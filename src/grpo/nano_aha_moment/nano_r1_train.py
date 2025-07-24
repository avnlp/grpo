# This code is based on the implementation from: https://github.com/McGill-NLP/nano-aha-moment

import gc
import time
from pathlib import Path
from typing import Any

import deepspeed
import numpy as np
import torch
import wandb
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from .config import (
    DATASET_NAME,
    EPISODES_PER_ITERATION,
    GENERATIONS_PER_SAMPLE,
    KL_COEFFICIENT,
    LEARNING_RATE,
    MAX_RESPONSE_TOKENS,
    MODEL_CHAT_NAME,
    MODEL_NAME,
    NUM_ITERATIONS,
    PER_DEVICE_BATCH_SIZE,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    deepspeed_config,
    ref_deepspeed_config,
)
from .episode_creation import create_training_episodes
from .policy_gradient_loss import compute_pg_loss
from .reward_functions import EOS_TOKEN, EOS_TOKEN_ID, compute_reward
from .utils import dump_episodes, evaluate_on_test_set, find_last_checkpoint, load_model_into_vllm, prepare_model_inputs

RUN_NAME = "r1-zero"
EXP_DIR = Path("r1_zero") / RUN_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

# Processing the Countdown Tasks dataset
# The Countdown task is where one must reach a target number using a set of numbers and basic arithmetic operations: addition, subtraction, multiplication, and division. Each number must be used exactly once.

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in the mind "
    "and then provide the user with the answer."
)
PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT_NAME)


# Load and process dataset
def preprocess_example(example: dict[str, Any]):
    numbers: list[int] = example["nums"]
    target: int = example["target"]

    # We prefix the assistant with "Let me solve this step by step.\n<think>" to make it easier for the model to generate the answer.
    # We do this because we are training the base model which has no prior understanding of system prompts or chat formatting.
    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target)},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"prompt": prompt, "input_ids": input_ids}


dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(preprocess_example, num_proc=6)

# Split dataset
train_test_split = dataset.train_test_split(test_size=500, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

policy_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map=0,
)
reference_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map=0,
)
policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


# Initialize DeepSpeed engines
policy_model, *_ = deepspeed.initialize(
    model=policy_model,
    config=deepspeed_config,
    model_parameters=policy_model.parameters(),
)
reference_model, *_ = deepspeed.initialize(
    model=reference_model,
    config=ref_deepspeed_config,
)

reference_model.module.cpu()

# Initialize vLLM (Inference) engine
inference_engine = LLM(
    model=MODEL_NAME,
    skip_tokenizer_init=False,
    gpu_memory_utilization=0.2,
    enable_prefix_caching=True,
    swap_space=1,
    scheduling_policy="fcfs",
    dtype=torch.bfloat16,
    max_model_len=2048,
    enable_sleep_mode=True,
)

# Wandb for logging
wandb.init(
    project="r1-aha-moment",
    name=RUN_NAME,
    config={
        "model_name": MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "num_iterations": NUM_ITERATIONS,
        "episodes_per_iteration": EPISODES_PER_ITERATION,
        "rollouts_per_episode": GENERATIONS_PER_SAMPLE,
        "kl_coefficient": KL_COEFFICIENT,
        "temperature": TEMPERATURE,
    },
)

# Load checkpoint if it exists
begin_iter = 0
ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
if ckpt_path is not None:
    print(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
    out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
    if out is None:
        msg = f"Failed to load checkpoint {ckpt_path}"
        raise RuntimeError(msg)
    begin_iter = ckpt_iter + 1
    load_model_into_vllm(policy_model, inference_engine)

# Training loop

# Each iteration of the training loop performs the following steps:

# 1. Evaluation: Every few iterations, the model is evaluated on a test set to monitor progress.
# 2. Episode Generation: A batch of prompts is sampled, and multiple responses are generated for each prompt using the
# inference engine. Then we put the inference engine to sleep.
# 3. Reward Computation: Rewards and advantages for each generated episode are computed.
# 4. Policy Gradient Training: Using the computed advantages, we calculate the policy gradient loss and update the model
#  parameters. The training is done using gradient accumulation to handle large batches.
#  Note that we apply single gradient update per iteration.
# 5. Inference Engine Update: The inference engine is woken up and updated with the latest model weights.
# 6. Logging: Training and evaluation metrics are logged using WandB.
# 7. Checkpointing: Every 50 iterations, the model and optimizer states are saved.

for iteration in trange(NUM_ITERATIONS):
    print(f"Iteration {iteration}/{NUM_ITERATIONS}")

    metrics = {}

    # Evaluation: Evaluate on the test set
    eval_stats = None
    if iteration % 25 == 0:
        print("Evaluating on eval set...")
        eval_episodes, eval_stats = evaluate_on_test_set(
            inference_engine=inference_engine,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            eos_token=EOS_TOKEN,
            eval_sampling_params=SamplingParams(
                temperature=0.3,
                max_tokens=1024,
                n=1,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
            reward_func=lambda completion, sample: compute_reward(completion, sample),
        )
        eval_episode_table = dump_episodes(
            episodes=eval_episodes,
            episodes_stats=eval_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
            is_eval=True,
        )
        wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})

    # Generate Episodes
    # Sample training batch
    num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
    indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
    samples = train_dataset.select(indices)

    # Sample responses
    outputs = inference_engine.generate(
        prompt_token_ids=samples["input_ids"],
        sampling_params=SamplingParams(
            n=GENERATIONS_PER_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_RESPONSE_TOKENS,
            detokenize=False,
            stop_token_ids=[EOS_TOKEN_ID],
        ),
    )
    all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
    all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
    inference_engine.sleep(1)

    print(f"Generated {len(all_generations)} responses")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    # Process responses and calculate rewards
    episodes, episodes_stats = create_training_episodes(
        samples,
        all_generations,
        all_finish_reasons,
    )
    for k, v in episodes_stats.items():
        metrics.setdefault(k, []).extend(v)

    episode_table = dump_episodes(
        episodes=episodes,
        episodes_stats=episodes_stats,
        exp_dir=EXP_DIR,
        tokenizer=tokenizer,
        iteration=iteration,
    )

    # Training
    # Prepare training batch
    model_inputs = prepare_model_inputs(
        query_token_ids=episodes["all_query_token_ids"],
        response_token_ids=episodes["all_response_token_ids"],
        advantages=episodes["all_advantages"],
        device="cuda",
    )

    # Calculate losses and update model
    policy_model.train()
    reference_model.module.cuda()
    reference_model.eval()

    total_response_len = (model_inputs["labels"] != -100).sum().item()

    for i in trange(0, EPISODES_PER_ITERATION, PER_DEVICE_BATCH_SIZE, desc="Gradient Accumulation"):
        batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}

        # Compute policy gradient loss
        loss, loss_metrics = compute_pg_loss(
            policy_model=policy_model,
            reference_model=reference_model,
            batch=batch,
            total_response_len=total_response_len,
        )

        # Track metrics
        metrics.setdefault("loss", []).append(loss.item())
        grad_norm = policy_model.get_global_grad_norm()
        if grad_norm is not None:
            grad_norm = grad_norm.item()
        metrics.setdefault("grad_norm", []).append(grad_norm)
        for k, v in loss_metrics.items():
            metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

        # Backpropagation and optimization step
        policy_model.backward(loss, scale_wrt_gas=False)

        # Free memory
        del loss, loss_metrics
        if policy_model.is_gradient_accumulation_boundary():
            reference_model.module.cpu()

        policy_model.step()

    # Update inference engine weights
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    inference_engine.wake_up()
    load_model_into_vllm(policy_model, inference_engine)

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    # Log metrics
    train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
    train_metrics["learning_rate"] = policy_model.get_lr()[0]
    logs = {
        "iteration": iteration,
        f"episodes/iter_{iteration:06d}": episode_table,
        **{f"train/{k}": v for k, v in train_metrics.items()},
    }
    if eval_stats is not None:
        eval_metrics = {k: np.mean(v) for k, v in eval_stats.items() if None not in v}
        logs.update({f"eval/{k}": v for k, v in eval_metrics.items()})
    wandb.log(logs)

    selected_keys = [
        "train/kl_penalty",
        "train/rewards",
        "train/reward_metrics/format_reward",
        "train/reward_metrics/equation_reward",
        "eval/rewards",
        "eval/reward_metrics/format_reward",
        "eval/reward_metrics/equation_reward",
    ]
    selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
    print(f"KEY METRICS: {selected_metrics}")

    if iteration % 50 == 0 and iteration != 0:
        policy_model.module.save_pretrained(str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model"))
        policy_model.save_checkpoint(str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "deepspeed"))

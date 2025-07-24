# This code is based on the implementation from: https://github.com/lsdefine/simple_GRPO

import json

import deepspeed
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .ref_model_server import bytes_list_to_list, bytes_to_tensor
from .reward_functions import reward_correct, reward_format

Q_batch_size = 1
model_path = "/data2/Qwen/Qwen2.5-7B"
num_pre_Q = 8
max_prompt_length = 400
ref_server = "http://localhost:59875"

ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size * num_pre_Q,
    "gradient_accumulation_steps": 2,
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-6}},
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"},
    },
}


def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b"empty":
            return None
    except:
        return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])
    data["inputs"] = bytes_to_tensor(dd[1])
    data["rewards"] = bytes_to_tensor(dd[2])
    data["refs"] = bytes_to_tensor(dd[3])
    if len(dd) == 5:
        data["gen_logps"] = bytes_to_tensor(dd[4])
    return data


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
gen_model = model

engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
gen_model = engine

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.9,
    num_return_sequences=num_pre_Q,
    pad_token_id=tokenizer.pad_token_id,
)


def gen_answers(prompts):
    tip_text = []
    for x in prompts:
        tip_text.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": x}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length:
        return []
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace("<|endoftext|>", "") for x in completion_ids]
    return answers


def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0:
        return None, None, None, None
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i * num_pre_Q : (i + 1) * num_pre_Q]:
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    prompts_text = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": x}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for x in prompts
    ]
    prompt_inputs = tokenizer(
        prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    )["input_ids"]
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)[
        "input_ids"
    ]
    return prompt_inputs, output_ids, torch.tensor(rewards, dtype=torch.float32), answers


def get_per_token_logps(logits, input_ids):
    per_token_logps = []  # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids, strict=False):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

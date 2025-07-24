# This code is based on the implementation from: https://github.com/aburkov/theLMbook/blob/main/GRPO.py

import copy
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .completions import compute_log_probabilities, generate_rollout_data
from .data_process import prepare_dataset
from .evaluation import evaluate_model
from .reward_functions import combined_reward


def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(42)


def compute_group_relative_advantages(rewards, num_generations):
    """Compute group-relative advantages for each prompt group.

    Args:
        rewards (torch.Tensor): Tensor of shape (batch_size * num_generations) containing rewards.
        num_generations (int): Number of completions generated per prompt.

    Returns:
        torch.Tensor: Tensor of advantages computed relative to the group mean.
    """
    # Reshape rewards to group by prompt
    rewards_by_group = rewards.view(-1, num_generations)

    # Compute mean and standard deviation for each prompt group
    group_means = rewards_by_group.mean(dim=1)
    group_stds = rewards_by_group.std(dim=1)

    # Expand the means and stds to match the original flat rewards tensor shape
    expanded_means = group_means.repeat_interleave(num_generations)
    expanded_stds = group_stds.repeat_interleave(num_generations)

    # Normalize rewards to get advantages
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)

    return advantages.unsqueeze(1)  # Add dimension for token-wise operations


def maximize_grpo_objective(model, ref_model, rollout_data, tokenizer, reward_function, optimizer, beta, epsilon):
    """Update the policy model by maximizing the GRPO objective.

    Args:
        model: The current policy model.
        ref_model: The reference model.
        rollout_data: Dictionary containing rollout data.
        tokenizer: The tokenizer.
        reward_function: Function to compute rewards.
        optimizer: The optimizer.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter.

    Returns:
        float: The loss value.
    """
    # Extract data from rollout
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    logits_to_keep = rollout_data["logits_to_keep"]

    # Compute current log probabilities
    current_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)

    # Compute policy ratio
    ratio = torch.exp(current_log_probs - old_log_probs)

    # Get rewards data
    formatted_completions = rollout_data["formatted_completions"]
    repeated_prompts = rollout_data["repeated_prompts"]
    repeated_answers = rollout_data["repeated_answers"]

    # Compute rewards
    rewards = torch.tensor(
        reward_function(prompts=repeated_prompts, completions=formatted_completions, answer=repeated_answers),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    avg_reward = rewards.mean().item()
    print(f"Average Reward: {avg_reward:.4f}")

    # Compute advantages using group-relative normalization
    rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    advantages = compute_group_relative_advantages(rewards, num_generations)

    # Compute surrogate loss with clipping
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surrogate1, surrogate2)

    # Compute KL divergence penalty
    kl_div = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1

    # Combine losses
    per_token_loss = surrogate_loss - beta * kl_div
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()

    return loss.item()


def train_with_grpo(
    model,
    tokenizer,
    train_data,
    num_iterations=1,
    steps_per_iteration=500,
    batch_size=4,
    num_generations=4,
    max_completion_length=128,
    beta=0.1,
    learning_rate=5e-6,
    mu=3,
    epsilon=0.2,
    reward_function=combined_reward,
):
    """Iterative Group Relative Policy Optimization algorithm.

    Args:
        model: The initial policy model to be fine-tuned.
        tokenizer: The tokenizer used for encoding prompts and decoding completions.
        train_data (list): List of training samples with "prompt" and "answer" fields.
        num_iterations (int): Number of outer iterations (reward model updates).
        steps_per_iteration (int): Number of policy update steps per iteration.
        batch_size (int): Number of prompt samples per batch.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL-divergence penalty coefficient.
        learning_rate (float): Learning rate for optimizer.
        mu (int): Number of GRPO updates per batch of generations.
        epsilon (float): Clipping parameter for surrogate objective.
        reward_function: Function that evaluates completions and returns rewards.

    Returns:
        The fine-tuned policy model.
    """
    # Initialize policy model
    policy_model = model
    device = next(policy_model.parameters()).device

    # Outer loop for iterations with reward model updates
    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting iteration {iteration}/{num_iterations}")

        # Create reference model for KL constraint
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        reference_model = reference_model.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
        policy_model.train()

        # Inner loop for policy updates
        for step in range(1, steps_per_iteration + 1):
            # Sample batch of prompts
            batch_samples = random.sample(train_data, batch_size)

            # Set old policy for this step
            with torch.no_grad():
                # Generate completions and compute log probs
                rollout_data = generate_rollout_data(
                    policy_model, reference_model, tokenizer, batch_samples, num_generations, max_completion_length
                )

            # Multiple GRPO updates per batch of generations
            for grpo_iter in range(1, mu + 1):
                loss_value = maximize_grpo_objective(
                    policy_model, reference_model, rollout_data, tokenizer, reward_function, optimizer, beta, epsilon
                )
                print(
                    f"Iteration {iteration}/{num_iterations}, Step {step}/{steps_per_iteration}, "
                    f"GRPO update {grpo_iter}/{mu}, Loss: {loss_value:.4f}"
                )

        # Optional: Update reward model here if using reward model training
        # This is not implemented in the original code but present in the pseudocode
        print(f"Completed iteration {iteration}. Reward model update would happen here.")

    return policy_model


def optimize_model_memory(model):
    """Apply memory optimizations like proper gradient checkpointing setup."""
    # Ensure model is in training mode
    model.train()

    # Disable caching for gradient checkpointing
    model.config.use_cache = False

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Enable input gradients properly
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def main():
    """Main function to run the complete training and evaluation pipeline.

    The process consists of:
      1. Loading the pre-trained model and tokenizer.
      2. Evaluating the initial model performance (before any finetuning).
      3. Performing reinforcement learning (GRPO) finetuning.
      4. Evaluating the final model after GRPO finetuning.
      5. Saving the finetuned model and tokenizer.

    Note: Functions such as prepare_dataset, evaluate_model, and reward_function
          are assumed to be defined elsewhere.
    """
    # Determine the device (GPU if available, otherwise CPU) from the model's parameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model name and output directory.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # The 0.5B model is not smart enough
    # to generate the <reasoning> and <answer> tags
    # so several iterations of SFT to teach it these tags
    # are recommended before RL

    # Load the pre-trained causal language model.
    # - torch_dtype specifies the precision (bfloat16 for efficiency on supported hardware).
    # - attn_implementation selects an optimized attention mechanism.
    # - device_map="auto" automatically distributes the model across available devices.
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=None,
    )
    print("Downloaded model")
    # Move the model to the determined device.
    model = model.to(device)

    # Load the tokenizer corresponding to the model.
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # Set the pad token to be the same as the end-of-sequence token.
    tokenizer.pad_token = tokenizer.eos_token
    # Update the model configuration with the correct token IDs.
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # -------------------------------
    # Step 0: INITIAL EVALUATION
    # -------------------------------
    # Load the complete training dataset using a helper function (assumed defined elsewhere).
    all_data = prepare_dataset("train")
    # Randomize the order of examples.
    random.shuffle(all_data)
    # Use a small subset (e.g., 30 examples) for evaluation.
    num_eval_examples = 1
    eval_data = all_data[:num_eval_examples]

    # Evaluate the initial performance of the model before any finetuning.
    print("\nInitial model evaluation before GRPO:")
    pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    model = optimize_model_memory(model)

    # -------------------------------
    # Step 1: RL FINETUNING (GRPO)
    # -------------------------------
    print("\nStarting RL finetuning using GRPO...")

    # Use the remaining examples (beyond the evaluation subset) for RL finetuning.
    train_data = all_data[num_eval_examples:]

    # Define RL training configuration.
    training_config = {
        "num_iterations": 1,
        "steps_per_iteration": 500,  # Total number of RL training steps.
        "batch_size": 4,  # Number of samples per training step.
        "num_generations": 16,  # Number of completions generated per prompt.
        "max_completion_length": 500,  # Maximum token length for each generated completion.
        "beta": 0.04,  # KL divergence penalty coefficient.
        "learning_rate": 5e-6,  # Learning rate for RL fine-tuning.
        "mu": 1,
        "epsilon": 0.1,
        "reward_function": combined_reward,
    }
    # Fine-tune the model using GRPO RL training.
    model = train_with_grpo(model=model, tokenizer=tokenizer, train_data=train_data, **training_config)

    # -------------------------------
    # Step 2: FINAL EVALUATION & SAVING
    # -------------------------------
    print("\nFinal model evaluation after GRPO RL finetuning:")
    # Evaluate the final model performance using the evaluation dataset.
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
    print(f"Total Improvement: {post_grpo_accuracy - pre_grpo_accuracy:.2f}%")

    print("\nSaving GRPO finetuned model...")
    # Save the final finetuned model and tokenizer to disk.
    model.save_pretrained("grpo_finetuned_model")
    tokenizer.save_pretrained("grpo_finetuned_model")


if __name__ == "__main__":
    main()

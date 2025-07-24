# Simple GRPO

Implementation of the **GRPO (Group Relative Policy Optimization)** training algorithm from:
[https://github.com/lsdefine/simple_GRPO](https://github.com/lsdefine/simple_GRPO).

We refactor the original code into a **modular and easy-to-understand layout**.
Each major step of the training process is separated into its own file for clarity and extensibility.

## Overview

1. A reference model (fixed) is used to compute the per-token log probabilities for the generated sequences. This is done in a separate server to offload the computation and avoid GPU memory issues on the training node.
2. The training process:
   a. Generates samples (using the current model) and computes rewards for them.
   b. Sends the generated sequences and rewards to the reference server, which returns the reference model's per-token log probabilities (for the generated part).
   c. Uses these to compute a policy gradient loss with a KL penalty (to stay close to the reference model) and updates the model parameters.

The use of a separate server for the reference model is a design choice to allow for distributed training and to keep the reference model fixed and efficient.

The code consists of two main parts:

1. A server (ref_server) that runs a reference model and computes per-token log probabilities for given input sequences.
2. The main training loop that uses DeepSpeed to train a model with a custom loss function (GRPO_step).
The code also includes functions for generating samples (using the GSM8K dataset) and computing rewards based on the correctness and format of the generated answers.
Let's break down the entire process step by step.

## Training

### Step 1: Setting up the reference model server (ref_server)

- The reference model is loaded (Qwen2.5-7B) and set to evaluation mode (no gradients).
- A Bottle server is started on port 59875 with two endpoints:
  - `/upload` (POST): Receives data (a batch of input sequences and rewards) and puts it in a queue (`raw_queue`).
  - `/get` (GET): Returns processed data (if available) from the `result_queue`.
- The server runs in a separate thread.
The server's main loop:
- It waits for data in `raw_queue`.
- For each received data item, it computes the per-token log probabilities for the input sequences (using the reference model) from the prompt length onward.
- It then packages the data (including the original inputs, rewards, and the computed log probabilities) and puts it in `result_queue` for the training process to pick up.

### Step 2: Main training script

#### Initialization

- The same model (Qwen2.5-7B) is loaded again, along with its tokenizer.
- The dataset (GSM8K training set) is loaded and formatted into a list of dictionaries (each with 'Q' for question and 'A' for answer).
- A generation configuration is set for sampling multiple sequences per prompt (with temperature 0.9 and max_new_tokens=512).

#### Generate Answers

- This function takes a list of prompts and generates answers using the model (initially the base model, then the training model as training progresses).
- It formats each prompt with a system message and the user's question using the tokenizer's chat template.
- The model generates `num_pre_Q` (8) completions per prompt.

#### Reward functions

- `reward_correct(item, answer)`: Checks if the answer contains the correct numeric answer (by comparing the last number in the generated answer with the ground truth). Returns 1 if correct, -1 otherwise.
- `reward_format(item, answer)`: Checks if the generated answer follows the expected format (with `<think>...</think><answer>...</answer>`). Returns 1.25 if the format is correct, -1 otherwise.
- The total reward is the sum of these two.

#### Generate Samples

- Takes a batch of questions (from `QAs`).
- Generates answers for each question (using `gen_answers`).
- Computes the reward for each generated answer.
- Tokenizes the prompts and the generated answers, and returns:
  - `prompt_inputs`: Tokenized prompts (input IDs).
  - `output_ids`: Tokenized generated answers (without the prompt).
  - `rewards`: A tensor of rewards for each generated answer.
  - `answers`: The generated answers (strings).

#### Generate Mode

- This function is responsible for generating samples and sending them to the reference server.
- It runs for `num` iterations (each iteration processes `Q_batch_size` (1) questions).
- For each question, it generates `num_pre_Q` (8) answers and computes their rewards.
- The rewards are normalized (subtract mean, divide by std) for the batch of generated answers for the same prompt.
- The entire sequence (prompt + generated answer) is tokenized and sent to the reference server along with the rewards and (optionally) the per-token log probabilities from the current model (if `compute_gen_logps` is True). This is done via a POST request to `/upload`.

#### Training Setup

- If the script is run with the 'genonly' argument, it only generates samples (runs `generate_mode` indefinitely) and exits.
- Otherwise, it initializes DeepSpeed with the given configuration (`ds_config`).
- The model is wrapped by DeepSpeed's engine.

- The training loop runs for `all_steps` (1000) steps.
- At the beginning, it runs `generate_mode` to populate the server with some initial data.
- In each step:
  - It tries to get a batch from the reference server (via the `/get` endpoint). If none is available, it runs `generate_mode` again until it gets a batch.
  - It computes the loss using `GRPO_step` and performs a backward pass and optimizer step with DeepSpeed.
  - Every `save_steps` (200) steps, the model is saved (only by rank 0).

#### Loss Function: `GRPO_step(batch)`

This function computes the loss for a batch of data. The batch contains:

- `inputs`: The entire token sequence (prompt + generated answer) of shape (B, L).
- `rewards`: The normalized rewards for each sequence (shape (B,)).
- `refs`: The per-token log probabilities from the reference model (for the generated part, shape (B, L_generated)).
- (Optionally) `gen_logps`: The per-token log probabilities from the current model at the time of generation (for the generated part, shape (B, L_generated)).
Steps in `GRPO_step`:

1. Extract `prompt_length` from the batch.
2. Move `inputs` and `rewards` to the current device.
3. Forward pass: compute logits for the entire sequence (shape (B, L, V)).
4. Remove the last logit (since we don't have a target for the next token after the last one) and the first token of the input (since we don't have a logit for the first token). So we get:
   - `logits`: (B, L-1, V)
   - `input_ids`: (B, L-1) [the targets for the logits]
5. Compute the per-token log probabilities for the current model (for the non-prompt part) by:
   - For each token position in the sequence (from the first token of the generated part to the end), compute the log probability of the actual token (using `get_per_token_logps`).
   - The result is `per_token_logps` of shape (B, L_generated) where L_generated = L - prompt_length.
6. Compute the per-token KL penalty:
   - `per_token_kl = exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1`
   - This is the KL divergence penalty (which is always non-negative) for each token.
7. Create a mask for the generated part (to ignore padding tokens).
8. Compute the policy gradient loss:
   - If `gen_logps` is provided (i.e., we are using PPO-style clipping):
        ratio = exp(current_logps - gen_logps)   [at the time of generation]
        clipped_ratio = clip(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = min(ratio *advantages, clipped_ratio* advantages)
   - Else (if not using clipping, then it's REINFORCE):
        per_token_loss = exp(current_logps - current_logps.detach()) *advantages

9. The total per-token loss is: `-(per_token_loss - beta * per_token_kl)`
   - We subtract the KL penalty (beta is a hyperparameter) to encourage the current policy not to deviate too far from the reference model.
10. The overall loss is the mean of the per-sequence loss (where each sequence's loss is the sum of the per-token losses divided by the number of non-padding tokens).

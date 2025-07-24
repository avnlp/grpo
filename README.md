# Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) is an algorithm proposed by Deepseek for training large language models with reinforcement learning. This repository aggregates and refactors **four distinct implementations** of GRPO, each demonstrating different approaches to the core algorithm while sharing common principles.

## Algorithm

The core GRPO algorithm follows these steps:

1. For each training step, randomly sample $N$ questions $q_1, q_2, \cdots, q_N$.
2. For each question $q_i$, sample $M$ answers $a_{i,1}, a_{i,2}, \cdots, a_{i,M}$.
3. Compute the reward $r_{i,j}$ for each answer $a_{i,j}$.
4. Compute group statistics for each question $q_i$:

$$
\begin{alignedat}{2}
&\mu_i &&\leftarrow \text{mean}(r_{i,1}, r_{i,2}, \cdots, r_{i,M}) \\
&\sigma_i &&\leftarrow \text{std}(r_{i,1}, r_{i,2}, \cdots, r_{i,M})
\end{alignedat}
$$

5. For each token $t$ in answer $a_{i,j}$, compute advantage:
$$A_{i,j}[t] \leftarrow \frac{r_{i,j} - \mu_i}{\sigma_i}$$

6. Update policy using PPO surrogate objective:
$$\nabla_\theta \log \pi_\theta(a_{i,j}[t]) \cdot A_{i,j}[t]$$

## Implementations

We provide four refactored implementations of GRPO, each with a different focus and design:

### 1. [nanoAhaMoment](src/grpo/nano_aha_moment)

An implementation from [nanoAhaMoment](https://github.com/McGill-NLP/nano-aha-moment), that separates each step of the GRPO loop into distinct components. It uses a rule-based reward function for a Countdown task and integrates with vLLM for efficient generation.

- Modular pipeline with separated components
- vLLM integration for efficient generation
- DeepSpeed training backend
- Format: `<think>...</think>\n<answer>...</answer>`
- Rule-based reward functions for Countdown tasks

### 2. [GRPO:Zero](src/grpo/grpo_zero)

An implementation from [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero), that uses a separate server for the reference model to offload computation. It uses the GSM8K dataset and a combined reward for correctness and format.

- Qwen2.5-3B-Instruct base model
- Countdown-Tasks-3to4 dataset
- Simplified training workflow
- Reward Function: Combined reward for correctness and format

### 3. [Simple GRPO](src/grpo/simple_grpo)

An implementation from [Simple GRPO](https://github.com/lsdefine/simple_GRPO), that uses DeepSpeed for training and a reference model server. It features a policy gradient loss with KL penalty and reward normalization within groups.

- Reference model server architecture
- GSM8K dataset
- KL divergence penalty term
- Per-token advantage calculation
- Distributed training support
- Loss Calculation: `loss = -(policy_ratio * advantage - beta * kl_divergence)`

### 4. [GRPO from Scratch](src/grpo/andriy_burkov_lm_book)

An implementation from ["The LM Book" by Andriy Burkov](https://github.com/aburkov/theLMbook/blob/main/GRPO.py), that demonstrates the core GRPO algorithm step-by-step. It uses a copy of the reference model and performs multiple updates per batch.

- Periodic reference model updates
- Multiple updates per batch (μ-PPO)
- Comprehensive reward decomposition
- Memory optimization techniques
- Reward Function: Combined reward for correctness and format

## Common Components

All implementations share the following steps:

- **Group Sampling**: For each prompt, multiple completions are generated to form a group.
- **Reward Calculation**: Each completion receives a scalar reward, typically combining correctness and format adherence.
- **Advantage Normalization**: Within each group, rewards are normalized to have zero mean and unit variance to form advantages.
- **Policy Update**: The policy is updated using a policy gradient method (with or without clipping) and often includes a KL penalty to prevent deviation from a reference policy.

## Variations

The implementations have different variations in the following aspects:

- Reward Functions: The implementations use different reward functions tailored to the task and different weights for format and correctness.
  - **Format Reward**: Enforces XML-style reasoning structure
  - **Correctness Reward**: Validates solution accuracy
  - **Combined Reward**: `R_total = R_format + R_correctness`

- Reference Model Handling: Some implementations use a fixed reference model (via a separate server or a frozen copy) while others update the reference model periodically.

- Training Framework: The implementations use different training frameworks (e.g., DeepSpeed, pure PyTorch) and optimization techniques (e.g., gradient checkpointing).

- Batching and Generation: The approaches to generation (vLLM, Hugging Face transformers) and batching vary.

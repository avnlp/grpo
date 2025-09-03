# Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) is an algorithm proposed by Deepseek for training large language models with reinforcement learning. This repository aggregates and refactors **four distinct implementations** of GRPO, each demonstrating different approaches to the core algorithm while sharing common principles.

## Algorithm

GRPO eliminates the need for additional value function approximation by using the average reward of multiple sampled responses to the same query as the baseline, significantly reducing computational and memory overhead in reinforcement learning training. This approach maintains policy optimization stability while removing the complexity of training a value function.

For each question $q$, GRPO samples a group of outputs $\{o_1, o_2, \cdots, o_G\}$ from the old policy $\pi_{\theta_{old}}$ and then optimizes the policy model by maximizing the following objective:

<img src="images/grpo_objective.png" width="95%" alt="GRPO Objective">

Instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of $\hat{A}_{i,t}$. The KL divergence is estimated with the following unbiased estimator, which is guaranteed to be positive:


<div align="center">
    <img src="images/kl_divergence.png" width="60%" alt="GRPO KL Divergence">
</div>

&nbsp;

A reward function is then used to score the outputs, yielding $G$ rewards $\mathbf{r}=\{R(q, o_1), R(q, o_2), \cdots, R(q, o_G)\}$ correspondingly. These rewards are normalized by subtracting the group average and dividing by the group standard deviation. The normalized rewards for each output $o_i$ define the advantages $\hat{A}_{i, t}$ for all tokens in the output. The policy is then optimized by maximizing the GRPO objective.

<div align="center">
    <img src="images/rewards.png" width="44%" alt="GRPO Rewards">
</div>

where,

- $\pi_{\theta}$ and $\pi_{\theta_{old}}$ are the current and old policy models,
- $\pi_{ref}$ is the reference model
- $q$ are questions from the question dataset
- $o$ are outputs sampled from the old policy $\pi_{\theta_{old}}$
- $R(q, o_i)$ is the reward for output $o_i$ to question $q$
- $\epsilon$ is a clipping-related hyper-parameter introduced in PPO for stabilizing training.
- $\beta$ is the coefficient of the KL penalty
- $\hat{A}_{i,t}$ is the advantage calculated based on relative rewards of the outputs inside each group (note: the advantage value is constant for all tokens within a single output sequence).

&nbsp;

<div style="border: 1px solid #ccc; padding: 15px; font-family: monospace;">
<pre>
<b>Algorithm:</b>
Input initial policy model π<sub>θ<sub>init</sub></sub>; reward function r<sub>φ</sub>; task prompts 𝓓; hyperparameters ε, β, μ
1: policy model π<sub>θ</sub> ← π<sub>θ<sub>init</sub></sub>
2: <b>for</b> iteration = 1, ..., I <b>do</b>
3:     reference model π<sub>ref</sub> ← π<sub>θ</sub>
4:     <b>for</b> step = 1, ..., M <b>do</b>
5:         Sample a batch 𝓓<sub>b</sub> from 𝓓
6:         Update the old policy model π<sub>θ<sub>old</sub></sub> ← π<sub>θ</sub>
7:         Sample G outputs {o<sub>i</sub>}<sup>G</sup><sub>i=1</sub> ∼ π<sub>θ<sub>old</sub></sub>(·|q) for each question q ∈ 𝓓<sub>b</sub>
8:         Compute rewards {r<sub>i</sub>}<sup>G</sup><sub>i=1</sub> = {R(q, o<sub>1</sub>), R(q, o<sub>2</sub>), ..., R(q, o<sub>G</sub>)} for each output o<sub>i</sub> using r<sub>φ</sub>
9:         Compute output-level advantage:
               Â<sub>i</sub> = (R(q, o<sub>i</sub>) - mean(R(q, o<sub>1</sub>), ..., R(q, o<sub>G</sub>))) / std(R(q, o<sub>1</sub>), ..., R(q, o<sub>G</sub>))
               and set Â<sub>i,t</sub> = Â<sub>i</sub> for all tokens t in output o<sub>i</sub> (constant within each output)
10:        <b>for</b> GRPO iteration = 1, ..., μ <b>do</b>
11:            Update the policy model π<sub>θ</sub> by maximizing the GRPO objective
<b>Output</b> π<sub>θ</sub>
</pre>
</div>

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


## References

[1] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

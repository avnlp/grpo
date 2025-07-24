# GRPO:Zero

Implementation of the **GRPO (Group Relative Policy Optimization)** training algorithm from:
[https://github.com/policy-gradient/GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero).

## CountDown Task

We are going to train the Qwen2.5 model on the [CountDown task](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4). Given a list of 3 or 4 numbers and a target number, the model needs to generate a mathematical expression using simple arithmetic operations (+, -, *, /) that evaluates to the target number. For example:

```
Question: Given 1 2 3 4 and a target number 11. Show an expression that evaluates to 11.
Answer: 1 + (2 * 3) + 4
```

## Reward Function

To solve the CountDown task, we will use the GRPO algorithm to train the model to generate the chain of thought reasoning before generating the final expression. Specifically, the model is trained to follow the format:

```
<think>Model step by step reasoning</think>
<answer>Final answer</answer>
```

The reward is the sum of two components:

1. **Format Reward**: The model earns a reward of `0.1` when it correctly follows the specified format with thinking and answer tags, and `0` otherwise.
2. **Answer Reward**: The model receives a reward of `1` if its final answer uses each provided number exactly once and correctly evaluates to the target value, otherwise it receives `0`.

## Training

We use the `Qwen2.5-3B-Instruct` model for training. To train the model, run the following commands:

```bash
# download the dataset
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

# download the pretrained model
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

# train the model
uv run train.py
```

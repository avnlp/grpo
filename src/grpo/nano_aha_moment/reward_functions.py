# This code is based on the implementation from: https://github.com/McGill-NLP/nano-aha-moment

# Reward Functions

# The DeepSeek R1 paper introduced rule-based rewards to evaluate whether the model-generated solutions were correct. We'll adopt a similar approach by defining two custom reward functions:

# - Format Reward: Checks if the output follows the required format:
#   `<think> [thinking] </think><answer> [answer] </answer>`

# - Equation Reward: Extracts the equation from within the `<answer>` tag, verifies that it evaluates to the target result, and ensures that all available numbers are used exactly once.


import re
from ast import literal_eval
from typing import Any

from config import MODEL_NAME
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


EOS_TOKEN_ID = tokenizer.eos_token_id
EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)


def format_reward_func(completion: str) -> float:
    r"""Format: <think>...</think>\n</answer>...</answer>.

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # add synthetic <think> as its already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: list[int], target: int) -> float:
    """Evaluate completion based on mathematical correctness of the answer.

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = literal_eval(equation)
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(completion: str, sample: dict[str, Any]) -> tuple[float, dict[str, float]]:
    r"""Compute the reward for a given completion and sample.

    The final reward assigned to an episode/trajectory (prompt+response) is simply the sum of the individual rewards.

    The reward is only computed at the last token of the output. From an RL perspective, this means that all intermediate actions receive zero reward.
    """
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion)
    equation_reward = equation_reward_func(completion=completion, nums=nums, target=target)

    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics

# This code is based on the implementation from: https://github.com/policy-gradient/GRPO-Zero

from dataclasses import dataclass


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: list[int]
    prefix_tokens: list[str]
    generated_token_ids: list[int]
    is_finished: bool
    reward: float
    reward_info: dict[str, float]


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: list[str]
    prefix_tokens: list[list[str]]
    prefix_token_ids: list[list[int]]
    numbers: list[list[int]]
    target: list[int]

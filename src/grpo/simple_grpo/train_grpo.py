# This code is based on the implementation from: https://github.com/lsdefine/simple_GRPO

import json
import os
import random
import sys
import time

import requests
import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from .completions import (
    engine,
    gen_samples,
    get_batch,
    get_per_token_logps,
    model,
    tokenizer,
)
from .ref_model_server import make_bytes_list, tensor_to_bytes

os.environ["TOKENIZERS_PARALLELISM"] = "true"
Q_batch_size = 1
model_path = "/data2/Qwen/Qwen2.5-7B"
beta = 0.04
num_pre_Q = 8
all_steps = 1000
max_prompt_length = 400
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"


dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [{"Q": x, "A": y.split("####")[-1].strip()} for x, y in zip(dataset["question"], dataset["answer"], strict=False)]


def generate_mode(num=10, rank=0):
    if rank == 0:
        print("enter generate mode")
    tic = time.time()
    for ii in range(num):
        inputs = random.sample(QAs, Q_batch_size)
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
        if prompt_inputs is None:
            continue
        if rank == 0:
            print("rewards:", rewards)
            if ii == 5:
                print("answers:", answers[0])
        if (rewards.max() - rewards.min()).item() < 0.01:
            continue
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        data = [json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)]

        if compute_gen_logps:
            with torch.inference_mode():
                mids = merged_ids.to(model.device)
                gen_logps = get_per_token_logps(model(mids).logits[:, :-1, :], mids[:, 1:])
            data.append(tensor_to_bytes(gen_logps[:, prompt_length - 1 :].cpu()))

        xdata = make_bytes_list(data)
        requests.post(f"{ref_server}/upload", data=xdata)
    if rank == 0:
        print("exit generate mode")
    print(f"{rank}: {time.time()-tic:.3f}s")


if "genonly" in sys.argv:
    model.to("cuda")
    generate_mode(999999)
    sys.exit()


def GRPO_step(batch):
    prompt_length = batch["plen"]
    inputs = batch["inputs"].to(engine.device)
    advantages = batch["rewards"].to(engine.device).unsqueeze(1)  # normalized in generation

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length - 1 :]
    ref_per_token_logps = batch["refs"].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if "gen_logps" in batch:
        ratio = torch.exp(per_token_logps - batch["gen_logps"].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


generate_mode(rank=torch.distributed.get_rank())


progress = range(1, all_steps + 1)
if torch.distributed.get_rank() == 0:
    progress = tqdm(progress)
for step in progress:
    batch = get_batch()
    while batch is None:
        generate_mode(rank=torch.distributed.get_rank())
        batch = get_batch()

    loss = GRPO_step(batch)
    engine.backward(loss)
    engine.step()

    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step % save_steps == 0:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print("saving model")
            save_name = f"./step_{step}"
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
        dist.barrier()

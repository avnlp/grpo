# This code is based on the implementation from: https://github.com/aburkov/theLMbook/blob/main/GRPO.py

import torch
import torch.nn.functional as F


def selective_log_softmax(logits, input_ids):
    """Compute the log probabilities for the tokens specified in input_ids using a selective log-softmax.

    Args:
        logits (torch.Tensor): A tensor of shape (batch_size, seq_len, vocab_size) containing raw logits from the model.
        input_ids (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the token indices for which we want the log probabilities.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, seq_len) where each element is the log probability
                      corresponding to the token in input_ids at that position.

    Explanation:
        1. F.log_softmax is applied along the vocabulary dimension (dim=-1) to convert logits into log probabilities.
        2. The tensor input_ids is reshaped (via unsqueeze) to have an extra dimension so that we can use it as indices
           in the log_probs tensor.
        3. torch.gather collects the log probability at the index specified in input_ids for each position.
        4. Finally, squeeze(-1) removes the extra dimension, returning a tensor with the same shape as input_ids.
    """
    # Convert raw logits into log probabilities along the vocabulary axis.
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)

    # Reshape input_ids from (batch_size, seq_len) to (batch_size, seq_len, 1) for gathering.
    # Then, gather the log probability for each token in input_ids.
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))

    # Remove the extra last dimension to get back to shape (batch_size, seq_len).
    return selected_log_probs.squeeze(-1)


def compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep):
    """Compute per-token log probabilities for a subset of tokens (typically the completion tokens).

    Args:
        model: The language model to use.
        input_ids (torch.Tensor): Tensor of shape (batch_size, total_seq_len) containing token ids
                                  for both prompt and completion.
        attention_mask (torch.Tensor): Tensor of shape (batch_size, total_seq_len) indicating which tokens are real (1) or padding (0).
        logits_to_keep (int): Number of tokens (from the completion part) for which we need log probabilities.

    Returns:
        torch.Tensor: Log probabilities for the last `logits_to_keep` tokens of each sequence.

    Explanation:
        1. We call the model with logits_to_keep + 1 so that the model outputs one extra logit than needed.
           This is common in next-token prediction setups.
        2. We slice off the last logit along the sequence dimension because it does not correspond to any input token.
        3. We then restrict both the input_ids and logits to the last logits_to_keep tokens, which should
           correspond to the generated completion portion.
        4. Finally, we use the selective_log_softmax to compute log probabilities only for those tokens.
    """
    # Run the model forward pass and obtain logits.
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,  # Request one extra logit for proper alignment.
    ).logits  # Shape: (batch_size, total_seq_len, vocab_size)

    # Remove the last logit as it does not have a corresponding target token.
    logits = logits[:, :-1, :]  # New shape: (batch_size, total_seq_len - 1, vocab_size)

    # Slice the input_ids to keep only the last logits_to_keep tokens.
    # This corresponds to the generated completion tokens.
    input_ids = input_ids[:, -logits_to_keep:]  # Shape: (batch_size, logits_to_keep)

    # Also slice the logits to keep only those corresponding to the completion tokens.
    logits = logits[:, -logits_to_keep:, :]  # Shape: (batch_size, logits_to_keep, vocab_size)

    # Compute and return the log probabilities for the selected tokens.
    return selective_log_softmax(logits, input_ids)


def create_completion_mask(completion_ids, eos_token_id):
    """Create a binary mask for the generated completion tokens so that tokens after the first EOS are ignored.

    Args:
        completion_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) with generated token ids.
        eos_token_id (int): The token id representing the end-of-sequence.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_len) with 1s for tokens up to and including the first EOS
                      and 0s for tokens following the first EOS.

    Explanation:
        1. First, a boolean mask (is_eos) is created indicating where in the sequence the EOS token appears.
        2. An index tensor (eos_idx) is initialized, assuming that no EOS is found (defaulting to the sequence length).
        3. For sequences where EOS exists, eos_idx is updated to the position (index) of the first EOS.
        4. A sequence index tensor is created that contains indices for each position in the sequence.
        5. The final mask is computed by comparing the sequence indices to eos_idx (after adding a dimension).
    """
    # Determine which positions in each sequence equal the EOS token.
    is_eos = completion_ids == eos_token_id  # Boolean tensor of shape (batch_size, seq_len)

    # Initialize a tensor to store the index of the first EOS for each sequence.
    # If no EOS is found, default to the full sequence length (is_eos.size(1)).
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)

    # Identify sequences that contain at least one EOS.
    mask_exists = is_eos.any(dim=1)
    # For sequences with an EOS, update eos_idx to the index of the first occurrence.
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

    # Create a tensor of indices [0, 1, 2, ..., seq_len-1] and replicate it for each sequence in the batch.
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)

    # Build the mask: positions with an index less than or equal to the first EOS index are marked as 1.
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    return completion_mask


def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """Generate multiple completions for each prompt and create corresponding attention masks.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer to process the prompts and decode the outputs.
        prompts (list of str): List of input prompt strings.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of new tokens to generate for the completion.

    Returns:
        tuple: Contains the following tensors:
            - prompt_ids: (batch_size * num_generations, prompt_seq_len)
            - prompt_mask: (batch_size * num_generations, prompt_seq_len)
            - completion_ids: (batch_size * num_generations, completion_seq_len)
            - completion_mask: (batch_size * num_generations, completion_seq_len)

    Explanation:
        1. The prompts are tokenized and padded (with padding added to the left).
        2. Each prompt is repeated num_generations times so that multiple completions are generated per prompt.
        3. The model.generate() function is called to generate new tokens.
        4. The generated output contains the prompt followed by the completion; we remove the prompt part to get the completions.
        5. A mask is created (via create_completion_mask) so that only tokens up to the first EOS are considered.
    """
    device = next(model.parameters()).device

    # Tokenize the list of prompts with padding. The padding_side="left" ensures alignment on the right.
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_mask = inputs["attention_mask"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_length = prompt_ids.size(1)  # Save the prompt length to later separate prompt from completion.

    # Repeat each prompt num_generations times.
    prompt_ids = prompt_ids.repeat_interleave(
        num_generations, dim=0
    )  # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_mask = prompt_mask.repeat_interleave(
        num_generations, dim=0
    )  # New shape: (batch_size*num_generations, prompt_seq_len)

    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Remove the prompt portion from the generated output to isolate the completion tokens.
    completion_ids = outputs[:, prompt_length:]  # Shape: (batch_size*num_generations, completion_seq_len)

    # Create a binary mask that ignores tokens beyond the first EOS token.
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)

    return prompt_ids, prompt_mask, completion_ids, completion_mask


def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """Generate rollouts and compute static log probabilities for both the old policy (current model)
    and the reference model. Gradients are disabled so that these remain fixed.

    Args:
        model: The current model (policy) used to generate rollouts.
        ref_model: The static reference model.
        tokenizer: The tokenizer.
        batch_samples: List of training samples.
        num_generations: Number of completions to generate per prompt.
        max_completion_length: Maximum completion length.

    Returns:
        A dictionary with rollout data including both old and reference log probabilities.
    """
    tokenizer.padding_side = "left"
    next(model.parameters()).device

    # Extract prompts and answers.
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    # Generate completions and associated masks.
    # We generate once, and then use the same completions to compute both sets of log probabilities.
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute old_log_probs from the current model, with gradients disabled.
        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)

        # Compute ref_log_probs from the reference model, which remains static.
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, logits_to_keep)

    formatted_completions = [[{"content": tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,  # Static log probs from the current model (old policy)
        "ref_log_probs": ref_log_probs,  # Static log probs from the reference model
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }

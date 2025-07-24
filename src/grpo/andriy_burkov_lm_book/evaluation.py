# This code is based on the implementation from: https://github.com/aburkov/theLMbook/blob/main/GRPO.py

import re

from .data_process import extract_answer_from_model_output


def _extract_last_number(text):
    """Extracts the last number from text if it's properly separated.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The extracted number as a float, or None if no valid number is found.

    Explanation:
        1. First removes $ and % signs from the text.
        2. Uses regex to find numbers that are:
           - Preceded by space, equals sign, or start of string
           - Followed by end of string or space
        3. Returns the first matching number as a float, or None if no match is found.
    """
    # Remove $ and % signs
    text = text.replace("$", "").replace("%", "")

    # Look for numbers that are:
    # - preceded by space or = or start of string (via \b or ^)
    # - followed by end of string or space
    pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _extract_single_number(text):
    """Extracts a single number from text if exactly one exists.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The extracted number as a float if exactly one number exists,
                      otherwise None.

    Explanation:
        1. Uses regex to find all numbers in the text.
        2. Returns the first number as a float if exactly one number is found.
        3. Returns None if zero or multiple numbers are found.
    """
    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None


def evaluate_model(model, tokenizer, eval_examples, device):
    """Evaluates the model on a set of examples and prints detailed results.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.

    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).

    Explanation:
        1. Sets the model to evaluation mode.
        2. For each example in the evaluation set:
           - Encodes the prompt and generates a response using the model.
           - Extracts the predicted answer from the generated response.
           - Compares the predicted answer with the expected answer using multiple methods:
             a. Exact string matching
             b. Single number extraction and comparison
             c. Last number extraction and comparison
           - Prints detailed information about each example.
        3. Calculates and returns the overall accuracy.
        4. Returns the model to training mode.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "=" * 50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("=" * 50)

    for example in eval_examples:
        # Build the full prompt using the same method as training.
        full_prompt = example["prompt"]
        expected = example["answer"]

        # Tokenize the full prompt and generate a response from the model.
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the predicted answer from the model output.
        try:
            predicted = extract_answer_from_model_output(response)

            # Check correctness in multiple ways
            if predicted == expected:  # First try exact match
                is_correct = True
            else:
                # Try single number
                pred_num = _extract_single_number(str(predicted))
                exp_num = _extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number
                    pred_num = _extract_last_number(str(predicted))
                    exp_num = _extract_last_number(str(expected))
                    is_correct = pred_num is not None and exp_num is not None and pred_num == exp_num

            if is_correct:
                correct += 1

            # Print details of the evaluation.
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-" * 50)

        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-" * 50)

    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 50)

    model.train()
    return accuracy

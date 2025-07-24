# GRPO from Scratch - LM Book by Andriy Burkov

Implementation of the **GRPO (Group Relative Policy Optimization)** training algorithm from:
[https://github.com/aburkov/theLMbook/blob/main/GRPO.py](https://github.com/aburkov/theLMbook/blob/main/GRPO.py).

We refactor the original code into a **modular and easy-to-understand layout**.
Each major step of the training process is separated into its own file for clarity and extensibility.

### Steps in the Training Process

1. **Initialization and Setup**:
   - Set random seeds for reproducibility across Python, NumPy, and PyTorch.
   - Load the pre-trained language model (Qwen2.5-0.5B-Instruct) and tokenizer.
   - Configure the model: set pad token to EOS token, enable memory optimizations (gradient checkpointing, disable cache, and ensure input gradients).
   - Prepare the GSM8K dataset: format each example with a system prompt and the question, and extract the answer.
2. **Initial Evaluation**:
   - Evaluate the model's initial performance on a small subset of the training data (before any fine-tuning).
   - The evaluation function generates responses for each prompt, extracts answers, and compares them to the expected answers using multiple methods (exact match, single number extraction, last number extraction).
3. **GRPO Training Loop**:
   - The training is divided into iterations (outer loop). In each iteration:
     - Create a reference model by making a deep copy of the current policy model (and freeze its parameters).
     - Set up an optimizer for the policy model.
     - For a specified number of steps (inner loop):
        - Sample a batch of training examples.
        - Generate multiple completions (rollouts) for each prompt in the batch using the current policy model.
        - For each generated completion, compute the log probabilities under the current policy and the reference model (without gradients).
        - Format the completions for reward computation.
        - Perform multiple GRPO updates (μ times) on the same batch of rollouts:
            - Compute rewards using the combined reward function (correctness and format).
            - Calculate group-relative advantages: within each group of completions for the same prompt, normalize the rewards by subtracting the group mean and dividing by the group standard deviation.
            - Compute the current log probabilities (with gradients) for the generated completions.
            - Calculate the policy ratio (exponential of the difference between current and old log probabilities).
            - Compute the surrogate loss (clipped to avoid large updates) and the KL divergence penalty (to prevent the policy from deviating too far from the reference model).
            - Combine the losses and update the policy model's parameters.
4. **Final Evaluation and Saving**:
   - After completing all iterations, evaluate the fine-tuned model on the same evaluation subset.
   - Calculate the improvement in accuracy.
   - Save the fine-tuned model and tokenizer.

## Code Structure

We refactor the original implementation into a modular, readable, and extensible format. Each component corresponds to a specific phase in the GRPO loop.

```bash
andriy_burkov_lm_book/
├── train_grpo.py         
├── data_process.py      
├── reward_functions.py      
├── evaluation.py  
└── completions.py              
```

- **`prepare_dataset`**:
  - Loads the GSM8K dataset and formats each example into a prompt string (combining system message and user question) and extracts the answer.
- **`evaluate_model`**:
  - Evaluates the model by generating responses for each evaluation example.
  - Extracts the predicted answer and compares it to the expected answer using multiple matching strategies (exact string, single number, last number).
  - Prints detailed results and returns the accuracy.
- **`correctness_reward`**:
  - Assigns a reward (0.0, 1.5, or 2.0) based on the correctness of the generated answer compared to the expected answer. Uses exact matching and numeric equivalence.
- **`format_reward`**:
  - Assigns a reward (up to 0.8) for adhering to the required XML format (presence of `<reasoning>`, `</reasoning>`, `<answer>`, and `</answer>` tags).
- **`combined_reward`**:
  - Sums the correctness reward and format reward for a total reward in the range [0.0, 2.8].
- **`generate_completions`**:
  - Generates multiple completions for each prompt in a batch using the current model.
  - Returns the tokenized prompts and completions, along with masks that ignore tokens after the first end-of-sequence (EOS) token.
- **`generate_rollout_data`**:
  - Uses `generate_completions` to generate rollouts (completions) for a batch of prompts.
  - Computes log probabilities for these completions under both the current policy and the reference model (without gradients).
  - Returns a dictionary containing the rollout data (inputs, masks, log probabilities, etc.).
- **`compute_group_relative_advantages`**:
  - Groups the rewards by prompt (each group has multiple completions for the same prompt).
  - For each group, normalizes the rewards by subtracting the group mean and dividing by the group standard deviation (adding a small epsilon to avoid division by zero).
  - Returns the normalized advantages for each completion.
- **`maximize_grpo_objective`**:
  - The core function that computes the GRPO loss and updates the model.
  - Computes current log probabilities (with gradients).
  - Computes the policy ratio (current log probability divided by old log probability, exponentiated).
  - Computes rewards and then group-relative advantages.
  - Computes the surrogate loss (min of two terms: unclipped and clipped) and the KL penalty.
  - Combines them and performs a gradient update.
- **`train_with_grpo`**:
  - Orchestrates the entire GRPO training process: sets up the reference model, optimizer, and loops over iterations and steps.
  - For each step, it generates rollout data and performs multiple GRPO updates.
- **`optimize_model_memory`**:
  - Applies memory optimization techniques: disables caching, enables gradient checkpointing, and ensures input gradients are required.
- **`main`**:
  - The main function that ties everything together: sets up the model, tokenizer, dataset, runs initial evaluation, performs GRPO training, runs final evaluation, and saves the model.

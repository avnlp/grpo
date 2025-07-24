# nanoAhaMoment

Implementation of the **GRPO (Group Relative Policy Optimization)** training algorithm from:
[https://github.com/McGill-NLP/nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment).

We refactor the original code into a **modular and easy-to-understand layout**.
Each major step of the training process is separated into its own file for clarity and extensibility.

## Code Structure

We refactor the original implementation into a modular, readable, and extensible format. Each component corresponds to a specific phase in the GRPO loop.

```bash
nanoAhaMoment/
├── nano_r1_train.py         # Main RL training loop using GRPO
├── config.py                # Configuration of hyperparameters and model settings
├── episode_creation.py      # Groups sampled generations, computes rewards and advantages
├── reward_functions.py      # Rule-based reward functions (format + equation correctness)
├── policy_gradient_loss.py  # Computes policy gradient loss with KL penalty
└── utils.py                 # Utilities for I/O, logging, inference engine sync, and evaluation
```

1. **Main Training Loop** (`nano_r1_train.py`): Controls the end-to-end RL loop.
   - Dataset preprocessing.  
   - Episode generation using vLLM.  
   - Reward computation.  
   - Loss calculation.  
   - Parameter updates with DeepSpeed.  
   - Logging and checkpointing.  

2. **Configuration** (`config.py`): Contains all model names, dataset paths, GRPO hyperparameters, and DeepSpeed settings in one place.

3. **Episode Generation** (`episode_creation.py`): Groups sampled outputs into episodes and normalizes rewards to derive token-level advantages.

4. **Reward Functions** (`reward_functions.py`): Implements:

   - `format_reward_func` – Enforces `<think>...</think>\n<answer>...</answer>` format.
   - `equation_reward_func` – Checks if the answer is correct and uses all numbers.  

5. **Policy Gradient Training** (`policy_gradient_loss.py`): Calculates loss:

   - KL-divergence with reference model.  
   - Token-wise policy gradient loss.  
   - Entropy logging for debugging.  

6. **Utilities** (`utils.py`): Provides helper functions for:

   - Prompt creation and model input preparation.  
   - Evaluation (`evaluate_on_test_set`).  
   - Inference engine weight updates (`load_model_into_vllm`).  
   - Checkpointing and logging.  

## Objective

Train a base LLM to solve reasoning-heavy algorithmic tasks using GRPO. This implementation follows the training methodology proposed in the **DeepSeek R1** paper.

- **Model**: Qwen2.5 3B-Base
- **Dataset**: [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
- **Algorithm**: GRPO

The `Countdown-Tasks-3to4` dataset is designed to test **reasoning and arithmetic**, with input prompts that only provide a final answer (no reasoning steps).
This setup helps induce behaviors such as:

- Self-reflection
- Verification
- Backtracking
- Even language-switching

## Reward Function

Each response is evaluated with two rule-based reward functions:

- **Format Reward**: Output must follow a structured pattern:

  ```text
  <think> [reasoning steps] </think><answer> [expression] </answer>
  ```

* **Equation Reward**: Validates if the answer expression:

  - Evaluates to the correct target value
  - Uses all input numbers **exactly once**

Final reward = `format_reward + equation_reward`

## Training

1. **Initialization**:  
   Start with a pretrained base LLM and a dataset of prompts paired with final answers.

2. **Main RL Loop**:  
   For each iteration:

   1. **Episode Generation**

      - Sample a batch of prompts

        $$
          \{x_i\}_{i=1}^N.
        $$
      - For each prompt $x_i$, generate a *group* of $G$ responses:

        $$
          y_{i,1},\,y_{i,2},\dots,y_{i,G}\;\sim\;\pi_\theta(y \mid x_i).
        $$

   2. **Reward Computation & Normalization**

      - Compute scalar rewards $R_{i,j}$ for each response $y_{i,j}$.
      - Within each group $i$, compute:

        $$
          \mu_i = \tfrac1G\sum_{j=1}^G R_{i,j},\quad
          \sigma_i = \sqrt{\tfrac1G\sum_{j=1}^G(R_{i,j}-\mu_i)^2},
        $$

        then normalize:

        $$
          R^*_{i,j} = \frac{R_{i,j} - \mu_i}{\sigma_i}.
        $$

   3. **Advantage Assignment**
      Assign the normalized reward as the token‑wise advantage:

      $$
        A_{i,j} \;=\; R^*_{i,j}
        \quad\text{for every token in }y_{i,j}.
      $$

   4. **Policy Gradient Computation**
      Build the episode dataset
      $\{(x_i,\,y_{i,j},\,A_{i,j})\}$
      and compute the gradient estimate:

      $$
        \mathbf{g}_{\mathrm{pg}}
        = \mathbb{E}_{(x,y,A)}\bigl[\nabla_\theta \log \pi_\theta(y \mid x)\;\cdot\;A\bigr].
      $$

   5. **Parameter Update**

      $$
        \theta \;\leftarrow\; \theta + \eta \,\mathbf{g}_{\mathrm{pg}}.
      $$

   6. **Inference Engine Sync**
      Wake up the inference engine and load the updated $\theta$.

3. **Evaluation, Logging & Checkpointing**:  

   - **Evaluation:** Every few iterations, evaluate on a held‑out test set to monitor progress.
   - **Logging:** Record training and evaluation metrics in Weights & Biases.
   - **Checkpointing:** Save model + optimizer states every 50 iterations.

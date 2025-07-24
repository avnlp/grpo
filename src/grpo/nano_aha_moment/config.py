# This code is based on the implementation from: https://github.com/McGill-NLP/nano-aha-moment

# Training Hyperparameters

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"

# Dataset configuration
DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"

# Total number of training iterations
NUM_ITERATIONS = 1000
# Number of episodes to collect per iteration for training
EPISODES_PER_ITERATION = 64
# Number of responses to generate for each input prompt (i.e. group size in GRPO)
GENERATIONS_PER_SAMPLE = 4
# Controls how much the policy can deviate from the reference model
KL_COEFFICIENT = 0.001

# Training hyperparameters
# Batch size for each GPU device during training
PER_DEVICE_BATCH_SIZE = 4
# Learning rate for model updates
LEARNING_RATE = 1e-6

# Sampling parameters
# Maximum number of tokens to generate in each response
MAX_RESPONSE_TOKENS = 1024
# Controls randomness in generation (higher = more random)
TEMPERATURE = 1.0
# Nucleus sampling parameter (1.0 = disabled)
TOP_P = 1.0
# Top-k sampling parameter (-1 = disabled)
TOP_K = -1  # no top k

# DeepSpeed configuration
# DeepSpeed config for the policy model
deepspeed_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 2, "overlap_comm": False},
    "train_batch_size": EPISODES_PER_ITERATION,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
    "gradient_clipping": 1.0,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": LEARNING_RATE,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "torch_adam": True,
        },
    },
}
# DeepSpeed config for the reference model
ref_deepspeed_config = {
    "bf16": {"enabled": True},
    # Note that we don't train the reference model
    # These are just for compatibility with DeepSpeed.
    "train_batch_size": EPISODES_PER_ITERATION,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
}

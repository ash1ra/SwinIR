from pathlib import Path
from typing import Literal, TypeAlias

ScalingFactor: TypeAlias = Literal[2, 4, 8]
DeviceType: TypeAlias = Literal["cuda", "cpu"]

# Architecture settings
HIDDEN_CHANNELS = 180
NUM_RSTB_BLOCKS = 6
NUM_STL_BLOCKS = 6
NUM_HEADS = 6
MLP_RATIO = 2
WINDOW_SIZE = 8
USE_GRADIENT_CHECKPOINTING = True

# Training settings
SCALING_FACTOR = 4
PATCH_SIZE = 48
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 1
NUM_ITERATIONS = 100_000
VAL_FREQ = 5000
LOG_FREQ = 10
GRADIENT_CLIPPING_NORM = 0.5

# Optimizer settings
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8

# Scheduler settings
SCHEDULER_MILESTONES = [50_000, 80_000, 90_000, 95_000]
SCHEDULER_GAMMA = 0.5

# Data loader settings
DATASET_REPEATS = 100
TRAIN_NUM_WORKERS = 8
TRAIN_PREFETCH_FACTOR = 4
VAL_NUM_WORKERS = 2
VAL_PREFETCH_FACTOR = 2

# Dataset pathes
TRAIN_DATASET_PATH = Path("data/DF2K")
VAL_DATASET_PATH = Path("data/DIV2K_val")
TEST_DATASET_PATHS = [
    Path("data/Set5.txt"),
    Path("data/Set14.txt"),
    Path("data/BSDS100.txt"),
    Path("data/Urban100.txt"),
    Path("data/Manga109.txt"),
]

# Checkpoint settings
LOAD_BEST_CHECKPOINT = False
LOAD_CHECKPOINT = False

BEST_CHECKPOINT_DIR_PATH = Path("checkpoints/best")
CHECKPOINT_DIR_PATH = Path("checkpoints/iter_0")

# WandB settings
USE_WANDB = True
WANDB_PROJECT_NAME = "SwinIR-SR"
WANDB_CONFIG = {
    "hidden_channels": HIDDEN_CHANNELS,
    "num_rstb_blocks": NUM_RSTB_BLOCKS,
    "num_stl_blocks": NUM_STL_BLOCKS,
    "num_heads": NUM_HEADS,
    "mlp_ratio": MLP_RATIO,
    "window_size": WINDOW_SIZE,
    "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "scaling_factor": SCALING_FACTOR,
    "patch_size": PATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "batch_size": TRAIN_BATCH_SIZE,
    "num_iteration": NUM_ITERATIONS,
    "gradient_clipping_norm": GRADIENT_CLIPPING_NORM,
    "train_num_workers": TRAIN_NUM_WORKERS,
    "train_prefetch_factor": TRAIN_PREFETCH_FACTOR,
}

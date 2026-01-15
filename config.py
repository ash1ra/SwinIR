from utils import create_logger
from typing import TypeAlias, Literal
from pathlib import Path

ScalingFactor: TypeAlias = Literal[2, 4, 8]

# Architecture settings
HIDDEN_CHANNELS = 96
NUM_RSTB_BLOCKS = 6
NUM_STL_BLOCKS = 6
NUM_HEADS = 6
MLP_RATIO = 4
WINDOW_SIZE = 8
USE_CHECKPOINT = True

# Training settings
SCALING_FACTOR = 4
PATCH_SIZE = 64
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
NUM_ITERATIONS = 500_000
VAL_FREQ = 5000
LOG_FREQ = 100

# Optimizer settings
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8

# Scheduler settings
SCHEDULER_MILESTONES = [250_000, 400_000, 450_000, 475_000]
SCHEDULER_GAMMA = 0.5

# Dataloader settings
TRAIN_NUM_WORKERS = 8
TRAIN_PREFETCH_FACTOR = 4
VAL_NUM_WORKERS = 0
VAL_PREFETCH_FACTOR = None

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
CHECKPOINT_DIR_PATH = Path("checkpoints/iter_5000")


logger = create_logger(log_level="INFO", log_file_name="SwinIR")

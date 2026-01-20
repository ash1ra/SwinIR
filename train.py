from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import config
from dataset import SRDataset
from models import SwinIR
from trainer import Trainer
from utils import InfiniteDataLoader, logger


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SRDataset(
        data_path=config.TRAIN_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        patch_size=config.PATCH_SIZE,
        test_mode=False,
        dev_mode=False,
    )

    val_dataset = SRDataset(
        data_path=config.VAL_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        patch_size=config.PATCH_SIZE,
        test_mode=True,
        dev_mode=True,
    )

    train_dataloader = InfiniteDataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN_NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=config.TRAIN_PREFETCH_FACTOR,
        persistent_workers=True if config.TRAIN_NUM_WORKERS > 0 else False,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.VAL_NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=config.VAL_PREFETCH_FACTOR,
        persistent_workers=True if config.VAL_NUM_WORKERS > 0 else False,
    )

    model = SwinIR(
        in_channels=3,
        hidden_channels=config.HIDDEN_CHANNELS,
        train_img_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        num_rstb_blocks=config.NUM_RSTB_BLOCKS,
        num_stl_blocks=config.NUM_STL_BLOCKS,
        num_heads=config.NUM_HEADS,
        window_size=config.WINDOW_SIZE,
        mlp_ratio=config.MLP_RATIO,
        upscale=config.SCALING_FACTOR,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
    )

    loss_fn = nn.L1Loss()

    optimizer = Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.ADAM_BETAS,
        eps=config.ADAM_EPS,
    )

    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config.SCHEDULER_MILESTONES,
        gamma=config.SCHEDULER_GAMMA,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dir_path=Path(""),
        scaling_factor=config.SCALING_FACTOR,
        num_iters=config.NUM_ITERATIONS,
        val_freq=config.VAL_FREQ,
        log_freq=config.LOG_FREQ,
        scheduler=scheduler,
        device=device,
        dtype=torch.bfloat16,
    )

    if config.LOAD_BEST_CHECKPOINT and config.BEST_CHECKPOINT_DIR_PATH.exists():
        trainer.load_checkpoint(config.BEST_CHECKPOINT_DIR_PATH)
        config.logger.info("Resuming training from best model checkpoint.")
    elif config.LOAD_CHECKPOINT and config.CHECKPOINT_DIR_PATH.exists():
        trainer.load_checkpoint(config.CHECKPOINT_DIR_PATH)
        config.logger.info(f"Resuming training from iteration #{trainer.current_iter}.")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by used. Saving last state...")
        trainer.save_checkpoint(is_best=False)


if __name__ == "__main__":
    main()

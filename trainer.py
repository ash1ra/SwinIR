from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from thop import profile
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import config
import wandb
from utils import Timer, calculate_psnr, calculate_ssim, format_time, logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        dir_path: Path,
        scaling_factor: config.ScalingFactor,
        num_iters: int,
        val_freq: int = 5000,
        log_freq: int = 100,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        device: config.DeviceType = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scaling_factor = scaling_factor
        self.num_iters = num_iters
        self.val_freq = val_freq
        self.log_freq = log_freq
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.dtype = dtype

        self.dir_path = dir_path
        self.checkpoints_dir_path = self.dir_path / "checkpoints"
        self.imgs_dir_path = self.dir_path / "images"
        self.logs_dir_path = self.dir_path / "logs"

        for dir in [self.checkpoints_dir_path, self.imgs_dir_path, self.logs_dir_path]:
            dir.mkdir(parents=True, exist_ok=True)

        self.timer = Timer(device=device)
        self.avg_iter_time = 0.0

        self.current_iter = 0
        self.best_psnr = float("-inf")

    def _log_model_info(self) -> None:
        dummy_input = torch.randn(1, 3, config.PATCH_SIZE, config.PATCH_SIZE).to(self.device)

        self.model.eval()
        flops, _ = profile(model=self.model, inputs=(dummy_input,), verbose=False)
        self.model.train()

        for module in self.model.modules():
            if hasattr(module, "total_ops"):
                del module.total_ops
            if hasattr(module, "total_params"):
                del module.total_params

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        dash_line = "-" * 60

        logger.info(dash_line)
        logger.info("Model & Training Information")
        logger.info(dash_line)
        logger.info(f"Model Architecture: {self.model.__class__.__name__}")
        logger.info(f"Total Parameters: {total_params / 1e6:.2f} M")
        logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
        logger.info(f"GFLOPs (per patch): {flops / 1e9:.4f} G")

        if self.device == "cuda":
            num_gpus = torch.cuda.device_count()
            logger.info(f"Device: CUDA ({num_gpus} GPUs active)")

            for i in range(num_gpus):
                gpu = torch.cuda.get_device_properties(i)
                total_memory = gpu.total_memory / (1024**3)
                logger.info(f"   [{i}] {gpu.name} ({total_memory:.1f} GB VRAM)")
        else:
            logger.info("Device: CPU")

        logger.info(dash_line)
        logger.info("Architecture Settings")
        logger.info(dash_line)
        logger.info(f"Hidden Channels: {config.HIDDEN_CHANNELS}")
        logger.info(f"RSTB Blocks: {config.NUM_RSTB_BLOCKS}")
        logger.info(f"STL Blocks: {config.NUM_STL_BLOCKS}")
        logger.info(f"Attention Heads: {config.NUM_HEADS}")
        logger.info(f"Window Size: {config.WINDOW_SIZE}")
        logger.info(f"MLP Ratio: {config.MLP_RATIO}")

        logger.info(dash_line)
        logger.info("Hyperparameters")
        logger.info(dash_line)
        logger.info(f"Scaling Factor: x{self.scaling_factor}")
        logger.info(f"Patch Size: {config.PATCH_SIZE}x{config.PATCH_SIZE}")
        logger.info(f"Batch Size: {config.TRAIN_BATCH_SIZE}")
        logger.info(f"Total iteration: {self.num_iters:,}")
        logger.info(f"Loss Function: {self.loss_fn.__class__.__name__}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")

        if isinstance(self.optimizer, torch.optim.Adam):
            logger.info(f"  - Betas: {config.ADAM_BETAS}")
            logger.info(f"  - Epsilon: {config.ADAM_EPS}")

        logger.info(f"Initial Learning Rate: {config.LEARNING_RATE}")
        logger.info(f"Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")

        if self.scheduler:
            logger.info(f"  - Milestones: {config.SCHEDULER_MILESTONES}")
            logger.info(f"  - Gamma: {config.SCHEDULER_GAMMA}")

        logger.info(f"Scaler: {'Enabled' if self.scaler else 'Disabled'}")
        logger.info(f"Precision (Data Type): {self.dtype}")
        logger.info(f"Gradient Clipping: {config.GRADIENT_CLIPPING_NORM}")
        logger.info(f"Gradient Checkpointing: {config.USE_GRADIENT_CHECKPOINTING}")

        logger.info(dash_line)
        logger.info("Data Processing")
        logger.info(dash_line)
        logger.info(f"Training Workers: {config.TRAIN_NUM_WORKERS} (Prefetch: {config.TRAIN_PREFETCH_FACTOR})")
        logger.info(f"Val Workers: {config.VAL_NUM_WORKERS} (Prefetch: {config.VAL_PREFETCH_FACTOR})")
        logger.info(dash_line)

    def _update_avg_time(self) -> None:
        alpha = 0.1

        if self.avg_iter_time == 0.0:
            self.avg_iter_time = self.timer.last_iter_duration
        else:
            self.avg_iter_time = (1 - alpha) * self.avg_iter_time + alpha * self.timer.last_iter_duration

    def _log_progress(self, loss: float) -> None:
        elapsed_time = format_time(self.timer.get_elapsed_time())
        remaining_time = format_time((self.num_iters - self.current_iter) * self.avg_iter_time)

        current_lr = self.optimizer.param_groups[0]["lr"]

        logger.info(
            (
                f"Iter: [{self.current_iter:>6d}/{self.num_iters}] "
                f"({format_time(self.avg_iter_time)} / {elapsed_time} / {remaining_time})"
            )
        )
        logger.info(f"Loss: {loss:.4f} | LR: {current_lr:.2e}")

        if config.USE_WANDB:
            wandb.log(
                {
                    "train/loss": loss,
                    "train/lr": current_lr,
                    "train/iteration": self.current_iter,
                },
                step=self.current_iter,
            )

    def _log_images(self, lr_img_tensor: Tensor, sr_img_tensor: Tensor, hr_img_tensor: Tensor) -> None:
        lr_img_tensor = lr_img_tensor.float().cpu().clamp(0, 1)
        sr_img_tensor = sr_img_tensor.float().cpu().clamp(0, 1)
        hr_img_tensor = hr_img_tensor.float().cpu().clamp(0, 1)

        lr_img_tensor_resized = torch.nn.functional.interpolate(
            input=lr_img_tensor.unsqueeze(0),
            size=(hr_img_tensor.shape[1], hr_img_tensor.shape[2]),
            mode="nearest",
        ).squeeze(0)

        combined_img_tensor = torch.cat([lr_img_tensor_resized, sr_img_tensor, hr_img_tensor], dim=2)

        wandb.log(
            {
                "val/visual_results": wandb.Image(
                    data_or_path=combined_img_tensor,
                    caption=f"Iter {self.current_iter}: LR (Nearest) | SR ({self.model.__class__.__name__}) | HR(Truth)",
                )
            },
            step=self.current_iter,
        )

    def _train_step(self, batch: dict[str, Tensor]) -> float:
        lr_img_tensor = batch["lr"].to(self.device, non_blocking=True)
        hr_img_tensor = batch["hr"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=True):
            sr_img_tensor = self.model(lr_img_tensor)
            loss = self.loss_fn(sr_img_tensor, hr_img_tensor)

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=config.GRADIENT_CLIPPING_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=config.GRADIENT_CLIPPING_NORM)
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    def train(self):
        self._log_model_info()

        if self.current_iter > 0:
            logger.info(
                f"Resuming training on {self.device} device from iteration {self.current_iter:,} / {self.num_iters:,}."
            )
        else:
            logger.info(f"Starting training on {self.device} device for {self.num_iters:,} iterations.")

        self.model.train()

        for batch in self.train_dataloader:
            with self.timer:
                loss = self._train_step(batch)

                if self.current_iter % self.val_freq == 0 and self.current_iter != 0:
                    logger.info("Validation started (it may take a few minutes)...")
                    self.validate()

                if self.current_iter % config.SAVE_CHECKPOINT_FREQ == 0 and self.current_iter != 0:
                    self.save_checkpoint(is_best=False)

            self.current_iter += 1
            self._update_avg_time()

            if self.current_iter % self.log_freq == 0:
                self._log_progress(loss)

            if self.current_iter >= self.num_iters:
                break

        self.save_checkpoint(is_best=False)
        logger.info("Training run completed successfully.")

    @torch.inference_mode()
    def validate(self):
        self.model.eval()

        avg_loss, avg_psnr, avg_ssim = 0.0, 0.0, 0.0

        for i, batch in enumerate(self.val_dataloader):
            lr_img_tensor = batch["lr"].to(self.device, non_blocking=True)
            hr_img_tensor = batch["hr"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=True):
                sr_img_tensor = self.model(lr_img_tensor)
                loss = self.loss_fn(sr_img_tensor, hr_img_tensor)
                avg_loss += loss.item()

            batch_psnr, batch_ssim = 0.0, 0.0
            batch_size = sr_img_tensor.size(0)

            for j in range(batch_size):
                sr_img_item = sr_img_tensor[j].float()
                hr_img_item = hr_img_tensor[j].float()

                batch_psnr += calculate_psnr(
                    sr_img_tensor=sr_img_item,
                    hr_img_tensor=hr_img_item,
                    crop_border=self.scaling_factor,
                )

                batch_ssim += calculate_ssim(
                    sr_img_tensor=sr_img_item,
                    hr_img_tensor=hr_img_item,
                    crop_border=self.scaling_factor,
                )

            avg_psnr += batch_psnr / batch_size
            avg_ssim += batch_ssim / batch_size

            if config.USE_WANDB and i == 0:
                self._log_images(
                    lr_img_tensor=lr_img_tensor[0],
                    sr_img_tensor=sr_img_tensor[0],
                    hr_img_tensor=hr_img_tensor[0],
                )

        self.model.train()

        avg_loss /= len(self.val_dataloader)
        avg_psnr /= len(self.val_dataloader)
        avg_ssim /= len(self.val_dataloader)

        logger.info(
            f"Validation | Iter: {self.current_iter} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}"
        )

        if config.USE_WANDB:
            wandb.log(
                {
                    "val/loss": avg_loss,
                    "val/psnr": avg_psnr,
                    "val/ssim": avg_ssim,
                    "val/iteration": self.current_iter,
                },
                step=self.current_iter,
            )

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint(is_best=True)

    def save_checkpoint(self, is_best: bool) -> None:
        model_state = self.model.state_dict()

        train_state = {
            "current_iter": self.current_iter,
            "best_psnr": self.best_psnr,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "wandb_id": wandb.run.id if wandb.run else None,
        }

        if is_best:
            save_dir = self.checkpoints_dir_path / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Checkpoint] New best model saved (PSNR: {self.best_psnr:.2f} dB).")
        else:
            save_dir = self.checkpoints_dir_path / f"iter_{self.current_iter}"
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Checkpoint] Checkpoint saved to '{save_dir}'.")

        save_file(model_state, save_dir / "model.safetensors")
        torch.save(train_state, save_dir / "state.pth")

    def load_checkpoint(self, checkpoint_dir_path: Path) -> None:
        model_path = checkpoint_dir_path / "model.safetensors"

        if model_path.exists():
            checkpoint_state_dict = load_file(model_path, device=self.device)

            try:
                self.model.load_state_dict(checkpoint_state_dict)
            except RuntimeError:
                logger.error("Architecture mismatch during loading! Raising error.")
                raise

            logger.info(f"[Checkpoint] Model weights loaded successfully from {checkpoint_dir_path.name}.")
        else:
            logger.warning(f"[Checkpoint] Model weights file not found at '{model_path}'")

        state_path = checkpoint_dir_path / "state.pth"

        if state_path.exists():
            state = torch.load(checkpoint_dir_path / "state.pth", map_location=self.device)

            self.current_iter = state["current_iter"]
            self.best_psnr = state["best_psnr"]

            try:
                self.optimizer.load_state_dict(state["optimizer"])

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = config.LEARNING_RATE
                    if "betas" in param_group:
                        param_group["betas"] = config.ADAM_BETAS

            except Exception:
                logger.warning("[Checkpoint] Optimizer type mismatch. Skipping the optimizer parameter loading.")

            if self.scheduler and state["scheduler"]:
                self.scheduler.load_state_dict(state["scheduler"])

            if self.scaler and state["scaler"]:
                self.scaler.load_state_dict(state["scaler"])

            logger.info("[Checkpoint] Training state loaded successfully.")
        else:
            logger.warning(f"[Checkpoint] Model state file not found at '{state_path}'")

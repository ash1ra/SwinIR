from pathlib import Path
from typing import Literal, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import config
from utils import calculate_psnr, calculate_ssim


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
        device: Literal["cpu", "cuda"] = "cpu",
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

        self.writer = SummaryWriter(log_dir=str(self.logs_dir_path))

        self.current_iter = 0
        self.best_psnr = 0.0

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
            clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    def train(self):
        config.logger.info(f"Starting training on {self.device} using {self.dtype} for {self.num_iters} iterations...")
        self.model.train()

        pbar = tqdm(total=self.num_iters, initial=self.current_iter, unit="iter")

        for batch in self.train_dataloader:
            self.current_iter += 1

            loss = self._train_step(batch)

            if self.current_iter % self.log_freq == 0:
                self._log_metrics(loss)
                pbar.set_description(f"Iter: {self.current_iter} | Loss: {loss:.4f}")

            pbar.update(1)

            if self.current_iter % self.val_freq == 0:
                self.validate()
                self.save_checkpoint(is_best=False)

            if self.current_iter >= self.num_iters:
                break

        pbar.close()
        self.writer.close()
        self.save_checkpoint(is_best=False)
        config.logger.info("Training finished.")

    def _log_metrics(self, loss: float) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Train/Loss", loss, self.current_iter)
        self.writer.add_scalar("Train/LR", current_lr, self.current_iter)

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        avg_psnr = 0.0
        avg_ssim = 0.0

        vis_lr, vis_sr, vis_hr = None, None, None

        for i, batch in enumerate(self.val_dataloader):
            lr_img_tensor = batch["lr"].to(self.device, non_blocking=True)
            hr_img_tensor = batch["hr"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=True):
                sr_img_tensor = self.model(lr_img_tensor)

            batch_psnr = 0.0
            batch_ssim = 0.0
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

            if i == 0:
                vis_lr = lr_img_tensor[0].detach().cpu().float()
                vis_sr = sr_img_tensor[0].detach().cpu().float().clamp(0, 1)
                vis_hr = hr_img_tensor[0].detach().cpu().float()

        avg_psnr /= len(self.val_dataloader)
        avg_ssim /= len(self.val_dataloader)

        self.writer.add_scalar("Val/PSNR", avg_psnr, self.current_iter)
        self.writer.add_scalar("Val/SSIM", avg_ssim, self.current_iter)

        config.logger.info(f"Validation | Iter: {self.current_iter} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        if vis_sr is not None:
            self._log_images(vis_lr, vis_sr, vis_hr)

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint(is_best=True)

    def _log_images(self, lr_img_tensor: Tensor, sr_img_tensor: Tensor, hr_img_tensor: Tensor) -> None:
        lr_up = F.interpolate(lr_img_tensor.unsqueeze(0), scale_factor=self.scaling_factor, mode="nearest").squeeze(0)

        grid = make_grid([lr_up, sr_img_tensor, hr_img_tensor], nrow=3, padding=5, normalize=False)
        self.writer.add_image("Val_Images", grid, self.current_iter)

    def save_checkpoint(self, is_best: bool) -> None:
        model_state = self.model.state_dict()

        train_state = {
            "current_iter": self.current_iter,
            "best_psnr": self.best_psnr,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

        if is_best:
            best_model_dir = self.checkpoints_dir_path / "best"
            best_model_dir.mkdir(parents=True, exist_ok=True)

            save_file(model_state, best_model_dir / "model.safetensors")
            torch.save(train_state, best_model_dir / "state.pth")

            config.logger.info(f"Saved best model (PSNR: {self.best_psnr:.2f})")
        else:
            current_iter_dir = self.checkpoints_dir_path / f"iter_{self.current_iter}"
            current_iter_dir.mkdir(parents=True, exist_ok=True)

            save_file(model_state, current_iter_dir / "model.safetensors")
            torch.save(train_state, current_iter_dir / "state.pth")

            config.logger.info(f"Saved checkpoint to '{current_iter_dir}'")

    def load_checkpoint(self, checkpoint_dir_path: Path) -> None:
        model_path = checkpoint_dir_path / "model.safetensors"

        if model_path:
            self.model.load_state_dict(load_file(model_path, device=self.device))

            config.logger.info("Model weights was successfully loaded.")
        else:
            config.logger.warning(f"Model path not found at '{model_path}'")

        state_path = checkpoint_dir_path / "state.pth"

        if state_path:
            state = torch.load(checkpoint_dir_path / "state.pth", map_location=self.device)

            self.current_iter = state["current_iter"]
            self.best_psnr = state["best_psnr"]

            self.optimizer.load_state_dict(state["optimizer"])

            if self.scheduler and state["scheduler"]:
                self.scheduler.load_state_dict(state["scheduler"])

            if self.scaler and state["scaler"]:
                self.scaler.load_state_dict(state["scaler"])

            config.logger.info("Model state was successfully loaded.")
        else:
            config.logger.warning(f"State path not found at '{state_path}'")

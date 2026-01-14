import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from math import exp
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)  # type: ignore
        except StopIteration:
            self._iterator = super().__iter__()
            batch = next(self._iterator)
        return batch


def create_logger(
    log_level: str,
    log_file_name: str,
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    Path("logs").mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_file_name = f"logs/{log_file_name}_{current_date}.log"

    file_handler = RotatingFileHandler(
        filename=log_file_name,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def split_img_into_windows(img_tensor: Tensor, window_size: int) -> Tensor:
    """
    Einops variables:
        b = batch size
        nw_h = number of windows (height)
        nw_w = number of windows (width)
        ws_h = window size (height)
        ws_w = window size (width)
        c = channels
    """
    return rearrange(
        img_tensor,
        "b (nw_h ws_h) (nw_w ws_w) c -> (b nw_h nw_w) ws_h ws_w c",
        ws_h=window_size,
        ws_w=window_size,
    )


def combine_windows_into_img(windows_tensor: Tensor, img_height: int, img_width: int) -> Tensor:
    """
    Einops variables:
        b = batch size
        nw_h = number of windows (height)
        nw_w = number of windows (width)
        ws_h = window size (height)
        ws_w = window size (width)
        c = channels
    """
    return rearrange(
        windows_tensor,
        "(b nw_h nw_w) ws_h ws_w c -> b (nw_h ws_h) (nw_w ws_w) c",
        nw_h=img_height // windows_tensor.shape[1],
        nw_w=img_width // windows_tensor.shape[2],
    )


def rgb2y(img_tensor: Tensor) -> Tensor:
    weights = torch.tensor(
        [65.481, 128.553, 24.966],
        device=img_tensor.device,
        dtype=img_tensor.dtype,
    ).view(1, 3, 1, 1)

    bias = 16.0

    return (torch.sum(img_tensor * weights, dim=1, keepdim=True) + bias) / 255.0


def calculate_psnr(sr_img_tensor: Tensor, hr_img_tensor: Tensor, crop_border: int) -> float:
    sr_img_tensor = rgb2y(sr_img_tensor)
    hr_img_tensor = rgb2y(hr_img_tensor)

    sr_img_tensor *= 255.0
    hr_img_tensor *= 255.0

    sr_img_tensor.round_()
    hr_img_tensor.round_()

    if crop_border > 0:
        sr_img_tensor = sr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        hr_img_tensor = hr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((sr_img_tensor - hr_img_tensor) ** 2)

    if mse == 0:
        return float("inf")
    else:
        return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()


def create_window_for_ssim_metric(window_size: int, sigma: float, channel: int) -> Tensor:
    window_1d = torch.tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    window_1d /= window_1d.sum()

    window_2d = window_1d.unsqueeze(1) @ window_1d.unsqueeze(0)

    window = window_2d.float().unsqueeze(0).unsqueeze(0)

    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim_metric(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    window_size: int,
    window: Optional[Tensor] = None,
    return_map: bool = False,
):
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    if window is None:
        window = (
            create_window_for_ssim_metric(
                window_size=window_size,
                sigma=1.5,
                channel=sr_img_tensor.size(1),
            )
            .to(sr_img_tensor.device)
            .type_as(sr_img_tensor)
        )

    mu1 = F.conv2d(
        input=sr_img_tensor,
        weight=window,
        stride=1,
        padding=0,
        groups=sr_img_tensor.size(1),
    )
    mu2 = F.conv2d(
        input=hr_img_tensor,
        weight=window,
        stride=1,
        padding=0,
        groups=hr_img_tensor.size(1),
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            input=sr_img_tensor * sr_img_tensor,
            weight=window,
            stride=1,
            padding=0,
            groups=sr_img_tensor.size(1),
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            input=hr_img_tensor * hr_img_tensor,
            weight=window,
            stride=1,
            padding=0,
            groups=hr_img_tensor.size(1),
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            input=sr_img_tensor * hr_img_tensor,
            weight=window,
            stride=1,
            padding=0,
            groups=sr_img_tensor.size(1),
        )
        - mu1_mu2
    )

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    mean_metric = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    ssim_map = mean_metric * contrast_metric
    ssim_value = ssim_map.mean().item()

    if return_map:
        return ssim_value, ssim_map
    else:
        return ssim_value


def calculate_ssim(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    crop_border: int,
    window_size: int = 11,
    return_map: bool = False,
) -> float | tuple[float, Tensor]:
    sr_img_tensor = rgb2y(sr_img_tensor)
    hr_img_tensor = rgb2y(hr_img_tensor)

    sr_img_tensor *= 255.0
    hr_img_tensor *= 255.0

    if crop_border > 0:
        sr_img_tensor = sr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        hr_img_tensor = hr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]

    if sr_img_tensor.size(-1) < window_size or sr_img_tensor.size(-2) < window_size:
        return 0.0

    if sr_img_tensor.dim() == 3:
        sr_img_tensor = sr_img_tensor.unsqueeze(0)
        hr_img_tensor = hr_img_tensor.unsqueeze(0)

    return ssim_metric(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        window_size=window_size,
        return_map=return_map,
    )

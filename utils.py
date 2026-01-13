import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

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

import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2 as transforms

import config


class SRDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        scaling_factor: config.ScalingFactor,
        patch_size: int,
        test_mode: bool = False,
        dev_mode: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.patch_size = patch_size
        self.test_mode = test_mode
        self.dev_mode = dev_mode

        self.hr_dir_path = data_path / "HR"
        self.lr_dir_path = data_path / f"LR_x{scaling_factor}"

        if not self.hr_dir_path.exists() or not self.lr_dir_path.exists():
            raise FileNotFoundError(f"Datasets directories not found in '{data_path}'")

        hr_img_names = {hr_img.name for hr_img in self.hr_dir_path.glob("*.png")}
        lr_img_names = {lr_img.name for lr_img in self.lr_dir_path.glob("*.png")}

        self.img_names = sorted(list(hr_img_names & lr_img_names))

        if len(self.img_names) == 0:
            raise FileNotFoundError(f"No matching files found between '{self.hr_dir_path}' and '{self.lr_dir_path}'")

        if len(hr_img_names) != len(lr_img_names):
            config.logger.warning(
                f"Mismatch in file counts! HR: {len(hr_img_names)}, LR: {len(lr_img_names)}. "
                f"Using {len(self.img_names)} common files."
            )

        if dev_mode:
            self.img_names = self.img_names[: int(len(self.img_names) * 0.1)]

        self.normalize = transforms.ToDtype(torch.float32, scale=True)

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_name = self.img_names[index]

        hr_img_path = str(self.hr_dir_path / img_name)
        lr_img_path = str(self.lr_dir_path / img_name)

        try:
            hr_img_tensor = decode_image(hr_img_path, mode=ImageReadMode.RGB)
            lr_img_tensor = decode_image(lr_img_path, mode=ImageReadMode.RGB)
        except Exception as e:
            if self.test_mode:
                config.logger.error(f"Error loading '{img_name}'.")
                raise e
            else:
                config.logger.warning(f"Error loading '{img_name}'. Using random sample instead.")
                return self.__getitem__(random.randint(0, len(self) - 1))

        hr_img_tensor = self.normalize(hr_img_tensor)
        lr_img_tensor = self.normalize(lr_img_tensor)

        if self.test_mode:
            return {"hr": hr_img_tensor, "lr": lr_img_tensor}

        _, img_height, img_width = lr_img_tensor.shape

        if img_height <= self.patch_size or img_width <= self.patch_size:
            config.logger.warning(
                f"Size of the image '{img_name}' ({img_height}x{img_width}) is less than patch size ({self.patch_size}). Using random sample instead."
            )
            return self.__getitem__(random.randint(0, len(self) - 1))

        lr_y = random.randint(0, img_height - self.patch_size)
        lr_x = random.randint(0, img_width - self.patch_size)

        hr_y = lr_y * self.scaling_factor
        hr_x = lr_x * self.scaling_factor
        hr_patch_size = self.patch_size * self.scaling_factor

        lr_patch = lr_img_tensor[:, lr_y : lr_y + self.patch_size, lr_x : lr_x + self.patch_size]
        hr_patch = hr_img_tensor[:, hr_y : hr_y + hr_patch_size, hr_x : hr_x + hr_patch_size]

        if random.random() < 0.5:
            lr_patch = transforms.functional.hflip(lr_patch)
            hr_patch = transforms.functional.hflip(hr_patch)

        if random.random() < 0.5:
            lr_patch = transforms.functional.vflip(lr_patch)
            hr_patch = transforms.functional.vflip(hr_patch)

        k = random.randint(0, 3)
        if k > 0:
            lr_patch = torch.rot90(lr_patch, k, [1, 2])
            hr_patch = torch.rot90(hr_patch, k, [1, 2])

        return {"hr": hr_patch, "lr": lr_patch}

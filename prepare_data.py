import concurrent.futures
from functools import partial
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import config
from utils import logger


def process_single_image(
    img_path: Path,
    hr_dir: Path,
    lr_dir: Path,
    scaling_factor: config.ScalingFactor,
) -> None:
    img_name = f"{img_path.stem}.png"
    hr_img_output_path = hr_dir / img_name
    lr_img_output_path = lr_dir / img_name

    if hr_img_output_path.exists() and lr_img_output_path.exists():
        return

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")

            img_width, img_height = img.size

            remainder_w = img_width % scaling_factor
            remainder_h = img_height % scaling_factor

            if remainder_w != 0 or remainder_h != 0:
                img = img.crop((0, 0, img_width - remainder_w, img_height - remainder_h))

            img.save(hr_img_output_path, compress_level=1)

            lr_img = img.resize(
                (img.width // scaling_factor, img.height // scaling_factor), resample=Image.Resampling.BICUBIC
            )
            lr_img.save(lr_img_output_path, compress_level=1)

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")


def prepare_data(
    input_data_path: Path,
    output_data_path: Path,
    scaling_factor: config.ScalingFactor,
    num_workers: int | None = None,
) -> None:
    output_data_path.mkdir(parents=True, exist_ok=True)

    hr_dir_output_path = output_data_path / "HR"
    hr_dir_output_path.mkdir(parents=True, exist_ok=True)

    lr_dir_output_path = output_data_path / f"LR_x{scaling_factor}"
    lr_dir_output_path.mkdir(parents=True, exist_ok=True)

    if input_data_path.exists():
        if input_data_path.is_dir():
            logger.info(f"Reading images from directory ({input_data_path})...")
            img_paths = sorted([p for p in input_data_path.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        elif input_data_path.is_file():
            logger.info(f"Reading images from file ({input_data_path})...")
            with open(input_data_path, "r") as f:
                img_paths = sorted([Path(line.strip()) for line in f if line.strip()])
    else:
        raise FileNotFoundError(f"Input data path {input_data_path} not found.")

    logger.info(f"Found {len(img_paths)} images. Starting multiprocessing...")

    worker = partial(
        process_single_image, hr_dir=hr_dir_output_path, lr_dir=lr_dir_output_path, scaling_factor=scaling_factor
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(worker, img_paths), total=len(img_paths), desc="Processing images..."))


if __name__ == "__main__":
    prepare_data(
        input_data_path=Path("data/DIV2K_val.txt"),
        output_data_path=Path("data/DIV2K_val/"),
        scaling_factor=4,
    )

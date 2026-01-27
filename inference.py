import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.io import ImageReadMode, decode_image
from torchvision.utils import save_image
from tqdm import tqdm

import config
from models import SwinIR
from utils import logger


def tiled_inference(
    model: nn.Module,
    input_img_tensor: Tensor,
    scaling_factor: int,
    tile_size: int,
    tile_overlap: int,
    device: config.DeviceType,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_channels, input_img_height, input_img_width = input_img_tensor.shape

    output_img_height = input_img_height * scaling_factor
    output_img_width = input_img_width * scaling_factor
    output_shape = (num_channels, output_img_height, output_img_width)

    output_accumulated_values = torch.zeros(output_shape, dtype=torch.float32, device="cpu")
    output_weight_map = torch.zeros(output_shape, dtype=torch.float32, device="cpu")

    stride = tile_size - tile_overlap
    height_idx_list = list(range(0, input_img_height - tile_size, stride)) + [input_img_height - tile_size]
    width_idx_list = list(range(0, input_img_width - tile_size, stride)) + [input_img_width - tile_size]

    if input_img_height < tile_size:
        height_idx_list = height_idx_list[0]

    if input_img_width < tile_size:
        width_idx_list = width_idx_list[0]

    pbar = tqdm(total=len(height_idx_list) * len(width_idx_list), desc="Processing tiles", leave=False)

    for height_idx in height_idx_list:
        for width_idx in width_idx_list:
            height_end = min(height_idx + tile_size, input_img_height)
            width_end = min(width_idx + tile_size, input_img_width)
            height_start = max(0, height_end - tile_size)
            width_start = max(0, width_end - tile_size)

            in_patch = input_img_tensor[:, height_start:height_end, width_start:width_end].to(device)

            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
                out_patch = model(in_patch.unsqueeze(dim=0)).squeeze(dim=0).cpu()

            out_height_start = height_start * scaling_factor
            out_height_end = height_end * scaling_factor
            out_width_start = width_start * scaling_factor
            out_width_end = width_end * scaling_factor

            output_accumulated_values[:, out_height_start:out_height_end, out_width_start:out_width_end] += out_patch
            output_weight_map[:, out_height_start:out_height_end, out_width_start:out_width_end] += 1.0

            pbar.update(1)
    pbar.close()

    return output_accumulated_values.div_(output_weight_map)


@torch.inference_mode()
def inference(
    model: nn.Module,
    input_img_path: Path,
    output_img_path: Path,
    scaling_factor: int,
    device: config.DeviceType,
    dtype: torch.dtype = torch.bfloat16,
    tile_size: Optional[int] = None,
    tile_overlap: int = 32,
) -> None:
    model.eval()

    if not input_img_path.exists():
        raise FileNotFoundError(f"File '{input_img_path.name}' not found.")

    if not input_img_path.is_file():
        raise ValueError("Not a file")

    if input_img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        raise ValueError("Image must be in PNG or JPG format.")

    logger.info(f"Loading image from '{input_img_path}'...")

    input_img_tensor = decode_image(str(input_img_path), mode=ImageReadMode.RGB)
    input_img_tensor = transforms.ToDtype(dtype=dtype, scale=True)(input_img_tensor)

    logger.info(f"Image loaded ({input_img_tensor.shape[2]}x{input_img_tensor.shape[1]})")

    if tile_size is not None:
        logger.info(f"Mode: Tiling Inference | Tile size: {tile_size} | Overlap: {tile_overlap}.")

        output_img_tensor = tiled_inference(
            model=model,
            input_img_tensor=input_img_tensor,
            scaling_factor=scaling_factor,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            device=device,
            dtype=dtype,
        )
    else:
        logger.info("Mode: Full Image Inference (No Tiling).")

        input_img_tensor = input_img_tensor.to(device)
        input_img_height, input_img_width = input_img_tensor.shape[1], input_img_tensor.shape[2]

        padding_right = (config.WINDOW_SIZE - (input_img_width % config.WINDOW_SIZE)) % config.WINDOW_SIZE
        padding_bottom = (config.WINDOW_SIZE - (input_img_height % config.WINDOW_SIZE)) % config.WINDOW_SIZE

        input_img_tensor_padded = F.pad(input_img_tensor, pad=(0, padding_right, 0, padding_bottom), mode="reflect")

        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
            output_img_tensor_padded = model(input_img_tensor_padded.unsqueeze(dim=0)).squeeze(dim=0)

        output_img_tensor = output_img_tensor_padded[
            :, : input_img_height * scaling_factor, : input_img_width * scaling_factor
        ]

    output_img_tensor.clamp_(0, 1)

    logger.info(f"Saving result to '{output_img_path}'...")
    save_image(output_img_tensor, output_img_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="SwinIR Inference")
    parser.add_argument("-s", "--scale", type=int, default=config.SCALING_FACTOR, help="Upscaling factor")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to input image")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to save output image")
    parser.add_argument(
        "-ts", "--tile_size", type=int, default=None, help="Tile size (e.g., 512). None = processing without tiling"
    )
    parser.add_argument("-to", "--tile_overlap", type=int, default=32, help="Overlapping pixels between tiles")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Initializing SwinIR model (x{args.scale}) on {device}...")

    model = SwinIR(
        in_channels=3,
        hidden_channels=config.HIDDEN_CHANNELS,
        train_img_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        num_rstb_blocks=config.NUM_RSTB_BLOCKS,
        num_stl_blocks=config.NUM_STL_BLOCKS,
        num_heads=config.NUM_HEADS,
        window_size=config.WINDOW_SIZE,
        mlp_ratio=config.MLP_RATIO,
        upscale=args.scale,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
    ).to(device)

    if config.BEST_CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.BEST_CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    elif config.CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    else:
        logger.error("Could not locate model weights. Please check your checkpoint paths.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    inference(
        model=model,
        input_img_path=args.input,
        output_img_path=args.output,
        scaling_factor=args.scale,
        device=device,
        dtype=torch.bfloat16,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()

import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import SRDataset
from models import SwinIR
from utils import calculate_psnr, calculate_ssim, logger


@torch.inference_mode()
def test(
    data_loader: DataLoader,
    dataset_name: str,
    model: nn.Module,
    loss_fn: nn.Module,
    scaling_factor: config.ScalingFactor,
    device: config.DeviceType,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, float, float]:
    model.eval()

    avg_loss, avg_psnr, avg_ssim = 0.0, 0.0, 0.0

    for batch in tqdm(data_loader, desc=dataset_name, leave=False):
        lr_img_tensor = batch["lr"].to(device, non_blocking=True)
        hr_img_tensor = batch["hr"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
            sr_img_tensor = model(lr_img_tensor)
            loss = loss_fn(sr_img_tensor, hr_img_tensor)
            avg_loss += loss.item()

        sr_img_item = sr_img_tensor[0].float()
        hr_img_item = hr_img_tensor[0].float()

        avg_psnr += calculate_psnr(
            sr_img_tensor=sr_img_item,
            hr_img_tensor=hr_img_item,
            crop_border=scaling_factor,
        )

        avg_ssim += calculate_ssim(
            sr_img_tensor=sr_img_item,
            hr_img_tensor=hr_img_item,
            crop_border=scaling_factor,
        )

    avg_loss /= len(data_loader)
    avg_psnr /= len(data_loader)
    avg_ssim /= len(data_loader)

    return avg_loss, avg_psnr, avg_ssim


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    ).to(device)

    if config.BEST_CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.BEST_CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    elif config.CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    else:
        logger.error("Could not locate model weights. Please check your checkpoint paths.")

    loss_fn = nn.L1Loss()

    for test_dataset_path in config.TEST_DATASET_PATHS:
        dataset = SRDataset(
            data_path=test_dataset_path,
            scaling_factor=config.SCALING_FACTOR,
            patch_size=config.PATCH_SIZE,
            test_mode=True,
            dev_mode=False,
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.VAL_NUM_WORKERS,
            pin_memory=True if device == "cuda" else False,
            prefetch_factor=config.VAL_PREFETCH_FACTOR,
            persistent_workers=True if config.VAL_NUM_WORKERS > 0 else False,
        )

        test_loss, test_psnr, test_ssim = test(
            data_loader=data_loader,
            dataset_name=test_dataset_path.name,
            model=model,
            loss_fn=loss_fn,
            scaling_factor=config.SCALING_FACTOR,
            device=device,
            dtype=torch.bfloat16,
        )

        logger.info(
            f"Dataset: {test_dataset_path.name} | PSNR: {test_psnr:.2f} | SSIM: {test_ssim:.4f} | Loss: {test_loss:.4f}"
        )


if __name__ == "__main__":
    main()

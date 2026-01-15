import math

import torch

from utils import calculate_psnr, calculate_ssim, rgb2y


def test_rgb2y():
    input_img_tensor = torch.rand(1, 3, 64, 64)
    output_img_tensor = rgb2y(img_tensor=input_img_tensor)

    # Shape tests
    assert input_img_tensor.shape[0] == output_img_tensor.shape[0]
    assert output_img_tensor.shape[1] == 1
    assert input_img_tensor.shape[2] == output_img_tensor.shape[2]
    assert input_img_tensor.shape[3] == output_img_tensor.shape[3]

    # Values tests
    tolerance = 1e-5

    assert output_img_tensor.min().item() >= 0.0 - tolerance
    assert output_img_tensor.max().item() <= 1.0 + tolerance


def test_psnr_identical():
    img1 = torch.rand(1, 3, 64, 64)
    img2 = img1.clone()

    psnr = calculate_psnr(img1, img2, crop_border=4)
    assert psnr == float("inf")


def test_psnr_math_correctness():
    img_black = torch.zeros(1, 3, 64, 64)

    diff_rgb = 10.0
    img_gray = torch.ones(1, 3, 64, 64) * (diff_rgb / 255.0)

    psnr = calculate_psnr(img_black, img_gray, crop_border=0)
    expected = 20 * math.log10(255.0 / 9.0)

    assert math.isclose(psnr, expected, abs_tol=0.1)


def test_ssim_bounds():
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)

    ssim = calculate_ssim(img1, img2, crop_border=4)

    assert -1.0 <= ssim <= 1.0  # type: ignore

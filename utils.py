from einops import rearrange
from torch import Tensor


def split_img_into_windows(img_tensor: Tensor, window_size: int) -> Tensor:
    return rearrange(
        img_tensor,
        "b (h gh) (w gw) c -> (b h w) gh gw c",
        gh=window_size,
        gw=window_size,
    )


def combine_windows_into_img(windows_tensor: Tensor, img_height: int, img_width: int) -> Tensor:
    return rearrange(
        windows_tensor,
        "(b h w) gh gw c -> b (h gh) (w gw) c",
        h=img_height // windows_tensor.shape[1],
        w=img_width // windows_tensor.shape[2],
    )

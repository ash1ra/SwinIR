import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

import config
from utils import combine_windows_into_img, split_img_into_windows


class WMSA(nn.Module):
    def __init__(self, num_channels: int, window_size: int, num_heads: int) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.window_size = window_size
        self.num_heads = num_heads

        self.scale = (num_channels // num_heads) ** -0.5

        self.qkv_layer = nn.Linear(in_features=num_channels, out_features=num_channels * 3)
        self.projection = nn.Linear(in_features=num_channels, out_features=num_channels)

        self.register_buffer("relative_position_index", self._get_relative_position_index())

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _get_relative_position_index(self) -> Tensor:
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)

        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1

        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)

        return relative_position_index

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        num_windows, num_pixels_in_window, num_channels = x.shape

        qkv_tensor = (
            self.qkv_layer(x)
            .reshape(num_windows, num_pixels_in_window, 3, self.num_heads, num_channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = qkv_tensor[0], qkv_tensor[1], qkv_tensor[2]

        queries *= self.scale
        attention_scores = queries @ keys.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.flatten()]  # type: ignore
            .view(num_pixels_in_window, num_pixels_in_window, -1)
            .permute(2, 0, 1)
            .contiguous()
        )

        attention_scores += relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows_per_img = mask.shape[0]

            attention_scores = attention_scores.view(
                num_windows // num_windows_per_img,
                num_windows_per_img,
                self.num_heads,
                num_pixels_in_window,
                num_pixels_in_window,
            )
            attention_scores += mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, num_pixels_in_window, num_pixels_in_window)

        attention_probs = F.softmax(attention_scores, dim=-1)

        x = (attention_probs @ values).transpose(1, 2).reshape(num_windows, num_pixels_in_window, num_channels)
        x = self.projection(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()

        self.layers_sequence = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers_sequence(x)


class STL(nn.Module):
    def __init__(
        self,
        num_channels: int,
        train_img_size: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.train_img_size = train_img_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(train_img_size) <= window_size:
            self.shift_size = 0
            self.window_size = min(train_img_size)

        if not (0 <= shift_size < window_size):
            raise ValueError(f"Shift size ({shift_size}) must be >= 0 and less than wiindow_size ({window_size})")

        self.layer_norm_1 = nn.LayerNorm(num_channels)
        self.attention_block = WMSA(
            num_channels=num_channels,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.layer_norm_2 = nn.LayerNorm(num_channels)
        self.mlp = MLP(
            in_features=num_channels,
            hidden_features=num_channels * mlp_ratio,
            out_features=num_channels,
        )

        if shift_size > 0:
            attention_mask = self.calculate_mask(x_size=train_img_size)
        else:
            attention_mask = None

        self.register_buffer("attention_mask", attention_mask)

    def calculate_mask(self, x_size: tuple[int, int]) -> Tensor:
        img_height, img_width = x_size
        img_mask = torch.zeros((1, img_height, img_width, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = split_img_into_windows(img_tensor=img_mask, window_size=self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask.masked_fill_(attention_mask != 0, float(-100.0))
        attention_mask.masked_fill_(attention_mask == 0, float(0.0))

        return attention_mask

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        img_height, img_width = x_size
        batch_size, num_pixels_in_img, num_channels = x.shape

        residual = x

        x = self.layer_norm_1(x)
        x = x.view(batch_size, img_height, img_width, num_channels)

        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_shifted = x

        x_windows = split_img_into_windows(img_tensor=x_shifted, window_size=self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, num_channels)

        if self.train_img_size == x_size:
            attention_windows = self.attention_block(x_windows, mask=self.attention_mask)
        else:
            attention_windows = self.attention_block(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, num_channels)
        x_shifted = combine_windows_into_img(
            windows_tensor=attention_windows, img_height=img_height, img_width=img_width
        )

        if self.shift_size > 0:
            x = torch.roll(x_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = x_shifted

        x = x.view(batch_size, num_pixels_in_img, num_channels)

        x += residual
        x += self.mlp(self.layer_norm_2(x))

        return x


class RSTB(nn.Module):
    def __init__(
        self,
        num_channels: int,
        train_img_size: tuple[int, int],
        num_stl_blocks: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: int,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.train_img_size = train_img_size
        self.num_stl_blocks = num_stl_blocks
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.layers = nn.ModuleList()

        for i in range(num_stl_blocks):
            self.layers.append(
                STL(
                    num_channels=num_channels,
                    train_img_size=train_img_size,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    mlp_ratio=mlp_ratio,
                )
            )

        self.conv_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        img_height, img_width = x_size
        batch_size, num_pixels_in_img, num_channels = x.shape

        residual = x

        for layer in self.layers:
            if self.use_gradient_checkpointing and x.requires_grad:
                x = checkpoint(layer, x, x_size, use_reentrant=False)
            else:
                x = layer(x, x_size)

        x = x.view(batch_size, img_height, img_width, num_channels).permute(0, 3, 1, 2)
        x = self.conv_layer(x)
        x = x.flatten(2).transpose(1, 2)
        x += residual

        return x


class SwinIR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        train_img_size: tuple[int, int],
        num_rstb_blocks: int,
        num_stl_blocks: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: int,
        upscale: config.ScalingFactor,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.train_img_size = train_img_size
        self.num_stl_blocks = num_stl_blocks
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.upscale = upscale
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, in_channels, 1, 1) + 0.5

        self.register_buffer("imgs_mean", self.mean)

        self.shallow_feature_extraction = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deep_feature_extraction = nn.ModuleList()

        for _ in range(num_rstb_blocks):
            self.deep_feature_extraction.append(
                RSTB(
                    num_channels=hidden_channels,
                    train_img_size=train_img_size,
                    num_stl_blocks=num_stl_blocks,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )
            )

        self.layer_norm_after_dfe = nn.LayerNorm(hidden_channels)
        self.conv_after_dfe = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.img_reconstruction = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=64 * (upscale**2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(upscale),
            nn.Conv2d(
                in_channels=64,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def _add_padding(self, x: Tensor) -> Tensor:
        _, _, img_height, img_width = x.size()

        mod_pad_h = (self.window_size - img_height % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - img_width % self.window_size) % self.window_size

        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        return x

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_channels, img_height, img_width = x.shape

        x = self._add_padding(x)
        _, _, padded_img_height, padded_img_width = x.shape

        self.imgs_mean = self.imgs_mean.type_as(x)
        x -= self.imgs_mean

        x = self.shallow_feature_extraction(x)
        x_after_sfe = x

        x = x.flatten(2).transpose(1, 2)

        for layer in self.deep_feature_extraction:
            x = layer(x, (padded_img_height, padded_img_width))

        x = self.layer_norm_after_dfe(x)

        x = x.view(batch_size, padded_img_height, padded_img_width, self.hidden_channels).permute(0, 3, 1, 2)

        x = self.conv_after_dfe(x)
        x += x_after_sfe

        x = self.img_reconstruction(x)

        x = x[:, :, : img_height * self.upscale, : img_width * self.upscale]

        x += self.imgs_mean

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinIR(
        in_channels=3,
        hidden_channels=96,
        train_img_size=(64, 64),
        num_rstb_blocks=4,
        num_stl_blocks=6,
        num_heads=6,
        window_size=8,
        mlp_ratio=4,
        upscale=4,
        use_gradient_checkpointing=True,
    ).to(device)

    input_tensor = torch.randn(2, 3, 48, 48).to(device)

    print(f"Input shape: {input_tensor.shape}")

    with torch.no_grad():
        output_tensor = model(input_tensor)

    print(f"Output shape: {output_tensor.shape}")

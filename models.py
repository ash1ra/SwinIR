import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        num_windows, num_patches, num_channels = x.shape

        qkv_tensor = (
            self.qkv_layer(x)
            .reshape(num_windows, num_patches, 3, self.num_heads, num_channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = qkv_tensor[0], qkv_tensor[1], qkv_tensor[2]

        queries *= self.scale
        attention_scores = queries @ keys.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.flatten()]  # type: ignore
            .view(num_patches, num_patches, -1)
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
                num_patches,
                num_patches,
            )
            attention_scores += mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, num_patches, num_patches)

        attention_probs = F.softmax(attention_scores, dim=-1)

        x = (attention_probs @ values).transpose(1, 2).reshape(num_windows, num_patches, num_channels)
        x = self.projection(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int | None = None) -> None:
        super().__init__()

        out_features = out_features or in_features

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
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

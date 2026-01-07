import torch
from torch import Tensor, nn


class WMSA(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        self.register_buffer("relative_position_index", self._get_relative_position_index())

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

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
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(N, N, -1)
            .permute(2, 0, 1)
            .contiguous()
        )

        attn += relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int | None) -> None:
        super().__init__()

        out_features = out_features or in_features

        self.layers_sequence = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers_sequence(x)

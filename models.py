from torch import Tensor, nn


class WMSA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: ...


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

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn


ArchitectureName = Literal["small", "v1"]
ActivationName = Literal["crelu", "screlu"]


@dataclass(frozen=True)
class NnueArchitecture:
    name: ArchitectureName
    input_features: int
    hidden_size: int
    activation: ActivationName
    feature_bias: bool = True
    side_to_move_ordering: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


ARCHITECTURES: dict[str, NnueArchitecture] = {
    "small": NnueArchitecture(
        name="small",
        input_features=768,
        hidden_size=512,
        activation="crelu",
    ),
    "v1": NnueArchitecture(
        name="v1",
        input_features=768,
        hidden_size=1024,
        activation="screlu",
    ),
}


def resolve_architecture(name: str) -> NnueArchitecture:
    try:
        return ARCHITECTURES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unsupported architecture '{name}'; expected one of: {supported}") from exc


def crelu(values: torch.Tensor) -> torch.Tensor:
    return torch.clamp(values, min=0.0, max=1.0)


def screlu(values: torch.Tensor) -> torch.Tensor:
    clipped = crelu(values)
    return clipped * clipped


class PerspectiveNnueModel(nn.Module):
    def __init__(self, architecture: str | NnueArchitecture) -> None:
        super().__init__()
        resolved = (
            resolve_architecture(architecture) if isinstance(architecture, str) else architecture
        )
        self.architecture = resolved
        self.white = nn.EmbeddingBag(
            resolved.input_features,
            resolved.hidden_size,
            mode="sum",
        )
        self.black = nn.EmbeddingBag(
            resolved.input_features,
            resolved.hidden_size,
            mode="sum",
        )
        self.white_bias = nn.Parameter(torch.zeros(resolved.hidden_size))
        self.black_bias = nn.Parameter(torch.zeros(resolved.hidden_size))
        self.output = nn.Linear(resolved.hidden_size * 2, 1)

    def activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.architecture.activation == "crelu":
            return crelu(values)
        if self.architecture.activation == "screlu":
            return screlu(values)
        raise RuntimeError(f"unsupported activation: {self.architecture.activation}")

    def forward(
        self,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        offsets: torch.Tensor,
        side_to_move: torch.Tensor | None = None,
    ) -> torch.Tensor:
        white_accumulator = self.white(white_indices, offsets)
        black_accumulator = self.black(black_indices, offsets)
        if self.architecture.feature_bias:
            white_accumulator = white_accumulator + self.white_bias
            black_accumulator = black_accumulator + self.black_bias
        white = self.activate(white_accumulator)
        black = self.activate(black_accumulator)
        if self.architecture.side_to_move_ordering and side_to_move is not None:
            white_to_move = side_to_move.reshape(-1, 1) == 0
            first = torch.where(white_to_move, white, black)
            second = torch.where(white_to_move, black, white)
            return self.output(torch.cat((first, second), dim=1)).squeeze(1)
        return self.output(torch.cat((white, black), dim=1)).squeeze(1)


__all__ = [
    "ARCHITECTURES",
    "ActivationName",
    "ArchitectureName",
    "NnueArchitecture",
    "PerspectiveNnueModel",
    "crelu",
    "resolve_architecture",
    "screlu",
]

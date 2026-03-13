from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol

import torch


@dataclass
class JSCCMethodOutput:
    """Standardized method output for fair benchmarking."""

    reconstruction: torch.Tensor
    rate_proxy: torch.Tensor | None = None
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


class JSCCMethodProtocol(Protocol):
    """Minimal plug-in protocol for single-modal JSCC methods."""

    def encode(self, batch: Dict[str, torch.Tensor], condition: Dict[str, float]) -> torch.Tensor:
        ...

    def decode(
        self,
        latent: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        condition: Dict[str, float],
    ) -> JSCCMethodOutput:
        ...

    def compute_specialized_loss(
        self,
        output: JSCCMethodOutput,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ...

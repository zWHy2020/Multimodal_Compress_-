from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from benchmark.interfaces import JSCCMethodOutput


def build_core_loss(
    output: JSCCMethodOutput,
    target: torch.Tensor,
    lambda_rate: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """Core benchmark objective: distortion + rate proxy.

    L_core = MSE(x_hat, x) + lambda_rate * R_hat
    """

    distortion = F.mse_loss(output.reconstruction, target)
    rate = output.rate_proxy if output.rate_proxy is not None else distortion.new_zeros(())
    total = distortion + lambda_rate * rate
    return {
        "core_total": total,
        "core_distortion": distortion,
        "core_rate": rate,
    }

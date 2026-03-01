"""MINE-based information estimation utilities for OMIB beta upper-bound calculation."""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Dict

import torch
import torch.nn as nn


class MINE(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def mi_nat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_perm = y[torch.randperm(y.size(0), device=y.device)]
        joint = torch.cat([x, y], dim=-1)
        marg = torch.cat([x, y_perm], dim=-1)
        t_joint = self.net(joint)
        t_marg = self.net(marg)
        return t_joint.mean() - torch.log(torch.exp(t_marg).mean().clamp_min(1e-8))


def _flatten_depth_video(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    depth = batch['depth'].to(device)
    video = batch['video'].to(device)
    depth_vec = depth.flatten(start_dim=1)
    video_vec = video.flatten(start_dim=1)
    return depth_vec, video_vec


def estimate_beta_upper_bound_mine(
    data_iter: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    max_steps: int = 50,
    mine_steps: int = 100,
    hidden_dim: int = 128,
    lr: float = 1e-4,
) -> Dict[str, float]:
    """
    Estimate Mu = 1 / (3 * (H(v1)+H(v2)-I(v1;v2))) using MINE with H(X)=I(X;X).
    Returns a dict with H1, H2, I12 and Mu.
    """
    batches = []
    for i, b in enumerate(data_iter):
        batches.append(b)
        if i + 1 >= max_steps:
            break
    if not batches:
        raise RuntimeError('No batches available for MINE estimation.')

    d0, v0 = _flatten_depth_video(batches[0], device)
    mine_dv = MINE(d0.size(1), v0.size(1), hidden_dim=hidden_dim).to(device)
    mine_dd = MINE(d0.size(1), d0.size(1), hidden_dim=hidden_dim).to(device)
    mine_vv = MINE(v0.size(1), v0.size(1), hidden_dim=hidden_dim).to(device)

    opts = [
        torch.optim.Adam(mine_dv.parameters(), lr=lr),
        torch.optim.Adam(mine_dd.parameters(), lr=lr),
        torch.optim.Adam(mine_vv.parameters(), lr=lr),
    ]
    nets = [mine_dv, mine_dd, mine_vv]

    # train estimators to maximize MI lower bound
    for _ in range(mine_steps):
        for b in batches:
            d, v = _flatten_depth_video(b, device)
            for net, opt, pair in zip(nets, opts, ['dv', 'dd', 'vv']):
                opt.zero_grad(set_to_none=True)
                if pair == 'dv':
                    mi = net.mi_nat(d, v)
                elif pair == 'dd':
                    mi = net.mi_nat(d, d)
                else:
                    mi = net.mi_nat(v, v)
                loss = -mi
                loss.backward()
                opt.step()

    with torch.no_grad():
        mi_dv = []
        mi_dd = []
        mi_vv = []
        for b in batches:
            d, v = _flatten_depth_video(b, device)
            mi_dv.append(mine_dv.mi_nat(d, v))
            mi_dd.append(mine_dd.mi_nat(d, d))
            mi_vv.append(mine_vv.mi_nat(v, v))

        I12_nat = torch.stack(mi_dv).mean().clamp_min(0.0)
        H1_nat = torch.stack(mi_dd).mean().clamp_min(0.0)
        H2_nat = torch.stack(mi_vv).mean().clamp_min(0.0)

        denom_nat = (H1_nat + H2_nat - I12_nat).clamp_min(1e-8)
        Mu_nat = 1.0 / (3.0 * denom_nat)

    return {
        'H1_nat': float(H1_nat.item()),
        'H2_nat': float(H2_nat.item()),
        'I12_nat': float(I12_nat.item()),
        'Mu_nat': float(Mu_nat.item()),
        'Mu_bits': float((Mu_nat * math.log(2.0)).item()),
    }

"""Strict structural adaptation of Deep-JSCC-PyTorch for depth (1ch->1ch).

Source reference (external implementation):
https://github.com/chunbaobao/Deep-JSCC-PyTorch

Adaptation rule:
- keep architecture and normalization/channeling style
- only switch image channels from RGB(3) to depth(1) at input/output ends.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def ratio2filtersize_depth(x: torch.Tensor, ratio: float) -> int:
    """Depth-version helper following external repo's ratio2filtersize idea."""
    if x.dim() == 4:
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception("Unknown size of input")
    encoder_temp = _EncoderDepth(is_temp=True)
    z_temp = encoder_temp(x)
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activate: nn.Module = nn.PReLU(),
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        )
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode="fan_out", nonlinearity="leaky_relu")
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _EncoderDepth(nn.Module):
    """External _Encoder with in_channels changed 3->1."""

    def __init__(self, c: int = 64, is_temp: bool = False, p: float = 1.0):
        super().__init__()
        self.is_temp = is_temp
        self.conv1 = _ConvWithPReLU(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2 * c, kernel_size=5, padding=2)
        self.norm = self._normalization_layer(p=p)

    @staticmethod
    def _normalization_layer(p: float = 1.0):
        def _inner(z_hat: torch.Tensor) -> torch.Tensor:
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception("Unknown size of input")
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(torch.tensor(p) * k) * z_hat / torch.sqrt((z_temp @ z_trans).clamp_min(1e-8))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor

        return _inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
        return x


class _DecoderDepth(nn.Module):
    """External _Decoder with out_channels changed 3->1."""

    def __init__(self, c: int = 64):
        super().__init__()
        self.tconv1 = _TransConvWithPReLU(in_channels=2 * c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16,
            out_channels=1,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            activate=nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


class DeepJSCCDepthEncoder(nn.Module):
    """Wrapper that follows repo-friendly encoder API."""

    def __init__(self, c: int = 64, power_constraint: float = 1.0):
        super().__init__()
        self.impl = _EncoderDepth(c=c, p=power_constraint)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.impl(x), None


class DeepJSCCDepthDecoder(nn.Module):
    """Wrapper that follows repo-friendly decoder API."""

    def __init__(self, c: int = 64):
        super().__init__()
        self.impl = _DecoderDepth(c=c)

    def forward(self, z: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.impl(z)

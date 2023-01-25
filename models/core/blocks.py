from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import *


class Stage(nn.Sequential):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        super().__init__(*args)

    def append(self, m: Union[nn.Module, List[nn.Module]]):
        if isinstance(m, nn.Module):
            self.add_module(str(len(self)), m)
        elif isinstance(m, list):
            [self.append(i) for i in m]
        else:
            ValueError('')


class Conv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1
    ):
        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False,
    ):
        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        super().__init__(
            inp, oup, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=inp
        )


class PointwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bias: bool = False,
        groups: int = 1
    ):
        super().__init__(inp, oup, 1, stride=stride, padding=0, bias=bias, groups=groups)


class DepthwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1
    ):
        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size, stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )


class PointwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        groups: int = 1
    ):
        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride, groups=groups),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )


class SemanticallyDeterminedDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1,
        ratio: float = 3 / 4,
        learnable: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = (3, 3)
        self.padding = (dilation, dilation)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = in_channels
        self.padding_mode = 'zeros'

        self.i = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        self.mask_l4 = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]) / 4.0
        self.mask_l8 = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]) / 8.0

        # Sobel
        self.sx = torch.tensor([
            [0, 0, 0],
            [-1, 0, 1],
            [0, 0, 0]
        ])

        self.mask_sx = torch.tensor([
            [-1, 0, 1],
            [0, 0, 0],
            [-1, 0, 1]
        ]) / 4.0

        self.sy = torch.tensor([
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])

        self.mask_sy = torch.tensor([
            [-1, 0, -1],
            [0, 0, 0],
            [1, 0, 1]
        ]) / 4.0

        # Roberts
        self.rx = torch.tensor([
            [-1, 0, 0],
            [0,  0, 0],
            [0,  0, 1]
        ])
        self.ry = torch.tensor([
            [0,  0, 1],
            [0,  0, 0],
            [-1, 0, 0]
        ])

        self.mask_rx = torch.tensor([
            [0, -1, 0],
            [-1, 0, 1],
            [0,  1, 0]
        ]) / 2.0

        self.mask_ry = torch.tensor([
            [0,  1, 0],
            [-1, 0, 1],
            [0, -1, 0]
        ]) / 2.0

        self.ratio = ratio
        self.learnable = learnable

        self.edge_chs = make_divisible(self.in_channels * ratio, 6, 6)
        if self.edge_chs > self.in_channels:
            self.edge_chs -= 6
        self.gaussian_chs = self.in_channels - self.edge_chs

        if self.edge_chs != 0:
            self.l4 = nn.Parameter(torch.ones(self.edge_chs // 6, 1, 1), self.learnable)
            self.l8 = nn.Parameter(torch.ones(self.edge_chs // 6, 1, 1), self.learnable)

            self.sobel = nn.Parameter(torch.ones(self.edge_chs // 6, 1, 1) / 2.0, self.learnable)

            self.roberts = nn.Parameter(torch.ones(self.edge_chs // 6, 1, 1) / 4.0, self.learnable)

        if self.gaussian_chs != 0:
            self.sigma = nn.Parameter(torch.ones(self.gaussian_chs), self.learnable)

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.in_channels)

    @property
    def weight(self):
        kernels = []
        if self.edge_chs != 0:
            # self.l4.data = torch.max(self.l4.data, torch.tensor(0.0, device=self.l4.device))
            # self.l4.data = torch.min(self.l4.data, torch.tensor(2.0, device=self.l4.device))

            # self.l8.data = torch.max(self.l8.data, torch.tensor(0.0, device=self.l4.device))
            # self.l8.data = torch.min(self.l8.data, torch.tensor(2.0, device=self.l4.device))

            # self.sobel.data = torch.max(self.sobel.data, torch.tensor(0.0, device=self.sobel.device))
            # self.sobel.data = torch.min(self.sobel.data, torch.tensor(2.0, device=self.sobel.device))

            # self.roberts.data = torch.max(self.roberts.data, torch.tensor(0.0, device=self.l4.device))
            # self.roberts.data = torch.min(self.roberts.data, torch.tensor(1.0, device=self.l4.device))

            kernels.append((self.l4 * self.i.to(self.l4.device) - self.mask_l4.to(self.l4.device)).unsqueeze(1))
            kernels.append((self.l8 * self.i.to(self.l4.device) - self.mask_l8.to(self.l4.device)).unsqueeze(1))

            kernels.append((self.sobel * self.sx.to(self.sobel.device) +
                           self.mask_sx.to(self.sobel.device)).unsqueeze(1))
            kernels.append((self.sobel * self.sy.to(self.sobel.device) +
                           self.mask_sy.to(self.sobel.device)).unsqueeze(1))

            kernels.append((self.roberts * self.rx.to(self.l4.device) + self.mask_rx.to(self.l4.device)).unsqueeze(1))
            kernels.append((self.roberts * self.ry.to(self.l4.device) + self.mask_ry.to(self.l4.device)).unsqueeze(1))

        if self.gaussian_chs != 0:
            # self.sigma.data = torch.max(self.sigma.data, torch.tensor(0.25, device=self.sigma.device))
            kernels.append(get_3x3_gaussian_weight2d(self.sigma))

        return torch.cat(kernels)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, ratio={ratio}({edge_chs}:{gaussian_chs}), kernel_size={kernel_size}'
             ', learnable={learnable}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class GaussianBlur(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        learnable: bool = True
    ):
        super().__init__()

        padding = padding or ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.channels = channels
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.padding_mode = 'zeros'
        self.learnable = learnable

        self.sigma = nn.Parameter(torch.ones(channels), self.learnable)

        self.standard_w = None if self.learnable else nn.Parameter(
            get_3x3_gaussian_weight2d(torch.ones(channels)), False)

    def forward(self, x):
        return F.conv2d(x, self.weight if self.learnable else self.standard_w, None, self.stride, self.padding, self.dilation, self.channels)

    @property
    def weight(self):
        return get_3x3_gaussian_weight2d(self.sigma)

    @property
    def out_channels(self):
        return self.channels

    def extra_repr(self):
        s = ('{channels}, kernel_size={kernel_size}'
             ', learnable={learnable}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class SDDCBlock(nn.Sequential):
    def __init__(
        self,
        channels,
        stride: int = 1,
        dilation: int = 1,
        ratio: float = 0.75,
        learnable: bool = True
    ):
        super().__init__(
            SemanticallyDeterminedDepthwiseConv2d(
                channels, stride, dilation=dilation, ratio=ratio, learnable=learnable),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


class GaussianBlurBlock(nn.Sequential):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        learnable: bool = True
    ):
        super().__init__(
            GaussianBlur(channels, kernel_size, stride, padding, dilation, learnable=learnable),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

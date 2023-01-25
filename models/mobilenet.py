import torch
import torch.nn as nn
from .core import blocks
from typing import Any, OrderedDict, Type, Union, List


class MobileBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.75  # unused
    ):
        super().__init__(
            blocks.DepthwiseBlock(inp, inp, kernel_size, stride, padding, dilation=dilation),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class SDKBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.75
    ):
        if ratio is None:
            super().__init__(
                blocks.GaussianBlurBlock(inp, kernel_size, stride, padding=padding, dilation=dilation),
                blocks.PointwiseBlock(inp, oup, groups=groups)
            )
        else:
            super().__init__(
                blocks.SDDCBlock(inp, stride, dilation=dilation, ratio=ratio),
                blocks.PointwiseBlock(inp, oup, groups=groups)
            )


class MobileNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        base_width: int = 32,
        block: Type[Union[MobileBlock, SDKBlock]] = MobileBlock,
        depth_multiplier: float = 1.0,
        dropout_rate: float = 0.2,
        dilations: List[int] = None,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        def depth(d): return max(int(d * depth_multiplier), 8)

        dilations = dilations or [1, 1, 1, 1]
        assert len(dilations) == 4, ''

        FRONT_S = 1 if thumbnail else 2

        layers = [2, 2, 6, 2]
        strides = [FRONT_S, 2, 2, 2]
        # ratios = [8/8, 6/8, 4/8, 2/8, 1/16]
        ratios = [6/8, 6/8, 5/8, 5/8, 1/8]
        # ratios = [1/16, 4/8, 4/8, 6/8, 7/8]
        # ratios = [1, 1, 1, 1, 1]
        # ratios = [0, 0, 0, 0, 0]
        # ratios = [0.5, 0.5, 0.5, 0.5, 0.5]
        # ratios = [0.25, 0.25, 0.25, 0.25, 0.25]
        # ratios = [0.75, 0, 0, 0, 0]
        # ratios = [0, 0, 0, 0, 0.75]

        nn.LayerNorm

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                blocks.Conv2dBlock(in_channels, depth(base_width), stride=FRONT_S),
                block(depth(base_width), depth(base_width) * 2, ratio=ratios[0])
            ))
        ]))

        for stage, stride in enumerate(strides):
            inp = depth(base_width * 2 ** (stage + 1))
            oup = depth(base_width * 2 ** (stage + 2))

            self.features.add_module(f'stage{stage+1}', blocks.Stage(
                [block(
                    inp if i == 0 else oup,
                    oup,
                    stride=stride if (i == 0 and dilations[stage] == 1) else 1,
                    dilation=max(dilations[stage] // (stride if i == 0 else 1), 1),
                    ratio=None if i == 0 else ratios[stage+1]
                ) for i in range(layers[stage])]
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(oup, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v1_x1_0():
    return MobileNet(depth_multiplier=1.0, block=MobileBlock)


def sdk_mobilenet_v1_x1_0():
    return MobileNet(depth_multiplier=1.0, block=SDKBlock)

import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import math
import numpy as np
sys.path.append(r'E:\PythonPro\v5-medical')

from models.common import Conv  # noqa


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


# Burstormer: Burst Image Restoration and Enhancement Transformer | BFF Module light-detial.
class BFF(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.bffcv1 = Conv(c1, c2, k)
        self.bffcv2 = Conv(c1, c2, k)
        self.bffcv3 = Conv(c1, c2, k)

    def forward(self, x):
        x1 = self.bffcv1(x)
        x2 = self.bffcv2(x1)
        residual = x2 - x
        x3 = self.bffcv3(residual)
        return x2 + x3


# GiraffeDet: A Heavy-Neck Paradigm for Object Detection | Queen-fusion light-detial.
class QueenFusion(nn.Module):
    def __init__(self, c1, c2, r=0.625):
        super().__init__()
        mdc = make_divisible(int(c1 * r), 8)
        cin = [c1 // 2 ** x for x in range(3)]
        self.up = nn.Sequential(Conv(cin[0], mdc, 1), nn.Upsample(
            None, 2, 'bilinear', align_corners=False))
        self.align = Conv(cin[1], mdc, 1)
        self.down = nn.Sequential(Conv(cin[2], mdc, 1), nn.MaxPool2d(3, 2, 1))
        self.fusion = Conv(int(mdc * 3), c2, 1)

    def forward(self, x):
        x1, x2, x3 = x  # pixel low->high
        x1 = self.up(x1)
        x2 = self.align(x2)
        x3 = self.down(x3)
        x = self.fusion(torch.cat([x1, x2, x3], 1))
        return x


# Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism | Low-FAM light-detial.
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int = 192,
            oup: int = 192,
            global_inp=192,
    ) -> None:
        super().__init__()

        if not global_inp:
            global_inp = inp

        self.local_embedding = Conv(inp, oup, 1, act=None)
        self.global_embedding = Conv(global_inp, oup, 1, act=None)
        self.global_act = Conv(global_inp, oup, 1, act=None)
        self.act = h_sigmoid()

    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_l, x_g = x
        _, _, H, W = x_l.shape

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)

        sig_act = F.interpolate(self.act(global_act), size=(H, W),
                                mode='bilinear', align_corners=False)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out


class FAM_IFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fam = SimFusion_4in()
        self.ifm = nn.Sequential(
            Conv(960, 192, 1),
            *[Conv(192, 192, 3, g=192) for _ in range(2)],
        )

    def forward(self, x):
        x = self.fam(x)
        x = self.ifm(x)
        return x


if __name__ == '__main__':
    im0 = torch.randn(2, 512, 16, 16)
    im1 = torch.randn(2, 256, 32, 32)
    im2 = torch.randn(2, 128, 64, 64)
    im3 = torch.randn(2, 64, 128, 128)
    model = FAM_IFM()
    model2 = InjectionMultiSum_Auto_pool(128, 128)

    g = model([im0, im1, im2, im3][::-1])
    y = model2([im2, g])
    print(g.shape)
    print(y.shape)

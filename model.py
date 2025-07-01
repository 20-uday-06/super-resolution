#   VDSR, DMCN‑prelu, MRUNet  (resolution‑agnostic)
#   Now parameterised for ASTER super‑resolution
# -----------------------------------------------------------
import math
from math import sqrt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────── VDSR ───────────────────────────────────────────
class ConvReLU(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, n_channels: int = 1, depth: int = 20):
        super().__init__()
        self.in_conv  = nn.Conv2d(n_channels, 64, 3, 1, 1, bias=False)
        self.residual = nn.Sequential(*[ConvReLU() for _ in range(depth-2)])
        self.out_conv = nn.Conv2d(64, n_channels, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # He init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = 3 * 3 * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        res = self.relu(self.in_conv(x))
        res = self.residual(res)
        res = self.out_conv(res)
        return x + res           # global residual


# ─────────── DMCN‑prelu (unchanged except n_channels) ───────
class DwSample(nn.Module):
    def __init__(self, ch: int, groups=1, bn=False):
        super().__init__()
        layers = [nn.Conv2d(ch, ch, 3, 1, 1, groups=groups),
                  nn.PReLU()]
        if bn:
            layers.insert(1, nn.BatchNorm2d(ch))
        self.body = nn.Sequential(*layers)

    def forward(self, x): return x + self.body(x)


class BasicBlock(nn.Module):
    def __init__(self, ch: int, bn=False):
        super().__init__()
        seq = [nn.Conv2d(ch, ch, 3, 1, 1),
               nn.PReLU(),
               nn.Conv2d(ch, ch, 3, 1, 1)]
        if bn:
            seq.insert(1, nn.BatchNorm2d(ch))
        self.body = nn.Sequential(*seq)

    def forward(self, x): return x + self.body(x)


class PixelShuffleBlock(nn.Module):
    def __init__(self, ch, upscale):
        super().__init__()
        self.block = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch, ch * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x): return self.block(x)


class DMCN_prelu(nn.Module):
    def __init__(self, n_channels: int = 1, width=64, bn=True):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(n_channels, width, 3, 1, 1),
            nn.PReLU()
        )
        self.body   = nn.Sequential(
            *[DwSample(width, bn=bn) for _ in range(5)],
            nn.Conv2d(width, width, 3, 2, 1),          # ↓2
            *[DwSample(width, bn=bn) for _ in range(2)],
            nn.Conv2d(width, width, 3, 2, 1),          # ↓2
            *[DwSample(width, bn=bn) for _ in range(2)],
            PixelShuffleBlock(width, 2),               # ↑2
            *[BasicBlock(width, bn=bn) for _ in range(2)],
            PixelShuffleBlock(width, 2),               # ↑2
            *[BasicBlock(width, bn=bn) for _ in range(5)],
        )
        self.exit = nn.Conv2d(width, n_channels, 3, 1, 1)

    def forward(self, x):
        res = self.entry(x)
        res = self.body(res)
        res = self.exit(res)
        return x + res


# ─────────── MRUNet (minor fixes & flexibility) ─────────────
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), Mish(),
        nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), Mish()
    )


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, res=False):
        super().__init__()
        self.res = res
        if res:
            self.down = nn.Conv2d(in_ch, in_ch, 2, 2)
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.body = nn.Sequential(
                nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        if self.res:
            d = self.down(x)
            return self.conv(d) + d
        return self.body(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.up = (nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
                   if bilinear else
                   nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2))
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        return self.conv(torch.cat([x2, x1], 1))


class MRUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1,
                 res_down=True, n_resblocks=1, bilinear=False):
        super().__init__()
        self.inc   = double_conv(n_channels, 64)
        self.down1 = Down(64, 128, res_down)
        self.down2 = Down(128, 256, res_down)
        self.down3 = Down(256, 512, res_down)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor, res_down)

        self.resblocks = nn.Sequential(
            *[double_conv(1024//factor, 1024//factor)
              for _ in range(n_resblocks)])

        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x_in = x
        x1, x2 = self.inc(x), None
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.resblocks(x5)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return x_in + self.outc(x)    # residual add

# ─────────── convenience helpers ────────────────────────────
def make_mrunet(bands=1, sr_factor=4):
    """Factory for your training script."""
    return MRUNet(n_channels=bands, n_classes=bands,
                  res_down=True, n_resblocks=1,
                  bilinear=False)

__all__ = ["VDSR", "DMCN_prelu", "MRUNet", "make_mrunet"]



# HOW TO USE?

# from model import make_mrunet

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model  = make_mrunet(bands=1).to(device)   # 1 band NIR or TIR

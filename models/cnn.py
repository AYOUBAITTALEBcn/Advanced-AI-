"""EDSR-style residual CNN for demosaicing.

Head conv -> n_resblocks x (Conv-ReLU-Conv + skip, no BatchNorm, optional
residual scaling) -> tail conv -> residual added to the bilinear base.
"""
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class CNNDemosaic(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=16, res_scale=1.0):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x, base):
        feat = self.head(x)
        feat = feat + self.body(feat)
        res = self.tail(feat)
        return (base + res).clamp(0.0, 1.0)

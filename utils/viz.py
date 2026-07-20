"""Image/figure helpers: crop saving, error maps, training curves."""
import os

import numpy as np
import torch
from PIL import Image

DPI = 300


def tensor_to_uint8(t):
    """[3,H,W] float [0,1] -> HWC uint8."""
    arr = (t.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return arr.transpose(1, 2, 0)


def save_image(t, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(tensor_to_uint8(t)).save(path)


def error_map_image(pred, gt, vmax=0.15):
    """Mean-abs-error heatmap (inferno), shared scale -> HWC uint8."""
    import matplotlib
    err = (pred - gt).abs().mean(dim=0).detach().cpu().numpy()
    err = np.clip(err / vmax, 0.0, 1.0)
    rgba = matplotlib.colormaps['inferno'](err)
    return (rgba[..., :3] * 255).astype(np.uint8)


def save_error_map(pred, gt, path, vmax=0.15):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(error_map_image(pred, gt, vmax)).save(path)


def hardest_crops(err_full, k=4, crop=96, min_dist=64):
    """Coordinates of the k highest-error crop windows in one error map.

    err_full: [H,W] tensor of per-pixel error; greedy non-overlapping picks.
    """
    import torch.nn.functional as F
    e = err_full[None, None]
    score = F.avg_pool2d(e, crop, stride=8).squeeze()
    coords = []
    s = score.clone()
    for _ in range(k):
        idx = int(torch.argmax(s))
        r, c = divmod(idx, s.shape[1])
        y, x = r * 8, c * 8
        coords.append((y, x))
        r0 = max(0, r - min_dist // 8)
        c0 = max(0, c - min_dist // 8)
        s[r0:r + min_dist // 8, c0:c + min_dist // 8] = -1
    return coords

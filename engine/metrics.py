"""PSNR / CPSNR / per-channel PSNR / SSIM.

All metrics are computed in the 8-bit domain (x255, no rounding) after
shaving a `shave`-px boundary. CPSNR = mean of the three per-channel PSNRs;
PSNR = joint-MSE PSNR over all channels.
"""
import numpy as np
from skimage.metrics import structural_similarity

METRIC_KEYS = ['psnr', 'cpsnr', 'psnr_r', 'psnr_g', 'psnr_b', 'ssim']


def _prep(t, shave):
    """[3,H,W] float tensor in [0,1] -> HWC float32 in [0,255], shaved.

    float32 (not float64): SSIM allocates several full-size intermediate
    arrays per call, and on a RAM-constrained host that's doubled needlessly
    -- PSNR/SSIM values only need ~4-6 significant digits, well within
    float32's ~7.
    """
    a = t.detach().cpu().numpy().astype(np.float32) * 255.0
    if shave > 0:
        a = a[:, shave:-shave, shave:-shave]
    return a.transpose(1, 2, 0)


def _psnr_from_mse(mse):
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def img_metrics(pred, gt, shave=8):
    """pred, gt: [3,H,W] float tensors in [0,1] -> dict of METRIC_KEYS."""
    p, g = _prep(pred, shave), _prep(gt, shave)
    diff2 = (p - g) ** 2
    per_ch = [_psnr_from_mse(diff2[..., c].mean()) for c in range(3)]
    return {
        'psnr': _psnr_from_mse(diff2.mean()),
        'cpsnr': float(np.mean(per_ch)),
        'psnr_r': per_ch[0], 'psnr_g': per_ch[1], 'psnr_b': per_ch[2],
        'ssim': structural_similarity(g, p, channel_axis=2, data_range=255.0),
    }


def mean_metrics(rows):
    """Average a list of img_metrics dicts."""
    return {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}

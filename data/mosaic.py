"""Bayer RGGB mosaicing and the bilinear demosaic used as the residual base.

Conventions (strict):
  - float32, range [0,1]
  - Bayer pattern RGGB on a 2x2 tile: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B
"""
import torch
import torch.nn.functional as F

_MASK_CACHE = {}


def bayer_mask(h, w, device=None, dtype=torch.float32):
    """[3,h,w] binary mask; exactly one channel is 1 per pixel (RGGB)."""
    key = (h, w, str(device), dtype)
    if key not in _MASK_CACHE:
        m = torch.zeros(3, h, w, device=device, dtype=dtype)
        m[0, 0::2, 0::2] = 1.0  # R
        m[1, 0::2, 1::2] = 1.0  # G at R-rows
        m[1, 1::2, 0::2] = 1.0  # G at B-rows
        m[2, 1::2, 1::2] = 1.0  # B
        _MASK_CACHE[key] = m
    return _MASK_CACHE[key]


def mosaic(rgb):
    """rgb [3,H,W] or [B,3,H,W] -> masked3 of the same shape.

    R values at R positions, G at both G positions, B at B positions;
    all other entries zero. This masked-3ch tensor is the model input.
    """
    h, w = rgb.shape[-2:]
    return rgb * bayer_mask(h, w, rgb.device, rgb.dtype)


# Classic bilinear demosaic expressed as per-channel convolutions over the
# masked mosaic. Interior pixels are mathematically identical to
# colour_demosaicing.demosaicing_CFA_Bayer_bilinear; borders use reflect101
# padding, which preserves RGGB phase for any pad amount.
_K_G = torch.tensor([[0., 1., 0.], [1., 4., 1.], [0., 1., 0.]]) / 4.0
_K_RB = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 4.0


def bilinear_base(masked3):
    """Bilinear-interpolated demosaic from masked3. [_,3,H,W] -> same shape.

    Every model predicts a residual added to this base.
    """
    single = masked3.dim() == 3
    x = masked3.unsqueeze(0) if single else masked3
    k = torch.stack([_K_RB, _K_G, _K_RB]).unsqueeze(1).to(x.device, x.dtype)
    out = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), k, groups=3)
    out = out.clamp(0.0, 1.0)
    return out.squeeze(0) if single else out


def bilinear_base_colour(masked3):
    """Reference implementation via colour-demosaicing (CPU, per image).

    Kept for cross-checking `bilinear_base`; not used in the training path.
    """
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
    single = masked3.dim() == 3
    x = masked3.unsqueeze(0) if single else masked3
    outs = []
    for img in x:
        cfa = img.sum(dim=0).cpu().numpy()  # one non-zero channel per pixel
        rgb = demosaicing_CFA_Bayer_bilinear(cfa, pattern='RGGB')
        outs.append(torch.from_numpy(rgb).permute(2, 0, 1).to(x.dtype).clamp(0, 1))
    out = torch.stack(outs).to(masked3.device)
    return out.squeeze(0) if single else out

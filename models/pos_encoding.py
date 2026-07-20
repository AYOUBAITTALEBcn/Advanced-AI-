"""Swappable positional encodings for the plain windowed ViT.

All four share one interface:
  pe(tokens, hw)     -> tokens (+PE)   additive path, applied once after embed
  pe.rope(q, k, hw)  -> q, k           rotary path, applied inside attention
  pe.is_rope                            True only for rope2d

Token layout everywhere: [B, N, C] with N = H*W row-major over the token grid.
Tiled inference keeps the token grid fixed, so grid-indexed tables never need
interpolation.
"""
import math

import torch
import torch.nn as nn


def build_pe(pe_type, dim, num_heads, window):
    if pe_type == 'none':
        return NoPE()
    if pe_type == 'learnable':
        return LearnablePE(dim, window)
    if pe_type == 'sinusoidal2d':
        return Sinusoidal2D(dim)
    if pe_type == 'rope2d':
        return Rope2D(dim, num_heads)
    raise ValueError(f'unknown pe_type: {pe_type}')


class BasePE(nn.Module):
    is_rope = False

    def forward(self, x, hw):
        return x

    def rope(self, q, k, hw):
        return q, k


class NoPE(BasePE):
    """Identity (control)."""


class LearnablePE(BasePE):
    """Learned table [window*window, dim], tiled across the token grid.

    Per-window indexing matches the windowed attention partition, so the same
    table is valid at any image size under tiled inference.
    """

    def __init__(self, dim, window):
        super().__init__()
        self.window = window
        self.table = nn.Parameter(torch.empty(window * window, dim))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, x, hw):
        H, W = hw
        w = self.window
        assert H % w == 0 and W % w == 0, f'grid {hw} not divisible by window {w}'
        t = self.table.view(w, w, -1).repeat(H // w, W // w, 1)  # [H,W,C]
        return x + t.reshape(1, H * W, -1)


def _sincos_1d(pos, dim, device, dtype):
    """[N] positions -> [N, dim] standard sin/cos embedding."""
    omega = torch.arange(dim // 2, device=device, dtype=torch.float32) / (dim // 2)
    omega = 1.0 / (10000.0 ** omega)                     # [dim/2]
    out = pos.to(torch.float32)[:, None] * omega[None]   # [N, dim/2]
    return torch.cat([out.sin(), out.cos()], dim=1).to(dtype)


class Sinusoidal2D(BasePE):
    """First half of dim encodes the row index, second half the column."""

    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, 'sinusoidal2d needs dim divisible by 4'
        self.dim = dim
        self._cache = {}

    def _table(self, H, W, device, dtype):
        key = (H, W, str(device), dtype)
        if key not in self._cache:
            rows = torch.arange(H, device=device)
            cols = torch.arange(W, device=device)
            er = _sincos_1d(rows, self.dim // 2, device, dtype)  # [H, C/2]
            ec = _sincos_1d(cols, self.dim // 2, device, dtype)  # [W, C/2]
            pe = torch.cat([
                er[:, None, :].expand(H, W, -1),
                ec[None, :, :].expand(H, W, -1)], dim=2)          # [H,W,C]
            self._cache[key] = pe.reshape(1, H * W, self.dim)
        return self._cache[key]

    def forward(self, x, hw):
        H, W = hw
        return x + self._table(H, W, x.device, x.dtype)


class Rope2D(BasePE):
    """2D axial rotary embedding on Q/K.

    Per head: the first half of head_dim is rotated by row-dependent angles,
    the second half by column-dependent angles.
    """
    is_rope = True

    def __init__(self, dim, num_heads):
        super().__init__()
        head_dim = dim // num_heads
        assert head_dim % 4 == 0, 'rope2d needs head_dim divisible by 4'
        self.half = head_dim // 2       # dims per axis
        npairs = self.half // 2         # rotation pairs per axis
        inv_freq = 1.0 / (10000.0 ** (torch.arange(npairs, dtype=torch.float32) / npairs))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cache = {}

    def _angles(self, H, W, device, dtype):
        key = (H, W, str(device), dtype)
        if key not in self._cache:
            inv = self.inv_freq.to(device=device, dtype=torch.float32)
            rows = torch.arange(H, device=device, dtype=torch.float32)
            cols = torch.arange(W, device=device, dtype=torch.float32)
            ar = torch.outer(rows, inv)  # [H, npairs]
            ac = torch.outer(cols, inv)  # [W, npairs]
            ar = ar[:, None, :].expand(H, W, -1).reshape(H * W, -1)
            ac = ac[None, :, :].expand(H, W, -1).reshape(H * W, -1)
            self._cache[key] = tuple(
                t.to(dtype) for t in (ar.cos(), ar.sin(), ac.cos(), ac.sin()))
        return self._cache[key]

    @staticmethod
    def _rotate(x, cos, sin):
        # x: [..., N, half]; cos/sin: [N, half/2] broadcast over leading dims.
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def rope(self, q, k, hw):
        """q, k: [B, heads, N, head_dim] with N = H*W row-major."""
        H, W = hw
        cr, sr, cc, sc = self._angles(H, W, q.device, q.dtype)
        h = self.half

        def rot(t):
            return torch.cat([
                self._rotate(t[..., :h], cr, sr),
                self._rotate(t[..., h:], cc, sc)], dim=-1)

        return rot(q), rot(k)

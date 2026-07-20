"""Plain windowed ViT for demosaicing - the PE-ablation model.

Stem: conv patch-embed (patch=4), so a 64x64 input -> 16x16 token grid.
Body: pre-norm Transformer blocks; attention partitions the token grid into
non-overlapping window x window windows with full attention inside each
(window == grid_size -> global attention).
PE:   swappable module from pos_encoding.py (none/learnable/sinusoidal2d
      applied additively after the stem; rope2d inside attention).
Head: conv -> PixelShuffle(patch) back to full res -> 3ch residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .pos_encoding import build_pe


class WindowedAttention(nn.Module):
    def __init__(self, dim, heads, window, pe):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.pe = pe  # shared PE module (used here only if rotary)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, hw):
        B, N, C = x.shape
        H, W = hw
        w = self.window
        nh, nw = H // w, W // w
        q, k, v = self.qkv(x).reshape(
            B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)

        if self.pe.is_rope:
            q, k = self.pe.rope(q, k, hw)

        def part(t):  # [B,h,N,d] -> [B*nWin, h, w*w, d]
            t = t.reshape(B, self.heads, nh, w, nw, w, self.head_dim)
            t = t.permute(0, 2, 4, 1, 3, 5, 6)
            return t.reshape(B * nh * nw, self.heads, w * w, self.head_dim)

        out = F.scaled_dot_product_attention(part(q), part(k), part(v))
        out = out.reshape(B, nh, nw, self.heads, w, w, self.head_dim)
        out = out.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, heads, window, pe, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim, heads, window, pe)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, hw):
        x = x + self.attn(self.norm1(x), hw)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTDemosaic(nn.Module):
    def __init__(self, dim=120, depth=6, heads=6, patch_embed=4, window=8,
                 pe_type='rope2d', mlp_ratio=4.0, grad_checkpoint=False):
        super().__init__()
        self.patch = patch_embed
        self.window = window
        self.grad_checkpoint = grad_checkpoint
        self.stem = nn.Conv2d(3, dim, patch_embed, stride=patch_embed)
        self.pe = build_pe(pe_type, dim, heads, window)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, window, self.pe, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 3 * patch_embed ** 2, 3, padding=1),
            nn.PixelShuffle(patch_embed))

    def forward(self, x, base):
        B, C, H, W = x.shape
        mult = self.patch * self.window
        ph = (mult - H % mult) % mult
        pw = (mult - W % mult) % mult
        if ph or pw:  # reflect101 keeps Bayer phase for any pad amount
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        Hp, Wp = x.shape[-2:]
        gh, gw = Hp // self.patch, Wp // self.patch

        feat = self.stem(x)                                   # [B,dim,gh,gw]
        tok = feat.flatten(2).transpose(1, 2)                 # [B,N,dim]
        tok = self.pe(tok, (gh, gw))
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                tok = checkpoint(blk, tok, (gh, gw), use_reentrant=False)
            else:
                tok = blk(tok, (gh, gw))
        tok = self.norm(tok)
        feat = tok.transpose(1, 2).reshape(B, -1, gh, gw)
        res = self.head(feat)[:, :, :H, :W]
        return (base + res).clamp(0.0, 1.0)

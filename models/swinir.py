"""SwinIR-style restorer: shallow conv -> RSTBs (windowed + shifted-window
attention with relative-position bias) -> conv -> residual to the base.

Position info is baked in via the relative-position bias; this model is NOT
part of the PE ablation.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def window_partition(x, w):
    """[B,H,W,C] -> [B*nWin, w*w, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, w * w, C)


def window_reverse(win, w, H, W):
    """[B*nWin, w*w, C] -> [B,H,W,C]"""
    B = win.shape[0] // (H * W // w // w)
    x = win.view(B, H // w, W // w, w, w, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads = heads
        self.window = window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.rpb_table = nn.Parameter(
            torch.empty((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(window), torch.arange(window), indexing='ij'))
        flat = coords.flatten(1)                                # [2, w*w]
        rel = flat[:, :, None] - flat[:, None, :]               # [2, w*w, w*w]
        rel = rel.permute(1, 2, 0) + window - 1
        idx = rel[..., 0] * (2 * window - 1) + rel[..., 1]
        self.register_buffer('rpb_index', idx, persistent=False)

    def forward(self, x, mask=None):
        """x: [B_, N, C] windows; mask: [nWin, N, N] or None."""
        B_, N, C = x.shape
        q, k, v = self.qkv(x).reshape(
            B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)           # [B_,h,N,N]
        bias = self.rpb_table[self.rpb_index.view(-1)].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_ // nw, nw, self.heads, N, N) + \
                mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_, self.heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, mlp_ratio=4.0):
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self._mask_cache = {}

    def _mask(self, H, W, device):
        if self.shift == 0:
            return None
        key = (H, W, str(device))
        if key not in self._mask_cache:
            w, s = self.window, self.shift
            img = torch.zeros(1, H, W, 1, device=device)
            cnt = 0
            for hs in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                for ws in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                    img[:, hs, ws, :] = cnt
                    cnt += 1
            win = window_partition(img, w).squeeze(-1)          # [nWin, w*w]
            diff = win.unsqueeze(1) - win.unsqueeze(2)
            mask = torch.zeros_like(diff).masked_fill(diff != 0, -100.0)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x, hw):
        H, W = hw
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.window)
        win = self.attn(win, self._mask(H, W, x.device))
        x = window_reverse(win, self.window, H, W)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(B, N, C)
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: swin blocks + conv + skip."""

    def __init__(self, dim, depth, heads, window, mlp_ratio, grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, window,
                      shift=0 if i % 2 == 0 else window // 2,
                      mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, feat):
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(blk, x, (H, W), use_reentrant=False)
            else:
                x = blk(x, (H, W))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return feat + self.conv(x)


class SwinIRDemosaic(nn.Module):
    def __init__(self, embed_dim=60, depths=(6, 6, 6, 6), heads=(6, 6, 6, 6),
                 window=8, mlp_ratio=4.0, grad_checkpoint=False):
        super().__init__()
        self.window = window
        self.shallow = nn.Conv2d(3, embed_dim, 3, padding=1)
        self.body = nn.ModuleList([
            RSTB(embed_dim, d, h, window, mlp_ratio, grad_checkpoint)
            for d, h in zip(depths, heads)])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.recon = nn.Conv2d(embed_dim, 3, 3, padding=1)

    def forward(self, x, base):
        H, W = x.shape[-2:]
        w = self.window
        ph, pw = (w - H % w) % w, (w - W % w) % w
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')

        shallow = self.shallow(x)
        feat = shallow
        for rstb in self.body:
            feat = rstb(feat)
        feat = self.conv_after_body(feat) + shallow
        res = self.recon(feat)[:, :, :H, :W]
        return (base + res).clamp(0.0, 1.0)

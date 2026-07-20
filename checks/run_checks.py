"""Static acceptance checks (spec section 12, checks 1 and 4 + sanity).

  python -m checks.run_checks            # fast: shapes, mosaic, PE, params
  python -m checks.run_checks --budgets  # also count lite/base param match

Data-dependent checks live elsewhere:
  check 2 (bilinear floor):  python -m engine.eval --config configs/cnn.yaml --bilinear
  check 3 (overfit):         python -m engine.train --config configs/cnn.yaml --smoke --overfit
  check 5 (determinism):     two `--smoke --iters 100 --deterministic` runs, diff
                              the curves CSVs (loss/metric columns; the elapsed-time
                              column always differs). Without --deterministic,
                              cudnn.benchmark=True is free to pick nondeterministic
                              algorithms and the traces will drift slightly.
  check 6 (smoke e2e):       bash scripts/run_all.sh --smoke
"""
import argparse

import torch

from data.mosaic import bayer_mask, bilinear_base, mosaic
from models.build import build_model, count_params

OK, BAD = '[ok]', '[FAIL]'
failures = []


def report(cond, msg):
    print(f'{OK if cond else BAD} {msg}')
    if not cond:
        failures.append(msg)


def check_mosaic():
    gt = torch.rand(2, 3, 64, 64)
    m = mosaic(gt)
    nz = (m > 0).sum(dim=1)
    report(m.shape == gt.shape, 'mosaic keeps shape')
    report(int(nz.max()) <= 1, 'mosaic: at most one non-zero channel per pixel')
    mask = bayer_mask(4, 4)
    expect = torch.tensor([[0, 1], [1, 2]])  # channel index on the 2x2 tile
    got = mask.argmax(dim=0)[:2, :2]
    report(bool((got == expect).all()), 'RGGB layout: (0,0)=R (0,1)=G (1,0)=G (1,1)=B')
    base = bilinear_base(m)
    report(base.shape == gt.shape, 'bilinear_base keeps shape')
    # measured values must pass through exactly (kernel center weight = 1)
    sel = bayer_mask(64, 64).bool()
    diff = (base[0] - gt[0])[sel].abs().max()
    report(float(diff) < 1e-6, f'bilinear_base preserves sensor samples (max diff {float(diff):.2e})')
    try:
        from data.mosaic import bilinear_base_colour
        ref = bilinear_base_colour(m[0])
        inner = (slice(None), slice(8, -8), slice(8, -8))
        d = (base[0][inner] - ref[inner]).abs().max()
        report(float(d) < 1e-5,
               f'bilinear_base matches colour-demosaicing in interior (max diff {float(d):.2e})')
    except ImportError:
        print('[skip] colour-demosaicing not installed; conv fallback in use')


def check_shapes():
    for name in ('cnn', 'swinir', 'vit'):
        model = build_model({'name': name, 'budget': 'tiny'}, verbose=False).eval()
        for h, w in ((64, 64), (100, 76)):  # odd-ish size exercises padding
            gt = torch.rand(2, 3, h, w)
            m = mosaic(gt)
            out = model(m, bilinear_base(m))
            report(out.shape == gt.shape,
                   f'{name}: [2,3,{h},{w}] -> {list(out.shape)}')
            report(bool((out >= 0).all() and (out <= 1).all()),
                   f'{name}: output clamped to [0,1]')


def check_pe():
    gt = torch.rand(1, 3, 64, 64)
    m = mosaic(gt)
    b = bilinear_base(m)
    for pe in ('none', 'learnable', 'sinusoidal2d', 'rope2d'):
        for w in (4, 8, 16):
            model = build_model({'name': 'vit', 'budget': 'tiny',
                                 'pe_type': pe, 'window': w},
                                verbose=False).eval()
            out = model(m, b)
            report(out.shape == gt.shape and torch.isfinite(out).all(),
                   f'vit pe={pe} window={w}: forward ok')


def check_param_match(budgets=('lite',)):
    for budget in budgets:
        counts = {}
        for name in ('cnn', 'swinir', 'vit'):
            model = build_model({'name': name, 'budget': budget})
            counts[name] = count_params(model)
        lo, hi = min(counts.values()), max(counts.values())
        spread = hi / lo - 1
        report(spread <= 0.10,
               f'{budget}: params within 10% (spread {spread * 100:.1f}%: ' +
               ', '.join(f'{k}={v / 1e6:.3f}M' for k, v in counts.items()) + ')')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--budgets', action='store_true',
                   help='check lite AND base budgets (slower)')
    args = p.parse_args()
    print('--- mosaic / bilinear base ---')
    check_mosaic()
    print('--- model shapes (check 1) ---')
    check_shapes()
    print('--- PE x window forward sanity ---')
    check_pe()
    print('--- param match (check 4) ---')
    check_param_match(('lite', 'base') if args.budgets else ('lite',))
    print()
    if failures:
        raise SystemExit(f'{len(failures)} check(s) FAILED')
    print('all checks passed')


if __name__ == '__main__':
    main()

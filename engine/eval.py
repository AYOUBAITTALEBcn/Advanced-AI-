"""Evaluation: tiled inference + metrics + efficiency profiling + CSV rows.

Tiled inference splits each image into overlapping tiles matching the
training patch size, runs the model per tile, and blends overlaps with a
feathered weight window. Tile origins are kept even so every tile has RGGB
phase, and the ViT token grid stays fixed (learnable PE valid at any size).

Usage:
  python -m engine.eval --run_name cnn_lite                 # main-results row
  python -m engine.eval --config configs/cnn.yaml --bilinear # bilinear floor
  python -m engine.eval --run_name vit_lite_pe-rope2d_w8 --ablation
"""
import argparse
import json
import os

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data.datasets import EvalImages, resolve_test_dir
from data.mosaic import bilinear_base, mosaic
from engine.metrics import img_metrics, mean_metrics
from models.build import build_model
from utils.profile import profile_model
from utils.viz import save_error_map, save_image

MAIN_COLS = ['model', 'params_M', 'flops_G', 'latency_ms',
             'kodak_PSNR', 'kodak_CPSNR', 'kodak_SSIM',
             'mcm_PSNR', 'mcm_CPSNR', 'mcm_SSIM']
ABL_COLS = ['pe_type', 'window', 'params_M',
            'kodak_CPSNR', 'kodak_SSIM', 'mcm_CPSNR', 'mcm_SSIM']


def _feather(tile, overlap):
    """[1,tile,tile] blending weights: 1 inside, linear ramp over the margin."""
    r = torch.arange(tile, dtype=torch.float32)
    ramp = torch.minimum(r + 1, tile - r).clamp(max=overlap + 1) / (overlap + 1)
    return (ramp[:, None] * ramp[None, :]).unsqueeze(0)


def _positions(full, tile, stride):
    xs = list(range(0, max(full - tile, 0) + 1, stride))
    if xs[-1] != full - tile and full > tile:
        xs.append(full - tile)
    return [x - x % 2 for x in xs]  # even origins keep RGGB phase


@torch.no_grad()
def tiled_inference(model, gt, tile=64, overlap=16, device='cpu',
                    batch_tiles=32):
    """gt [3,H,W] float in [0,1] -> (pred, base) at full size on CPU."""
    masked = mosaic(gt)
    base = bilinear_base(masked)
    if model is None:  # bilinear floor
        return base, base

    H, W = gt.shape[-2:]
    ph, pw = max(tile - H, 0), max(tile - W, 0)
    if ph or pw:  # reflect101 keeps Bayer phase for any pad amount
        masked = torch.nn.functional.pad(masked[None], (0, pw, 0, ph),
                                         mode='reflect')[0]
        base = torch.nn.functional.pad(base[None], (0, pw, 0, ph),
                                       mode='reflect')[0]
    Hp, Wp = masked.shape[-2:]
    stride = tile - overlap
    coords = [(y, x) for y in _positions(Hp, tile, stride)
              for x in _positions(Wp, tile, stride)]
    weight = _feather(tile, overlap)

    out = torch.zeros(3, Hp, Wp)
    acc = torch.zeros(1, Hp, Wp)
    for i in range(0, len(coords), batch_tiles):
        chunk = coords[i:i + batch_tiles]
        mt = torch.stack([masked[:, y:y + tile, x:x + tile] for y, x in chunk])
        bt = torch.stack([base[:, y:y + tile, x:x + tile] for y, x in chunk])
        pred = model(mt.to(device), bt.to(device)).float().cpu()
        for j, (y, x) in enumerate(chunk):
            out[:, y:y + tile, x:x + tile] += pred[j] * weight
            acc[:, y:y + tile, x:x + tile] += weight
    pred_full = (out / acc)[:, :H, :W].clamp(0.0, 1.0)
    return pred_full, base[:, :H, :W]


def eval_dataset(model, img_dir, tile, overlap, shave, device,
                 max_images=None, desc=''):
    ds = EvalImages(img_dir, max_images=max_images)
    rows, per_image = [], {}
    for name, gt in tqdm(ds, desc=desc, leave=False):
        pred, _ = tiled_inference(model, gt, tile, overlap, device)
        m = img_metrics(pred, gt, shave)
        rows.append(m)
        per_image[name] = m
    return mean_metrics(rows), per_image


# ---------------------------------------------------------------- qualitative

def select_hard_crops(img_dir, qual_dir, k=4, crop=96):
    """Pick the k hardest (highest bilinear-error) crops in a test set.

    Deterministic, cached to crops.json; also writes GT/bilinear crops and
    the bilinear error maps so every model is compared at the same places.
    """
    from utils.viz import hardest_crops
    os.makedirs(qual_dir, exist_ok=True)
    cache = os.path.join(qual_dir, 'crops.json')
    ds = EvalImages(img_dir)
    if not os.path.exists(cache):
        best = []  # (score, name, y, x)
        for name, gt in ds:
            base = bilinear_base(mosaic(gt))
            err = (base - gt).abs().mean(dim=0)
            y, x = hardest_crops(err, k=1, crop=crop)[0]
            y, x = y - y % 2, x - x % 2
            score = float(err[y:y + crop, x:x + crop].mean())
            best.append((score, name, y, x))
        best.sort(reverse=True)
        chosen = [{'name': n, 'y': y, 'x': x} for _, n, y, x in best[:k]]
        with open(cache, 'w') as f:
            json.dump(chosen, f, indent=2)
    with open(cache) as f:
        chosen = json.load(f)

    by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in ds.files}
    for i, c in enumerate(chosen):
        gt_path = os.path.join(qual_dir, f'crop{i}_gt.png')
        # keyed off the last-written file so partially-written states heal
        if not os.path.exists(os.path.join(qual_dir,
                                           f'crop{i}_bilinear_err.png')):
            from data.datasets import load_image
            gt = load_image(by_name[c['name']])
            g = gt[:, c['y']:c['y'] + crop, c['x']:c['x'] + crop]
            b = bilinear_base(mosaic(gt))[:, c['y']:c['y'] + crop,
                                          c['x']:c['x'] + crop]
            save_image(g, gt_path)
            save_image(b, os.path.join(qual_dir, f'crop{i}_bilinear.png'))
            save_error_map(b, g, os.path.join(qual_dir,
                                              f'crop{i}_bilinear_err.png'))
    return chosen


def save_qual_crops(model, img_dir, qual_dir, run_name, tile, overlap,
                    device, crop=96):
    from data.datasets import load_image
    chosen = select_hard_crops(img_dir, qual_dir, k=4, crop=crop)
    ds = EvalImages(img_dir)
    by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in ds.files}
    for i, c in enumerate(chosen):
        gt = load_image(by_name[c['name']])
        pred, _ = tiled_inference(model, gt, tile, overlap, device)
        g = gt[:, c['y']:c['y'] + crop, c['x']:c['x'] + crop]
        p = pred[:, c['y']:c['y'] + crop, c['x']:c['x'] + crop]
        save_image(p, os.path.join(qual_dir, f'crop{i}_{run_name}.png'))
        save_error_map(p, g, os.path.join(qual_dir,
                                          f'crop{i}_{run_name}_err.png'))


# ------------------------------------------------------------------- CSV I/O

def upsert_row(csv_path, row, cols, key_cols):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for c in cols:
            if c not in df.columns:
                df[c] = float('nan')
    else:
        df = pd.DataFrame(columns=cols)
    mask = pd.Series(True, index=df.index)
    for k in key_cols:
        mask &= df[k].astype(str) == str(row[k])
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)[cols]
    df.to_csv(csv_path, index=False, float_format='%.4f')


# --------------------------------------------------------------------- main

def load_run(run_name, ckpt_dir, which='best', device='cpu'):
    path = os.path.join(ckpt_dir, run_name, f'{which}.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f'checkpoint not found: {path}')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = build_model(cfg['model'])
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)
    return model, cfg, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default=None,
                   help='loads checkpoints/<run_name>/best.pth (+its config)')
    p.add_argument('--config', default=None,
                   help='yaml config (required for --bilinear, else fallback)')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--which', default='best', choices=['best', 'last'])
    p.add_argument('--bilinear', action='store_true',
                   help='evaluate the bilinear base as the floor row')
    p.add_argument('--ablation', action='store_true',
                   help='write to results/pe_ablation.csv (Exp B schema)')
    p.add_argument('--csv', default=None)
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--datasets_root', default='./datasets')
    p.add_argument('--max_images', type=int, default=None)
    p.add_argument('--no_qual', action='store_true')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.bilinear:
        if not args.config:
            raise SystemExit('--bilinear needs --config for eval settings')
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model, run_name = None, 'bilinear'
    else:
        if not args.run_name:
            raise SystemExit('need --run_name (or --bilinear)')
        model, cfg, _ = load_run(args.run_name, args.checkpoint_dir,
                                 args.which, device)
        run_name = args.run_name

    ecfg = cfg['eval']
    tile, overlap, shave = ecfg['tile'], ecfg['tile_overlap'], ecfg['shave']
    test_sets = args.datasets or cfg['data']['test']

    row = {'model': run_name}
    if model is not None:
        prof = profile_model(model, ecfg.get('flops_size', 256), device)
        row.update({k: round(v, 4) for k, v in prof.items()})
        print(f"[eval] {run_name}: {row['params_M']:.3f}M params, "
              f"{row['flops_G']:.2f} GFLOPs@{ecfg.get('flops_size', 256)}, "
              f"{row['latency_ms']:.2f} ms")
    else:
        row.update({'params_M': 0.0, 'flops_G': float('nan'),
                    'latency_ms': float('nan')})

    for ds_name in test_sets:
        img_dir = resolve_test_dir(ds_name, args.datasets_root)
        if not os.path.isdir(img_dir):
            print(f'[eval] WARNING: {ds_name} not found at {img_dir}; skipping')
            continue
        m, per_image = eval_dataset(model, img_dir, tile, overlap, shave,
                                    device, args.max_images, desc=ds_name)
        row[f'{ds_name}_PSNR'] = round(m['psnr'], 4)
        row[f'{ds_name}_CPSNR'] = round(m['cpsnr'], 4)
        row[f'{ds_name}_SSIM'] = round(m['ssim'], 4)
        print(f"[eval] {run_name} on {ds_name}: "
              f"PSNR {m['psnr']:.3f}  CPSNR {m['cpsnr']:.3f}  "
              f"R/G/B {m['psnr_r']:.2f}/{m['psnr_g']:.2f}/{m['psnr_b']:.2f}  "
              f"SSIM {m['ssim']:.4f}")
        if not args.no_qual:
            qual_dir = os.path.join('figures', 'qual', ds_name)
            if model is None:
                select_hard_crops(img_dir, qual_dir)
            else:
                save_qual_crops(model, img_dir, qual_dir, run_name,
                                tile, overlap, device)

    if args.ablation:
        csv_path = args.csv or 'results/pe_ablation.csv'
        abl_row = {'pe_type': cfg['model']['pe_type'],
                   'window': cfg['model']['window'],
                   'params_M': row['params_M']}
        for ds in ('kodak', 'mcm'):
            for met in ('CPSNR', 'SSIM'):
                abl_row[f'{ds}_{met}'] = row.get(f'{ds}_{met}', float('nan'))
        upsert_row(csv_path, abl_row, ABL_COLS, ['pe_type', 'window'])
    else:
        csv_path = args.csv or 'results/main_results.csv'
        for c in MAIN_COLS:
            row.setdefault(c, float('nan'))
        upsert_row(csv_path, row, MAIN_COLS, ['model'])
    print(f'[eval] wrote {csv_path}')


if __name__ == '__main__':
    main()

"""Exp B runner: train + eval the ViT over pe_type x window grid.

Usage (from repo root):
  python scripts/run_ablation.py                    # full grid, ablation.yaml
  python scripts/run_ablation.py --smoke            # tiny models, ~100 iters
  python scripts/run_ablation.py --pe_types rope2d none --windows 8 16
"""
import argparse
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, retries=3):
    print('+', ' '.join(cmd), flush=True)
    for attempt in range(retries + 1):
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode == 0:
            return
        print(f'[ablation] attempt {attempt + 1} failed '
              f'(exit {proc.returncode}); '
              + ('retrying' if attempt < retries else 'giving up'), flush=True)
    raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/ablation.yaml')
    p.add_argument('--pe_types', nargs='+', default=None)
    p.add_argument('--windows', nargs='+', type=int, default=None)
    p.add_argument('--iters', type=int, default=None)
    p.add_argument('--val_dir', default=None,
                   help='forwarded to engine.train')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--gpu_cache', action='store_true',
                   help='forwarded to engine.train: precompute patches onto '
                        'the GPU instead of using the CPU DataLoader')
    p.add_argument('--resume', action='store_true',
                   help='forwarded to engine.train: resume each cell from '
                        'its last.pth if present, instead of restarting')
    p.add_argument('--skip_train', action='store_true',
                   help='eval existing checkpoints only')
    args = p.parse_args()

    with open(os.path.join(ROOT, args.config)) as f:
        cfg = yaml.safe_load(f)
    grid = cfg.get('ablation', {})
    pe_types = args.pe_types or grid.get('pe_types',
                                         ['none', 'learnable', 'sinusoidal2d',
                                          'rope2d'])
    windows = args.windows or grid.get('windows', [4, 8, 16])
    budget = 'tiny' if args.smoke else cfg['model'].get('budget', 'lite')

    py = sys.executable
    for pe in pe_types:
        for w in windows:
            name = f'vit_{budget}_pe-{pe}_w{w}'
            if not args.skip_train:
                cmd = [py, '-m', 'engine.train', '--config', args.config,
                       '--run_name', name, '--pe_type', pe, '--window', str(w)]
                if args.iters:
                    cmd += ['--iters', str(args.iters)]
                if args.val_dir:
                    cmd += ['--val_dir', args.val_dir]
                if args.smoke:
                    cmd += ['--smoke']
                if args.gpu_cache:
                    cmd += ['--gpu_cache']
                if args.resume:
                    cmd += ['--resume']
                run(cmd)
            run([py, '-m', 'engine.eval', '--run_name', name, '--ablation',
                 '--no_qual'])
    print('[ablation] grid complete -> results/pe_ablation.csv')


if __name__ == '__main__':
    main()

import argparse
import copy
import math
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import (EvalImages, TrainPatches, build_gpu_patch_pool,
                           gpu_pool_batches)
from engine.eval import tiled_inference
from engine.metrics import img_metrics, mean_metrics
from models.build import WIDTH_KEYS, build_model, model_tag
from utils.logger import TrainLogger
from utils.seed import seed_everything, worker_init_fn

SMOKE = {'iters': 500, 'val_every': 250, 'batch': 8, 'num_workers': 0,
         'val_max_images': 2, 'budget': 'tiny'}
# --overfit needs more iters than the generic smoke budget: the cosine decay
# has to reach a small LR *after* loss has actually converged, or it just
# looks like a truncated descent (see acceptance check 3).
OVERFIT_ITERS = 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run_name', default=None)
    p.add_argument('--smoke', action='store_true',
                   help='~500 iters, tiny model: fast end-to-end sanity pass')
    p.add_argument('--overfit', action='store_true',
                   help='train on one fixed batch (acceptance check 3)')
    p.add_argument('--resume', action='store_true',
                   help='resume from checkpoints/<run_name>/last.pth if it '
                        'exists (model + optimizer + scaler + iter count)')
    p.add_argument('--l2', action='store_true', help='L2 loss instead of L1')
    p.add_argument('--grad_checkpoint', action='store_true')
    p.add_argument('--tensorboard', action='store_true')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--gpu_cache', action='store_true',
                   help='precompute a fixed pool of GT patches straight '
                        'onto the GPU and skip the CPU DataLoader entirely '
                        '(no num_workers/host-RAM tradeoff; see '
                        'data.datasets.build_gpu_patch_pool)')
    p.add_argument('--gpu_cache_patches', type=int, default=20000,
                   help='pool size for --gpu_cache (patches, not images)')
    # config overrides
    p.add_argument('--train_dir', default=None)
    p.add_argument('--val_dir', default=None)
    p.add_argument('--iters', type=int, default=None)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--patch', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--val_every', type=int, default=None)
    p.add_argument('--val_max_images', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--budget', default=None)
    p.add_argument('--pe_type', default=None)
    p.add_argument('--window', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    return p.parse_args()


def resolve_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m, d, t = cfg['model'], cfg['data'], cfg['train']
    orig_budget = m.get('budget', 'lite')
    for key, val in [('budget', args.budget), ('pe_type', args.pe_type),
                     ('window', args.window)]:
        if val is not None:
            m[key] = val
    for key, val in [('patch', args.patch), ('val_max_images', args.val_max_images),
                     ('num_workers', args.num_workers),
                     ('train_dir', args.train_dir), ('val_dir', args.val_dir)]:
        if val is not None:
            d[key] = val
    for key, val in [('total_iters', args.iters), ('batch', args.batch),
                     ('lr', args.lr), ('seed', args.seed),
                     ('val_every', args.val_every)]:
        if val is not None:
            t[key] = val
    if args.l2:
        t['loss'] = 'l2'
    if args.deterministic:
        t['deterministic'] = True
    if args.smoke:
        default_iters = OVERFIT_ITERS if args.overfit else SMOKE['iters']
        t['total_iters'] = args.iters or default_iters
        t['val_every'] = args.val_every or SMOKE['val_every']
        t['batch'] = args.batch or SMOKE['batch']
        d['num_workers'] = SMOKE['num_workers']
        d['val_max_images'] = args.val_max_images or SMOKE['val_max_images']
        if args.budget is None:
            m['budget'] = SMOKE['budget']
    if m.get('budget', 'lite') != orig_budget:
        for k in WIDTH_KEYS[m['name']]:
            m.pop(k, None)
    return cfg


def cosine_lr(base_lr, it, total):
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * it / total))


@torch.no_grad()
def validate(model, val_ds, ecfg, device):
    model.eval()
    rows = [img_metrics(
        tiled_inference(model, gt, ecfg['tile'], ecfg['tile_overlap'],
                        device)[0], gt, ecfg['shave'])
        for _, gt in val_ds]
    model.train()
    return mean_metrics(rows)


def main():
    args = parse_args()
    cfg = resolve_config(args)
    mcfg, dcfg, tcfg, ecfg = (cfg['model'], cfg['data'], cfg['train'],
                              cfg['eval'])
    run_name = args.run_name or model_tag(mcfg) + ('_smoke' if args.smoke else '')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(tcfg.get('amp', True)) and device == 'cuda'

    seed_everything(tcfg['seed'], tcfg.get('deterministic', False))

    model = build_model(mcfg, grad_checkpoint=args.grad_checkpoint).to(device)

    if args.gpu_cache:
        print(f'[gpu_cache] building patch pool ({args.gpu_cache_patches} '
              f'patches) from {dcfg["train_dir"]} ...', flush=True)
        pool = build_gpu_patch_pool(dcfg['train_dir'], dcfg['patch'],
                                    args.gpu_cache_patches, device,
                                    seed=tcfg['seed'])
        pool_gb = pool.element_size() * pool.nelement() / 1e9
        print(f'[gpu_cache] pool ready: {pool.shape[0]} patches, '
              f'{pool_gb:.2f} GB on {device}', flush=True)
        data_iter = gpu_pool_batches(pool, tcfg['batch'], device)
    else:
        train_ds = TrainPatches(dcfg['train_dir'], patch=dcfg['patch'])
        loader = DataLoader(
            train_ds, batch_size=tcfg['batch'], num_workers=dcfg.get('num_workers', 4),
            pin_memory=(device == 'cuda'), drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=dcfg.get('num_workers', 4) > 0,
            generator=torch.Generator().manual_seed(tcfg['seed']))
        data_iter = iter(loader)
    val_ds = EvalImages(dcfg['val_dir'],
                        max_images=dcfg.get('val_max_images', 20))

    opt = torch.optim.Adam(model.parameters(), lr=tcfg['lr'],
                           betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    loss_fn = (torch.nn.MSELoss() if tcfg.get('loss', 'l1') == 'l2'
               else torch.nn.L1Loss())

    ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    start_iter = 0
    best_cpsnr = -1.0
    last_path = os.path.join(ckpt_dir, 'last.pth')
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            opt.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_iter = ckpt['iter']
        best_cpsnr = ckpt.get('best_cpsnr', ckpt.get('val_cpsnr', -1.0))
        print(f'[{run_name}] resumed from {last_path} @ iter {start_iter} '
              f'(best CPSNR so far: {best_cpsnr:.3f})', flush=True)

    logger = TrainLogger(run_name, args.results_dir, args.tensorboard)
    logger.console(f'device={device} amp={use_amp} iters={tcfg["total_iters"]} '
                   f'batch={tcfg["batch"]} patch={dcfg["patch"]} '
                   f'loss={tcfg.get("loss", "l1")} seed={tcfg["seed"]}')

    total = tcfg['total_iters']
    fixed_batch = None
    pbar = tqdm(range(start_iter + 1, total + 1), desc=run_name,
               dynamic_ncols=True, initial=start_iter, total=total)
    for it in pbar:
        if args.overfit:
            if fixed_batch is None:
                fixed_batch = next(data_iter)
            masked, base, gt = fixed_batch
        else:
            masked, base, gt = next(data_iter)
        masked = masked.to(device, non_blocking=True)
        base = base.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        lr = cosine_lr(tcfg['lr'], it - 1, total)
        for g in opt.param_groups:
            g['lr'] = lr

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(masked, base)
            loss = loss_fn(out, gt)
        scaler.scale(loss).backward()
        if tcfg.get('grad_clip'):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           tcfg['grad_clip'])
        scaler.step(opt)
        scaler.update()

        logger.train_step(it, loss.item(), lr)
        if it % 50 == 0:
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr:.1e}')

        if it % tcfg['val_every'] == 0 or it == total:
            metrics = validate(model, val_ds, ecfg, device)
            logger.val_step(it, metrics)
            is_best = metrics['cpsnr'] > best_cpsnr
            if is_best:
                best_cpsnr = metrics['cpsnr']
            state = {'model': model.state_dict(), 'iter': it,
                     'val_cpsnr': metrics['cpsnr'], 'config': cfg,
                     'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
                     'best_cpsnr': best_cpsnr}
            torch.save(state, os.path.join(ckpt_dir, 'last.pth'))
            if is_best:
                torch.save(state, os.path.join(ckpt_dir, 'best.pth'))
                logger.console(f'new best CPSNR {best_cpsnr:.3f} @ iter {it}')

    logger.console(f'done. best val CPSNR {best_cpsnr:.3f}; '
                   f'checkpoints in {ckpt_dir}')
    logger.close()


if __name__ == '__main__':
    main()

"""Model factory: config dict -> model, with per-budget presets.

Widths are tuned so the three models land within ~10% of each other at each
budget (lite ~1.25M, base ~4.9M); actual counts are printed on init.
"""
import torch.nn as nn

from .cnn import CNNDemosaic
from .swinir import SwinIRDemosaic
from .vit import ViTDemosaic

PRESETS = {
    'cnn': {
        'lite': dict(n_feats=64, n_resblocks=16, res_scale=1.0),
        'base': dict(n_feats=128, n_resblocks=16, res_scale=1.0),
        'tiny': dict(n_feats=24, n_resblocks=3, res_scale=1.0),
    },
    'swinir': {
        'lite': dict(embed_dim=60, depths=[6, 6, 6, 6], heads=[6, 6, 6, 6],
                     window=8, mlp_ratio=4.0),
        'base': dict(embed_dim=120, depths=[6, 6, 6, 6], heads=[6, 6, 6, 6],
                     window=8, mlp_ratio=4.0),
        'tiny': dict(embed_dim=24, depths=[2, 2], heads=[4, 4],
                     window=8, mlp_ratio=2.0),
    },
    'vit': {
        'lite': dict(dim=120, depth=6, heads=6, patch_embed=4, window=8,
                     pe_type='rope2d', mlp_ratio=4.0),
        'base': dict(dim=240, depth=6, heads=6, patch_embed=4, window=8,
                     pe_type='rope2d', mlp_ratio=4.0),
        'tiny': dict(dim=48, depth=2, heads=4, patch_embed=4, window=8,
                     pe_type='rope2d', mlp_ratio=2.0),
    },
}

# width/size keys that belong to a budget preset: when a run switches budget
# (e.g. --smoke -> tiny, --budget base), these yaml keys are dropped so the
# preset wins; knobs like pe_type/window survive the switch.
WIDTH_KEYS = {
    'cnn': ['n_feats', 'n_resblocks', 'res_scale'],
    'swinir': ['embed_dim', 'depths', 'heads', 'mlp_ratio'],
    'vit': ['dim', 'depth', 'heads', 'mlp_ratio', 'patch_embed'],
}

# model-config keys that are forwarded to each constructor
ARG_KEYS = {
    'cnn': {'n_feats', 'n_resblocks', 'res_scale'},
    'swinir': {'embed_dim', 'depths', 'heads', 'window', 'mlp_ratio',
               'grad_checkpoint'},
    'vit': {'dim', 'depth', 'heads', 'patch_embed', 'window', 'pe_type',
            'mlp_ratio', 'grad_checkpoint'},
}

CLASSES = {'cnn': CNNDemosaic, 'swinir': SwinIRDemosaic, 'vit': ViTDemosaic}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_cfg, grad_checkpoint=False, verbose=True):
    name = model_cfg['name']
    budget = model_cfg.get('budget', 'lite')
    if name not in CLASSES:
        raise ValueError(f'unknown model: {name}')
    args = dict(PRESETS[name][budget])
    # explicit config keys override the preset
    for k, v in model_cfg.items():
        if k in ARG_KEYS[name]:
            args[k] = v
    if grad_checkpoint and name in ('swinir', 'vit'):
        args['grad_checkpoint'] = True

    model = CLASSES[name](**args)
    n = count_params(model)
    if verbose:
        desc = ', '.join(f'{k}={v}' for k, v in args.items())
        print(f'[build] {name} ({budget}): {n / 1e6:.3f}M params ({desc})')
    return model


def model_tag(model_cfg):
    """Short run tag, e.g. vit_lite or vit_lite_pe-rope2d_w8 for ablations."""
    tag = f"{model_cfg['name']}_{model_cfg.get('budget', 'lite')}"
    if model_cfg.get('ablation_tag'):
        tag += f"_pe-{model_cfg['pe_type']}_w{model_cfg['window']}"
    return tag

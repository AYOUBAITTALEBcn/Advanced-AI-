"""Seeding + determinism helpers."""
import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    """Give each DataLoader worker a distinct, seed-derived RNG state."""
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)

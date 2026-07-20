"""Datasets: random training patches, and full images for val/test.

Training: random-crop `patch` px from GT, augment (h/v flip + 90/180/270
rotation) BEFORE mosaicing so the crop always has RGGB phase, then mosaic.
Returns (masked3, bilinear_base, gt).

Val/test use full GT images; tiled inference happens in engine/eval.py.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .mosaic import mosaic, bilinear_base

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# Named test sets resolve to these folders under the datasets root.
TESTSET_DIRS = {'kodak': 'Kodak', 'mcm': 'McM'}


def list_images(root):
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f'Image folder not found: {root}\n'
            f'Run `python -m data.download` or drop images in manually.')
    files = sorted(
        os.path.join(root, f) for f in os.listdir(root)
        if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f'No images found in {root}')
    return files


def load_image(path):
    """-> float32 tensor [3,H,W] in [0,1], H/W cropped to even (RGGB phase)."""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    arr = arr[:h - h % 2, :w - w % 2]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class TrainPatches(Dataset):
    """Random augmented GT patches with mosaic + bilinear base."""

    def __init__(self, root, patch=64, augment=True, virtual_len=10 ** 9):
        self.files = list_images(root)
        self.patch = patch
        self.augment = augment
        self.virtual_len = virtual_len

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        path = self.files[idx % len(self.files)]
        img = Image.open(path).convert('RGB')
        w, h = img.size
        p = self.patch
        if w < p or h < p:
            raise ValueError(f'{path} ({w}x{h}) smaller than patch {p}')
        x0 = random.randint(0, w - p)
        y0 = random.randint(0, h - p)
        arr = np.asarray(img.crop((x0, y0, x0 + p, y0 + p)), dtype=np.float32) / 255.0
        if self.augment:
            if random.random() < 0.5:
                arr = arr[:, ::-1]
            if random.random() < 0.5:
                arr = arr[::-1, :]
            arr = np.rot90(arr, k=random.randint(0, 3))
        gt = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        masked3 = mosaic(gt)
        base = bilinear_base(masked3)
        return masked3, base, gt


def build_gpu_patch_pool(root, patch, total_patches, device, seed=0):
    """Decode every training image once, extract a large fixed pool of GT
    crops, and move the whole pool to `device`.

    Trades unlimited random crops for a large-but-fixed pool sampled once up
    front. With the pool GPU-resident, training needs zero CPU/disk work per
    step (no DataLoader, no `num_workers` -> host-RAM tradeoff): see
    `gpu_pool_batches`, which augments and mosaics directly on `device`.
    """
    files = list_images(root)
    rng = random.Random(seed)
    per_image = max(1, total_patches // len(files))
    pool = np.empty((per_image * len(files), patch, patch, 3), dtype=np.float32)
    n = 0
    for path in files:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if w < patch or h < patch:
            continue
        arr = np.asarray(img, dtype=np.float32) / 255.0
        for _ in range(per_image):
            x0 = rng.randint(0, w - patch)
            y0 = rng.randint(0, h - patch)
            pool[n] = arr[y0:y0 + patch, x0:x0 + patch]
            n += 1
    t = torch.from_numpy(pool[:n]).permute(0, 3, 1, 2).contiguous()
    return t.to(device)


def _augment_gpu(gt):
    """In-place random h/v flip + 90/180/270 rotation, per sample."""
    for i in range(gt.shape[0]):
        if random.random() < 0.5:
            gt[i] = gt[i].flip(-1)
        if random.random() < 0.5:
            gt[i] = gt[i].flip(-2)
        k = random.randint(0, 3)
        if k:
            gt[i] = torch.rot90(gt[i], k, dims=(-2, -1))
    return gt


def gpu_pool_batches(pool, batch_size, device):
    """Infinite generator of (masked3, base, gt) batches, entirely on-device."""
    n = pool.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,), device=device)
        gt = _augment_gpu(pool[idx].clone())
        masked3 = mosaic(gt)
        base = bilinear_base(masked3)
        yield masked3, base, gt


class EvalImages(Dataset):
    """Full GT images (name, gt); mosaic/base are built at inference time."""

    def __init__(self, root, max_images=None):
        self.files = list_images(root)
        if max_images:
            self.files = self.files[:max_images]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        return name, load_image(path)


def resolve_test_dir(name, datasets_root='./datasets'):
    """'kodak' / 'mcm' -> folder path; a path passes through unchanged."""
    if name.lower() in TESTSET_DIRS:
        return os.path.join(datasets_root, TESTSET_DIRS[name.lower()])
    return name

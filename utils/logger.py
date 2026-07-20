"""Console + CSV logging (+ optional TensorBoard)."""
import csv
import os
import sys
import time


class CSVLogger:
    """Append dict rows to a CSV; header written once, buffered writes."""

    def __init__(self, path, fieldnames, flush_every=100):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, 'a', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames,
                                     extrasaction='ignore')
        if new:
            self.writer.writeheader()
            self.f.flush()
        self.flush_every = flush_every
        self._n = 0

    def log(self, row):
        self.writer.writerow(row)
        self._n += 1
        if self._n % self.flush_every == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()


class TrainLogger:
    """Curves CSV (train + val rows) with console echo and optional TB."""

    FIELDS = ['iter', 'split', 'loss', 'lr', 'cpsnr', 'psnr', 'ssim', 'time']

    def __init__(self, run_name, results_dir='./results', tensorboard=False):
        self.run_name = run_name
        self.csv = CSVLogger(os.path.join(results_dir, f'{run_name}_curves.csv'),
                             self.FIELDS)
        self.t0 = time.time()
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(results_dir, 'tb', run_name))
            except ImportError:
                print('[logger] tensorboard not installed; skipping')

    def train_step(self, it, loss, lr):
        self.csv.log({'iter': it, 'split': 'train', 'loss': f'{loss:.6f}',
                      'lr': f'{lr:.3e}', 'time': f'{time.time() - self.t0:.1f}'})
        if self.tb:
            self.tb.add_scalar('train/loss', loss, it)

    def val_step(self, it, metrics):
        self.csv.log({'iter': it, 'split': 'val',
                      'cpsnr': f"{metrics['cpsnr']:.4f}",
                      'psnr': f"{metrics['psnr']:.4f}",
                      'ssim': f"{metrics['ssim']:.4f}",
                      'time': f'{time.time() - self.t0:.1f}'})
        self.csv.f.flush()
        if self.tb:
            for k in ('cpsnr', 'psnr', 'ssim'):
                self.tb.add_scalar(f'val/{k}', metrics[k], it)
        print(f"[{self.run_name}] iter {it}: val CPSNR {metrics['cpsnr']:.3f} "
              f"PSNR {metrics['psnr']:.3f} SSIM {metrics['ssim']:.4f}")

    def console(self, msg):
        print(f'[{self.run_name}] {msg}')
        sys.stdout.flush()

    def close(self):
        self.csv.close()
        if self.tb:
            self.tb.close()

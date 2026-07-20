# demosaic-tc — Transformer vs. CNN for Bayer Demosaicing (+ PE Ablation)

Single-image RGGB demosaicing: an EDSR-style CNN vs. a SwinIR-style
Transformer vs. a plain windowed ViT, all predicting a residual over the same
bilinear base. **Exp A** benchmarks accuracy + params/FLOPs/latency; **Exp B**
ablates positional-encoding type x attention-window size on the ViT (a 2D
"length-aware" study).

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Python 3.10+ / PyTorch 2.x, single GPU. AMP is on by default; add
`--grad_checkpoint` to the Transformer trainings if memory is tight.

## Data

```bash
python -m data.download                      # DIV2K train/val + Kodak + McM
python -m data.download --datasets kodak mcm # just the test sets
```

If a mirror is down the script prints the expected `datasets/` layout so you
can drop files in manually (DIV2K_train_HR / DIV2K_valid_HR / Kodak / McM).

## Smoke test (acceptance check 6)

End-to-end sanity pass with tiny models (~500 iters each):

```bash
bash scripts/run_all.sh --smoke
```

## Full pipeline

```bash
bash scripts/run_all.sh
```

Or stage by stage:

```bash
# Exp A: train + eval the three main models (150k iters each)
python -m engine.train --config configs/cnn.yaml --gpu_cache
python -m engine.train --config configs/swinir.yaml --gpu_cache --grad_checkpoint
python -m engine.train --config configs/vit.yaml --gpu_cache
python -m engine.eval  --config configs/cnn.yaml --bilinear   # floor row
python -m engine.eval  --run_name cnn_lite
python -m engine.eval  --run_name swinir_lite
python -m engine.eval  --run_name vit_lite

# Exp B: PE x window grid on the ViT (60k iters per cell)
python scripts/run_ablation.py --gpu_cache

# tables + figures + summary
python scripts/make_figures.py
```

Or, equivalently: `bash scripts/run_all.sh --gpu_cache --resume`.

Useful switches: `--budget base` (~5M-param models), `--l2`, `--overfit`
(one-batch sanity), `--deterministic`, `--iters/--batch/--patch/--lr/--seed`,
`--resume` (continue from `checkpoints/<run>/last.pth`: model + optimizer +
AMP scaler + iter count, so an interrupted multi-hour run doesn't lose
progress -- safe to always pass, a no-op if there's nothing to resume).

### `--gpu_cache`: fixing the real bottleneck on this hardware

With the default `num_workers=0` (kept low deliberately -- each DataLoader
worker re-imports torch, and this machine only has ~15GB RAM with little
free), the per-iteration cost is dominated by decoding a full-resolution
DIV2K image on CPU just to crop a 64x64 patch. Measured on the reference
laptop (RTX 3060, 6GB VRAM): **~1.0s/iter** regardless of model (CNN and ViT
lite measured nearly identically, which is the signature of a CPU-bound, not
GPU-bound, loop) -- at that rate the full 150k/60k schedule is ~14 days of
sequential compute.

`--gpu_cache` (`data.datasets.build_gpu_patch_pool` +
`data.datasets.gpu_pool_batches`) decodes the training set **once**,
extracts a large fixed pool of GT crops (`--gpu_cache_patches`, default
20000 -> ~1GB), and moves the whole pool to VRAM. Every training step then
samples, augments (flip/rot90), mosaics, and builds the bilinear base
entirely on-device -- zero CPU/disk work in the hot loop, and no
`num_workers` vs. host-RAM tradeoff at all. Measured throughput jumps to
**~20 it/s (CNN), ~22.5 it/s (ViT)** -- a ~20x speedup -- which brings the
full schedule down to **roughly a day** of sequential compute:

| stage | iters | it/s | time |
|---|---|---|---|
| CNN lite | 150k | ~20 | ~2.1 h |
| ViT lite | 150k | ~22.5 | ~1.9 h |
| SwinIR lite (batch 8, `--grad_checkpoint`) | 150k | ~3.9 | ~10.7 h |
| ablation, 12 cells x 60k | 60k/cell | ~20-22 | ~9.5 h |
| **total (sequential)** | | | **~24 h** |

SwinIR is the deepest model (24 transformer blocks) and needs
`--grad_checkpoint`; even then, batch 16 (the cnn/vit default) OOMs a 6GB
GPU here, both in VRAM (no checkpointing) and in host RAM during validation
(with checkpointing) -- `configs/swinir.yaml` defaults its batch to 8
accordingly. Add validation overhead (periodic tiled inference over
`val_max_images`) on top of the table above; it isn't included.

The pool trades unlimited random crops for a large-but-fixed sample --
20000 patches from 800 images is ~25/image, refreshed every run via
`--seed`, not every epoch. Bump `--gpu_cache_patches` if VRAM allows; each
1000 patches at patch=64 costs ~49MB.

## Outputs

- `results/main_results.csv` — model, params_M, flops_G, latency_ms,
  kodak/mcm PSNR + CPSNR + SSIM (plus the bilinear floor row)
- `results/pe_ablation.csv` — pe_type, window, params_M, kodak/mcm CPSNR + SSIM
- `results/<run>_curves.csv` — per-iter train loss + val metrics
- `results/main_results.md|.tex`, `results/pe_ablation.md|.tex` — tables
- `results/summary.md` — **the single hand-back file** (all tables, figure
  list, raw CSVs)
- `figures/*.png` @300dpi + `figures/captions.md`
- `checkpoints/<run>/{best,last}.pth` + the resolved `config.yaml` per run

## Conventions

- float32 in [0,1] internally; PSNR/CPSNR/SSIM in the 8-bit domain after
  shaving 8 px. CPSNR = mean of per-channel PSNRs.
- Bayer RGGB everywhere: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B.
- Test/val use tiled inference (64-px tiles, 16-px feathered overlap, even
  tile origins to preserve RGGB phase) so the ViT token grid stays fixed and
  learnable PE is valid at any image size.
- SwinIR keeps its relative-position bias (PE baked in) and is not part of
  the PE ablation.

## Acceptance checks

```bash
python -m checks.run_checks            # shapes, param match, PE sanity
python -m engine.train --config configs/cnn.yaml --smoke --overfit  # check 3
python -m engine.eval --config configs/cnn.yaml --bilinear          # check 2
bash scripts/run_all.sh --smoke                                     # check 6
```

Check 3 (`--overfit`) trains on one fixed batch with the normal cosine
schedule, over `OVERFIT_ITERS=2000` iters (longer than the generic
`--smoke` default of 500, so the decay actually reaches a converged tail
instead of cutting the run off mid-descent). "Near-zero" loss in practice
means a sharp drop (~90%+) that flattens into a clean plateau, not literally
0.000 — the tiny/lite CNN has a nonzero representational floor for real
photo texture at 8 fixed patches (empirically ~0.006-0.008 L1 for the tiny
smoke preset), and that floor is independent of LR, schedule, AMP, and grad
clipping. What the check actually verifies is that the loop drives loss down
sharply and plateaus stably — not that it diverges, NaNs, or sits flat at
the initial value.

Check 5 (determinism) needs `--deterministic` on both runs to be bit-exact:
`python -m engine.train --config configs/cnn.yaml --smoke --iters 100 --deterministic`,
twice with the same `--run_name` avoided (so neither checkpoint clobbers the
other), then diff the two `results/<run>_curves.csv`. Without `--deterministic`,
`cudnn.benchmark=True` lets cuDNN pick from several algorithms at runtime and
the loss trace drifts by the 5th-6th significant digit — expected, not a bug.
With it, the loss/metric columns match exactly (only the elapsed-time column
differs, since that's wall-clock).

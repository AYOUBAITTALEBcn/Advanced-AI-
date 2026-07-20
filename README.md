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



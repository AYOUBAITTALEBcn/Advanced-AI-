# Figure captions

**pe_ablation_cpsnr_vs_window.png** - CPSNR vs. attention-window size on the plain windowed ViT, one line per positional-encoding type (window 16 = global attention on the 16x16 token grid). The length-aware hypothesis predicts the gap between none and rope2d/sinusoidal2d widening with window size.

**efficiency_frontier.png** - Efficiency frontier: Kodak CPSNR vs. parameters (left) and FLOPs at 256x256 (right) for the three param-matched models.

**qualitative_kodak.png** - Qualitative comparison on the four hardest Kodak crops (highest bilinear error): GT / bilinear / models, with mean-abs-error heatmaps (shared scale) beneath each row.

**qualitative_mcm.png** - Qualitative comparison on the four hardest McMaster crops, laid out as in the Kodak panel.

**training_curves.png** - Validation CPSNR vs. training iteration for the three main models (DIV2K val subset).


"""CSV -> tables (md + tex) + figures (png @300dpi) + captions + summary.

Reads results/main_results.csv, results/pe_ablation.csv and the per-run
curves CSVs; writes tables to results/, figures to figures/, and compiles
results/summary.md (the single hand-back file).

Run from the repo root:  python scripts/make_figures.py
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS, FIGURES = 'results', 'figures'
DPI = 300

# Validated reference palette (dataviz skill): categorical slots in fixed
# order — color follows the entity in every figure, never its rank.
SLOT = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948']
MODEL_COLOR = {'bilinear': '#898781', 'cnn': SLOT[0], 'swinir': SLOT[1],
               'vit': SLOT[2]}
PE_COLOR = {'none': SLOT[0], 'learnable': SLOT[1], 'sinusoidal2d': SLOT[2],
            'rope2d': SLOT[3]}
PE_MARKER = {'none': 'o', 'learnable': 's', 'sinusoidal2d': '^', 'rope2d': 'D'}
MODEL_MARKER = {'bilinear': 'x', 'cnn': 'o', 'swinir': 's', 'vit': '^'}

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE = '#e1e0d9', '#c3c2b7'

DISPLAY = {'bilinear': 'Bilinear', 'cnn': 'CNN (EDSR-style)',
           'swinir': 'SwinIR', 'vit': 'ViT'}


def model_key(name):
    return str(name).split('_')[0]


def display_name(name):
    key = model_key(name)
    disp = DISPLAY.get(key, name)
    if '_base' in str(name):
        disp += ' (base)'
    return disp


def style_ax(ax):
    ax.set_facecolor('white')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=1.0)
    ax.set_axisbelow(True)


def savefig(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[figures] wrote {path}')
    return path


# ------------------------------------------------------------------- tables

def df_to_md(df, floatfmt='.3f'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        cells = [f'{v:{floatfmt}}' if isinstance(v, float) and not np.isnan(v)
                 else ('--' if isinstance(v, float) else str(v)) for v in r]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def df_to_tex(df, caption, label, floatfmt='.3f'):
    cols = list(df.columns)
    head = ' & '.join(c.replace('_', r'\_') for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = [f'{v:{floatfmt}}' if isinstance(v, float) and not np.isnan(v)
                 else ('--' if isinstance(v, float) else str(v).replace('_', r'\_'))
                 for v in r]
        rows.append(' & '.join(cells) + r' \\')
    return '\n'.join([
        r'\begin{table}[t]', r'\centering', r'\small',
        rf'\caption{{{caption}}}', rf'\label{{{label}}}',
        r'\begin{tabular}{l' + 'r' * (len(cols) - 1) + '}',
        r'\toprule', head + r' \\', r'\midrule', *rows,
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def write_tables(main_df, abl_df):
    outs = {}
    if main_df is not None:
        df = main_df.copy()
        df.insert(0, 'Model', df.pop('model').map(display_name))
        with open(f'{RESULTS}/main_results.md', 'w') as f:
            f.write('# Main results (Exp A)\n\n' + df_to_md(df))
        with open(f'{RESULTS}/main_results.tex', 'w') as f:
            f.write(df_to_tex(df, 'Demosaicing on Kodak and McMaster: '
                              'accuracy and efficiency.', 'tab:main'))
        outs['main'] = df
        print(f'[figures] wrote {RESULTS}/main_results.md/.tex')
    if abl_df is not None:
        df = abl_df.sort_values(['pe_type', 'window']).reset_index(drop=True)
        with open(f'{RESULTS}/pe_ablation.md', 'w') as f:
            f.write('# PE x window ablation (Exp B)\n\n' + df_to_md(df))
        with open(f'{RESULTS}/pe_ablation.tex', 'w') as f:
            f.write(df_to_tex(df, 'Positional-encoding ablation on the '
                              'windowed ViT.', 'tab:ablation'))
        outs['abl'] = df
        print(f'[figures] wrote {RESULTS}/pe_ablation.md/.tex')
    return outs


# ------------------------------------------------------------------ figures

def fig_pe_lines(abl_df):
    """The money figure: CPSNR vs window, one line per PE type."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, ds, title in zip(axes, ('kodak', 'mcm'), ('Kodak', 'McMaster')):
        col = f'{ds}_CPSNR'
        for pe in ('none', 'learnable', 'sinusoidal2d', 'rope2d'):
            sub = abl_df[abl_df.pe_type == pe].sort_values('window')
            if sub.empty:
                continue
            ax.plot(sub.window, sub[col], color=PE_COLOR[pe],
                    marker=PE_MARKER[pe], markersize=5, linewidth=1.8,
                    label=pe)
            ax.annotate(pe, (sub.window.iloc[-1], sub[col].iloc[-1]),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=7, color=PE_COLOR[pe], va='center')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(abl_df.window.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('attention window (tokens)')
        ax.set_title(title, fontsize=9)
        style_ax(ax)
    axes[0].set_ylabel('CPSNR (dB)')
    axes[0].legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout()
    return savefig(fig, 'pe_ablation_cpsnr_vs_window.png')


def fig_efficiency(main_df):
    df = main_df[main_df.model != 'bilinear']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, xcol, xlabel in zip(axes, ('params_M', 'flops_G'),
                                ('parameters (M)', 'FLOPs (G) @ 256x256')):
        for _, r in df.iterrows():
            key = model_key(r.model)
            ax.scatter(r[xcol], r.kodak_CPSNR, s=45,
                       color=MODEL_COLOR.get(key, INK2),
                       marker=MODEL_MARKER.get(key, 'o'), zorder=3)
            ax.annotate(display_name(r.model), (r[xcol], r.kodak_CPSNR),
                        xytext=(5, 4), textcoords='offset points',
                        fontsize=7, color=INK2)
        ax.set_xlabel(xlabel)
        style_ax(ax)
    axes[0].set_ylabel('Kodak CPSNR (dB)')
    fig.tight_layout()
    return savefig(fig, 'efficiency_frontier.png')


def fig_curves():
    paths = []
    runs = []
    for f in sorted(glob.glob(f'{RESULTS}/*_curves.csv')):
        run = os.path.basename(f)[:-len('_curves.csv')]
        if 'pe-' in run or 'smoke' in run:  # main models only
            continue
        df = pd.read_csv(f)
        val = df[df.split == 'val']
        if len(val):
            runs.append((run, val))
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for run, val in runs:
        key = model_key(run)
        ax.plot(val['iter'], val.cpsnr, color=MODEL_COLOR.get(key, INK2),
                linewidth=1.8, label=display_name(run))
        ax.annotate(display_name(run),
                    (val['iter'].iloc[-1], val.cpsnr.iloc[-1]),
                    xytext=(4, 0), textcoords='offset points',
                    fontsize=7, color=MODEL_COLOR.get(key, INK2), va='center')
    ax.set_xlabel('iteration')
    ax.set_ylabel('val CPSNR (dB)')
    style_ax(ax)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout()
    paths.append(savefig(fig, 'training_curves.png'))
    return paths[0]


def fig_qual_panels(main_df):
    """GT / bilinear / models columns; error-map row under each crop row."""
    paths = []
    models = [m for m in main_df.model if m != 'bilinear'] if main_df is not None else []
    for ds in ('kodak', 'mcm'):
        qdir = os.path.join(FIGURES, 'qual', ds)
        if not os.path.isdir(qdir):
            continue
        crops = sorted({f.split('_')[0] for f in os.listdir(qdir)
                        if f.startswith('crop') and f.endswith('_gt.png')})
        if not crops:
            continue
        col_names = ['gt', 'bilinear'] + models
        col_disp = ['GT', 'Bilinear'] + [display_name(m) for m in models]
        ncols, nrows = len(col_names), 2 * len(crops)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(1.55 * ncols, 1.62 * nrows),
            gridspec_kw={'wspace': 0.04, 'hspace': 0.04})
        axes = np.atleast_2d(axes)
        for ci, crop in enumerate(crops):
            for mi, (mname, mdisp) in enumerate(zip(col_names, col_disp)):
                ax_img = axes[2 * ci, mi]
                ax_err = axes[2 * ci + 1, mi]
                img_p = os.path.join(qdir, f'{crop}_{mname}.png')
                err_p = os.path.join(qdir, f'{crop}_{mname}_err.png')
                for ax, p in ((ax_img, img_p), (ax_err, err_p)):
                    ax.set_xticks([]), ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_visible(False)
                    if os.path.exists(p):
                        ax.imshow(np.asarray(Image.open(p)))
                    else:
                        ax.set_facecolor('white')
                if ci == 0:
                    ax_img.set_title(mdisp, fontsize=8, color=INK)
                if mi == 0:
                    ax_img.set_ylabel(crop, fontsize=7, color=INK2)
                    ax_err.set_ylabel('error', fontsize=7, color=MUTED)
        paths.append(savefig(fig, f'qualitative_{ds}.png'))
    return paths


# ---------------------------------------------------------------- summaries

CAPTIONS = {
    'pe_ablation_cpsnr_vs_window.png':
        'CPSNR vs. attention-window size on the plain windowed ViT, one line '
        'per positional-encoding type (window 16 = global attention on the '
        '16x16 token grid). The length-aware hypothesis predicts the gap '
        'between none and rope2d/sinusoidal2d widening with window size.',
    'efficiency_frontier.png':
        'Efficiency frontier: Kodak CPSNR vs. parameters (left) and FLOPs at '
        '256x256 (right) for the three param-matched models.',
    'training_curves.png':
        'Validation CPSNR vs. training iteration for the three main models '
        '(DIV2K val subset).',
    'qualitative_kodak.png':
        'Qualitative comparison on the four hardest Kodak crops (highest '
        'bilinear error): GT / bilinear / models, with mean-abs-error '
        'heatmaps (shared scale) beneath each row.',
    'qualitative_mcm.png':
        'Qualitative comparison on the four hardest McMaster crops, laid out '
        'as in the Kodak panel.',
}


def write_captions(paths):
    os.makedirs(FIGURES, exist_ok=True)
    with open(f'{FIGURES}/captions.md', 'w') as f:
        f.write('# Figure captions\n\n')
        for p in paths:
            name = os.path.basename(p)
            f.write(f'**{name}** - {CAPTIONS.get(name, "(no caption)")}\n\n')
    print(f'[figures] wrote {FIGURES}/captions.md')


def write_summary(fig_paths):
    lines = ['# Results summary', '',
             '_Auto-generated by scripts/make_figures.py. Paste this file '
             '(plus the CSVs it inlines) back for paper writing._', '']
    for title, path in [('Main results (Exp A)', f'{RESULTS}/main_results.md'),
                        ('PE ablation (Exp B)', f'{RESULTS}/pe_ablation.md')]:
        if os.path.exists(path):
            with open(path) as f:
                lines += [f.read(), '']
    lines += ['## Figures', '']
    lines += [f'- `{p}` - {CAPTIONS.get(os.path.basename(p), "")}'
              for p in fig_paths]
    lines += ['', '## Raw CSVs', '']
    for csv in ('main_results.csv', 'pe_ablation.csv'):
        path = f'{RESULTS}/{csv}'
        if os.path.exists(path):
            with open(path) as f:
                lines += [f'### {csv}', '```csv', f.read().rstrip(), '```', '']
    for f_ in sorted(glob.glob(f'{RESULTS}/*_curves.csv')):
        lines += [f'- curves: `{f_}`']
    with open(f'{RESULTS}/summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[figures] wrote {RESULTS}/summary.md')


def main():
    main_df = (pd.read_csv(f'{RESULTS}/main_results.csv')
               if os.path.exists(f'{RESULTS}/main_results.csv') else None)
    abl_df = (pd.read_csv(f'{RESULTS}/pe_ablation.csv')
              if os.path.exists(f'{RESULTS}/pe_ablation.csv') else None)
    if main_df is None and abl_df is None:
        raise SystemExit('no CSVs in results/ - run eval first')

    write_tables(main_df, abl_df)
    fig_paths = []
    if abl_df is not None and len(abl_df):
        fig_paths.append(fig_pe_lines(abl_df))
    if main_df is not None and len(main_df):
        fig_paths.append(fig_efficiency(main_df))
        fig_paths += fig_qual_panels(main_df)
    c = fig_curves()
    if c:
        fig_paths.append(c)
    write_captions(fig_paths)
    write_summary(fig_paths)


if __name__ == '__main__':
    main()

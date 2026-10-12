"""
make_paper_figures.py
=====================
Figuras del artículo a partir de results/E*_metrics.csv:

  Fig. 5  Resumen de la validación sintética (6 paneles, un solo eje Y por panel)
  Fig. S1 Barrido de parámetros sobre el conjunto de ajuste (tuning/)

Estilo: trazos finos, marcadores con anillo blanco, rejilla tenue, una sola
escala por panel, etiquetas directas cuando hay pocas series y paleta
categórica validada (azul, naranja, aqua) apta para daltonismo.

Uso:
    python make_paper_figures.py [--results_dir results] [--tuning_dir tuning] [--out_dir figures]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.frameon': False, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'mathtext.default': 'regular',
})

LINE = dict(lw=1.4, ms=4.5, mec='white', mew=0.9, solid_capstyle='round')


def _panel(ax, letter, title=None):
    ax.text(-0.17, 1.06, f'({letter})', transform=ax.transAxes, fontsize=9,
            fontweight='bold', color=INK, va='bottom')
    if title:
        ax.set_title(title, loc='left', color=INK, pad=4)


def _logx(ax, ticks):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def _label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords='offset points',
                color=INK2, fontsize=7.5, va='center')


def fig5(res, out_dir):
    e1, e3, e4, e5 = (res[k] for k in ('E1', 'E3', 'E4', 'E5'))
    gauss = e3[e3['noise_type'] == 'gaussian'].sort_values('noise_std')
    sp = e3[e3['noise_type'] == 'salt_pepper'].sort_values('sp_fraction')

    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.5))
    fig.subplots_adjust(wspace=0.42, hspace=0.62)
    ticks = [3, 5, 10, 20, 50, 80]

    # (a) error relativo del diámetro frente al tamaño
    ax = axs[0, 0]
    ax.plot(e1['diameter_px'], 100 * e1['MRE_diam'], 'o-', color=BLUE, label='E1, no noise', **LINE)
    ax.plot(e4['diameter_px'], 100 * e4['MRE_diam'], 's-', color=ORANGE, label='E4, σ = 8', **LINE)
    _logx(ax, ticks)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.axhline(10, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(80, 10, '10 %', color=INK2, fontsize=7, ha='right', va='bottom')
    ax.set_xlabel('Prescribed diameter, $d_{GT}$ (px)')
    ax.set_ylabel('$MRE_d$ (%)')
    ax.legend(loc='upper right', handlelength=1.6)
    _panel(ax, 'a', 'Relative sizing error')

    # (b) sesgo absoluto con signo
    ax = axs[0, 1]
    ax.axhline(0, color=INK2, lw=0.6)
    ax.plot(e1['diameter_px'], e1['bias_diam'], 'o-', color=BLUE, label='E1, no noise', **LINE)
    ax.plot(e4['diameter_px'], e4['bias_diam'], 's-', color=ORANGE, label='E4, σ = 8', **LINE)
    _logx(ax, ticks)
    ax.set_xlabel('Prescribed diameter, $d_{GT}$ (px)')
    ax.set_ylabel('Mean bias, $d_{det} - d_{GT}$ (px)')
    ax.legend(loc='best', handlelength=1.6)
    _panel(ax, 'b', 'Absolute sizing bias')

    # (c) detección frente al tamaño (E4)
    ax = axs[0, 2]
    ax.plot(e4['diameter_px'], e4['recall'], 's-', color=ORANGE, label='Recall', **LINE)
    ax.plot(e4['diameter_px'], e4['precision'], 'o-', color=BLUE, label='Precision', **LINE)
    _logx(ax, ticks)
    ax.set_ylim(min(0.0, e4['recall'].min()) - 0.03, 1.04)
    ax.set_xlabel('Prescribed diameter, $d_{GT}$ (px)')
    ax.set_ylabel('Detection rate')
    ax.legend(loc='lower right', handlelength=1.6)
    _panel(ax, 'c', 'Detection limit (E4)')

    # (d) ruido gaussiano (E3)
    ax = axs[1, 0]
    ax.plot(gauss['noise_std'], gauss['recall'], 's-', color=ORANGE, label='Recall', **LINE)
    ax.plot(gauss['noise_std'], gauss['precision'], 'o-', color=BLUE, label='Precision', **LINE)
    ax.set_ylim(min(0.75, gauss[['recall', 'precision']].min().min() - 0.03), 1.03)
    ax.set_xlabel('Gaussian noise, σ (grey levels)')
    ax.set_ylabel('Detection rate')
    ax.legend(loc='lower left', handlelength=1.6)
    _panel(ax, 'd', 'Gaussian noise (E3)')

    # (e) sal y pimienta (E3)
    ax = axs[1, 1]
    ax.plot(100 * sp['sp_fraction'], sp['recall'], 's-', color=ORANGE, label='Recall', **LINE)
    ax.plot(100 * sp['sp_fraction'], sp['precision'], 'o-', color=BLUE, label='Precision', **LINE)
    ax.set_ylim(min(0.75, sp[['recall', 'precision']].min().min() - 0.03), 1.03)
    ax.set_xlabel('Salt-and-pepper noise (% of pixels)')
    ax.set_ylabel('Detection rate')
    ax.legend(loc='lower left', handlelength=1.6)
    _panel(ax, 'e', 'Impulse noise (E3)')

    # (f) densidad (E5): fragmentos por gota real
    ax = axs[1, 2]
    ax.axhline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.plot(e5['n_drops'], e5['fragments_per_gt'], 'o-', color=BLUE, **LINE)
    ax.set_xticks(e5['n_drops'])
    ax.set_ylim(0.85, max(1.5, e5['fragments_per_gt'].max() * 1.1))
    ax.text(e5['n_drops'].iloc[-1], 0.98, 'ideal (one track per droplet)', color=INK2,
            fontsize=6.8, ha='right', va='top')
    ax.set_xlabel('Number of droplets, $N_d$')
    ax.set_ylabel('Tracks per ground-truth droplet')
    _panel(ax, 'f', 'Tracking under density (E5)')

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'Fig5_synthetic_validation.{ext}'))
    plt.close(fig)


def figS1(tuning_dir, out_dir):
    seg = pd.read_csv(os.path.join(tuning_dir, 'segmentation_sweep.csv'))
    flt = pd.read_csv(os.path.join(tuning_dir, 'filter_sweep.csv'))
    trk = pd.read_csv(os.path.join(tuning_dir, 'tracking_sweep.csv'))

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(wspace=0.38)

    ax = axs[0]
    clahe = seg['use_clahe'].astype(bool)
    ax.scatter(seg.loc[clahe, 'RMSE_diam_mean'], seg.loc[clahe, 'F1_mean'], s=14, color=BLUE,
               edgecolor='white', lw=0.4, label='Stage 1, CLAHE')
    ax.scatter(seg.loc[~clahe, 'RMSE_diam_mean'], seg.loc[~clahe, 'F1_mean'], s=14, facecolor='none',
               edgecolor=INK2, lw=0.6, label='Stage 1, percentile stretching')
    ax.scatter(flt['RMSE_diam_mean'], flt['F1_mean'], s=16, marker='D', color=AQUA,
               edgecolor='white', lw=0.4, label='Stage 2, contour filters')
    best = flt.iloc[0]
    ax.scatter([best['RMSE_diam_mean']], [best['F1_mean']], s=70, facecolor='none',
               edgecolor=ORANGE, lw=1.4, label='Selected')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('Mean $RMSE_d$ on tuning set (px)')
    ax.set_ylabel('Mean F1-score on tuning set')
    ax.legend(loc='upper right', handlelength=1.0, fontsize=6.8)
    _panel(ax, 'a', 'Segmentation and filtering')

    ax = axs[1]
    ax.axhline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.plot(trk['max_tracking_distance'], trk['fragments_per_gt'], 'o-', color=BLUE, **LINE)
    sel = trk.loc[trk['max_tracking_distance'] == 15]
    ax.plot(sel['max_tracking_distance'], sel['fragments_per_gt'], 'o', ms=9, mfc='none', mec=ORANGE, mew=1.4)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(trk['max_tracking_distance']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('Maximum association distance (px)')
    ax.set_ylabel('Tracks per ground-truth droplet')
    _panel(ax, 'b', 'Association distance')

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'FigS1_parameter_sweep.{ext}'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--tuning_dir', default='tuning')
    ap.add_argument('--out_dir', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    res = {e: pd.read_csv(os.path.join(args.results_dir, f'{e}_metrics.csv')) for e in ('E1', 'E2', 'E3', 'E4', 'E5')}
    fig5(res, args.out_dir)
    if os.path.exists(os.path.join(args.tuning_dir, 'tracking_sweep.csv')):
        figS1(args.tuning_dir, args.out_dir)
    print('Figuras guardadas en', os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()

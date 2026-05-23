"""
plot_experiment.py
==================
Genera las gráficas de resultados para UN experimento a partir de su
CSV de métricas (generado por run_experiments.py).
Cada gráfica se guardará como un archivo PNG independiente.

Uso:
    python plot_experiment.py E1 --metrics_csv results/E1_metrics.csv
    python plot_experiment.py E2 --metrics_csv results/E2_metrics.csv
    python plot_experiment.py E3 --metrics_csv results/E3_metrics.csv
    python plot_experiment.py E4 --metrics_csv results/E4_metrics.csv
    python plot_experiment.py E5 --metrics_csv results/E5_metrics.csv

Argumentos opcionales:
    --out_dir   Directorio de salida para las imágenes PNG (default: results/)
    --dpi       Resolución de las imágenes (default: 150)
    --show      Mostrar las figuras en pantalla además de guardarlas
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Estilo global
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'blue':   '#3498db',
    'red':    '#e74c3c',
    'green':  '#2ecc71',
    'purple': '#9b59b6',
    'orange': '#e67e22',
    'gray':   '#95a5a6',
}

def _style_ax(ax, xlabel='', ylabel='', title='', ylim=None):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if ylim is not None:
        ax.set_ylim(*ylim)

def _save_and_show(out_path, dpi, show):
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"✓ Guardado → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# E1 – Precisión de segmentación
# ─────────────────────────────────────────────────────────────────────────────

def plot_E1(df: pd.DataFrame, base_out_path: str, dpi: int = 150, show: bool = False):
    x = df['diameter_px'].astype(float)

    # (a) MRE diámetro
    fig, ax = plt.subplots(figsize=(6, 4))
    mre = df['MRE_diam'].astype(float) * 100
    ax.plot(x, mre, 'o-', color=COLORS['blue'], linewidth=2, markersize=6)
    ax.axhline(5,  color='red',    linestyle='--', alpha=0.6, label='5 %')
    ax.axhline(10, color='orange', linestyle='--', alpha=0.6, label='10 %')
    ax.legend(fontsize=8)
    _style_ax(ax, xlabel='Diámetro GT (px)', ylabel='Error Relativo Medio (%)', title='E1 - Error en Diámetro')
    _save_and_show(f"{base_out_path}_mre_diam.png", dpi, show)

    # (b) RMSE diámetro
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, df['RMSE_diam'].astype(float), 's-', color=COLORS['red'], linewidth=2, markersize=6)
    _style_ax(ax, xlabel='Diámetro GT (px)', ylabel='RMSE (px)', title='E1 - RMSE del Diámetro')
    _save_and_show(f"{base_out_path}_rmse_diam.png", dpi, show)

    # (c) Circularidad y Redondez
    fig, ax = plt.subplots(figsize=(6, 4))
    if 'circ_mean' in df.columns:
        circ_mean = df['circ_mean'].astype(float)
        circ_std  = df['circ_std'].astype(float) if 'circ_std' in df.columns else pd.Series(np.zeros(len(df)))
        ax.errorbar(x, circ_mean, yerr=circ_std, fmt='D-', color=COLORS['green'], linewidth=2, markersize=6, capsize=4, label='Circularidad')
    
    if 'round_mean' in df.columns:
        round_mean = df['round_mean'].astype(float)
        round_std  = df['round_std'].astype(float) if 'round_std' in df.columns else pd.Series(np.zeros(len(df)))
        ax.errorbar(x, round_mean, yerr=round_std, fmt='o-', color=COLORS['orange'], linewidth=2, markersize=6, capsize=4, label='Redondez')

    if 'circ_mean' in df.columns or 'round_mean' in df.columns:
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal (1.0)')
        ax.legend(fontsize=8)
        
    _style_ax(ax, xlabel='Diámetro GT (px)', ylabel='Valor', title='E1 - Circularidad y Redondez', ylim=(0, 1.1))
    _save_and_show(f"{base_out_path}_shape_metrics.png", dpi, show)


# ─────────────────────────────────────────────────────────────────────────────
# E2 – Precisión de velocidad
# ─────────────────────────────────────────────────────────────────────────────

def plot_E2(df: pd.DataFrame, base_out_path: str, dpi: int = 150, show: bool = False):
    labels = df['tag'].str.replace('E2_', '', regex=False)
    x = np.arange(len(df))

    # (a) MRE velocidad
    fig, ax = plt.subplots(figsize=(7, 4))
    mre = df['MRE_vel'].astype(float) * 100
    bars = ax.bar(x, mre, color=COLORS['purple'], edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    ax.axhline(5,  color='red',    linestyle='--', alpha=0.6, label='5 %')
    ax.axhline(10, color='orange', linestyle='--', alpha=0.6, label='10 %')
    ax.legend(fontsize=8)
    _style_ax(ax, ylabel='Error Relativo Medio (%)', title='E2 - MRE de la Velocidad')
    _save_and_show(f"{base_out_path}_mre_vel.png", dpi, show)

    # (b) Velocidad detectada vs GT
    if 'gt_vel_px_frame' in df.columns and 'det_vel_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        gt_vel  = df['gt_vel_px_frame'].astype(float)
        det_vel = df['det_vel_mean'].astype(float)
        det_std = df['det_vel_std'].astype(float) if 'det_vel_std' in df.columns else pd.Series(np.zeros(len(df)))
        
        ax.errorbar(gt_vel, det_vel, yerr=det_std, fmt='o', color=COLORS['orange'], linewidth=2, markersize=7, capsize=4, label='Detectada')
        lims = [min(gt_vel.min(), det_vel.min()) * 0.9, max(gt_vel.max(), det_vel.max()) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5, label='Ideal (y=x)')
        ax.legend(fontsize=8)
        _style_ax(ax, xlabel='Velocidad GT (px/frame)', ylabel='Velocidad detectada (px/frame)', title='E2 - GT vs. Detectada')
        _save_and_show(f"{base_out_path}_gt_vs_det.png", dpi, show)


# ─────────────────────────────────────────────────────────────────────────────
# E3 – Robustez al ruido
# ─────────────────────────────────────────────────────────────────────────────

def plot_E3(df: pd.DataFrame, base_out_path: str, dpi: int = 150, show: bool = False):
    gauss = df[df['tag'].str.contains('gauss')].copy()
    sp    = df[df['tag'].str.contains('_sp_')].copy()

    # ── Ruido gaussiano ──────────────────────────────────────────────────────
    if not gauss.empty and 'noise_std' in gauss.columns:
        xg = gauss['noise_std'].astype(float)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xg, gauss['F1'].astype(float), 'o-', color=COLORS['blue'], linewidth=2, markersize=6)
        ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='F1 = 0.9')
        ax.legend(fontsize=8)
        _style_ax(ax, xlabel='Desv. est. ruido gaussiano (σ)', ylabel='F1', title='E3 - F1 (Ruido Gaussiano)', ylim=(0, 1.05))
        _save_and_show(f"{base_out_path}_gauss_f1.png", dpi, show)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xg, gauss['MRE_diam'].astype(float) * 100, 's-', color=COLORS['red'], linewidth=2, markersize=6)
        _style_ax(ax, xlabel='Desv. est. ruido gaussiano (σ)', ylabel='MRE Diámetro (%)', title='E3 - Error Diámetro (Ruido Gaussiano)')
        _save_and_show(f"{base_out_path}_gauss_mre.png", dpi, show)

    # ── Sal y pimienta ───────────────────────────────────────────────────────
    if not sp.empty and 'sp_fraction' in sp.columns:
        xs = sp['sp_fraction'].astype(float) * 100   # → porcentaje

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, sp['F1'].astype(float), 'o-', color=COLORS['purple'], linewidth=2, markersize=6)
        ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='F1 = 0.9')
        ax.legend(fontsize=8)
        _style_ax(ax, xlabel='Fracción píxeles sal y pimienta (%)', ylabel='F1', title='E3 - F1 (Sal y Pimienta)', ylim=(0, 1.05))
        _save_and_show(f"{base_out_path}_sp_f1.png", dpi, show)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, sp['MRE_diam'].astype(float) * 100, 's-', color=COLORS['orange'], linewidth=2, markersize=6)
        _style_ax(ax, xlabel='Fracción píxeles sal y pimienta (%)', ylabel='MRE Diámetro (%)', title='E3 - Error Diámetro (Sal y Pimienta)')
        _save_and_show(f"{base_out_path}_sp_mre.png", dpi, show)


# ─────────────────────────────────────────────────────────────────────────────
# E4 – Rango de tamaños detectables
# ─────────────────────────────────────────────────────────────────────────────

def plot_E4(df: pd.DataFrame, base_out_path: str, dpi: int = 150, show: bool = False):
    x = df['diameter_px'].astype(float)

    # (a) Precision / Recall / F1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, df['recall'].astype(float),    'o-', color=COLORS['green'], linewidth=2, markersize=6, label='Recall')
    ax.plot(x, df['precision'].astype(float), 's-', color=COLORS['blue'], linewidth=2, markersize=6, label='Precision')
    ax.plot(x, df['F1'].astype(float),        'D-', color=COLORS['red'], linewidth=2, markersize=6, label='F1')
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.legend(fontsize=9)
    _style_ax(ax, xlabel='Diámetro GT (px)', ylabel='Métrica', title='E4 - Tasa de Detección por Tamaño', ylim=(0.75, 1.05))
    _save_and_show(f"{base_out_path}_detection_rates.png", dpi, show)

    # (b) MRE diámetro
    fig, ax = plt.subplots(figsize=(6, 4))
    mre = df['MRE_diam'].astype(float) * 100
    ax.plot(x, mre, 'o-', color=COLORS['purple'], linewidth=2, markersize=6)
    ax.axhline(5,  color='red',    linestyle='--', alpha=0.6, label='5 %')
    ax.axhline(10, color='orange', linestyle='--', alpha=0.6, label='10 %')
    ax.legend(fontsize=8)
    _style_ax(ax, xlabel='Diámetro GT (px)', ylabel='MRE Diámetro (%)', title='E4 - Error de Medición por Tamaño')
    _save_and_show(f"{base_out_path}_mre_diam.png", dpi, show)


# ─────────────────────────────────────────────────────────────────────────────
# E5 – Densidad de gotas
# ─────────────────────────────────────────────────────────────────────────────

def plot_E5(df: pd.DataFrame, base_out_path: str, dpi: int = 150, show: bool = False):
    x = df['n_drops'].astype(float)

    # (a) Recall
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, df['recall'].astype(float), 'o-', color=COLORS['green'], linewidth=2, markersize=7)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    _style_ax(ax, xlabel='Número de gotas simultáneas', ylabel='Recall', title='E5 - Recall vs. Densidad', ylim=(0, 1.05))
    _save_and_show(f"{base_out_path}_recall.png", dpi, show)

    # (b) F1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, df['F1'].astype(float), 's-', color=COLORS['blue'], linewidth=2, markersize=7)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    _style_ax(ax, xlabel='Número de gotas simultáneas', ylabel='F1', title='E5 - F1 vs. Densidad', ylim=(0, 1.05))
    _save_and_show(f"{base_out_path}_f1.png", dpi, show)

    # (c) Fragmentación
    if 'frag_rate' in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        frag = df['frag_rate'].astype(float)
        ax.plot(x, frag, 'D-', color=COLORS['red'], linewidth=2, markersize=7)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Sin fragmentación')
        ax.fill_between(x, 0, frag.clip(lower=0), alpha=0.15, color=COLORS['red'])
        ax.legend(fontsize=8)
        _style_ax(ax, xlabel='Número de gotas simultáneas', ylabel='Tasa de fragmentación', title='E5 - Fragmentación de Tracks')
        _save_and_show(f"{base_out_path}_frag_rate.png", dpi, show)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

PLOT_FUNCTIONS = {
    'E1': plot_E1,
    'E2': plot_E2,
    'E3': plot_E3,
    'E4': plot_E4,
    'E5': plot_E5,
}

REQUIRED_COLUMNS = {
    'E1': ['diameter_px', 'MRE_diam', 'RMSE_diam'],
    'E2': ['tag', 'MRE_vel'],
    'E3': ['tag'],
    'E4': ['diameter_px', 'recall', 'precision', 'F1', 'MRE_diam'],
    'E5': ['n_drops', 'recall', 'F1'],
}


def main():
    parser = argparse.ArgumentParser(
        description='Genera las gráficas de resultados de un experimento de evaluación.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python plot_experiment.py E1 --metrics_csv results/E1_metrics.csv
  python plot_experiment.py E4 --metrics_csv results/E4_metrics.csv --dpi 300
  python plot_experiment.py E3 --metrics_csv results/E3_metrics.csv --show
        """,
    )
    parser.add_argument('experiment',
                        choices=list(PLOT_FUNCTIONS.keys()),
                        help='Identificador del experimento (E1–E5).')
    parser.add_argument('--metrics_csv', required=True,
                        help='Ruta al CSV de métricas generado por run_experiments.py.')
    parser.add_argument('--out_dir', default=None,
                        help='Directorio de salida (default: mismo directorio que --metrics_csv).')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Resolución de las imágenes PNG (default: 150).')
    parser.add_argument('--show', action='store_true',
                        help='Mostrar las figuras en pantalla después de guardarlas.')
    args = parser.parse_args()

    # ── Leer CSV ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.metrics_csv):
        print(f"ERROR: no se encuentra el CSV: {args.metrics_csv}")
        sys.exit(1)

    df = pd.read_csv(args.metrics_csv)

    # Comprobar columnas mínimas necesarias
    missing = [c for c in REQUIRED_COLUMNS[args.experiment] if c not in df.columns]
    if missing:
        print(f"ERROR: al CSV le faltan las columnas: {missing}")
        print(f"  Columnas presentes: {list(df.columns)}")
        sys.exit(1)

    # ── Ruta de salida ────────────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.metrics_csv))
    os.makedirs(out_dir, exist_ok=True)
    
    # Creamos un prefijo para las imágenes (ej: results/E1)
    base_out_path = os.path.join(out_dir, args.experiment)

    # ── Graficar ──────────────────────────────────────────────────────────────
    PLOT_FUNCTIONS[args.experiment](df, base_out_path, dpi=args.dpi, show=args.show)


if __name__ == '__main__':
    main()
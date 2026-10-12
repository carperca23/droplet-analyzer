"""
run_experiments.py
==================
Orquesta el pipeline completo de evaluación:
  1. Genera los vídeos sintéticos (synthetic_video_generator.py)
  2. Ejecuta el analizador de gotas sobre cada vídeo
  3. Evalúa los resultados con el evaluador (evaluator.py)
  4. Genera tablas CSV y gráficos PNG de resumen

Uso:
    python run_experiments.py [--experiments E1 E2 ...] [--output_dir results]

NOTA: El script asume que droplet_analyzer.py está en la carpeta raíz
del proyecto (un nivel por encima de donde está este script, o en el
mismo directorio). Ajusta ANALYZER_DIR si fuera necesario.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')   # consola de Windows (cp1252)
import json
import shutil
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Rutas ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# Ajusta esta ruta a donde está droplet_analyzer.py
ANALYZER_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, ANALYZER_DIR)
sys.path.insert(0, SCRIPT_DIR)

from synthetic_video_generator import (
    experiment_E1_segmentation,
    experiment_E2_velocity,
    experiment_E3_noise,
    experiment_E4_size_range,
    experiment_E5_density,
)
from droplet_evaluator import DropletEvaluator
from experiment_config import params_for, export_parameter_table

try:
    from droplet_analyzer import DropletAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("⚠  droplet_analyzer.py no encontrado. Se generarán los vídeos y el")
    print("   framework de evaluación, pero el análisis deberá lanzarse manualmente.")


def run_analyzer(video_path: str, output_dir: str, csv_name: str,
                 params: dict):
    """
    Ejecuta DropletAnalyzer sobre video_path con `params` y devuelve
    (ruta al CSV generado, resumen de tiempos por etapa).
    Si el analizador no está disponible, devuelve (None, None).
    """
    if not ANALYZER_AVAILABLE:
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    analyzer = DropletAnalyzer(video_path, output_dir)

    for k, v in params.items():
        analyzer.set_parameter(k, v)

    analyzer.set_scale(None)       # trabajamos en píxeles
    analyzer.set_fps(1.0)          # 1 frame/s → velocidad en px/frame

    analyzer.process_video()
    # Cabeceras en inglés: el evaluador busca diameter_*, track_id, velocity_*
    _, csv_path = analyzer.export_results(filename=csv_name, lang='en')
    return csv_path, analyzer.get_timing_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline por experimento
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_experiment(synth_dir: str, results_dir: str,
                         tag: str, params: dict) -> dict | None:
    """
    Dado el tag de un sub-experimento, busca su GT CSV, lanza el analizador
    y devuelve las métricas de evaluación.
    """
    gt_csv   = os.path.join(synth_dir, f"{tag}_gt.csv")
    video    = os.path.join(synth_dir, f"{tag}.avi")
    csv_name = f"{tag}_result"
    out_dir  = os.path.join(results_dir, tag)

    if not os.path.exists(video):
        print(f"  Vídeo no encontrado: {video}")
        return None

    csv_path, timing = run_analyzer(video, out_dir, csv_name, params)

    if csv_path is None or not os.path.exists(csv_path):
        print(f"   No se pudo analizar {tag}.")
        return None

    ev = DropletEvaluator(gt_csv, csv_path, match_dist=15.0)
    if ev.det.empty:
        print(f"   {tag}: sin detecciones (CSV resultado vacío). Posible gota demasiado pequeña.")
        metrics = {'tag': tag, 'TP': 0, 'FP': 0, 'FN': len(ev.gt),
                   'precision': 0.0, 'recall': 0.0, 'F1': 0.0,
                   'MRE_diam': None, 'RMSE_diam': None}
    else:
        metrics = ev.evaluate()
    metrics['tag'] = tag
    metrics['fps_processing'] = round(timing['fps'], 2)
    metrics['ms_per_frame'] = round(1000 * timing['total_s'] / max(timing['frames'], 1), 2)
    for stage, t in timing['stages'].items():
        metrics[f'pct_{stage}'] = round(t['percent'], 2)

    meta_path = os.path.join(synth_dir, f"{tag}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        # solo los campos escalares (la configuración completa queda en el JSON)
        metrics.update({k: v for k, v in meta.items() if not isinstance(v, (dict, list))})

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de plotting por experimento
# ─────────────────────────────────────────────────────────────────────────────

def plot_E1(df: pd.DataFrame, out_path: str):
    """E1: Error de diámetro y circularidad vs. diámetro GT."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('E1 - Precisión de Segmentación (Diámetros Conocidos)', fontweight='bold')

    x = df['diameter_px']

    axes[0].plot(x, df['MRE_diam'] * 100, 'o-', color='#3498db', linewidth=2)
    axes[0].set_xlabel('Diámetro GT (px)')
    axes[0].set_ylabel('Error Relativo Medio (%)')
    axes[0].set_title('Error en Diámetro')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(5, color='red', linestyle='--', alpha=0.5, label='5% umbral')
    axes[0].legend()

    axes[1].plot(x, df['RMSE_diam'], 's-', color='#e74c3c', linewidth=2)
    axes[1].set_xlabel('Diámetro GT (px)')
    axes[1].set_ylabel('RMSE (px)')
    axes[1].set_title('RMSE del Diámetro')
    axes[1].grid(True, alpha=0.3)

    if 'circ_mean' in df.columns:
        axes[2].errorbar(x, df['circ_mean'], yerr=df.get('circ_std', 0),
                         fmt='D-', color='#2ecc71', linewidth=2, capsize=4)
        axes[2].set_xlabel('Diámetro GT (px)')
        axes[2].set_ylabel('Circularidad media')
        axes[2].set_title('Circularidad Detectada')
        axes[2].set_ylim(0, 1.05)
        axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='ideal')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Gráfico E1: {out_path}")


def plot_E2(df: pd.DataFrame, out_path: str):
    """E2: Error de velocidad."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('E2 - Precisión de Velocidad', fontweight='bold')

    labels = df['tag'].str.replace('E2_', '', regex=False)

    axes[0].bar(range(len(df)), df['MRE_vel'] * 100, color='#9b59b6')
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Error Relativo Medio (%)')
    axes[0].set_title('MRE Velocidad')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(range(len(df)), df['RMSE_vel'], color='#e67e22')
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('RMSE (px/frame)')
    axes[1].set_title('RMSE Velocidad')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Gráfico E2: {out_path}")


def plot_E3(df: pd.DataFrame, out_path: str):
    """E3: Degradación de métricas con el ruido."""
    gauss = df[df['tag'].str.contains('gauss')].copy()
    sp    = df[df['tag'].str.contains('_sp_')].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('E3 - Robustez al Ruido', fontweight='bold')

    # Gauss - F1
    if not gauss.empty and 'noise_std' in gauss.columns:
        axes[0, 0].plot(gauss['noise_std'], gauss['F1'], 'o-', color='#3498db', linewidth=2)
        axes[0, 0].set_xlabel('Ruido gaussiano σ')
        axes[0, 0].set_ylabel('F1')
        axes[0, 0].set_title('F1 vs. Ruido Gaussiano')
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(gauss['noise_std'], gauss['MRE_diam'] * 100,
                        's-', color='#e74c3c', linewidth=2)
        axes[0, 1].set_xlabel('Ruido gaussiano σ')
        axes[0, 1].set_ylabel('MRE Diámetro (%)')
        axes[0, 1].set_title('Error Diámetro vs. Ruido Gaussiano')
        axes[0, 1].grid(True, alpha=0.3)

    # Sal & pimienta - F1
    if not sp.empty and 'sp_fraction' in sp.columns:
        axes[1, 0].plot(sp['sp_fraction'] * 100, sp['F1'],
                        'o-', color='#9b59b6', linewidth=2)
        axes[1, 0].set_xlabel('Ruido Sal & Pimienta (%)')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].set_title('F1 vs. Ruido Sal & Pimienta')
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(sp['sp_fraction'] * 100, sp['MRE_diam'] * 100,
                        's-', color='#e67e22', linewidth=2)
        axes[1, 1].set_xlabel('Ruido Sal & Pimienta (%)')
        axes[1, 1].set_ylabel('MRE Diámetro (%)')
        axes[1, 1].set_title('Error Diámetro vs. Sal & Pimienta')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Gráfico E3: {out_path}")


def plot_E4(df: pd.DataFrame, out_path: str):
    """E4: Rango de tamaños detectables."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('E4 - Rango de Tamaños Detectables', fontweight='bold')

    x = df['diameter_px']

    axes[0].plot(x, df['recall'], 'o-', color='#2ecc71', linewidth=2, label='Recall')
    axes[0].plot(x, df['precision'], 's-', color='#3498db', linewidth=2, label='Precision')
    axes[0].plot(x, df['F1'], 'D-', color='#e74c3c', linewidth=2, label='F1')
    axes[0].set_xlabel('Diámetro GT (px)')
    axes[0].set_ylabel('Métrica')
    axes[0].set_title('Tasa de Detección por Tamaño')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, df['MRE_diam'] * 100, 'o-', color='#9b59b6', linewidth=2)
    axes[1].set_xlabel('Diámetro GT (px)')
    axes[1].set_ylabel('MRE Diámetro (%)')
    axes[1].set_title('Error de Medición por Tamaño')
    axes[1].axhline(10, color='red', linestyle='--', alpha=0.5, label='10%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Gráfico E4: {out_path}")


def plot_E5(df: pd.DataFrame, out_path: str):
    """E5: Densidad de gotas."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('E5 - Impacto de la Densidad de Gotas', fontweight='bold')

    x = df['n_drops']

    axes[0].plot(x, df['recall'], 'o-', color='#2ecc71', linewidth=2)
    axes[0].set_xlabel('N.º de gotas')
    axes[0].set_ylabel('Recall')
    axes[0].set_title('Recall vs. Densidad')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, df['F1'], 's-', color='#3498db', linewidth=2)
    axes[1].set_xlabel('N.º de gotas')
    axes[1].set_ylabel('F1')
    axes[1].set_title('F1 vs. Densidad')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    if 'frag_rate' in df.columns:
        axes[2].plot(x, df['frag_rate'], 'D-', color='#e74c3c', linewidth=2)
        axes[2].set_xlabel('N.º de gotas')
        axes[2].set_ylabel('Tasa de fragmentación')
        axes[2].set_title('Fragmentación de Tracks')
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Gráfico E5: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluación completa del analizador de gotas.')
    parser.add_argument('--experiments', nargs='+',
                        default=['E1', 'E2', 'E3', 'E4', 'E5'],
                        choices=['E1', 'E2', 'E3', 'E4', 'E5'])
    parser.add_argument('--synth_dir',   default='synthetic_dataset',
                        help='Directorio con vídeos sintéticos (se crearán si no existen).')
    parser.add_argument('--results_dir', default='results',
                        help='Directorio para resultados de análisis y métricas.')
    parser.add_argument('--skip_generation', action='store_true',
                        help='No regenerar los vídeos si ya existen.')
    args = parser.parse_args()

    os.makedirs(args.synth_dir,   exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Generar vídeos sintéticos
    if not args.skip_generation:
        print("\n═══ FASE 1: Generación de vídeos sintéticos ═══")
        if 'E1' in args.experiments:
            experiment_E1_segmentation(args.synth_dir)
        if 'E2' in args.experiments:
            experiment_E2_velocity(args.synth_dir)
        if 'E3' in args.experiments:
            experiment_E3_noise(args.synth_dir)
        if 'E4' in args.experiments:
            experiment_E4_size_range(args.synth_dir)
        if 'E5' in args.experiments:
            experiment_E5_density(args.synth_dir)

    # 2. Analizar y evaluar
    print("\n═══ FASE 2: Análisis y evaluación ═══")

    # Tags de cada experimento
    exp_tags = {
        'E1': [f"E1_diam{d:03d}px" for d in [6, 10, 15, 20, 30, 40, 50]],
        'E2': [],   # se infieren de los archivos presentes
        'E3': (
            [f"E3_gauss_std{s:03d}" for s in [0, 5, 10, 20, 35, 50]] +
            [f"E3_sp_{int(p*100):03d}pct" for p in [0.0, 0.02, 0.05, 0.10]]
        ),
        'E4': [f"E4_diam{d:03d}px" for d in [3,4,5,6,8,10,12,15,20,25,30,40,50,60,70,80]],
        'E5': [f"E5_n{n:03d}drops" for n in [5, 10, 20, 35, 50]],
    }

    # E2: inferir tags de ficheros existentes
    if 'E2' in args.experiments:
        exp_tags['E2'] = sorted(
            f.replace('_gt.csv', '')
            for f in os.listdir(args.synth_dir)
            if f.startswith('E2_') and f.endswith('_gt.csv')
        )

    print(f"  Tabla S1 de parámetros: {export_parameter_table(args.results_dir)}")
    all_results = {}

    for exp in args.experiments:
        print(f"\n── {exp} ──")
        metrics_list = []

        # Parámetros: única fuente en experiment_config.py (Tabla S1)
        params = params_for(exp)

        for tag in exp_tags.get(exp, []):
            print(f"  Procesando {tag}...")
            m = evaluate_experiment(args.synth_dir, args.results_dir, tag, params)
            if m:
                metrics_list.append(m)

        if not metrics_list:
            print(f"  (sin resultados para {exp})")
            continue

        df_exp = pd.DataFrame(metrics_list)
        csv_out = os.path.join(args.results_dir, f"{exp}_metrics.csv")
        df_exp.to_csv(csv_out, index=False)
        print(f"  Tabla métricas: {csv_out}")
        all_results[exp] = df_exp

    # 3. Gráficos
    print("\n═══ FASE 3: Gráficos de resultados ═══")
    plot_fns = {'E1': plot_E1, 'E2': plot_E2, 'E3': plot_E3,
                'E4': plot_E4, 'E5': plot_E5}

    for exp, df_exp in all_results.items():
        png_path = os.path.join(args.results_dir, f"{exp}_results.png")
        try:
            plot_fns[exp](df_exp, png_path)
        except Exception as e:
            print(f"  Error al graficar {exp}: {e}")

    print("\nEvaluación completa. Resultados en:", os.path.abspath(args.results_dir))


if __name__ == '__main__':
    main()

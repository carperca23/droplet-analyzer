"""
tune_parameters.py
==================
Barrido de parámetros del analizador sobre un conjunto de ajuste sintético
INDEPENDIENTE de E1-E5 (otros diámetros, otras semillas de ruido y otra
disposición de gotas), para elegir la configuración común de la validación.

Fase 1 (segmentación): rejilla de contraste, NL-Means, umbral adaptativo y
morfología sobre gotas estáticas de varios tamaños y niveles de ruido, con
los filtros de contorno fijos y permisivos.
Criterio: mayor F1 media sobre el conjunto de ajuste; en caso de empate
(diferencia < 0.005), menor RMSE medio del diámetro (el sesgo es casi
constante en px, por eso se usa el error absoluto) y después la más rápida.

Fase 1b (filtros): con las 3 mejores segmentaciones, barrido del área mínima
y la circularidad mínima, que controlan el compromiso entre falsos
positivos del ruido y detección de las gotas más pequeñas. Mismo criterio.

Fase 2 (tracking): con la segmentación elegida, barrido de la distancia
máxima de asociación sobre escenas con varias gotas en movimiento.
Criterio: mínimo de cambios de identidad + fragmentos por gota, con
recall >= 0.95 del máximo alcanzado.

Uso:
    python tune_parameters.py [--out_dir tuning] [--workers 8]
"""

import os
import sys
import json
import time
import argparse
import itertools
import multiprocessing as mp

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
sys.path.insert(0, SCRIPT_DIR)

from synthetic_video_generator import generate_video, _grid_positions, _seed
from droplet_evaluator import DropletEvaluator
from droplet_analyzer import DropletAnalyzer
from experiment_config import COMMON_PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# Conjunto de ajuste (disjunto de E1-E5)
# ─────────────────────────────────────────────────────────────────────────────

SEG_DIAMETERS = [5, 9, 14, 25, 45]
SEG_NOISE = [('gauss', 0), ('gauss', 10), ('gauss', 25), ('sp', 0.05)]
TUNING_LAYOUT_SEED = 2024


def build_tuning_set(out_dir, n_frames_seg=10, n_frames_trk=40, size=512):
    os.makedirs(out_dir, exist_ok=True)
    seg, trk = [], []

    def make(tag, droplets, n_frames, noise_std=0.0, sp=0.0):
        video = os.path.join(out_dir, f'{tag}.avi')
        gt = os.path.join(out_dir, f'{tag}_gt.csv')
        if not os.path.exists(video):
            generate_video(video, gt, droplets, n_frames=n_frames, width=size, height=size,
                           noise_std=noise_std, noise_salt_pepper=sp,
                           polarity='dark', seed=_seed(tag),
                           meta_path=os.path.join(out_dir, f'{tag}_meta.json'),
                           meta={'set': 'tuning', 'tag': tag})
        return tag

    for d in SEG_DIAMETERS:
        pos = _grid_positions(2, 2, size, size, margin=max(80, d + 20))
        drops = [{'id': i, 'x0': x, 'y0': y, 'vx': 0.0, 'vy': 0.0, 'diameter': float(d)}
                 for i, (x, y) in enumerate(pos)]
        for kind, level in SEG_NOISE:
            name = f'T_seg_d{d:03d}_{kind}{level:g}'.replace('.', 'p')
            seg.append(make(name, drops, n_frames_seg,
                            noise_std=level if kind == 'gauss' else 0.0,
                            sp=level if kind == 'sp' else 0.0))

    rng = np.random.default_rng(TUNING_LAYOUT_SEED)
    for n in (10, 25):
        drops = []
        for i in range(n):
            ang = rng.uniform(0, 2 * np.pi)
            drops.append({'id': i, 'x0': rng.uniform(60, size - 60), 'y0': rng.uniform(60, size - 60),
                          'vx': 3.0 * np.cos(ang), 'vy': 3.0 * np.sin(ang), 'diameter': 16.0})
        trk.append(make(f'T_trk_n{n:03d}', drops, n_frames_trk, noise_std=8.0))
    # Gota rápida aislada (6 px/frame en diagonal)
    trk.append(make('T_trk_fast', [{'id': 0, 'x0': 80.0, 'y0': 80.0, 'vx': 4.5, 'vy': 4.0,
                                    'diameter': 16.0}], n_frames_trk, noise_std=8.0))
    return seg, trk


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución de una configuración sobre un vídeo
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(job):
    cfg_id, params, video_dir, tag, work_dir = job
    import cv2
    cv2.setNumThreads(1)    # el paralelismo lo dan los procesos del pool
    video = os.path.join(video_dir, f'{tag}.avi')
    out_dir = os.path.join(work_dir, f'cfg{cfg_id:03d}')
    a = DropletAnalyzer(video, out_dir)
    for k, v in params.items():
        a.set_parameter(k, v)
    a.set_scale(None)
    a.set_fps(1.0)
    a.process_video()
    _, csv = a.export_results(filename=f'{tag}.csv', lang='en')
    m = DropletEvaluator(os.path.join(video_dir, f'{tag}_gt.csv'), csv).evaluate() \
        if not a.export_to_dataframe().empty else {'F1': 0.0, 'recall': 0.0, 'precision': 0.0}
    t = a.get_timing_summary()
    m.update({'cfg_id': cfg_id, 'tag': tag, 'ms_per_frame': 1000 * t['total_s'] / max(t['frames'], 1)})
    return m


def _parallel(jobs, workers):
    with mp.Pool(workers) as pool:
        return list(pool.imap_unordered(_run_one, jobs, chunksize=4))


# ─────────────────────────────────────────────────────────────────────────────
# Fase 1: segmentación
# ─────────────────────────────────────────────────────────────────────────────

SEG_GRID = {
    'use_clahe':              [True, False],
    'denoise_strength':       [5, 10, 20, 30],
    'adaptive_block_size':    [3, 11, 31],
    'adaptive_c':             [2, 5],
    'morphology_kernel_size': [1, 3],
}


def segmentation_sweep(video_dir, seg_tags, work_dir, workers):
    keys = list(SEG_GRID)
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*SEG_GRID.values())]
    jobs = []
    for i, c in enumerate(configs):
        params = dict(COMMON_PARAMS)
        params.update(c)
        params['max_tracking_distance'] = 1     # gotas estáticas
        jobs += [(i, params, video_dir, t, work_dir) for t in seg_tags]

    raw = pd.DataFrame(_parallel(jobs, workers))
    agg = raw.groupby('cfg_id').agg(
        F1_mean=('F1', 'mean'), F1_min=('F1', 'min'),
        MRE_diam_mean=('MRE_diam', 'mean'), RMSE_diam_mean=('RMSE_diam', 'mean'),
        bias_diam_mean=('bias_diam', 'mean'), ms_per_frame=('ms_per_frame', 'mean'),
    ).reset_index()
    agg = pd.concat([agg, pd.DataFrame([configs[i] for i in agg['cfg_id']])], axis=1)

    agg = _rank(agg)
    return raw, agg, [{k: agg.iloc[i][k] for k in keys} for i in range(3)]


def _rank(agg):
    """Ordena por F1 media (agrupada en pasos de 0.005), RMSE y tiempo."""
    agg = agg.assign(_f1=-(agg['F1_mean'] / 0.005).round())
    return agg.sort_values(['_f1', 'RMSE_diam_mean', 'ms_per_frame']).drop(columns='_f1')


# ─────────────────────────────────────────────────────────────────────────────
# Fase 1b: filtros de contorno
# ─────────────────────────────────────────────────────────────────────────────

FILTER_GRID = {
    'min_droplet_area':        [3, 8, 15, 25],
    'min_droplet_circularity': [0.5, 0.6, 0.7],
}


def filter_sweep(video_dir, seg_tags, work_dir, workers, seg_candidates):
    keys = list(FILTER_GRID)
    configs = []
    for seg in seg_candidates:
        for vals in itertools.product(*FILTER_GRID.values()):
            configs.append({**seg, **dict(zip(keys, vals))})
    jobs = []
    for i, c in enumerate(configs):
        params = dict(COMMON_PARAMS)
        params.update(c)
        params['max_tracking_distance'] = 1
        jobs += [(500 + i, params, video_dir, t, work_dir) for t in seg_tags]

    raw = pd.DataFrame(_parallel(jobs, workers))
    agg = raw.groupby('cfg_id').agg(
        F1_mean=('F1', 'mean'), F1_min=('F1', 'min'),
        MRE_diam_mean=('MRE_diam', 'mean'), RMSE_diam_mean=('RMSE_diam', 'mean'),
        bias_diam_mean=('bias_diam', 'mean'), ms_per_frame=('ms_per_frame', 'mean'),
    ).reset_index()
    agg = pd.concat([agg, pd.DataFrame([configs[i - 500] for i in agg['cfg_id']])], axis=1)
    agg = _rank(agg)
    return raw, agg, {k: agg.iloc[0][k] for k in configs[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Fase 2: tracking
# ─────────────────────────────────────────────────────────────────────────────

TRACK_DISTANCES = [4, 6, 8, 10, 15, 20, 30, 60]


def tracking_sweep(video_dir, trk_tags, work_dir, workers, seg_params):
    jobs = []
    for i, dist in enumerate(TRACK_DISTANCES):
        params = dict(COMMON_PARAMS)
        params.update(seg_params)
        params['max_tracking_distance'] = dist
        jobs += [(1000 + i, params, video_dir, t, work_dir) for t in trk_tags]

    raw = pd.DataFrame(_parallel(jobs, workers))
    raw['max_tracking_distance'] = raw['cfg_id'].map(lambda c: TRACK_DISTANCES[c - 1000])
    agg = raw.groupby('max_tracking_distance').agg(
        recall=('recall', 'mean'), id_switches=('id_switches', 'sum'),
        fragments_per_gt=('fragments_per_gt', 'mean'), MRE_vel=('MRE_vel', 'mean'),
    ).reset_index()
    ok = agg[agg['recall'] >= 0.95 * agg['recall'].max()]
    best = ok.assign(score=ok['id_switches'] + ok['fragments_per_gt']) \
             .sort_values(['score', 'MRE_vel']).iloc[0]
    return raw, agg, int(best['max_tracking_distance'])


# ─────────────────────────────────────────────────────────────────────────────

def _native(d):
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default='tuning')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    video_dir = os.path.join(args.out_dir, 'videos')
    work_dir = os.path.join(args.out_dir, 'runs')
    t0 = time.time()
    seg_tags, trk_tags = build_tuning_set(video_dir)
    print(f'Conjunto de ajuste: {len(seg_tags)} vídeos de segmentación, {len(trk_tags)} de tracking')

    raw, agg, top3 = segmentation_sweep(video_dir, seg_tags, work_dir, args.workers)
    raw.to_csv(os.path.join(args.out_dir, 'segmentation_sweep_raw.csv'), index=False)
    agg.to_csv(os.path.join(args.out_dir, 'segmentation_sweep.csv'), index=False)
    print(f'Fase 1 ({time.time() - t0:.0f} s). Mejores segmentaciones: {top3}')
    print(agg.head(12).to_string(index=False))

    raw_f, agg_f, best_seg = filter_sweep(video_dir, seg_tags, work_dir, args.workers,
                                          [_native(c) for c in top3])
    raw_f.to_csv(os.path.join(args.out_dir, 'filter_sweep_raw.csv'), index=False)
    agg_f.to_csv(os.path.join(args.out_dir, 'filter_sweep.csv'), index=False)
    print(f'\nFase 1b ({time.time() - t0:.0f} s). Mejor segmentación + filtros: {best_seg}')
    print(agg_f.head(12).to_string(index=False))

    sys.stdout.flush()
    raw_t, agg_t, best_dist = tracking_sweep(video_dir, trk_tags, work_dir, args.workers, _native(best_seg))
    raw_t.to_csv(os.path.join(args.out_dir, 'tracking_sweep_raw.csv'), index=False)
    agg_t.to_csv(os.path.join(args.out_dir, 'tracking_sweep.csv'), index=False)
    print(f'\nFase 2 ({time.time() - t0:.0f} s). Mejor distancia de asociación: {best_dist} px')
    print(agg_t.to_string(index=False))

    selected = {**_native(best_seg), 'max_tracking_distance': best_dist}
    with open(os.path.join(args.out_dir, 'selected_parameters.json'), 'w') as f:
        json.dump(selected, f, indent=2)
    print(f'\nSeleccionado: {selected}')


if __name__ == '__main__':
    main()

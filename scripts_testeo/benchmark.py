"""
benchmark.py
============
Mide la velocidad de procesado del analizador (frames/s en CPU) y el reparto
del tiempo entre etapas, sobre uno o varios vídeos.

Uso:
    python benchmark.py video1.avi [video2.avi ...] [--repeats 3] [--out benchmark.csv]

Se usan los parámetros comunes de experiment_config.py. Cada vídeo se procesa
`repeats` veces y se informa de la mediana. OpenCV usa su paralelismo interno
por defecto (como en la aplicación); --threads 1 fuerza un solo hilo.
"""

import os
import sys
import argparse
import platform

import cv2
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
sys.path.insert(0, SCRIPT_DIR)

from droplet_analyzer import DropletAnalyzer, TIMING_STAGES
from experiment_config import COMMON_PARAMS


def cpu_name():
    if platform.system() == 'Windows':
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor()


def benchmark_video(path, repeats, out_dir, params):
    rows = []
    for _ in range(repeats):
        a = DropletAnalyzer(path, out_dir)
        for k, v in params.items():
            a.set_parameter(k, v)
        a.process_video()
        t = a.get_timing_summary()
        row = {'video': os.path.basename(path), 'width': a.width, 'height': a.height,
               'frames': t['frames'], 'fps': t['fps'],
               'ms_per_frame': 1000 * t['total_s'] / max(t['frames'], 1),
               'detections': sum(len(tr['frames']) for tr in a.droplet_tracks)}
        for s in TIMING_STAGES:
            row[f'ms_{s}'] = t['stages'][s]['ms_per_frame']
            row[f'pct_{s}'] = t['stages'][s]['percent']
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.groupby(['video', 'width', 'height', 'frames']).median(numeric_only=True).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('videos', nargs='+')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--out', default='benchmark.csv')
    args = ap.parse_args()

    if args.threads:
        cv2.setNumThreads(args.threads)
    print(f'CPU: {cpu_name()} | OpenCV {cv2.__version__} | hilos OpenCV: {cv2.getNumThreads()}')

    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)) or '.', 'benchmark_tmp')
    res = pd.concat([benchmark_video(v, args.repeats, out_dir, COMMON_PARAMS) for v in args.videos])
    res['cpu'] = cpu_name()
    res['opencv'] = cv2.__version__
    res['opencv_threads'] = cv2.getNumThreads()
    res.to_csv(args.out, index=False)

    cols = ['video', 'width', 'height', 'frames', 'fps', 'ms_per_frame'] + [f'pct_{s}' for s in TIMING_STAGES]
    print(res[cols].round(2).to_string(index=False))
    print(f'\nGuardado en {args.out}')


if __name__ == '__main__':
    main()

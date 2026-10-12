"""
droplet_evaluator.py
====================
Compara la salida del analizador de gotas (CSV resultado) con el ground
truth generado por synthetic_video_generator.py y calcula métricas de
evaluación cuantitativas.

Emparejamiento
--------------
En cada frame, las detecciones se emparejan con las gotas GT por distancia
euclídea entre centroides, de forma greedy (primero el par más cercano) y
sin reasignación, con una distancia máxima `match_dist` (15 px por defecto).
Todas las métricas se calculan sobre este único emparejamiento.

Métricas calculadas
-------------------
Detección (micro-promedio: TP, FP y FN se acumulan sobre todos los frames
y después se calculan precision, recall y F1):
  - precision  : TP / (TP + FP)
  - recall     : TP / (TP + FN)
  - F1         : 2PR / (P + R)
  - F1_frame_mean : media de la F1 calculada frame a frame (macro-promedio),
                    solo como referencia; las tablas usan la F1 micro.

Segmentación / medición (sobre las detecciones emparejadas):
  - MRE_diam   : Error Relativo Medio en diámetro, mean(|d_det - d_gt| / d_gt)
  - RMSE_diam  : RMSE del diámetro detectado vs. GT (px)
  - bias_diam  : sesgo medio con signo, mean(d_det - d_gt) (px)
  - MRE_area   : Error Relativo Medio en área
  - circ_mean  : Circularidad media detectada (≈ 1.0 en gotas sintéticas)
  - round_mean : Redondez media detectada (≈ 1.0 en gotas sintéticas)

Velocidad (sobre las detecciones emparejadas que tienen velocidad, es decir,
todas las de cada track salvo la última; la velocidad de la fila del frame f
es el desplazamiento hasta la siguiente detección del mismo track):
  - gt_vel_px_frame : magnitud media de la velocidad GT de esas detecciones
  - det_vel_mean/std: magnitud de la velocidad detectada (px/frame)
  - MRE_vel    : Error Relativo Medio de la magnitud de velocidad
  - RMSE_vel   : RMSE de la magnitud de velocidad (px/frame)

Tracking:
  - frag_rate  : (N_tracks_det - N_tracks_GT) / N_tracks_GT. Cuenta también
                 los tracks de falsos positivos.
  - fragments_per_gt : nº medio de tracks detectados distintos asignados a
                 cada gota GT (1 = sin fragmentación)
  - id_switches: nº de veces que cambia el track detectado asignado a una
                 misma gota GT entre detecciones emparejadas consecutivas

La velocidad detectada se lee de la columna velocity_magnitude_* del CSV del
analizador. run_experiments.py fija fps = 1 y escala = None, de modo que
esa columna está en px/frame, las mismas unidades que el GT.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')   # consola de Windows (cp1252)
import pandas as pd
import numpy as np
import os
import json
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _match_detections_to_gt(gt_pos: np.ndarray, det_pos: np.ndarray,
                            max_dist: float = 15.0):
    """
    Empareja detecciones con GT por distancia euclídea mínima (greedy, sin
    reasignación). Devuelve lista de tuplas (idx_gt, idx_det).
    """
    if len(gt_pos) == 0 or len(det_pos) == 0:
        return []

    D = cdist(gt_pos, det_pos)
    matches = []
    for _ in range(min(len(gt_pos), len(det_pos))):
        ig, id_ = np.unravel_index(np.argmin(D), D.shape)
        if D[ig, id_] > max_dist:
            break
        matches.append((ig, id_))
        D[ig, :] = np.inf
        D[:, id_] = np.inf
    return matches


def _get_col(df, prefix):
    """Devuelve el nombre de la primera columna que empieza por `prefix`."""
    return next((c for c in df.columns if c.startswith(prefix)), None)


def _f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluador principal
# ─────────────────────────────────────────────────────────────────────────────

class DropletEvaluator:
    """
    Parameters
    ----------
    gt_csv   : ruta al CSV de ground truth generado por synthetic_video_generator
    det_csv  : ruta al CSV de salida del analizador de gotas
    match_dist: radio máximo (px) para emparejar detección ↔ GT
    """

    def __init__(self, gt_csv: str, det_csv: str, match_dist: float = 15.0):
        self.gt = pd.read_csv(gt_csv)
        try:
            self.det = pd.read_csv(det_csv)
            if self.det.empty or len(self.det.columns) == 0:
                self.det = pd.DataFrame()
        except Exception:
            self.det = pd.DataFrame()
        self.match_dist = match_dist

        # Detectar columnas de la salida del analizador
        self._x_col = _get_col(self.det, 'x_') if not self.det.empty else None
        self._y_col = _get_col(self.det, 'y_') if not self.det.empty else None
        self._d_col = _get_col(self.det, 'diameter_') if not self.det.empty else None
        self._a_col = _get_col(self.det, 'area_') if not self.det.empty else None
        self._v_col = _get_col(self.det, 'velocity_magnitude_') if not self.det.empty else None

        # Normalizar nombres para uso interno
        if self._x_col and not self.det.empty:
            self.det['x_px'] = self.det[self._x_col]
            self.det['y_px'] = self.det[self._y_col]
            self.det['diameter_px'] = self.det[self._d_col]

        self._matched = None

    # ── Emparejamiento (se calcula una vez) ──────────────────────────────────

    def _match_all(self):
        """
        Empareja todos los frames. Devuelve (matched, counts):
          matched : DataFrame con una fila por par GT↔detección
          counts  : DataFrame con TP, FP, FN por frame
        """
        if self._matched is not None:
            return self._matched

        rows, counts = [], []
        has_det = 'frame' in self.det.columns
        for f in sorted(self.gt['frame'].unique()):
            gt_f  = self.gt[self.gt['frame'] == f]
            det_f = self.det[self.det['frame'] == f] if has_det else pd.DataFrame()

            matches = _match_detections_to_gt(
                gt_f[['x_px', 'y_px']].values,
                det_f[['x_px', 'y_px']].values if not det_f.empty else np.empty((0, 2)),
                self.match_dist,
            )
            counts.append({'frame': f, 'TP': len(matches),
                           'FP': len(det_f) - len(matches),
                           'FN': len(gt_f) - len(matches)})

            for ig, id_ in matches:
                g, d = gt_f.iloc[ig], det_f.iloc[id_]
                rows.append({
                    'frame': f,
                    'gt_track': g['track_id'],
                    'det_track': d['track_id'] if 'track_id' in det_f.columns else None,
                    'd_gt': g['diameter_px'],
                    'd_det': d['diameter_px'],
                    'a_det': d[self._a_col] if self._a_col else np.nan,
                    'circ': d['circularity'] if 'circularity' in det_f.columns else np.nan,
                    'round': d['roundness'] if 'roundness' in det_f.columns else np.nan,
                    'v_gt': np.hypot(g['vx_px_frame'], g['vy_px_frame']),
                    'v_det': d[self._v_col] if self._v_col else np.nan,
                })

        self._matched = (pd.DataFrame(rows), pd.DataFrame(counts))
        return self._matched

    # ── Métricas de detección y segmentación ─────────────────────────────────

    def segmentation_metrics(self) -> dict:
        m, c = self._match_all()
        tp, fp, fn = int(c['TP'].sum()), int(c['FP'].sum()), int(c['FN'].sum())
        precision, recall, f1 = _f1(tp, fp, fn)
        f1_frames = [_f1(r.TP, r.FP, r.FN)[2] for r in c.itertuples() if r.TP + r.FP + r.FN > 0]

        out = {
            'TP': tp, 'FP': fp, 'FN': fn,
            'precision': _r(precision), 'recall': _r(recall), 'F1': _r(f1),
            'F1_frame_mean': _r(np.mean(f1_frames)) if f1_frames else None,
            'MRE_diam': None, 'RMSE_diam': None, 'bias_diam': None, 'MRE_area': None,
            'circ_mean': None, 'circ_std': None, 'round_mean': None, 'round_std': None,
        }
        if m.empty:
            return out

        err = m['d_det'] - m['d_gt']
        a_gt = np.pi * (m['d_gt'] / 2) ** 2
        out.update({
            'MRE_diam':   _r((err.abs() / m['d_gt']).mean()),
            'RMSE_diam':  _r(np.sqrt((err ** 2).mean())),
            'bias_diam':  _r(err.mean()),
            'MRE_area':   _r(((m['a_det'] - a_gt).abs() / a_gt).mean()),
            'circ_mean':  _r(m['circ'].mean()),
            'circ_std':   _r(m['circ'].std(ddof=0)),
            'round_mean': _r(m['round'].mean()),
            'round_std':  _r(m['round'].std(ddof=0)),
        })
        return out

    # ── Métricas de velocidad ────────────────────────────────────────────────

    def velocity_metrics(self) -> dict:
        """
        Compara, para cada detección emparejada que tenga velocidad, la
        magnitud detectada con la de la gota GT con la que se emparejó.
        """
        empty = {'gt_vel_px_frame': None, 'det_vel_mean': None, 'det_vel_std': None,
                 'MRE_vel': None, 'RMSE_vel': None}
        if self._v_col is None:
            return {**empty, 'note': 'no velocity column in output'}
        if (self.gt['vx_px_frame'].abs() + self.gt['vy_px_frame'].abs()).max() == 0:
            return {**empty, 'note': 'static droplets, no velocity GT'}

        m, _ = self._match_all()
        m = m.dropna(subset=['v_det']) if not m.empty else m
        m = m[m['v_gt'] > 0] if not m.empty else m
        if m.empty:
            return {**empty, 'note': 'no matched velocity detections'}

        err = m['v_det'] - m['v_gt']
        return {
            'gt_vel_px_frame': _r(m['v_gt'].mean(), 3),
            'det_vel_mean':    _r(m['v_det'].mean(), 3),
            'det_vel_std':     _r(m['v_det'].std(), 3),
            'MRE_vel':         _r((err.abs() / m['v_gt']).mean()),
            'RMSE_vel':        _r(np.sqrt((err ** 2).mean())),
        }

    # ── Métricas de tracking ─────────────────────────────────────────────────

    def tracking_metrics(self) -> dict:
        n_gt_tracks  = self.gt['track_id'].nunique()
        n_det_tracks = self.det['track_id'].nunique() if 'track_id' in self.det.columns else None

        frag_rate = None
        if n_det_tracks is not None and n_gt_tracks > 0:
            frag_rate = _r((n_det_tracks - n_gt_tracks) / n_gt_tracks)

        fragments, switches = None, None
        m, _ = self._match_all()
        if not m.empty and m['det_track'].notna().all():
            per_gt = m.sort_values('frame').groupby('gt_track')['det_track']
            fragments = _r(per_gt.nunique().mean())
            switches = int(per_gt.apply(lambda s: int((s.values[1:] != s.values[:-1]).sum())).sum())

        return {
            'n_gt_tracks':  n_gt_tracks,
            'n_det_tracks': n_det_tracks,
            'frag_rate':    frag_rate,   # >0 → exceso de tracks (fragmentación o FP)
            'fragments_per_gt': fragments,
            'id_switches':  switches,
        }

    # ── Informe completo ─────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        results = {}
        results.update(self.segmentation_metrics())
        results.update(self.velocity_metrics())
        results.update(self.tracking_metrics())
        return results

    def print_report(self, title: str = ""):
        r = self.evaluate()
        print(f"\n{'─'*50}")
        if title:
            print(f"  {title}")
            print(f"{'─'*50}")
        for k, v in r.items():
            print(f"  {k:<20} {v}")
        print(f"{'─'*50}")
        return r


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Evaluador de resultados del analizador de gotas.')
    p.add_argument('gt_csv',  help='CSV de ground truth')
    p.add_argument('det_csv', help='CSV de salida del analizador')
    p.add_argument('--match_dist', type=float, default=15.0,
                   help='Distancia máxima (px) para emparejar GT ↔ detección.')
    p.add_argument('--out_json', default=None, help='Guardar métricas en JSON.')
    args = p.parse_args()

    ev = DropletEvaluator(args.gt_csv, args.det_csv, args.match_dist)
    r  = ev.print_report(title=f"{os.path.basename(args.gt_csv)}")

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(r, f, indent=2)
        print(f"\n  Métricas guardadas en: {args.out_json}")

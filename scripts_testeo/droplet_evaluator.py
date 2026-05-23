"""
evaluator.py
============
Compara la salida del analizador de gotas (CSV resultado) con el ground
truth generado por synthetic_video_generator.py y calcula métricas de
evaluación cuantitativas.

Métricas calculadas
-------------------
Segmentación / medición:
  - MRE_diam   : Error Relativo Medio en diámetro  (|d_det - d_gt| / d_gt)
  - RMSE_diam  : RMSE del diámetro detectado vs. GT
  - MRE_area   : Error Relativo Medio en área
  - circ_mean  : Circularidad media detectada (debería ser ≈ 1.0 en gotas sintéticas)
  - round_mean : Redondez media detectada (debería ser ≈ 1.0 en gotas sintéticas)

Detección:
  - recall     : gotas GT detectadas / gotas GT totales (por frame)
  - precision  : gotas GT de las detectadas / total detectadas
  - F1         : media armónica de precision y recall

Tracking:
  - frag_rate  : tasa de fragmentación de tracks (tracks_det / tracks_gt - 1)
  - id_switches: número de cambios de ID estimado

Velocidad:
  - MRE_vel    : Error Relativo Medio en magnitud de velocidad
  - RMSE_vel   : RMSE de la magnitud de velocidad
"""

import pandas as pd
import numpy as np
import os
import json
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _match_detections_to_gt(gt_frame: pd.DataFrame, det_frame: pd.DataFrame,
                             max_dist: float = 15.0):
    """
    Empareja detecciones con GT por distancia euclídea mínima (greedy).
    Devuelve lista de tuplas (idx_gt, idx_det).
    """
    if gt_frame.empty or det_frame.empty:
        return []

    gt_pos  = gt_frame[['x_px', 'y_px']].values
    det_pos = det_frame[['x_px', 'y_px']].values if 'x_px' in det_frame.columns else None

    if det_pos is None:
        # Intentar columnas con unidad
        x_col = next((c for c in det_frame.columns if c.startswith('x_')), None)
        y_col = next((c for c in det_frame.columns if c.startswith('y_')), None)
        if x_col is None:
            return []
        det_pos = det_frame[[x_col, y_col]].values

    D = cdist(gt_pos, det_pos)
    matches = []
    used_det = set()
    used_gt  = set()

    for _ in range(min(len(gt_frame), len(det_frame))):
        if D.size == 0:
            break
        idx = np.unravel_index(np.argmin(D), D.shape)
        ig, id_ = idx
        if D[ig, id_] > max_dist:
            break
        if ig not in used_gt and id_ not in used_det:
            matches.append((ig, id_))
            used_gt.add(ig)
            used_det.add(id_)
            D[ig, :] = np.inf
            D[:, id_] = np.inf

    return matches


def _get_col(df, prefix):
    """Devuelve el nombre de la primera columna que empieza por `prefix`."""
    return next((c for c in df.columns if c.startswith(prefix)), None)


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
        self.gt  = pd.read_csv(gt_csv)
        try:
            self.det = pd.read_csv(det_csv)
            if self.det.empty or len(self.det.columns) == 0:
                self.det = pd.DataFrame()
        except Exception:
            self.det = pd.DataFrame()
        self.match_dist = match_dist

        # Detectar columnas de la salida del analizador
        self._x_col   = _get_col(self.det, 'x_')   if not self.det.empty else None
        self._y_col   = _get_col(self.det, 'y_')   if not self.det.empty else None
        self._d_col   = _get_col(self.det, 'diameter_') if not self.det.empty else None
        self._a_col   = _get_col(self.det, 'area_')    if not self.det.empty else None
        self._v_col   = _get_col(self.det, 'velocity_magnitude_') if not self.det.empty else None

        # Normalizar nombres para uso interno
        if self._x_col and not self.det.empty:
            self.det['x_px'] = self.det[self._x_col]
            self.det['y_px'] = self.det[self._y_col]
            self.det['diameter_px'] = self.det[self._d_col]

    # ── Métricas de segmentación ─────────────────────────────────────────────

    def segmentation_metrics(self) -> dict:
        """
        Por cada frame, empareja GT ↔ detecciones y acumula errores de
        diámetro, área, circularidad y redondez.
        """
        errors_diam_rel = [] # Para el MRE (porcentaje)
        errors_diam_px  = [] # Para el RMSE (píxeles absolutos)
        errors_area = []
        circularities = []
        roundnesses = []
        tp = fp = fn = 0

        frames = sorted(self.gt['frame'].unique())

        for f in frames:
            gt_f  = self.gt[self.gt['frame'] == f].reset_index(drop=True)
            det_f = self.det[self.det['frame'] == f].reset_index(drop=True) \
                    if 'frame' in self.det.columns else pd.DataFrame()

            matches = _match_detections_to_gt(gt_f, det_f, self.match_dist)
            tp += len(matches)
            fn += len(gt_f) - len(matches)
            fp += len(det_f) - len(matches)

            for ig, id_ in matches:
                d_gt  = gt_f.loc[ig, 'diameter_px']
                d_det = det_f.loc[id_, 'diameter_px']
                
                # Guardamos ambos errores por separado
                errors_diam_rel.append(abs(d_det - d_gt) / d_gt)
                errors_diam_px.append(abs(d_det - d_gt))

                if self._a_col:
                    a_gt  = np.pi * (d_gt / 2) ** 2
                    a_det = det_f.loc[id_, self._a_col]
                    errors_area.append(abs(a_det - a_gt) / a_gt)

                if 'circularity' in det_f.columns:
                    circularities.append(det_f.loc[id_, 'circularity'])
                
                if 'roundness' in det_f.columns:
                    roundnesses.append(det_f.loc[id_, 'roundness'])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) \
                    if (precision + recall) > 0 else 0.0

        return {
            'TP': tp, 'FP': fp, 'FN': fn,
            'precision':  round(precision, 4),
            'recall':     round(recall, 4),
            'F1':         round(f1, 4),
            'MRE_diam':   round(float(np.mean(errors_diam_rel)),  4) if errors_diam_rel else None,
            'RMSE_diam':  round(float(np.sqrt(np.mean(np.array(errors_diam_px)**2))), 4)
                          if errors_diam_px else None,
            'MRE_area':   round(float(np.mean(errors_area)),  4) if errors_area else None,
            'circ_mean':  round(float(np.mean(circularities)), 4) if circularities else None,
            'circ_std':   round(float(np.std(circularities)),  4) if circularities else None,
            'round_mean': round(float(np.mean(roundnesses)), 4) if roundnesses else None,
            'round_std':  round(float(np.std(roundnesses)), 4) if roundnesses else None,
        }

    # ── Métricas de velocidad ────────────────────────────────────────────────

    def velocity_metrics(self) -> dict:
        """
        Compara la magnitud de velocidad detectada con la del GT.
        Solo tiene sentido cuando hay movimiento (vx o vy != 0 en el GT).
        """
        if self._v_col is None:
            return {'MRE_vel': None, 'RMSE_vel': None, 'note': 'no velocity column in output'}

        gt_with_vel = self.gt[(self.gt['vx_px_frame'] != 0) | (self.gt['vy_px_frame'] != 0)].copy()
        if gt_with_vel.empty:
            return {'MRE_vel': None, 'RMSE_vel': None, 'note': 'static droplets, no velocity GT'}

        gt_mag = np.sqrt(gt_with_vel['vx_px_frame']**2 + gt_with_vel['vy_px_frame']**2).iloc[0]

        # El analizador calcula velocidad en px/s; convertimos a px/frame usando FPS=25
        # Aceptamos cualquier columna de velocidad_magnitude
        vel_vals = self.det[self._v_col].dropna()
        if vel_vals.empty:
            return {'MRE_vel': None, 'RMSE_vel': None, 'note': 'no velocity detections'}

        errors = np.abs(vel_vals - gt_mag) / gt_mag
        return {
            'gt_vel_px_frame': round(gt_mag, 3),
            'det_vel_mean':    round(float(vel_vals.mean()), 3),
            'det_vel_std':     round(float(vel_vals.std()),  3),
            'MRE_vel':         round(float(errors.mean()), 4),
            'RMSE_vel':        round(float(np.sqrt((errors**2).mean())), 4),
        }

    # ── Métricas de tracking ─────────────────────────────────────────────────

    def tracking_metrics(self) -> dict:
        """
        Cuenta fragmentación de tracks: si una gota GT (1 track) se mapea a
        varios tracks detectados, hay fragmentación.
        """
        n_gt_tracks  = self.gt['track_id'].nunique()
        n_det_tracks = self.det['track_id'].nunique() \
                       if 'track_id' in self.det.columns else None

        frag_rate = None
        if n_det_tracks is not None and n_gt_tracks > 0:
            frag_rate = round((n_det_tracks - n_gt_tracks) / n_gt_tracks, 4)

        return {
            'n_gt_tracks':  n_gt_tracks,
            'n_det_tracks': n_det_tracks,
            'frag_rate':    frag_rate,   # >0 → exceso de tracks (fragmentación)
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
    import argparse, json

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
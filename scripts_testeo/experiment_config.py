"""
experiment_config.py
====================
Única fuente de los parámetros del analizador usados en la validación
sintética (E1-E5). run_experiments.py lee de aquí la configuración y
export_parameter_table() genera la Tabla S1 del artículo, de modo que los
valores publicados coinciden siempre con los ejecutados.

Los valores de COMMON_PARAMS se eligieron con tune_parameters.py sobre un
conjunto de ajuste independiente de E1-E5 (otros diámetros, otras semillas
de ruido y otra disposición de gotas).
"""

import os
import pandas as pd


# Parámetros comunes a E1-E5: resultado de tune_parameters.py
# (tuning/selected_parameters.json)
COMMON_PARAMS = {
    # Preprocesado
    'use_clahe':               True,    # True = CLAHE, False = estiramiento por percentiles (0,4 %)
    'clahe_clip_limit':        2.0,
    'denoise_strength':        30,      # h de cv2.fastNlMeansDenoising
    # Segmentación
    'adaptive_block_size':     11,
    'adaptive_c':              5,
    'morphology_kernel_size':  3,
    # Filtrado de contornos
    'min_droplet_area':        15,
    'max_droplet_area':        8000,
    'min_droplet_circularity': 0.5,
    'min_droplet_roundness':   0.0,
    # Tracking
    'max_tracking_distance':   15,
    'use_lucas_kanade':        True,
    'lk_window_size':          15,
}

# Diferencias por experimento respecto a COMMON_PARAMS (vacío = ninguna)
EXPERIMENT_OVERRIDES = {
    'E1': {},
    'E2': {},
    'E3': {},
    'E4': {},
    'E5': {},
}

# Valores fijos en el código del analizador (no configurables desde la GUI)
FIXED_SETTINGS = [
    ('Preprocessing', 'Grayscale conversion', 'cv2.cvtColor (COLOR_BGR2GRAY)', '-'),
    ('Preprocessing', 'CLAHE tile grid size', 'cv2.createCLAHE tileGridSize', '8 × 8'),
    ('Preprocessing', 'Percentile stretching limits', 'np.percentile', '0.4 % / 99.6 %'),
    ('Preprocessing', 'NL-Means template / search window', 'cv2.fastNlMeansDenoising', '7 px / 21 px (OpenCV defaults)'),
    ('Segmentation', 'Adaptive method / polarity', 'cv2.adaptiveThreshold',
     'ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV'),
    ('Segmentation', 'Structuring element shape', 'cv2.getStructuringElement', 'MORPH_ELLIPSE'),
    ('Segmentation', 'Morphological sequence', 'cv2.morphologyEx', 'opening, then closing (same kernel)'),
    ('Contours', 'Pass 1 (hole filling)', 'cv2.findContours',
     'RETR_EXTERNAL, CHAIN_APPROX_SIMPLE; contours with A > min area filled'),
    ('Contours', 'Pass 2 (measurement)', 'cv2.findContours', 'RETR_EXTERNAL, CHAIN_APPROX_NONE'),
    ('Features', 'Circularity', '4πA/P² (cv2.contourArea, cv2.arcLength)', 'clipped to ≤ 1'),
    ('Features', 'Roundness', '4A/(π·MA²) (cv2.fitEllipse)', 'requires ≥ 5 contour points; clipped to ≤ 1'),
    ('Features', 'Centroid', 'cv2.moments', 'bounding-box centre if M00 = 0'),
    ('Tracking', 'Lucas–Kanade pyramid levels', 'cv2.calcOpticalFlowPyrLK maxLevel', '2'),
    ('Tracking', 'Lucas–Kanade stop criteria', 'cv2.calcOpticalFlowPyrLK criteria', '10 iterations or ε = 0.03'),
    ('Tracking', 'Maximum gap before a track is closed', '-', '5 frames'),
    ('Tracking', 'Association', '-', 'greedy nearest neighbour, one detection per track and frame'),
]

# Descripción de cada parámetro configurable para la Tabla S1
PARAM_INFO = {
    'use_clahe':               ('Preprocessing', 'Contrast enhancement method', '-'),
    'clahe_clip_limit':        ('Preprocessing', 'CLAHE clip limit', 'cv2.createCLAHE clipLimit'),
    'denoise_strength':        ('Preprocessing', 'Non-Local Means filter strength h', 'cv2.fastNlMeansDenoising h'),
    'adaptive_block_size':     ('Segmentation', 'Adaptive threshold block size (px)', 'cv2.adaptiveThreshold blockSize'),
    'adaptive_c':              ('Segmentation', 'Adaptive threshold constant C', 'cv2.adaptiveThreshold C'),
    'morphology_kernel_size':  ('Segmentation', 'Elliptical structuring element size (px)', 'cv2.getStructuringElement ksize'),
    'min_droplet_area':        ('Filtering', 'Minimum projected area (px²)', '-'),
    'max_droplet_area':        ('Filtering', 'Maximum projected area (px²)', '-'),
    'min_droplet_circularity': ('Filtering', 'Minimum circularity', '-'),
    'min_droplet_roundness':   ('Filtering', 'Minimum roundness', '-'),
    'max_tracking_distance':   ('Tracking', 'Maximum association distance (px)', '-'),
    'use_lucas_kanade':        ('Tracking', 'Lucas–Kanade position prediction', '-'),
    'lk_window_size':          ('Tracking', 'Lucas–Kanade window size (px)', 'cv2.calcOpticalFlowPyrLK winSize'),
}


def params_for(experiment: str) -> dict:
    """Parámetros completos del analizador para un experimento (E1-E5)."""
    p = dict(COMMON_PARAMS)
    p.update(EXPERIMENT_OVERRIDES.get(experiment, {}))
    return p


def _fmt(key, value):
    if key == 'use_clahe':
        return 'CLAHE' if value else 'Percentile stretching'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    return f'{value:g}' if isinstance(value, float) else str(value)


def parameter_table() -> pd.DataFrame:
    """Tabla S1: parámetros configurables (por experimento) y valores fijos."""
    rows = []
    exps = list(EXPERIMENT_OVERRIDES)
    for key, (stage, desc, opencv) in PARAM_INFO.items():
        values = {e: _fmt(key, params_for(e)[key]) for e in exps}
        same = len(set(values.values())) == 1
        row = {'Stage': stage, 'Parameter': desc, 'OpenCV function / argument': opencv,
               'Value (E1–E5)': values[exps[0]] if same else
               '; '.join(f'{e}: {v}' for e, v in values.items()),
               'User-configurable': 'Yes'}
        if key == 'clahe_clip_limit' and not params_for(exps[0])['use_clahe']:
            row['Value (E1–E5)'] += ' (not used)'
        rows.append(row)
    for stage, desc, opencv, value in FIXED_SETTINGS:
        rows.append({'Stage': stage, 'Parameter': desc, 'OpenCV function / argument': opencv,
                     'Value (E1–E5)': value, 'User-configurable': 'No'})
    return pd.DataFrame(rows)


def export_parameter_table(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'TableS1_parameters.csv')
    parameter_table().to_csv(path, index=False, encoding='utf-8-sig')
    return path

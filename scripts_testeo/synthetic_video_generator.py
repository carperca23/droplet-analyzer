"""
synthetic_video_generator.py
============================
Genera vídeos sintéticos de gotas con ground truth conocido para la
evaluación cuantitativa del analizador de gotas.

Por defecto (polarity='dark') cada gota es un disco oscuro con borde
gaussiano suavizado sobre fondo claro, igual que las sombras de las gotas
en la retroiluminación real y como espera el umbral invertido del
analizador. Internamente se dibuja un disco claro sobre fondo oscuro
(fondo 20, gota 230) y se invierte el frame antes de añadir ruido, de modo
que el resultado es su negativo exacto (fondo 235, gota 25).
polarity='bright' reproduce los vídeos de la versión anterior.

Posición, diámetro y velocidad de cada gota son exactos y se exporta un
CSV de ground truth con columnas:
    frame, track_id, x_px, y_px, diameter_px, vx_px_frame, vy_px_frame

Reproducibilidad: el ruido de cada vídeo usa un generador aleatorio con
semilla derivada de su etiqueta, los vídeos se guardan sin pérdidas
(FFV1) y el JSON de metadatos recoge todos los parámetros de generación.

Experimentos disponibles
------------------------
E1  - Precisión de segmentación: diámetros fijos, sin movimiento, sin ruido.
E2  - Precisión de velocidad  : diámetro fijo, velocidades conocidas.
E3  - Robustez al ruido       : diámetro y velocidad fijos, ruido creciente.
E4  - Rango de detección      : barrido de diámetros (límites del analizador).
E5  - Densidad de gotas       : número creciente de gotas simultáneas.
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
import zlib
import itertools


CODEC = 'FFV1'   # códec sin pérdidas: evita artefactos de compresión en el GT
E5_LAYOUT_SEED = 72


def _seed(tag: str) -> int:
    """Semilla estable para el ruido de cada vídeo, derivada de su etiqueta."""
    return zlib.crc32(tag.encode('utf-8'))


# ─────────────────────────────────────────────────────────────────────────────
# Primitiva de dibujo
# ─────────────────────────────────────────────────────────────────────────────

def draw_droplet(frame: np.ndarray, cx: float, cy: float, radius: float,
                 intensity: int = 255, blur_sigma: float = 0.8):
    """
    Dibuja una gota (disco blanco con borde gaussiano) sobre `frame`.
    
    Parameters
    ----------
    frame       : imagen en escala de grises (uint8), modificada in-place
    cx, cy      : centro de la gota (pueden ser flotantes)
    radius      : radio en píxeles
    intensity   : intensidad máxima de la gota (0-255)
    blur_sigma  : suavizado de borde (sigma del gaussiano); 0 = disco duro
    """
    h, w = frame.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    ir = int(np.ceil(radius + 3 * blur_sigma + 1))

    # Ventana local para no recalcular toda la imagen
    x0 = max(0, ix - ir)
    x1 = min(w, ix + ir + 1)
    y0 = max(0, iy - ir)
    y1 = min(h, iy + ir + 1)

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if blur_sigma > 0:
        # Perfil de intensidad: 1 dentro, caída gaussiana en el borde
        mask = np.clip(
            np.exp(-0.5 * ((dist - radius) / blur_sigma) ** 2)
            * (dist <= radius + 3 * blur_sigma),
            0, 1
        )
        mask[dist <= radius] = 1.0
    else:
        mask = (dist <= radius).astype(np.float32)

    patch = frame[y0:y1, x0:x1].astype(np.float32)
    patch = np.maximum(patch, mask * intensity)
    frame[y0:y1, x0:x1] = patch.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Generador principal
# ─────────────────────────────────────────────────────────────────────────────

def generate_video(
    output_path: str,
    gt_csv_path: str,
    droplets: list,          # [{x0, y0, vx, vy, diameter, id}]
    n_frames: int = 60,
    width: int = 512,
    height: int = 512,
    fps: float = 25.0,
    noise_std: float = 0.0,  # desviación estándar del ruido gaussiano (0-255)
    noise_salt_pepper: float = 0.0,  # fracción de píxeles salt&pepper
    background: int = 20,    # nivel de gris del fondo
    blur_sigma: float = 0.8,
    intensity: int = 230,
    polarity: str = 'dark',  # 'dark' = gota oscura sobre fondo claro (retroiluminación)
    seed: int = None,        # semilla del ruido (None = no reproducible)
    meta_path: str = None,   # JSON con metadatos del experimento
    meta: dict = None,
):
    """
    Genera un vídeo AVI sintético (sin pérdidas) y su ground truth CSV.

    `background` e `intensity` se refieren al dibujo en polaridad clara;
    con polarity='dark' el frame se invierte antes de añadir el ruido.

    Returns
    -------
    gt_df : pd.DataFrame con el ground truth completo
    """
    if polarity not in ('dark', 'bright'):
        raise ValueError("polarity debe ser 'dark' o 'bright'")
    rng = np.random.default_rng(seed)

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    if not out.isOpened():
        raise RuntimeError(f"No se pudo abrir el VideoWriter con el códec {CODEC}")

    records = []

    for frame_idx in range(n_frames):
        canvas = np.full((height, width), background, dtype=np.uint8)

        for d in droplets:
            cx = d['x0'] + d['vx'] * frame_idx
            cy = d['y0'] + d['vy'] * frame_idx
            radius = d['diameter'] / 2.0

            # Ignorar gotas que salgan del frame
            if cx - radius < 0 or cx + radius > width:
                continue
            if cy - radius < 0 or cy + radius > height:
                continue

            draw_droplet(canvas, cx, cy, radius, intensity=intensity,
                         blur_sigma=blur_sigma)

            records.append({
                'frame':        frame_idx,
                'track_id':     d['id'],
                'x_px':         round(cx, 3),
                'y_px':         round(cy, 3),
                'diameter_px':  round(d['diameter'], 3),
                'vx_px_frame':  d['vx'],
                'vy_px_frame':  d['vy'],
            })

        if polarity == 'dark':
            canvas = 255 - canvas

        # Ruido gaussiano
        if noise_std > 0:
            noise = rng.normal(0, noise_std, canvas.shape).astype(np.float32)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Ruido sal y pimienta
        if noise_salt_pepper > 0:
            rnd = rng.random(canvas.shape)
            canvas[rnd < noise_salt_pepper / 2] = 0
            canvas[rnd > 1 - noise_salt_pepper / 2] = 255

        out.write(canvas)

    out.release()

    gt_df = pd.DataFrame(records)
    gt_df.to_csv(gt_csv_path, index=False)

    if meta_path:
        full_meta = dict(meta or {})
        full_meta['generator'] = {
            'width': width, 'height': height, 'n_frames': n_frames, 'fps': fps,
            'polarity': polarity,
            'background_level': 255 - background if polarity == 'dark' else background,
            'droplet_level': 255 - intensity if polarity == 'dark' else intensity,
            'edge_blur_sigma_px': blur_sigma,
            'noise_gaussian_std': noise_std,
            'noise_salt_pepper_fraction': noise_salt_pepper,
            'noise_seed': seed,
            'codec': CODEC,
            'droplets_initial': droplets,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(full_meta, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Vídeo  : {output_path}")
    print(f"  ✓ GT CSV : {gt_csv_path}  ({len(records)} filas)")
    return gt_df


# ─────────────────────────────────────────────────────────────────────────────
# Definición de los experimentos
# ─────────────────────────────────────────────────────────────────────────────

def _grid_positions(n_cols, n_rows, width, height, margin=60):
    """Devuelve posiciones en cuadrícula para colocar gotas estáticas."""
    xs = np.linspace(margin, width - margin, n_cols)
    ys = np.linspace(margin, height - margin, n_rows)
    positions = [(float(x), float(y)) for y in ys for x in xs]
    return positions


def experiment_E1_segmentation(output_dir: str, width=512, height=512, n_frames=30, polarity='dark'):
    """
    E1 - Precisión de segmentación
    Gotas estáticas de diámetros conocidos, sin ruido.
    Permite medir el error en diámetro, área y circularidad.
    """
    print("\n── E1: Precisión de segmentación (diámetros conocidos) ──")
    diameters = [6, 10, 15, 20, 30, 40, 50]

    for diam in diameters:
        tag = f"E1_diam{diam:03d}px"
        # 4 gotas por frame para estadística
        positions = _grid_positions(2, 2, width, height, margin=max(80, diam + 20))
        droplets = [
            {'id': i, 'x0': x, 'y0': y, 'vx': 0.0, 'vy': 0.0, 'diameter': float(diam)}
            for i, (x, y) in enumerate(positions)
        ]

        generate_video(
            output_path=os.path.join(output_dir, f"{tag}.avi"),
            gt_csv_path=os.path.join(output_dir, f"{tag}_gt.csv"),
            droplets=droplets,
            n_frames=n_frames,
            polarity=polarity, seed=_seed(tag),
            width=width, height=height,
            noise_std=0.0,
            meta_path=os.path.join(output_dir, f"{tag}_meta.json"),
            meta={'experiment': 'E1', 'diameter_px': diam, 'n_droplets': len(droplets)},
        )


def experiment_E2_velocity(output_dir: str, width=512, height=512, n_frames=60, polarity='dark'):
    """
    E2  Precisión de velocidad
    Gota de 20 px moviéndose a velocidades conocidas, sin ruido y con ruido
    gaussiano (σ = 8). Las velocidades enteras solo comprueban el
    procedimiento (la gota se traslada exactamente sobre la rejilla de
    píxeles); las subpíxel miden la precisión de localización del centroide,
    y las más lentas (0.4 y 0.2 px/frame) el régimen en el que el
    desplazamiento se acerca a la incertidumbre de localización.
    """
    print("\n── E2: Precisión de velocidad ──")
    # Velocidades en px/frame
    velocities = [
        (2.0, 0.0),    # enteras
        (5.0, 0.0),
        (0.0, 3.0),
        (3.0, 3.0),
        (-4.0, 2.0),
        (1.3, 0.7),    # subpíxel
        (4.25, -1.6),
        (0.4, 0.0),    # lentas
        (0.2, 0.0),
    ]
    noise_levels = [0.0, 8.0]
    diameter = 20.0
    margin = 80

    for (vx, vy), noise in itertools.product(velocities, noise_levels):
        tag = (f"E2_vx{vx:+.2f}_vy{vy:+.2f}".replace('+', 'p').replace('-', 'n').replace('.', 'd')
               + f"_s{int(noise):03d}")
        # La gota parte del borde opuesto al sentido de avance en cada eje,
        # de modo que la trayectoria completa queda dentro del frame
        x0 = margin if vx >= 0 else width - margin
        y0 = margin if vy >= 0 else height - margin
        assert abs(vx) * n_frames <= width - 2 * margin and abs(vy) * n_frames <= height - 2 * margin

        droplets = [{'id': 0, 'x0': float(x0), 'y0': float(y0), 'vx': vx, 'vy': vy, 'diameter': diameter}]

        generate_video(
            output_path=os.path.join(output_dir, f"{tag}.avi"),
            gt_csv_path=os.path.join(output_dir, f"{tag}_gt.csv"),
            droplets=droplets,
            n_frames=n_frames,
            polarity=polarity, seed=_seed(tag),
            width=width, height=height,
            noise_std=noise,
            meta_path=os.path.join(output_dir, f"{tag}_meta.json"),
            meta={'experiment': 'E2', 'vx_px_frame': vx, 'vy_px_frame': vy,
                  'noise_std': noise, 'diameter_px': diameter},
        )


def experiment_E3_noise(output_dir: str, width=512, height=512, n_frames=40, polarity='dark'):
    """
    E3 – Robustez al ruido
    Gota fija de 20 px con niveles crecientes de ruido gaussiano y sal&pimienta.
    """
    print("\n── E3: Robustez al ruido ──")
    diameter = 20.0
    droplets = [{'id': 0, 'x0': 256.0, 'y0': 256.0, 'vx': 0.0, 'vy': 0.0, 'diameter': diameter}]

    noise_levels_gauss = [0, 5, 10, 20, 35, 50]
    noise_levels_sp    = [0.00, 0.02, 0.05, 0.10]

    for noise_std in noise_levels_gauss:
        tag = f"E3_gauss_std{noise_std:03d}"
        generate_video(
            output_path=os.path.join(output_dir, f"{tag}.avi"),
            gt_csv_path=os.path.join(output_dir, f"{tag}_gt.csv"),
            droplets=droplets,
            n_frames=n_frames,
            polarity=polarity, seed=_seed(tag),
            width=width, height=height,
            noise_std=float(noise_std),
            meta_path=os.path.join(output_dir, f"{tag}_meta.json"),
            meta={'experiment': 'E3', 'noise_type': 'gaussian', 'noise_std': noise_std, 'diameter_px': diameter},
        )

    for sp in noise_levels_sp:
        tag = f"E3_sp_{int(sp*100):03d}pct"
        generate_video(
            output_path=os.path.join(output_dir, f"{tag}.avi"),
            gt_csv_path=os.path.join(output_dir, f"{tag}_gt.csv"),
            droplets=droplets,
            n_frames=n_frames,
            polarity=polarity, seed=_seed(tag),
            width=width, height=height,
            noise_std=0.0,
            noise_salt_pepper=sp,
            meta_path=os.path.join(output_dir, f"{tag}_meta.json"),
            meta={'experiment': 'E3', 'noise_type': 'salt_pepper', 'sp_fraction': sp, 'diameter_px': diameter},
        )


def experiment_E4_size_range(output_dir: str, width=512, height=512, n_frames=30, polarity='dark'):
    """
    E4 – Rango de tamaños detectables
    Barrido de diámetros: 3 → 80 px para encontrar límites.
    """
    print("\n── E4: Rango de detección (barrido de diámetros) ──")
    diameters = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80]

    for diam in diameters:
        tag = f"E4_diam{diam:03d}px"
        cx = width / 2
        cy = height / 2
        droplets = [{'id': 0, 'x0': cx, 'y0': cy, 'vx': 0.0, 'vy': 0.0, 'diameter': float(diam)}]

        generate_video(
            output_path=os.path.join(output_dir, f"{tag}.avi"),
            gt_csv_path=os.path.join(output_dir, f"{tag}_gt.csv"),
            droplets=droplets,
            n_frames=n_frames,
            polarity=polarity, seed=_seed(tag),
            width=width, height=height,
            noise_std=8.0,   # algo de ruido realista
            meta_path=os.path.join(output_dir, f"{tag}_meta.json"),
            meta={'experiment': 'E4', 'diameter_px': diam},
        )


def experiment_E5_density(output_dir: str, width=512, height=512, n_frames=60, polarity='dark'):
    """
    E5 – Densidad de gotas (robustez del tracker con oclusiones)
    Número creciente de gotas moviéndose aleatoriamente.
    """
    print("\n── E5: Densidad de gotas ──")
    rng = np.random.default_rng(E5_LAYOUT_SEED)   # disposición inicial reproducible
    density_levels = [5, 10, 20, 35, 50]
    diameter = 18.0
    speed = 3.0  # px/frame

    for n_drops in density_levels:
        tag = f"E5_n{n_drops:03d}drops"
        margin = 60
        droplets = []
        for i in range(n_drops):
            x0 = rng.uniform(margin, width - margin)
            y0 = rng.uniform(margin, height - margin)
            angle = rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            droplets.append({'id': i, 'x0': x0, 'y0': y0, 'vx': vx, 'vy': vy,
                             'diameter': diameter})

        generate_video(
            output_path=os.path.join(output_dir, f"{tag}.avi"),
            gt_csv_path=os.path.join(output_dir, f"{tag}_gt.csv"),
            droplets=droplets,
            n_frames=n_frames,
            polarity=polarity, seed=_seed(tag),
            width=width, height=height,
            noise_std=8.0,
            meta_path=os.path.join(output_dir, f"{tag}_meta.json"),
            meta={'experiment': 'E5', 'n_drops': n_drops, 'diameter_px': diameter, 'speed_px_frame': speed},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generador de vídeos sintéticos para evaluación.')
    parser.add_argument('--output_dir', default='synthetic_dataset',
                        help='Directorio donde se guardan los vídeos y CSVs de GT.')
    parser.add_argument('--experiments', nargs='+',
                        default=['E1', 'E2', 'E3', 'E4', 'E5'],
                        choices=['E1', 'E2', 'E3', 'E4', 'E5'],
                        help='Experimentos a generar.')
    parser.add_argument('--polarity', default='dark', choices=['dark', 'bright'],
                        help="'dark' = gotas oscuras sobre fondo claro (por defecto); "
                             "'bright' = polaridad de la versión anterior.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generando vídeos sintéticos en: {args.output_dir}")

    if 'E1' in args.experiments:
        experiment_E1_segmentation(args.output_dir, polarity=args.polarity)
    if 'E2' in args.experiments:
        experiment_E2_velocity(args.output_dir, polarity=args.polarity)
    if 'E3' in args.experiments:
        experiment_E3_noise(args.output_dir, polarity=args.polarity)
    if 'E4' in args.experiments:
        experiment_E4_size_range(args.output_dir, polarity=args.polarity)
    if 'E5' in args.experiments:
        experiment_E5_density(args.output_dir, polarity=args.polarity)

    print("\n✓ Todos los vídeos generados.")

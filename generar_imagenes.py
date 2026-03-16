"""
debug_pipeline.py
Genera una imagen por cada paso del algoritmo de análisis de gotas,
mostrando cómo se transforma el frame a lo largo del pipeline.
 
Uso:
    python debug_pipeline.py <ruta_video> [numero_frame] [directorio_salida]
 
Ejemplos:
    python debug_pipeline.py video.mp4
    python debug_pipeline.py video.mp4 42
    python debug_pipeline.py video.mp4 0 debug_frames/
"""
 
import cv2
import numpy as np
import os
import sys
 
 
# ── Parámetros por defecto (igual que DropletAnalyzer) ────────────────────────
DEFAULT_PARAMS = {
    'min_droplet_area': 10,
    'max_droplet_area': 5000,
    'min_droplet_circularity': 0.6,
    'clahe_clip_limit': 2.0,
    'denoise_strength': 10,
    'adaptive_block_size': 11,
    'adaptive_c': 2,
    'morphology_kernel_size': 2,
    'use_clahe': False,
}
 
 
def save_step(img, step_num, name, output_dir, label=None):
    """
    Guarda una imagen tal cual, sin ninguna anotación.
 
    Args:
        img:        imagen numpy (gris o BGR)
        step_num:   índice del paso (para ordenar archivos)
        name:       nombre descriptivo del paso (usado solo en el nombre de archivo)
        output_dir: directorio de salida
        label:      ignorado, mantenido por compatibilidad
    """
    filename = f"{step_num:02d}_{name.replace(' ', '_')}.png"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, img)
    print(f"  guardado: {filename}")
    return path
 
 
def run_pipeline_debug(video_path, frame_index=0, output_dir='debug_frames', params=None):
    """
    Extrae un frame del video y guarda una imagen por cada paso del pipeline.
 
    Args:
        video_path:  ruta al archivo de video
        frame_index: índice del frame a analizar (0 = primer frame)
        output_dir:  directorio donde guardar los pasos
        params:      dict de parámetros; si es None usa DEFAULT_PARAMS
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
 
    os.makedirs(output_dir, exist_ok=True)
    step = 0
 
    # ── 0. Leer frame ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
 
    if frame_index >= total:
        print(f"Advertencia: frame {frame_index} fuera de rango ({total} frames). Usando frame 0.")
        frame_index = 0
 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame_bgr = cap.read()
    cap.release()
 
    if not ret:
        raise RuntimeError(f"No se pudo leer el frame {frame_index} de '{video_path}'")
 
    print(f"\nVideo: {video_path}")
    print(f"Frame {frame_index}/{total}  |  {frame_bgr.shape[1]}x{frame_bgr.shape[0]} px  |  {fps:.1f} fps")
    print(f"Guardando pasos en: {output_dir}/\n")
 
    save_step(frame_bgr, step, "frame original (BGR)", output_dir,
              label=f"frame {frame_index}/{total}")
    step += 1
 
    # ── FASE 1: Preprocesamiento ───────────────────────────────────────────────
 
    # 1. Conversión a escala de grises
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    save_step(gray, step, "escala de grises", output_dir)
    step += 1
 
    # 2a. Mejora de contraste
    if params['use_clahe']:
        clh = cv2.createCLAHE(
            clipLimit=params['clahe_clip_limit'],
            tileGridSize=(8, 8)
        )
        enhanced = clh.apply(gray)
        save_step(enhanced, step, "contraste CLAHE", output_dir,
                  label=f"clipLimit={params['clahe_clip_limit']}")
    else:
        flattened  = gray.flatten()
        low_val, high_val = np.percentile(flattened, [0.4, 99.6])
        dst      = np.clip(gray, low_val, high_val)
        enhanced = ((dst - low_val) / (high_val - low_val) * 255).astype(np.uint8)
        save_step(enhanced, step, "contraste por percentil", output_dir,
                  label=f"p0.4={low_val:.0f}  p99.6={high_val:.0f}")
    step += 1
 
    # 2b. Reducción de ruido
    denoised = cv2.fastNlMeansDenoising(enhanced, h=params['denoise_strength'])
    save_step(denoised, step, "denoising NlMeans", output_dir,
              label=f"h={params['denoise_strength']}")
    step += 1
 
    # ── FASE 2: Segmentación ───────────────────────────────────────────────────
 
    # 3. Umbralización adaptativa
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params['adaptive_block_size'],
        params['adaptive_c']
    )
    save_step(binary, step, "umbral adaptativo (gaussiano inv)", output_dir,
              label=f"blockSize={params['adaptive_block_size']}  C={params['adaptive_c']}")
    step += 1
 
    # 4. Morfología OPEN
    k = params['morphology_kernel_size']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    after_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    save_step(after_open, step, "morfologia OPEN", output_dir,
              label=f"kernel elipse {k}x{k}  — elimina ruido puntual")
    step += 1
 
    # 5. Morfología CLOSE
    after_close = cv2.morphologyEx(after_open, cv2.MORPH_CLOSE, kernel, iterations=1)
    save_step(after_close, step, "morfologia CLOSE", output_dir,
              label=f"kernel elipse {k}x{k}  — cierra huecos internos")
    step += 1
 
    binary_morph = after_close
 
    # 6. Primera pasada findContours → relleno de huecos
    contours_pass1, _ = cv2.findContours(
        binary_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    binary_filled = binary_morph.copy()
    filled_count = 0
    for cnt in contours_pass1:
        if cv2.contourArea(cnt) > params['min_droplet_area']:
            cv2.drawContours(binary_filled, [cnt], -1, 255, thickness=-1)
            filled_count += 1
 
    save_step(binary_filled, step, "relleno de huecos (1a pasada)", output_dir,
              label=f"{filled_count} contornos rellenados")
    step += 1
 
    # 7. Segunda pasada findContours sobre máscara rellena
    contours_pass2, _ = cv2.findContours(
        binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 
    vis_all = cv2.cvtColor(binary_filled, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_all, contours_pass2, -1, (0, 180, 255), 1)
    save_step(vis_all, step, "contornos antes de filtrar", output_dir,
              label=f"{len(contours_pass2)} contornos totales")
    step += 1
 
    # ── FASE 3: Filtrado y medición ────────────────────────────────────────────
 
    rejected_area   = []
    rejected_circ   = []
    accepted        = []
 
    for cnt in contours_pass2:
        area = cv2.contourArea(cnt)
 
        if not (params['min_droplet_area'] <= area <= params['max_droplet_area']):
            rejected_area.append(cnt)
            continue
 
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
        circularity = min(1.0, circularity)
 
        if circularity < params['min_droplet_circularity']:
            rejected_circ.append(cnt)
            continue
 
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])
        else:
            x_, y_, w_, h_ = cv2.boundingRect(cnt)
            cx, cy = x_ + w_ / 2, y_ + h_ / 2
 
        diameter = 2 * np.sqrt(area / np.pi)
        accepted.append({
            'centroid': (cx, cy),
            'area': area,
            'diameter': diameter,
            'circularity': circularity,
            'contour': cnt,
        })
 
    # 8. Resultado tras filtro de área
    vis_area = frame_bgr.copy()
    cv2.drawContours(vis_area, rejected_area, -1, (0, 0, 200), 1)
    cv2.drawContours(vis_area, rejected_circ + [d['contour'] for d in accepted], -1, (0, 200, 0), 1)
    save_step(vis_area, step, "filtro area (rojo=rechazado)", output_dir,
              label=(f"rechazadas por area: {len(rejected_area)}  "
                     f"|  min={params['min_droplet_area']}  max={params['max_droplet_area']}"))
    step += 1
 
    # 9. Resultado tras filtro de circularidad
    vis_circ = frame_bgr.copy()
    cv2.drawContours(vis_circ, rejected_area + rejected_circ, -1, (0, 0, 200), 1)
    cv2.drawContours(vis_circ, [d['contour'] for d in accepted], -1, (0, 220, 0), 1)
    save_step(vis_circ, step, "filtro circularidad (rojo=rechazado)", output_dir,
              label=(f"rechazadas por circularidad: {len(rejected_circ)}  "
                     f"|  min_circ={params['min_droplet_circularity']}"))
    step += 1
 
    # 10. Resultado final con anotaciones
    vis_final = frame_bgr.copy()
    for d in accepted:
        cx, cy = int(d['centroid'][0]), int(d['centroid'][1])
        r = max(1, int(d['diameter'] / 2))
        cv2.circle(vis_final, (cx, cy), r,  (0, 255, 0), 1)
        cv2.circle(vis_final, (cx, cy), 2,  (0, 0, 255), -1)
 
    save_step(vis_final, step, "gotas detectadas final", output_dir,
              label=f"{len(accepted)} gotas validas detectadas")
    step += 1
 
    # ── Resumen ────────────────────────────────────────────────────────────────
    print(f"\nResumen:")
    print(f"  Contornos totales (2ª pasada) : {len(contours_pass2)}")
    print(f"  Rechazadas por área           : {len(rejected_area)}")
    print(f"  Rechazadas por circularidad   : {len(rejected_circ)}")
    print(f"  Gotas válidas                 : {len(accepted)}")
    if accepted:
        diameters = [d['diameter'] for d in accepted]
        circs     = [d['circularity'] for d in accepted]
        print(f"  Diámetro medio (px)           : {np.mean(diameters):.2f}  "
              f"[{np.min(diameters):.1f} – {np.max(diameters):.1f}]")
        print(f"  Circularidad media            : {np.mean(circs):.3f}  "
              f"[{np.min(circs):.3f} – {np.max(circs):.3f}]")
 
    print(f"\n{step} imágenes guardadas en '{output_dir}/'")
    return accepted
 
 
# ── Punto de entrada ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
 
    video_path_arg  = sys.argv[1]
    frame_idx_arg   = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_dir_arg  = sys.argv[3] if len(sys.argv) > 3 else 'debug_frames'
 
    run_pipeline_debug(video_path_arg, frame_idx_arg, output_dir_arg)
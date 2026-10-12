"""
droplet_analyzer.py
Clase principal para el análisis de gotas
"""

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from collections import defaultdict
import os
import time


# Etapas del pipeline para las que se mide el tiempo de ejecución
TIMING_STAGES = ('decode', 'preprocess', 'threshold_morphology',
                 'contours_features', 'tracking', 'output')


class DropletAnalyzer:
    """Analizador de gotas en video"""
    
    def __init__(self, video_path, output_dir='output'):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.params = {
            'min_droplet_area': 10,
            'max_droplet_area': 5000,
            'min_droplet_circularity': 0.6,
            'min_droplet_roundness': 0.0,
            'threshold_value': 100,
            'max_tracking_distance': 50,
            'clahe_clip_limit': 2.0,
            'denoise_strength': 10,
            'adaptive_block_size': 11,
            'adaptive_c': 2,
            'morphology_kernel_size': 3,
            'use_clahe': False,
            'use_lucas_kanade': True,     
            'lk_window_size': 15          
        }

        self.pixel_scale = None   
        self.scale_unit = 'px'    
        
        self.droplet_tracks = []
        self.next_id = 0
        self.prev_gray = None
        self.timings = {s: 0.0 for s in TIMING_STAGES}
        self.frames_processed = 0

        self._load_video_info()
        self.effective_fps = self.fps
        self.progress_callback = None
        self.cancel_requested = False
    
    def _load_video_info(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    def set_parameter(self, param_name, value):
        if param_name in self.params:
            self.params[param_name] = value
    
    def get_parameter(self, param_name):
        return self.params.get(param_name)
    
    def get_all_parameters(self):
        return self.params.copy()
    
    def set_fps(self, fps):
        if fps and fps > 0:
            self.effective_fps = float(fps)
        else:
            self.effective_fps = self.fps

    def set_scale(self, pixel_scale, unit='µm'):
        if pixel_scale and pixel_scale > 0:
            self.pixel_scale = float(pixel_scale)
            self.scale_unit = unit
        else:
            self.pixel_scale = None
            self.scale_unit = 'px'
    
    def px_to_real(self, value_px):
        if self.pixel_scale:
            return value_px * self.pixel_scale
        return value_px
    
    def px2_to_real2(self, area_px2):
        if self.pixel_scale:
            return area_px2 * (self.pixel_scale ** 2)
        return area_px2
    
    def set_progress_callback(self, callback):
        self.progress_callback = callback
    
    def cancel_processing(self):
        self.cancel_requested = True
    
    def preprocess_frame(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if self.params['use_clahe']:
            clh = cv2.createCLAHE(
                clipLimit=self.params['clahe_clip_limit'], 
                tileGridSize=(8, 8)
            )
            enhanced = clh.apply(gray)
        else:
            flattened = gray.flatten()
            low_percentile = 0.4
            high_percentile = 100.0 - 0.4
            low_val, high_val = np.percentile(flattened, [low_percentile, high_percentile])
            if high_val - low_val < 1:
                # Frame casi uniforme (p. ej. pocas gotas sobre fondo limpio):
                # ambos percentiles caen en el fondo y el estiramiento dividiría
                # por cero, así que se deja el frame sin estirar.
                enhanced = gray
            else:
                dst = np.clip(gray, low_val, high_val)
                enhanced = ((dst - low_val) / (high_val - low_val) * 255).astype(np.uint8)

        denoised = cv2.fastNlMeansDenoising(
            enhanced, 
            h=self.params['denoise_strength']
        )
        return denoised
    
    def segment_droplets(self, frame, processed=None):
        """Segmenta las gotas de un frame. Si se pasa `processed` (salida de
        preprocess_frame) se reutiliza para no repetir el preprocesado."""
        t0 = time.perf_counter()
        if processed is None:
            processed = self.preprocess_frame(frame)
        t1 = time.perf_counter()
        binary = self._binarize(processed)
        t2 = time.perf_counter()
        droplets = self._extract_droplets(binary)
        t3 = time.perf_counter()

        self.timings['preprocess'] += t1 - t0
        self.timings['threshold_morphology'] += t2 - t1
        self.timings['contours_features'] += t3 - t2
        return binary, droplets

    def _binarize(self, processed):
        binary = cv2.adaptiveThreshold(
            processed, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            self.params['adaptive_block_size'],
            self.params['adaptive_c']
        )
        
        kernel_size = self.params['morphology_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _extract_droplets(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        for cnt in contours:
            if cv2.contourArea(cnt) > self.params['min_droplet_area']:
                cv2.drawContours(binary, [cnt], -1, 255, thickness=-1)
        
        droplets = []
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.params["min_droplet_area"] <= area <= self.params["max_droplet_area"]:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centroid_x = float(M["m10"] / M["m00"])
                    centroid_y = float(M["m01"] / M["m00"])
                else:
                    x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
                    centroid_x, centroid_y = x_box + w_box/2, y_box + h_box/2

                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    circularity = min(1.0, circularity)
                else:
                    circularity = 0

                if circularity >= self.params["min_droplet_circularity"]:
                    roundness = 0
                    if len(cnt) >= 5:
                        _, (ma, MA), _ = cv2.fitEllipse(cnt)
                        if MA > 0:
                            roundness = (4 * area) / (np.pi * (MA**2))
                            roundness = min(1.0, roundness)

                    if roundness < self.params.get("min_droplet_roundness", 0.0):
                        continue

                    diameter = 2 * np.sqrt(area / np.pi)
                    bbox = cv2.boundingRect(cnt)
                    mask = np.zeros(binary.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    
                    droplets.append({
                        'centroid': (centroid_x, centroid_y),
                        'area': area,
                        'diameter': diameter,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'roundness': roundness, 
                        'bbox': bbox,
                        'mask': mask,
                        'contour': cnt # NUEVO: Guardamos el contorno para dibujarlo en naranja luego
                    })

        return droplets

    def track_droplets(self, current_droplets, frame_idx, gray_frame=None):
        t0 = time.perf_counter()
        self._track_droplets(current_droplets, frame_idx, gray_frame)
        self.timings['tracking'] += time.perf_counter() - t0

    def _track_droplets(self, current_droplets, frame_idx, gray_frame=None):
        max_distance = self.params['max_tracking_distance']
        use_lk = self.params.get('use_lucas_kanade', True)
        lk_win = int(self.params.get('lk_window_size', 15))
        
        predicted_positions = {} 

        if use_lk and frame_idx > 0 and self.prev_gray is not None and gray_frame is not None:
            active_indices = [j for j, t in enumerate(self.droplet_tracks) if t['frames'][-1] == frame_idx - 1]
            
            if active_indices:
                p0 = np.array([[self.droplet_tracks[j]['positions'][-1]] for j in active_indices], dtype=np.float32)
                
                lk_params = dict(winSize=(lk_win, lk_win), maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_frame, p0, None, **lk_params)
                
                for k, j in enumerate(active_indices):
                    if st[k][0] == 1:
                        predicted_positions[j] = tuple(p1[k][0])

        if frame_idx == 0:
            for droplet in current_droplets:
                track = {
                    'id': self.next_id,
                    'positions': [droplet['centroid']],
                    'frames': [frame_idx],
                    'areas': [droplet['area']],
                    'diameters': [droplet['diameter']],
                    'circularities': [droplet['circularity']],
                    'roundnesses': [droplet['roundness']]
                }
                self.droplet_tracks.append(track)
                self.next_id += 1
        else:
            used_tracks = set()
            used_droplets = set()
            
            for i, droplet in enumerate(current_droplets):
                best_track_idx = None
                best_distance = float('inf')
                
                for j, track in enumerate(self.droplet_tracks):
                    if j in used_tracks:
                        continue
                    
                    if track['frames'][-1] < frame_idx - 5:
                        continue
                    
                    reference_pos = predicted_positions.get(j, track['positions'][-1])
                    
                    distance = np.sqrt(
                        (droplet['centroid'][0] - reference_pos[0])**2 + 
                        (droplet['centroid'][1] - reference_pos[1])**2
                    )
                    
                    if distance < best_distance and distance < max_distance:
                        best_distance = distance
                        best_track_idx = j
                
                if best_track_idx is not None:
                    track = self.droplet_tracks[best_track_idx]
                    track['positions'].append(droplet['centroid']) 
                    track['frames'].append(frame_idx)
                    track['areas'].append(droplet['area'])
                    track['diameters'].append(droplet['diameter'])
                    track['circularities'].append(droplet['circularity'])
                    track.setdefault('roundnesses', []).append(droplet['roundness'])
                    used_tracks.add(best_track_idx)
                    used_droplets.add(i)
            
            for i, droplet in enumerate(current_droplets):
                if i not in used_droplets:
                    track = {
                        'id': self.next_id,
                        'positions': [droplet['centroid']],
                        'frames': [frame_idx],
                        'areas': [droplet['area']],
                        'diameters': [droplet['diameter']],
                        'circularities': [droplet['circularity']],
                        'roundnesses': [droplet['roundness']]
                    }
                    self.droplet_tracks.append(track)
                    self.next_id += 1

        if use_lk and gray_frame is not None:
            self.prev_gray = gray_frame.copy()
    
    def calculate_velocities(self):
        for track in self.droplet_tracks:
            if len(track['positions']) < 2:
                track['velocity_x'] = []
                track['velocity_y'] = []
                track['velocity_magnitude'] = []
                continue
            
            positions = np.array(track['positions'])
            frames = np.array(track['frames'])
            
            dx = np.diff(positions[:, 0])
            dy = np.diff(positions[:, 1])
            dt = np.diff(frames) / self.effective_fps
            
            vx = dx / dt
            vy = dy / dt
            v_mag = np.sqrt(vx**2 + vy**2)
            
            track['velocity_x'] = vx.tolist()
            track['velocity_y'] = vy.tolist()
            track['velocity_magnitude'] = v_mag.tolist()
    
    def process_video(self, output_video=None, frame_callback=None):
        """
        Procesa el vídeo completo: preprocesado, segmentación, tracking y
        cálculo de velocidades.

        Parameters
        ----------
        output_video   : ruta del vídeo de visualización a guardar (None = no se guarda)
        frame_callback : función opcional f(frame_idx, frame, preprocessed, binary, droplets)
                         llamada en cada frame. Si devuelve una imagen BGR, esa
                         imagen es la que se escribe en `output_video`; si devuelve
                         None se escribe la visualización por defecto.

        Returns
        -------
        True si se completó, False si se canceló.
        """
        self.reset()
        cap = cv2.VideoCapture(self.video_path)
        out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))

        frame_idx = 0
        t_start = time.perf_counter()
        try:
            while True:
                if self.cancel_requested:
                    return False

                t0 = time.perf_counter()
                ret, frame = cap.read()
                self.timings['decode'] += time.perf_counter() - t0
                if not ret:
                    break

                t0 = time.perf_counter()
                preprocessed = self.preprocess_frame(frame)
                self.timings['preprocess'] += time.perf_counter() - t0
                binary, droplets = self.segment_droplets(frame, processed=preprocessed)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
                self.track_droplets(droplets, frame_idx, gray)

                t0 = time.perf_counter()
                vis = None
                if frame_callback is not None:
                    vis = frame_callback(frame_idx, frame, preprocessed, binary, droplets)
                if out is not None:
                    out.write(vis if vis is not None else self.draw_detections(frame, droplets, frame_idx))
                self.timings['output'] += time.perf_counter() - t0

                if self.progress_callback and self.total_frames > 0:
                    pct = int((frame_idx + 1) / self.total_frames * 100)
                    self.progress_callback(pct, frame_idx + 1, self.total_frames)

                frame_idx += 1
        finally:
            cap.release()
            if out is not None:
                out.release()

        self.frames_processed = frame_idx
        self.total_time = time.perf_counter() - t_start
        self.calculate_velocities()
        return True

    def draw_detections(self, frame, droplets, frame_idx, show_diam=True,
                        show_perim=False, show_lines=True):
        """Dibuja detecciones y trayectorias sobre una copia del frame (BGR)."""
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        for d in droplets:
            cx, cy = d['centroid']
            if show_diam:
                cv2.circle(vis, (int(cx), int(cy)), int(d['diameter'] / 2), (0, 255, 0), 1)
            if show_perim and 'contour' in d:
                cv2.drawContours(vis, [d['contour']], -1, (0, 165, 255), 1)
            cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), -1)

        if show_lines:
            for track in self.droplet_tracks:
                if len(track['positions']) > 1 and track['frames'][-1] >= frame_idx - 10:
                    pts = np.array(track['positions'], dtype=np.int32)
                    cv2.polylines(vis, [pts], False, (255, 0, 0), 1)
        return vis

    def get_timing_summary(self):
        """Tiempo por etapa (s, ms/frame y % del total) y frames/s globales."""
        n = max(self.frames_processed, 1)
        total = sum(self.timings.values())
        rows = {}
        for stage, t in self.timings.items():
            rows[stage] = {
                'total_s': t,
                'ms_per_frame': 1000 * t / n,
                'percent': 100 * t / total if total > 0 else 0.0,
            }
        return {
            'frames': self.frames_processed,
            'total_s': total,
            'fps': n / total if total > 0 else 0.0,
            'stages': rows,
        }

    def reset(self):
        self.droplet_tracks = []
        self.next_id = 0
        self.cancel_requested = False
        self.prev_gray = None
        self.timings = {s: 0.0 for s in TIMING_STAGES}
        self.frames_processed = 0
    
    def export_to_dataframe(self):
        # Mantenemos las columnas internas en inglés por defecto
        u = self.scale_unit  
        data = []
        for track in self.droplet_tracks:
            track_id = track['id']
            
            for i in range(len(track['frames'])):
                row = {
                    'track_id': track_id,
                    'frame': track['frames'][i],
                    'time_s': track['frames'][i] / self.effective_fps,
                    f'x_{u}': self.px_to_real(track['positions'][i][0]),
                    f'y_{u}': self.px_to_real(track['positions'][i][1]),
                    f'area_{u}2': self.px2_to_real2(track['areas'][i]),
                    f'diameter_{u}': self.px_to_real(track['diameters'][i]),
                    'circularity': track['circularities'][i],
                    'roundness': track.get('roundnesses', [0]*len(track['frames']))[i]
                }
                
                if i < len(track.get('velocity_x', [])):
                    row[f'velocity_x_{u}_s'] = self.px_to_real(track['velocity_x'][i])
                    row[f'velocity_y_{u}_s'] = self.px_to_real(track['velocity_y'][i])
                    row[f'velocity_magnitude_{u}_s'] = self.px_to_real(track['velocity_magnitude'][i])
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def _translate_df_headers(self, df, lang):
        """Traduce las cabeceras del DataFrame al idioma seleccionado antes de exportar"""
        if lang == 'en':
            return df
        
        # Traducción a español
        new_cols = []
        for c in df.columns:
            c = c.replace('track_id', 'id_trayectoria')
            c = c.replace('time', 'tiempo')
            c = c.replace('area', 'area')
            c = c.replace('diameter', 'diametro')
            c = c.replace('circularity', 'circularidad')
            c = c.replace('roundness', 'redondez')
            c = c.replace('velocity', 'velocidad')
            new_cols.append(c)
        
        df_translated = df.copy()
        df_translated.columns = new_cols
        return df_translated

    def export_results(self, filename='droplet_data.csv', lang='es'):
        df = self.export_to_dataframe()
        df_export = self._translate_df_headers(df, lang)
        
        if not filename.endswith('.csv'):
            filename += '.csv'
        csv_path = os.path.join(self.output_dir, filename)
        df_export.to_csv(csv_path, index=False)
        return df, csv_path
    
    def get_statistics(self):
        df = self.export_to_dataframe()
        if len(df) == 0:
            return None
        
        u = self.scale_unit
        diam_col = f'diameter_{u}'
        area_col = f'area_{u}2'
        vel_col = f'velocity_magnitude_{u}_s'
        
        stats = {
            'total_tracks': df['track_id'].nunique(),
            'total_detections': len(df),
            'diameter_mean': df[diam_col].mean(),
            'diameter_median': df[diam_col].median(),
            'diameter_std': df[diam_col].std(),
            'diameter_min': df[diam_col].min(),
            'diameter_max': df[diam_col].max(),
            'area_mean': df[area_col].mean(),
            'area_median': df[area_col].median(),
            'area_std': df[area_col].std(),
            'circularity_mean': df['circularity'].mean(),
            'circularity_median': df['circularity'].median(),
            'roundness_mean': df['roundness'].mean() if 'roundness' in df.columns else None,
            'roundness_median': df['roundness'].median() if 'roundness' in df.columns else None,
            'unit': u,
        }
        
        if vel_col in df.columns:
            velocities = df[vel_col].dropna()
            if len(velocities) > 0:
                stats['velocity_mean'] = velocities.mean()
                stats['velocity_median'] = velocities.median()
                stats['velocity_std'] = velocities.std()
                stats['velocity_min'] = velocities.min()
                stats['velocity_max'] = velocities.max()
        
        return stats
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


class DropletAnalyzer:
    """Analizador de gotas en video"""
    
    def __init__(self, video_path, output_dir='output'):
        """
        Inicializa el analizador de gotas
        
        Args:
            video_path: ruta al archivo de video
            output_dir: directorio para guardar resultados
        """
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Parámetros de segmentación (ajustables)
        self.params = {
            'min_droplet_area': 10,
            'max_droplet_area': 5000,
            'min_droplet_circularity': 0.6,
            'threshold_value': 100,
            'max_tracking_distance': 50,
            'clahe_clip_limit': 2.0,
            'denoise_strength': 10,
            'adaptive_block_size': 11,
            'adaptive_c': 2,
            'morphology_kernel_size': 3,
            'use_clahe': False
        }

        # Escala de calibración: cuántas unidades reales por píxel
        # Si es None, las medidas se reportan en píxeles
        self.pixel_scale = None   # unidades reales / píxel
        self.scale_unit = 'px'    # nombre de la unidad (ej: 'µm', 'mm')
        
        # Para tracking
        self.droplet_tracks = []
        self.next_id = 0
        
        # Info del video
        self._load_video_info()
        
        # Callback para actualización de progreso
        self.progress_callback = None
        
        # Flag para cancelación
        self.cancel_requested = False
    
    def _load_video_info(self):
        """Carga información del video"""
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    def set_parameter(self, param_name, value):
        """Actualiza un parámetro"""
        if param_name in self.params:
            self.params[param_name] = value
    
    def get_parameter(self, param_name):
        """Obtiene un parámetro"""
        return self.params.get(param_name)
    
    def get_all_parameters(self):
        """Obtiene todos los parámetros"""
        return self.params.copy()
    
    def set_scale(self, pixel_scale, unit='µm'):
        """
        Establece la relación píxel/distancia real.
        
        Args:
            pixel_scale: distancia real por píxel (ej: 0.5 para 0.5 µm/px)
                         Si es None o 0, se desactiva la conversión.
            unit: nombre de la unidad real (ej: 'µm', 'mm')
        """
        if pixel_scale and pixel_scale > 0:
            self.pixel_scale = float(pixel_scale)
            self.scale_unit = unit
        else:
            self.pixel_scale = None
            self.scale_unit = 'px'
    
    def px_to_real(self, value_px):
        """Convierte un valor en píxeles a la unidad real configurada"""
        if self.pixel_scale:
            return value_px * self.pixel_scale
        return value_px
    
    def px2_to_real2(self, area_px2):
        """Convierte área en píxeles² a la unidad real al cuadrado"""
        if self.pixel_scale:
            return area_px2 * (self.pixel_scale ** 2)
        return area_px2
    
    def save_diagnostic_frames(self, preprocessed_filename='frame_preprocesado.png',
                                binary_filename='frame_binarizado.png'):
        """
        Guarda el primer frame del video preprocesado y binarizado para diagnóstico.
        
        Args:
            preprocessed_filename: nombre del archivo para el frame preprocesado
            binary_filename: nombre del archivo para el frame binarizado
        Returns:
            Tupla con las rutas de los dos archivos guardados
        """
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("No se pudo leer el primer frame del video")
        
        # Frame preprocesado
        preprocessed = self.preprocess_frame(frame)
        pre_path = os.path.join(self.output_dir, preprocessed_filename)
        cv2.imwrite(pre_path, preprocessed)
        
        # Frame binarizado
        binary, _ = self.segment_droplets(frame)
        bin_path = os.path.join(self.output_dir, binary_filename)
        cv2.imwrite(bin_path, binary)
        
        return pre_path, bin_path
    
    def set_progress_callback(self, callback):
        """Establece función callback para progreso"""
        self.progress_callback = callback
    
    def cancel_processing(self):
        """Cancela el procesamiento"""
        self.cancel_requested = True
    
    def preprocess_frame(self, frame):
        """
        Preprocesa el frame para mejorar la detección
        """
        # Convertir a escala de grises si es necesario
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
            dst = np.clip(gray, low_val, high_val)
            enhanced = ((dst - low_val) / (high_val - low_val) * 255).astype(np.uint8)

        # Reducir ruido
        denoised = cv2.fastNlMeansDenoising(
            enhanced, 
            h=self.params['denoise_strength']
        )
        
        return denoised
    
    def segment_droplets(self, frame):
        """
        Segmenta las gotas en un frame
        
        Returns:
            binary: imagen binaria con gotas segmentadas
            droplets: lista de propiedades de cada gota
        """
        processed = self.preprocess_frame(frame)
        
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
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    
        for cnt in contours:
            if cv2.contourArea(cnt) > self.params['min_droplet_area']:
                cv2.drawContours(binary, [cnt], -1, 255, thickness=-1)
        droplets = []

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                        'mask': mask
                    })
        
        return binary, droplets
    
    def track_droplets(self, current_droplets, frame_idx):
        """
        Rastrea gotas entre frames consecutivos
        """
        max_distance = self.params['max_tracking_distance']
        
        if frame_idx == 0:
            for droplet in current_droplets:
                track = {
                    'id': self.next_id,
                    'positions': [droplet['centroid']],
                    'frames': [frame_idx],
                    'areas': [droplet['area']],
                    'diameters': [droplet['diameter']],
                    'circularities': [droplet['circularity']]
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
                    
                    last_pos = track['positions'][-1]
                    distance = np.sqrt(
                        (droplet['centroid'][0] - last_pos[0])**2 + 
                        (droplet['centroid'][1] - last_pos[1])**2
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
                        'circularities': [droplet['circularity']]
                    }
                    self.droplet_tracks.append(track)
                    self.next_id += 1
    
    def calculate_velocities(self):
        """
        Calcula velocidades para cada track
        """
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
            dt = np.diff(frames) / self.fps 
            
            vx = dx / dt
            vy = dy / dt
            v_mag = np.sqrt(vx**2 + vy**2)
            
            track['velocity_x'] = vx.tolist()
            track['velocity_y'] = vy.tolist()
            track['velocity_magnitude'] = v_mag.tolist()
    
    def reset(self):
        """Reinicia el análisis"""
        self.droplet_tracks = []
        self.next_id = 0
        self.cancel_requested = False
    
    def process_video(self, save_visualization=True, output_video_filename='tracked_droplets.avi'):
        """
        Procesa todo el video
        """
        self.reset()
        
        cap = cv2.VideoCapture(self.video_path)
        
        if save_visualization:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                os.path.join(self.output_dir, output_video_filename),
                fourcc, self.fps, (self.width, self.height)
            )
        
        frame_idx = 0
        
        while True:
            if self.cancel_requested:
                print("Procesamiento cancelado por el usuario")
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            _, droplets = self.segment_droplets(frame)
            
            self.track_droplets(droplets, frame_idx)
            
            if save_visualization:
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
                
                for droplet in droplets:
                    cx, cy = droplet['centroid']
                    cv2.circle(vis_frame, (int(cx), int(cy)), int(droplet['diameter']/2), (0, 255, 0), 1)
                    cv2.circle(vis_frame, (int(cx), int(cy)), 2, (0, 0, 255), -1)
                
                for track in self.droplet_tracks:
                    if len(track['positions']) > 1 and track['frames'][-1] >= frame_idx - 10:
                        points = np.array(track['positions'], dtype=np.int32)
                        cv2.polylines(vis_frame, [points], False, (255, 0, 0), 1)
                
                out.write(vis_frame)
            
            if self.progress_callback:
                progress = int((frame_idx + 1) / self.total_frames * 100)
                self.progress_callback(progress, frame_idx + 1, self.total_frames)
            
            frame_idx += 1
        
        cap.release()
        if save_visualization:
            out.release()
        
        if not self.cancel_requested:
            self.calculate_velocities()
        
        return not self.cancel_requested
    
    def export_to_dataframe(self):
        """
        Exporta resultados a DataFrame.
        Si hay escala configurada, las medidas se convierten a unidades reales.
        """
        u = self.scale_unit  # unidad activa
        data = []
        for track in self.droplet_tracks:
            track_id = track['id']
            
            for i in range(len(track['frames'])):
                row = {
                    'track_id': track_id,
                    'frame': track['frames'][i],
                    'time_s': track['frames'][i] / self.fps,
                    f'x_{u}': self.px_to_real(track['positions'][i][0]),
                    f'y_{u}': self.px_to_real(track['positions'][i][1]),
                    f'area_{u}2': self.px2_to_real2(track['areas'][i]),
                    f'diameter_{u}': self.px_to_real(track['diameters'][i]),
                    'circularity': track['circularities'][i]
                }
                
                if i < len(track.get('velocity_x', [])):
                    row[f'velocity_x_{u}_s'] = self.px_to_real(track['velocity_x'][i])
                    row[f'velocity_y_{u}_s'] = self.px_to_real(track['velocity_y'][i])
                    row[f'velocity_magnitude_{u}_s'] = self.px_to_real(track['velocity_magnitude'][i])
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def export_results(self, filename='droplet_data.csv'):
        """
        Exporta resultados a CSV con nombre de archivo personalizable.
        """
        df = self.export_to_dataframe()
        if not filename.endswith('.csv'):
            filename += '.csv'
        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False)
        return df, csv_path
    
    def get_statistics(self):
        """
        Calcula estadísticas del análisis
        """
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

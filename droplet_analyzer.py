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
            'threshold_value': 100,
            'max_tracking_distance': 50,
            'clahe_clip_limit': 2.0,
            'denoise_strength': 10,
            'adaptive_block_size': 11,
            'adaptive_c': 2,
            'morphology_kernel_size': 3
        }
        
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
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip_limit'], 
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(gray)
        
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
        # Preprocesar
        processed = self.preprocess_frame(frame)
        
        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(
            processed, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            self.params['adaptive_block_size'],
            self.params['adaptive_c']
        )
        
        # Operaciones morfológicas para limpiar
        kernel_size = self.params['morphology_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Etiquetar componentes conectados
        labeled, num_features = ndimage.label(binary)
        
        # Extraer propiedades de cada gota
        droplets = []
        min_area = self.params['min_droplet_area']
        max_area = self.params['max_droplet_area']
        
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            area = np.sum(mask)
            
            # Filtrar por área
            if area < min_area or area > max_area:
                continue
            
            # Calcular centroide
            coords = np.argwhere(mask)
            centroid_y = np.mean(coords[:, 0])
            centroid_x = np.mean(coords[:, 1])
            
            # Calcular diámetro equivalente
            diameter = 2 * np.sqrt(area / np.pi)
            
            # Calcular perímetro y circularidad
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            else:
                perimeter = 0
                circularity = 0
            
            # Bounding box
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            
            droplets.append({
                'centroid': (centroid_x, centroid_y),
                'area': area,
                'diameter': diameter,
                'perimeter': perimeter,
                'circularity': circularity,
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
            # Primer frame: inicializar tracks
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
            # Frames subsiguientes: asociar con tracks existentes
            used_tracks = set()
            used_droplets = set()
            
            # Calcular distancias entre gotas actuales y últimas posiciones de tracks
            for i, droplet in enumerate(current_droplets):
                best_track_idx = None
                best_distance = float('inf')
                
                for j, track in enumerate(self.droplet_tracks):
                    if j in used_tracks:
                        continue
                    
                    # Solo considerar tracks activos (último frame reciente)
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
                    # Actualizar track existente
                    track = self.droplet_tracks[best_track_idx]
                    track['positions'].append(droplet['centroid'])
                    track['frames'].append(frame_idx)
                    track['areas'].append(droplet['area'])
                    track['diameters'].append(droplet['diameter'])
                    track['circularities'].append(droplet['circularity'])
                    used_tracks.add(best_track_idx)
                    used_droplets.add(i)
            
            # Crear nuevos tracks para gotas no asociadas
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
            
            # Calcular diferencias
            dx = np.diff(positions[:, 0])
            dy = np.diff(positions[:, 1])
            dt = np.diff(frames) / self.fps  # tiempo en segundos
            
            # Velocidades en píxeles/segundo
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
    
    def process_video(self, save_visualization=True):
        """
        Procesa todo el video
        """
        self.reset()
        
        cap = cv2.VideoCapture(self.video_path)
        
        if save_visualization:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                os.path.join(self.output_dir, 'tracked_droplets.avi'),
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
            
            # Segmentar gotas
            binary, droplets = self.segment_droplets(frame)
            
            # Rastrear gotas
            self.track_droplets(droplets, frame_idx)
            
            # Visualización
            if save_visualization:
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
                
                # Dibujar gotas detectadas
                for droplet in droplets:
                    cx, cy = droplet['centroid']
                    cv2.circle(vis_frame, (int(cx), int(cy)), int(droplet['diameter']/2), (0, 255, 0), 1)
                    cv2.circle(vis_frame, (int(cx), int(cy)), 2, (0, 0, 255), -1)
                
                # Dibujar trayectorias de tracks activos
                for track in self.droplet_tracks:
                    if len(track['positions']) > 1 and track['frames'][-1] >= frame_idx - 10:
                        points = np.array(track['positions'], dtype=np.int32)
                        cv2.polylines(vis_frame, [points], False, (255, 0, 0), 1)
                
                out.write(vis_frame)
            
            # Actualizar progreso
            if self.progress_callback:
                progress = int((frame_idx + 1) / self.total_frames * 100)
                self.progress_callback(progress, frame_idx + 1, self.total_frames)
            
            frame_idx += 1
        
        cap.release()
        if save_visualization:
            out.release()
        
        if not self.cancel_requested:
            # Calcular velocidades
            self.calculate_velocities()
        
        return not self.cancel_requested
    
    def export_to_dataframe(self):
        """
        Exporta resultados a DataFrame
        """
        data = []
        for track in self.droplet_tracks:
            track_id = track['id']
            
            for i in range(len(track['frames'])):
                row = {
                    'track_id': track_id,
                    'frame': track['frames'][i],
                    'time_s': track['frames'][i] / self.fps,
                    'x_px': track['positions'][i][0],
                    'y_px': track['positions'][i][1],
                    'area_px2': track['areas'][i],
                    'diameter_px': track['diameters'][i],
                    'circularity': track['circularities'][i]
                }
                
                # Añadir velocidades si existen
                if i < len(track.get('velocity_x', [])):
                    row['velocity_x_px_s'] = track['velocity_x'][i]
                    row['velocity_y_px_s'] = track['velocity_y'][i]
                    row['velocity_magnitude_px_s'] = track['velocity_magnitude'][i]
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def export_results(self):
        """
        Exporta resultados a CSV
        """
        df = self.export_to_dataframe()
        csv_path = os.path.join(self.output_dir, 'droplet_data.csv')
        df.to_csv(csv_path, index=False)
        return df, csv_path
    
    def get_statistics(self):
        """
        Calcula estadísticas del análisis
        """
        df = self.export_to_dataframe()
        
        if len(df) == 0:
            return None
        
        stats = {
            'total_tracks': df['track_id'].nunique(),
            'total_detections': len(df),
            'diameter_mean': df['diameter_px'].mean(),
            'diameter_median': df['diameter_px'].median(),
            'diameter_std': df['diameter_px'].std(),
            'diameter_min': df['diameter_px'].min(),
            'diameter_max': df['diameter_px'].max(),
            'area_mean': df['area_px2'].mean(),
            'area_median': df['area_px2'].median(),
            'area_std': df['area_px2'].std(),
            'circularity_mean': df['circularity'].mean(),
            'circularity_median': df['circularity'].median(),
        }
        
        if 'velocity_magnitude_px_s' in df.columns:
            velocities = df['velocity_magnitude_px_s'].dropna()
            if len(velocities) > 0:
                stats['velocity_mean'] = velocities.mean()
                stats['velocity_median'] = velocities.median()
                stats['velocity_std'] = velocities.std()
                stats['velocity_min'] = velocities.min()
                stats['velocity_max'] = velocities.max()
        
        return stats

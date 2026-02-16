"""
plotter.py
Módulo para generar gráficos y visualizaciones
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para evitar conflictos
import numpy as np
import os


class DropletPlotter:
    """Clase para generar visualizaciones del análisis de gotas"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_all_plots(self, df, filename='analysis_plots.png'):
        """
        Genera todos los gráficos de análisis
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Análisis de Gotas', fontsize=16, fontweight='bold')
        
        # 1. Distribución de diámetros
        self._plot_diameter_distribution(df, axes[0, 0])
        
        # 2. Distribución de áreas
        self._plot_area_distribution(df, axes[0, 1])
        
        # 3. Distribución de velocidades
        self._plot_velocity_distribution(df, axes[0, 2])
        
        # 4. Trayectorias de gotas
        self._plot_trajectories(df, axes[1, 0])
        
        # 5. Diámetro vs Velocidad
        self._plot_diameter_vs_velocity(df, axes[1, 1])
        
        # 6. Circularidad
        self._plot_circularity_distribution(df, axes[1, 2])
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_diameter_distribution(self, df, ax):
        """Gráfico de distribución de diámetros"""
        ax.hist(df['diameter_px'].dropna(), bins=30, edgecolor='black', color='#3498db')
        ax.set_xlabel('Diámetro (píxeles)', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title('Distribución de Diámetros', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Añadir estadísticas
        mean_val = df['diameter_px'].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
        ax.legend(fontsize=8)
    
    def _plot_area_distribution(self, df, ax):
        """Gráfico de distribución de áreas"""
        ax.hist(df['area_px2'].dropna(), bins=30, edgecolor='black', color='#2ecc71')
        ax.set_xlabel('Área (píxeles²)', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title('Distribución de Áreas', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        mean_val = df['area_px2'].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
        ax.legend(fontsize=8)
    
    def _plot_velocity_distribution(self, df, ax):
        """Gráfico de distribución de velocidades"""
        if 'velocity_magnitude_px_s' in df.columns:
            velocities = df['velocity_magnitude_px_s'].dropna()
            if len(velocities) > 0:
                ax.hist(velocities, bins=30, edgecolor='black', color='#e74c3c')
                ax.set_xlabel('Velocidad (píxeles/s)', fontsize=10)
                ax.set_ylabel('Frecuencia', fontsize=10)
                ax.set_title('Distribución de Velocidades', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                mean_val = velocities.mean()
                ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, 
                          label=f'Media: {mean_val:.2f}')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No hay datos de velocidad', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No hay datos de velocidad', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_trajectories(self, df, ax):
        """Gráfico de trayectorias"""
        tracks_to_plot = min(20, df['track_id'].nunique())
        
        # Colores variados
        colors = plt.cm.tab20(np.linspace(0, 1, tracks_to_plot))
        
        for idx, track_id in enumerate(df['track_id'].unique()[:tracks_to_plot]):
            track_data = df[df['track_id'] == track_id].sort_values('frame')
            ax.plot(track_data['x_px'], track_data['y_px'], 
                   alpha=0.6, linewidth=1.5, color=colors[idx])
        
        ax.set_xlabel('X (píxeles)', fontsize=10)
        ax.set_ylabel('Y (píxeles)', fontsize=10)
        ax.set_title(f'Trayectorias de Gotas (primeras {tracks_to_plot})', 
                    fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_diameter_vs_velocity(self, df, ax):
        """Gráfico de diámetro vs velocidad"""
        if 'velocity_magnitude_px_s' in df.columns:
            ax.scatter(df['diameter_px'], df['velocity_magnitude_px_s'], 
                      alpha=0.3, s=10, color='#3498db')
            ax.set_xlabel('Diámetro (píxeles)', fontsize=10)
            ax.set_ylabel('Velocidad (píxeles/s)', fontsize=10)
            ax.set_title('Diámetro vs Velocidad', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No hay datos de velocidad', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_circularity_distribution(self, df, ax):
        """Gráfico de distribución de circularidad"""
        ax.hist(df['circularity'].dropna(), bins=30, edgecolor='black', color='#9b59b6')
        ax.set_xlabel('Circularidad', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title('Distribución de Circularidad', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        mean_val = df['circularity'].mean()
        ax.axvline(mean_val, color='darkviolet', linestyle='--', linewidth=2, 
                  label=f'Media: {mean_val:.3f}')
        ax.legend(fontsize=8)
    
    def plot_single_metric(self, df, metric, filename=None):
        """
        Genera un gráfico individual para una métrica específica
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if metric == 'diameter':
            self._plot_diameter_distribution(df, ax)
        elif metric == 'area':
            self._plot_area_distribution(df, ax)
        elif metric == 'velocity':
            self._plot_velocity_distribution(df, ax)
        elif metric == 'trajectories':
            self._plot_trajectories(df, ax)
        elif metric == 'diameter_vs_velocity':
            self._plot_diameter_vs_velocity(df, ax)
        elif metric == 'circularity':
            self._plot_circularity_distribution(df, ax)
        
        plt.tight_layout()
        
        if filename:
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            return plot_path
        else:
            return fig
    
    def plot_time_series(self, df, track_id, filename=None):
        """
        Genera gráficos de series temporales para un track específico
        """
        track_data = df[df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) == 0:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Análisis Temporal - Track {track_id}', fontsize=14, fontweight='bold')
        
        # Posición X vs tiempo
        axes[0, 0].plot(track_data['time_s'], track_data['x_px'], 'o-', color='#3498db')
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('Posición X (px)')
        axes[0, 0].set_title('Posición X')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Posición Y vs tiempo
        axes[0, 1].plot(track_data['time_s'], track_data['y_px'], 'o-', color='#2ecc71')
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('Posición Y (px)')
        axes[0, 1].set_title('Posición Y')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Diámetro vs tiempo
        axes[1, 0].plot(track_data['time_s'], track_data['diameter_px'], 'o-', color='#e74c3c')
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('Diámetro (px)')
        axes[1, 0].set_title('Diámetro')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Velocidad vs tiempo
        if 'velocity_magnitude_px_s' in track_data.columns:
            axes[1, 1].plot(track_data['time_s'].iloc[:-1], 
                          track_data['velocity_magnitude_px_s'].dropna(), 
                          'o-', color='#9b59b6')
            axes[1, 1].set_xlabel('Tiempo (s)')
            axes[1, 1].set_ylabel('Velocidad (px/s)')
            axes[1, 1].set_title('Velocidad')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            return plot_path
        else:
            return fig

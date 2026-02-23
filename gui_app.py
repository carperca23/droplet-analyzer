"""
gui_app.py
Interfaz gráfica para el análisis de gotas
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

from droplet_analyzer import DropletAnalyzer
from plotter import DropletPlotter


class DropletAnalyzerGUI:
    """Interfaz gráfica para el análisis de gotas"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Gotas - Droplet Analyzer")
        self.root.geometry("1200x800")
        
        # Variables
        self.video_path = None
        self.analyzer = None
        self.plotter = None
        self.output_dir = 'output'
        self.processing = False
        
        # Variables de escala
        self.pixel_scale_var = tk.DoubleVar(value=0.0)   # 0 = desactivado
        self.scale_unit_var = tk.StringVar(value='µm')
        
        # Variables de nombres de archivo
        self.csv_filename_var = tk.StringVar(value='droplet_data')
        self.plots_filename_var = tk.StringVar(value='analysis_plots')
        self.video_out_filename_var = tk.StringVar(value='tracked_droplets')
        self.pre_frame_filename_var = tk.StringVar(value='frame_preprocesado')
        self.bin_frame_filename_var = tk.StringVar(value='frame_binarizado')
        self.save_pre_frame_var = tk.BooleanVar(value=True)   # guardar frame preprocesado
        self.save_bin_frame_var = tk.BooleanVar(value=True)   # guardar frame binarizado
        
        # Crear interfaz
        self.create_widgets()
        
        # Estilo
        self.setup_styles()
    
    def setup_styles(self):
        """Configura estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colores
        style.configure('TFrame', background='#f0f0f0')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'), background='#f0f0f0')
        style.configure('TButton', font=('Arial', 10))
        style.configure('Action.TButton', font=('Arial', 10, 'bold'))
    
    def create_widgets(self):
        """Crea los widgets de la interfaz"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Analizador de Gotas", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # ============ SECCIÓN 1: Carga de video ============
        video_frame = ttk.LabelFrame(main_frame, text="1. Cargar Video", padding="10")
        video_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        video_frame.columnconfigure(1, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="No se ha cargado ningún video")
        self.video_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Button(video_frame, text="Seleccionar Video", 
                  command=self.load_video).grid(row=1, column=0, sticky=tk.W, padx=5)
        
        self.video_info_label = ttk.Label(video_frame, text="")
        self.video_info_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        # ============ SECCIÓN 2: Escala de calibración ============
        scale_frame = ttk.LabelFrame(main_frame, text="2. Calibración (píxel → distancia real)", 
                                     padding="10")
        scale_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        scale_frame.columnconfigure(1, weight=1)
        
        ttk.Label(scale_frame, text="Escala (unidad/px):").grid(row=0, column=0, sticky=tk.W, padx=5)
        scale_entry = ttk.Entry(scale_frame, textvariable=self.pixel_scale_var, width=12)
        scale_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(scale_frame, text="  Unidad:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(scale_frame, textvariable=self.scale_unit_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=5)
        ttk.Label(scale_frame, 
                  text="(Dejar en 0 para trabajar en píxeles. Ej: 0.325 µm/px)", 
                  foreground='gray').grid(row=0, column=4, sticky=tk.W, padx=10)
        
        # ============ SECCIÓN 3: Nombres de archivos de salida ============
        names_frame = ttk.LabelFrame(main_frame, text="3. Nombres de Archivos de Salida", 
                                     padding="10")
        names_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        for col in range(7):
            names_frame.columnconfigure(col, weight=1 if col % 3 == 1 else 0)
        
        # Fila 0: CSV y Gráficos (sin checkbox, se generan bajo demanda)
        ttk.Label(names_frame, text="CSV de datos:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(names_frame, textvariable=self.csv_filename_var, width=22).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Label(names_frame, text=".csv", foreground='gray').grid(row=0, column=2, sticky=tk.W)
        
        ttk.Label(names_frame, text="Gráficos:").grid(row=0, column=3, sticky=tk.W, padx=(15,5), pady=2)
        ttk.Entry(names_frame, textvariable=self.plots_filename_var, width=22).grid(row=0, column=4, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Label(names_frame, text=".png", foreground='gray').grid(row=0, column=5, sticky=tk.W)
        
        # Fila 1: Video rastreado (sin checkbox)
        ttk.Label(names_frame, text="Video rastreado:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(names_frame, textvariable=self.video_out_filename_var, width=22).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Label(names_frame, text=".avi", foreground='gray').grid(row=1, column=2, sticky=tk.W)
        
        # Fila 2: Frame preprocesado — con checkbox
        self.pre_frame_check = ttk.Checkbutton(
            names_frame, text="Frame preprocesado:", variable=self.save_pre_frame_var,
            command=self._toggle_pre_frame_entry
        )
        self.pre_frame_check.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.pre_frame_entry = ttk.Entry(names_frame, textvariable=self.pre_frame_filename_var, width=22)
        self.pre_frame_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Label(names_frame, text=".png", foreground='gray').grid(row=2, column=2, sticky=tk.W)
        
        # Fila 2 (col 3): Frame binarizado — con checkbox
        self.bin_frame_check = ttk.Checkbutton(
            names_frame, text="Frame binarizado:", variable=self.save_bin_frame_var,
            command=self._toggle_bin_frame_entry
        )
        self.bin_frame_check.grid(row=2, column=3, sticky=tk.W, padx=(15,5), pady=2)
        self.bin_frame_entry = ttk.Entry(names_frame, textvariable=self.bin_frame_filename_var, width=22)
        self.bin_frame_entry.grid(row=2, column=4, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Label(names_frame, text=".png", foreground='gray').grid(row=2, column=5, sticky=tk.W)
        
        # ============ SECCIÓN 4: Parámetros ============
        params_frame = ttk.LabelFrame(main_frame, text="4. Parámetros de Segmentación", 
                                     padding="10")
        params_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                         pady=5, padx=(0, 5))
        
        # Crear scrollable frame para parámetros
        canvas = tk.Canvas(params_frame, height=400)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Parámetros ajustables
        self.param_vars = {}
        self.create_parameter_controls(scrollable_frame)
        
        # ============ SECCIÓN 3: Control y Progreso ============
        control_frame = ttk.LabelFrame(main_frame, text="5. Procesamiento", padding="10")
        control_frame.grid(row=4, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        control_frame.columnconfigure(0, weight=1)
        
        # Botones de control
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        self.process_btn = ttk.Button(btn_frame, text="▶ Procesar Video", 
                                     command=self.process_video, 
                                     style='Action.TButton')
        self.process_btn.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.cancel_btn = ttk.Button(btn_frame, text="⏹ Cancelar", 
                                    command=self.cancel_processing,
                                    state=tk.DISABLED)
        self.cancel_btn.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Barra de progreso
        ttk.Label(control_frame, text="Progreso:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_label = ttk.Label(control_frame, text="")
        self.progress_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        
        # Estadísticas
        stats_subframe = ttk.LabelFrame(control_frame, text="Estadísticas", padding="10")
        stats_subframe.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        stats_subframe.rowconfigure(0, weight=1)
        
        self.stats_text = tk.Text(stats_subframe, height=15, width=40, 
                                 font=('Courier', 9), state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Botones de exportación
        export_frame = ttk.Frame(control_frame)
        export_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=10)
        export_frame.columnconfigure(0, weight=1)
        
        self.export_csv_btn = ttk.Button(export_frame, text="💾 Exportar CSV", 
                                        command=self.export_csv, state=tk.DISABLED)
        self.export_csv_btn.grid(row=0, column=0, pady=2, sticky=(tk.W, tk.E))
        
        self.generate_plots_btn = ttk.Button(export_frame, text="📊 Generar Gráficos", 
                                            command=self.generate_plots, state=tk.DISABLED)
        self.generate_plots_btn.grid(row=1, column=0, pady=2, sticky=(tk.W, tk.E))
        
        self.save_frames_btn = ttk.Button(export_frame, text="🔬 Guardar Frames Diagnóstico",
                                          command=self.save_diagnostic_frames)
        self.save_frames_btn.grid(row=2, column=0, pady=2, sticky=(tk.W, tk.E))
        
        self.open_folder_btn = ttk.Button(export_frame, text="📁 Abrir Carpeta Resultados", 
                                         command=self.open_output_folder)
        self.open_folder_btn.grid(row=3, column=0, pady=2, sticky=(tk.W, tk.E))
        
        # ============ SECCIÓN 6: Estado ============
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Estado:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Listo", foreground='green')
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=10)
    
    def create_parameter_controls(self, parent):
        """Crea controles para los parámetros"""
        
        params_info = [
            ('min_droplet_area', 'Área mínima de gota (px²)', 1, 500, 10),
            ('max_droplet_area', 'Área máxima de gota (px²)', 100, 10000, 5000),
            ('max_tracking_distance', 'Distancia máxima tracking (px)', 5, 200, 50),
            ('clahe_clip_limit', 'Contraste CLAHE', 0.5, 10.0, 2.0),
            ('denoise_strength', 'Fuerza reducción ruido', 1, 30, 10),
            ('adaptive_block_size', 'Tamaño bloque adaptativo', 3, 31, 11),
            ('adaptive_c', 'Constante adaptativa', -10, 10, 2),
            ('morphology_kernel_size', 'Tamaño kernel morfología', 1, 15, 3),
        ]
        
        for i, (param_name, label, min_val, max_val, default_val) in enumerate(params_info):
            # Frame para cada parámetro
            param_frame = ttk.Frame(parent)
            param_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
            param_frame.columnconfigure(1, weight=1)
            
            # Etiqueta
            ttk.Label(param_frame, text=label, width=30).grid(row=0, column=0, sticky=tk.W)
            
            # Variable
            if isinstance(default_val, float):
                var = tk.DoubleVar(value=default_val)
            else:
                var = tk.IntVar(value=default_val)
            self.param_vars[param_name] = var
            
            # Slider
            if param_name == 'adaptive_block_size':
                # Este debe ser impar
                scale = ttk.Scale(param_frame, from_=min_val, to=max_val, 
                                variable=var, orient=tk.HORIZONTAL,
                                command=lambda v, pn=param_name: self.validate_odd_parameter(pn))
            else:
                scale = ttk.Scale(param_frame, from_=min_val, to=max_val, 
                                variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            
            # Entry para valor
            entry = ttk.Entry(param_frame, textvariable=var, width=8)
            entry.grid(row=0, column=2, sticky=tk.W, padx=5)
            
            # Botón reset
            ttk.Button(param_frame, text="↺", width=3,
                      command=lambda v=var, d=default_val: v.set(d)).grid(row=0, column=3)
    
    def validate_odd_parameter(self, param_name):
        """Asegura que ciertos parámetros sean impares"""
        value = int(self.param_vars[param_name].get())
        if value % 2 == 0:
            value += 1
        self.param_vars[param_name].set(value)
    
    def load_video(self):
        """Carga un archivo de video"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Videos", "*.avi *.mp4 *.mov *.mkv"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if filepath:
            self.video_path = filepath
            filename = os.path.basename(filepath)
            self.video_label.config(text=f"Video: {filename}")
            
            # Crear analizador temporal para obtener info
            temp_analyzer = DropletAnalyzer(filepath, self.output_dir)
            info_text = (f"{temp_analyzer.width}x{temp_analyzer.height}, "
                        f"{temp_analyzer.fps:.1f} fps, "
                        f"{temp_analyzer.total_frames} frames")
            self.video_info_label.config(text=info_text)
            
            self.update_status("Video cargado", "green")
    
    def update_status(self, message, color="black"):
        """Actualiza el mensaje de estado"""
        self.status_label.config(text=message, foreground=color)
    
    def get_parameters_from_gui(self):
        """Obtiene los parámetros de la GUI"""
        params = {}
        for param_name, var in self.param_vars.items():
            params[param_name] = var.get()
        return params
    
    def process_video(self):
        """Procesa el video en un hilo separado"""
        if not self.video_path:
            messagebox.showwarning("Advertencia", "Primero debes cargar un video")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.update_status("Procesando...", "orange")
        
        # Crear analizador con parámetros actuales
        self.analyzer = DropletAnalyzer(self.video_path, self.output_dir)
        
        # Actualizar parámetros
        params = self.get_parameters_from_gui()
        for param_name, value in params.items():
            self.analyzer.set_parameter(param_name, value)
        
        # Configurar escala
        scale_val = self.pixel_scale_var.get()
        unit = self.scale_unit_var.get().strip() or 'px'
        self.analyzer.set_scale(scale_val if scale_val > 0 else None, unit)
        
        # Configurar callback de progreso
        self.analyzer.set_progress_callback(self.update_progress)
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=self._process_video_thread)
        thread.daemon = True
        thread.start()
    
    def _process_video_thread(self):
        """Hilo de procesamiento del video"""
        try:
            vid_name = self.video_out_filename_var.get().strip() or 'tracked_droplets'
            if not vid_name.endswith('.avi'):
                vid_name += '.avi'
            success = self.analyzer.process_video(save_visualization=True,
                                                   output_video_filename=vid_name)
            
            if success:
                # Actualizar UI en el hilo principal
                self.root.after(0, self._on_processing_complete)
            else:
                self.root.after(0, self._on_processing_cancelled)
                
        except Exception as e:
            self.root.after(0, lambda: self._on_processing_error(str(e)))
    
    def _on_processing_complete(self):
        """Callback cuando el procesamiento se completa"""
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.export_csv_btn.config(state=tk.NORMAL)
        self.generate_plots_btn.config(state=tk.NORMAL)
        
        # Mostrar estadísticas
        self.display_statistics()
        
        self.update_status("¡Procesamiento completado!", "green")
        messagebox.showinfo("Éxito", "El video ha sido procesado correctamente")
    
    def _on_processing_cancelled(self):
        """Callback cuando el procesamiento se cancela"""
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.update_status("Procesamiento cancelado", "red")
    
    def _on_processing_error(self, error_msg):
        """Callback cuando hay un error en el procesamiento"""
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.update_status("Error en procesamiento", "red")
        messagebox.showerror("Error", f"Error durante el procesamiento:\n{error_msg}")
    
    def cancel_processing(self):
        """Cancela el procesamiento"""
        if self.analyzer:
            self.analyzer.cancel_processing()
            self.update_status("Cancelando...", "orange")
    
    def update_progress(self, percentage, current_frame, total_frames):
        """Actualiza la barra de progreso"""
        self.progress_var.set(percentage)
        self.progress_label.config(
            text=f"Frame {current_frame}/{total_frames} ({percentage}%)"
        )
    
    def display_statistics(self):
        """Muestra estadísticas en el área de texto"""
        if not self.analyzer:
            return
        
        stats = self.analyzer.get_statistics()
        if not stats:
            return
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        text = "═══════════════════════════════\n"
        text += "     ESTADÍSTICAS\n"
        text += "═══════════════════════════════\n\n"
        text += f"Gotas rastreadas: {stats['total_tracks']}\n"
        text += f"Detecciones totales: {stats['total_detections']}\n\n"
        
        u = stats.get('unit', 'px')
        
        text += f"DIÁMETRO ({u}):\n"
        text += f"  Media:    {stats['diameter_mean']:.2f}\n"
        text += f"  Mediana:  {stats['diameter_median']:.2f}\n"
        text += f"  Desv.Est: {stats['diameter_std']:.2f}\n"
        text += f"  Min:      {stats['diameter_min']:.2f}\n"
        text += f"  Max:      {stats['diameter_max']:.2f}\n\n"
        
        text += f"ÁREA ({u}²):\n"
        text += f"  Media:    {stats['area_mean']:.2f}\n"
        text += f"  Mediana:  {stats['area_median']:.2f}\n"
        text += f"  Desv.Est: {stats['area_std']:.2f}\n\n"
        
        if 'velocity_mean' in stats:
            text += f"VELOCIDAD ({u}/s):\n"
            text += f"  Media:    {stats['velocity_mean']:.2f}\n"
            text += f"  Mediana:  {stats['velocity_median']:.2f}\n"
            text += f"  Desv.Est: {stats['velocity_std']:.2f}\n"
            text += f"  Min:      {stats['velocity_min']:.2f}\n"
            text += f"  Max:      {stats['velocity_max']:.2f}\n\n"
        
        text += "CIRCULARIDAD:\n"
        text += f"  Media:    {stats['circularity_mean']:.3f}\n"
        text += f"  Mediana:  {stats['circularity_median']:.3f}\n"
        
        self.stats_text.insert(1.0, text)
        self.stats_text.config(state=tk.DISABLED)
    
    def export_csv(self):
        """Exporta los resultados a CSV"""
        if not self.analyzer:
            return
        
        try:
            fname = self.csv_filename_var.get().strip() or 'droplet_data'
            df, csv_path = self.analyzer.export_results(filename=fname)
            messagebox.showinfo("Éxito", f"CSV exportado a:\n{csv_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar CSV:\n{str(e)}")
    
    def generate_plots(self):
        """Genera los gráficos"""
        if not self.analyzer:
            return
        
        try:
            df = self.analyzer.export_to_dataframe()
            self.plotter = DropletPlotter(self.output_dir)
            fname = self.plots_filename_var.get().strip() or 'analysis_plots'
            if not fname.endswith('.png'):
                fname += '.png'
            plot_path = self.plotter.generate_all_plots(df, filename=fname)
            messagebox.showinfo("Éxito", f"Gráficos generados en:\n{plot_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráficos:\n{str(e)}")
    
    def _toggle_pre_frame_entry(self):
        """Activa/desactiva el campo de nombre del frame preprocesado"""
        state = tk.NORMAL if self.save_pre_frame_var.get() else tk.DISABLED
        self.pre_frame_entry.config(state=state)
    
    def _toggle_bin_frame_entry(self):
        """Activa/desactiva el campo de nombre del frame binarizado"""
        state = tk.NORMAL if self.save_bin_frame_var.get() else tk.DISABLED
        self.bin_frame_entry.config(state=state)
    
    def save_diagnostic_frames(self):
        """Guarda el primer frame preprocesado y/o binarizado para diagnóstico"""
        if not self.video_path:
            messagebox.showwarning("Advertencia", "Primero debes cargar un video")
            return
        
        save_pre = self.save_pre_frame_var.get()
        save_bin = self.save_bin_frame_var.get()
        
        if not save_pre and not save_bin:
            messagebox.showwarning("Advertencia", 
                                   "Marca al menos una opción (preprocesado o binarizado)")
            return
        
        try:
            from droplet_analyzer import DropletAnalyzer
            diag_analyzer = DropletAnalyzer(self.video_path, self.output_dir)
            params = self.get_parameters_from_gui()
            for param_name, value in params.items():
                diag_analyzer.set_parameter(param_name, value)
            
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("No se pudo leer el primer frame del video")
            
            saved = []
            if save_pre:
                pre_name = (self.pre_frame_filename_var.get().strip() or 'frame_preprocesado') + '.png'
                preprocessed = diag_analyzer.preprocess_frame(frame)
                pre_path = os.path.join(self.output_dir, pre_name)
                cv2.imwrite(pre_path, preprocessed)
                saved.append(f"  Preprocesado: {pre_path}")
            
            if save_bin:
                bin_name = (self.bin_frame_filename_var.get().strip() or 'frame_binarizado') + '.png'
                binary, _ = diag_analyzer.segment_droplets(frame)
                bin_path = os.path.join(self.output_dir, bin_name)
                cv2.imwrite(bin_path, binary)
                saved.append(f"  Binarizado:   {bin_path}")
            
            messagebox.showinfo("Éxito", "Frames guardados:\n" + "\n".join(saved))
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar frames:\n{str(e)}") 
    
    def open_output_folder(self):
        """Abre la carpeta de salida, creándola si no existe"""
        import platform
        import subprocess
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if platform.system() == 'Windows':
            os.startfile(self.output_dir)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.Popen(['open', self.output_dir])
        else:  # Linux
            subprocess.Popen(['xdg-open', self.output_dir])


def main():
    """Función principal"""
    root = tk.Tk()
    app = DropletAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

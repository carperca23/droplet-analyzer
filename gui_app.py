"""
gui_app.py
Interfaz gráfica para el análisis de gotas
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import queue
import time

from droplet_analyzer import DropletAnalyzer
from plotter import DropletPlotter


class DropletAnalyzerGUI:
    """Interfaz gráfica para el análisis de gotas"""

    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Gotas - Droplet Analyzer")
        self.root.geometry("1350x900")

        # Variables de estado
        self.video_path = None
        self.analyzer = None
        self.plotter = None
        self.output_dir = 'output'
        self.processing = False

        self._log_queue = queue.Queue()
        self._preview_queue = queue.Queue(maxsize=2)
        self._preview_win = None

        self.pixel_scale_var = tk.DoubleVar(value=0.0)
        self.scale_unit_var = tk.StringVar(value='µm')
        self.fps_override_var = tk.DoubleVar(value=1_000_000.0)

        self.csv_filename_var = tk.StringVar(value='droplet_data')
        self.plots_filename_var = tk.StringVar(value='analysis_plots')
        self.video_out_filename_var = tk.StringVar(value='tracked_droplets')
        self.pre_frame_filename_var = tk.StringVar(value='frame_preprocesado')
        self.bin_frame_filename_var = tk.StringVar(value='frame_binarizado')
        self.save_pre_frame_var = tk.BooleanVar(value=True)
        self.save_bin_frame_var = tk.BooleanVar(value=True)
        self.show_tracking_lines_var = tk.BooleanVar(value=True)

        self.create_widgets()
        self.setup_styles()
        self.root.after(100, self._poll_log_queue)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'), background='#f0f0f0')
        style.configure('TButton', font=('Arial', 10))
        style.configure('Action.TButton', font=('Arial', 11, 'bold'))

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(5, weight=1)

        ttk.Label(main_frame, text="Analizador de Gotas",
                  style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 8))

        video_frame = ttk.LabelFrame(main_frame, text="1. Cargar Video", padding="8")
        video_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)
        video_frame.columnconfigure(1, weight=1)

        self.video_label = ttk.Label(video_frame, text="No se ha cargado ningún video")
        self.video_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=3)

        ttk.Button(video_frame, text="Seleccionar Video",
                   command=self.load_video).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.video_info_label = ttk.Label(video_frame, text="")
        self.video_info_label.grid(row=1, column=1, sticky=tk.W, padx=10)

        scale_frame = ttk.LabelFrame(main_frame, text="2. Calibración (píxel → distancia real)", padding="8")
        scale_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)

        ttk.Label(scale_frame, text="Escala (unidad/px):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(scale_frame, textvariable=self.pixel_scale_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(scale_frame, text="  Unidad:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(scale_frame, textvariable=self.scale_unit_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=5)
        ttk.Label(scale_frame, text="(0 = trabajar en píxeles · Ej: 0.325 µm/px)", foreground='gray').grid(row=0, column=4, sticky=tk.W, padx=10)

        ttk.Label(scale_frame, text="FPS reales adquisición:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(6, 2))
        ttk.Entry(scale_frame, textvariable=self.fps_override_var, width=16).grid(row=1, column=1, sticky=tk.W, padx=5, pady=(6, 2))
        ttk.Label(scale_frame, text="fps  (por defecto 1 000 000 · sobreescribe el FPS del archivo)", foreground='gray').grid(row=1, column=2, columnspan=3, sticky=tk.W, padx=5, pady=(6, 2))

        names_frame = ttk.LabelFrame(main_frame, text="3. Nombres de Archivos de Salida", padding="8")
        names_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)
        for col in range(6):
            names_frame.columnconfigure(col, weight=1 if col % 3 == 1 else 0)

        ttk.Label(names_frame, text="CSV datos:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(names_frame, textvariable=self.csv_filename_var, width=22).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)
        ttk.Label(names_frame, text=".csv", foreground='gray').grid(row=0, column=2, sticky=tk.W)

        ttk.Label(names_frame, text="Gráficos:").grid(row=0, column=3, sticky=tk.W, padx=(15, 5), pady=2)
        ttk.Entry(names_frame, textvariable=self.plots_filename_var, width=22).grid(row=0, column=4, sticky=(tk.W, tk.E), padx=2)
        ttk.Label(names_frame, text=".png", foreground='gray').grid(row=0, column=5, sticky=tk.W)

        ttk.Label(names_frame, text="Video rastreado:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(names_frame, textvariable=self.video_out_filename_var, width=22).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=2)
        ttk.Label(names_frame, text=".avi", foreground='gray').grid(row=1, column=2, sticky=tk.W)

        self.pre_frame_check = ttk.Checkbutton(names_frame, text="Frame preprocesado:", variable=self.save_pre_frame_var, command=self._toggle_pre_frame_entry)
        self.pre_frame_check.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.pre_frame_entry = ttk.Entry(names_frame, textvariable=self.pre_frame_filename_var, width=22)
        self.pre_frame_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=2)
        ttk.Label(names_frame, text=".png", foreground='gray').grid(row=2, column=2, sticky=tk.W)

        self.bin_frame_check = ttk.Checkbutton(names_frame, text="Frame binarizado:", variable=self.save_bin_frame_var, command=self._toggle_bin_frame_entry)
        self.bin_frame_check.grid(row=2, column=3, sticky=tk.W, padx=(15, 5), pady=2)
        self.bin_frame_entry = ttk.Entry(names_frame, textvariable=self.bin_frame_filename_var, width=22)
        self.bin_frame_entry.grid(row=2, column=4, sticky=(tk.W, tk.E), padx=2)
        ttk.Label(names_frame, text=".png", foreground='gray').grid(row=2, column=5, sticky=tk.W)

        params_frame = ttk.LabelFrame(main_frame, text="4. Parámetros de Segmentación", padding="8")
        params_frame.grid(row=4, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=3, padx=(0, 5))
        params_frame.columnconfigure(0, weight=1)
        params_frame.rowconfigure(0, weight=1)

        canvas = tk.Canvas(params_frame, height=350)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.param_vars = {}
        self.create_parameter_controls(scrollable_frame)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=4, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=3)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.rowconfigure(2, weight=1)

        action_frame = ttk.LabelFrame(right_frame, text="5. Procesamiento", padding="10")
        action_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

        self.run_btn = ttk.Button(action_frame, text="▶▶ ANALIZAR TODO", command=self.run_full_pipeline, style='Action.TButton')
        self.run_btn.grid(row=0, column=0, padx=5, pady=8, sticky=(tk.W, tk.E))

        self.cancel_btn = ttk.Button(action_frame, text="⏹ Cancelar", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_btn.grid(row=0, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3, padx=5)
        self.progress_label = ttk.Label(action_frame, text="")
        self.progress_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)

        ttk.Checkbutton(action_frame, text="🔵 Mostrar líneas de tracking en video", variable=self.show_tracking_lines_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=4)

        aux_frame = ttk.Frame(action_frame)
        aux_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=4)
        aux_frame.columnconfigure(0, weight=1)
        aux_frame.columnconfigure(1, weight=1)

        ttk.Button(aux_frame, text="🔬 Frames Diagnóstico", command=self.save_diagnostic_frames).grid(row=0, column=0, padx=3, sticky=(tk.W, tk.E))
        ttk.Button(aux_frame, text="📁 Abrir Resultados", command=self.open_output_folder).grid(row=0, column=1, padx=3, sticky=(tk.W, tk.E))

        log_frame = ttk.LabelFrame(right_frame, text="Log de procesamiento", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=10, width=46, font=('Courier', 8), state=tk.DISABLED, bg='#1a1a2e', fg='#00ff88', insertbackground='white')
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        stats_frame = ttk.LabelFrame(right_frame, text="Estadísticas", padding="5")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)

        self.stats_text = tk.Text(stats_frame, height=14, width=46, font=('Courier', 8), state=tk.DISABLED)
        stats_scroll = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Estado:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Listo", foreground='green')
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=10)

    def create_parameter_controls(self, parent):
        algo_frame = ttk.Frame(parent)
        algo_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        algo_frame.columnconfigure(1, weight=1)

        ttk.Label(algo_frame, text="Método de contraste:", width=32).grid(row=0, column=0, sticky=tk.W)
        self.algo_var = tk.StringVar(value="ImageJ (Global)")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var, values=["ImageJ (Global)", "CLAHE (Local)"], state="readonly")
        algo_combo.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        self.algo_var.trace_add("write", self._on_algo_change)

        # NUEVO: Toggle de Lucas-Kanade
        self.use_lk_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(algo_frame, text="Mejorar tracking con Lucas-Kanade", variable=self.use_lk_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        self.param_widgets = {}

        # NUEVO: Se añadió lk_window_size a la lista para controlar su tamaño
        params_info = [
            ('min_droplet_area',        'Área mínima gota (px²)',        1,    500,   10),
            ('max_droplet_area',        'Área máxima gota (px²)',       100,  10000, 5000),
            ('min_droplet_circularity', 'Circularidad mínima (0-1)',    0.0,    1.0,  0.6),
            ('min_droplet_roundness',   'Redondez mínima (0-1)',        0.0,    1.0,  0.0),
            ('max_tracking_distance',   'Dist. máxima tracking (px)',     5,    200,   50),
            ('lk_window_size',          'Tamaño ventana Lucas-Kanade',    5,     51,   15),
            ('clahe_clip_limit',        'Contraste CLAHE',              0.5,   10.0,  2.0),
            ('denoise_strength',        'Fuerza reducción ruido',         1,     30,   10),
            ('adaptive_block_size',     'Tamaño bloque adaptativo',       3,     31,   11),
            ('adaptive_c',              'Constante adaptativa',         -10,     10,    2),
            ('morphology_kernel_size',  'Tamaño kernel morfología',       1,     15,    3),
        ]

        for i, (param_name, label, min_val, max_val, default_val) in enumerate(params_info, start=1):
            param_frame = ttk.Frame(parent)
            param_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2, padx=5)
            param_frame.columnconfigure(1, weight=1)

            lbl = ttk.Label(param_frame, text=label, width=32)
            lbl.grid(row=0, column=0, sticky=tk.W)

            if isinstance(default_val, float):
                var = tk.DoubleVar(value=default_val)
            else:
                var = tk.IntVar(value=default_val)
            self.param_vars[param_name] = var

            cmd = None
            if param_name == 'adaptive_block_size':
                cmd = lambda v, pn=param_name: self.validate_odd_parameter(pn)

            scale = ttk.Scale(param_frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, command=cmd)
            scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

            entry = ttk.Entry(param_frame, textvariable=var, width=8)
            entry.grid(row=0, column=2, sticky=tk.W, padx=5)

            reset_btn = ttk.Button(param_frame, text="↺", width=3, command=lambda v=var, d=default_val: v.set(d))
            reset_btn.grid(row=0, column=3)

            self.param_widgets[param_name] = [lbl, scale, entry, reset_btn]

        ttk.Label(
            parent,
            text="ℹ️  Las áreas se introducen en px². Si hay escala configurada, el equivalente en µm² aparece en el log al analizar.",
            foreground='#555555', wraplength=400, justify=tk.LEFT
        ).grid(row=len(params_info) + 1, column=0, sticky=tk.W, padx=5, pady=4)

        self._on_algo_change()

    def _on_algo_change(self, *args):
        use_clahe = (self.algo_var.get() == "CLAHE (Local)")
        state = tk.NORMAL if use_clahe else tk.DISABLED
        for widget in self.param_widgets.get('clahe_clip_limit', []):
            try:
                widget.config(state=state)
            except Exception:
                pass

    def validate_odd_parameter(self, param_name):
        value = int(self.param_vars[param_name].get())
        if value % 2 == 0:
            value += 1
        self.param_vars[param_name].set(value)

    def get_parameters_from_gui(self):
        params = {}
        for param_name, var in self.param_vars.items():
            params[param_name] = var.get()
        params['use_clahe'] = (self.algo_var.get() == "CLAHE (Local)")
        params['use_lucas_kanade'] = self.use_lk_var.get() # NUEVO: Leer toggle de LK de la GUI
        return params

    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Videos", "*.avi *.mp4 *.mov *.mkv"), ("Todos los archivos", "*.*")]
        )
        if filepath:
            self.video_path = filepath
            filename = os.path.basename(filepath)
            self.video_label.config(text=f"Video: {filename}")
            temp = DropletAnalyzer(filepath, self.output_dir)
            self.video_info_label.config(text=(f"{temp.width}x{temp.height}, {temp.fps:.1f} fps, {temp.total_frames} frames"))
            self.update_status("Video cargado", "green")
            self.log(f"Video cargado: {filename}")
            self.log(f"  {temp.width}x{temp.height} · {temp.fps:.1f} fps · {temp.total_frames} frames")

    def run_full_pipeline(self):
        if not self.video_path:
            messagebox.showwarning("Advertencia", "Primero debes cargar un video")
            return
        if self.processing:
            return

        self.processing = True
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="")
        self.update_status("Procesando...", "orange")
        self.log("=" * 42)
        self.log("  INICIANDO ANÁLISIS COMPLETO")
        self.log("=" * 42)

        self.analyzer = DropletAnalyzer(self.video_path, self.output_dir)
        params = self.get_parameters_from_gui()
        for pname, val in params.items():
            self.analyzer.set_parameter(pname, val)

        scale_val = self.pixel_scale_var.get()
        unit = self.scale_unit_var.get().strip() or 'px'
        self.analyzer.set_scale(scale_val if scale_val > 0 else None, unit)

        fps_val = self.fps_override_var.get()
        self.analyzer.set_fps(fps_val if fps_val > 0 else None)
        self.log(f"FPS efectivos: {self.analyzer.effective_fps:,.0f} fps (archivo: {self.analyzer.fps:.2f} fps)")

        if scale_val > 0:
            min_area_px = self.param_vars['min_droplet_area'].get()
            max_area_px = self.param_vars['max_droplet_area'].get()
            factor = scale_val ** 2
            self.log(f"Escala: {scale_val} {unit}/px  →  1 px² = {factor:.4f} {unit}²")
            self.log(f"Área mín: {min_area_px:.0f} px² = {min_area_px * factor:.2f} {unit}²")
            self.log(f"Área máx: {max_area_px:.0f} px² = {max_area_px * factor:.2f} {unit}²")

        self.analyzer.set_progress_callback(self.update_progress)
        self._open_preview_window()
        
        thread = threading.Thread(target=self._full_pipeline_thread, daemon=True)
        thread.start()

    def _full_pipeline_thread(self):
        try:
            self._log_from_thread("► Paso 1/4: Procesando video...")
            vid_name = self.video_out_filename_var.get().strip() or 'tracked_droplets'
            if not vid_name.endswith('.avi'):
                vid_name += '.avi'
            show_lines = self.show_tracking_lines_var.get()
            
            # NUEVO: Usar el flag de Lucas-Kanade del GUI para el log
            use_lk = self.get_parameters_from_gui().get('use_lucas_kanade', True)
            self._log_from_thread(f"   Modo Tracking: {'Lucas-Kanade + Distancia' if use_lk else 'Solo Distancia Euclidean'}")
            
            success = self._process_video_with_lines(vid_name, show_lines)
            if not success:
                self.root.after(0, self._on_processing_cancelled)
                return

            self._log_from_thread("► Paso 2/4: Exportando CSV de datos...")
            fname = self.csv_filename_var.get().strip() or 'droplet_data'
            df, csv_path = self.analyzer.export_results(filename=fname)
            self._log_from_thread(f"   ✓ CSV guardado: {csv_path}")

            self._log_from_thread("► Paso 3/4: Generando gráficos y CSVs individuales...")
            self.plotter = DropletPlotter(self.output_dir)
            pfname = self.plots_filename_var.get().strip() or 'analysis_plots'
            if not pfname.endswith('.png'):
                pfname += '.png'
            plot_path = self.plotter.generate_all_plots(df, filename=pfname)
            self._log_from_thread(f"   ✓ Gráficos: {plot_path}")

            base = self.csv_filename_var.get().strip() or 'droplet_data'
            exported = self.plotter.export_individual_csvs(df, base_name=base)
            for name, path in exported.items():
                self._log_from_thread(f"   ✓ CSV {name}: {os.path.basename(path)}")

            self._log_from_thread("► Paso 4/4: Guardando frames de diagnóstico...")
            self._save_diag_frames_internal()

            self.root.after(0, self._on_pipeline_complete)

        except Exception as e:
            import traceback
            self._log_from_thread(f"ERROR: {e}")
            self._log_from_thread(traceback.format_exc())
            self.root.after(0, lambda err=str(e): self._on_processing_error(err))

    def _process_video_with_lines(self, vid_name, show_lines):
        import cv2
        import numpy as np

        self.analyzer.reset()
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            os.path.join(self.output_dir, vid_name),
            fourcc, self.analyzer.fps,
            (self.analyzer.width, self.analyzer.height)
        )

        PREVIEW_EVERY = max(1, int(self.analyzer.fps / 10))
        frame_idx = 0
        
        while True:
            if self.analyzer.cancel_requested:
                cap.release()
                out.release()
                return False

            ret, frame = cap.read()
            if not ret:
                break

            preprocessed = self.analyzer.preprocess_frame(frame)
            binary, droplets = self.analyzer.segment_droplets(frame)
            
            # NUEVO: Para Lucas Kanade, pasamos el frame en escala de grises
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            self.analyzer.track_droplets(droplets, frame_idx, gray_frame)

            vis = (cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy())

            for d in droplets:
                cx, cy = d['centroid']
                cv2.circle(vis, (int(cx), int(cy)), int(d['diameter'] / 2), (0, 255, 0), 1)
                cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), -1)

            if show_lines:
                for track in self.analyzer.droplet_tracks:
                    if len(track['positions']) > 1 and track['frames'][-1] >= frame_idx - 10:
                        pts = np.array(track['positions'], dtype=np.int32)
                        cv2.polylines(vis, [pts], False, (255, 0, 0), 1)

            out.write(vis)

            if frame_idx % PREVIEW_EVERY == 0:
                orig_bgr = (cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy())
                pre_bgr  = (cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR) if preprocessed.ndim == 2 else preprocessed.copy())
                bin_bgr  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                self._push_preview_frames(orig_bgr, pre_bgr, bin_bgr, vis)

            if self.analyzer.progress_callback:
                pct = int((frame_idx + 1) / self.analyzer.total_frames * 100)
                self.analyzer.progress_callback(pct, frame_idx + 1, self.analyzer.total_frames)

            frame_idx += 1

        cap.release()
        out.release()
        self.analyzer.calculate_velocities()
        return True

    def _save_diag_frames_internal(self):
        import cv2
        save_pre = self.save_pre_frame_var.get()
        save_bin = self.save_bin_frame_var.get()
        if not save_pre and not save_bin:
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self._log_from_thread("   ⚠ No se pudo leer el primer frame para diagnóstico")
                return
            if save_pre:
                name = (self.pre_frame_filename_var.get().strip() or 'frame_preprocesado') + '.png'
                pre = self.analyzer.preprocess_frame(frame)
                path = os.path.join(self.output_dir, name)
                cv2.imwrite(path, pre)
                self._log_from_thread(f"   ✓ Frame preprocesado: {path}")
            if save_bin:
                name = (self.bin_frame_filename_var.get().strip() or 'frame_binarizado') + '.png'
                binary, _ = self.analyzer.segment_droplets(frame)
                path = os.path.join(self.output_dir, name)
                cv2.imwrite(path, binary)
                self._log_from_thread(f"   ✓ Frame binarizado: {path}")
        except Exception as e:
            self._log_from_thread(f"   ⚠ Error frames diagnóstico: {e}")

    def _on_pipeline_complete(self):
        self.processing = False
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.display_statistics()
        self.update_status("Análisis completado", "green")
        self.log("=" * 42)
        self.log("  ANÁLISIS COMPLETADO EXITOSAMENTE")
        self.log(f"  Resultados en: {os.path.abspath(self.output_dir)}")
        self.log("=" * 42)

    def _on_processing_cancelled(self):
        self.processing = False
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.update_status("Procesamiento cancelado", "red")
        self.log("Procesamiento cancelado por el usuario")

    def _on_processing_error(self, error_msg):
        self.processing = False
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.update_status("Error en procesamiento", "red")
        messagebox.showerror("Error", f"Error durante el procesamiento:\n{error_msg}")

    def cancel_processing(self):
        if self.analyzer:
            self.analyzer.cancel_processing()
            self.update_status("Cancelando...", "orange")
            self.log("Cancelación solicitada...")

    def update_progress(self, percentage, current_frame, total_frames):
        self.progress_var.set(percentage)
        self.progress_label.config(text=f"Frame {current_frame}/{total_frames} ({percentage}%)")
        if percentage % 10 == 0:
            self._log_from_thread(f"   Progreso video: {percentage}%  ({current_frame}/{total_frames})")

    def _open_preview_window(self):
        if self._preview_win and tk.Toplevel.winfo_exists(self._preview_win):
            self._preview_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Previsualización en tiempo real")
        win.resizable(True, True)
        self._preview_win = win

        PANEL_LABELS = ["Original", "Preprocesado", "Binarizado", "Tracking + Detecciones"]
        self._preview_labels = []
        self._preview_photo  = [None] * 4

        for col, text in enumerate(PANEL_LABELS):
            ttk.Label(win, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=4, pady=(6, 2))
            lbl = tk.Label(win, bg="black")
            lbl.grid(row=1, column=col, padx=4, pady=(0, 6))
            self._preview_labels.append(lbl)

        self.root.after(30, self._poll_preview_queue)

    def _poll_preview_queue(self):
        if not (self._preview_win and tk.Toplevel.winfo_exists(self._preview_win)):
            while not self._preview_queue.empty():
                try:
                    self._preview_queue.get_nowait()
                except queue.Empty:
                    break
            return

        try:
            frames = self._preview_queue.get_nowait()
            self._render_preview_frames(frames)
        except queue.Empty:
            pass

        if self.processing:
            self.root.after(30, self._poll_preview_queue)

    def _render_preview_frames(self, frames):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        import numpy as np

        THUMB_W, THUMB_H = 320, 240

        for idx, (frame, lbl) in enumerate(zip(frames, self._preview_labels)):
            if frame is None:
                continue
            img = frame.astype("uint8")
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            else:
                img = img[:, :, ::-1]

            h, w = img.shape[:2]
            scale = min(THUMB_W / w, THUMB_H / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            pil_img = Image.fromarray(img).resize((nw, nh), Image.NEAREST)
            photo = ImageTk.PhotoImage(pil_img)
            self._preview_photo[idx] = photo
            lbl.config(image=photo, width=nw, height=nh)

    def _push_preview_frames(self, orig, preprocessed, binary, tracking):
        try:
            self._preview_queue.put_nowait([orig, preprocessed, binary, tracking])
        except queue.Full:
            pass

    def log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _log_from_thread(self, msg: str):
        self._log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def display_statistics(self):
        if not self.analyzer:
            return
        stats = self.analyzer.get_statistics()
        if not stats:
            return

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        u = stats.get('unit', 'px')
        t = "═══════════════════════════════\n"
        t += "       ESTADÍSTICAS\n"
        t += "═══════════════════════════════\n\n"
        t += f"Gotas rastreadas:    {stats['total_tracks']}\n"
        t += f"Detecciones totales: {stats['total_detections']}\n\n"
        t += f"DIÁMETRO ({u}):\n"
        t += f"  Media:    {stats['diameter_mean']:.2f}\n"
        t += f"  Mediana:  {stats['diameter_median']:.2f}\n"
        t += f"  Desv.Est: {stats['diameter_std']:.2f}\n"
        t += f"  Min:      {stats['diameter_min']:.2f}\n"
        t += f"  Max:      {stats['diameter_max']:.2f}\n\n"
        t += f"ÁREA ({u}²):\n"
        t += f"  Media:    {stats['area_mean']:.2f}\n"
        t += f"  Mediana:  {stats['area_median']:.2f}\n"
        t += f"  Desv.Est: {stats['area_std']:.2f}\n\n"
        if 'velocity_mean' in stats:
            t += f"VELOCIDAD ({u}/s):\n"
            t += f"  Media:    {stats['velocity_mean']:.2f}\n"
            t += f"  Mediana:  {stats['velocity_median']:.2f}\n"
            t += f"  Desv.Est: {stats['velocity_std']:.2f}\n"
            t += f"  Min:      {stats['velocity_min']:.2f}\n"
            t += f"  Max:      {stats['velocity_max']:.2f}\n\n"
        t += "CIRCULARIDAD:\n"
        t += f"  Media:    {stats['circularity_mean']:.3f}\n"
        t += f"  Mediana:  {stats['circularity_median']:.3f}\n\n"
        if stats.get('roundness_mean') is not None:
            t += "REDONDEZ:\n"
            t += f"  Media:    {stats['roundness_mean']:.3f}\n"
            t += f"  Mediana:  {stats['roundness_median']:.3f}\n"

        self.stats_text.insert(1.0, t)
        self.stats_text.config(state=tk.DISABLED)

    def save_diagnostic_frames(self):
        if not self.video_path:
            messagebox.showwarning("Advertencia", "Primero debes cargar un video")
            return
        save_pre = self.save_pre_frame_var.get()
        save_bin = self.save_bin_frame_var.get()
        if not save_pre and not save_bin:
            messagebox.showwarning("Advertencia", "Marca al menos una opción")
            return
        try:
            import cv2
            diag = DropletAnalyzer(self.video_path, self.output_dir)
            for pname, val in self.get_parameters_from_gui().items():
                diag.set_parameter(pname, val)
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("No se pudo leer el primer frame")
            saved = []
            if save_pre:
                name = (self.pre_frame_filename_var.get().strip() or 'frame_preprocesado') + '.png'
                pre = diag.preprocess_frame(frame)
                path = os.path.join(self.output_dir, name)
                cv2.imwrite(path, pre)
                saved.append(f"  Preprocesado: {path}")
                self.log(f"Frame preprocesado: {path}")
            if save_bin:
                name = (self.bin_frame_filename_var.get().strip() or 'frame_binarizado') + '.png'
                binary, _ = diag.segment_droplets(frame)
                path = os.path.join(self.output_dir, name)
                cv2.imwrite(path, binary)
                saved.append(f"  Binarizado: {path}")
                self.log(f"Frame binarizado: {path}")
            messagebox.showinfo("Éxito", "Frames guardados:\n" + "\n".join(saved))
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar frames:\n{str(e)}")

    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)

    def _toggle_pre_frame_entry(self):
        state = tk.NORMAL if self.save_pre_frame_var.get() else tk.DISABLED
        self.pre_frame_entry.config(state=state)

    def _toggle_bin_frame_entry(self):
        state = tk.NORMAL if self.save_bin_frame_var.get() else tk.DISABLED
        self.bin_frame_entry.config(state=state)

    def open_output_folder(self):
        import platform
        import subprocess
        os.makedirs(self.output_dir, exist_ok=True)
        if platform.system() == 'Windows':
            os.startfile(self.output_dir)
        elif platform.system() == 'Darwin':
            subprocess.Popen(['open', self.output_dir])
        else:
            subprocess.Popen(['xdg-open', self.output_dir])


def main():
    root = tk.Tk()
    app = DropletAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
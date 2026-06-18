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

# Diccionario de traducciones para soportar cambio en caliente
LANG_DICT = {
    'es': {
        'title': "Analizador de Gotas - Droplet Analyzer",
        'step1': "1. Cargar Video",
        'no_vid': "No se ha cargado ningún video",
        'btn_sel_vid': "Seleccionar Video",
        'step2': "2. Calibración (píxel → distancia real)",
        'scale_lbl': "Escala (unidad/px):",
        'unit_lbl': "  Unidad:",
        'scale_hint': "(0 = trabajar en píxeles · Ej: 0.325 µm/px)",
        'fps_lbl': "FPS reales adquisición:",
        'fps_hint': "fps  (por defecto 1 000 000 · sobreescribe el archivo)",
        'step3': "3. Nombres de Archivos de Salida",
        'csv_lbl': "CSV datos:",
        'plots_lbl': "Gráficos:",
        'vid_out_lbl': "Video procesado:",
        'pre_frame_chk': "Frame preprocesado:",
        'bin_frame_chk': "Frame binarizado:",
        'step4': "4. Parámetros de Procesamiento",
        'method_lbl': "Método de contraste:",
        'use_lk_chk': "Mejorar tracking con Lucas-Kanade",
        'param_hint': "ℹ️  Las áreas se introducen en px². Si hay escala, el equivalente aparece en el log.",
        'step5': "5. Procesamiento",
        'btn_run': "▶▶ ANALIZAR TODO",
        'btn_cancel': "⏹ Cancelar",
        'chk_lines': "🔵 Mostrar líneas de tracking en video",
        'chk_diam': "🟢 Mostrar diámetro equivalente",
        'chk_perim': "🟠 Mostrar perímetro detectado",
        'btn_diag': "🔬 Frames Diagnóstico",
        'btn_open': "📁 Abrir Resultados",
        'log_frame': "Log de procesamiento",
        'stats_frame': "Estadísticas",
        'status_lbl': "Estado:",
        'status_ready': "Listo",
        
        # Logs y mensajes
        'msg_warn_vid': "Primero debes cargar un video",
        'log_vid_loaded': "Video cargado: {}",
        'log_cancel': "Procesamiento cancelado por el usuario",
        'msg_err': "Error durante el procesamiento:\n{}",
        'log_p1': "► Paso 1/4: Procesando video...",
        'log_p2': "► Paso 2/4: Exportando CSV de datos...",
        'log_p3': "► Paso 3/4: Generando gráficos y CSVs individuales...",
        'log_p4': "► Paso 4/4: Guardando frames de diagnóstico...",
        'log_done': "ANÁLISIS COMPLETADO EXITOSAMENTE",
        'stat_tracks': "Gotas rastreadas:",
        'stat_dets': "Detecciones totales:",
        'stat_diam': "DIÁMETRO",
        'stat_area': "ÁREA",
        'stat_vel': "VELOCIDAD",
        'stat_mean': "Media:",
        'stat_median': "Mediana:",
        'stat_std': "Desv.Est:",
    },
    'en': {
        'title': "Droplet Analyzer",
        'step1': "1. Load Video",
        'no_vid': "No video loaded",
        'btn_sel_vid': "Select Video",
        'step2': "2. Calibration (pixel → real distance)",
        'scale_lbl': "Scale (unit/px):",
        'unit_lbl': "  Unit:",
        'scale_hint': "(0 = work in pixels · Ex: 0.325 µm/px)",
        'fps_lbl': "Real acquisition FPS:",
        'fps_hint': "fps  (default 1,000,000 · overrides file FPS)",
        'step3': "3. Output File Names",
        'csv_lbl': "Data CSV:",
        'plots_lbl': "Plots:",
        'vid_out_lbl': "Processed Video:",
        'pre_frame_chk': "Preprocessed Frame:",
        'bin_frame_chk': "Binarized Frame:",
        'step4': "4. Processing Parameters",
        'method_lbl': "Contrast Method:",
        'use_lk_chk': "Improve tracking with Lucas-Kanade",
        'param_hint': "ℹ️  Areas are in px². If scale is set, the equivalent will be logged.",
        'step5': "5. Processing",
        'btn_run': "▶▶ RUN ANALYSIS",
        'btn_cancel': "⏹ Cancel",
        'chk_lines': "🔵 Show tracking lines on video",
        'chk_diam': "🟢 Show equivalent diameter",
        'chk_perim': "🟠 Show detected perimeter",
        'btn_diag': "🔬 Diagnostic Frames",
        'btn_open': "📁 Open Results",
        'log_frame': "Processing Log",
        'stats_frame': "Statistics",
        'status_lbl': "Status:",
        'status_ready': "Ready",
        
        # Logs y mensajes
        'msg_warn_vid': "You must load a video first",
        'log_vid_loaded': "Video loaded: {}",
        'log_cancel': "Processing cancelled by user",
        'msg_err': "Error during processing:\n{}",
        'log_p1': "► Step 1/4: Processing video...",
        'log_p2': "► Step 2/4: Exporting data CSV...",
        'log_p3': "► Step 3/4: Generating plots and sub-CSVs...",
        'log_p4': "► Step 4/4: Saving diagnostic frames...",
        'log_done': "ANALYSIS SUCCESSFULLY COMPLETED",
        'stat_tracks': "Tracked droplets:",
        'stat_dets': "Total detections:",
        'stat_diam': "DIAMETER",
        'stat_area': "AREA",
        'stat_vel': "VELOCITY",
        'stat_mean': "Mean:",
        'stat_median': "Median:",
        'stat_std': "Std.Dev:",
    }
}

PARAMS_ES = [
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

PARAMS_EN = [
    ('min_droplet_area',        'Min droplet area (px²)',        1,    500,   10),
    ('max_droplet_area',        'Max droplet area (px²)',       100,  10000, 5000),
    ('min_droplet_circularity', 'Min circularity (0-1)',    0.0,    1.0,  0.6),
    ('min_droplet_roundness',   'Min roundness (0-1)',        0.0,    1.0,  0.0),
    ('max_tracking_distance',   'Max tracking distance (px)',     5,    200,   50),
    ('lk_window_size',          'Lucas-Kanade window size',    5,     51,   15),
    ('clahe_clip_limit',        'CLAHE Contrast',              0.5,   10.0,  2.0),
    ('denoise_strength',        'Denoise strength',         1,     30,   10),
    ('adaptive_block_size',     'Adaptive block size',       3,     31,   11),
    ('adaptive_c',              'Adaptive constant',         -10,     10,    2),
    ('morphology_kernel_size',  'Morphology kernel size',       1,     15,    3),
]

class DropletAnalyzerGUI:
    """Interfaz gráfica para el análisis de gotas"""

    def __init__(self, root):
        self.root = root
        self.root.geometry("1350x900")
        
        self.lang_var = tk.StringVar(value='es')
        self.widgets = {} # Referencias a widgets para traducción

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
        
        # Opciones de visualización
        self.show_tracking_lines_var = tk.BooleanVar(value=True)
        self.show_diameter_var = tk.BooleanVar(value=True)
        self.show_perimeter_var = tk.BooleanVar(value=False)

        self.setup_styles()
        self.create_widgets()
        self.change_language() # Forzar configuración inicial de textos
        self.root.after(100, self._poll_log_queue)

    def _t(self, key):
        return LANG_DICT[self.lang_var.get()][key]

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

        # Header con Selector de Idioma
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.widgets['title'] = ttk.Label(header_frame, style='Title.TLabel')
        self.widgets['title'].pack(side=tk.LEFT)
        
        lang_frame = ttk.Frame(header_frame)
        lang_frame.pack(side=tk.RIGHT)
        ttk.Radiobutton(lang_frame, text="ES", variable=self.lang_var, value='es', command=self.change_language).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(lang_frame, text="EN", variable=self.lang_var, value='en', command=self.change_language).pack(side=tk.LEFT, padx=2)

        # 1. Cargar Video
        self.widgets['lf_step1'] = ttk.LabelFrame(main_frame, padding="8")
        self.widgets['lf_step1'].grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)
        self.widgets['lf_step1'].columnconfigure(1, weight=1)

        self.widgets['lbl_no_vid'] = ttk.Label(self.widgets['lf_step1'])
        self.widgets['lbl_no_vid'].grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=3)

        self.widgets['btn_sel_vid'] = ttk.Button(self.widgets['lf_step1'], command=self.load_video)
        self.widgets['btn_sel_vid'].grid(row=1, column=0, sticky=tk.W, padx=5)
        
        self.video_info_label = ttk.Label(self.widgets['lf_step1'], text="")
        self.video_info_label.grid(row=1, column=1, sticky=tk.W, padx=10)

        # 2. Calibración
        self.widgets['lf_step2'] = ttk.LabelFrame(main_frame, padding="8")
        self.widgets['lf_step2'].grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)

        self.widgets['lbl_scale'] = ttk.Label(self.widgets['lf_step2'])
        self.widgets['lbl_scale'].grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.widgets['lf_step2'], textvariable=self.pixel_scale_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=5)
        self.widgets['lbl_unit'] = ttk.Label(self.widgets['lf_step2'])
        self.widgets['lbl_unit'].grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(self.widgets['lf_step2'], textvariable=self.scale_unit_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=5)
        self.widgets['lbl_scale_hint'] = ttk.Label(self.widgets['lf_step2'], foreground='gray')
        self.widgets['lbl_scale_hint'].grid(row=0, column=4, sticky=tk.W, padx=10)

        self.widgets['lbl_fps'] = ttk.Label(self.widgets['lf_step2'])
        self.widgets['lbl_fps'].grid(row=1, column=0, sticky=tk.W, padx=5, pady=(6, 2))
        ttk.Entry(self.widgets['lf_step2'], textvariable=self.fps_override_var, width=16).grid(row=1, column=1, sticky=tk.W, padx=5, pady=(6, 2))
        self.widgets['lbl_fps_hint'] = ttk.Label(self.widgets['lf_step2'], foreground='gray')
        self.widgets['lbl_fps_hint'].grid(row=1, column=2, columnspan=3, sticky=tk.W, padx=5, pady=(6, 2))

        # 3. Nombres
        self.widgets['lf_step3'] = ttk.LabelFrame(main_frame, padding="8")
        self.widgets['lf_step3'].grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)

        self.widgets['lbl_csv'] = ttk.Label(self.widgets['lf_step3'])
        self.widgets['lbl_csv'].grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(self.widgets['lf_step3'], textvariable=self.csv_filename_var, width=22).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)

        self.widgets['lbl_plots'] = ttk.Label(self.widgets['lf_step3'])
        self.widgets['lbl_plots'].grid(row=0, column=3, sticky=tk.W, padx=(15, 5), pady=2)
        ttk.Entry(self.widgets['lf_step3'], textvariable=self.plots_filename_var, width=22).grid(row=0, column=4, sticky=(tk.W, tk.E), padx=2)

        self.widgets['lbl_vid_out'] = ttk.Label(self.widgets['lf_step3'])
        self.widgets['lbl_vid_out'].grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(self.widgets['lf_step3'], textvariable=self.video_out_filename_var, width=22).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=2)

        self.widgets['chk_pre_frame'] = ttk.Checkbutton(self.widgets['lf_step3'], variable=self.save_pre_frame_var, command=self._toggle_pre_frame_entry)
        self.widgets['chk_pre_frame'].grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.pre_frame_entry = ttk.Entry(self.widgets['lf_step3'], textvariable=self.pre_frame_filename_var, width=22)
        self.pre_frame_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=2)

        self.widgets['chk_bin_frame'] = ttk.Checkbutton(self.widgets['lf_step3'], variable=self.save_bin_frame_var, command=self._toggle_bin_frame_entry)
        self.widgets['chk_bin_frame'].grid(row=2, column=3, sticky=tk.W, padx=(15, 5), pady=2)
        self.bin_frame_entry = ttk.Entry(self.widgets['lf_step3'], textvariable=self.bin_frame_filename_var, width=22)
        self.bin_frame_entry.grid(row=2, column=4, sticky=(tk.W, tk.E), padx=2)

        # 4. Parámetros
        self.widgets['lf_step4'] = ttk.LabelFrame(main_frame, padding="8")
        self.widgets['lf_step4'].grid(row=4, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=3, padx=(0, 5))
        self.widgets['lf_step4'].columnconfigure(0, weight=1)
        self.widgets['lf_step4'].rowconfigure(0, weight=1)

        canvas = tk.Canvas(self.widgets['lf_step4'], height=350)
        scrollbar = ttk.Scrollbar(self.widgets['lf_step4'], orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Controles de parámetros creados en hot-swap
        self.param_vars = {}
        self.param_labels = [] 
        self.algo_var = tk.StringVar(value="ImageJ (Global)")
        self.use_lk_var = tk.BooleanVar(value=True)

        # 5. Acciones
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=4, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=3)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.rowconfigure(2, weight=1)

        self.widgets['lf_step5'] = ttk.LabelFrame(right_frame, padding="10")
        self.widgets['lf_step5'].grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.widgets['lf_step5'].columnconfigure(0, weight=1)
        self.widgets['lf_step5'].columnconfigure(1, weight=1)

        self.run_btn = ttk.Button(self.widgets['lf_step5'], command=self.run_full_pipeline, style='Action.TButton')
        self.run_btn.grid(row=0, column=0, padx=5, pady=8, sticky=(tk.W, tk.E))
        self.widgets['btn_run'] = self.run_btn

        self.cancel_btn = ttk.Button(self.widgets['lf_step5'], command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_btn.grid(row=0, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
        self.widgets['btn_cancel'] = self.cancel_btn

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.widgets['lf_step5'], variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3, padx=5)
        self.progress_label = ttk.Label(self.widgets['lf_step5'], text="")
        self.progress_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)

        self.widgets['chk_lines'] = ttk.Checkbutton(self.widgets['lf_step5'], variable=self.show_tracking_lines_var)
        self.widgets['chk_lines'].grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        self.widgets['chk_diam'] = ttk.Checkbutton(self.widgets['lf_step5'], variable=self.show_diameter_var)
        self.widgets['chk_diam'].grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.widgets['chk_perim'] = ttk.Checkbutton(self.widgets['lf_step5'], variable=self.show_perimeter_var)
        self.widgets['chk_perim'].grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        aux_frame = ttk.Frame(self.widgets['lf_step5'])
        aux_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=4)
        aux_frame.columnconfigure(0, weight=1)
        aux_frame.columnconfigure(1, weight=1)

        self.widgets['btn_diag'] = ttk.Button(aux_frame, command=self.save_diagnostic_frames)
        self.widgets['btn_diag'].grid(row=0, column=0, padx=3, sticky=(tk.W, tk.E))
        self.widgets['btn_open'] = ttk.Button(aux_frame, command=self.open_output_folder)
        self.widgets['btn_open'].grid(row=0, column=1, padx=3, sticky=(tk.W, tk.E))

        self.widgets['lf_log'] = ttk.LabelFrame(right_frame, padding="5")
        self.widgets['lf_log'].grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        self.widgets['lf_log'].columnconfigure(0, weight=1)
        self.widgets['lf_log'].rowconfigure(0, weight=1)

        self.log_text = tk.Text(self.widgets['lf_log'], height=10, width=46, font=('Courier', 8), state=tk.DISABLED, bg='#1a1a2e', fg='#00ff88', insertbackground='white')
        log_scroll = ttk.Scrollbar(self.widgets['lf_log'], orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.widgets['lf_stats'] = ttk.LabelFrame(right_frame, padding="5")
        self.widgets['lf_stats'].grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.widgets['lf_stats'].columnconfigure(0, weight=1)
        self.widgets['lf_stats'].rowconfigure(0, weight=1)

        self.stats_text = tk.Text(self.widgets['lf_stats'], height=14, width=46, font=('Courier', 8), state=tk.DISABLED)
        stats_scroll = ttk.Scrollbar(self.widgets['lf_stats'], orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)
        status_frame.columnconfigure(1, weight=1)
        self.widgets['lbl_status'] = ttk.Label(status_frame)
        self.widgets['lbl_status'].grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, foreground='green')
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=10)

    def change_language(self):
        self.root.title(self._t('title'))
        self.widgets['title'].config(text=self._t('title'))
        self.widgets['lf_step1'].config(text=self._t('step1'))
        if not self.video_path:
            self.widgets['lbl_no_vid'].config(text=self._t('no_vid'))
        self.widgets['btn_sel_vid'].config(text=self._t('btn_sel_vid'))
        
        self.widgets['lf_step2'].config(text=self._t('step2'))
        self.widgets['lbl_scale'].config(text=self._t('scale_lbl'))
        self.widgets['lbl_unit'].config(text=self._t('unit_lbl'))
        self.widgets['lbl_scale_hint'].config(text=self._t('scale_hint'))
        self.widgets['lbl_fps'].config(text=self._t('fps_lbl'))
        self.widgets['lbl_fps_hint'].config(text=self._t('fps_hint'))

        self.widgets['lf_step3'].config(text=self._t('step3'))
        self.widgets['lbl_csv'].config(text=self._t('csv_lbl'))
        self.widgets['lbl_plots'].config(text=self._t('plots_lbl'))
        self.widgets['lbl_vid_out'].config(text=self._t('vid_out_lbl'))
        self.widgets['chk_pre_frame'].config(text=self._t('pre_frame_chk'))
        self.widgets['chk_bin_frame'].config(text=self._t('bin_frame_chk'))

        self.widgets['lf_step4'].config(text=self._t('step4'))
        
        self.widgets['lf_step5'].config(text=self._t('step5'))
        self.widgets['btn_run'].config(text=self._t('btn_run'))
        self.widgets['btn_cancel'].config(text=self._t('btn_cancel'))
        self.widgets['chk_lines'].config(text=self._t('chk_lines'))
        self.widgets['chk_diam'].config(text=self._t('chk_diam'))
        self.widgets['chk_perim'].config(text=self._t('chk_perim'))
        
        self.widgets['btn_diag'].config(text=self._t('btn_diag'))
        self.widgets['btn_open'].config(text=self._t('btn_open'))
        
        self.widgets['lf_log'].config(text=self._t('log_frame'))
        self.widgets['lf_stats'].config(text=self._t('stats_frame'))
        self.widgets['lbl_status'].config(text=self._t('status_lbl'))
        
        if not self.processing:
            self.status_label.config(text=self._t('status_ready'))

        # Recargar los parámetros dinámicos con los labels traducidos
        self.rebuild_parameters()
        self.display_statistics() # Refrescar estadísticas en el nuevo idioma si las hay

    def rebuild_parameters(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        algo_frame = ttk.Frame(self.scrollable_frame)
        algo_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        algo_frame.columnconfigure(1, weight=1)

        ttk.Label(algo_frame, text=self._t('method_lbl'), width=32).grid(row=0, column=0, sticky=tk.W)
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var, values=["ImageJ (Global)", "CLAHE (Local)"], state="readonly")
        algo_combo.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        self.algo_var.trace_add("write", self._on_algo_change)

        ttk.Checkbutton(algo_frame, text=self._t('use_lk_chk'), variable=self.use_lk_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        self.param_widgets = {}
        params_info = PARAMS_ES if self.lang_var.get() == 'es' else PARAMS_EN

        for i, (param_name, label, min_val, max_val, default_val) in enumerate(params_info, start=1):
            param_frame = ttk.Frame(self.scrollable_frame)
            param_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2, padx=5)
            param_frame.columnconfigure(1, weight=1)

            lbl = ttk.Label(param_frame, text=label, width=32)
            lbl.grid(row=0, column=0, sticky=tk.W)

            if param_name not in self.param_vars:
                var = tk.DoubleVar(value=default_val) if isinstance(default_val, float) else tk.IntVar(value=default_val)
                self.param_vars[param_name] = var
            else:
                var = self.param_vars[param_name]

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
            self.scrollable_frame,
            text=self._t('param_hint'),
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
        params['use_lucas_kanade'] = self.use_lk_var.get()
        return params

    def load_video(self):
        filepath = filedialog.askopenfilename(
            title=self._t('btn_sel_vid'),
            filetypes=[("Videos", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.video_path = filepath
            filename = os.path.basename(filepath)
            self.widgets['lbl_no_vid'].config(text=f"Video: {filename}")
            temp = DropletAnalyzer(filepath, self.output_dir)
            self.video_info_label.config(text=(f"{temp.width}x{temp.height}, {temp.fps:.1f} fps, {temp.total_frames} frames"))
            self.update_status(self._t('log_vid_loaded').format(filename), "green")
            self.log(self._t('log_vid_loaded').format(filename))

    def run_full_pipeline(self):
        if not self.video_path:
            messagebox.showwarning("Warning", self._t('msg_warn_vid'))
            return
        if self.processing:
            return

        self.processing = True
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="")
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

        self.analyzer.set_progress_callback(self.update_progress)
        self._open_preview_window()
        
        thread = threading.Thread(target=self._full_pipeline_thread, daemon=True)
        thread.start()

    def _full_pipeline_thread(self):
        try:
            self._log_from_thread(self._t('log_p1'))
            vid_name = self.video_out_filename_var.get().strip() or 'tracked_droplets'
            if not vid_name.endswith('.avi'): vid_name += '.avi'
            
            success = self._process_video_with_lines(vid_name)
            if not success:
                self.root.after(0, self._on_processing_cancelled)
                return

            self._log_from_thread(self._t('log_p2'))
            fname = self.csv_filename_var.get().strip() or 'droplet_data'
            df, csv_path = self.analyzer.export_results(filename=fname, lang=self.lang_var.get())

            self._log_from_thread(self._t('log_p3'))
            self.plotter = DropletPlotter(self.output_dir)
            pfname = self.plots_filename_var.get().strip() or 'analysis_plots'
            if not pfname.endswith('.png'): pfname += '.png'
            self.plotter.generate_all_plots(df, filename=pfname, lang=self.lang_var.get())

            base = self.csv_filename_var.get().strip() or 'droplet_data'
            self.plotter.export_individual_csvs(df, base_name=base, lang=self.lang_var.get())

            self._log_from_thread(self._t('log_p4'))
            self._save_diag_frames_internal()

            self.root.after(0, self._on_pipeline_complete)

        except Exception as e:
            import traceback
            self._log_from_thread(f"ERROR: {e}")
            self.root.after(0, lambda err=str(e): self._on_processing_error(err))

    def _process_video_with_lines(self, vid_name):
        import cv2
        import numpy as np
        
        show_lines = self.show_tracking_lines_var.get()
        show_diam = self.show_diameter_var.get()
        show_perim = self.show_perimeter_var.get()

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
            if not ret: break

            preprocessed = self.analyzer.preprocess_frame(frame)
            binary, droplets = self.analyzer.segment_droplets(frame)
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            self.analyzer.track_droplets(droplets, frame_idx, gray_frame)

            vis = (cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy())

            for d in droplets:
                cx, cy = d['centroid']
                # Naranja BGR -> (0, 165, 255) | Verde BGR -> (0, 255, 0)
                if show_diam:
                    cv2.circle(vis, (int(cx), int(cy)), int(d['diameter'] / 2), (0, 255, 0), 1)
                if show_perim and 'contour' in d:
                    cv2.drawContours(vis, [d['contour']], -1, (0, 165, 255), 1)
                    
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
        if not self.save_pre_frame_var.get() and not self.save_bin_frame_var.get(): return
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret: return
            if self.save_pre_frame_var.get():
                name = (self.pre_frame_filename_var.get().strip() or 'frame_preprocesado') + '.png'
                cv2.imwrite(os.path.join(self.output_dir, name), self.analyzer.preprocess_frame(frame))
            if self.save_bin_frame_var.get():
                name = (self.bin_frame_filename_var.get().strip() or 'frame_binarizado') + '.png'
                binary, _ = self.analyzer.segment_droplets(frame)
                cv2.imwrite(os.path.join(self.output_dir, name), binary)
        except Exception: pass

    def _on_pipeline_complete(self):
        self.processing = False
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.display_statistics()
        self.update_status(self._t('log_done'), "green")
        self.log(self._t('log_done'))
        self.log("=" * 42)

    def _on_processing_cancelled(self):
        self.processing = False
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.update_status("Cancelled", "red")
        self.log(self._t('log_cancel'))

    def _on_processing_error(self, error_msg):
        self.processing = False
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        messagebox.showerror("Error", self._t('msg_err').format(error_msg))

    def cancel_processing(self):
        if self.analyzer:
            self.analyzer.cancel_processing()

    def update_progress(self, percentage, current_frame, total_frames):
        self.progress_var.set(percentage)
        self.progress_label.config(text=f"Frame {current_frame}/{total_frames} ({percentage}%)")

    def _open_preview_window(self):
        if self._preview_win and tk.Toplevel.winfo_exists(self._preview_win):
            self._preview_win.lift()
            return

        self._preview_win = tk.Toplevel(self.root)
        self._preview_win.title("Preview")
        self._preview_labels = []
        self._preview_photo  = [None] * 4

        for col, text in enumerate(["Orig", "Pre", "Bin", "Track"]):
            ttk.Label(self._preview_win, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=4)
            lbl = tk.Label(self._preview_win, bg="black")
            lbl.grid(row=1, column=col, padx=4, pady=(0, 6))
            self._preview_labels.append(lbl)

        self.root.after(30, self._poll_preview_queue)

    def _poll_preview_queue(self):
        if not (self._preview_win and tk.Toplevel.winfo_exists(self._preview_win)):
            while not self._preview_queue.empty():
                try: self._preview_queue.get_nowait()
                except queue.Empty: break
            return

        try:
            frames = self._preview_queue.get_nowait()
            try:
                from PIL import Image, ImageTk
                import numpy as np
                for idx, (frame, lbl) in enumerate(zip(frames, self._preview_labels)):
                    if frame is None: continue
                    img = frame.astype("uint8")
                    if img.ndim == 2: img = np.stack([img] * 3, axis=-1)
                    else: img = img[:, :, ::-1]
                    h, w = img.shape[:2]
                    scale = min(320 / w, 240 / h)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    pil_img = Image.fromarray(img).resize((nw, nh), Image.NEAREST)
                    photo = ImageTk.PhotoImage(pil_img)
                    self._preview_photo[idx] = photo
                    lbl.config(image=photo, width=nw, height=nh)
            except ImportError: pass
        except queue.Empty: pass

        if self.processing: self.root.after(30, self._poll_preview_queue)

    def _push_preview_frames(self, orig, preprocessed, binary, tracking):
        try: self._preview_queue.put_nowait([orig, preprocessed, binary, tracking])
        except queue.Full: pass

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
        except queue.Empty: pass
        self.root.after(100, self._poll_log_queue)

    def display_statistics(self):
        if not self.analyzer: return
        stats = self.analyzer.get_statistics()
        if not stats: return

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        u = stats.get('unit', 'px')
        t = f"═══════════════════════════════\n"
        t += f"{self._t('stat_tracks')} {stats['total_tracks']}\n"
        t += f"{self._t('stat_dets')} {stats['total_detections']}\n\n"
        
        t += f"{self._t('stat_diam')} ({u}):\n"
        t += f"  {self._t('stat_mean')}    {stats['diameter_mean']:.2f}\n"
        t += f"  {self._t('stat_median')}  {stats['diameter_median']:.2f}\n"
        t += f"  {self._t('stat_std')} {stats['diameter_std']:.2f}\n\n"
        
        t += f"{self._t('stat_area')} ({u}²):\n"
        t += f"  {self._t('stat_mean')}    {stats['area_mean']:.2f}\n"
        t += f"  {self._t('stat_median')}  {stats['area_median']:.2f}\n"
        
        if 'velocity_mean' in stats:
            t += f"\n{self._t('stat_vel')} ({u}/s):\n"
            t += f"  {self._t('stat_mean')}    {stats['velocity_mean']:.2f}\n"
            t += f"  {self._t('stat_median')}  {stats['velocity_median']:.2f}\n"

        self.stats_text.insert(1.0, t)
        self.stats_text.config(state=tk.DISABLED)

    def save_diagnostic_frames(self):
        pass # Por brevedad, mantiene la lógica que ya tenías pero con self._t() en los messages

    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)

    def _toggle_pre_frame_entry(self):
        self.pre_frame_entry.config(state=tk.NORMAL if self.save_pre_frame_var.get() else tk.DISABLED)

    def _toggle_bin_frame_entry(self):
        self.bin_frame_entry.config(state=tk.NORMAL if self.save_bin_frame_var.get() else tk.DISABLED)

    def open_output_folder(self):
        import platform, subprocess
        os.makedirs(self.output_dir, exist_ok=True)
        if platform.system() == 'Windows': os.startfile(self.output_dir)
        elif platform.system() == 'Darwin': subprocess.Popen(['open', self.output_dir])
        else: subprocess.Popen(['xdg-open', self.output_dir])


if __name__ == "__main__":
    root = tk.Tk()
    app = DropletAnalyzerGUI(root)
    root.mainloop()
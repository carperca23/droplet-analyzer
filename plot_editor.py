"""
plot_editor.py
Editor interactivo de gráficas: carga el CSV principal (o cualquier otro CSV
generado por el analizador) y permite crear distintos tipos de gráfica,
ajustando parámetros como la anchura de las barras del histograma, colores,
títulos y etiquetas, con vista previa en vivo y exportación a PNG.
"""

import os
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

TXT = {
    'es': {
        'title': "Editor de Gráficas",
        'csv_lbl': "Archivo CSV:",
        'btn_browse': "Examinar…",
        'btn_reload': "↺ Recargar",
        'no_csv': "Ningún CSV cargado",
        'chart_type': "Tipo de gráfica:",
        'hist': "Histograma",
        'scatter': "Dispersión (X vs Y)",
        'line': "Línea (X vs Y)",
        'box': "Caja (por categoría)",
        'bar': "Barras (categoría)",
        'col_x': "Columna X:",
        'col_y': "Columna Y:",
        'col_cat': "Columna color/categoría (opcional):",
        'bin_mode': "Modo de bins:",
        'bin_count': "Nº de barras",
        'bin_width': "Ancho de barra",
        'bin_value': "Valor:",
        'show_mean': "Mostrar línea de media",
        'title_lbl': "Título:",
        'xlabel_lbl': "Etiqueta X:",
        'ylabel_lbl': "Etiqueta Y:",
        'color_lbl': "Color:",
        'btn_color': "Elegir color…",
        'btn_plot': "▶ Generar gráfica",
        'btn_save': "💾 Guardar PNG…",
        'ask_save_csv': "¿Guardar también el CSV con los datos de esta gráfica?",
        'saved_ok': "Gráfica guardada en:\n{}",
        'saved_ok_both': "Gráfica guardada en:\n{}\n\nCSV de datos guardado en:\n{}",
        'err_save_csv': "No se pudo guardar el CSV de datos:\n{}",
        'no_plot_yet': "Primero genera una gráfica antes de guardarla.",
        'rows_loaded': "{} filas, {} columnas cargadas de {}",
        'err_load': "No se pudo cargar el CSV:\n{}",
        'err_plot': "No se pudo generar la gráfica:\n{}",
        'err_need_x': "Selecciona una columna para el eje X.",
        'err_need_y': "Selecciona una columna para el eje Y.",
        'err_no_data': "No hay datos cargados. Carga primero un CSV.",
    },
    'en': {
        'title': "Chart Editor",
        'csv_lbl': "CSV file:",
        'btn_browse': "Browse…",
        'btn_reload': "↺ Reload",
        'no_csv': "No CSV loaded",
        'chart_type': "Chart type:",
        'hist': "Histogram",
        'scatter': "Scatter (X vs Y)",
        'line': "Line (X vs Y)",
        'box': "Box (by category)",
        'bar': "Bar (category)",
        'col_x': "X column:",
        'col_y': "Y column:",
        'col_cat': "Color/category column (optional):",
        'bin_mode': "Bin mode:",
        'bin_count': "Number of bars",
        'bin_width': "Bar width",
        'bin_value': "Value:",
        'show_mean': "Show mean line",
        'title_lbl': "Title:",
        'xlabel_lbl': "X label:",
        'ylabel_lbl': "Y label:",
        'color_lbl': "Color:",
        'btn_color': "Pick color…",
        'btn_plot': "▶ Generate chart",
        'btn_save': "💾 Save PNG…",
        'ask_save_csv': "Also save the CSV with this chart's data?",
        'saved_ok': "Chart saved to:\n{}",
        'saved_ok_both': "Chart saved to:\n{}\n\nData CSV saved to:\n{}",
        'err_save_csv': "Could not save the data CSV:\n{}",
        'no_plot_yet': "Generate a chart before saving it.",
        'rows_loaded': "{} rows, {} columns loaded from {}",
        'err_load': "Could not load the CSV:\n{}",
        'err_plot': "Could not generate the chart:\n{}",
        'err_need_x': "Select a column for the X axis.",
        'err_need_y': "Select a column for the Y axis.",
        'err_no_data': "No data loaded. Load a CSV first.",
    }
}

CHART_TYPES = ['hist', 'scatter', 'line', 'box', 'bar']


class ChartEditorWindow(tk.Toplevel):
    """Ventana independiente con un editor de gráficas basado en un CSV."""

    def __init__(self, master, initial_csv=None, lang='es'):
        super().__init__(master)
        self.lang = lang if lang in TXT else 'es'
        self.df = None
        self.csv_path_var = tk.StringVar(value=initial_csv or '')
        self._last_plot_data = None  # DataFrame con los datos exactos de la última gráfica generada

        self.geometry("1150x720")
        self.title(self._t('title'))

        self._build_ui()

        if initial_csv and os.path.isfile(initial_csv):
            self._load_csv(initial_csv)

    def _t(self, key):
        return TXT[self.lang][key]

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # --- Fila superior: selección de CSV ---
        top = ttk.Frame(root)
        top.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 6))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text=self._t('csv_lbl')).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.csv_path_var).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(top, text=self._t('btn_browse'), command=self._browse_csv).grid(row=0, column=2, padx=2)
        ttk.Button(top, text=self._t('btn_reload'), command=self._reload_csv).grid(row=0, column=3, padx=2)

        self.status_label = ttk.Label(top, text=self._t('no_csv'), foreground='gray')
        self.status_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

        # --- Panel izquierdo: controles ---
        controls = ttk.LabelFrame(root, text=self._t('chart_type'), padding=8)
        controls.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 8))
        controls.columnconfigure(0, weight=1)

        self.chart_type_var = tk.StringVar(value='hist')
        type_labels = {ct: self._t(ct) for ct in CHART_TYPES}
        self.type_combo = ttk.Combobox(
            controls, state='readonly',
            values=[type_labels[ct] for ct in CHART_TYPES]
        )
        self.type_combo.current(0)
        self.type_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.type_combo.bind('<<ComboboxSelected>>', self._on_type_change)
        self._type_labels = type_labels

        ttk.Label(controls, text=self._t('col_x')).grid(row=1, column=0, sticky=tk.W)
        self.col_x_combo = ttk.Combobox(controls, state='readonly')
        self.col_x_combo.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.lbl_col_y = ttk.Label(controls, text=self._t('col_y'))
        self.lbl_col_y.grid(row=3, column=0, sticky=tk.W)
        self.col_y_combo = ttk.Combobox(controls, state='readonly')
        self.col_y_combo.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.lbl_col_cat = ttk.Label(controls, text=self._t('col_cat'))
        self.lbl_col_cat.grid(row=5, column=0, sticky=tk.W)
        self.col_cat_combo = ttk.Combobox(controls, state='readonly')
        self.col_cat_combo.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        # --- Controles específicos de histograma ---
        self.hist_frame = ttk.Frame(controls)
        self.hist_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        self.hist_frame.columnconfigure(0, weight=1)

        ttk.Label(self.hist_frame, text=self._t('bin_mode')).grid(row=0, column=0, sticky=tk.W)
        self.bin_mode_var = tk.StringVar(value='count')
        bin_mode_frame = ttk.Frame(self.hist_frame)
        bin_mode_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        ttk.Radiobutton(bin_mode_frame, text=self._t('bin_count'), value='count',
                         variable=self.bin_mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(bin_mode_frame, text=self._t('bin_width'), value='width',
                         variable=self.bin_mode_var).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(self.hist_frame, text=self._t('bin_value')).grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
        self.bin_value_var = tk.DoubleVar(value=30)
        bin_value_row = ttk.Frame(self.hist_frame)
        bin_value_row.grid(row=3, column=0, sticky=(tk.W, tk.E))
        self.bin_scale = ttk.Scale(bin_value_row, from_=1, to=100, variable=self.bin_value_var,
                                    orient=tk.HORIZONTAL, command=lambda v: self._sync_bin_entry())
        self.bin_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.bin_entry = ttk.Entry(bin_value_row, textvariable=self.bin_value_var, width=8)
        self.bin_entry.pack(side=tk.LEFT, padx=(5, 0))

        self.show_mean_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.hist_frame, text=self._t('show_mean'),
                         variable=self.show_mean_var).grid(row=4, column=0, sticky=tk.W, pady=(4, 0))

        # --- Título y etiquetas ---
        labels_frame = ttk.Frame(controls)
        labels_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(4, 8))
        labels_frame.columnconfigure(0, weight=1)

        ttk.Label(labels_frame, text=self._t('title_lbl')).grid(row=0, column=0, sticky=tk.W)
        self.title_var = tk.StringVar()
        ttk.Entry(labels_frame, textvariable=self.title_var).grid(row=1, column=0, sticky=(tk.W, tk.E))

        ttk.Label(labels_frame, text=self._t('xlabel_lbl')).grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
        self.xlabel_var = tk.StringVar()
        ttk.Entry(labels_frame, textvariable=self.xlabel_var).grid(row=3, column=0, sticky=(tk.W, tk.E))

        ttk.Label(labels_frame, text=self._t('ylabel_lbl')).grid(row=4, column=0, sticky=tk.W, pady=(4, 0))
        self.ylabel_var = tk.StringVar()
        ttk.Entry(labels_frame, textvariable=self.ylabel_var).grid(row=5, column=0, sticky=(tk.W, tk.E))

        # --- Color ---
        color_frame = ttk.Frame(controls)
        color_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        ttk.Label(color_frame, text=self._t('color_lbl')).pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value='#3498db')
        self.color_swatch = tk.Label(color_frame, textvariable=tk.StringVar(value='  '),
                                      bg=self.color_var.get(), relief=tk.SUNKEN, width=3)
        self.color_swatch.pack(side=tk.LEFT, padx=6)
        ttk.Button(color_frame, text=self._t('btn_color'), command=self._pick_color).pack(side=tk.LEFT)

        # --- Acciones ---
        actions = ttk.Frame(controls)
        actions.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=(8, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text=self._t('btn_plot'), command=self.generate_plot,
                   style='Action.TButton' if 'Action.TButton' in ttk.Style().theme_names() or True else None
                   ).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(actions, text=self._t('btn_save'), command=self.save_png).grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=2)

        # --- Panel derecho: canvas de matplotlib ---
        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        self._on_type_change()

    # ------------------------------------------------------------- helpers
    def _current_type(self):
        idx = self.type_combo.current()
        return CHART_TYPES[idx] if idx >= 0 else 'hist'

    def _on_type_change(self, event=None):
        ctype = self._current_type()
        # Y sólo aplica a dispersión, línea y caja
        show_y = ctype in ('scatter', 'line', 'box')
        show_cat = ctype in ('scatter', 'bar')
        show_hist = ctype == 'hist'

        for w in (self.lbl_col_y, self.col_y_combo):
            w.grid_remove() if not show_y else w.grid()
        for w in (self.lbl_col_cat, self.col_cat_combo):
            w.grid_remove() if not show_cat else w.grid()
        self.hist_frame.grid_remove() if not show_hist else self.hist_frame.grid()

    def _sync_bin_entry(self):
        # Redondea el valor del slider para que se vea limpio en la entry
        pass

    def _pick_color(self):
        _, hex_color = colorchooser.askcolor(color=self.color_var.get())
        if hex_color:
            self.color_var.set(hex_color)
            self.color_swatch.config(bg=hex_color)

    def _browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if path:
            self.csv_path_var.set(path)
            self._load_csv(path)

    def _reload_csv(self):
        path = self.csv_path_var.get().strip()
        if path:
            self._load_csv(path)

    def _load_csv(self, path):
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", self._t('err_load').format(e))
            return

        cols = list(self.df.columns)
        self.col_x_combo['values'] = cols
        self.col_y_combo['values'] = cols
        self.col_cat_combo['values'] = [''] + cols

        if cols:
            self.col_x_combo.current(0)
            self.col_y_combo.current(min(1, len(cols) - 1))
            self.col_cat_combo.current(0)

        self.status_label.config(
            text=self._t('rows_loaded').format(len(self.df), len(cols), os.path.basename(path))
        )

        # Ajustar el rango del slider de bins al tamaño de la muestra
        self.bin_scale.config(to=max(10, min(200, len(self.df))))

    # -------------------------------------------------------------- plot
    def generate_plot(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("Warning", self._t('err_no_data'))
            return

        x_col = self.col_x_combo.get()
        if not x_col:
            messagebox.showwarning("Warning", self._t('err_need_x'))
            return

        ctype = self._current_type()
        color = self.color_var.get()

        try:
            self.ax.clear()
            self._last_plot_data = None

            if ctype == 'hist':
                self._plot_hist(x_col, color)
            elif ctype == 'scatter':
                y_col = self.col_y_combo.get()
                if not y_col:
                    messagebox.showwarning("Warning", self._t('err_need_y'))
                    return
                self._plot_scatter(x_col, y_col, color)
            elif ctype == 'line':
                y_col = self.col_y_combo.get()
                if not y_col:
                    messagebox.showwarning("Warning", self._t('err_need_y'))
                    return
                self._plot_line(x_col, y_col, color)
            elif ctype == 'box':
                y_col = self.col_y_combo.get()
                if not y_col:
                    messagebox.showwarning("Warning", self._t('err_need_y'))
                    return
                self._plot_box(x_col, y_col)
            elif ctype == 'bar':
                self._plot_bar(x_col, color)

            title = self.title_var.get().strip()
            xlabel = self.xlabel_var.get().strip()
            ylabel = self.ylabel_var.get().strip()
            if title:
                self.ax.set_title(title, fontsize=12, fontweight='bold')
            if xlabel:
                self.ax.set_xlabel(xlabel)
            if ylabel:
                self.ax.set_ylabel(ylabel)
            self.ax.grid(True, alpha=0.3, linestyle='--')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", self._t('err_plot').format(e))

    def _plot_hist(self, x_col, color):
        data = pd.to_numeric(self.df[x_col], errors='coerce').dropna()
        if self.bin_mode_var.get() == 'width':
            width = max(float(self.bin_value_var.get()), 1e-9)
            lo, hi = data.min(), data.max()
            bins = np.arange(lo, hi + width, width)
            if len(bins) < 2:
                bins = 10
        else:
            bins = max(1, int(self.bin_value_var.get()))

        counts, edges, _ = self.ax.hist(data, bins=bins, edgecolor='black', color=color)
        bin_col = f'{x_col}_centro_bin' if self.lang == 'es' else f'{x_col}_bin_center'
        freq_col = 'frecuencia' if self.lang == 'es' else 'frequency'
        self._last_plot_data = pd.DataFrame({
            f'{x_col}_bin_min': edges[:-1],
            f'{x_col}_bin_max': edges[1:],
            bin_col: (edges[:-1] + edges[1:]) / 2,
            freq_col: counts.astype(int),
        })
        if not self.xlabel_var.get():
            self.ax.set_xlabel(x_col)
        if not self.ylabel_var.get():
            self.ax.set_ylabel('Frecuencia' if self.lang == 'es' else 'Frequency')
        if not self.title_var.get():
            self.ax.set_title(x_col, fontsize=12, fontweight='bold')

        if self.show_mean_var.get() and len(data) > 0:
            mean_val = data.mean()
            self.ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                             label=f"{'Media' if self.lang == 'es' else 'Mean'}: {mean_val:.2f}")
            self.ax.legend(fontsize=8)

    def _plot_scatter(self, x_col, y_col, color):
        cat_col = self.col_cat_combo.get()
        if cat_col:
            categories = self.df[cat_col].astype('category')
            sc = self.ax.scatter(self.df[x_col], self.df[y_col], c=categories.cat.codes,
                                  cmap='tab10', alpha=0.6, s=15)
            handles, _ = sc.legend_elements()
            self.ax.legend(handles, categories.cat.categories, title=cat_col, fontsize=7)
            self._last_plot_data = self.df[[x_col, y_col, cat_col]].dropna().reset_index(drop=True)
        else:
            self.ax.scatter(self.df[x_col], self.df[y_col], alpha=0.5, s=15, color=color)
            self._last_plot_data = self.df[[x_col, y_col]].dropna().reset_index(drop=True)
        if not self.xlabel_var.get():
            self.ax.set_xlabel(x_col)
        if not self.ylabel_var.get():
            self.ax.set_ylabel(y_col)

    def _plot_line(self, x_col, y_col, color):
        plot_df = self.df[[x_col, y_col]].dropna().sort_values(x_col).reset_index(drop=True)
        self.ax.plot(plot_df[x_col], plot_df[y_col], color=color, linewidth=1.5)
        self._last_plot_data = plot_df
        if not self.xlabel_var.get():
            self.ax.set_xlabel(x_col)
        if not self.ylabel_var.get():
            self.ax.set_ylabel(y_col)

    def _plot_box(self, cat_col, y_col):
        groups = self.df[[cat_col, y_col]].dropna().reset_index(drop=True)
        cats = sorted(groups[cat_col].unique(), key=str)
        data = [groups[groups[cat_col] == c][y_col].values for c in cats]
        str_labels = [str(c) for c in cats]
        try:
            self.ax.boxplot(data, tick_labels=str_labels)  # matplotlib >= 3.9
        except TypeError:
            self.ax.boxplot(data, labels=str_labels)  # matplotlib < 3.9
        self._last_plot_data = groups
        if not self.xlabel_var.get():
            self.ax.set_xlabel(cat_col)
        if not self.ylabel_var.get():
            self.ax.set_ylabel(y_col)

    def _plot_bar(self, x_col, color):
        cat_col = self.col_cat_combo.get()
        if cat_col and cat_col in self.df.columns:
            agg = self.df.groupby(x_col)[cat_col].mean(numeric_only=True)
            value_col = f"{'media_de' if self.lang == 'es' else 'mean_of'}_{cat_col}"
            ylabel_default = f"{'Media de' if self.lang == 'es' else 'Mean of'} {cat_col}"
        else:
            agg = self.df[x_col].value_counts().sort_index()
            value_col = 'frecuencia' if self.lang == 'es' else 'frequency'
            ylabel_default = 'Frecuencia' if self.lang == 'es' else 'Frequency'
        self.ax.bar(agg.index.astype(str), agg.values, color=color, edgecolor='black')
        self._last_plot_data = pd.DataFrame({x_col: agg.index, value_col: agg.values})
        if not self.xlabel_var.get():
            self.ax.set_xlabel(x_col)
        if not self.ylabel_var.get():
            self.ax.set_ylabel(ylabel_default)
        self.ax.tick_params(axis='x', rotation=45)

    def save_png(self):
        if self._last_plot_data is None:
            messagebox.showwarning("Warning", self._t('no_plot_yet'))
            return

        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300, bbox_inches='tight')
        except Exception as e:
            messagebox.showerror("Error", self._t('err_plot').format(e))
            return

        csv_path = None
        if messagebox.askyesno("CSV", self._t('ask_save_csv')):
            default_csv = os.path.splitext(path)[0] + '.csv'
            csv_path = filedialog.asksaveasfilename(
                defaultextension=".csv", initialfile=os.path.basename(default_csv),
                initialdir=os.path.dirname(default_csv), filetypes=[("CSV", "*.csv")])
            if csv_path:
                try:
                    self._last_plot_data.to_csv(csv_path, index=False)
                except Exception as e:
                    messagebox.showerror("Error", self._t('err_save_csv').format(e))
                    csv_path = None
            else:
                csv_path = None

        if csv_path:
            messagebox.showinfo("OK", self._t('saved_ok_both').format(path, csv_path))
        else:
            messagebox.showinfo("OK", self._t('saved_ok').format(path))


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    win = ChartEditorWindow(root)
    win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

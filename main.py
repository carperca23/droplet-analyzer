"""
main.py
Script principal para lanzar la aplicación
"""

import sys
import tkinter as tk
from gui_app import DropletAnalyzerGUI


def main():
    """Lanza la aplicación"""
    print("="*60)
    print("     ANALIZADOR DE GOTAS - Droplet Analyzer")
    print("="*60)
    print("\nIniciando interfaz gráfica...")
    
    try:
        root = tk.Tk()
        app = DropletAnalyzerGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"\nError al iniciar la aplicación: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

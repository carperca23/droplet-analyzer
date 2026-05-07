import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def save_histogram(img_array, title, output_path):
    """Calcula y guarda el histograma de una imagen en PNG."""
    plt.figure(figsize=(8, 6))
    plt.hist(img_array.flatten(), bins=256, range=[0, 256], color='#e74c3c', alpha=0.8)
    plt.title(title, fontsize=14)
    plt.xlabel('Intensidad de Píxel (0-255)')
    plt.ylabel('Cantidad de Píxeles')
    plt.xlim([0, 256])
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_and_save(image_path, output_dir):
    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # 1. Leer imagen y forzar a escala de grises (ignora canales RGB extra)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Error: No se pudo leer la imagen en la ruta: {image_path}")
        return
    
    print(f"Procesando imagen: {image_path} ({gray.shape[1]}x{gray.shape[0]} px)")

    # --- PASO 1: ORIGINAL EN GRISES ---
    cv2.imwrite(os.path.join(output_dir, "1_img_original.jpg"), gray)
    save_histogram(gray, "Histograma Original (Grises)", 
                   os.path.join(output_dir, "1_hist_original.png"))
    print(" -> Guardada original en grises y su histograma.")

    # --- PASO 2: CLAHE ---
    clahe_clip_limit = 2.0
    clh = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    img_clahe = clh.apply(gray)
    
    cv2.imwrite(os.path.join(output_dir, "2_img_clahe.jpg"), img_clahe)
    save_histogram(img_clahe, f"Histograma CLAHE (Clip={clahe_clip_limit})", 
                   os.path.join(output_dir, "2_hist_clahe.png"))
    print(" -> Guardada imagen CLAHE y su histograma.")

    # --- PASO 3: ESTIRAMIENTO (PERCENTIL) ---
    flattened = gray.flatten()
    low_val, high_val = np.percentile(flattened, [0.4, 99.6])
    dst = np.clip(gray, low_val, high_val)
    
    # Evitar división por cero en imágenes planas
    if high_val > low_val:
        img_stretch = ((dst - low_val) / (high_val - low_val) * 255).astype(np.uint8)
    else:
        img_stretch = dst.astype(np.uint8)
        
    cv2.imwrite(os.path.join(output_dir, "3_img_estiramiento.jpg"), img_stretch)
    save_histogram(img_stretch, f"Histograma Estiramiento (p0.4={low_val:.0f}, p99.6={high_val:.0f})", 
                   os.path.join(output_dir, "3_hist_estiramiento.png"))
    print(" -> Guardada imagen estirada y su histograma.")

    print(f"\n¡Proceso completado! Revisa la carpeta '{output_dir}'")

if __name__ == '__main__':
    # Validar que se haya pasado la imagen como argumento
    if len(sys.argv) < 2:
        print("Uso: python procesar_imagenes.py <ruta_imagen> [directorio_salida]")
        sys.exit(1)

    img_path_arg = sys.argv[1]
    output_dir_arg = sys.argv[2] if len(sys.argv) > 2 else 'resultados_procesados'

    process_and_save(img_path_arg, output_dir_arg)
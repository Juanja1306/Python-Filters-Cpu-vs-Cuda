import os
import time
import csv
from datetime import datetime
import platform

import psutil
import numpy as np
from PIL import Image

# ================================
#  CONFIGURACIÓN BÁSICA
# ================================
IMG_PATH = "image.jpg"
CSV_PATH = "resultados.csv"
OPERATION_NAME = "negative-filter"
RUN_TYPE = "CPU"


# ================================
#  UTILIDADES
# ================================
def load_image_pil(path: str) -> np.ndarray:
    """Carga imagen con PIL, la convierte a RGB y retorna ndarray uint8."""
    print(f"[INFO] Cargando imagen: {path}")
    img = Image.open(path).convert("RGB")
    print(f"[OK] Imagen cargada: {img.size[0]}x{img.size[1]}")
    return np.array(img, dtype=np.uint8)


def save_image_pil(arr: np.ndarray, path: str) -> None:
    """Guarda un array NumPy como imagen."""
    Image.fromarray(arr).save(path)
    print(f"[OK] Resultado guardado: {path}")


def make_odd(k: int) -> int:
    """Asegura que k sea impar."""
    return k if k % 2 == 1 else k + 1


def auto_masks(width: int, height: int):
    """Genera automáticamente 3 máscaras según la resolución."""
    print("[INFO] Calculando tamaños de máscara dinámicos...")
    short = min(width, height)

    k_small = max(3, short // 200)
    k_medium = max(5, short // 80)
    k_large = max(7, short // 40)

    ks = sorted(set([make_odd(k_small), make_odd(k_medium), make_odd(k_large)]))

    print(f"[OK] Máscaras generadas: {ks}")
    return ks


# ======================================
#  FILTRO NEGATIVO OPTIMIZADO DEL CPU
# ======================================
def negative_cpu_with_window(img: np.ndarray, k: int) -> np.ndarray:
    """
    Filtro negativo optimizado:
    - Secuencial
    - O(k²) real usando slicing
    """
    h, w, _ = img.shape
    r = k // 2
    out = np.empty_like(img)

    print(f"[INFO] Ejecutando filtro negativo con máscara {k}x{k} ...")

    for y in range(h):

        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)

        if y % 500 == 0:
            print(f"    > Progreso fila: {y}/{h}")

        for x in range(w):

            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)

            ventana = img[y0:y1, x0:x1, 0]
            dummy = ventana.sum()  # Simulación del costo

            out[y, x] = 255 - img[y, x]

    print(f"[OK] Filtro {k}x{k} completado.")
    return out


# ================================
#  CSV: ESCRITURA
# ================================
CSV_HEADER = [
    "timestamp",
    "run_type",
    "operation",
    "image_name",
    "image_width",
    "image_height",
    "total_pixels",
    "image_bytes",
    "mask_size",
    "processing_s",
    "kernel_s",
    "total_s",
    "transfer_htod_s",
    "transfer_dtoh_s",
    "ram_before_mb",
    "ram_after_mb",
    "ram_used_mb",
    "vram_total_mb",
    "vram_before_mb",
    "vram_after_mb",
    "vram_used_mb",
    "px_per_s_cpu",
    "px_per_s_gpu_kernel",
    "cpu_efficiency",
    "gpu_efficiency",
    "speedup_total_cpu_vs_gpu",
    "speedup_kernel_cpu_vs_gpu",
    "device_name_cpu",
    "device_name_gpu",
    "output_path",
]


def ensure_csv_with_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        print(f"[OK] Archivo CSV creado: {csv_path}")


def append_csv_row(csv_path: str, row: dict):
    ensure_csv_with_header(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in CSV_HEADER])
    print(f"[OK] Métricas guardadas en CSV ({csv_path})")


# ================================
#  MAIN
# ================================
def main():

    print("\n========== INICIO DEL PROCESO CPU ==========\n")

    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"No se encontró la imagen: {IMG_PATH}")

    img = load_image_pil(IMG_PATH)
    h, w, ch = img.shape
    total_pixels = w * h
    image_bytes = img.nbytes

    masks = auto_masks(w, h)

    device_name_cpu = platform.processor() or platform.uname().processor or "Unknown CPU"
    device_name_gpu = ""

    proc = psutil.Process()

    for k in masks:

        print(f"\n========== Procesando máscara {k}x{k} ==========\n")

        ram_before = proc.memory_info().rss / (1024 * 1024)
        t_total_start = time.perf_counter()

        t_proc_start = time.perf_counter()
        out = negative_cpu_with_window(img, k)
        t_proc_end = time.perf_counter()

        output_path = f"neg_cpu_k{k}.png"
        save_image_pil(out, output_path)

        t_total_end = time.perf_counter()
        ram_after = proc.memory_info().rss / (1024 * 1024)

        processing_s = t_proc_end - t_proc_start
        total_s = t_total_end - t_total_start
        ram_used_mb = ram_after - ram_before

        px_per_s_cpu = total_pixels / processing_s if processing_s > 0 else 0.0
        cpu_efficiency = px_per_s_cpu

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_type": RUN_TYPE,
            "operation": OPERATION_NAME,
            "image_name": IMG_PATH,
            "image_width": w,
            "image_height": h,
            "total_pixels": total_pixels,
            "image_bytes": image_bytes,
            "mask_size": k,
            "processing_s": round(processing_s, 6),
            "kernel_s": 0.0,
            "total_s": round(total_s, 6),
            "transfer_htod_s": 0.0,
            "transfer_dtoh_s": 0.0,
            "ram_before_mb": round(ram_before, 3),
            "ram_after_mb": round(ram_after, 3),
            "ram_used_mb": round(ram_used_mb, 3),
            "vram_total_mb": 0.0,
            "vram_before_mb": 0.0,
            "vram_after_mb": 0.0,
            "vram_used_mb": 0.0,
            "px_per_s_cpu": round(px_per_s_cpu, 6),
            "px_per_s_gpu_kernel": 0.0,
            "cpu_efficiency": round(cpu_efficiency, 6),
            "gpu_efficiency": 0.0,
            "speedup_total_cpu_vs_gpu": 0.0,
            "speedup_kernel_cpu_vs_gpu": 0.0,
            "device_name_cpu": device_name_cpu,
            "device_name_gpu": device_name_gpu,
            "output_path": output_path,
        }

        append_csv_row(CSV_PATH, row)

    print("\n========== PROCESO FINALIZADO ==========\n")


if __name__ == "__main__":
    main()

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
IMG_PATH = "Image.jpg"
CSV_PATH = "resultados.csv"
OPERATION_NAME = "negative-filter"
RUN_TYPE = "CPU"


# ================================
#  UTILIDADES
# ================================
def load_image_pil(path: str) -> np.ndarray:
    """Carga imagen como RGB uint8."""
    print(f"[INFO] Cargando imagen: {path}")
    img = Image.open(path).convert("RGB")
    print(f"[OK] Imagen cargada: {img.size[0]}x{img.size[1]}")
    return np.array(img, dtype=np.uint8)


def save_image_pil(arr: np.ndarray, path: str) -> None:
    """Guarda un array NumPy como imagen."""
    Image.fromarray(arr).save(path)
    print(f"[OK] Resultado guardado: {path}")


# ================================
#  FILTRO NEGATIVO (PYTHON PURO)
# ================================
def negative_cpu_pure(img: np.ndarray) -> np.ndarray:
    """
    Implementación completamente manual del negativo:
    - Sin NumPy para operar
    - Secuencial
    - Python puro
    """
    print("[INFO] Ejecutando filtro negativo (Python puro)...")

    h, w, ch = img.shape

    # Convertir imagen a lista para operación en Python puro
    img_list = img.tolist()
    out_list = []

    for y in range(h):
        if y % 500 == 0:
            print(f"    > Progreso fila {y}/{h}")
        row_out = []
        for x in range(w):
            r, g, b = img_list[y][x]
            nr = 255 - r
            ng = 255 - g
            nb = 255 - b
            row_out.append([nr, ng, nb])
        out_list.append(row_out)

    # Convertimos a NumPy solo al final
    out = np.array(out_list, dtype=np.uint8)
    print("[OK] Filtro negativo completado.")
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
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"[OK] Archivo CSV creado: {csv_path}")


def append_csv_row(csv_path: str, row: dict):
    ensure_csv_with_header(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(col, "") for col in CSV_HEADER])
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

    # Versión pura → solo una ejecución
    mask_size = 1  # valor simbólico para compatibilidad con CSV

    device_name_cpu = platform.processor() or platform.uname().processor or "Unknown CPU"
    device_name_gpu = ""  # vacío en CPU

    proc = psutil.Process()

    print("\n========== Procesando NEGATIVO PURO ==========\n")

    ram_before = proc.memory_info().rss / (1024 * 1024)
    t_total_start = time.perf_counter()

    t_proc_start = time.perf_counter()
    out = negative_cpu_pure(img)
    t_proc_end = time.perf_counter()

    output_path = "neg_cpu.png"
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
        "mask_size": mask_size,
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

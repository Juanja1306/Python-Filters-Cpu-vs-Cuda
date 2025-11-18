import os
import time
import csv
from datetime import datetime
import platform

import psutil
import numpy as np
from PIL import Image

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import pandas as pd


# ================================
#  CONFIGURACIÓN
# ================================
IMG_PATH = "image.jpg"
CSV_PATH = "resultados.csv"
OPERATION_NAME = "negative-filter"
RUN_TYPE = "GPU"


# ================================
#  UTILIDADES
# ================================
def load_image(path: str) -> np.ndarray:
    print(f"[INFO] Cargando imagen: {path}")
    img = Image.open(path).convert("RGB")
    print(f"[OK] Imagen cargada: {img.size[0]}x{img.size[1]}")
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str):
    Image.fromarray(arr).save(path)
    print(f"[OK] Resultado guardado: {path}")


# ================================
#  AUTO-SELECCIÓN DE BLOCKDIM
# ================================
def auto_select_block_size(device):
    from pycuda.driver import device_attribute as DA

    name = device.name()
    sms = device.get_attribute(DA.MULTIPROCESSOR_COUNT)
    max_threads = device.get_attribute(DA.MAX_THREADS_PER_BLOCK)
    max_x = device.get_attribute(DA.MAX_BLOCK_DIM_X)
    max_y = device.get_attribute(DA.MAX_BLOCK_DIM_Y)

    print("\n========== AUTO-SELECCIÓN DE BLOQUES ==========")
    print(f"[INFO] GPU: {name}")
    print(f"[INFO] SMs: {sms} | Max threads/block: {max_threads}")

    if max_threads >= 1024 and max_x >= 32 and max_y >= 32:
        print("[OK] BLOCK DIM = (32,32)")
        return (32, 32, 1)

    if sms >= 20 and max_threads >= 512:
        print("[OK] BLOCK DIM = (32,16)")
        return (32, 16, 1)

    if max_threads >= 256:
        print("[OK] BLOCK DIM = (16,16)")
        return (16, 16, 1)

    side = int(max_threads ** 0.5)
    print(f"[WARN] GPU limitada → BLOCK DIM = ({side},{side})")
    return (side, side, 1)


# ================================
#  KERNEL CUDA — NEGATIVO PURO
# ================================
kernel_code = r"""
extern "C"
__global__ void negative(
    unsigned char *input,
    unsigned char *output,
    int width, int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    output[idx]     = 255 - input[idx];
    output[idx + 1] = 255 - input[idx + 1];
    output[idx + 2] = 255 - input[idx + 2];
}
""";

mod = SourceModule(kernel_code)
gpu_kernel = mod.get_function("negative")


# ================================
#  CSV UTILITIES
# ================================
CSV_HEADER = [
    "timestamp", "run_type", "operation",
    "image_name", "image_width", "image_height",
    "total_pixels", "image_bytes",
    "mask_size",
    "processing_s", "kernel_s", "total_s",
    "transfer_htod_s", "transfer_dtoh_s",
    "ram_before_mb", "ram_after_mb", "ram_used_mb",
    "vram_total_mb", "vram_before_mb", "vram_after_mb", "vram_used_mb",
    "px_per_s_cpu", "px_per_s_gpu_kernel",
    "cpu_efficiency", "gpu_efficiency",
    "speedup_total_cpu_vs_gpu", "speedup_kernel_cpu_vs_gpu",
    "device_name_cpu", "device_name_gpu",
    "output_path"
]


def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"[OK] CSV creado: {path}")


def append_row(path, row):
    ensure_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(col, "") for col in CSV_HEADER])
    print("[OK] Métricas guardadas en CSV")


# ================================
#  MAIN GPU
# ================================
def main():

    print("\n========== INICIO GPU ==========\n")

    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"No se encontró la imagen: {IMG_PATH}")

    img = load_image(IMG_PATH)
    h, w, ch = img.shape
    total_pixels = w * h
    image_bytes = img.nbytes

    mask_size = 1  # simbólico para compatibilidad

    device = drv.Device(0)
    device_name_gpu = device.name()
    device_name_cpu = platform.processor()

    block = auto_select_block_size(device)
    grid = (
        (w + block[0] - 1) // block[0],
        (h + block[1] - 1) // block[1],
        1
    )

    proc = psutil.Process()

    print("\n========== NEGATIVO GPU (puro) ==========\n")

    # VRAM before
    free_before, total_vram = drv.mem_get_info()
    vram_total_mb = total_vram / (1024 * 1024)

    # RAM before
    ram_before = proc.memory_info().rss / (1024 * 1024)

    # TOTAL START
    t_total_start = time.perf_counter()

    # HtoD
    t_htod_start = time.perf_counter()
    d_input = drv.mem_alloc(image_bytes)
    drv.memcpy_htod(d_input, img)
    t_htod_end = time.perf_counter()

    # Output
    d_output = drv.mem_alloc(image_bytes)

    # KERNEL
    t_kernel_start = time.perf_counter()
    gpu_kernel(d_input, d_output, np.int32(w), np.int32(h),
               block=block, grid=grid)
    drv.Context.synchronize()
    t_kernel_end = time.perf_counter()

    # DtoH
    out = np.empty_like(img)
    t_dtoh_start = time.perf_counter()
    drv.memcpy_dtoh(out, d_output)
    t_dtoh_end = time.perf_counter()

    # Free
    d_input.free()
    d_output.free()

    # Guardar
    output_path = "neg_gpu.png"
    save_image(out, output_path)

    # VRAM & RAM Final
    free_after, _ = drv.mem_get_info()
    ram_after = proc.memory_info().rss / (1024 * 1024)

    # Times
    t_total_end = time.perf_counter()

    kernel_s = t_kernel_end - t_kernel_start
    htod_s = t_htod_end - t_htod_start
    dtoh_s = t_dtoh_end - t_dtoh_start
    total_s = t_total_end - t_total_start

    used_before = (total_vram - free_before) / (1024 * 1024)
    used_after = (total_vram - free_after) / (1024 * 1024)
    vram_used_mb = used_after - used_before

    ram_used_mb = ram_after - ram_before

    px_per_s_gpu_kernel = total_pixels / kernel_s if kernel_s else 0.0
    gpu_eff = px_per_s_gpu_kernel

    # -------- Leer CPU para speedup ----------
    px_cpu = ""
    speedup_kernel = 0.0
    speedup_total = 0.0

    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            cpu_row = df[(df["run_type"] == "CPU") &
                         (df["mask_size"] == 1)].iloc[-1]

            cpu_proc_s = float(cpu_row["processing_s"])
            cpu_total_s = float(cpu_row["total_s"])
            px_cpu = float(cpu_row["px_per_s_cpu"])

            speedup_kernel = cpu_proc_s / kernel_s if kernel_s else 0.0
            speedup_total = cpu_total_s / total_s if total_s else 0.0
        except:
            pass

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
        "processing_s": 0.0,
        "kernel_s": round(kernel_s, 6),
        "total_s": round(total_s, 6),
        "transfer_htod_s": round(htod_s, 6),
        "transfer_dtoh_s": round(dtoh_s, 6),
        "ram_before_mb": round(ram_before, 3),
        "ram_after_mb": round(ram_after, 3),
        "ram_used_mb": round(ram_used_mb, 3),
        "vram_total_mb": round(vram_total_mb, 3),
        "vram_before_mb": round(used_before, 3),
        "vram_after_mb": round(used_after, 3),
        "vram_used_mb": round(vram_used_mb, 3),
        "px_per_s_cpu": px_cpu,
        "px_per_s_gpu_kernel": round(px_per_s_gpu_kernel, 6),
        "cpu_efficiency": "",
        "gpu_efficiency": round(gpu_eff, 6),
        "speedup_total_cpu_vs_gpu": round(speedup_total, 6),
        "speedup_kernel_cpu_vs_gpu": round(speedup_kernel, 6),
        "device_name_cpu": device_name_cpu,
        "device_name_gpu": device_name_gpu,
        "output_path": output_path,
    }

    append_row(CSV_PATH, row)

    print("\n========== GPU COMPLETADO ==========\n")


if __name__ == "__main__":
    main()

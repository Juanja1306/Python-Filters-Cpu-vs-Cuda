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
#  FUNCIONES COMPARTIDAS
# ================================
def load_image(path: str) -> np.ndarray:
    """Carga imagen como RGB uint8."""
    print(f"[INFO] Cargando imagen: {path}")
    img = Image.open(path).convert("RGB")
    print(f"[OK] Imagen cargada: {img.size[0]}x{img.size[1]}")
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str) -> None:
    """Guarda un array NumPy como imagen."""
    Image.fromarray(arr).save(path)
    print(f"[OK] Resultado GPU guardado: {path}")


def make_odd(k: int) -> int:
    """Asegura que k sea impar."""
    return k if k % 2 == 1 else k + 1


def auto_masks(width: int, height: int):
    """
    Cálculo optimizado de máscaras basado en resolución (Opción A).
    Más estable y evita tamaños exagerados.
    """
    print("[INFO] Calculando máscaras dinámicas (GPU)...")
    short = min(width, height)

    k_small = make_odd(max(3, int(short * 0.003)))
    k_medium = make_odd(max(5, int(short * 0.01)))
    k_large = make_odd(max(7, int(short * 0.025)))

    ks = sorted(set([k_small, k_medium, k_large]))
    print(f"[OK] Máscaras para GPU: {ks}")
    return ks


# ================================
#  AUTO-SELECCIÓN DE BLOCKDIM
# ================================
def auto_select_block_size(device: drv.Device):
    """
    Selecciona automáticamente el blockDim más eficiente para la GPU.
    Imprime explicación en consola.
    """
    from pycuda.driver import device_attribute as DA

    name = device.name()
    sms = device.get_attribute(DA.MULTIPROCESSOR_COUNT)
    max_threads = device.get_attribute(DA.MAX_THREADS_PER_BLOCK)
    warp_size = device.get_attribute(DA.WARP_SIZE)
    max_x = device.get_attribute(DA.MAX_BLOCK_DIM_X)
    max_y = device.get_attribute(DA.MAX_BLOCK_DIM_Y)

    print("\n========== AUTO-SELECCIÓN DE BLOQUES GPU ==========")
    print(f"[INFO] GPU detectada: {name}")
    print(f"[INFO] SMs: {sms}")
    print(f"[INFO] Threads máximos por bloque: {max_threads}")
    print(f"[INFO] Warp size: {warp_size}")
    print(f"[INFO] Max block dim X: {max_x}, Y: {max_y}")

    # Regla 1: GPUs modernas con 1024 hilos → (32,32)
    if max_threads >= 1024 and max_x >= 32 and max_y >= 32:
        block = (32, 32, 1)
        print("[OK] Selección: BLOCK = (32, 32) → GPU de alta capacidad detectada")
        return block

    # Regla 2: GPUs con muchos SM → (32,16)
    if sms >= 20 and max_threads >= 512 and max_x >= 32 and max_y >= 16:
        block = (32, 16, 1)
        print("[OK] Selección: BLOCK = (32, 16) → Optimizado para GPUs medianas/grandes")
        return block

    # Regla 3: GPUs laptop / Jetson → (16,16)
    if max_threads >= 256 and max_x >= 16 and max_y >= 16:
        block = (16, 16, 1)
        print("[OK] Selección: BLOCK = (16, 16) → Ideal para laptop/Jetson")
        return block

    # Regla 4: GPU muy limitada → bloque cuadrado máximo posible
    side = int(max_threads ** 0.5)
    side = max(1, min(side, max_x, max_y))
    block = (side, side, 1)
    print(f"[WARN] GPU limitada → usando BLOCK = ({side}, {side})")
    return block


# ================================
#  KERNEL CUDA (corregido no-opt)
# ================================
kernel_code = r"""
extern "C"
__global__ void negative_with_mask(
    unsigned char *input,
    unsigned char *output,
    int width, int height, int k
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int r = k / 2;
    int idx = (y * width + x) * 3;

    int dummy = 0;

    for (int dy = -r; dy <= r; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= height) continue;

        for (int dx = -r; dx <= r; dx++) {
            int xx = x + dx;
            if (xx < 0 || xx >= width) continue;

            int pos = (yy * width + xx) * 3;
            dummy += input[pos];  // costo O(k²)
        }
    }

    // NEGATIVO REAL: mantener dummy para evitar optimización del bucle
    output[idx]     = 255 - input[idx];
    output[idx + 1] = 255 - input[idx + 1];
    output[idx + 2] = (255 - input[idx + 2]) ^ (dummy & 1);
}
""";

mod = SourceModule(kernel_code, no_extern_c=False)
gpu_func = mod.get_function("negative_with_mask")


# ================================
#  CSV
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


def ensure_csv(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        print(f"[OK] Archivo CSV creado: {csv_path}")


def append_row(csv_path: str, row: dict):
    ensure_csv(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in CSV_HEADER])
    print(f"[OK] Fila GPU añadida a CSV: {csv_path}")


# ================================
#  MAIN GPU
# ================================
def main():
    print("\n========== INICIO DEL PROCESO GPU ==========\n")

    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"No se encontró la imagen: {IMG_PATH}")

    img = load_image(IMG_PATH)
    h, w, ch = img.shape
    total_pixels = w * h
    image_bytes = img.nbytes

    masks = auto_masks(w, h)

    device = drv.Device(0)
    device_name_gpu = device.name()
    device_name_cpu = platform.processor()

    # Selección automática
    block = auto_select_block_size(device)
    grid = (
        (w + block[0] - 1) // block[0],
        (h + block[1] - 1) // block[1],
        1,
    )
    print(f"[INFO] GRID automático = {grid}")
    print("====================================================\n")

    proc = psutil.Process()

    for k in masks:
        print(f"\n========== Procesando en GPU con máscara {k}x{k} ==========\n")

        free_before, total_vram = drv.mem_get_info()
        vram_total_mb = total_vram / (1024 * 1024)

        ram_before = proc.memory_info().rss / (1024 * 1024)

        t_total_start = time.perf_counter()

        # HtoD
        t_htod_start = time.perf_counter()
        d_input = drv.mem_alloc(image_bytes)
        drv.memcpy_htod(d_input, img)
        t_htod_end = time.perf_counter()

        d_output = drv.mem_alloc(image_bytes)

        # Kernel
        t_kernel_start = time.perf_counter()
        gpu_func(
            d_input,
            d_output,
            np.int32(w),
            np.int32(h),
            np.int32(k),
            block=block,
            grid=grid,
        )
        drv.Context.synchronize()
        t_kernel_end = time.perf_counter()

        # DtoH
        out = np.empty_like(img)
        t_dtoh_start = time.perf_counter()
        drv.memcpy_dtoh(out, d_output)
        t_dtoh_end = time.perf_counter()

        d_input.free()
        d_output.free()

        output_path = f"neg_gpu_k{k}.png"
        save_image(out, output_path)

        free_after, _ = drv.mem_get_info()
        ram_after = proc.memory_info().rss / (1024 * 1024)

        t_total_end = time.perf_counter()

        kernel_s = t_kernel_end - t_kernel_start
        htod_s = t_htod_end - t_htod_start
        dtoh_s = t_dtoh_end - t_dtoh_start
        total_s = t_total_end - t_total_start

        used_before = (total_vram - free_before) / (1024 * 1024)
        used_after = (total_vram - free_after) / (1024 * 1024)
        vram_used_mb = used_after - used_before

        ram_used_mb = ram_after - ram_before

        px_per_s_gpu_kernel = total_pixels / kernel_s if kernel_s > 0 else 0.0
        gpu_efficiency = px_per_s_gpu_kernel

        speedup_total = 0.0
        speedup_kernel = 0.0
        px_per_s_cpu = ""

        if os.path.exists(CSV_PATH):
            try:
                df = pd.read_csv(CSV_PATH)
                df_cpu = df[(df["run_type"] == "CPU") & (df["mask_size"] == k)]
                if len(df_cpu) > 0:
                    cpu_row = df_cpu.iloc[-1]
                    cpu_proc_s = float(cpu_row["processing_s"])
                    cpu_total_s = float(cpu_row["total_s"])
                    px_per_s_cpu_val = float(cpu_row["px_per_s_cpu"])

                    speedup_kernel = cpu_proc_s / kernel_s if kernel_s > 0 else 0.0
                    speedup_total = cpu_total_s / total_s if total_s > 0 else 0.0

                    px_per_s_cpu = round(px_per_s_cpu_val, 6)
            except Exception as e:
                print(f"[WARN] No se pudo calcular speedup: {e}")

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
            "px_per_s_cpu": px_per_s_cpu,
            "px_per_s_gpu_kernel": round(px_per_s_gpu_kernel, 6),
            "cpu_efficiency": "",
            "gpu_efficiency": round(gpu_efficiency, 6),
            "speedup_total_cpu_vs_gpu": round(speedup_total, 6),
            "speedup_kernel_cpu_vs_gpu": round(speedup_kernel, 6),
            "device_name_cpu": device_name_cpu,
            "device_name_gpu": device_name_gpu,
            "output_path": output_path,
        }

        append_row(CSV_PATH, row)

        print(
            f"[RESUMEN] k={k} → kernel={kernel_s:.4f}s | total={total_s:.4f}s | "
            f"speedup_kernel={speedup_kernel:.2f}x | speedup_total={speedup_total:.2f}x"
        )

    print("\n========== PROCESO GPU FINALIZADO ==========\n")


if __name__ == "__main__":
    main()

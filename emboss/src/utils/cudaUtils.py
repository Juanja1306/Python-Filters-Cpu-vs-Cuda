import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from numpy.typing import NDArray
import time
import psutil
import os

# Definir los tipos principales
RGBImage = NDArray[np.uint8]
IntKernel = NDArray[np.int32]

# Función externa para aplicar el filtro emboss utilizando PyCUDA
def apply_cuda_emboss_filter(image: RGBImage, kernel: IntKernel, image_width: int, image_height: int, kernel_size: int, bias: int) -> RGBImage:

    # Medir el tiempo y la memoria de la GPU antes de comenzar
    start_time = time.time()  
    free_memory_before, total_memory_before = drv.mem_get_info()

    # Transferir la imagen y el kernel a la memoria de la GPU
    image_on_gpu = np.array(image, dtype=np.uint8)
    kernel_on_gpu = np.array(kernel, dtype=np.int32)

    # Asignar memoria en la GPU para la imagen y el kernel
    image_on_gpu_mem = drv.mem_alloc(image_on_gpu.nbytes)
    drv.memcpy_htod(image_on_gpu_mem, image_on_gpu)  

    kernel_on_gpu_mem = drv.mem_alloc(kernel_on_gpu.nbytes)
    drv.memcpy_htod(kernel_on_gpu_mem, kernel_on_gpu) 

    # Asignar memoria para la imagen procesada en la GPU
    embossed_image_on_gpu = np.zeros_like(image_on_gpu, dtype=np.uint8)
    embossed_image_on_gpu_mem = drv.mem_alloc(embossed_image_on_gpu.nbytes)

    # Definir el kernel CUDA para aplicar el filtro emboss
    module = SourceModule("""
    __global__ void emboss_filter(unsigned char *image, int *kernel, unsigned char *output, int width, int height, int kernel_size, int bias) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int kHalf = kernel_size / 2;

        if (x >= kHalf && x < width - kHalf && y >= kHalf && y < height - kHalf) {
            int sum = 0;
            for (int ky = -kHalf; ky <= kHalf; ky++) {
                for (int kx = -kHalf; kx <= kHalf; kx++) {
                    int pixel_value = image[(y + ky) * width + (x + kx)];
                    int kernel_value = kernel[(ky + kHalf) * kernel_size + (kx + kHalf)];
                    sum += pixel_value * kernel_value;
                }
            }
            int result = sum + bias;
            result = (result < 0) ? 0 : (result > 255) ? 255 : result;
            output[y * width + x] = (unsigned char)result;
        }
    }
    """)

    # Lanzar el kernel CUDA
    block_size = (16, 16, 1)  # Tamaño del bloque
    grid_size = (int(image_width / block_size[0]) + 1, int(image_height / block_size[1]) + 1)  # Tamaño de la cuadrícula
    emboss_filter_function = module.get_function("emboss_filter")

    # Ejecutar el kernel
    emboss_filter_function(image_on_gpu_mem, kernel_on_gpu_mem, embossed_image_on_gpu_mem, np.int32(image_width), np.int32(image_height), np.int32(kernel_size), np.int32(bias),
                           block=block_size, grid=grid_size)

    # Copiar el resultado de vuelta a la memoria de la CPU
    drv.memcpy_dtoh(embossed_image_on_gpu, embossed_image_on_gpu_mem)

    # Medir el tiempo y la memoria de la GPU después de terminar
    end_time = time.time()  # Tiempo después de aplicar el filtro
    free_memory_after, total_memory_after = drv.mem_get_info()  # Memoria libre después de aplicar el filtro

    # Calcular el tiempo de ejecución y la memoria utilizada
    elapsed_time = end_time - start_time  # Tiempo en segundos
    memory_used = total_memory_before - free_memory_after  # Memoria utilizada en bytes

    # Imprimir los resultados
    print(f"Execution time: {elapsed_time:.4f} seconds")
    print(f"Memory used: {memory_used / 1024 / 1024:.4f} MB")

    # Devolver la imagen procesada
    return embossed_image_on_gpu

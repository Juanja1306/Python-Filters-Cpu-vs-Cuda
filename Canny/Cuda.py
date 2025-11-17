import numpy as np
import cv2
import time
import csv
import os
from datetime import datetime

# Configurar variables de entorno de Visual Studio para nvcc
vs_compiler_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64"

if vs_compiler_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + vs_compiler_path

vs_include = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\ATLMFC\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\winrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\cppwinrt"

os.environ["INCLUDE"] = vs_include

vs_lib = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\ATLMFC\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\um\x64"

os.environ["LIB"] = vs_lib

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


def exp_manual(x):
    """Calcula e^x manualmente usando la serie de Taylor."""
    resultado = 1.0
    termino = 1.0
    
    for n in range(1, 50):
        termino *= x / n
        resultado += termino
        if abs(termino) < 1e-15:
            break
    
    return resultado


def generar_kernel_gaussiano(tamaño, sigma):
    """Genera un kernel gaussiano manualmente."""
    kernel = np.zeros((tamaño, tamaño), dtype=np.float32)
    centro = tamaño // 2
    
    suma_total = 0.0
    for y in range(tamaño):
        for x in range(tamaño):
            dx = x - centro
            dy = y - centro
            valor = exp_manual(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
            kernel[y, x] = valor
            suma_total += valor
    
    for y in range(tamaño):
        for x in range(tamaño):
            kernel[y, x] /= suma_total
    
    return kernel


def convertir_a_grises(imagen):
    """Convierte imagen a escala de grises."""
    b, g, r = cv2.split(imagen)
    grises = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    return grises


# ==================== KERNELS CUDA ====================

# Kernel para suavizado gaussiano
kernel_suavizado = """
__global__ void convolve_gaussian(float *imagen, float *kernel, float *resultado, 
                                   int altura, int ancho, int tam_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    int offset = tam_kernel / 2;
    float suma = 0.0f;
    
    for (int ky = 0; ky < tam_kernel; ky++) {
        for (int kx = 0; kx < tam_kernel; kx++) {
            int py = y + ky - offset;
            int px = x + kx - offset;
            
            // Manejar bordes
            if (py < 0) py = 0;
            if (py >= altura) py = altura - 1;
            if (px < 0) px = 0;
            if (px >= ancho) px = ancho - 1;
            
            suma += imagen[py * ancho + px] * kernel[ky * tam_kernel + kx];
        }
    }
    
    // Clamp entre 0 y 255
    if (suma < 0.0f) suma = 0.0f;
    if (suma > 255.0f) suma = 255.0f;
    
    resultado[y * ancho + x] = suma;
}
"""

# Kernel para cálculo de gradientes con Sobel
kernel_gradientes = """
__global__ void calcular_gradientes(float *imagen, float *magnitud, float *direccion,
                                      int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    // Bordes = 0
    if (x == 0 || x == ancho - 1 || y == 0 || y == altura - 1) {
        magnitud[y * ancho + x] = 0.0f;
        direccion[y * ancho + x] = 0.0f;
        return;
    }
    
    // Operadores Sobel
    float gx = 0.0f, gy = 0.0f;
    
    // Sobel X
    gx += -1.0f * imagen[(y-1) * ancho + (x-1)];
    gx +=  0.0f * imagen[(y-1) * ancho + x];
    gx +=  1.0f * imagen[(y-1) * ancho + (x+1)];
    gx += -2.0f * imagen[y * ancho + (x-1)];
    gx +=  0.0f * imagen[y * ancho + x];
    gx +=  2.0f * imagen[y * ancho + (x+1)];
    gx += -1.0f * imagen[(y+1) * ancho + (x-1)];
    gx +=  0.0f * imagen[(y+1) * ancho + x];
    gx +=  1.0f * imagen[(y+1) * ancho + (x+1)];
    
    // Sobel Y
    gy += -1.0f * imagen[(y-1) * ancho + (x-1)];
    gy += -2.0f * imagen[(y-1) * ancho + x];
    gy += -1.0f * imagen[(y-1) * ancho + (x+1)];
    gy +=  0.0f * imagen[y * ancho + (x-1)];
    gy +=  0.0f * imagen[y * ancho + x];
    gy +=  0.0f * imagen[y * ancho + (x+1)];
    gy +=  1.0f * imagen[(y+1) * ancho + (x-1)];
    gy +=  2.0f * imagen[(y+1) * ancho + x];
    gy +=  1.0f * imagen[(y+1) * ancho + (x+1)];
    
    magnitud[y * ancho + x] = sqrtf(gx * gx + gy * gy);
    direccion[y * ancho + x] = atan2f(gy, gx);
}
"""

# Kernel para supresión no-máxima
kernel_supresion = """
#define PI 3.14159265359f

__global__ void supresion_no_maxima(float *magnitud, float *direccion, float *resultado,
                                     int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    // Bordes = 0
    if (x == 0 || x == ancho - 1 || y == 0 || y == altura - 1) {
        resultado[y * ancho + x] = 0.0f;
        return;
    }
    
    float dir = direccion[y * ancho + x];
    float mag = magnitud[y * ancho + x];
    
    // Convertir a grados
    float angulo = dir * 180.0f / PI;
    if (angulo < 0.0f) angulo += 180.0f;
    
    float mag1, mag2;
    
    // Determinar dirección y vecinos
    if ((angulo >= 0.0f && angulo < 22.5f) || (angulo >= 157.5f && angulo <= 180.0f)) {
        // 0 grados (horizontal)
        mag1 = magnitud[y * ancho + (x-1)];
        mag2 = magnitud[y * ancho + (x+1)];
    } else if (angulo >= 22.5f && angulo < 67.5f) {
        // 45 grados (diagonal /)
        mag1 = magnitud[(y-1) * ancho + (x+1)];
        mag2 = magnitud[(y+1) * ancho + (x-1)];
    } else if (angulo >= 67.5f && angulo < 112.5f) {
        // 90 grados (vertical)
        mag1 = magnitud[(y-1) * ancho + x];
        mag2 = magnitud[(y+1) * ancho + x];
    } else {
        // 135 grados (diagonal \)
        mag1 = magnitud[(y-1) * ancho + (x-1)];
        mag2 = magnitud[(y+1) * ancho + (x+1)];
    }
    
    // Suprimir si no es máximo local
    if (mag >= mag1 && mag >= mag2) {
        resultado[y * ancho + x] = mag;
    } else {
        resultado[y * ancho + x] = 0.0f;
    }
}
"""

# Kernel para umbralización (primera fase)
kernel_umbralizacion = """
__global__ void umbralizar(float *magnitud, unsigned char *resultado,
                           float umbral_alto, float umbral_bajo,
                           int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    float mag = magnitud[y * ancho + x];
    
    if (mag >= umbral_alto) {
        resultado[y * ancho + x] = 255;  // Borde fuerte
    } else if (mag >= umbral_bajo) {
        resultado[y * ancho + x] = 128;  // Borde débil
    } else {
        resultado[y * ancho + x] = 0;    // No es borde
    }
}
"""

# Kernel para histéresis (conectar bordes)
kernel_histeresis = """
__global__ void conectar_bordes(unsigned char *imagen, unsigned char *resultado,
                                 int altura, int ancho, int *cambio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    // Bordes
    if (x == 0 || x == ancho - 1 || y == 0 || y == altura - 1) {
        resultado[y * ancho + x] = imagen[y * ancho + x];
        return;
    }
    
    unsigned char val = imagen[y * ancho + x];
    
    // Si es borde débil, verificar si está conectado a un borde fuerte
    if (val == 128) {
        bool conectado = false;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (imagen[(y + dy) * ancho + (x + dx)] == 255) {
                    conectado = true;
                    break;
                }
            }
            if (conectado) break;
        }
        
        if (conectado) {
            resultado[y * ancho + x] = 255;
            *cambio = 1;  // Marcar que hubo cambio
        } else {
            resultado[y * ancho + x] = 128;
        }
    } else {
        resultado[y * ancho + x] = val;
    }
}
"""

# Kernel para limpieza final
kernel_limpieza = """
__global__ void limpiar_bordes_debiles(unsigned char *imagen, unsigned char *resultado,
                                        int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    unsigned char val = imagen[y * ancho + x];
    
    if (val == 128) {
        resultado[y * ancho + x] = 0;  // Eliminar bordes débiles no conectados
    } else {
        resultado[y * ancho + x] = val;
    }
}
"""


def aplicar_canny_cuda(imagen_grises, tamaño_kernel=5, sigma=1.4):
    """
    Aplica el filtro Canny completo usando CUDA.
    
    Args:
        imagen_grises: Imagen en escala de grises (numpy array)
        tamaño_kernel: Tamaño del kernel gaussiano (debe ser impar)
        sigma: Desviación estándar del filtro gaussiano
    
    Returns:
        numpy.ndarray: Imagen con bordes detectados
    """
    altura, ancho = imagen_grises.shape
    
    # Compilar kernels
    mod_suavizado = SourceModule(kernel_suavizado)
    mod_gradientes = SourceModule(kernel_gradientes)
    mod_supresion = SourceModule(kernel_supresion)
    mod_umbralizacion = SourceModule(kernel_umbralizacion)
    mod_histeresis = SourceModule(kernel_histeresis)
    mod_limpieza = SourceModule(kernel_limpieza)
    
    convolve_func = mod_suavizado.get_function("convolve_gaussian")
    gradientes_func = mod_gradientes.get_function("calcular_gradientes")
    supresion_func = mod_supresion.get_function("supresion_no_maxima")
    umbralizar_func = mod_umbralizacion.get_function("umbralizar")
    conectar_func = mod_histeresis.get_function("conectar_bordes")
    limpiar_func = mod_limpieza.get_function("limpiar_bordes_debiles")
    
    # Configurar bloques y grids
    block_size = (16, 16, 1)
    grid_size = (
        (ancho + block_size[0] - 1) // block_size[0],
        (altura + block_size[1] - 1) // block_size[1],
        1
    )
    
    print("  1) Generando kernel gaussiano...")
    kernel_gauss = generar_kernel_gaussiano(tamaño_kernel, sigma)
    tam_kernel = kernel_gauss.shape[0]
    
    # Transferir a GPU
    imagen_gpu = gpuarray.to_gpu(imagen_grises.astype(np.float32))
    kernel_gpu = gpuarray.to_gpu(kernel_gauss.flatten().astype(np.float32))
    resultado_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    
    # Paso 1: Suavizado gaussiano
    print("  2) Aplicando suavizado gaussiano (GPU)...")
    convolve_func(
        imagen_gpu, kernel_gpu, resultado_gpu,
        np.int32(altura), np.int32(ancho), np.int32(tam_kernel),
        block=block_size, grid=grid_size
    )
    
    suavizada_gpu = resultado_gpu
    
    # Paso 2: Cálculo de gradientes
    print("  3) Calculando gradientes (Sobel) (GPU)...")
    magnitud_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    direccion_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    
    gradientes_func(
        suavizada_gpu, magnitud_gpu, direccion_gpu,
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Paso 3: Supresión no-máxima
    print("  4) Aplicando supresión no-máxima (GPU)...")
    suprimida_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    
    supresion_func(
        magnitud_gpu, direccion_gpu, suprimida_gpu,
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Paso 4: Calcular umbrales
    print("  5) Calculando umbrales automáticos...")
    suprimida_cpu = suprimida_gpu.get()
    
    # Calcular máximo manualmente
    max_magnitud = 0.0
    for y in range(altura):
        for x in range(ancho):
            if suprimida_cpu[y, x] > max_magnitud:
                max_magnitud = suprimida_cpu[y, x]
    
    umbral_alto = max_magnitud * 0.15
    umbral_bajo = umbral_alto * 0.4
    
    # Paso 5: Umbralización
    print("  6) Aplicando umbralización con histéresis (GPU)...")
    resultado_uint8_gpu = gpuarray.empty((altura, ancho), dtype=np.uint8)
    
    umbralizar_func(
        suprimida_gpu, resultado_uint8_gpu,
        np.float32(umbral_alto), np.float32(umbral_bajo),
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Paso 6: Histéresis (conectar bordes) - iterativo
    print("  7) Conectando bordes débiles a fuertes (GPU)...")
    temp_gpu = gpuarray.empty_like(resultado_uint8_gpu)
    cambio_gpu = gpuarray.zeros(1, dtype=np.int32)
    
    MAX_ITERACIONES = 100
    for iteracion in range(MAX_ITERACIONES):
        cambio_gpu.fill(0)
        
        conectar_func(
            resultado_uint8_gpu, temp_gpu,
            np.int32(altura), np.int32(ancho),
            cambio_gpu,
            block=block_size, grid=grid_size
        )
        
        # Copiar resultado temporal al principal
        resultado_uint8_gpu = temp_gpu.copy()
        
        # Verificar si hubo cambios
        if cambio_gpu.get()[0] == 0:
            break
    
    # Paso 7: Limpieza final
    print("  8) Limpiando bordes débiles no conectados (GPU)...")
    resultado_final_gpu = gpuarray.empty_like(resultado_uint8_gpu)
    
    limpiar_func(
        resultado_uint8_gpu, resultado_final_gpu,
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Transferir resultado de vuelta a CPU
    resultado = resultado_final_gpu.get()
    
    return resultado


def main():
    try:
        # Configuración
        archivo_entrada = os.path.join("Images", "casa.jpg")
        archivo_salida = os.path.join("Salida", "canny_cuda.jpg")
        archivo_resultados = os.path.join("Resultados", "canny_cuda.csv")
        
        print("=== FILTRO CANNY DE DETECCIÓN DE BORDES (CUDA/GPU) ===")
        print(f"Leyendo imagen: {archivo_entrada}")
        
        if not os.path.exists(archivo_entrada):
            print(f"Error: No se encontró el archivo {archivo_entrada}")
            return
        
        # Leer la imagen
        imagen_original = cv2.imread(archivo_entrada)
        if imagen_original is None:
            print(f"Error: No se pudo leer la imagen {archivo_entrada}")
            return
        
        altura, ancho = imagen_original.shape[:2]
        print(f"Dimensiones: {ancho}x{altura}")
        
        # Información de la GPU
        print("\n--- INFORMACIÓN DE LA GPU ---")
        device = cuda.Device(0)
        print(f"GPU: {device.name()}")
        print(f"Compute Capability: {device.compute_capability()}")
        total_memory = device.total_memory() // (1024**2)
        print(f"Memoria Total: {total_memory} MB")
        
        # Convertir a escala de grises
        print("\n--- CONVERSIÓN A ESCALA DE GRISES ---")
        imagen_grises = convertir_a_grises(imagen_original)
        print("Conversión completada.")
        
        # Tamaños de kernel a probar: 1%, 3% y 5%
        porcentajes_kernel = [0.01, 0.03, 0.05]
        tamaño_minimo = min(ancho, altura)
        
        # Procesar con cada tamaño de kernel
        for idx, porcentaje_kernel in enumerate(porcentajes_kernel, 1):
            print(f"\n{'='*70}")
            print(f"PROCESAMIENTO {idx}/3 - KERNEL {int(porcentaje_kernel*100)}%")
            print(f"{'='*70}")
            
            # Calcular tamaño del kernel
            tamaño_kernel_float = tamaño_minimo * porcentaje_kernel
            tamaño_kernel = int(tamaño_kernel_float)
            if tamaño_kernel % 2 == 0:
                tamaño_kernel += 1
            if tamaño_kernel < 3:
                tamaño_kernel = 3
            
            sigma = tamaño_kernel / 6.0
            
            print(f"\nKernel gaussiano: {tamaño_kernel}x{tamaño_kernel} (sigma={sigma:.2f})")
            
            # Ejecutar filtro Canny con CUDA
            print("\n--- PROCESAMIENTO CON CUDA (GPU) ---")
            print("Aplicando filtro Canny...")
            
            # Calentar GPU (primera ejecución)
            print("\nCalentamiento de GPU...")
            _ = aplicar_canny_cuda(imagen_grises, tamaño_kernel, sigma)
            
            # Ejecución medida
            print("\nEjecución medida:")
            tiempo_inicio = time.time()
            imagen_bordes = aplicar_canny_cuda(imagen_grises, tamaño_kernel, sigma)
            tiempo_fin = time.time()
            
            tiempo_ms = int((tiempo_fin - tiempo_inicio) * 1000)
            print(f"\nTiempo de ejecución: {tiempo_ms} ms")
            
            # Guardar imagen con el porcentaje en el nombre
            archivo_salida_kernel = os.path.join("Salida", f"canny_cuda_{int(porcentaje_kernel*100)}pct.jpg")
            os.makedirs(os.path.dirname(archivo_salida_kernel), exist_ok=True)
            cv2.imwrite(archivo_salida_kernel, imagen_bordes)
            print(f"Imagen con bordes detectados guardada: {archivo_salida_kernel}")
            
            # Guardar resultados en CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(archivo_resultados), exist_ok=True)
            
            # Verificar si el archivo existe para decidir si escribir el header
            archivo_existe = os.path.exists(archivo_resultados)
            
            with open(archivo_resultados, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Solo escribir header si el archivo no existe
                if not archivo_existe:
                    writer.writerow(['Timestamp', 'Kernel_Percent', 'Kernel_Size', 'Time_ms', 'Method'])
                writer.writerow([timestamp, int(porcentaje_kernel*100), tamaño_kernel, tiempo_ms, 'CUDA'])
        
        print(f"\n{'='*70}")
        print("=== RESUMEN COMPLETO ===")
        print(f"Procesamiento completado con 3 tamaños de kernel: 1%, 3%, 5%")
        print(f"Resultados guardados en: {archivo_resultados}")
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


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
    """
    Calcula e^x manualmente usando la serie de Taylor.
    e^x = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
    
    Args:
        x: Exponente
    
    Returns:
        float: e^x
    """
    resultado = 1.0
    termino = 1.0
    
    # Usar suficientes términos para buena precisión
    for n in range(1, 50):
        termino *= x / n
        resultado += termino
        
        # Si el término es muy pequeño, podemos parar
        if abs(termino) < 1e-15:
            break
    
    return resultado


def generar_kernel_gaussiano(tamanio, sigma):
    """
    Genera un kernel gaussiano manualmente.
    
    Args:
        tamanio: tamanio del kernel (debe ser impar)
        sigma: Desviación estándar del filtro gaussiano
    
    Returns:
        numpy.ndarray: Kernel gaussiano normalizado
    """
    kernel = np.zeros((tamanio, tamanio), dtype=np.float32)
    centro = tamanio // 2
    
    # Fórmula gaussiana: G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
    for y in range(tamanio):
        for x in range(tamanio):
            dx = x - centro
            dy = y - centro
            valor = exp_manual(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
            kernel[y, x] = valor
    
    # Normalizar el kernel para que la suma sea 1
    # Calcular la suma total manualmente
    suma_total = 0.0
    for y in range(tamanio):
        for x in range(tamanio):
            suma_total += kernel[y, x]
    
    # Dividir cada elemento por la suma total
    for y in range(tamanio):
        for x in range(tamanio):
            kernel[y, x] /= suma_total
    
    return kernel


def convertir_a_grises(imagen):
    """
    Convierte imagen a escala de grises.
    
    Args:
        imagen: Imagen BGR de OpenCV (numpy array)
    
    Returns:
        numpy.ndarray: Imagen en escala de grises
    """
    # OpenCV lee imágenes en formato BGR
    b, g, r = cv2.split(imagen)
    
    # Fórmula estándar para convertir a escala de grises
    # Promedio ponderado que considera la percepción humana
    grises = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    
    return grises


def aplicar_convolucion_cuda(imagen, kernel):
    """
    Aplica convolución usando CUDA en la GPU.
    
    Args:
        imagen: Imagen en escala de grises (numpy array 2D)
        kernel: Kernel de convolución (numpy array 2D)
    
    Returns:
        numpy.ndarray: Imagen filtrada
    """
    altura, ancho = imagen.shape
    tam_kernel = kernel.shape[0]
    offset = tam_kernel // 2
    
    # Kernel CUDA para convolución
    kernel_code = """
    __global__ void convolucion_gaussiana(
        float *imagen, 
        float *kernel, 
        float *resultado, 
        int altura, 
        int ancho, 
        int tam_kernel, 
        int offset)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (idx >= ancho || idy >= altura) {
            return;
        }
        
        float suma = 0.0f;
        
        for (int ky = 0; ky < tam_kernel; ky++) {
            for (int kx = 0; kx < tam_kernel; kx++) {
                int py = idy + ky - offset;
                int px = idx + kx - offset;
                
                // Manejar bordes con extensión (replicar píxeles del borde)
                if (py < 0) py = 0;
                if (py >= altura) py = altura - 1;
                if (px < 0) px = 0;
                if (px >= ancho) px = ancho - 1;
                
                suma += imagen[py * ancho + px] * kernel[ky * tam_kernel + kx];
            }
        }
        
        // Asegurar que el valor esté en el rango válido [0, 255]
        if (suma < 0.0f) suma = 0.0f;
        if (suma > 255.0f) suma = 255.0f;
        
        resultado[idy * ancho + idx] = suma;
    }
    """
    
    # Compilar el kernel CUDA
    mod = SourceModule(kernel_code)
    convolucion_func = mod.get_function("convolucion_gaussiana")
    
    # Preparar datos en la GPU
    imagen_gpu = cuda.mem_alloc(imagen.nbytes)
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)
    resultado_gpu = cuda.mem_alloc(imagen.nbytes)
    
    # Transferir datos a la GPU
    cuda.memcpy_htod(imagen_gpu, imagen)
    cuda.memcpy_htod(kernel_gpu, kernel)
    
    # Configurar grid y block size
    block_size = (16, 16, 1)
    grid_size = ((ancho + block_size[0] - 1) // block_size[0], 
                  (altura + block_size[1] - 1) // block_size[1], 1)
    
    # Ejecutar kernel
    convolucion_func(
        imagen_gpu, kernel_gpu, resultado_gpu,
        np.int32(altura), np.int32(ancho), 
        np.int32(tam_kernel), np.int32(offset),
        block=block_size, grid=grid_size
    )
    
    # Transferir resultado de vuelta a CPU
    resultado = np.empty_like(imagen)
    cuda.memcpy_dtoh(resultado, resultado_gpu)
    
    return resultado.astype(np.uint8)


def main():
    try:
        # Configuración (usar os.path.join para compatibilidad Windows/Linux)
        archivo_entrada = os.path.join("Images", "casa.jpg")
        archivo_resultados = os.path.join("Resultados", "gaussiana_cuda.csv")
        
        print("=== FILTRO GAUSSIANO POR CONVOLUCIÓN (CUDA) ===")
        print(f"Leyendo imagen: {archivo_entrada}")
        
        # Información de la GPU
        print(f"GPU: {cuda.Device(0).name()}")
        print(f"Memoria GPU: {cuda.Device(0).total_memory() / (1024**3):.2f} GB")
        
        # Verificar que el archivo existe
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
        
        # Convertir a escala de grises (solo una vez)
        print("\n--- CONVERSIÓN A ESCALA DE GRISES ---")
        imagen_grises = convertir_a_grises(imagen_original)
        print("Conversión completada.")
        
        # tamanios de kernel a probar: 5%, 10% y 15%
        porcentajes_kernel = [0.05, 0.10, 0.15]
        tamanio_minimo = min(ancho, altura)
        
        # Procesar con cada tamanio de kernel
        for idx, porcentaje_kernel in enumerate(porcentajes_kernel, 1):
            print(f"\n{'='*60}")
            print(f"PROCESAMIENTO {idx}/3 - KERNEL {int(porcentaje_kernel*100)}%")
            print(f"{'='*60}")
            
            # Calcular tamanio del kernel
            tamanio_kernel_float = tamanio_minimo * porcentaje_kernel
            tamanio_kernel = int(tamanio_kernel_float)
            if tamanio_kernel % 2 == 0:
                tamanio_kernel += 1
            if tamanio_kernel < 3:
                tamanio_kernel = 3
            
            sigma = tamanio_kernel / 6.0
            
            # Generar kernel gaussiano
            print(f"\n--- GENERANDO KERNEL {int(porcentaje_kernel*100)}% ---")
            print(f"Kernel gaussiano {tamanio_kernel}x{tamanio_kernel} (sigma={sigma:.2f})...")
            kernel_gaussiano = generar_kernel_gaussiano(tamanio_kernel, sigma)
            
            # Ejecutar convolución CUDA
            print("\n--- PROCESAMIENTO CUDA (GPU) ---")
            print("Aplicando convolución gaussiana en GPU...")
            
            tiempo_inicio = time.time()
            imagen_filtrada = aplicar_convolucion_cuda(imagen_grises, kernel_gaussiano)
            tiempo_fin = time.time()
            
            tiempo_ms = int((tiempo_fin - tiempo_inicio) * 1000)
            print(f"Tiempo de ejecución: {tiempo_ms} ms ({tiempo_ms / 1000:.2f} segundos)")
            
            # Guardar la imagen procesada
            archivo_salida = os.path.join("Salida", f"gaussiana_cuda_{int(porcentaje_kernel*100)}pct.jpg")
            cv2.imwrite(archivo_salida, imagen_filtrada)
            print(f"Imagen filtrada guardada: {archivo_salida}")
            
            # Guardar resultados en archivo CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            archivo_existe = os.path.exists(archivo_resultados)
            
            with open(archivo_resultados, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not archivo_existe:
                    writer.writerow(['Timestamp', 'Kernel_Percent', 'Kernel_Size', 'Time_ms', 'Method'])
                    archivo_existe = True
                writer.writerow([timestamp, int(porcentaje_kernel*100), tamanio_kernel, tiempo_ms, 'CUDA'])
        
        print(f"\n{'='*60}")
        print("=== RESUMEN COMPLETO ===")
        print(f"Procesamiento completado con 3 tamanios de kernel: 5%, 10%, 15%")
        print(f"Resultados guardados en: {archivo_resultados}")
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


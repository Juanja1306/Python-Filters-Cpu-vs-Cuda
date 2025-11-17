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


def generar_kernel_gaussiano(tamaño, sigma):
    """
    Genera un kernel gaussiano manualmente.
    
    Args:
        tamaño: Tamaño del kernel (debe ser impar)
        sigma: Desviación estándar del filtro gaussiano
    
    Returns:
        numpy.ndarray: Kernel gaussiano normalizado
    """
    kernel = np.zeros((tamaño, tamaño), dtype=np.float32)
    centro = tamaño // 2
    
    # Fórmula gaussiana: G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
    for y in range(tamaño):
        for x in range(tamaño):
            dx = x - centro
            dy = y - centro
            valor = np.exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
            kernel[y, x] = valor
    
    # Normalizar el kernel para que la suma sea 1
    kernel /= np.sum(kernel)
    
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
        # Configuración
        archivo_entrada = r"Images\casa.jpg"
        archivo_salida = r"Salida\gaussiana_cuda.jpg"
        archivo_resultados = r"Resultados\gaussiana_cuda.csv"
        
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
        
        # Obtener número de procesadores disponibles
        procesadores_disponibles = os.cpu_count()
        print(f"Procesadores CPU disponibles: {procesadores_disponibles}")
        
        # Configuración del kernel como porcentaje del tamaño de la imagen
        porcentaje_kernel = 0.05  # 5% del tamaño de la imagen (puedes ajustar este valor)
        tamaño_minimo = min(ancho, altura)
        tamaño_kernel_float = tamaño_minimo * porcentaje_kernel
        
        # Asegurar que el tamaño del kernel sea impar (necesario para kernels gaussianos)
        tamaño_kernel = int(tamaño_kernel_float)
        if tamaño_kernel % 2 == 0:
            tamaño_kernel += 1
        
        # Asegurar un tamaño mínimo razonable (al menos 3x3)
        if tamaño_kernel < 3:
            tamaño_kernel = 3
            if tamaño_kernel % 2 == 0:
                tamaño_kernel = 3
        
        sigma = tamaño_kernel / 6.0
        
        # Generar kernel gaussiano
        print("\n--- GENERANDO KERNEL ---")
        print(f"Porcentaje del tamaño de imagen: {porcentaje_kernel * 100:.2f}%")
        print(f"Tamaño mínimo de imagen: {tamaño_minimo}")
        print(f"Kernel gaussiano {tamaño_kernel}x{tamaño_kernel} (sigma={sigma:.2f})...")
        kernel_gaussiano = generar_kernel_gaussiano(tamaño_kernel, sigma)
        
        # Convertir a escala de grises
        print("\n--- CONVERSIÓN A ESCALA DE GRISES ---")
        imagen_grises = convertir_a_grises(imagen_original)
        print("Conversión completada.")
        
        # Ejecutar convolución CUDA
        print("\n--- PROCESAMIENTO CUDA ---")
        print("Aplicando convolución gaussiana en GPU...")
        
        tiempo_inicio = time.time()
        imagen_filtrada = aplicar_convolucion_cuda(imagen_grises, kernel_gaussiano)
        tiempo_fin = time.time()
        
        tiempo_ms = int((tiempo_fin - tiempo_inicio) * 1000)
        print(f"Tiempo de ejecución: {tiempo_ms} ms")
        
        # Guardar la imagen procesada
        cv2.imwrite(archivo_salida, imagen_filtrada)
        print(f"Imagen filtrada guardada: {archivo_salida}")
        
        # Mostrar resumen de resultados
        print("\n=== RESUMEN DE RENDIMIENTO ===")
        print(f"Tiempo de ejecución: {tiempo_ms} ms")
        print(f"Tiempo de ejecución: {tiempo_ms / 1000:.2f} segundos")
        
        # Guardar resultados en archivo CSV
        # Timestamp debe estar en formato ISO 8601
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Verificar si el archivo existe
        archivo_existe = os.path.exists(archivo_resultados)
        
        # Si existe, agregar fila; si no, crear con headers
        with open(archivo_resultados, 'a' if archivo_existe else 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Si el archivo no existe, escribir headers
            if not archivo_existe:
                writer.writerow(['Timestamp', 'Time'])
            # Agregar la nueva fila de datos
            writer.writerow([timestamp, tiempo_ms])
        
        print(f"\nResultados guardados en: {archivo_resultados}")
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


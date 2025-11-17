import numpy as np
import cv2
import time
import csv
import os
from multiprocessing import Pool, shared_memory
import multiprocessing as mp


def generar_kernel_gaussiano(tamaño, sigma):
    """
    Genera un kernel gaussiano manualmente.
    
    Args:
        tamaño: Tamaño del kernel (debe ser impar)
        sigma: Desviación estándar del filtro gaussiano
    
    Returns:
        numpy.ndarray: Kernel gaussiano normalizado
    """
    kernel = np.zeros((tamaño, tamaño), dtype=np.float64)
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
    grises = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    
    return grises


def procesar_banda(args):
    """
    Procesa una banda de la imagen (función para multiprocessing).
    
    Args:
        args: Tupla con (y_inicio, y_fin, ancho, altura, tam_kernel, offset, 
              shm_imagen_name, shm_kernel_name, shm_resultado_name)
    
    Returns:
        None (escribe directamente en memoria compartida)
    """
    (y_inicio, y_fin, ancho, altura, tam_kernel, offset,
     shm_imagen_name, shm_kernel_name, shm_resultado_name) = args
    
    # Acceder a memoria compartida
    shm_imagen = shared_memory.SharedMemory(name=shm_imagen_name)
    shm_kernel = shared_memory.SharedMemory(name=shm_kernel_name)
    shm_resultado = shared_memory.SharedMemory(name=shm_resultado_name)
    
    # Crear arrays numpy desde memoria compartida
    imagen = np.ndarray((altura, ancho), dtype=np.uint8, buffer=shm_imagen.buf)
    kernel = np.ndarray((tam_kernel, tam_kernel), dtype=np.float64, buffer=shm_kernel.buf)
    resultado = np.ndarray((altura, ancho), dtype=np.float64, buffer=shm_resultado.buf)
    
    # Procesar la banda de filas asignada
    for y in range(y_inicio, y_fin):
        for x in range(ancho):
            suma = 0.0
            
            # Aplicar el kernel
            for ky in range(tam_kernel):
                for kx in range(tam_kernel):
                    py = y + ky - offset
                    px = x + kx - offset
                    
                    # Manejar bordes con extensión (replicar píxeles del borde)
                    if py < 0:
                        py = 0
                    if py >= altura:
                        py = altura - 1
                    if px < 0:
                        px = 0
                    if px >= ancho:
                        px = ancho - 1
                    
                    suma += imagen[py, px] * kernel[ky, kx]
            
            # Asegurar que el valor esté en el rango válido [0, 255]
            resultado[y, x] = np.clip(suma, 0, 255)
    
    # Cerrar memoria compartida
    shm_imagen.close()
    shm_kernel.close()
    shm_resultado.close()
    
    return None


def aplicar_convolucion_paralela(imagen, kernel, num_procesos):
    """
    Aplica convolución manual a la imagen en escala de grises (PARALELO con multiprocessing).
    
    Args:
        imagen: Imagen en escala de grises (numpy array 2D)
        kernel: Kernel de convolución (numpy array 2D)
        num_procesos: Número de procesos a usar
    
    Returns:
        numpy.ndarray: Imagen filtrada
    """
    altura, ancho = imagen.shape
    tam_kernel = kernel.shape[0]
    offset = tam_kernel // 2
    
    # Crear memoria compartida para la imagen, kernel y resultado
    shm_imagen = shared_memory.SharedMemory(create=True, size=imagen.nbytes)
    shm_kernel = shared_memory.SharedMemory(create=True, size=kernel.nbytes)
    shm_resultado = shared_memory.SharedMemory(create=True, size=imagen.nbytes * 8)  # float64
    
    # Copiar datos a memoria compartida
    imagen_shm = np.ndarray(imagen.shape, dtype=imagen.dtype, buffer=shm_imagen.buf)
    kernel_shm = np.ndarray(kernel.shape, dtype=kernel.dtype, buffer=shm_kernel.buf)
    resultado_shm = np.ndarray((altura, ancho), dtype=np.float64, buffer=shm_resultado.buf)
    
    np.copyto(imagen_shm, imagen)
    np.copyto(kernel_shm, kernel)
    resultado_shm.fill(0.0)
    
    # Dividir la imagen en bandas horizontales
    filas_por_proceso = altura // num_procesos
    
    # Preparar argumentos para cada proceso
    args_list = []
    for p in range(num_procesos):
        y_inicio = p * filas_por_proceso
        y_fin = altura if (p == num_procesos - 1) else (p + 1) * filas_por_proceso
        
        args_list.append((
            y_inicio, y_fin, ancho, altura, tam_kernel, offset,
            shm_imagen.name, shm_kernel.name, shm_resultado.name
        ))
    
    # Ejecutar en paralelo
    with Pool(processes=num_procesos) as pool:
        pool.map(procesar_banda, args_list)
    
    # Copiar resultado desde memoria compartida
    resultado = np.ndarray((altura, ancho), dtype=np.float64)
    np.copyto(resultado, resultado_shm)
    
    # Limpiar memoria compartida
    shm_imagen.close()
    shm_kernel.close()
    shm_resultado.close()
    shm_imagen.unlink()
    shm_kernel.unlink()
    shm_resultado.unlink()
    
    return resultado.astype(np.uint8)


def aplicar_convolucion_secuencial(imagen, kernel):
    """
    Aplica convolución manual a la imagen en escala de grises (SECUENCIAL).
    
    Args:
        imagen: Imagen en escala de grises (numpy array 2D)
        kernel: Kernel de convolución (numpy array 2D)
    
    Returns:
        numpy.ndarray: Imagen filtrada
    """
    altura, ancho = imagen.shape
    tam_kernel = kernel.shape[0]
    offset = tam_kernel // 2
    
    resultado = np.zeros((altura, ancho), dtype=np.float64)
    
    # Recorrer cada pixel de la imagen
    for y in range(altura):
        for x in range(ancho):
            suma = 0.0
            
            # Aplicar el kernel
            for ky in range(tam_kernel):
                for kx in range(tam_kernel):
                    py = y + ky - offset
                    px = x + kx - offset
                    
                    # Manejar bordes con extensión (replicar píxeles del borde)
                    if py < 0:
                        py = 0
                    if py >= altura:
                        py = altura - 1
                    if px < 0:
                        px = 0
                    if px >= ancho:
                        px = ancho - 1
                    
                    suma += imagen[py, px] * kernel[ky, kx]
            
            # Asegurar que el valor esté en el rango válido [0, 255]
            resultado[y, x] = np.clip(suma, 0, 255)
    
    return resultado.astype(np.uint8)


def main():
    try:
        # Configuración (usar os.path.join para compatibilidad Windows/Linux)
        archivo_entrada = os.path.join("Images", "casa.jpg")
        archivo_salida = os.path.join("Salida", "gaussiana_paralelo.jpg")
        archivo_resultados = os.path.join("Resultados", "gaussiana_paralelo.csv")
        
        print("=== FILTRO GAUSSIANO POR CONVOLUCIÓN (MULTIPROCESAMIENTO) ===")
        print(f"Leyendo imagen: {archivo_entrada}")
        
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
        print(f"Procesadores disponibles: {procesadores_disponibles}")
        
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
        
        # Array para almacenar los procesos a probar
        procesos_array = [1, 4, 8]
        tiempos = []
        
        # Ejecutar con diferentes números de procesos
        for num_procesos in procesos_array:
            print(f"\n--- PROCESAMIENTO CON {num_procesos} PROCESO(S) ---")
            print("Aplicando convolución gaussiana...")
            
            tiempo_inicio = time.time()
            
            if num_procesos == 1:
                imagen_filtrada = aplicar_convolucion_secuencial(imagen_grises, kernel_gaussiano)
            else:
                imagen_filtrada = aplicar_convolucion_paralela(imagen_grises, kernel_gaussiano, num_procesos)
            
            tiempo_fin = time.time()
            tiempo_ms = int((tiempo_fin - tiempo_inicio) * 1000)
            tiempos.append(tiempo_ms)
            
            print(f"Tiempo de ejecución: {tiempo_ms} ms")
            
            # Guardar la última imagen procesada
            if num_procesos == procesos_array[-1]:
                cv2.imwrite(archivo_salida, imagen_filtrada)
                print(f"Imagen filtrada guardada: {archivo_salida}")
        
        # Mostrar resumen de resultados
        print("\n=== RESUMEN DE RENDIMIENTO ===")
        print("Fórmulas:")
        print("  - Speedup:    S = Ts/Tp")
        print("  - Eficiencia: E = S/p = Ts/(p*Tp)")
        print("\nProcesos(p) | Tiempo(Tp) | Speedup(S) | Eficiencia(E) | Clasificación")
        print("------------|------------|------------|---------------|---------------------------------------")
        
        resultados_csv = []
        for i, num_procesos in enumerate(procesos_array):
            speedup = tiempos[0] / tiempos[i]
            eficiencia_decimal = speedup / num_procesos  # E = S/p
            eficiencia_porcentaje = eficiencia_decimal * 100
            
            # Determinar tipo de aceleramiento
            tipo_aceleramiento = ""
            if speedup > num_procesos:
                tipo_aceleramiento = "Aceleramiento Superlineal (S > p)"
            elif abs(speedup - num_procesos) < 0.1:
                tipo_aceleramiento = "Aceleramiento Lineal (S = p)"
            else:
                tipo_aceleramiento = "Aceleramiento Real (S < p)"
            
            # Determinar clasificación de eficiencia
            tipo_eficiencia = ""
            if abs(eficiencia_decimal - 1.0) < 0.05:
                tipo_eficiencia = "E = 1 (caso lineal)"
            elif eficiencia_decimal < 0.5:
                tipo_eficiencia = "E << 1 (poca eficiencia)"
            elif eficiencia_decimal < 1.0:
                tipo_eficiencia = "E < 1 (caso real)"
            else:
                tipo_eficiencia = "E > 1 (superlineal)"
            
            clasificacion_completa = tipo_aceleramiento + ", " + tipo_eficiencia
            
            print(f"{num_procesos:11d} | {tiempos[i]:10d} | {speedup:9.2f}x | {eficiencia_porcentaje:10.2f}% | {clasificacion_completa}")
            
            resultados_csv.append([num_procesos, tiempos[i], speedup, eficiencia_porcentaje])
        
        # Guardar resultados en archivo CSV
        # Verificar si el archivo existe para decidir si escribir el header
        archivo_existe = os.path.exists(archivo_resultados)
        
        with open(archivo_resultados, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Solo escribir header si el archivo no existe
            if not archivo_existe:
                writer.writerow(['procesos', 'tiempo_ms', 'speedup', 'eficiencia'])
            for row in resultados_csv:
                writer.writerow([row[0], row[1], f"{row[2]:.2f}", f"{row[3]:.2f}"])
        
        print(f"\nResultados guardados en: {archivo_resultados}")
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Necesario para multiprocessing en Windows
    mp.set_start_method('spawn', force=True)
    main()


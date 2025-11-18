import numpy as np
import cv2
import time
import csv
import os
from datetime import datetime


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
            valor = exp_manual(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
            kernel[y, x] = valor
    
    # Normalizar el kernel para que la suma sea 1
    # Calcular la suma total manualmente
    suma_total = 0.0
    for y in range(tamaño):
        for x in range(tamaño):
            suma_total += kernel[y, x]
    
    # Dividir cada elemento por la suma total
    for y in range(tamaño):
        for x in range(tamaño):
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
    grises = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    
    return grises


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
            if suma < 0:
                resultado[y, x] = 0
            elif suma > 255:
                resultado[y, x] = 255
            else:
                resultado[y, x] = suma
    
    return resultado.astype(np.uint8)


def main():
    try:
        # Configuración (usar os.path.join para compatibilidad Windows/Linux)
        archivo_entrada = os.path.join("Images", "casa.jpg")
        archivo_resultados = os.path.join("Resultados", "gaussiana_secuencial.csv")
        
        print("=== FILTRO GAUSSIANO POR CONVOLUCIÓN (SECUENCIAL) ===")
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
        
        # Convertir a escala de grises (solo una vez)
        print("\n--- CONVERSIÓN A ESCALA DE GRISES ---")
        imagen_grises = convertir_a_grises(imagen_original)
        print("Conversión completada.")
        
        # Tamaños de kernel a probar: 5%, 10% y 15%
        porcentajes_kernel = [0.05, 0.10, 0.15]
        tamaño_minimo = min(ancho, altura)
        
        # Procesar con cada tamaño de kernel
        for idx, porcentaje_kernel in enumerate(porcentajes_kernel, 1):
            print(f"\n{'='*60}")
            print(f"PROCESAMIENTO {idx}/3 - KERNEL {int(porcentaje_kernel*100)}%")
            print(f"{'='*60}")
            
            # Calcular tamaño del kernel
            tamaño_kernel_float = tamaño_minimo * porcentaje_kernel
            tamaño_kernel = int(tamaño_kernel_float)
            if tamaño_kernel % 2 == 0:
                tamaño_kernel += 1
            
            # Asegurar un tamaño mínimo razonable (al menos 3x3)
            if tamaño_kernel < 3:
                tamaño_kernel = 3
            
            sigma = tamaño_kernel / 6.0
            
            # Generar kernel gaussiano
            print("\n--- GENERANDO KERNEL ---")
            print(f"Porcentaje del tamaño de imagen: {porcentaje_kernel * 100:.2f}%")
            print(f"Tamaño mínimo de imagen: {tamaño_minimo}")
            print(f"Kernel gaussiano {tamaño_kernel}x{tamaño_kernel} (sigma={sigma:.2f})...")
            kernel_gaussiano = generar_kernel_gaussiano(tamaño_kernel, sigma)
            
            # Ejecutar convolución secuencial
            print("\n--- PROCESAMIENTO SECUENCIAL ---")
            print("Aplicando convolución gaussiana...")
            
            tiempo_inicio = time.time()
            imagen_filtrada = aplicar_convolucion_secuencial(imagen_grises, kernel_gaussiano)
            tiempo_fin = time.time()
            
            tiempo_ms = int((tiempo_fin - tiempo_inicio) * 1000)
            print(f"Tiempo de ejecución: {tiempo_ms} ms ({tiempo_ms / 1000:.2f} segundos)")
            
            # Guardar la imagen procesada
            archivo_salida = os.path.join("Salida", f"gaussiana_secuencial_{int(porcentaje_kernel*100)}pct.jpg")
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
                writer.writerow([timestamp, int(porcentaje_kernel*100), tamaño_kernel, tiempo_ms, 'Secuencial'])
        
        print(f"\n{'='*60}")
        print("=== RESUMEN COMPLETO ===")
        print(f"Procesamiento completado con 3 tamaños de kernel: 5%, 10%, 15%")
        print(f"Resultados guardados en: {archivo_resultados}")
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


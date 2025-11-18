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


def sqrt_manual(x):
    """
    Calcula raíz cuadrada manualmente usando el método de Newton.
    
    Args:
        x: Número del cual calcular la raíz
    
    Returns:
        float: sqrt(x)
    """
    if x < 0:
        return 0
    if x == 0:
        return 0
    
    # Aproximación inicial
    aproximacion = x / 2.0
    
    # Método de Newton: x_n+1 = (x_n + S/x_n) / 2
    for _ in range(20):
        if aproximacion == 0:
            break
        nueva = (aproximacion + x / aproximacion) / 2.0
        if abs(nueva - aproximacion) < 1e-10:
            break
        aproximacion = nueva
    
    return aproximacion


def atan2_manual(y, x):
    """
    Calcula atan2(y, x) manualmente usando aproximación de arcotangente.
    
    Args:
        y: Componente y
        x: Componente x
    
    Returns:
        float: ángulo en radianes
    """
    # Casos especiales
    if x == 0:
        if y > 0:
            return 3.14159265359 / 2.0  # π/2
        elif y < 0:
            return -3.14159265359 / 2.0  # -π/2
        else:
            return 0.0
    
    # Calcular arctan(y/x) usando serie de Taylor para valores pequeños
    # o aproximaciones para valores grandes
    if abs(x) > abs(y):
        # Usar atan(y/x)
        z = y / x
        atan = z - (z**3)/3 + (z**5)/5 - (z**7)/7 + (z**9)/9
        
        if x < 0:
            if y >= 0:
                return atan + 3.14159265359  # π
            else:
                return atan - 3.14159265359  # -π
        return atan
    else:
        # Usar π/2 - atan(x/y)
        z = x / y
        atan = z - (z**3)/3 + (z**5)/5 - (z**7)/7 + (z**9)/9
        
        if y > 0:
            return 3.14159265359 / 2.0 - atan
        else:
            return -3.14159265359 / 2.0 - atan


def generar_kernel_gaussiano(tamanio, sigma):
    """
    Genera un kernel gaussiano manualmente.
    
    Args:
        tamanio: tamanio del kernel (debe ser impar)
        sigma: Desviación estándar del filtro gaussiano
    
    Returns:
        list: Kernel gaussiano normalizado (lista de listas)
    """
    kernel = [[0.0 for _ in range(tamanio)] for _ in range(tamanio)]
    centro = tamanio // 2
    
    # Fórmula gaussiana: G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
    suma_total = 0.0
    for y in range(tamanio):
        for x in range(tamanio):
            dx = x - centro
            dy = y - centro
            valor = exp_manual(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
            kernel[y][x] = valor
            suma_total += valor
    
    # Normalizar el kernel para que la suma sea 1
    for y in range(tamanio):
        for x in range(tamanio):
            kernel[y][x] /= suma_total
    
    return kernel


def convertir_a_grises(imagen):
    """
    Convierte imagen a escala de grises.
    
    Args:
        imagen: Imagen BGR de OpenCV (numpy array)
    
    Returns:
        list: Imagen en escala de grises (lista de listas)
    """
    altura, ancho = imagen.shape[:2]
    grises = [[0 for _ in range(ancho)] for _ in range(altura)]
    
    # Fórmula estándar para convertir a escala de grises
    for y in range(altura):
        for x in range(ancho):
            b, g, r = imagen[y, x]
            valor_gris = int(0.299 * r + 0.587 * g + 0.114 * b)
            grises[y][x] = valor_gris
    
    return grises


def aplicar_suavizado_gaussiano(imagen, kernel):
    """
    Aplica convolución gaussiana para suavizado (SECUENCIAL).
    
    Args:
        imagen: Imagen en escala de grises (lista de listas)
        kernel: Kernel de convolución (lista de listas)
    
    Returns:
        list: Imagen suavizada (lista de listas)
    """
    altura = len(imagen)
    ancho = len(imagen[0])
    tam_kernel = len(kernel)
    offset = tam_kernel // 2
    
    resultado = [[0 for _ in range(ancho)] for _ in range(altura)]
    
    for y in range(altura):
        for x in range(ancho):
            suma = 0.0
            
            for ky in range(tam_kernel):
                for kx in range(tam_kernel):
                    py = y + ky - offset
                    px = x + kx - offset
                    
                    # Manejar bordes con extensión
                    if py < 0:
                        py = 0
                    if py >= altura:
                        py = altura - 1
                    if px < 0:
                        px = 0
                    if px >= ancho:
                        px = ancho - 1
                    
                    suma += imagen[py][px] * kernel[ky][kx]
            
            # Asegurar que el valor esté en el rango válido [0, 255]
            valor = int(suma)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[y][x] = valor
    
    return resultado


def calcular_gradientes_secuencial(imagen):
    """
    Calcula gradientes usando operadores Sobel (SECUENCIAL).
    
    Args:
        imagen: Imagen suavizada (lista de listas)
    
    Returns:
        tuple: (magnitud, direccion) - ambas como listas de listas
    """
    altura = len(imagen)
    ancho = len(imagen[0])
    
    magnitud = [[0.0 for _ in range(ancho)] for _ in range(altura)]
    direccion = [[0.0 for _ in range(ancho)] for _ in range(altura)]
    
    # Operadores Sobel
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    for y in range(1, altura - 1):
        for x in range(1, ancho - 1):
            gx = 0.0
            gy = 0.0
            
            # Aplicar operadores Sobel
            for ky in range(3):
                for kx in range(3):
                    py = y + ky - 1
                    px = x + kx - 1
                    pixel = imagen[py][px]
                    
                    gx += pixel * sobel_x[ky][kx]
                    gy += pixel * sobel_y[ky][kx]
            
            magnitud[y][x] = sqrt_manual(gx * gx + gy * gy)
            direccion[y][x] = atan2_manual(gy, gx)
    
    return magnitud, direccion


def supresion_no_maxima_secuencial(magnitud, direccion):
    """
    Supresión no-máxima (SECUENCIAL).
    
    Args:
        magnitud: Magnitud del gradiente (lista de listas)
        direccion: Dirección del gradiente (lista de listas)
    
    Returns:
        list: Imagen con supresión no-máxima aplicada
    """
    altura = len(magnitud)
    ancho = len(magnitud[0])
    
    resultado = [[0.0 for _ in range(ancho)] for _ in range(altura)]
    
    # Conversión de radianes a grados (π rad = 180°)
    PI = 3.14159265359
    
    for y in range(1, altura - 1):
        for x in range(1, ancho - 1):
            direccion_rad = direccion[y][x]
            mag = magnitud[y][x]
            
            # Convertir a grados
            angulo = direccion_rad * 180.0 / PI
            if angulo < 0:
                angulo += 180
            
            # Redondear dirección a uno de 4 ángulos
            if (angulo >= 0 and angulo < 22.5) or (angulo >= 157.5 and angulo <= 180):
                # 0 grados (horizontal)
                mag1 = magnitud[y][x - 1]
                mag2 = magnitud[y][x + 1]
            elif angulo >= 22.5 and angulo < 67.5:
                # 45 grados (diagonal /)
                mag1 = magnitud[y - 1][x + 1]
                mag2 = magnitud[y + 1][x - 1]
            elif angulo >= 67.5 and angulo < 112.5:
                # 90 grados (vertical)
                mag1 = magnitud[y - 1][x]
                mag2 = magnitud[y + 1][x]
            else:
                # 135 grados (diagonal \)
                mag1 = magnitud[y - 1][x - 1]
                mag2 = magnitud[y + 1][x + 1]
            
            # Suprimir si no es máximo local
            if mag >= mag1 and mag >= mag2:
                resultado[y][x] = mag
            else:
                resultado[y][x] = 0
    
    return resultado


def umbralizacion_histeresis_secuencial(magnitud, umbral_alto, umbral_bajo):
    """
    Umbralización con histéresis (SECUENCIAL).
    
    Args:
        magnitud: Magnitud después de supresión no-máxima (lista de listas)
        umbral_alto: Umbral alto
        umbral_bajo: Umbral bajo
    
    Returns:
        list: Imagen con bordes detectados
    """
    altura = len(magnitud)
    ancho = len(magnitud[0])
    
    resultado = [[0 for _ in range(ancho)] for _ in range(altura)]
    
    # Clasificar píxeles en bordes fuertes, débiles y no-bordes
    for y in range(altura):
        for x in range(ancho):
            if magnitud[y][x] >= umbral_alto:
                resultado[y][x] = 255  # Borde fuerte
            elif magnitud[y][x] >= umbral_bajo:
                resultado[y][x] = 128  # Borde débil
            else:
                resultado[y][x] = 0  # No es borde
    
    # Conectar bordes débiles con bordes fuertes (rastreo de bordes)
    cambio = True
    while cambio:
        cambio = False
        for y in range(1, altura - 1):
            for x in range(1, ancho - 1):
                if resultado[y][x] == 128:  # Si es borde débil
                    # Verificar si está conectado a un borde fuerte
                    conectado = False
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if resultado[y + dy][x + dx] == 255:
                                conectado = True
                                break
                        if conectado:
                            break
                    
                    if conectado:
                        resultado[y][x] = 255  # Promover a borde fuerte
                        cambio = True
    
    # Eliminar bordes débiles no conectados
    for y in range(altura):
        for x in range(ancho):
            if resultado[y][x] == 128:
                resultado[y][x] = 0
    
    return resultado


def aplicar_canny_secuencial(imagen_grises, tamanio_kernel=5, sigma=1.4):
    """
    Filtro Canny completo (SECUENCIAL).
    
    Args:
        imagen_grises: Imagen en escala de grises (lista de listas)
        tamanio_kernel: tamanio del kernel gaussiano
        sigma: Desviación estándar del kernel gaussiano
    
    Returns:
        list: Imagen con bordes detectados
    """
    print(f"  1) Generando kernel gaussiano {tamanio_kernel}x{tamanio_kernel}...")
    kernel_gaussiano = generar_kernel_gaussiano(tamanio_kernel, sigma)
    
    print("  2) Aplicando suavizado gaussiano...")
    suavizada = aplicar_suavizado_gaussiano(imagen_grises, kernel_gaussiano)
    
    print("  3) Calculando gradientes (Sobel)...")
    magnitud, direccion = calcular_gradientes_secuencial(suavizada)
    
    print("  4) Aplicando supresión no-máxima...")
    suprimida = supresion_no_maxima_secuencial(magnitud, direccion)
    
    print("  5) Calculando umbrales automáticos...")
    # Calcular máximo de magnitud
    max_magnitud = 0.0
    for y in range(len(suprimida)):
        for x in range(len(suprimida[0])):
            if suprimida[y][x] > max_magnitud:
                max_magnitud = suprimida[y][x]
    
    umbral_alto = max_magnitud * 0.15
    umbral_bajo = umbral_alto * 0.4
    
    print("  6) Aplicando umbralización con histéresis...")
    resultado = umbralizacion_histeresis_secuencial(suprimida, umbral_alto, umbral_bajo)
    
    return resultado


def main():
    try:
        # Configuración
        archivo_entrada = os.path.join("Images", "casa.jpg")
        archivo_resultados = os.path.join("Resultados", "canny_secuencial.csv")
        
        print("=== FILTRO CANNY DE DETECCIÓN DE BORDES (SECUENCIAL) ===")
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
        
        # Convertir a escala de grises (solo una vez)
        print("\n--- CONVERSIÓN A ESCALA DE GRISES ---")
        imagen_grises = convertir_a_grises(imagen_original)
        print("Conversión completada.")
        
        # tamanios de kernel a probar: 1%, 3% y 5%
        porcentajes_kernel = [0.01, 0.03, 0.05]
        tamanio_minimo = min(ancho, altura)
        
        import numpy as np
        
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
            
            # Ejecutar filtro Canny secuencial
            print("\n--- PROCESAMIENTO SECUENCIAL ---")
            print("Aplicando filtro Canny...")
            print("Pasos: Suavizado → Gradientes → Supresión → Histéresis")
            
            tiempo_inicio = time.time()
            imagen_bordes = aplicar_canny_secuencial(imagen_grises, tamanio_kernel, sigma)
            tiempo_fin = time.time()
            
            tiempo_ms = int((tiempo_fin - tiempo_inicio) * 1000)
            print(f"\nTiempo de ejecución: {tiempo_ms} ms ({tiempo_ms / 1000:.2f} segundos)")
            
            # Convertir resultado a formato numpy para guardar con OpenCV
            resultado_np = np.array(imagen_bordes, dtype=np.uint8)
            
            # Crear directorio de salida si no existe
            archivo_salida = os.path.join("Salida", f"canny_secuencial_{int(porcentaje_kernel*100)}pct.jpg")
            os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
            cv2.imwrite(archivo_salida, resultado_np)
            print(f"Imagen con bordes detectados guardada: {archivo_salida}")
            
            # Guardar resultados en archivo CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(archivo_resultados), exist_ok=True)
            archivo_existe = os.path.exists(archivo_resultados)
            
            with open(archivo_resultados, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not archivo_existe:
                    writer.writerow(['Timestamp', 'Kernel_Percent', 'Kernel_Size', 'Time_ms', 'Method'])
                    archivo_existe = True
                writer.writerow([timestamp, int(porcentaje_kernel*100), tamanio_kernel, tiempo_ms, 'Secuencial'])
        
        print(f"\n{'='*60}")
        print("=== RESUMEN COMPLETO ===")
        print(f"Procesamiento completado con 3 tamanios de kernel: 1%, 3%, 5%")
        print(f"Resultados guardados en: {archivo_resultados}")
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


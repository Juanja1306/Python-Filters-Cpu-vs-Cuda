import cv2
import time
import csv
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import multiprocessing as mp


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


def sqrt_manual(x):
    """Calcula raíz cuadrada manualmente usando el método de Newton."""
    if x <= 0:
        return 0
    
    aproximacion = x / 2.0
    for _ in range(20):
        if aproximacion == 0:
            break
        nueva = (aproximacion + x / aproximacion) / 2.0
        if abs(nueva - aproximacion) < 1e-10:
            break
        aproximacion = nueva
    
    return aproximacion


def atan2_manual(y, x):
    """Calcula atan2(y, x) manualmente."""
    if x == 0:
        if y > 0:
            return 3.14159265359 / 2.0
        elif y < 0:
            return -3.14159265359 / 2.0
        else:
            return 0.0
    
    if abs(x) > abs(y):
        z = y / x
        atan = z - (z**3)/3 + (z**5)/5 - (z**7)/7 + (z**9)/9
        
        if x < 0:
            if y >= 0:
                return atan + 3.14159265359
            else:
                return atan - 3.14159265359
        return atan
    else:
        z = x / y
        atan = z - (z**3)/3 + (z**5)/5 - (z**7)/7 + (z**9)/9
        
        if y > 0:
            return 3.14159265359 / 2.0 - atan
        else:
            return -3.14159265359 / 2.0 - atan


def generar_kernel_gaussiano(tamaño, sigma):
    """Genera un kernel gaussiano manualmente."""
    kernel = [[0.0 for _ in range(tamaño)] for _ in range(tamaño)]
    centro = tamaño // 2
    
    suma_total = 0.0
    for y in range(tamaño):
        for x in range(tamaño):
            dx = x - centro
            dy = y - centro
            valor = exp_manual(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
            kernel[y][x] = valor
            suma_total += valor
    
    for y in range(tamaño):
        for x in range(tamaño):
            kernel[y][x] /= suma_total
    
    return kernel


def convertir_a_grises(imagen):
    """Convierte imagen a escala de grises."""
    altura, ancho = imagen.shape[:2]
    grises = [[0 for _ in range(ancho)] for _ in range(altura)]
    
    for y in range(altura):
        for x in range(ancho):
            b, g, r = imagen[y, x]
            valor_gris = int(0.299 * r + 0.587 * g + 0.114 * b)
            grises[y][x] = valor_gris
    
    return grises


# ============== FUNCIONES PARALELAS ==============

def worker_suavizado(args):
    """Worker para suavizado gaussiano paralelo."""
    y_inicio, y_fin, imagen, kernel = args
    
    altura = len(imagen)
    ancho = len(imagen[0])
    tam_kernel = len(kernel)
    offset = tam_kernel // 2
    
    resultado_parcial = []
    
    for y in range(y_inicio, y_fin):
        fila = []
        for x in range(ancho):
            suma = 0.0
            
            for ky in range(tam_kernel):
                for kx in range(tam_kernel):
                    py = y + ky - offset
                    px = x + kx - offset
                    
                    if py < 0:
                        py = 0
                    if py >= altura:
                        py = altura - 1
                    if px < 0:
                        px = 0
                    if px >= ancho:
                        px = ancho - 1
                    
                    suma += imagen[py][px] * kernel[ky][kx]
            
            valor = int(suma)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            fila.append(valor)
        resultado_parcial.append(fila)
    
    return resultado_parcial


def aplicar_suavizado_gaussiano_paralelo(imagen, kernel, num_threads):
    """Aplica suavizado gaussiano en paralelo."""
    altura = len(imagen)
    filas_por_thread = altura // num_threads
    
    tareas = []
    for t in range(num_threads):
        y_inicio = t * filas_por_thread
        y_fin = altura if t == num_threads - 1 else (t + 1) * filas_por_thread
        tareas.append((y_inicio, y_fin, imagen, kernel))
    
    with Pool(processes=num_threads) as pool:
        resultados = pool.map(worker_suavizado, tareas)
    
    # Combinar resultados
    resultado_final = []
    for resultado_parcial in resultados:
        resultado_final.extend(resultado_parcial)
    
    return resultado_final


def worker_gradientes(args):
    """Worker para cálculo de gradientes paralelo."""
    y_inicio, y_fin, imagen = args
    
    altura = len(imagen)
    ancho = len(imagen[0])
    
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    magnitud_parcial = []
    direccion_parcial = []
    
    for y in range(y_inicio, y_fin):
        fila_mag = []
        fila_dir = []
        
        for x in range(ancho):
            if y == 0 or y == altura - 1 or x == 0 or x == ancho - 1:
                fila_mag.append(0.0)
                fila_dir.append(0.0)
            else:
                gx = 0.0
                gy = 0.0
                
                for ky in range(3):
                    for kx in range(3):
                        py = y + ky - 1
                        px = x + kx - 1
                        pixel = imagen[py][px]
                        
                        gx += pixel * sobel_x[ky][kx]
                        gy += pixel * sobel_y[ky][kx]
                
                fila_mag.append(sqrt_manual(gx * gx + gy * gy))
                fila_dir.append(atan2_manual(gy, gx))
        
        magnitud_parcial.append(fila_mag)
        direccion_parcial.append(fila_dir)
    
    return magnitud_parcial, direccion_parcial


def calcular_gradientes_paralelo(imagen, num_threads):
    """Calcula gradientes en paralelo."""
    altura = len(imagen)
    filas_por_thread = altura // num_threads
    
    tareas = []
    for t in range(num_threads):
        y_inicio = t * filas_por_thread
        y_fin = altura if t == num_threads - 1 else (t + 1) * filas_por_thread
        tareas.append((y_inicio, y_fin, imagen))
    
    with Pool(processes=num_threads) as pool:
        resultados = pool.map(worker_gradientes, tareas)
    
    # Combinar resultados
    magnitud_final = []
    direccion_final = []
    
    for mag_parcial, dir_parcial in resultados:
        magnitud_final.extend(mag_parcial)
        direccion_final.extend(dir_parcial)
    
    return magnitud_final, direccion_final


def worker_supresion(args):
    """Worker para supresión no-máxima paralelo."""
    y_inicio, y_fin, magnitud, direccion = args
    
    altura = len(magnitud)
    ancho = len(magnitud[0])
    PI = 3.14159265359
    
    resultado_parcial = []
    
    for y in range(y_inicio, y_fin):
        fila = []
        
        for x in range(ancho):
            if y == 0 or y == altura - 1 or x == 0 or x == ancho - 1:
                fila.append(0.0)
            else:
                direccion_rad = direccion[y][x]
                mag = magnitud[y][x]
                
                angulo = direccion_rad * 180.0 / PI
                if angulo < 0:
                    angulo += 180
                
                if (angulo >= 0 and angulo < 22.5) or (angulo >= 157.5 and angulo <= 180):
                    mag1 = magnitud[y][x - 1]
                    mag2 = magnitud[y][x + 1]
                elif angulo >= 22.5 and angulo < 67.5:
                    mag1 = magnitud[y - 1][x + 1]
                    mag2 = magnitud[y + 1][x - 1]
                elif angulo >= 67.5 and angulo < 112.5:
                    mag1 = magnitud[y - 1][x]
                    mag2 = magnitud[y + 1][x]
                else:
                    mag1 = magnitud[y - 1][x - 1]
                    mag2 = magnitud[y + 1][x + 1]
                
                if mag >= mag1 and mag >= mag2:
                    fila.append(mag)
                else:
                    fila.append(0.0)
        
        resultado_parcial.append(fila)
    
    return resultado_parcial


def supresion_no_maxima_paralelo(magnitud, direccion, num_threads):
    """Aplica supresión no-máxima en paralelo."""
    altura = len(magnitud)
    filas_por_thread = altura // num_threads
    
    tareas = []
    for t in range(num_threads):
        y_inicio = t * filas_por_thread
        y_fin = altura if t == num_threads - 1 else (t + 1) * filas_por_thread
        tareas.append((y_inicio, y_fin, magnitud, direccion))
    
    with Pool(processes=num_threads) as pool:
        resultados = pool.map(worker_supresion, tareas)
    
    resultado_final = []
    for resultado_parcial in resultados:
        resultado_final.extend(resultado_parcial)
    
    return resultado_final


def worker_clasificacion(args):
    """Worker para clasificación de píxeles (primera fase histéresis)."""
    y_inicio, y_fin, magnitud, umbral_alto, umbral_bajo = args
    
    ancho = len(magnitud[0])
    resultado_parcial = []
    
    for y in range(y_inicio, y_fin):
        fila = []
        for x in range(ancho):
            if magnitud[y][x] >= umbral_alto:
                fila.append(255)
            elif magnitud[y][x] >= umbral_bajo:
                fila.append(128)
            else:
                fila.append(0)
        resultado_parcial.append(fila)
    
    return resultado_parcial


def umbralizacion_histeresis_paralelo(magnitud, umbral_alto, umbral_bajo, num_threads):
    """Aplica umbralización con histéresis en paralelo."""
    altura = len(magnitud)
    ancho = len(magnitud[0])
    filas_por_thread = altura // num_threads
    
    # Fase 1: Clasificación paralela
    tareas = []
    for t in range(num_threads):
        y_inicio = t * filas_por_thread
        y_fin = altura if t == num_threads - 1 else (t + 1) * filas_por_thread
        tareas.append((y_inicio, y_fin, magnitud, umbral_alto, umbral_bajo))
    
    with Pool(processes=num_threads) as pool:
        resultados = pool.map(worker_clasificacion, tareas)
    
    resultado = []
    for resultado_parcial in resultados:
        resultado.extend(resultado_parcial)
    
    # Fase 2: Rastreo de bordes (secuencial por simplicidad)
    cambio = True
    while cambio:
        cambio = False
        for y in range(1, altura - 1):
            for x in range(1, ancho - 1):
                if resultado[y][x] == 128:
                    conectado = False
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if resultado[y + dy][x + dx] == 255:
                                conectado = True
                                break
                        if conectado:
                            break
                    
                    if conectado:
                        resultado[y][x] = 255
                        cambio = True
    
    # Fase 3: Limpieza
    for y in range(altura):
        for x in range(ancho):
            if resultado[y][x] == 128:
                resultado[y][x] = 0
    
    return resultado


def worker_max_magnitud(args):
    """Worker para calcular máximo de magnitud."""
    y_inicio, y_fin, magnitud = args
    max_local = 0.0
    for y in range(y_inicio, y_fin):
        for x in range(len(magnitud[0])):
            if magnitud[y][x] > max_local:
                max_local = magnitud[y][x]
    return max_local


def calcular_maximo_paralelo(magnitud, num_threads):
    """Calcula el máximo de la magnitud en paralelo."""
    altura = len(magnitud)
    filas_por_thread = altura // num_threads
    
    tareas = []
    for t in range(num_threads):
        y_inicio = t * filas_por_thread
        y_fin = altura if t == num_threads - 1 else (t + 1) * filas_por_thread
        tareas.append((y_inicio, y_fin, magnitud))
    
    with Pool(processes=num_threads) as pool:
        maximos_locales = pool.map(worker_max_magnitud, tareas)
    
    return max(maximos_locales)


def aplicar_canny_paralelo(imagen_grises, num_threads):
    """Filtro Canny completo (PARALELO)."""
    print("  1) Generando kernel gaussiano...")
    kernel_gaussiano = generar_kernel_gaussiano(5, 1.4)
    
    print("  2) Aplicando suavizado gaussiano (paralelo)...")
    suavizada = aplicar_suavizado_gaussiano_paralelo(imagen_grises, kernel_gaussiano, num_threads)
    
    print("  3) Calculando gradientes (Sobel) (paralelo)...")
    magnitud, direccion = calcular_gradientes_paralelo(suavizada, num_threads)
    
    print("  4) Aplicando supresión no-máxima (paralelo)...")
    suprimida = supresion_no_maxima_paralelo(magnitud, direccion, num_threads)
    
    print("  5) Calculando umbrales automáticos (paralelo)...")
    max_magnitud = calcular_maximo_paralelo(suprimida, num_threads)
    
    umbral_alto = max_magnitud * 0.15
    umbral_bajo = umbral_alto * 0.4
    
    print("  6) Aplicando umbralización con histéresis (paralelo)...")
    resultado = umbralizacion_histeresis_paralelo(suprimida, umbral_alto, umbral_bajo, num_threads)
    
    return resultado


def main():
    try:
        # Configuración
        archivo_entrada = os.path.join("Images", "casa.jpg")
        archivo_salida = os.path.join("Salida", "canny_paralelo.jpg")
        archivo_resultados = os.path.join("Resultados", "canny_paralelo.csv")
        
        print("=== FILTRO CANNY DE DETECCIÓN DE BORDES (MULTIPROCESAMIENTO) ===")
        print(f"Leyendo imagen: {archivo_entrada}")
        
        if not os.path.exists(archivo_entrada):
            print(f"Error: No se encontró el archivo {archivo_entrada}")
            return
        
        imagen_original = cv2.imread(archivo_entrada)
        if imagen_original is None:
            print(f"Error: No se pudo leer la imagen {archivo_entrada}")
            return
        
        altura, ancho = imagen_original.shape[:2]
        print(f"Dimensiones: {ancho}x{altura}")
        
        procesadores_disponibles = cpu_count()
        print(f"Procesadores disponibles: {procesadores_disponibles}")
        
        # Convertir a escala de grises
        print("\n--- CONVERSIÓN A ESCALA DE GRISES ---")
        imagen_grises = convertir_a_grises(imagen_original)
        print("Conversión completada.")
        
        # Array de threads a probar
        threads_array = [1, 4, 8]
        tiempos = []
        
        # Ejecutar con diferentes números de threads
        for num_threads in threads_array:
            print(f"\n--- PROCESAMIENTO CON {num_threads} THREAD(S) ---")
            print("Aplicando filtro Canny...")
            print("Pasos: Suavizado → Gradientes → Supresión → Histéresis")
            
            tiempo_inicio = time.time()
            
            if num_threads == 1:
                # Usar versión secuencial para 1 thread
                from secuencial import aplicar_canny_secuencial
                imagen_bordes = aplicar_canny_secuencial(imagen_grises)
            else:
                imagen_bordes = aplicar_canny_paralelo(imagen_grises, num_threads)
            
            tiempo_fin = time.time()
            tiempo_ms = int((tiempo_fin - tiempo_inicio) * 1000)
            tiempos.append(tiempo_ms)
            
            print(f"\nTiempo de ejecución: {tiempo_ms} ms")
            
            # Guardar la última imagen procesada
            if num_threads == threads_array[-1]:
                import numpy as np
                resultado_np = np.array(imagen_bordes, dtype=np.uint8)
                os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
                cv2.imwrite(archivo_salida, resultado_np)
                print(f"Imagen con bordes detectados guardada: {archivo_salida}")
        
        # Mostrar resumen de resultados
        print("\n=== RESUMEN DE RENDIMIENTO ===")
        print("Fórmulas:")
        print("  - Speedup:    S = Ts/Tp")
        print("  - Eficiencia: E = S/p = Ts/(p*Tp)")
        print("\nThreads(p) | Tiempo(Tp) | Speedup(S) | Eficiencia(E) | Clasificación")
        print("-----------|------------|------------|---------------|---------------------------------------")
        
        for i, num_threads in enumerate(threads_array):
            speedup = tiempos[0] / tiempos[i]
            eficiencia_decimal = speedup / num_threads
            eficiencia_porcentaje = eficiencia_decimal * 100
            
            if speedup > num_threads:
                tipo_aceleramiento = "Aceleramiento Superlineal (S > p)"
            elif abs(speedup - num_threads) < 0.1:
                tipo_aceleramiento = "Aceleramiento Lineal (S = p)"
            else:
                tipo_aceleramiento = "Aceleramiento Real (S < p)"
            
            if abs(eficiencia_decimal - 1.0) < 0.05:
                tipo_eficiencia = "E = 1 (caso lineal)"
            elif eficiencia_decimal < 0.5:
                tipo_eficiencia = "E << 1 (poca eficiencia)"
            elif eficiencia_decimal < 1.0:
                tipo_eficiencia = "E < 1 (caso real)"
            else:
                tipo_eficiencia = "E > 1 (superlineal)"
            
            clasificacion = f"{tipo_aceleramiento}, {tipo_eficiencia}"
            
            print(f"{num_threads:10d} | {tiempos[i]:10d} | {speedup:9.2f}x | {eficiencia_porcentaje:10.2f}% | {clasificacion}")
        
        # Guardar resultados en CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(archivo_resultados), exist_ok=True)
        
        # Verificar si el archivo existe para decidir si escribir el header
        archivo_existe = os.path.exists(archivo_resultados)
        
        with open(archivo_resultados, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Solo escribir header si el archivo no existe
            if not archivo_existe:
                writer.writerow(['threads', 'tiempo_ms', 'speedup', 'eficiencia'])
            for i, num_threads in enumerate(threads_array):
                speedup = tiempos[0] / tiempos[i]
                eficiencia = (speedup / num_threads) * 100
                writer.writerow([num_threads, tiempos[i], f"{speedup:.2f}", f"{eficiencia:.2f}"])
        
        print(f"\nResultados guardados en: {archivo_resultados}")
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


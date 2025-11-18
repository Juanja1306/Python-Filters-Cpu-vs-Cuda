# Python Filters: CPU vs CUDA

Comparación de rendimiento entre implementaciones secuenciales (CPU) y CUDA (GPU) de filtros de procesamiento de imágenes en Python.

---

## 📂 Estructura del Proyecto

```
Python-Filters-Cpu-vs-Cuda/
├── Gaussian/              # Filtro Gaussiano (Suavizado)
│   ├── secuencial.py      # Versión secuencial (1 core CPU)
│   ├── Cuda.py            # Versión GPU con CUDA
│   ├── Images/            # Imágenes de entrada
│   └── requirements.txt   # Dependencias Python
│
├── emboss/                             # Filtro Emboss (Relieve)
│   ├── assets/                         # Imagenes
│   └── src/    
│       ├── filters/                    # Implementaciones CPU y CUDA
│       │   ├── domain/                 # Reglas (abstracs)
│       │   │    └── abstracFilter.py   # Define las reglas que deben acatarse los filters
│       │   │
│       │   ├── Cuda.py                 # Implementación del filtro Embos con PyCuda
│       │   └── Sequential.py           # Implementación del filtro Embos de forma secuencial
│       │   
│       ├── utils/                      # Utilidades de imagen y CUDA
│       │   ├── cudaUtils.py            # Funciones de utilidad para la implementación en GPU con PyCUDA
│       │   └── processImage.py         # Clase para procesar la imagen, como lectura, obtencion de parametros, y guardado
│       └── main_emboss.py              # Punto de entrada
│
└── Canny/                 # Filtro Canny (Detección de bordes)
    ├── secuencial.py      # Versión secuencial (1 core CPU)
    ├── Cuda.py            # Versión GPU con CUDA
    ├── Images/            # Imágenes de entrada
    └── requirements.txt   # Dependencias Python
```

---

## 🎯 Filtros Implementados

### 1. **Filtro Gaussiano** 🌫️

Suavizado de imágenes mediante convolución con kernel gaussiano.

**Versiones:**
- `secuencial.py` - Ejecución en CPU (1 core)
- `Cuda.py` - Ejecución en GPU con PyCUDA

**Kernels probados:**
- 5%, 10% y 15% del tamaño mínimo de la imagen

---

### 2. **Filtro Canny** 🔍

Detección de bordes mediante el algoritmo Canny completo.

**Etapas del algoritmo:**
1. 🌫️ Suavizado Gaussiano
2. 📐 Cálculo de Gradientes (Sobel)
3. ✂️ Supresión No-Máxima
4. 🎯 Umbralización Doble
5. 🔗 Histéresis para conectar bordes

**Versiones:**
- `secuencial.py` - Ejecución en CPU (1 core)
- `Cuda.py` - Ejecución en GPU con PyCUDA

**Kernels probados:**
- 1%, 3% y 5% del tamaño mínimo de la imagen

---

### 3. **Filtro Emboss (Relieve)** �

Crea un efecto de relieve o grabado en la imagen, simulando una fuente de luz.

**Versiones:**
- `Sequential.py` - Ejecución en CPU (1 core)
- `Cuda.py` - Ejecución en GPU con PyCUDA

**Kernels probados:**
- Tamaño de kernel dinámico (3x3, 5x5, 7x7, 9x9) basado en las dimensiones de la imagen.

---

## �🛠️ Características Técnicas

### ✅ **Implementación Manual**
- **Sin numpy** para operaciones matemáticas críticas
- Funciones implementadas manualmente:
  - `exp()` - Exponencial (serie de Taylor)
  - `sqrt()` - Raíz cuadrada (método de Newton)
  - `atan2()` - Arcotangente de dos argumentos
  - `max()`, `sum()`, `clip()` - Operaciones básicas

### 🚀 **Optimizaciones CUDA**
- Kernels CUDA optimizados para GPU
- Manejo eficiente de memoria GPU
- Transferencia CPU ↔ GPU minimizada
- Calentamiento de GPU antes de mediciones

### 📊 **Métricas y Análisis**
- ⏱️ Tiempo de ejecución (milisegundos)
- 🔄 Múltiples tamaños de kernel
- 📈 Exportación de resultados a CSV
- 🖼️ Guardado de imágenes procesadas

---

## 📋 Requisitos

### Software necesario:
```
Python 3.9+
CUDA Toolkit (para versiones GPU)
```

### Dependencias Python:
```bash
# Instalar dependencias para Gaussian
cd Gaussian
pip install -r requirements.txt

# Instalar dependencias para Canny
cd Canny
pip install -r requirements.txt
```

**Principales librerías:**
- `numpy>=1.26.0` - Arrays y operaciones numéricas
- `opencv-python-headless>=4.8.0` - Lectura/escritura de imágenes
- `pycuda>=2024.1` - Interfaz Python-CUDA (solo versiones GPU)
- `psutil` - Medición de uso de memoria (para filtro Emboss)

---

## 🚀 Ejecución

### **Filtro Gaussiano**

#### Versión Secuencial (CPU):
```bash
cd Gaussian
python secuencial.py
```

#### Versión CUDA (GPU):
```bash
cd Gaussian
python Cuda.py
```

---

### **Filtro Canny**

#### Versión Secuencial (CPU):
```bash
cd Canny
python secuencial.py
```

#### Versión CUDA (GPU):
```bash
cd Canny
python Cuda.py
```

--- 
### **Filtro Emboss**

Para ejecutar el archivo ``main_emboss``


```bash
python -m emboss.src.main_emboss
```

>  **Nota:**  
> Actualizar la ruta de la imagen a la que se desea aplicar el filtro
> Actualizar el servicio que se desea aplicar, ya sea secuencial o cuda en el ``main_emboss.py``
---

## 🔬 Detalles de Implementación

### **Gaussiano - Proceso**
1. Lectura de imagen desde `Images/casa.jpg`
2. Conversión a escala de grises
3. Generación de kernel gaussiano (manual)
4. Convolución 2D (píxel por píxel)
5. Repetir para 3 tamaños de kernel diferentes

### **Emboss - Proceso**
1. Lectura de la imagen de entrada.
2. Selección dinámica del tamaño del kernel (3x3, 5x5, 7x7 o 9x9) según las dimensiones de la imagen.
3. Generación del kernel de relieve correspondiente.
4. Aplicación de la convolución 2D:
    - **CPU**: Bucle anidado sobre cada píxel y canal de la imagen.
    - **GPU**: Kernel CUDA optimizado para procesar la imagen en paralelo.
5. Adición de un valor de sesgo (bias) de 128 para ajustar el brillo final.
6. Guardado de la imagen con el efecto de relieve.


### **Canny - Proceso**
1. Lectura de imagen desde `Images/casa.jpg`
2. Conversión a escala de grises
3. Suavizado Gaussiano (con kernel variable)
4. Cálculo de gradientes (operador Sobel)
5. Supresión no-máxima (adelgazamiento de bordes)
6. Umbralización doble (clasificación de píxeles)
7. Histéresis (conexión de bordes fuertes y débiles)
8. Repetir para 3 tamaños de kernel diferentes

---

## 🧮 Implementaciones Matemáticas Manuales

### Función Exponencial (Taylor Series)
```python
def exp_manual(x, terminos=20):
    resultado = 1.0
    termino = 1.0
    for n in range(1, terminos):
        termino *= x / n
        resultado += termino
    return resultado
```

### Raíz Cuadrada (Newton's Method)
```python
def sqrt_manual(x, iteraciones=10):
    if x == 0:
        return 0
    estimacion = x / 2.0
    for _ in range(iteraciones):
        estimacion = (estimacion + x / estimacion) / 2.0
    return estimacion
```

### Arcotangente (CORDIC Algorithm)
```python
def atan2_manual(y, x):
    # Implementación basada en aproximación de Taylor
    # y manejo de cuadrantes
    ...
```

---

## 💡 Ventajas de esta Implementación

| Aspecto | Ventaja |
|---------|---------|
| 🎓 **Educativo** | Comprensión profunda de algoritmos |
| 🔧 **Control Total** | Sin dependencias de librerías externas |
| ⚡ **Optimizable** | Código adaptable a diferentes plataformas |
| 🧪 **Experimental** | Facilita comparaciones CPU vs GPU |
| 📊 **Medible** | Métricas precisas de rendimiento |

---

## 📈 Comparación CPU vs GPU

### **CPU (Secuencial)**
- ✅ Sin requisitos especiales de hardware
- ✅ Código más simple y portable
- ❌ Más lento para imágenes grandes
- ❌ No aprovecha paralelismo masivo

### **GPU (CUDA)**
- ✅ Procesamiento masivamente paralelo
- ✅ Hasta 100x más rápido en imágenes grandes
- ❌ Requiere NVIDIA GPU con CUDA
- ❌ Mayor complejidad de código

---

## 🎓 Conceptos Aplicados

- **Convolución 2D**: Operación fundamental en procesamiento de imágenes
- **Filtro Gaussiano**: Suavizado basado en distribución normal
- **Operador Sobel**: Detección de bordes por gradientes
- **Supresión No-Máxima**: Adelgazamiento de bordes
- **Histéresis**: Conexión inteligente de píxeles de borde
- **CUDA Programming**: Programación en GPU
- **Optimización de Performance**: Medición y comparación

---

## 📚 Referencias

- [OpenCV Documentation](https://docs.opencv.org/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Canny Edge Detection (Wikipedia)](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Gaussian Blur (Wikipedia)](https://en.wikipedia.org/wiki/Gaussian_blur)
- [Emboss (Wikipedia)](https://en.wikipedia.org/wiki/Image_embossing)

---

## 👨‍💻 Uso Académico

Este proyecto es ideal para:
- 📖 Aprender procesamiento de imágenes desde cero
- 🔬 Experimentar con algoritmos de visión por computador
- ⚡ Comparar rendimiento CPU vs GPU
- 🎯 Entender implementaciones sin librerías de alto nivel
- 📊 Analizar impacto de parámetros (tamaño de kernel)

---

## 📝 Notas

- Las imágenes de entrada deben estar en formato `.jpg` o `.png`
- Los kernels se calculan como porcentaje del tamaño mínimo de la imagen
- Todos los cálculos matemáticos críticos están implementados manualmente
- Los resultados se guardan automáticamente al ejecutar los scripts
- La versión CUDA requiere una GPU NVIDIA compatible

---

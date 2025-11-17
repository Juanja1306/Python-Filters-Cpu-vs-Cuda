# Filtro Canny - Detección de Bordes

## 🚀 Ejecución con Docker

### Paso 1: Construir la imagen

```powershell
docker build -t canny-filter:py3.14 .
```

### Paso 2: Ejecutar el contenedor

```powershell
docker run -v ${PWD}/Salida:/app/Salida -v ${PWD}/Resultados:/app/Resultados canny-filter:py3.14
```

---

## 📁 Resultados

Los archivos generados estarán en:
- `Salida/canny_paralelo.jpg` - Imagen con bordes detectados
- `Resultados/canny_paralelo.csv` - Métricas de rendimiento

---

## ℹ️ Información

El script `paralelo314.py` implementa el algoritmo Canny completo:
1. 🌫️ Suavizado Gaussiano
2. 📐 Gradientes con Sobel
3. ✂️ Supresión no-máxima
4. 🎯 Umbralización
5. 🔗 Histéresis

Prueba automáticamente con **1, 4 y 8 procesos** y calcula speedup y eficiencia.


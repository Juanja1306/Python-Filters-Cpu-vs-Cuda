# Filtro Gaussiano - Multiprocesamiento

## 🚀 Ejecución con Docker

### Paso 1: Construir la imagen

```powershell
docker build -t gaussian-filter:py3.14 .
```

### Paso 2: Ejecutar el contenedor

```powershell
docker run -v ${PWD}/Salida:/app/Salida -v ${PWD}/Resultados:/app/Resultados gaussian-filter:py3.14
```

---

## 📁 Resultados

Los archivos generados estarán en:
- `Salida/gaussiana_paralelo.jpg` - Imagen procesada
- `Resultados/gaussiana_paralelo.csv` - Métricas de rendimiento

---

## ℹ️ Información

El script `paralelo314.py` prueba automáticamente con **1, 4 y 8 procesos** y calcula:
- ⏱️ Tiempo de ejecución
- 🚀 Speedup
- 📊 Eficiencia


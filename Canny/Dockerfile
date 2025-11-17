# Dockerfile para ejecutar el filtro Canny paralelo314 en Python 3.14
FROM python:3.14-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy>=1.26.0 opencv-python-headless>=4.8.0

# Crear directorios necesarios para la salida
RUN mkdir -p Salida Resultados Images

# Copiar los scripts de Python
COPY paralelo314.py .
COPY secuencial.py .

# Copiar las imágenes
COPY Images/ Images/

# Configurar variables de entorno para multiprocessing
ENV PYTHONUNBUFFERED=1

# Comando por defecto para ejecutar el script paralelo314
CMD ["python", "paralelo314.py"]


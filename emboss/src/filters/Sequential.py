# Dependencies
from dataclasses import dataclass
import numpy as np
import time
import psutil
import os


# Modules
from emboss.src.filters.domain.abstracFilter import AplicationFilter
from emboss.src.utils.processImage import ProcessImage, SaveImageProps
from emboss.src.types.index import RGBImage, IntKernel




@dataclass (frozen=True)
class SequentialProps:
    processImage :ProcessImage
    pathImage    :str


class Sequential(AplicationFilter):
    def __init__(self, options: SequentialProps):
        self.__pathImage    = options.pathImage
        self.__pathNewImage = f'{options.pathImage}_boss.png'
        self.__processImage = options.processImage
    

    def aplyFilter(self):
        image     = self.__processImage.readImage(self.__pathImage)
        bossImage = self.__embossFilter(image)        
        
        self.__processImage.saveImage(SaveImageProps(
            formatImage = "PNG",
            pathImage   = self.__pathNewImage,
            image       = bossImage
        ))        
        
        
    def __embossFilter(self, image: RGBImage) -> RGBImage:
        print("Applying Emboss filter sequentially")

        # Validación mínima del shape esperado (H, W, 3)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3) in RGB")

        # Kernel de emboss direccional y bias para recentrar intensidades
        kernel: IntKernel = np.array(
            [[-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]],
            dtype=np.int32,
        )
        biasValue: int = 128

        imageHeight, imageWidth, imageChannels = image.shape

        # Padding por borde para mantener tamaño
        paddedImage: RGBImage = np.pad(
            image, pad_width=((1, 1), (1, 1), (0, 0)), mode="edge"
        )

        embossedImage: RGBImage = np.zeros_like(image, dtype=np.uint8)

        # Medir el tiempo y memoria antes de comenzar
        start_time = time.time()  # Tiempo antes de empezar el filtro
        process = psutil.Process(os.getpid())  # Obtener el proceso actual
        start_memory = process.memory_info().rss / 1024 / 1024  # Memoria en MB antes de empezar

        for channelIndex in range(imageChannels):
            for rowIndex in range(imageHeight):
                for colIndex in range(imageWidth):
                    # Obtener el vecindario de la imagen alrededor del píxel actual
                    neighborhood = [
                        [paddedImage[rowIndex + i, colIndex + j, channelIndex] 
                        for j in range(3)] 
                        for i in range(3)
                    ]

                    # Realizar la convolución (producto punto) sin usar numpy
                    convolutionSum = 0
                    for i in range(3):
                        for j in range(3):
                            convolutionSum += neighborhood[i][j] * kernel[i, j]

                    # Aplicar el sesgo (bias) y ajustar el valor
                    valueWithBias = convolutionSum + biasValue

                    # Clamp manual para asegurar que el valor esté entre 0 y 255
                    clampedValue = max(0, min(255, valueWithBias))

                    embossedImage[rowIndex, colIndex, channelIndex] = np.uint8(clampedValue)

        # Medir el tiempo y la memoria después de terminar
        end_time = time.time()  # Tiempo después de aplicar el filtro
        end_memory = process.memory_info().rss / 1024 / 1024  # Memoria en MB después de aplicar el filtro

        # Calcular el tiempo de ejecución y la memoria utilizada
        elapsed_time = end_time - start_time  # Tiempo en segundos
        memory_used = end_memory - start_memory  # Memoria en MB utilizada

        # Imprimir los resultados
        print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")
        print(f"Memoria utilizada: {memory_used:.4f} MB")

        return embossedImage
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



@dataclass(frozen=True)
class EmbossParams:
    kernel    : IntKernel
    biasValue : int
    height    : int
    width     : int
    channels  : int



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
        image        = self.__processImage.readImage(self.__pathImage)
        imageParams  = self.__processImage.getImageParamters(image)
        embossParams = self.__createEmbossKernel(imageParams)
        bossImage    = self.__embossFilter(image, embossParams)        
        
        self.__processImage.saveImage(SaveImageProps(
            formatImage = "PNG",
            pathImage   = self.__pathNewImage,
            image       = bossImage
        ))        
        
        
    def __embossFilter(self, image: RGBImage, embossParams:EmbossParams) -> RGBImage:
        print("Applying Emboss filter sequentially")

        # Validación mínima del shape esperado (H, W, 3)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3) in RGB")

        # Usar el kernel y sesgo proporcionados por los parámetros de emboss
        biasValue = embossParams.biasValue
        kernel    = embossParams.kernel

        imageChannels = embossParams.channels
        imageHeight   = embossParams.height
        imageWidth    = embossParams.width


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
                        for j in range(kernel.shape[1])]  # Cambiar de 3 a kernel.shape[1]
                        for i in range(kernel.shape[0])  # Cambiar de 3 a kernel.shape[0]
                    ]

                    # Realizar la convolución (producto punto) sin usar numpy
                    convolutionSum = 0
                    for i in range(kernel.shape[0]):  # Iterar sobre las filas del kernel
                        for j in range(kernel.shape[1]):  # Iterar sobre las columnas del kernel
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
    
    
    def __createEmbossKernel(self, imageParams: dict[str, int]) -> EmbossParams:
        # Obtener parámetros de la imagen
        height  = imageParams['height']
        width   = imageParams['width']
        chanels = imageParams['channels']
       

        # Ajustar el tamaño del kernel según las dimensiones de la imagen
       # Ajustar el tamaño del kernel según las dimensiones de la imagen
        if height > 4000 and width > 4000:
            kernel_size = 9  # Para imágenes muy grandes (más de 4000x4000)
        elif height > 1000 and width > 1000:
            kernel_size = 7  # Para imágenes grandes (más de 1000x1000)
        else:
            kernel_size = 5  # Para imágenes medianas (menos de 1000x1000)
        
        if kernel_size == 9:
            kernel = np.array(
                [
                    [ -3, -2, -1, 0, 1, 2,  3,  4,  5  ],
                    [ -2, -1,  0, 1, 2, 3,  4,  5,  6  ],
                    [ -1,  0,  1, 2, 3, 4,  5,  6,  7  ],
                    [  0,  1,  2, 3, 4, 5,  6,  7,  8  ],
                    [  1,  2,  3, 4, 5, 6,  7,  8,  9  ],
                    [  2,  3,  4, 5, 6, 7,  8,  9,  10 ],
                    [  3,  4,  5, 6, 7, 8,  9,  10, 11 ],
                    [  4,  5,  6, 7, 8, 9,  10, 11, 12 ],
                    [  5,  6,  7, 8, 9, 10, 11, 12, 13 ]
                ],
                dtype=np.int32
            )
        elif kernel_size == 7:
            kernel = np.array(
                [
                    [ -2, -1,  0, 1, 2, 3, 4  ],
                    [ -1,  0,  1, 2, 3, 4, 5  ],
                    [  0,  1,  2, 3, 4, 5, 6  ],
                    [  1,  2,  3, 4, 5, 6, 7  ],
                    [  2,  3,  4, 5, 6, 7, 8  ],
                    [  3,  4,  5, 6, 7, 8, 9  ],
                    [  4,  5,  6, 7, 8, 9, 10 ]
                ],
                dtype=np.int32
            )
        else:  # Kernel de tamaño 5x5 para imágenes más pequeñas
            kernel = np.array(
                [
                    [ -2, -1,  0,  1,   2 ],
                    [ -1,  1,  1,  1,   1 ],
                    [  0,  1,  2,  1,   0 ],
                    [  1,  1,  1,  1,  -1 ],
                    [  2,  1,  0, -1,  -2 ]
                ],
                dtype=np.int32
            )

        # El sesgo (bias) es un valor constante que puedes ajustar según sea necesario
        biasValue = 128

        # Retornar el kernel y el sesgo en un diccionario o cualquier otra estructura que desees        

        embossParams = EmbossParams(
            biasValue = biasValue,
            channels  = chanels,
            kernel    = kernel,
            height    = height,
            width     = width
        )
        
        return embossParams
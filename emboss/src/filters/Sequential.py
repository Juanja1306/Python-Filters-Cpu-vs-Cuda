# Dependencies
from dataclasses import dataclass
import numpy as np

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

        
        imageHeight  : int
        imageWidth   : int
        imageChannels: int
        imageHeight, imageWidth, imageChannels = image.shape

        # Padding por borde para mantener tamaño
        paddedImage: RGBImage = np.pad(
            image, pad_width = ((1, 1), (1, 1), (0, 0)), mode="edge"
        )

        embossedImage: RGBImage = np.zeros_like(image, dtype=np.uint8)

        
        channelIndex : int
        rowIndex     : int
        colIndex     : int

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


        return embossedImage
# Dependencies
from dataclasses import dataclass
import numpy as np

# Modules
from emboss.src.filters.domain.abstracFilter import AplicationFilter
from emboss.src.utils.processImage import ProcessImage #, SaveImageProps
from emboss.src.utils.cudaUtils import apply_cuda_emboss_filter #, SaveImageProps
from emboss.src.types.index import RGBImage, IntKernel

@dataclass(frozen=True)
class EmbossParams:
    kernel    : IntKernel
    biasValue : int
    height    : int
    width     : int
    channels  : int


@dataclass (frozen=True)
class CudaProps:
    processImage:ProcessImage
    pathImage:str
    
class CudaEmboss(AplicationFilter):
    def __init__(self, options: CudaProps):
        self.__pathImage    = options.pathImage
        self.__pathNewImage = f'{options.pathImage}_boss_cuda.png'
        self.__processImage = options.processImage
        
    def aplyFilter(self):
        image     = self.__processImage.readImage(self.__pathImage)
        imageParams  = self.__processImage.getImageParamters(image)
        embossParams = self.__createEmbossKernel(imageParams)
        bossImage = self.__embossFilter(image, embossParams)
        print(bossImage)
    
    def __embossFilter(self, image: RGBImage, embossParams: EmbossParams) -> RGBImage:
        print("Applying Emboss filter with CUDA")

        # Llamar a la función externa para realizar la operación CUDA
        embossed_image = apply_cuda_emboss_filter(
            image,
            embossParams.kernel,
            embossParams.width,
            embossParams.height,
            len(embossParams.kernel[0]),
            embossParams.biasValue
        )

        return embossed_image

    def __createEmbossKernel(self, imageParams: dict[str, int]) -> EmbossParams:
        # Obtener parámetros de la imagen
        height  = imageParams['height']
        width   = imageParams['width']
        chanels = imageParams['channels']
       

        # Ajustar el tamaño del kernel según las dimensiones de la imagen
       # Ajustar el tamaño del kernel según las dimensiones de la imagen
        if height > 5000 and width > 7000:
            kernel_size = 9  # Para imágenes muy grandes (más de 4000x4000)
        elif height > 3000 and width > 3000:
            kernel_size = 7  # Para imágenes grandes (más de 1000x1000)
        elif height > 1000 and width > 1000:
            kernel_size = 5  # Para imágenes medianas (menos de 1000x1000)
        else:
            kernel_size = 3  # Para imágenes pequeñas (menos de 1000x1000)
        
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
        elif kernel_size == 5:  # Kernel de tamaño 5x5 para imágenes más pequeñas
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
        else:
            kernel = np.array(
                [[-2, -1, 0],
                 [-1,  1, 1],
                 [ 0,  1, 2]],
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
        
        print(f'Kernel size: {kernel_size}, bias: {biasValue}')
        print(f'Kernel: \n{kernel}')
        
        return embossParams
        
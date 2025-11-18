# Dependencies
from dataclasses import dataclass

# Modules
from emboss.src.filters.domain.abstracFilter import AplicationFilter
from emboss.src.utils.processImage import ProcessImage #, SaveImageProps
from emboss.src.types.index import RGBImage


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
        bossImage = self.__embossFilter(image)
        print(bossImage)
    
    def __embossFilter(self, image: RGBImage) -> RGBImage:
        print("Applying Emboss filter with CUDA")
        return image

        
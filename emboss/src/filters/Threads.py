# Dependencies
from dataclasses import dataclass
#import numpy as np

# Modules
from emboss.src.filters.domain.abstracFilter import AplicationFilter
from emboss.src.utils.processImage import ProcessImage#, SaveImageProps
from emboss.src.types.index import RGBImage




@dataclass (frozen=True)
class ThreardsProps:
    processImage:ProcessImage
    pathImage:str


class ThreadsEmboss(AplicationFilter):
    def __init__(self, options: ThreardsProps):
        self.__pathImage    = options.pathImage
        self.__pathNewImage = f'{options.pathImage}_boss_threads.png'
        self.__processImage = options.processImage
    

    def aplyFilter(self):
        image     = self.__processImage.readImage(self.__pathImage)
        bossImage = self.__embossFilter(image)
        print(bossImage)
    
    def __embossFilter(self, image: RGBImage) -> RGBImage:
        print("Applying Emboss filter with threads")
        return image


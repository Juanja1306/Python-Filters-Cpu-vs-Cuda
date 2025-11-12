from dataclasses import dataclass
from emboss.src.filters.abstracFilter import AplicationFilter
from emboss.src.utils.processImage import ProcessImage


@dataclass (frozen=True)
class SequentialProps:
    pathImage:str
    processImage:ProcessImage


class Sequential(AplicationFilter):
    def __init__(self, options: SequentialProps):
        self.__pathImage    = options.pathImage
        self.__processImage = options.processImage
    

    def aplyFilter(self):
        print("Aplicando filtro secuencial...")
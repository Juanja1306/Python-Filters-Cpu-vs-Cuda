from dataclasses import dataclass

from emboss.src.filters.domain.abstracFilter import AplicationFilter

@dataclass(frozen=True)
class EmbossProps:
    serviceEmboss: AplicationFilter

class Emboss:
    def __init__(self, options: EmbossProps):
        self.__serviceEmboss = options.serviceEmboss
    
    def aplyFilter(self):
        self.__serviceEmboss.aplyFilter()
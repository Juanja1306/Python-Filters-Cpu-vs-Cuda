from emboss.src.emboss import Emboss, EmbossProps
from emboss.src.filters.Cuda import CudaEmboss, CudaProps
from emboss.src.filters.Sequential import Sequential, SequentialProps
from emboss.src.utils.processImage import ProcessImage


processImage = ProcessImage()
pathImage    = "C:/Users/Jonna/Desktop/ups/8ctavo ciclo/paralelo/practica_2/Python-Filters-Cpu-vs-Cuda/emboss/assets/casa.jpg"


# Sequential Service
sequential = Sequential(SequentialProps (
    pathImage    = pathImage,
    processImage = processImage
))

# Cuda Service
cuda = CudaEmboss(CudaProps(
    pathImage    = pathImage,
    processImage = processImage
))


# Embos
embossFilter = Emboss(EmbossProps(
    serviceEmboss = sequential
    # serviceEmboss = threads
    #serviceEmboss = cuda
))

embossFilter.aplyFilter()
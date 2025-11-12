from emboss.src.filters.pySequential import Sequential, SequentialProps
from emboss.src.utils.processImage import ProcessImage


processImage = ProcessImage()
options = SequentialProps(
    pathImage="C:/Users/Jonna/Desktop/ups/8ctavo ciclo/paralelo/practica_2/Python-Filters-Cpu-vs-Cuda/emboss/assets/hyena-img.jpg",
    processImage=processImage
)

sequential = Sequential(options)
sequential.aplyFilter()
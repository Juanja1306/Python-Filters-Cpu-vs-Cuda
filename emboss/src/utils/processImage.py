from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from PIL import Image
import numpy as np
from numpy.typing import NDArray




# Alias de tipos
RGBImage = NDArray[np.uint8]
IntKernel = NDArray[np.int32]
FormatImage = Literal["PNG", "JPG"]

@dataclass(frozen=True)
class SaveImageProps:
    pathImage: str
    image: RGBImage
    formatImage: FormatImage


class ProcessImage():
    
    def readImage(self, pathImage: str) -> RGBImage:
        print("Reading image...")
        with Image.open(pathImage) as pilImage:
            pilImage = pilImage.convert("RGB")
            imageArray: RGBImage = np.array(pilImage, dtype=np.uint8)
        return imageArray
    
    def saveImage(self, options: SaveImageProps) -> None:
        pathImage = options.pathImage
        image = options.image
        formatImage = options.formatImage
        
        print(f"Saving image to: {pathImage}")        
        try:            
            pilImage: Image.Image = Image.fromarray(image)
            
            if pilImage.mode != "RGB":
                pilImage = pilImage.convert("RGB")
            
            pilImage.save(pathImage, format=formatImage)
            print("Image saved successfully ✅")
        except Exception as error:
            print(f"Error saving image: {error}")

    
    def embossSequential(self, image: RGBImage) -> RGBImage:
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

        
        imageHeight: int
        imageWidth: int
        imageChannels: int
        imageHeight, imageWidth, imageChannels = image.shape

        # Padding por borde para mantener tamaño
        paddedImage: RGBImage = np.pad(
            image, pad_width=((1, 1), (1, 1), (0, 0)), mode="edge"
        )

        embossedImage: RGBImage = np.zeros_like(image, dtype=np.uint8)

        
        channelIndex: int
        rowIndex: int
        colIndex: int

        for channelIndex in range(imageChannels):
            for rowIndex in range(imageHeight):
                for colIndex in range(imageWidth):
                    neighborhood: RGBImage = paddedImage[
                        rowIndex:rowIndex + 3, colIndex:colIndex + 3, channelIndex
                    ]

                    # Convertimos a int32 para evitar overflow al multiplicar por el kernel
                    convolutionSum: int = int(
                        np.sum(neighborhood.astype(np.int32) * kernel)
                    )
                    valueWithBias: int = convolutionSum + biasValue

                    # Clamp manual para Pylance (evita quejas por tipos)
                    if valueWithBias < 0:
                        clampedValue: int = 0
                    elif valueWithBias > 255:
                        clampedValue = 255
                    else:
                        clampedValue = valueWithBias

                    embossedImage[rowIndex, colIndex, channelIndex] = np.uint8(clampedValue)

        return embossedImage
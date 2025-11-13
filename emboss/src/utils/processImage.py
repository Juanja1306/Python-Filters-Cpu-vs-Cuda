# Dependencies
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from PIL import Image
import numpy as np

# Modules
from emboss.src.types.index import RGBImage


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
            print("Image saved successfully")
        except Exception as error:
            print(f"Error saving image: {error}")
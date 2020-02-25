import numpy as np
import pandas as pd
import skimage

class ImageTransform():
    def __init__(self, config):
        self.config = config

    def transform(self, 
        image,
        pad_width = 400,
        pad_value = 0,
        anti_aliasing = True):

        image = np.pad(image,
            mode = 'constant',
            constant_values = pad_value,
            pad_width = ((pad_width,pad_width),(pad_width,pad_width), (0,0))
            )
        image = skimage.img_as_ubyte(skimage.transform.resize(image, (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM),
            anti_aliasing=True) )
        return image

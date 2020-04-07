import numpy as np
import skimage

class ImageTransform():
    def __init__(self, config):
        """config: COCO style configuration object for the Mask RCNN.
        """
        self.config = config

    def transform(self, image, pad_width = 400, pad_value = 0):
        """Resizes and/or pads the input image.
        image: Input image
        pad_width: number of pixels to pad each image vertice with
        pad_value: The value with which to pad the image (0 to 255)
        """
        image = np.pad(
            image,
            mode = 'constant',
            constant_values = pad_value,
            pad_width = ((pad_width,pad_width),(pad_width,pad_width), (0,0)))

        image = skimage.transform.resize(
            image,
            (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM),
            anti_aliasing=True,
            preserve_range=True).astype(np.uint8)

        return image

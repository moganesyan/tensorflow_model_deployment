# User Define parameters
import os
# Make it True if you want to use the provided coco weights

# keras model directory path
MODEL_DIR = os.path.abspath('./keras_model/')

# keras model file path
H5_WEIGHT_PATH = os.path.abspath('./keras_model/mask_rcnn_tags_0001.h5')

# Path where the Frozen PB will be save
PATH_TO_SAVE_FROZEN_PB = os.path.abspath('./frozen_model/')

# Name for the Frozen PB name
FROZEN_NAME = 'mask_frozen_graph.pb'

# PATH where to save serving model
PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL = os.path.abspath('./serving_model')

# Version of the serving model
VERSION_NUMBER = 1

# Number of classes that you have trained your model
NUMBER_OF_CLASSES = 5

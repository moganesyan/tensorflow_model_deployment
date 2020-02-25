# User Define parameters
import os

# path to the mask r-cnn source code
MRCNN_DIR = os.path.abspath('../external/mask_rcnn/')
COCO_DIR = os.path.abspath('../external/mask_rcnn/samples')

# keras model directory path
KERAS_MODEL_DIR = os.path.abspath('./keras_model/')

# keras model file path
KERAS_WEIGHTS_PATH = os.path.abspath('./keras_model/mask_rcnn_tags_0001.h5')

# Path where the Frozen PB will be save
FROZEN_MODEL_PATH = os.path.abspath('./frozen_model/')

# Name for the Frozen PB name
FROZEN_NAME = 'mrcnn_frozen_graph.pb'

# PATH where to save serving model
TF_SERVING_MODEL_PATH = os.path.abspath('./serving_model')

# Version of the serving model
VERSION_NUMBER = 1

# Graph optimisation transforms
TRANSFORMS = ["remove_nodes(op=Identity)", 
             "merge_duplicate_nodes",
             "strip_unused_nodes",
             "fold_constants(ignore_errors=true)",
             "fold_batch_norms",
             # "quantize_nodes", 
             # "quantize_weights"
             ]

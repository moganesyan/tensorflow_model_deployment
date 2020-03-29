# User Define parameters
import os

# path to the mask r-cnn source code
MRCNN_DIR = os.path.abspath('../external/mask_rcnn/')
COCO_DIR = os.path.abspath('../external/mask_rcnn/samples')

# keras model directory path
KERAS_MODEL_DIR = os.path.abspath('./keras_model/')

# keras model file path
KERAS_WEIGHTS_PATH = os.path.abspath('./keras_model/mask_rcnn_tags_0001.h5')

# tf serving export dir
EXPORT_DIR = os.path.abspath('./serving_model')

# Version of the tf serving model
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

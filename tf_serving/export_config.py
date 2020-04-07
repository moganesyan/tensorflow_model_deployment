import os

class ExportConfig(object):
    # model name
    MODEL_NAME = "coco_mrcnn"

    # path to the mask r-cnn source code
    MRCNN_DIR = os.path.abspath('../external/mask_rcnn/')
    COCO_DIR = os.path.abspath('../external/mask_rcnn/samples')

    # keras model path
    KERAS_MODEL_DIR = os.path.abspath('./keras_model/')

    # keras model weights path
    KERAS_WEIGHTS_PATH = os.path.abspath('./keras_model/mask_rcnn_tags_0001.h5')

    # tf serving export dir
    EXPORT_DIR = os.path.abspath('./exported_models')

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

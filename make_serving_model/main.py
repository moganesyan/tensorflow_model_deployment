from user_config import *
import tensorflow as tf
import keras.backend as K
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.tools.graph_transforms import TransformGraph

import os
import sys
MRCNN_DIR = os.path.abspath('../external/mask_rcnn/')
COCO_DIR = os.path.abspath('../external/mask_rcnn/samples')
sys.path.append(MRCNN_DIR)
sys.path.append(COCO_DIR)

from coco.coco import CocoConfig
from mrcnn.model import MaskRCNN

sess = tf.Session()
K.set_session(sess)


def get_config():
    class InferenceConfig(CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    return config


def describe_graph(graph_def, show_nodes=False):
  print('Input Feature Nodes: {}'.format(
      [node.name for node in graph_def.node if node.op=='Placeholder']))
  print('')
  print('Unused Nodes: {}'.format(
      [node.name for node in graph_def.node if 'unused'  in node.name]))
  print('')
  print('Output Nodes: {}'.format( 
      [node.name for node in graph_def.node if (
          'predictions' in node.name or 'softmax' in node.name)]))
  print('')
  print('Quantization Nodes: {}'.format(
      [node.name for node in graph_def.node if 'quant' in node.name]))
  print('')
  print('Constant Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Const'])))
  print('')
  print('Variable Count: {}'.format(
      len([node for node in graph_def.node if 'Variable' in node.op])))
  print('')
  print('Identity Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Identity'])))
  print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

  if show_nodes==True:
    for node in graph_def.node:
      print('Op:{} - Name: {}'.format(node.op, node.name))


def get_size(model_dir, model_file='saved_model.pb'):
  model_file_path = os.path.join(model_dir, model_file)
  print(model_file_path, '')
  pb_size = os.path.getsize(model_file_path)
  variables_size = 0
  if os.path.exists(
      os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
    variables_size = os.path.getsize(os.path.join(
        model_dir,'variables/variables.data-00000-of-00001'))
    variables_size += os.path.getsize(os.path.join(
        model_dir,'variables/variables.index'))
  print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
  print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
  print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))


def freeze_session(session, keep_var_names=None, input_names = None, output_names=None, clear_devices=True):
    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_names = input_names or []
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)

        transforms = [
                         "remove_nodes(op=Identity)", 
                         "merge_duplicate_nodes",
                         "strip_unused_nodes",
                         "fold_constants(ignore_errors=true)",
                         "fold_batch_norms",
                         "quantize_nodes", 
                         "quantize_weights"
                        ]
        optimized_graph_def = TransformGraph(
                                              frozen_graph,
                                              input_names,
                                              output_names,
                                              transforms)
        return optimized_graph_def


def freeze_model(model, name):
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.outputs][:4],
        input_names=[input_tensor.op.name for input_tensor in model.inputs][:4])
    directory = PATH_TO_SAVE_FROZEN_PB
    tf.train.write_graph(frozen_graph, directory, name , as_text=False)
    print("*"*80)
    print("Finish converting keras model to Frozen PB")
    print('PATH: ', PATH_TO_SAVE_FROZEN_PB)
    print("*" * 80)


def make_serving_ready(model_path, save_serve_path, version_number):
    import tensorflow as tf

    export_dir = os.path.join(save_serve_path, str(version_number))
    graph_pb = model_path

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    print("*" * 80)
    describe_graph(graph_def)
    print("*" * 80)

    with tf.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        input_image = g.get_tensor_by_name("input_image:0")
        input_image_meta = g.get_tensor_by_name("input_image_meta:0")
        input_anchors = g.get_tensor_by_name("input_anchors:0")

        output_detection = g.get_tensor_by_name("mrcnn_detection/Reshape_1:0")
        output_mask = g.get_tensor_by_name("mrcnn_mask/Reshape_1:0")

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"input_image": input_image, 'input_image_meta': input_image_meta, 'input_anchors': input_anchors},
                {"mrcnn_detection/Reshape_1": output_detection, 'mrcnn_mask/Reshape_1': output_mask})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

    builder.save()
    print("*" * 80)
    print("FINISH CONVERTING FROZEN PB TO SERVING READY")
    print("PATH:", PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL)
    print("*" * 80)


# Load Mask RCNN config
# you can also load your own config in here.
# config = your_custom_config_class
config = get_config()


# LOAD MODEL
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(H5_WEIGHT_PATH, by_name=True)

# Converting keras model to PB frozen graph
freeze_model(model.keras_model, FROZEN_NAME)

# Now convert frozen graph to Tensorflow Serving Ready
make_serving_ready(os.path.join(PATH_TO_SAVE_FROZEN_PB, FROZEN_NAME),
                     PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL,
                     VERSION_NUMBER)

print("*" * 80)
get_size(os.path.join(PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL, str(VERSION_NUMBER)))
print("*" * 80)

print("COMPLETED")
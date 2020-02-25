from user_config import *
import tensorflow as tf
import keras.backend as K
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.tools.graph_transforms import TransformGraph

import os
import sys

sys.path.append(MRCNN_DIR)
sys.path.append(COCO_DIR)

from coco.coco import CocoConfig
from mrcnn.model import MaskRCNN

sess = tf.Session()
K.set_session(sess)


def get_coco_config():
  '''
    Get inference config for MS-COCO
    Ammend hyperparameters as necessary
  '''
  class InferenceConfig(CocoConfig):
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1

  return InferenceConfig()


def describe_graph(graph_def, show_nodes = False):
  '''
    Describe the graph of the tensorflow model. This is a diagnostic function.
    Graph is broken down by node types: Input, output, quantization, etc.
  '''
  print(f"Input Feature Nodes: {[node.name for node in graph_def.node if node.op == 'Placeholder']}")
  print(f"Unused Nodes: {[node.name for node in graph_def.node if 'unused' in node.name]}")
  print(f"Output Nodes: {[node.name for node in graph_def.node if ('predictions' in node.name or 'softmax' in node.name)]}")
  print(f"Quantization Node Count: {len([node.name for node in graph_def.node if 'quant' in node.name])}")
  print(f"Constant Count: {len([node for node in graph_def.node if node.op =='Const'])}")
  print(f"Variable Count: {len([node for node in graph_def.node if 'Variable' in node.op])}")
  print(f"Identity Count: {len([node for node in graph_def.node if node.op =='Identity'])}")
  print("", f"Total nodes: {len(graph_def.node)}", "")

  if show_nodes == True:
    for node in graph_def.node:
      print(f"Op:{node.op} - Name: {node.name}")


def get_model_size(model_dir, model_file = "saved_model.pb"):
  '''
    Get size of the produced tf-serving model and count number of variables. This is a diagnostic function.
  '''

  model_file_path = os.path.join(model_dir, model_file)
  print(model_file_path, '')
  pb_size = os.path.getsize(model_file_path)
  variables_size = 0

  if os.path.exists(
      os.path.join(model_dir,"variables/variables.data-00000-of-00001")):
    variables_size = os.path.getsize(os.path.join(
        model_dir,"variables/variables.data-00000-of-00001"))
    variables_size += os.path.getsize(os.path.join(
        model_dir,"variables/variables.index"))

  print(f"Model size: {round(pb_size / (1024.0),3)} KB")
  print(f"Variables size: {round( variables_size / (1024.0),3)} KB")
  print(f"Total Size: {round((pb_size + variables_size) / (1024.0),3)} KB")


def freeze_session(session, keep_var_names = None, input_names = None, output_names = None, clear_devices = True, transforms = None):
  '''
    Freeze the tensorflow session and produce a frozen graph.
    :transforms: list of transforms to apply when optimising the graph
  '''
  graph = sess.graph

  with graph.as_default():
      freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))

      output_names = output_names or []
      input_names = input_names or []
      input_graph_def = graph.as_graph_def()

      if clear_devices:
          for node in input_graph_def.node:
              node.device = ""

      frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
          session, input_graph_def, output_names, freeze_var_names)

      print("*" * 80)
      print("FROZEN GRAPH SUMMARY")
      describe_graph(frozen_graph)
      print("*" * 80)

      if transforms:
        optimized_graph = TransformGraph(frozen_graph, input_names, output_names, transforms)
        print("*" * 80)
        print("OPTIMIZED GRAPH SUMMARY")
        describe_graph(optimized_graph)
        print("*" * 80)
        return optimized_graph
      else:
        return frozen_graph


def freeze_model(model, name):
  '''
    Freeze the keras model and write the frozen graph 
  '''
  frozen_graph = freeze_session(sess,
      output_names = [out.op.name for out in model.outputs][:4],
      input_names = [input_tensor.op.name for input_tensor in model.inputs][:4],
      transforms = TRANSFORMS)

  directory = FROZEN_MODEL_PATH
  tf.compat.v1.train.write_graph(frozen_graph, directory, name , as_text = False)
  print("*" * 80)
  print("Finished converting the keras model to a frozen graph")
  print("PATH: ", directory)
  print("*" * 80)


def make_serving_ready(model_path, save_serve_path, version_number):
  '''
    Converts frozen graph to a tf-serving compatible model
  '''
  export_dir = os.path.join(save_serve_path, str(version_number))
  graph_pb = model_path

  builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

  with tf.io.gfile.GFile(graph_pb, "rb") as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())

  sigs = {}

  with tf.Session(graph = tf.Graph()) as sess:
      # name="" is important to ensure we don't get spurious prefixing
      tf.import_graph_def(graph_def, name = "")
      g = tf.get_default_graph()
      input_image = g.get_tensor_by_name("input_image:0")
      input_image_meta = g.get_tensor_by_name("input_image_meta:0")
      input_anchors = g.get_tensor_by_name("input_anchors:0")

      output_detection = g.get_tensor_by_name("mrcnn_detection/Reshape_1:0")
      output_mask = g.get_tensor_by_name("mrcnn_mask/Reshape_1:0")

      sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
          tf.saved_model.signature_def_utils.predict_signature_def(
              {"input_image": input_image, "input_image_meta": input_image_meta, "input_anchors": input_anchors},
              {"mrcnn_detection/Reshape_1": output_detection, "mrcnn_mask/Reshape_1": output_mask})

      builder.add_meta_graph_and_variables(sess,
                                           [tag_constants.SERVING],
                                           signature_def_map=sigs)

  builder.save()


if __name__ == '__main__':
  # Get coco config
  coco_config = get_coco_config()

  # Load maask rcnn keras model and the pretrained weights
  model = MaskRCNN(mode = "inference", model_dir = KERAS_MODEL_DIR, config = coco_config)
  model.load_weights(KERAS_WEIGHTS_PATH, by_name = True)

  # Converts the keras model to a frozen graph
  freeze_model(model.keras_model, FROZEN_NAME)

  # Convert the frozen graph to a tf serving model
  make_serving_ready(os.path.join(FROZEN_MODEL_PATH, FROZEN_NAME),
                       TF_SERVING_MODEL_PATH,
                       VERSION_NUMBER)

  # Print the size of the tf-serving model
  print("*" * 80)
  get_model_size(os.path.join(TF_SERVING_MODEL_PATH, str(VERSION_NUMBER)))
  print("*" * 80)
  print("COMPLETED")

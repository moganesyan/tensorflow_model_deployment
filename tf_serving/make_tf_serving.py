import tensorflow as tf
import keras.backend as K
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.compat.v1 import saved_model
import os
import sys

import export_config

sys.path.append(export_config.MRCNN_DIR)
sys.path.append(export_config.COCO_DIR)

from coco.coco import CocoConfig
from mrcnn.model import MaskRCNN

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


def get_model_size(export_dir, version, model_file = "saved_model.pb"):
  '''
    Get size of the produced tf-serving model and count number of variables. This is a diagnostic function.
  '''

  model_dir = os.path.join(export_dir, str(version))
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


def freeze_model(model, transforms = None, clear_devices = True):
  input_names = [input_tensor.op.name for input_tensor in model.inputs][:4]
  output_names = [out.op.name for out in model.outputs][:4]
  freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()))

  g = tf.compat.v1.get_default_graph()
  input_graph_def = g.as_graph_def()

  if clear_devices:
      for node in input_graph_def.node:
          node.device = ""

  frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
      master_session, input_graph_def, output_names, freeze_var_names)

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


def export_saved_model(export_dir, version):
  export_dir = os.path.join(export_dir, str(version))
  builder = saved_model.builder.SavedModelBuilder(export_dir)
  signature = {}

  g = tf.compat.v1.get_default_graph()

  input_image = saved_model.build_tensor_info(g.get_tensor_by_name("input_image:0"))
  input_image_meta = saved_model.build_tensor_info(g.get_tensor_by_name("input_image_meta:0"))
  input_anchors = saved_model.build_tensor_info(g.get_tensor_by_name("input_anchors:0"))

  output_detection = saved_model.build_tensor_info(g.get_tensor_by_name("mrcnn_detection/Reshape_1:0"))
  output_mask = saved_model.build_tensor_info(g.get_tensor_by_name("mrcnn_mask/Reshape_1:0"))

  signature[saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
    saved_model.signature_def_utils.build_signature_def(
      inputs = {"input_image": input_image, "input_image_meta": input_image_meta, "input_anchors": input_anchors},
      outputs = {"mrcnn_detection/Reshape_1": output_detection, "mrcnn_mask/Reshape_1": output_mask},
      method_name = saved_model.signature_constants.PREDICT_METHOD_NAME)

  builder.add_meta_graph_and_variables(export_session,
                                       [saved_model.tag_constants.SERVING],
                                       signature_def_map = signature)
  builder.save()


if __name__ == '__main__':
  # Get coco config
  coco_config = get_coco_config()

  # Load maask rcnn keras model and the pretrained weights
  model = MaskRCNN(mode = "inference", model_dir = export_config.KERAS_MODEL_DIR, config = coco_config)
  model.load_weights(export_config.KERAS_WEIGHTS_PATH, by_name = True)

  with K.get_session() as master_session:
    graph_def = freeze_model(model.keras_model, transforms = export_config.TRANSFORMS)

    with tf.Session(graph = tf.Graph()) as export_session:
      tf.import_graph_def(graph_def, name = "")
      export_saved_model(export_config.EXPORT_DIR, export_config.VERSION_NUMBER)

  # Print the size of the tf-serving model
  print("*" * 80)
  get_model_size(export_config.EXPORT_DIR, export_config.VERSION_NUMBER)
  print("*" * 80)
  print("COMPLETED")

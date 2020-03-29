import os
import io
import sys

import flask
import cv2

import numpy as np

# import tensorflow.python.keras.backend as K
# import tensorflow as tf

MRCNN_DIR = os.path.abspath('../external/mask_rcnn/')
COCO_DIR = os.path.abspath('../external/mask_rcnn/samples')
MODEL_DIR = os.path.abspath('../tf_serving/keras_model/')
MODEL_PATH = os.path.abspath('../tf_serving/keras_model/mask_rcnn_tags_0001.h5')

sys.path.append(MRCNN_DIR)
sys.path.append(COCO_DIR)

from coco.coco import CocoConfig
from mrcnn.model import MaskRCNN

app = flask.Flask(__name__)


def get_coco_config():
  '''
    Get inference config for MS-COCO
    Ammend hyperparameters as necessary
  '''
  class InferenceConfig(CocoConfig):
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1

  return InferenceConfig()

# sess = K.get_session()
# graph = sess.graph
# graph.as_default()
# print([n.name for n in tf.get_default_graph().as_graph_def().node])

def load_model():
  global model
  coco_config = get_coco_config()
  model = MaskRCNN(mode = "inference", model_dir = MODEL_DIR, config = coco_config)
  model.load_weights(MODEL_PATH, by_name = True)
  model.keras_model._make_predict_function()

  print('Loaded model...')


@app.route("/predict", methods=["POST"]) 
def predict():
  output_payload = {"success": False}
  if flask.request.method == 'POST':
      if flask.request.files.get("image"):
          input_payload = flask.request.files.get("image")
          img_stream = io.BytesIO(input_payload.read())
          image = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          results = model.detect([image], verbose = 1)
          results = results[0]
          results['rois'] = results['rois'].tolist()
          results['class_ids'] = results['class_ids'].tolist()
          results['scores'] = results['scores'].tolist()
          results['masks'] = results['masks'].tolist()
          output_payload["predictions"] = results
          output_payload["success"] = True

  return flask.jsonify(output_payload)

if __name__ == '__main__':
  print('Hello world')
  load_model()
  app.run()

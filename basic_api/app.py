import os
import io
import sys
import cv2
import flask
import numpy as np

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
    Get inference config for MS-COCO.
    Ammend hyperparameters as necessary.
    '''
    class InferenceConfig(CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    return InferenceConfig()


def load_model():
    """Initialises the Mask RCNN and loads weights.
    """
    global model
    coco_config = get_coco_config()
    model = MaskRCNN(mode = "inference", model_dir = MODEL_DIR, config = coco_config)
    model.load_weights(MODEL_PATH, by_name = True)
    model.keras_model._make_predict_function()
    print('Loaded model...')


@app.route("/predict", methods=["POST"]) 
def predict():
    """Returns JSON object outout_payload.
    output_payload['detections']: Output masks, rois, classes and scores for each object and image
    output_payload['success']: True/False tag for detection success
    """
    output_payload = {"detections": [],"success": False}
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

            output_payload["detections"] = results
            output_payload["success"] = True
    return flask.jsonify(output_payload)

if __name__ == '__main__':
    print('Hello world')
    load_model()
    app.run()

import json
import numpy as np
import io
import os
import requests
from PIL import Image
from flask import Blueprint, jsonify, request

from configs.flask_config import Config
from configs.logging_config import get_logger
from configs.mrcnn_config import MrcnnConfig

from utils.transform import ImageTransform
from utils.preprocess import PreprocessImage
from utils.postprocess import PostprocessImage


_logger = get_logger(logger_name = __name__)
coco_detector = Blueprint('coco_detector', __name__)


flask_config = Config()
mrcnn_config = MrcnnConfig()
transformClass = ImageTransform(mrcnn_config)
preprocessClass = PreprocessImage(mrcnn_config)
postprocessClass = PostprocessImage(mrcnn_config)


def prepare_payload(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.asarray(image)
    image = transformClass.transform(image)
    molded_image, image_meta, anchors, window =preprocessClass.preprocess_input(image)
    payload = {
                "signature_name": "serving_default",
                "instances": [
                {'input_anchors': anchors.tolist(),
                'input_image': molded_image.tolist(),
                'input_image_meta': image_meta.tolist()}
                ]
                }
    return payload, image.shape, molded_image.shape, window


@coco_detector.route('/health', methods = ['GET'])
def health():
	if request.method == 'GET':
		#_logger.info('health status is ok')
		return 'ok'


@coco_detector.route("/predict", methods=["POST"])    
def predict():
    data = {"success": False,
    		"predictions": []}
    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()

            image = Image.open(io.BytesIO(image))
            payload, image_shape, molded_image_shape, window = prepare_payload(image)
            # _logger.debug(f'{payload}')
            
            headers = {"content-type": "application/json"}
            target_host = str(os.environ['TARGET_HOST'])
            json_response = requests.post(f'http://{target_host}/v1/models/coco_test:predict', data=json.dumps(payload), headers=headers)
            response = json_response.json()

            _logger.debug(f'{json_response.status_code}')
            # _logger.debug(f'{response}')
            mrcnn_detection = np.array(response['predictions'][0]['mrcnn_detection/Reshape_1']).reshape(-1, *(mrcnn_config.IMAGES_PER_GPU, mrcnn_config.DETECTION_MAX_INSTANCES, 6))
            mrcnn_mask = np.array(response['predictions'][0]['mrcnn_mask/Reshape_1']).reshape(-1, *(mrcnn_config.IMAGES_PER_GPU, mrcnn_config.DETECTION_MAX_INSTANCES, mrcnn_config.MASK_SHAPE[0], mrcnn_config.MASK_SHAPE[1], mrcnn_config.NUM_CLASSES))

            final_rois, final_class_ids, final_scores, final_masks = \
                postprocessClass.unmold_detections(
                    mrcnn_detection,
                    mrcnn_mask,
                    image_shape,
                    molded_image_shape,
                    window
                )

            results = {'class_ids': final_class_ids.tolist(),
                        'rois': final_rois.tolist(),
                        'scores': final_scores.tolist(),
                        'masks': (final_masks*1).tolist()}

            data["predictions"] = results
            data["success"] = True
    return jsonify(data)
			

@coco_detector.route('/version', methods = ['GET'])
def version():
	if request.method == 'GET':
		return jsonify({'model_version': mrcnn_config.MODEL_VERSION,
						'api_version': flask_config.API_VERSION
						})

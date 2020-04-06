#!/bin/bash
nvidia-docker build -t coco-mrcnn .
nvidia-docker run -p 8500:8500 --mount type=bind,source=/home/mikheil/Documents/CLOTHES/dev/tensorflow_model_deployment/tf_serving/serving_model,target=/models/coco_mrcnn -t coco-mrcnn &
 
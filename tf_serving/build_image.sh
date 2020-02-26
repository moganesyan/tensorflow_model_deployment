#!/bin/bash
docker build -t coco-mrcnn .
docker run -p 8500:8500 --mount type=bind,source=/Users/moganesyan/Documents/sphere_python/dev/tensorflow_model_deployment/tf_serving/serving_model,target=/models/coco_mrcnn -t coco-mrcnn &
 
#!/bin/bash
docker run -d --name serving_base tensorflow/serving:latest
PWD=$(pwd)
docker cp $PWD/serving_model/ serving_base:/models/coco_test
docker commit --change "ENV MODEL_NAME coco_test" serving_base mikheil/coco_test
docker kill serving_base
docker rm serving_base
docker run -p 8501:8501 -t mikheil/coco_test &

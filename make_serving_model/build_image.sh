#!/bin/bash
nvidia-docker run -d --name serving-base tensorflow/serving:latest
PWD=$(pwd)
nvidia-docker cp $PWD/serving_model/ serving-base:/models/coco_test
nvidia-docker commit --change "ENV MODEL_NAME coco_test" serving-base mikheil/coco-test
nvidia-docker kill serving-base
nvidia-docker rm serving-base
#docker run -p 8501:8501 -t mikheil/coco_test &
nvidia-docker tag  mikheil/coco-test gcr.io/scraper-playground/coco-test:latest
gcloud docker -- push gcr.io/scraper-playground/coco-test:latest

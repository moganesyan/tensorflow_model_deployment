#!/bin/bash
WORKDIR=$(pwd)
nvidia-docker build -t coco-mrcnn .
nvidia-docker run -p 8500:8500 -p 8501:8501 -t coco-mrcnn --model_config_file=/models/model_config.tf &
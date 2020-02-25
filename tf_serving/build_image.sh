#!/bin/bash
docker build -t coco-mrcnn .
docker run -p 8501:8501 -t coco-mrcnn &

#!/bin/bash
nvidia-docker build -t mikheil/coco-test-flask .
nvidia-docker tag mikheil/coco-test-flask gcr.io/scraper-playground/coco-test-flask
gcloud docker -- push gcr.io/scraper-playground/coco-test-flask


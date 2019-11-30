#!/bin/bash
gcloud container clusters create load-test-coco \
   --zone us-east1-b \
   --enable-autoscaling \
   --enable-stackdriver-kubernetes \
   --min-nodes "3" --max-nodes "10" \
   --addons HorizontalPodAutoscaling,HttpLoadBalancing
nvidia-docker build -t slackdrop/load-test-coco ./docker-image
nvidia-docker tag slackdrop/load-test-coco gcr.io/scraper-playground/load-test-coco
gcloud docker -- push gcr.io/scraper-playground/load-test-coco

gcloud config set container/cluster load-test-coco
gcloud container clusters get-credentials load-test-coco --zone us-east1-b --project scraper-playground

kubectl apply -f kubernetes-config/locust-master-service.yaml
kubectl apply -f kubernetes-config/locust-master-controller.yaml
kubectl apply -f kubernetes-config/locust-worker-controller.yaml

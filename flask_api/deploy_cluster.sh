#!/bin/bash
gcloud container clusters create coco-test-flask \
   --zone us-east1-b \
   --machine-type=n1-highmem-2 \
   --enable-autoscaling \
   --enable-stackdriver-kubernetes \
   --min-nodes "3" --max-nodes "10" \
   --addons HorizontalPodAutoscaling,HttpLoadBalancing
gcloud config set container/cluster coco-test-flask
gcloud container clusters get-credentials coco-test-flask --zone us-east1-b --project scraper-playground
kubectl apply -f kubernetes_config/namespace.yaml
kubectl apply -f kubernetes_config/service.yaml
kubectl apply -f kubernetes_config/deployment.yaml

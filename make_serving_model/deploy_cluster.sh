#!/bin/bash
sudo apt install jq
echo "Creating cluster coco-test"
gcloud container clusters create coco-test \
--machine-type=n1-standard-4 \
--region us-east1-d \
--node-labels=label-a=model-node \
--node-locations=us-east1-d \
--image-type=ubuntu \
--num-nodes=2 \
--min-nodes=0 \
--max-nodes=2 \
--enable-autoscaling \
--scopes "https://www.googleapis.com/auth/compute","https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
--enable-stackdriver-kubernetes \
--addons HorizontalPodAutoscaling,HttpLoadBalancing
echo "Succesfully created cluster coco-test"
gcloud container clusters get-credentials coco-test --region us-east1-d
echo "Installing NVIDIA drivers (may take up to a minute)..."
#kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-preloaded.yaml
sleep 30s
echo "Succesfully installed NVIDIA drivers..."
echo "Deploying the latest model image to the cluster coco-test..."
kubectl apply -f kubernetes_config/namespace.yaml
kubectl apply -f kubernetes_config/service.yaml
loadBalancerIP=""
while [[ "$loadBalancerIP" == "" ]] || [[ "$loadBalancerIP" == "null" ]]
do
  sleep 10s

  svc_json=$(kubectl get svc -l app=coco-test --namespace prod -o json)

  loadBalancerIP=$(echo $svc_json | jq -r ".items[].status.loadBalancer.ingress[0].ip")
done
kubectl apply -f kubernetes_config/deployment.yaml
echo "Building image..."
availableStatus="0"
while [[ "$availableStatus" == "0" ]] || [[ "$availableStatus" == "null" ]]
do
  sleep 10s

  status_json=$(kubectl get deployments --namespace prod -o json)

  availableStatus=$(echo $status_json | jq -r ".items[].status.availableReplicas")
done
kubectl autoscale deployment coco-test --cpu-percent=80 --min=1 --max=4 --namespace=prod
echo "Succesfully deployed the latest model to the cluster coco-test..."
echo "The external ip is $loadBalancerIP"

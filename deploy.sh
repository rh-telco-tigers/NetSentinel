#!/bin/bash

set -e  # Exit on any error
set -u  # Treat unset variables as an error

### Configuration Variables ###
# DOMAIN="cluster-4b8nj.4b8nj.sandbox81.opentlc.com"
# MAAS_TOKEN="c17e3bbfe5424cc594788243205d48b7"
# SLACK_BOT_TOKEN="xoxb-7834804921362-8126334417046-44zknlYfRlonLRFK1jXULGz0"
# SLACK_SIGNING_SECRET="ce06e4bf1c817f4611d5b339d3ad7f17"

read -p "Enter the LLM MAAS Token: " MAAS_TOKEN
read -p "Enter the Slack Bot Token: " SLACK_BOT_TOKEN
read -p "Enter the Slack Signing Secret: " SLACK_SIGNING_SECRET


# Dynamically derive the DOMAIN
SERVER_URL=$(oc whoami --show-server)
if [[ -z "$SERVER_URL" ]]; then
  echo "Error: Could not retrieve OpenShift server URL. Ensure you are logged in using 'oc login'."
  exit 1
fi
DOMAIN=$(echo "$SERVER_URL" | sed -E 's~https://api\.~~;s~:.*~~')
echo "Derived OpenShift DOMAIN: $DOMAIN"

### Utility Functions ###

# Waits for all pods in a given namespace to be ready
wait_for_pods() {
  local namespace=$1
  local timeout=300  # Timeout in seconds
  echo "Waiting for pods in namespace '${namespace}' to be ready..."
  
  SECONDS=0
  while [[ $SECONDS -lt $timeout ]]; do
    local all_ready=true
    local pods=$(oc get pods -n "$namespace" --no-headers || true)
    
    if [[ -z $pods ]]; then
      echo "No pods found in namespace '${namespace}'. Retrying..."
      all_ready=false
    fi

    while IFS= read -r pod; do
      local ready=$(echo "$pod" | awk '{print $2}')
      local total=$(echo "$ready" | cut -d'/' -f2)
      local current=$(echo "$ready" | cut -d'/' -f1)
      if [[ "$current" != "$total" ]]; then
        echo "Pod not ready: $pod"
        all_ready=false
        break
      fi
    done <<< "$pods"

    if [[ $all_ready == true ]]; then
      echo "All pods in namespace '${namespace}' are ready."
      return 0
    fi
    sleep 10
  done

  echo "Timeout waiting for pods in namespace '${namespace}'. Exiting."
  exit 1
}

# Applies resources from a kustomization path and waits for pod readiness
apply_and_wait() {
  local kustomization_path=$1
  echo "Applying resources from ${kustomization_path}..."
  oc apply -k "$kustomization_path"

  local namespaces=$(grep -r 'namespace:' "$kustomization_path" | awk -F'namespace:' '{print $2}' | sed 's/^[ \t]*//' | sort | uniq)
  for namespace in $namespaces; do
    if [[ -n $namespace ]]; then
      wait_for_pods "$namespace"
    fi
  done
}

# Sets up the S3 command for MinIO and ensures the bucket exists
setup_s3() {
  local bucket_name=$1
  local s3_endpoint=$(oc get routes minio-api -o jsonpath='{.spec.host}' -n netsentinel)
  
  echo "S3 Endpoint: $s3_endpoint"
  S3_COMMAND="aws s3 --endpoint-url https://$s3_endpoint"

  if ! $S3_COMMAND ls "s3://${bucket_name}" &>/dev/null; then
    echo "Creating S3 bucket '${bucket_name}'..."
    $S3_COMMAND mb "s3://${bucket_name}"
  else
    echo "S3 bucket '${bucket_name}' already exists."
  fi
}

# Uploads model files to S3 if they don't already exist
upload_model_files() {
  local bucket_name=$1
  if $S3_COMMAND ls "s3://${bucket_name}/predictive-model/config.pbtxt" &>/dev/null && \
     $S3_COMMAND ls "s3://${bucket_name}/predictive-model/1/model.onnx" &>/dev/null; then
    echo "Predictive model files already exist in S3. Skipping upload."
  else
    echo "Uploading predictive model files to S3..."
    $S3_COMMAND cp v1/config.pbtxt "s3://${bucket_name}/predictive-model/config.pbtxt"
    $S3_COMMAND cp v1/1/model.onnx "s3://${bucket_name}/predictive-model/1/model.onnx"
  fi
}

# Clones the predictive model repository if not already present
clone_predictive_model_repo() {
  local repo_url=$1
  local repo_dir="predictive-model"
  
  if [ -d "$repo_dir" ]; then
    echo "'$repo_dir' directory already exists. Skipping clone step."
  else
    echo "Cloning predictive model repository..."
    git clone "$repo_url" "$repo_dir"
  fi
}

### Deployment Steps ###

echo "Applying namespace configurations..."
oc apply -k k8s/namespaces/base

echo "Deploying operators..."
apply_and_wait "k8s/operators/overlays/common"

echo "Deploying instance configurations..."
apply_and_wait "k8s/instances/overlays/common"

echo "Updating Kafka configurations..."
find ./k8s/instances/overlays/rhlab/kafka/ -type f -exec sed -i '' "s/<CLUSTER_NAME_WITH_BASE_DOMAIN>/$DOMAIN/g" {} +
apply_and_wait "k8s/instances/overlays/rhlab"

echo "Setting up the predictive model..."
setup_s3 "netsentinel"
clone_predictive_model_repo "https://huggingface.co/bkpandey/netsentinel"
cd predictive-model/
upload_model_files "netsentinel"
cd ..

echo "Deploying model configurations..."
apply_and_wait "k8s/apps/base/models/"

echo "Updating application configuration..."
sed -i '' "s/token: \"<YOUR_API_KEY_HERE>\"/token: \"$MAAS_TOKEN\"/" k8s/apps/overlays/rhlab/netsentinel/app-config.yaml

echo "Deploying the application..."
apply_and_wait "k8s/apps/overlays/rhlab"

echo "Deployment completed successfully."

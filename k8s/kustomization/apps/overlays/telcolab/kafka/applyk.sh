#!/bin/bash

# Path to the expected secret file
SECRET_FILE="./console-ui-secrets.yaml"

# Check if the secret file exists
if [[ ! -f "$SECRET_FILE" ]]; then
  echo "Error: Missing secret file '$SECRET_FILE'."
  echo "Please run the following command to generate it:"
  echo "    ../../../base/kafka/generate-console-secrets.sh"
  exit 1
fi

# Run 'oc apply -k .' if the secret file exists
echo "Applying Kustomize overlay..."
oc apply -k .

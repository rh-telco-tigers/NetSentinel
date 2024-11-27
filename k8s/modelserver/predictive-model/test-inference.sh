#!/bin/bash

# Namespace and route name
NAMESPACE="multimodelserver"
ROUTE_NAME="netsentinel"

# Fetch the route host dynamically
ROUTE_HOST=$(oc get route $ROUTE_NAME -n $NAMESPACE -o jsonpath='{.spec.host}')

# Check if the route was fetched successfully
if [[ -z "$ROUTE_HOST" ]]; then
  echo "Error: Unable to fetch route. Ensure the route exists and you have the correct namespace."
  exit 1
fi

# Construct the full URL
FULL_URL="https://$ROUTE_HOST/v2/models/netsentinel/infer"

# Define the input data inline
INPUT_DATA=$(cat <<EOF
{
  "inputs": [
    {
      "name": "float_input",
      "shape": [1, 47],
      "datatype": "FP32",
      "data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
               1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
               2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
               3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
               4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7]
    }
  ]
}
EOF
)

# Make the curl request
echo "Sending request to: $FULL_URL"
curl -X POST \
  -H "Content-Type: application/json" \
  -d "$INPUT_DATA" \
  $FULL_URL

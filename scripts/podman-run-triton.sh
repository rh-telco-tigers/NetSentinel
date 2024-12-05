podman run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /Users/bkpandey/Downloads/models:/models \
  nvcr.io/nvidia/tritonserver:24.11-py3 \
  tritonserver --model-repository=/models

## Inference
podman run -it --rm --platform linux/amd64 \
  --net=host \
  nvcr.io/nvidia/tritonserver:24.11-py3-sdk

cat <<EOF > input.json 
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

curl -X POST -H "Content-Type: application/json" \
  -d @input.json \
  http://localhost:8000/v2/models/netsentinel/infer



## References:
# https://github.com/triton-inference-server/server
# https://github.com/NVIDIA-AI-IOT/tao-toolkit-triton-apps/
# https://ai-on-openshift.io/odh-rhoai/custom-runtime-triton
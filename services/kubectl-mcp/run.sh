
podman build -t kubectl-ai-api . 

export OPENAI_API_BASE=https://deepseek-r1-distill-qwen-14b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com/v1
export OPENAI_API_KEY=""
export LLM_PROVIDER=openai
export LLM_MODEL=deepseek-r1-distill-qwen-14b

export OPENAI_API_BASE=https://mistral-7b-instruct-v0-3-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1
export OPENAI_API_KEY=""
export LLM_PROVIDER=openai
export LLM_MODEL=mistral-7b-instruct

podman run --rm -it \
  -v ~/.kube/config:/root/.kube/config \
  -p 8088:8088 \
  -e OPENAI_API_BASE=$OPENAI_API_BASE \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e LLM_PROVIDER=$LLM_PROVIDER \
  -e LLM_MODEL=$LLM_MODEL \
  kubectl-ai-api



curl -X POST "$OPENAI_API_BASE/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'


curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     $OPENAI_API_BASE/models


curl -s -X POST http://localhost:8088/mcp/k8s \
  -H "Content-Type: application/json" \
  -d '{
    "query": "list all pods from namespace netsentinel"
  }' | jq .



curl -s -X POST http://localhost:8088/mcp/k8s \
  -H "Content-Type: application/json" \
  -d '{
    "query": "List all pods in namespace netsentinel. Respond ONLY with a JSON code block with the following format: ```json {\"action\": {\"name\": \"kubectl\", \"command\": \"kubectl get pods -n netsentinel\"}}```"
  }' | jq .

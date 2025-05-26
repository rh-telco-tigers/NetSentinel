FROM python:3.11-slim

# Install required packages including kubectl
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    && curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl \
    && pip install flask \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download kubectl-ai binary
RUN curl -L -o /tmp/kubectl-ai.tar.gz https://github.com/GoogleCloudPlatform/kubectl-ai/releases/download/v0.0.10/kubectl-ai_Linux_x86_64.tar.gz && \
    tar -xzf /tmp/kubectl-ai.tar.gz -C /usr/local/bin && \
    chmod +x /usr/local/bin/kubectl-ai

# Copy Flask app and config
WORKDIR /app
COPY app.py /app/app.py
COPY prompts/prompt_tool_use.txt /app/prompts/prompt_tool_use.txt
COPY tools/tools.yaml /app/tools/tools.yaml

EXPOSE 8088

CMD ["python", "app.py"]

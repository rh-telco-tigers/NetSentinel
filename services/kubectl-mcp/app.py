from flask import Flask, request, jsonify
import subprocess
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

# Paths and env variables
KUBECTL_AI_PATH = "/usr/local/bin/kubectl-ai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
TOOLS_CONFIG = "/app/tools/tools.yaml"
TRACE_PATH = "/tmp/kubectl-ai-trace.log"

@app.route("/mcp/k8s", methods=["POST"])
def handle_k8s():
    payload = request.json
    cmd_string = payload.get("query")

    if not cmd_string:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        env = os.environ.copy()

        cmd = [
            KUBECTL_AI_PATH,
            "--quiet",
            "--enable-tool-use-shim",
            "--llm-provider", LLM_PROVIDER,
            "--model", LLM_MODEL,
            "--custom-tools-config", TOOLS_CONFIG,
            "--trace-path", TRACE_PATH,
            # "--prompt-template-file-path", "/app/prompts/prompt_tool_use.txt",
            "--skip-permissions", 
            "-v=10",
            cmd_string
        ]

        logging.debug(f"Running command: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        logging.debug(f"[stdout]: {proc.stdout}")
        logging.debug(f"[stderr]: {proc.stderr}")

        # Optionally read trace file if exists
        if os.path.exists(TRACE_PATH):
            with open(TRACE_PATH, "r") as trace_file:
                trace_output = trace_file.read()
                logging.debug("[trace.log]:\n" + trace_output)

        if proc.returncode != 0:
            return jsonify({"error": proc.stderr.strip()}), 500

        return jsonify({"result": proc.stdout.strip()})

    except Exception as e:
        logging.exception("Exception occurred during command execution")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088)

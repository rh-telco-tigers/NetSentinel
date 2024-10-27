#!/bin/bash
# scripts/reconnaissance.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
TARGET_NETWORK=$(yq e '.reconnaissance.target_network' "$CONFIG_FILE")

echo "Starting Reconnaissance Attack: Stealth SYN Scan on $TARGET_NETWORK..."

# Perform stealth SYN scan
nmap -sS -p1-65535 "$TARGET_NETWORK"

echo "Reconnaissance Attack Completed."

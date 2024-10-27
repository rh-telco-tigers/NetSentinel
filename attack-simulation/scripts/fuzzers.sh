#!/bin/bash
# scripts/fuzzers.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
TARGET_URL=$(yq e '.fuzzers.target_url' "$CONFIG_FILE")
WORDLIST=$(yq e '.fuzzers.wordlist' "$CONFIG_FILE")

echo "Starting Fuzzers Attack on $TARGET_URL using wordlist $WORDLIST..."

# Execute wfuzz
wfuzz -c -z file,"$WORDLIST" -u "$TARGET_URL" --hc 404

echo "Fuzzers Attack Completed."

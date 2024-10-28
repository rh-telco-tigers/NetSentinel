#!/bin/bash
# scripts/worms.sh

CONFIG_FILE="/etc/attack-config.yml"

# Extract variables using yq
TARGET_IP=$(yq e '.worms.target_ip' "$CONFIG_FILE")
TARGET_PORT=$(yq e '.worms.target_port' "$CONFIG_FILE")

echo "Starting Worm Attack: Propagating to $TARGET_IP:$TARGET_PORT..."

# Create a simple worm script
cat << 'EOF' > /opt/attack-scripts/worm.py
import socket
import subprocess

TARGET_IP = "192.168.1.100"
TARGET_PORT = 4444

def propagate():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((TARGET_IP, TARGET_PORT))
        s.send(b'Worm message')
        s.close()
    except Exception as e:
        print(f"Propagation failed: {e}")

def execute():
    # Execute a payload or command
    subprocess.call(['echo', 'Worm Executed'])

if __name__ == "__main__":
    propagate()
    execute()
EOF

# Update worm.py with actual TARGET_IP and TARGET_PORT
sed -i "s/TARGET_IP = .*/TARGET_IP = \"$TARGET_IP\"/" /opt/attack-scripts/worm.py
sed -i "s/TARGET_PORT = .*/TARGET_PORT = $TARGET_PORT/" /opt/attack-scripts/worm.py

# Execute the worm
python3 /opt/attack-scripts/worm.py

echo "Worm Attack Completed."

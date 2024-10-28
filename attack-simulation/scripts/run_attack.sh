#!/bin/bash
# scripts/run_attack.sh

# Check if an attack type is provided
if [ -z "$ATTACK_TYPE" ]; then
    echo "Error: No attack type specified."
    echo "Usage: docker run --rm -e ATTACK_TYPE=<attack_type> attack-simulation"
    echo "Available attack types: fuzzers, analysis, backdoors, dos, exploits, generic, reconnaissance, shellcode, worms, brute_force"
    exit 1
fi

# Navigate to scripts directory
cd /opt/attack-scripts

# Execute the corresponding attack script
case "$ATTACK_TYPE" in
    fuzzers)
        ./fuzzers.sh
        ;;
    analysis)
        ./analysis.sh
        ;;
    backdoors)
        ./backdoors.sh
        ;;
    dos)
        ./dos.sh
        ;;
    exploits)
        ./exploits.sh
        ;;
    generic)
        ./generic.sh
        ;;
    reconnaissance)
        ./reconnaissance.sh
        ;;
    shellcode)
        ./shellcode.sh
        ;;
    worms)
        ./worms.sh
        ;;
    brute_force)
        ./brute_force.sh
        ;;
    *)
        echo "Error: Unknown attack type '$ATTACK_TYPE'."
        echo "Available attack types: fuzzers, analysis, backdoors, dos, exploits, generic, reconnaissance, shellcode, worms, brute_force"
        exit 1
        ;;
esac

# Attack Simulation Framework

## Overview

This project provides a unified framework to simulate nine distinct types of network attacks using Docker Compose and Kubernetes. It leverages various security tools to generate realistic attack traffic, enabling the evaluation of your Zeek-based classification model.

## Attack Types

1. **Fuzzers**
2. **Analysis**
3. **Backdoors**
4. **Denial of Service (DoS)**
5. **Exploits**
6. **Generic Attacks**
7. **Reconnaissance**
8. **Shellcode**
9. **Worms**
10. **Brute Force** (Kubernetes Deployment)


## Setup Instructions

### Prerequisites

- **Docker:** [Install Docker](https://docs.docker.com/engine/install/)
- **Docker Compose:** [Install Docker Compose](https://docs.docker.com/compose/install/)
- **Kubernetes Cluster:** Ensure access to a Kubernetes cluster (e.g., Minikube, GKE, EKS)
- **Kubectl:** [Install Kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)

### Configuration

1. **Edit Attack Configurations:**

   Update `config/attack-config.yml` with your target IPs, ports, and other parameters.

```yaml
   # config/attack-config.yml

   fuzzers:
     target_url: "http://192.168.1.100/login.php"
     wordlist: "/usr/share/wfuzz/wordlists/general/common.txt"

   analysis:
     target_ip: "192.168.1.100"
     target_port: 443

   backdoors:
     lhost: "192.168.1.50"
     lport: 4444

   dos:
     target_ip: "192.168.1.100"
     target_port: 80
     method: "GET"
     threads: 100

   exploits:
     target_ip: "192.168.1.100"
     payload: "windows/meterpreter/reverse_tcp"
     lhost: "192.168.1.50"
     lport: 4444

   generic:
     target_ip: "192.168.1.100"
     target_port: 80

   reconnaissance:
     target_network: "192.168.1.0/24"

   shellcode:
     lhost: "192.168.1.50"
     lport: 4444

   worms:
     target_ip: "192.168.1.100"
     target_port: 4444

   brute_force:
     target_service: "ssh://192.168.1.100:22"
     username: "admin"
     password_file: "/usr/share/wordlists/rockyou.txt"
```

### Building the Docker Image

1. Navigate to Project Directory:
```
cd attack-simulation
```

2. Build the Docker Image:
```
docker-compose build
```

### Running Attack Simulations with Docker Compose
1. Start a Specific Attack:
- Example: Running a DoS attack.
```
ATTACK_TYPE=dos docker-compose up
```
- Example: Running a Reconnaissance attack.
```
ATTACK_TYPE=reconnaissance docker-compose up
```
# Dockerfile

FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        python3-pip \
        wget \
        curl \
        mono-complete \
        hping3 \
        nmap \
        hydra \
        sqlmap \
        openjdk-11-jdk \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
        libpcap-dev \
        libsqlite3-dev \
        libyaml-dev \
        && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install additional tools
# Install wfuzz
RUN pip3 install wfuzz

# Install Metasploit Framework
RUN curl https://raw.githubusercontent.com/rapid7/metasploit-framework/master/scripts/docker_install.sh | bash

# Clone LOIC repository
RUN git clone https://github.com/NewEraCracker/LOIC.git /opt/LOIC && \
    cd /opt/LOIC/LowOrbitIonCannon && \
    xbuild /p:Configuration=Release

# Install yq for YAML parsing in scripts
RUN wget https://github.com/mikefarah/yq/releases/download/v4.6.1/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# Clone Atomic Red Team repository
RUN git clone https://github.com/redcanaryco/atomic-red-team.git /opt/atomic-red-team

# Set up scripts directory
RUN mkdir /opt/attack-scripts

# Copy attack scripts into the container
COPY scripts/ /opt/attack-scripts/
RUN chmod +x /opt/attack-scripts/*.sh

# Set the entrypoint script
ENTRYPOINT ["/opt/attack-scripts/run_attack.sh"]

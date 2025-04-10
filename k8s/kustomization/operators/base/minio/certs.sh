#!/bin/bash

mkdir -p ./base/certs

# Define the domain for MinIO
DOMAIN="minio-dspa-netsentinel.apps.cloud.xtoph152.dfw.ocp.run"

# Generate a private key
openssl genrsa -out ./base/certs/tls.key 2048

# Generate a self-signed certificate with the domain name
openssl req -x509 -new -nodes -key ./base/certs/tls.key -sha256 -days 365 -out ./base/certs/tls.crt -subj "/CN=$DOMAIN"
cp ./base/certs/tls.crt ./base/certs/ca-bundle.crt
openssl x509 -in ./base/certs/tls.crt -text -noout

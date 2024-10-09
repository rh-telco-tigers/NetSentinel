#!/bin/bash

mkdir -p ./base/certs

# Generate the private key and certificate
openssl genrsa -out ./base/certs/tls.key 2048
openssl req -x509 -new -nodes -key ./base/certs/tls.key -sha256 -days 365 -out ./base/certs/tls.crt -subj "/CN=minio/O=minio"

# Optionally, you can copy the certificate to the CA bundle
cp ./base/certs/tls.crt ./base/certs/ca-bundle.crt

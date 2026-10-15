#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== Generating CCVNN Root CA ==="
openssl req -x509 -newkey rsa:4096 -days 365 -nodes \
  -keyout ca.key -out ca.crt \
  -subj "/C=EG/ST=Cairo/L=Cairo/O=CCVNN-Edge/OU=Security/CN=CCVNN-Root-CA"

echo "=== Generating Triton Server Certificate ==="
openssl req -newkey rsa:2048 -nodes \
  -keyout server.key -out server.csr \
  -subj "/C=EG/ST=Cairo/L=Cairo/O=CCVNN-Edge/OU=Inference/CN=localhost"

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 365

echo "=== Generating Edge Ingestion Client Certificate ==="
openssl req -newkey rsa:2048 -nodes \
  -keyout client.key -out client.csr \
  -subj "/C=EG/ST=Cairo/L=Cairo/O=CCVNN-Edge/OU=Ingestion/CN=edge-worker-01"

openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out client.crt -days 365

chmod 600 ca.key server.key client.key
echo "=== PKI Certificates Successfully Generated in certs/ ==="

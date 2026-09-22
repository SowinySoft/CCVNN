#!/bin/bash
set -e

# Configuration
BACKUP_DIR="${1:-/mnt/usb/ccvnn_backup}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="ccvnn_stack_${TIMESTAMP}.tar.gz"

echo "==> Creating backup directory at ${BACKUP_DIR}..."
mkdir -p "${BACKUP_DIR}"

echo "==> Exporting Docker images..."
sudo docker save \
  ccvnn-plc-sim \
  ccvnn-watcher-engine \
  eclipse-mosquitto:2.0 \
  nvcr.io/nvidia/tritonserver:26.08-py3 \
  timescale/timescaledb:latest-pg15 \
  prom/prometheus:latest \
  grafana/grafana:latest \
  -o "${BACKUP_DIR}/ccvnn_images.tar"

echo "==> Backing up configuration files, rules, and init scripts..."
tar -czf "${BACKUP_DIR}/ccvnn_config_${TIMESTAMP}.tar.gz" \
  docker-compose.yml \
  config/ \
  db/ \
  prometheus.yml \
  prometheus_rules.yml \
  model_repository/ 2>/dev/null || true

echo "==> Exporting active Docker volumes (if any)..."
sudo docker run --rm \
  -v ccvnn_pgdata:/pgdata \
  -v "${BACKUP_DIR}:/backup" \
  alpine tar -czf "/backup/ccvnn_pgdata_${TIMESTAMP}.tar.gz" -C /pgdata . 2>/dev/null || echo "No existing pgdata volume found to back up."

echo "==> Export complete! All artifacts saved to ${BACKUP_DIR}"
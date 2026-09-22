#!/bin/bash
set -e

# Configuration (Point this to where your backup archive is located)
BACKUP_DIR="${1:-/mnt/usb/ccvnn_backup}"

if [ ! -d "$BACKUP_DIR" ]; then
  echo "Error: Backup directory $BACKUP_DIR does not exist!"
  exit 1
fi

echo "==> Loading Docker images..."
if [ -f "${BACKUP_DIR}/ccvnn_images.tar" ]; then
  sudo docker load -i "${BACKUP_DIR}/ccvnn_images.tar"
else
  echo "Warning: ccvnn_images.tar not found in ${BACKUP_DIR}"
fi

echo "==> Restoring configuration files..."
LATEST_CONFIG=$(ls -t ${BACKUP_DIR}/ccvnn_config_*.tar.gz 2>/dev/null | head -n 1)
if [ -n "$LATEST_CONFIG" ]; then
  tar -xzf "$LATEST_CONFIG"
  echo "Configurations restored successfully."
else
  echo "Warning: No configuration archive found."
fi

echo "==> Restoring database volume (if backup exists)..."
LATEST_PGDATA=$(ls -t ${BACKUP_DIR}/ccvnn_pgdata_*.tar.gz 2>/dev/null | head -n 1)
if [ -n "$LATEST_PGDATA" ]; then
  sudo docker volume create ccvnn_pgdata || true
  sudo docker run --rm \
    -v ccvnn_pgdata:/pgdata \
    -v "${BACKUP_DIR}:/backup" \
    alpine sh -c "rm -rf /pgdata/* && tar -xzf /backup/$(basename "$LATEST_PGDATA") -C /pgdata"
  echo "Database volume restored."
fi

echo "==> Bringing up the CCVNN stack..."
sudo docker compose up -d

echo "==> Import and deployment complete! Run 'sudo docker compose ps' to verify."
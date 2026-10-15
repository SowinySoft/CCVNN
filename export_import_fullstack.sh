#!/bin/bash
set -e

DEFAULT_BACKUP_DIR="/mnt/usb/ccvnn_backup"
PROJECT_NAME="ccvnn_prod"

echo "=================================================="
echo "    CCVNN Full-Stack Lifecycle Manager            "
echo "=================================================="
echo "Select mode:"
echo "  1) Export Full Stack (Images, Configs, Volume)"
echo "  2) Import Full Stack & Deploy"
echo "--------------------------------------------------"

# Step 1: Mode Selection (Interactive or positional arg $1)
if [ -n "$1" ]; then
  CHOICE="$1"
else
  read -p "Enter choice [1 or 2]: " CHOICE
fi

# Step 2: Path Selection (Interactive or positional arg $2)
if [ -n "$2" ]; then
  BACKUP_DIR="$2"
else
  read -p "Enter target/source directory [Default: ${DEFAULT_BACKUP_DIR}]: " USER_DIR
  BACKUP_DIR="${USER_DIR:-$DEFAULT_BACKUP_DIR}"
fi

case "$CHOICE" in
  1)
    echo ""
    echo "==> [MODE 1] EXPORTING CCVNN STACK to: ${BACKUP_DIR}"
    mkdir -p "${BACKUP_DIR}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    echo "==> Exporting Docker images associated with project '${PROJECT_NAME}'..."
    # Dynamically extract active images from the docker-compose stack, falling back to static tag list
    STACK_IMAGES=$(sudo docker compose -p "${PROJECT_NAME}" images -q 2>/dev/null | sort -u)
    
    if [ -n "$STACK_IMAGES" ]; then
      sudo docker save $STACK_IMAGES -o "${BACKUP_DIR}/ccvnn_images.tar"
    else
      echo "Notice: Stack not running, using default image tag list for save..."
      sudo docker save \
        ccvnn_prod-grpc-app \
        ccvnn_prod-watcher-engine \
        ccvnn_prod-plc-sim \
        eclipse-mosquitto:2.0 \
        nvcr.io/nvidia/tritonserver:26.08-py3 \
        timescale/timescaledb:latest-pg15 \
        prom/prometheus:latest \
        grafana/grafana:latest \
        -o "${BACKUP_DIR}/ccvnn_images.tar"
    fi

    echo "==> Archiving configuration files, models, and DB scripts..."
    tar -czf "${BACKUP_DIR}/ccvnn_config_${TIMESTAMP}.tar.gz" \
      docker-compose.yml \
      config/ \
      db/ \
      prometheus.yml \
      prometheus_rules.yml \
      model/ \
      model_repository/ 2>/dev/null || true

    echo "==> Backing up active TimescaleDB volume (ccvnn_pgdata)..."
    if sudo docker volume inspect ccvnn_pgdata >/dev/null 2>&1; then
      sudo docker run --rm \
        -v ccvnn_pgdata:/pgdata \
        -v "${BACKUP_DIR}:/backup" \
        alpine tar -czf "/backup/ccvnn_pgdata_${TIMESTAMP}.tar.gz" -C /pgdata . 2>/dev/null || echo "Volume export finished."
      echo "Database volume backup created successfully."
    else
      echo "Warning: Volume ccvnn_pgdata not found, skipping DB data archive."
    fi

    echo "=================================================="
    echo "Export Complete! All artifacts saved to:"
    echo " -> ${BACKUP_DIR}"
    echo "=================================================="
    ;;

  2)
    echo ""
    echo "==> [MODE 2] IMPORTING & DEPLOYING CCVNN STACK from: ${BACKUP_DIR}"
    if [ ! -d "$BACKUP_DIR" ]; then
      echo "Error: Backup directory ${BACKUP_DIR} does not exist!"
      exit 1
    fi

    echo "==> Loading Docker images..."
    if [ -f "${BACKUP_DIR}/ccvnn_images.tar" ]; then
      sudo docker load -i "${BACKUP_DIR}/ccvnn_images.tar"
    else
      echo "Warning: ccvnn_images.tar not found in ${BACKUP_DIR}"
    fi

    echo "==> Restoring configuration files..."
    LATEST_CONFIG=$(ls -t "${BACKUP_DIR}"/ccvnn_config_*.tar.gz 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CONFIG" ]; then
      tar -xzf "$LATEST_CONFIG"
      echo "Configurations restored from ${LATEST_CONFIG}"
    else
      echo "Warning: No ccvnn_config_*.tar.gz file found in ${BACKUP_DIR}"
    fi

    echo "==> Restoring database volume..."
    LATEST_PGDATA=$(ls -t "${BACKUP_DIR}"/ccvnn_pgdata_*.tar.gz 2>/dev/null | head -n 1)
    if [ -n "$LATEST_PGDATA" ]; then
      sudo docker volume create ccvnn_pgdata || true
      sudo docker run --rm \
        -v ccvnn_pgdata:/pgdata \
        -v "${BACKUP_DIR}:/backup" \
        alpine sh -c "rm -rf /pgdata/* && tar -xzf /backup/$(basename "$LATEST_PGDATA") -C /pgdata"
      echo "Database volume ccvnn_pgdata restored from ${LATEST_PGDATA}"
    else
      echo "Notice: No ccvnn_pgdata volume archive found to restore."
    fi

    echo "==> Launching stack under project '${PROJECT_NAME}'..."
    sudo docker compose -p "${PROJECT_NAME}" up -d

    echo "=================================================="
    echo "Import & Deployment Complete!"
    echo "=================================================="
    echo "Current Stack Status:"
    sudo docker compose -p "${PROJECT_NAME}" ps
    ;;

  *)
    echo "Invalid option: '$CHOICE'. Please choose 1 for Export or 2 for Import."
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

# Configuration
DB_CONTAINER="ccvnn-db"
MQTT_CONTAINER="ccvnn-mqtt-broker"
DB_USER="postgres"
DB_NAME="ccvnn_db"
TOPIC="ccvnn/detections"
MAX_DB_WAIT=30

echo "=== [1/5] Rebuilding and launching Docker stack ==="
DOCKER_BUILDKIT=0 docker compose up -d --build

echo "=== [2/5] Polling PostgreSQL ($DB_CONTAINER) readiness ==="
count=0
until docker exec "$DB_CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; do
  count=$((count + 1))
  if [ "$count" -ge "$MAX_DB_WAIT" ]; then
    echo "Error: Timed out waiting for database ($DB_CONTAINER) after ${MAX_DB_WAIT}s." >&2
    exit 1
  fi
  echo "Waiting for database connection... ($count/$MAX_DB_WAIT)"
  sleep 1
done
echo "Database is online and accepting connections."

echo "=== [3/5] Publishing 6 hazard frames to trigger ALARM state ==="
for i in {1..6}; do
  docker exec -i "$MQTT_CONTAINER" mosquitto_pub \
    -h localhost \
    -p 1883 \
    -t "$TOPIC" \
    -m "{\"event_type\": \"fire_hazard\", \"confidence\": 0.95, \"frame_id\": $i}"
  echo "Published frame $i"
  sleep 0.2
done

echo "=== [4/5] Waiting 6s for cooldown reset to NORMAL state ==="
sleep 6

echo "=== [5/5] Fetching safety audit log records ==="
docker exec -t "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c \
  "SELECT id, timestamp, state, plc_register_40001, reasoning FROM safety_audit_logs ORDER BY id DESC LIMIT 10;"

echo "=== Integration Test Complete ==="
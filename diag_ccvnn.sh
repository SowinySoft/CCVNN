#!/usr/bin/env bash
# CCVNN Stack Self-Healing Diagnostic Script

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "    CCVNN Stack Diagnostic & Auto-Recovery        "
echo "=================================================="

restart_and_wait() {
  local container="$1"
  local wait_sec="${2:-4}"
  echo -e "  ${YELLOW}➜ Auto-recovery: Restarting '${container}'...${NC}"
  docker restart "${container}" >/dev/null 2>&1
  sleep "${wait_sec}"
}

# 1. Container Running Status
echo -n "[1/5] Checking Container Running Status... "
CONTAINERS=("ccvnn-mqtt-broker" "ccvnn-plc-sim" "ccvnn-db" "ccvnn-watcher-engine")
ALL_RUNNING=true

for c in "${CONTAINERS[@]}"; do
  if ! docker ps --format '{{.Names}}' | grep -q "^${c}$"; then
    echo -e "\n  ${RED}✘ Container '${c}' is not running.${NC}"
    ALL_RUNNING=false
    restart_and_wait "${c}" 3
  fi
done

if [ "$ALL_RUNNING" = false ]; then
  echo -n "  Re-evaluating Container Status... "
  ALL_RUNNING=true
  for c in "${CONTAINERS[@]}"; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${c}$"; then
      ALL_RUNNING=false
    fi
  done
fi

if [ "$ALL_RUNNING" = true ]; then
  echo -e "${GREEN}PASS${NC}"
else
  echo -e "${RED}FAIL: Could not start all required containers.${NC}"
  exit 1
fi

# 2. MQTT Broker Health Check
echo -n "[2/5] Testing MQTT Broker (ccvnn-mqtt-broker:1883)... "
test_mqtt() {
  docker exec ccvnn-mqtt-broker mosquitto_pub -h localhost -p 1883 -t "healthcheck/diag" -m "ping" >/dev/null 2>&1
}

if test_mqtt; then
  echo -e "${GREEN}PASS${NC}"
else
  echo -e "\n  ${RED}✘ MQTT Broker non-responsive.${NC}"
  restart_and_wait "ccvnn-mqtt-broker" 3
  if test_mqtt; then
    echo -e "  Re-test MQTT Broker: ${GREEN}PASS (Recovered)${NC}"
  else
    echo -e "  Re-test MQTT Broker: ${RED}FAIL${NC}"
  fi
fi

# 3. Watcher -> Modbus PLC Communication
echo -n "[3/5] Testing Watcher -> Modbus PLC (ccvnn-plc-sim:5020)... "
test_plc() {
  docker exec ccvnn-watcher-engine python3 -c "
from pymodbus.client import ModbusTcpClient
client = ModbusTcpClient('ccvnn-plc-sim', port=5020)
if client.connect():
    res = client.read_holding_registers(address=0, count=1)
    if not res.isError():
        print('OK')
    client.close()
" 2>/dev/null || echo "FAIL"
}

if [ "$(test_plc)" = "OK" ]; then
  echo -e "${GREEN}PASS${NC}"
else
  echo -e "\n  ${RED}✘ Modbus PLC communication failed.${NC}"
  restart_and_wait "ccvnn-plc-sim" 3
  if [ "$(test_plc)" = "OK" ]; then
    echo -e "  Re-test Modbus PLC: ${GREEN}PASS (Recovered)${NC}"
  else
    echo -e "  Re-test Modbus PLC: ${RED}FAIL${NC}"
  fi
fi

# 4. Watcher -> TimescaleDB Communication
echo -n "[4/5] Testing Watcher -> TimescaleDB (ccvnn-db:5432)... "
test_db() {
  docker exec ccvnn-watcher-engine python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='ccvnn-db',
        port=5432,
        dbname='ccvnn_db',
        user='postgres',
        password='postgrespassword',
        connect_timeout=3
    )
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM safety_audit_logs LIMIT 1;')
    conn.close()
    print('OK')
except Exception:
    print('FAIL')
" 2>/dev/null || echo "FAIL"
}

if [ "$(test_db)" = "OK" ]; then
  echo -e "${GREEN}PASS${NC}"
else
  echo -e "\n  ${RED}✘ Database connection or schema check failed.${NC}"
  restart_and_wait "ccvnn-db" 5
  if [ "$(test_db)" = "OK" ]; then
    echo -e "  Re-test TimescaleDB: ${GREEN}PASS (Recovered)${NC}"
  else
    echo -e "  Re-test TimescaleDB: ${RED}FAIL${NC}"
  fi
fi

# 5. Watcher Process State
echo -n "[5/5] Verifying Watcher Daemon Process... "
test_watcher() {
  docker exec ccvnn-watcher-engine pgrep -f "watcher/daemon.py" >/dev/null 2>&1
}

if test_watcher; then
  echo -e "${GREEN}PASS${NC}"
else
  echo -e "\n  ${RED}✘ Watcher daemon process is not active inside container.${NC}"
  restart_and_wait "ccvnn-watcher-engine" 4
  if test_watcher; then
    echo -e "  Re-test Watcher Daemon: ${GREEN}PASS (Recovered)${NC}"
  else
    echo -e "  Re-test Watcher Daemon: ${RED}FAIL${NC}"
  fi
fi

echo "=================================================="
echo " Auto-Recovery Diagnostics Complete "
echo "=================================================="
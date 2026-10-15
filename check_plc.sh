#!/usr/bin/env bash

PLC_IP="192.168.10.45"
PLC_PORT="502"

echo "=================================================="
echo "  CCVNN V17 - PHYSICAL PLC NETWORK DIAGNOSTICS"
echo "  Target PLC: ${PLC_IP}:${PLC_PORT}"
echo "=================================================="

# 1. Check ICMP Ping Reachability
echo -n "[1/2] Checking ICMP ping reachability... "
if ping -c 1 -W 2 "$PLC_IP" > /dev/null 2>&1; then
    echo -e "\e[32m[REACHABLE]\e[0m"
else
    echo -e "\e[31m[UNREACHABLE]\e[0m"
    echo "[-] Error: Host ${PLC_IP} is not responding to ping. Check power and network cabling."
    exit 1
fi

# 2. Check Modbus TCP Port 502
echo -n "[2/2] Testing Modbus TCP Port ${PLC_PORT}... "
if python3 -c "import socket; s = socket.socket(); s.settimeout(2); exit(0 if s.connect_ex(('$PLC_IP', $PLC_PORT)) == 0 else 1)" 2>/dev/null; then
    echo -e "\e[32m[OPEN]\e[0m"
else
    echo -e "\e[31m[CLOSED/TIMEOUT]\e[0m"
    echo "[-] Error: Port ${PLC_PORT} on ${PLC_IP} is closed or blocked by firewall."
    exit 1
fi

echo -e "\n\e[32m[✓] All PLC diagnostics passed successfully!\e[0m"
echo "--- Launching Batch Verifier Script ---"
python test_batch_verifier.py

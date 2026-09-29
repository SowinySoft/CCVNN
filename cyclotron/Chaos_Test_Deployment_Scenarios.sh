
#Scenario A: Flapping Network / High-Frequency Dropped Packets

  python3 modbus_chaos_proxy.py \
  --listen-port 5020 \
  --target-host 192.168.1.100 \
  --target-port 502 \
  --failure-rate 0.40 \
  --mode timeout
  
  #Scenario B: High Latency / Slow PLC Response
  
  python3 modbus_chaos_proxy.py \
  --listen-port 5020 \
  --target-host 192.168.1.100 \
  --target-port 502 \
  --failure-rate 0.60 \
  --mode latency \
  --min-delay-ms 1500 \
  --max-delay-ms 3500
  
  
  #Scenario C: Abrupt Socket Resets (Hardware Cable Unplug)
  python3 modbus_chaos_proxy.py \
  --listen-port 5020 \
  --target-host 192.168.1.100 \
  --target-port 502 \
  --failure-rate 0.50 \
  --mode reset
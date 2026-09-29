python3 scripts/mock_modbus_server.py &
python3 scripts/modbus_chaos_proxy.py --listen-port 5020 --target-port 5022 --failure-rate 0.50 --mode timeout &
ctest --test-dir build --output-on-failure
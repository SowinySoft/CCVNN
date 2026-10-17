from pymodbus.client import ModbusTcpClient
import sys

def main():
    client = ModbusTcpClient("127.0.0.1", port=5020, timeout=2)
    try:
        if not client.connect():
            sys.exit(1)
        response = client.read_holding_registers(address=0, count=1, slave=1)
        if response is not None and not response.isError():
            sys.exit(0)
        else:
            sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    main()

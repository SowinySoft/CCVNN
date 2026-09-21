import socketserver
import struct

class ModbusTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(f"\n[+] Incoming Modbus TCP Connection from {self.client_address[0]}:{self.client_address[1]}")
        while True:
            data = self.request.recv(1024)
            if not data:
                break
            
            # Parse MBAP Header (Transaction ID, Protocol ID, Length, Unit ID)
            if len(data) >= 7:
                transaction_id, protocol_id, length, unit_id = struct.unpack(">HHHB", data[:7])
                function_code = data[7]
                
                print(f"[MODBUS RECV] Transaction ID: {transaction_id} | Function Code: 0x{function_code:02X}")
                
                # Function Code 0x05 (Write Single Coil) or 0x0F (Write Multiple Coils) or 0x06 (Write Holding Reg)
                if function_code in (0x05, 0x06, 0x0F, 0x10):
                    output_address = struct.unpack(">H", data[8:10])[0]
                    write_value = struct.unpack(">H", data[10:12])[0] if len(data) >= 12 else 1
                    
                    print(f"[✓] SUCCESS: Actuated Register Address: {output_address} | Value: {write_value}")
                
                # Echo back the Modbus response frame to acknowledge successful write
                response = data
                self.request.sendall(response)

def run_server():
    port = 5020
    print("==================================================")
    print("  STANDALONE MODBUS TCP MOCK SERVER ACTIVE")
    print(f"  Listening on: 127.0.0.1:{port}")
    print("==================================================\n")
    
    server = socketserver.TCPServer(("127.0.0.1", port), ModbusTCPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Stopping Mock Modbus Server...")
        server.server_close()

if __name__ == "__main__":
    run_server()

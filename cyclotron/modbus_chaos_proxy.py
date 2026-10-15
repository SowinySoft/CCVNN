#!/usr/bin/env python3
import asyncio
import random
import time
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CHAOS] %(message)s",
    datefmt="%H:%M:%S"
)

class ChaosMode:
    NOMINAL = "nominal"         # Passthrough without interference
    TIMEOUT = "timeout"         # Silent blackhole (causes client TCP timeout)
    RESET = "reset"             # Immediate socket termination (RST)
    LATENCY = "latency"         # Inject high delay before forwarding

class ModbusChaosProxy:
    def __init__(self, listen_host: str, listen_port: int, target_host: str, target_port: int,
                 failure_rate: float, chaos_mode: str, min_delay_ms: float, max_delay_ms: float):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.target_host = target_host
        self.target_port = target_port
        self.failure_rate = failure_rate
        self.chaos_mode = chaos_mode
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.listen_host, self.listen_port
        )
        logging.info(f"Modbus Chaos Proxy listening on {self.listen_host}:{self.listen_port} "
                     f"--> Target {self.target_host}:{self.target_port}")
        logging.info(f"Chaos Configuration: mode={self.chaos_mode}, failure_rate={self.failure_rate * 100:.1f}%")
        
        async with server:
            await server.serve_forever()

    async def handle_client(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter):
        client_addr = client_writer.get_extra_info('peername')
        
        # Decide if chaos applies to this connection / request batch
        inject_fault = random.random() < self.failure_rate

        if inject_fault:
            if self.chaos_mode == ChaosMode.RESET:
                logging.warning(f"[{client_addr}] INJECTING FAULT: Abrupt TCP Reset")
                client_writer.close()
                await client_writer.wait_closed()
                return

            elif self.chaos_mode == ChaosMode.TIMEOUT:
                logging.warning(f"[{client_addr}] INJECTING FAULT: Blackhole (Connection Timeout)")
                # Consume client data without forwarding or replying to induce client socket timeout
                try:
                    while True:
                        data = await client_reader.read(1024)
                        if not data:
                            break
                except Exception:
                    pass
                return

            elif self.chaos_mode == ChaosMode.LATENCY:
                delay = random.uniform(self.min_delay_ms, self.max_delay_ms) / 1000.0
                logging.warning(f"[{client_addr}] INJECTING FAULT: Artificial Delay ({delay*1000:.1f}ms)")
                await asyncio.sleep(delay)

        # Connect to upstream target Modbus TCP server
        try:
            target_reader, target_writer = await asyncio.open_connection(
                self.target_host, self.target_port
            )
        except Exception as e:
            logging.error(f"[{client_addr}] Upstream connection to Modbus server failed: {e}")
            client_writer.close()
            await client_writer.wait_closed()
            return

        # Bidirectional proxying
        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, name: str):
            try:
                while True:
                    data = await reader.read(1024)
                    if not data:
                        break
                    writer.write(data)
                    await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()

        await asyncio.gather(
            forward(client_reader, target_writer, "client->target"),
            forward(target_reader, client_writer, "target->client")
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Modbus TCP Fault Injection & Chaos Engineering Proxy")
    parser.add_argument("--listen-port", type=int, default=5020, help="Local proxy listening port")
    parser.add_argument("--target-host", type=str, default="127.0.0.1", help="Target Modbus TCP server host")
    parser.add_argument("--target-port", type=int, default=502, help="Target Modbus TCP server port")
    parser.add_argument("--failure-rate", type=float, default=0.4, help="Probability of fault injection (0.0 - 1.0)")
    parser.add_argument("--mode", type=str, choices=[ChaosMode.NOMINAL, ChaosMode.TIMEOUT, ChaosMode.RESET, ChaosMode.LATENCY],
                        default=ChaosMode.TIMEOUT, help="Chaos injection strategy")
    parser.add_argument("--min-delay-ms", type=float, default=1500.0, help="Minimum latency in ms (for latency mode)")
    parser.add_argument("--max-delay-ms", type=float, default=3000.0, help="Maximum latency in ms (for latency mode)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    proxy = ModbusChaosProxy(
        listen_host="0.0.0.0",
        listen_port=args.listen_port,
        target_host=args.target_host,
        target_port=args.target_port,
        failure_rate=args.failure_rate,
        chaos_mode=args.mode,
        min_delay_ms=args.min_delay_ms,
        max_delay_ms=args.max_delay_ms
    )
    try:
        asyncio.run(proxy.start())
    except KeyboardInterrupt:
        logging.info("Chaos proxy stopped by user.")
import argparse
import asyncio
import logging
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModbusSim")

async def run_server(host: str, port: int):
    # Initialize holding registers (40001+) initialized to zero
    store = ModbusSlaveContext(
        hr=ModbusSequentialDataBlock(0, [0] * 100),
        zero_mode=True
    )
    context = ModbusServerContext(slaves=store, single=True)

    logger.info(f"Starting Modbus TCP Simulator on {host}:{port}...")
    await StartAsyncTcpServer(context=context, address=(host, port))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCVNN Modbus TCP PLC Simulator")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5020, help="Port to listen on")
    args = parser.parse_args()

    asyncio.run(run_server(args.host, args.port))
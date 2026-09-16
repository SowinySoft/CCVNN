import argparse
import asyncio
import logging
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext

# Defensive import for PyModbus version variations (3.0-3.7 vs 3.8+)
try:
    from pymodbus.datastore import ModbusSlaveContext
except ImportError:
    try:
        from pymodbus.datastore.store import ModbusSlaveContext
    except ImportError:
        ModbusSlaveContext = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModbusSim")

async def run_server(host: str, port: int):
    block = ModbusSequentialDataBlock(0, [0] * 1000)
    
    if ModbusSlaveContext is not None:
        store = ModbusSlaveContext(di=block, co=block, hr=block, ir=block, zero_mode=True)
        context = ModbusServerContext(slaves=store, single=True)
    else:
        # PyModbus 3.8+ fallback
        try:
            context = ModbusServerContext(slaves=block, single=True)
        except TypeError:
            context = ModbusServerContext(di=block, co=block, hr=block, ir=block, single=True)

    logger.info(f"Starting Modbus TCP Simulator on {host}:{port}...")
    await StartAsyncTcpServer(context=context, address=(host, port))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCVNN Modbus TCP PLC Simulator")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5020, help="Port to listen on")
    args = parser.parse_args()

    asyncio.run(run_server(args.host, args.port))
import argparse
import asyncio
import logging
import sys
from typing import Any

import pymodbus
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
)
from pymodbus.server import StartAsyncTcpServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ModbusTCPSimulator")


def create_server_context() -> ModbusServerContext:
    """Constructs a robust datastore context supporting PyModbus v2/v3 variations."""
    block = ModbusSequentialDataBlock(0, [0] * 1000)

    try:
        store = ModbusSlaveContext(
            di=block, co=block, hr=block, ir=block, zero_mode=True
        )
    except TypeError:
        store = ModbusSlaveContext(di=block, co=block, hr=block, ir=block)

    try:
        return ModbusServerContext(slaves=store, single=True)
    except TypeError:
        try:
            return ModbusServerContext(devices=store, single=True)
        except TypeError:
            return ModbusServerContext(context=store, single=True)


async def run_server(host: str, port: int) -> None:
    """Runs the asynchronous Modbus TCP simulator server."""
    context = create_server_context()
    logger.info(
        f"Starting CCVNN Modbus TCP Simulator on {host}:{port} "
        f"(PyModbus {pymodbus.__version__})..."
    )

    try:
        await StartAsyncTcpServer(
            context=context,
            address=(host, port),
        )
    except asyncio.CancelledError:
        logger.info("Modbus TCP Simulator server task cancelled.")
    except Exception as e:
        logger.error(f"Error running Modbus TCP Simulator: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCVNN Modbus TCP Industrial PLC Simulator"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Binding IP address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5020, help="Port number (default: 5020)"
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_server(args.host, args.port))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Modbus TCP Simulator stopped cleanly.")


if __name__ == "__main__":
    main()
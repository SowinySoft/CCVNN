import argparse
import asyncio
import logging

from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
)
from pymodbus.server import StartAsyncTcpServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModbusSim")


async def run_server(host: str, port: int) -> None:
    block = ModbusSequentialDataBlock(0, [0] * 1000)

    try:
        # PyModbus versions exposing ModbusDeviceContext.
        from pymodbus.datastore import ModbusDeviceContext

        device = ModbusDeviceContext(
            di=block,
            co=block,
            hr=block,
            ir=block,
        )
        context = ModbusServerContext(devices=device, single=True)
    except ImportError:
        # Older PyModbus versions.
        from pymodbus.datastore import ModbusSlaveContext

        store = ModbusSlaveContext(
            di=block,
            co=block,
            hr=block,
            ir=block,
            zero_mode=True,
        )
        context = ModbusServerContext(slaves=store, single=True)

    logger.info("Starting Modbus TCP Simulator on %s:%s", host, port)
    await StartAsyncTcpServer(
        context=context,
        address=(host, port),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCVNN Modbus TCP PLC Simulator"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5020)
    args = parser.parse_args()

    asyncio.run(run_server(args.host, args.port))


if __name__ == "__main__":
    main()
import argparse
import asyncio
import logging

from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
)
from pymodbus.server import StartAsyncTcpServer

# Initialize logger and base config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_server(host: str, port: int) -> None:
    block = ModbusSequentialDataBlock(0, [0] * 1000)

    store = ModbusSlaveContext(
        di=block,
        co=block,
        hr=block,
        ir=block,
        zero_mode=True,
    )
    context = ModbusServerContext(
        slaves=store,
        single=True,
    )

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
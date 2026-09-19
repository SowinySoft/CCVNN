import argparse
import asyncio
import logging
from typing import Optional

import pymodbus
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import StartAsyncTcpServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MockPLC")


class MockPLCServer:
    """Programmable Mock PLC Modbus TCP Server for integration & test suites."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5020, register_count: int = 100):
        self.host = host
        self.port = port
        self.register_count = register_count
        self.block = ModbusSequentialDataBlock(1, [0] * self.register_count)
        self.context = self._build_context()

    def _build_context(self) -> ModbusServerContext:
        """Constructs datastore handling dynamic API variations across PyModbus versions."""
        store = None
        for cls_name in ["ModbusSlaveContext", "ModbusDeviceContext"]:
            try:
                mod = __import__("pymodbus.datastore", fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                try:
                    store = cls(hr=self.block, zero_mode=True)
                except TypeError:
                    store = cls(hr=self.block)
                if store is not None:
                    break
            except (ImportError, AttributeError):
                continue

        if store is None:
            store = self.block

        try:
            return ModbusServerContext(slaves=store, single=True)
        except TypeError:
            try:
                return ModbusServerContext(context=store, single=True)
            except TypeError:
                try:
                    return ModbusServerContext(devices=store, single=True)
                except TypeError:
                    return ModbusServerContext(store)

    def get_register_value(self, address: int) -> int:
        """Retrieves holding register value for test verification."""
        return self.block.getValues(address, 1)[0]

    def set_register_value(self, address: int, value: int) -> None:
        """Overrides holding register value to simulate hardware state transitions."""
        self.block.setValues(address, [value])

    async def start(self) -> None:
        """Starts the async Modbus TCP server."""
        logger.info(
            f"Starting Mock Modbus TCP PLC Server on {self.host}:{self.port} "
            f"(PyModbus {pymodbus.__version__})..."
        )
        await StartAsyncTcpServer(context=self.context, address=(self.host, self.port))


async def main() -> None:
    parser = argparse.ArgumentParser(description="CCVNN Mock PLC Test Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=5020, help="Port number")
    args = parser.parse_args()

    server = MockPLCServer(host=args.host, port=args.port)
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Mock PLC Server execution terminated.")
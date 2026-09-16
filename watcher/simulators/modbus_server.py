import argparse
import asyncio
import logging
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext

# Defensive import for PyModbus version variations (3.0-3.7 vs 3.8+)
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
)

async def run_server(host: str, port: int):
    block = ModbusSequentialDataBlock(0, [0] * 1000)

    try:
        # PyModbus 3.8+
        from pymodbus.datastore import ModbusDeviceContext

        device = ModbusDeviceContext(
            di=block,
            co=block,
            hr=block,
            ir=block,
        )
        context = ModbusServerContext(
            devices=device,
            single=True,
        )
    except ImportError:
        # Older PyModbus releases
        from pymodbus.datastore import ModbusSlaveContext

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
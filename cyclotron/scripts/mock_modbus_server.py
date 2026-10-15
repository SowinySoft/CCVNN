#!/usr/bin/env python3
import asyncio
import logging
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext

logging.basicConfig(level=logging.INFO)

async def main():
    block = ModbusSequentialDataBlock(0, [0] * 100)
    store = ModbusSlaveContext(hr=block, ir=block, co=block, di=block)
    context = ModbusServerContext(slaves=store, single=True)
    
    logging.info("Starting lightweight mock Modbus TCP server on port 5022...")
    await StartAsyncTcpServer(context, address=("127.0.0.1", 5022))

if __name__ == "__main__":
    asyncio.run(main())
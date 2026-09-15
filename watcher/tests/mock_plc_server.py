import asyncio
import logging
import pymodbus
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockPLC")

async def main():
    logger.info(f"Starting Mock Modbus TCP PLC Server on 0.0.0.0:5020 (PyModbus {pymodbus.__version__})...")
    
    # 1-based index block for holding registers
    block = ModbusSequentialDataBlock(1, [0] * 100)
    
    # Instantiate store without forcing zero_mode kwarg
    store = None
    for cls_name in ["ModbusSlaveContext", "ModbusDeviceContext"]:
        try:
            mod = __import__("pymodbus.datastore", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            try:
                store = cls(hr=block, zero_mode=True)
            except TypeError:
                store = cls(hr=block)
            if store is not None:
                break
        except (ImportError, AttributeError):
            continue
    
    if store is None:
        store = block

    # Instantiate ModbusServerContext dynamically based on API signature
    from pymodbus.datastore import ModbusServerContext
    try:
        context = ModbusServerContext(slaves=store, single=True)
    except TypeError:
        try:
            context = ModbusServerContext(context=store, single=True)
        except TypeError:
            try:
                context = ModbusServerContext(devices=store, single=True)
            except TypeError:
                context = ModbusServerContext(store)

    await StartAsyncTcpServer(context=context, address=("0.0.0.0", 5020))

if __name__ == "__main__":
    asyncio.run(main())

"""
Cyclotron Ingestion Buffer.
High-throughput, thread-safe dynamic ring buffer for multi-modal vector inputs
with backpressure handling and automatic batch aggregation.
"""

import asyncio
import logging
import time
from collections import deque
from typing import List, Optional
from watcher.schemas.vector_schema import VectorPayload, MultiVectorBatch

logger = logging.getLogger("WatcherIngestionBuffer")


class MultiVectorIngestionBuffer:
    def __init__(
        self,
        max_capacity: int = 1000,
        max_batch_size: int = 16,
        max_delay_ms: float = 50.0,
    ):
        self.max_capacity = max_capacity
        self.max_batch_size = max_batch_size
        self.max_delay_sec = max_delay_ms / 1000.0
        
        self._queue: deque[VectorPayload] = deque(maxlen=max_capacity)
        self._lock = asyncio.Lock()
        self._batch_counter = 0

    async def push(self, payload: VectorPayload) -> bool:
        """
        Pushes a new vector payload into the buffer.
        If capacity is reached, drops the oldest vector to guarantee low latency.
        """
        async with self._lock:
            if len(self._queue) >= self.max_capacity:
                dropped = self._queue.popleft()
                logger.warning(f"[IngestionBuffer] Capacity reached. Dropped vector ID: {dropped.vector_id}")
            
            self._queue.append(payload)
            return True

    async def get_batch(self) -> Optional[MultiVectorBatch]:
        """
        Extracts up to max_batch_size vectors. Waits up to max_delay_ms for vectors
        to accumulate before returning an available batch.
        """
        start_time = time.time()
        accumulated: List[VectorPayload] = []

        while (time.time() - start_time) < self.max_delay_sec:
            async with self._lock:
                while self._queue and len(accumulated) < self.max_batch_size:
                    accumulated.append(self._queue.popleft())

            if len(accumulated) >= self.max_batch_size:
                break

            await asyncio.sleep(0.005)  # 5ms poll interval

        if not accumulated:
            return None

        self._batch_counter += 1
        batch = MultiVectorBatch(
            batch_id=f"batch_cyclotron_{self._batch_counter}_{int(time.time() * 1000)}",
            vectors=accumulated,
        )
        batch.compute_aggregates()
        return batch

    @property
    def current_size(self) -> int:
        return len(self._queue)
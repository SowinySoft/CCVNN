import asyncio
import time
import numpy as np
from typing import List, Tuple, Any


class BatchCoalescer:
    """
    Edge dynamic request batching queue.
    Collects single REST requests into contiguous memory batches before
    dispatching them to Triton gRPC streams.
    """

    def __init__(
        self,
        dispatch_fn,
        max_batch_size: int = 8,
        max_delay_ms: float = 2.0,
        loop: asyncio.AbstractEventLoop = None,
    ):
        self.dispatch_fn = dispatch_fn
        self.max_batch_size = max_batch_size
        self.max_delay_sec = max_delay_ms / 1000.0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.loop = loop or asyncio.get_event_loop()
        self.worker_task: asyncio.Task = None
        self._running = False

    def start(self):
        self._running = True
        self.worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self):
        self._running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Enqueues a single request tensor and returns a Future awaiting the batched inference result.
        """
        future = self.loop.create_future()
        await self.queue.put((input_tensor, future))
        return await future

    async def _batch_worker(self):
        while self._running:
            # Block until at least one request arrives
            first_item, first_future = await self.queue.get()
            batch: List[Tuple[np.ndarray, asyncio.Future]] = [(first_item, first_future)]
            
            deadline = time.perf_counter() + self.max_delay_sec

            # Coalesce additional incoming requests up to max_batch_size or deadline
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    item, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append((item, future))
                except asyncio.TimeoutError:
                    break

            # Process the batch concurrently
            self.loop.create_task(self._process_batch(batch))

    async def _process_batch(self, batch: List[Tuple[np.ndarray, asyncio.Future]]):
        tensors, futures = zip(*batch)

        try:
            # Convert inputs into C-contiguous stacked batch
            batched_input = np.ascontiguousarray(np.stack(tensors, axis=0))

            # Dispatch batched tensor over gRPC stream to Triton
            batched_output = await self.dispatch_fn(batched_input)

            # Split batched results back to corresponding request Futures
            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result(batched_output[i])

        except Exception as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
# FUTURE: In Python 3.13 this can be replaced with asyncio.Queue.shutdown
from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncGenerator, Iterable
from typing import Generic, TypeVar

import aiohttp

from apicadabri import ApicadabriBulkResponse

A = TypeVar("A")
R = TypeVar("R")

class PoisonPill:
    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "inst"):
            cls.inst = PoisonPill()
        return cls.inst

class ApicadabriRecursiveResponse(ApicadabriBulkResponse[A, R], Generic[A, R], ABC):
    async def execute_task_group(self):
        self.result_queue: asyncio.Queue[R | PoisonPill] = asyncio.Queue()
        async def worker(client: aiohttp.ClientSession, args: A):
            _, res = await self.call_with_semaphore(client, 0, args)
            await self.result_queue.put(res)
        async with aiohttp.ClientSession() as client:
            async with asyncio.TaskGroup() as tg:
                self.task_group = tg
                for inst in self.instances():
                    tg.create_task(worker(client, inst))
        await self.result_queue.put(PoisonPill.get_instance())

    async def call_all(self) -> AsyncGenerator[R, None]:
        """Return an iterator that yields the results of the API calls.

        This uses a semaphore to limit the number of concurrent API calls.

        It returns results in the same order as the input arguments from
        `instances`. However, it also allows to inspect and process results
        as they arrive.
        """
        next_index = 0
        buffer: list[tuple[int, R]] = []
        # TODO: Concurrent task receives work items from queue and puts results in second queue, this task receives results from result queue and returns them
        asyncio.create_task(self.execute_task_group())
        result = await self.result_queue.get()
        while not isinstance(result, PoisonPill):
            # TODO maybe we also want to do some re-ordering here in the future?
            yield result
            result = await self.result_queue.get()            

    @abstractmethod
    def instances(self) -> Iterable[A]:
        """Generate instances of the API call arguments."""
        self.task_queue = asyncio.Queue()
# FUTURE: In Python 3.13 this can be replaced with asyncio.Queue.shutdown
import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable
from typing import Generic, Self, TypeVar

import aiohttp

from apicadabri import ApicadabriBulkResponse

A = TypeVar("A")
R = TypeVar("R")

class PoisonPill:
    """Poison pill singleton to signal that a queue should be terminated."""
    _instance = None

    def __new__(cls) -> Self:
        """Returns the singleton instance of this class."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

class ApicadabriRecursiveResponse(ApicadabriBulkResponse[A, R], Generic[A, R], ABC):
    """Response type for recursive calls that can spawn new tasks within the call.

    To use this base class you have to implement the following methods:

    * `call_api` should implement your main task(s). It will receive arguments from
        the `instances` method and can schedule new tasks via `self.task_group`.
    * `instances` must return an iterator over the arguments of the main tasks(s).

    Args:
        A: The type of the arguments that are passed to the API call.
        R: The return type of the API call.
    """
    async def execute_task_group(self) -> None:
        """Creates a task group to start the initial instances.

        This task group is saved as an instance variable allowing the instances
        to generate new tasks and schedule them within the same task group.

        This method only returns after all the tasks in the task group have actually finished.‚
        """
        async def worker(client: aiohttp.ClientSession, args: A) -> None:
            """Performs individual calls and puts result into result queue.

            Args:
                client: A shared client session to use for efficient HTTP calls.
                args: The arguments to the individual API call.
            """
            _, res = await self.call_with_semaphore(client, 0, args)
            await self.result_queue.put(res)
        async with aiohttp.ClientSession() as client, asyncio.TaskGroup() as tg:
            self.task_group = tg
            for inst in self.instances():
                tg.create_task(worker(client, inst))
        await self.result_queue.put(PoisonPill())

    async def call_all(self) -> AsyncGenerator[R, None]:
        """Return an iterator that yields the results of the API calls.

        This uses a semaphore to limit the number of concurrent API calls.

        It returns results in the same order as the input arguments from
        `instances`. However, it also allows to inspect and process results
        as they arrive.
        """
        # TODO: Concurrent task receives work items from queue and puts results in second queue, this task receives results from result queue and returns them
        # result needs to be stored to avoid garbage collection
        self.result_queue: asyncio.Queue[R | PoisonPill] = asyncio.Queue()
        self._main_task = asyncio.create_task(self.execute_task_group())
        result = await self.result_queue.get()
        while not isinstance(result, PoisonPill):
            # TODO maybe we also want to do some re-ordering here in the future?
            yield result
            result = await self.result_queue.get()

    @abstractmethod
    def instances(self) -> Iterable[A]:
        """Generate instances of the API call arguments."""
        self.task_queue = asyncio.Queue()

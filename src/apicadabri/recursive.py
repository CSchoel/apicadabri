"""Support for recursive calls where number of calls is not known beforehand."""

import asyncio
from abc import ABC
from bisect import insort_right
from collections.abc import AsyncGenerator, Iterable
from typing import Any, Generic, Literal, Protocol, Self, TypeVar

import aiohttp

from apicadabri import (
    JSON,
    ApicadabriBulkHTTPResponse,
    ApicadabriBulkResponse,
    ApicadabriCallArguments,
    ApicadabriCallInstance,
    AsyncRetrier,
    SyncedClientResponse,
)
from apicadabri.helpers import BufferedOrdererBase, Ordered

A = TypeVar("A")
R = TypeVar("R")


# FUTURE: In Python 3.13 this can be replaced with asyncio.Queue.shutdown
class PoisonPill:
    """Poison pill singleton to signal that a queue should be terminated."""

    _instance = None

    def __new__(cls) -> Self:
        """Returns the singleton instance of this class."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class ApicadabriRecursiveExecutionError(Exception):
    """Error during recursive execution.

    This error is thrown if a recursive subtask fails with an exception, leading to a
    premature insertion of a PoisonPill into the task queue.
    """

    def __init__(self) -> None:
        """Create a new error."""
        super().__init__(
            "Error during recursive task execution. Refer to the causing error to fix this.",
        )


class ApicadabriRecursiveResponse(ApicadabriBulkResponse[A, R], ABC, Generic[A, R]):
    """Response type for recursive calls that can spawn new tasks within the call.

    To use this base class you have to implement the following methods:

    * `call_api` should implement your main task(s). It will receive arguments from
        the `instances` method and can schedule new tasks via `self.schedule_subtask`.
    * `instances` must return an iterator over the arguments of the main tasks(s).

    Args:
        A: The type of the arguments that are passed to the API call. Please note
            that the same type is used for top-level and subtasks.
        R: The return type of the API call. Please note that the same type is used
            for top-level and subtasks.
    """

    def __init__(
        self,
        max_active_calls: int = 20,
        retrier: AsyncRetrier | None = None,
        *,
        return_in_order: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Creates a new response object.

        Args:
            max_active_calls: The maximum number of concurrent API calls to make.
            retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                    If None, a new instance will be created with default parameters.
            return_in_order: If True, results are returned in breadth-first order of their
                    creation in the recursive tree of subtasks. Set this to False if you
                    experience out of memory errors or long pauses and sudden bursts of
                    results in your pipeline. These effects can occur due to the buffering
                    of results required to return them in order.
            kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(max_active_calls=max_active_calls, retrier=retrier, **kwargs)
        self.indexer = SubtaskIndexer(remember_order=return_in_order)
        self.return_in_order = return_in_order

    async def execute_task_group(self) -> None:
        """Creates a task group to start the initial instances.

        This task group is saved as an instance variable allowing the instances
        to generate new tasks and schedule them within the same task group.

        This method only returns after all the tasks in the task group have actually finished.
        """
        try:
            async with aiohttp.ClientSession() as client, asyncio.TaskGroup() as tg:
                self.task_group = tg
                for idx, inst in enumerate(self.instances()):
                    await self.schedule_subtask(client, inst, (idx,))
        except Exception as e:  # noqa: BLE001 - we raise it later
            self.exception = e
        finally:
            self.result_queue.put_nowait((-1, PoisonPill()))

    async def schedule_subtask(
        self,
        client: aiohttp.ClientSession,
        instance_args: A,
        index: tuple[int, ...],
    ) -> None:
        """Schedules a subtask in the internal task group.

        The task will be called via `self.call_api` and benefits from retry
        and rate limiting functionality of Apicadabri.

        The result will be put into an internal result queue, which is
        processed automatically.
        """
        int_index = await self.indexer.add_tuple_index(index)
        self.task_group.create_task(self.execute_task(client, instance_args, int_index))

    async def execute_task(
        self,
        client: aiohttp.ClientSession,
        instance_args: A,
        index: int,
    ) -> None:
        """Performs individual API calls and puts result into result queue.

        Args:
            client: A shared client client to use for efficient HTTP calls.
            instance_args: The arguments to the individual API call.
            index: The index of the task (generated by `SubtaskCreator`).
        """
        index, res = await self.call_with_semaphore(client, index, instance_args)
        await self.result_queue.put((index, res))

    async def call_all(self) -> AsyncGenerator[R, None]:
        """Return an iterator that yields the results of the API calls.

        This uses a semaphore to limit the number of concurrent API calls.

        It returns results in the same order as the input arguments from
        `instances`. However, it also allows to inspect and process results
        as they arrive.
        """
        # result needs to be stored to avoid garbage collection
        self.result_queue: asyncio.Queue[tuple[int, R | PoisonPill]] = asyncio.Queue()
        self.exception = None
        self._main_task = asyncio.create_task(self.execute_task_group())
        orderer = BufferedOrdererTuple(self.indexer)
        result: R | PoisonPill = PoisonPill()

        async def pop_result() -> R | PoisonPill:
            idx, result = await self.result_queue.get()
            if not isinstance(result, PoisonPill) and self.return_in_order:
                tuple_idx = await self.indexer.get_tuple_index(idx)
                orderer.insert_in_order(tuple_idx, result)
            return result

        result = await pop_result()
        while not isinstance(result, PoisonPill):
            if self.return_in_order:
                for nxt in await orderer.retrieve_next_in_line():
                    yield nxt
            else:
                yield result
            result = await pop_result()
        await self._main_task
        if self.exception is not None:
            raise ApicadabriRecursiveExecutionError from self.exception


class SubtaskCreator(Protocol):
    """Protocol for providing subtask creator functions as callback."""

    def __call__(
        self,
        client: aiohttp.ClientSession,
        index: tuple[int, ...],
        instance_args: ApicadabriCallInstance,
        result: SyncedClientResponse,
    ) -> Iterable[ApicadabriCallInstance]:
        """Decide on additional HTTP tasks to take based on the response of the current one.

        Args:
            client: The aiohttp client to use for the request.
            index: The index of the instance in the list of instances. (WARNING: Currently unused)
            instance_args: The arguments that were used for the current instance.
            result: The result of the current instance.

        Returns:
            An iterable of new HTTP calls to add as subtasks.
        """
        ...


class SubtaskIndexer:
    """Manages the indexing of subtasks for recursive calls.

    Subtask indices are tuples that represent the hierarchical layers.
    The initial calls get indices (0,), (1,), and so on. The children of
    (0,) (i.e the tasks spawned when processing this task)
    become (0,0), (0,1), for children of (1,) we have (1,0), (1,1), etc.

    Since the rest of apicadabri assumes simple integers as indices, this
    class creates a mapping that generates integer indices an remembers
    the tuple-based indices they represent.
    """

    def __init__(
        self,
        *args: list[Any],
        remember_order: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """Creates a new subtask.

        Designed to play nice with any kind of subclass constructor under
        multiple inheritance.

        Args:
            args: Positional arguments (forwarded to other constructors).
            remember_order: If True, will keep a dictionary of order which
                costs O(n²) over the whole task.
            kwargs: Keyword arguments (forwarded rto other constructors).
        """
        super().__init__(*args, **kwargs)
        self.single_to_tuple: dict[int, tuple[int, ...]] = {}
        self.single_to_tuple_lock = asyncio.Lock()
        self.ordered_tuples = []
        self.remember_order = remember_order
        self.index_counter = 0

    async def get_tuple_index(self, index: int) -> tuple[int, ...]:
        """Retrieve the tuple associated with a specific integer index.

        Returns the tuple stored at the provided index. Access is ensured
        to be safe by acquiring an asyncio lock before execution.

        Args:
            index: The integer index corresponding to a tuple.

        Returns:
            The tuple associated with the index.

        Raises:
            KeyError: If the index does not exist in the internal dictionary.
        """
        async with self.single_to_tuple_lock:
            return self.single_to_tuple[index]

    async def add_tuple_index(self, index: tuple[int, ...]) -> int:
        """Assign a new integer index to a provided tuple index and register it.

        This method generates the next available integer index and maps it to
        the provided tuple.

        For `self.remember_order == False` this is in O(1), otherwise it's in
        O(n).

        Args:
            index: The tuple-based index to register.

        Returns:
            int: The assigned integer index that maps to the tuple index.
        """
        async with self.single_to_tuple_lock:
            int_index = self.index_counter
            self.single_to_tuple[int_index] = index
            self.index_counter += 1
            if self.remember_order:
                # NOTE: If this ever becomes a bottleneck, we can replace it with
                #       a SortedList if we accept sortedcontainers as dependency.
                insort_right(
                    self.ordered_tuples,
                    index,
                    key=lambda x: (-len(x), tuple(-i for i in x)),
                )
        return int_index

    async def next_index(self, n: int) -> tuple[int, ...] | None:
        """Return the next index that should be expected after n items have been retrieved.

        Args:
            n: The number of indices that have already been retrieved.

        Returns:
            The next index to expect.
        """
        if not self.remember_order:
            msg = (
                "SubtaskIndexer was created with self.remember_order = False, "
                "can't determine next index."
            )
            raise NotImplementedError(msg)
        async with self.single_to_tuple_lock:
            return self.ordered_tuples[-(n + 1)] if len(self.ordered_tuples) >= n + 1 else None


class BufferedOrdererTuple(BufferedOrdererBase[tuple[int, ...], R]):
    """BufferedOrderer that uses int tuples as indices.

    The indices are hierarchical, so (1,3,0) would mean the first child task
    of the 4th child task of the second initial task, for example.
    """

    def __init__(self, indexer: SubtaskIndexer, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Create a new orderer.

        Args:
            indexer: Indexer required for determining which index is next in line.
            args: Unused, just provided for compatibility with multiple inheritance.
            kwargs: Unused, just provided for compatibility with multiple inheritance.
        """
        super().__init__(*args, **kwargs)
        self.indexer = indexer

    def sorting_key(self, index: tuple[int, ...]) -> Ordered:
        """Sorts indices by length (shortest last) and reverse tuple order.

        The goal is to retrieve indices in breadth-first order.

        Args:
            index: The index.

        Returns:
            A key for sorting a buffer of indices.
        """
        return (-len(index), tuple(-i for i in index))

    async def next_expected_index(self, n: int) -> tuple[int, ...] | None:
        """Retrieves next expected index from Indexer.

        This is required, because we cannot easily determine how many children
        a particular element has. Is (1,3) the next index after (1,2) or is it (2,0)?

        The indexer can give us this information because it is aware of all
        currently existing indices before subtasks are even scheduled.

        Args:
            n: Number of results already retrieved from this buffer.

        Returns:
            The index that is next in line in breadth-first order.
        """
        return await self.indexer.next_index(n)


class ApicadabriRecursiveHTTPResponse(
    ApicadabriRecursiveResponse[ApicadabriCallInstance, SyncedClientResponse],
    ApicadabriBulkHTTPResponse,
):
    """Response class for recursive HTTP API calls."""

    def __init__(  # noqa: PLR0913
        self,
        apicadabri_args: ApicadabriCallArguments,
        method: Literal["POST", "GET"],
        max_active_calls: int = 20,
        retrier: AsyncRetrier | None = None,
        subtask_creator: SubtaskCreator = lambda client, index, instance_args, result: [],  # noqa: ARG005
        *,
        return_in_order: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the response object.

        Args:
            apicadabri_args: The arguments to pass to the API call.
            method: The HTTP method to use for the API call (GET or POST).
            max_active_calls: The maximum number of concurrent API calls to make.
            retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                    If None, a new instance will be created with default parameters.
            subtask_creator: Function that decides whether to spawn subtasks from an API call.
                    The response object will acquire an object-wide lock before calling this
                    function, so it should be safe to use shared state within this function.
            return_in_order: If True, results are returned in breadth-first order of their
                    creation in the recursive tree of subtasks. Set this to False if you
                    experience out of memory errors or long pauses and sudden bursts of
                    results in your pipeline. These effects can occur due to the buffering
                    of results required to return them in order.
            kwargs: Additional keyword arguments to pass to the aiohttp get/post method.
        """
        super().__init__(
            apicadabri_args=apicadabri_args,
            method=method,
            max_active_calls=max_active_calls,
            retrier=retrier,
            return_in_order=return_in_order,
            **kwargs,
        )
        self.create_subtasks = subtask_creator
        self.create_subtasks_lock = asyncio.Lock()

    async def call_api(
        self,
        client: aiohttp.ClientSession,
        index: int,
        instance_args: ApicadabriCallInstance,
    ) -> tuple[int, SyncedClientResponse]:
        """Call the API with the given arguments and return the response.

        This method will also schedule subtasks via `self.create_subtasks`
        if required.

        Args:
            client: The aiohttp client to use for the request.
            index: The index of the instance in the list of instances.
            instance_args: The arguments to pass to the API call.
        """
        idx, result = await super().call_api(client, index, instance_args)
        tuple_idx = await self.indexer.get_tuple_index(idx)
        async with self.create_subtasks_lock:
            subtasks = self.create_subtasks(client, tuple_idx, instance_args, result)
        for sub_idx, sub in enumerate(subtasks):
            sub_tuple_idx = (*tuple_idx, sub_idx)
            await self.schedule_subtask(client, sub, sub_tuple_idx)
        return (idx, result)


def recursive_get(  # noqa: PLR0913, PLR0917
    url: str | None = None,
    urls: Iterable[str] | None = None,
    params: dict[str, str] | None = None,
    param_sets: Iterable[dict[str, str]] | None = None,
    json: JSON | None = None,
    json_sets: Iterable[JSON] | None = None,
    headers: dict[str, str] | None = None,
    header_sets: Iterable[dict[str, str]] | None = None,
    mode: Literal["zip", "product"] = "zip",
    max_active_calls: int = 20,
    retrier: AsyncRetrier | None = None,
    subtask_creator: SubtaskCreator = lambda client, index, instance_args, result: [],  # noqa: ARG005
    *,
    return_in_order: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> ApicadabriRecursiveHTTPResponse:
    """Make a recursive GET request to the given API endpoint.

    For each of the typical HTTP call parameters, you can either pass a single value or an
    iterable of values.

    If more than one parameter is passed as an iterable, the `mode` parameter determines how the
    parameters are combined:

    - "zip": The parameters are combined in a way that each parameter is combined with the
        corresponding parameter from the other iterables. This means that the first element of each
        iterable is combined, then the second element, and so on. If one iterable is shorter than
        the others, it will be padded with None values.
    - "product": The parameters are combined in a way that each parameter is combined with all
        other parameters. This means that the first element of each iterable is combined with all
        other elements, then the second element, and so on. This will result in a Cartesian product
        of the parameters.

    Args:
        url: The URL of the API endpoint.
        urls: An iterable of URLs to make requests to.
        params: A dictionary of parameters to include in the request.
        param_sets: An iterable of dictionaries of parameters to include in the request.
        json: The JSON data to include in the request body.
        json_sets: An iterable of JSON data to include in the request body.
        headers: A dictionary of headers to include in the request.
        header_sets: An iterable of dictionaries of headers to include in the request.
        mode: The mode to use for combining the parameters. Either "zip" or "product".
        max_active_calls: The maximum number of concurrent API calls to make.
        retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                 If None, a new instance will be created with default parameters.
        subtask_creator: Function that decides whether to spawn subtasks from an API call.
        return_in_order: If True, results are returned in breadth-first order of their
                    creation in the recursive tree of subtasks. Set this to False if you
                    experience out of memory errors or long pauses and sudden bursts of
                    results in your pipeline. These effects can occur due to the buffering
                    of results required to return them in order.
        kwargs: Additional keyword arguments to pass to the aiohttp get method.

    Returns:
        A response object that can be used for further processing and retrieving the
        API responses.

    """
    if params is None and param_sets is None:
        params = {}
    if json is None and json_sets is None:
        json = {}
    if headers is None and header_sets is None:
        headers = {}
    return recursive_call(
        method="GET",
        apicadabri_args=ApicadabriCallArguments(
            url=url,
            urls=urls,
            params=params,
            param_sets=param_sets,
            json=json,
            json_sets=json_sets,
            headers=headers,
            header_sets=header_sets,
            mode=mode,
        ),
        max_active_calls=max_active_calls,
        retrier=retrier,
        subtask_creator=subtask_creator,
        return_in_order=return_in_order,
        **kwargs,
    )


def recursive_post(  # noqa: PLR0913, PLR0917
    url: str | None = None,
    urls: Iterable[str] | None = None,
    params: dict[str, str] | None = None,
    param_sets: Iterable[dict[str, str]] | None = None,
    json: JSON | None = None,
    json_sets: Iterable[JSON] | None = None,
    headers: dict[str, str] | None = None,
    header_sets: Iterable[dict[str, str]] | None = None,
    mode: Literal["zip", "product"] = "zip",
    max_active_calls: int = 20,
    retrier: AsyncRetrier | None = None,
    subtask_creator: SubtaskCreator = lambda client, index, instance_args, result: [],  # noqa: ARG005
    *,
    return_in_order: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> ApicadabriRecursiveHTTPResponse:
    """Make a recursive POST request to the given API endpoint.

    For each of the typical HTTP call parameters, you can either pass a single value or an
    iterable of values.

    If more than one parameter is passed as an iterable, the `mode` parameter determines how the
    parameters are combined:

    - "zip": The parameters are combined in a way that each parameter is combined with the
        corresponding parameter from the other iterables. This means that the first element of each
        iterable is combined, then the second element, and so on. If one iterable is shorter than
        the others, it will be padded with None values.
    - "product": The parameters are combined in a way that each parameter is combined with all
        other parameters. This means that the first element of each iterable is combined with all
        other elements, then the second element, and so on. This will result in a Cartesian product
        of the parameters.

    Args:
        url: The URL of the API endpoint.
        urls: An iterable of URLs to make requests to.
        params: A dictionary of parameters to include in the request.
        param_sets: An iterable of dictionaries of parameters to include in the request.
        json: The JSON data to include in the request body.
        json_sets: An iterable of JSON data to include in the request body.
        headers: A dictionary of headers to include in the request.
        header_sets: An iterable of dictionaries of headers to include in the request.
        mode: The mode to use for combining the parameters. Either "zip" or "product".
        max_active_calls: The maximum number of concurrent API calls to
            make.
        retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                 If None, a new instance will be created with default parameters.
        subtask_creator: Function that decides whether to spawn subtasks from an API call.
        return_in_order: If True, results are returned in breadth-first order of their
                    creation in the recursive tree of subtasks. Set this to False if you
                    experience out of memory errors or long pauses and sudden bursts of
                    results in your pipeline. These effects can occur due to the buffering
                    of results required to return them in order.
        kwargs: Additional keyword arguments to pass to the aiohttp post method.

    Returns:
        A response object that can be used for further processing and retrieving the
        API responses.

    """
    if params is None and param_sets is None:
        params = {}
    if json is None and json_sets is None:
        json = {}
    if headers is None and header_sets is None:
        headers = {}
    return recursive_call(
        method="POST",
        apicadabri_args=ApicadabriCallArguments(
            url=url,
            urls=urls,
            params=params,
            param_sets=param_sets,
            json=json,
            json_sets=json_sets,
            headers=headers,
            header_sets=header_sets,
            mode=mode,
        ),
        max_active_calls=max_active_calls,
        retrier=retrier,
        subtask_creator=subtask_creator,
        return_in_order=return_in_order,
        **kwargs,
    )


def recursive_call(  # noqa: PLR0913
    method: Literal["POST", "GET"],
    apicadabri_args: ApicadabriCallArguments,
    max_active_calls: int = 20,
    retrier: AsyncRetrier | None = None,
    subtask_creator: SubtaskCreator = lambda client, index, instance_args, result: [],  # noqa: ARG005
    *,
    return_in_order: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> ApicadabriRecursiveHTTPResponse:
    """Make a bulk API call to the given API endpoint.

    This is a convenience function that wraps the `ApicadabriBulkHTTPResponse` class.

    Args:
        method: The HTTP method to use for the API call (GET or POST).
        apicadabri_args: The arguments to pass to the API call.
        max_active_calls: The maximum number of concurrent API calls to make.
        retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                 If None, a new instance will be created with default parameters.
        subtask_creator: Function that decides whether to spawn subtasks from an API call.
        return_in_order: If True, results are returned in breadth-first order of their
                    creation in the recursive tree of subtasks. Set this to False if you
                    experience out of memory errors or long pauses and sudden bursts of
                    results in your pipeline. These effects can occur due to the buffering
                    of results required to return them in order.
        kwargs: Additional keyword arguments to pass to the aiohttp get/post method.

    Returns:
        A response object that can be used for further processing and retrieving the
        API responses.

    """
    return ApicadabriRecursiveHTTPResponse(
        apicadabri_args=apicadabri_args,
        method=method,
        max_active_calls=max_active_calls,
        retrier=retrier,
        subtask_creator=subtask_creator,
        return_in_order=return_in_order,
        **kwargs,
    )

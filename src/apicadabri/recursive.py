"""Support for recursive calls where number of calls is not known beforehand."""

import asyncio
from abc import ABC
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


class ApicadabriRecursiveResponse(ApicadabriBulkResponse[A, R], Generic[A, R], ABC):
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

    async def execute_task_group(self) -> None:
        """Creates a task group to start the initial instances.

        This task group is saved as an instance variable allowing the instances
        to generate new tasks and schedule them within the same task group.

        This method only returns after all the tasks in the task group have actually finished.
        """
        async with aiohttp.ClientSession() as client, asyncio.TaskGroup() as tg:
            self.task_group = tg
            for inst in self.instances():
                await self.schedule_subtask(client, inst)

        await self.result_queue.put(PoisonPill())

    async def schedule_subtask(self, client: aiohttp.ClientSession, instance_args: A) -> None:
        """Schedules a subtask in the internal task group.

        The task will be called via `self.call_api` and benefits from retry
        and rate limiting functionality of Apicadabri.

        The result will be put into an internal result queue, which is
        processed automatically.
        """
        self.task_group.create_task(self.execute_task(client, instance_args))

    async def execute_task(self, client: aiohttp.ClientSession, instance_args: A) -> None:
        """Performs individual API calls and puts result into result queue.

        Args:
            client: A shared client client to use for efficient HTTP calls.
            instance_args: The arguments to the individual API call.
        """
        _, res = await self.call_with_semaphore(client, 0, instance_args)
        await self.result_queue.put(res)

    async def call_all(self) -> AsyncGenerator[R, None]:
        """Return an iterator that yields the results of the API calls.

        This uses a semaphore to limit the number of concurrent API calls.

        It returns results in the same order as the input arguments from
        `instances`. However, it also allows to inspect and process results
        as they arrive.
        """
        # result needs to be stored to avoid garbage collection
        self.result_queue: asyncio.Queue[R | PoisonPill] = asyncio.Queue()
        self._main_task = asyncio.create_task(self.execute_task_group())
        result = await self.result_queue.get()
        while not isinstance(result, PoisonPill):
            # TODO maybe we also want to do some re-ordering here in the future?
            yield result
            result = await self.result_queue.get()


class SubtaskCreator(Protocol):
    """Function that decides whether subtasks should be created."""

    def __call__(
        self,
        client: aiohttp.ClientSession,
        index: int,
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


class ApicadabriRecursiveHTTPResponse(
    ApicadabriRecursiveResponse[ApicadabriCallInstance, SyncedClientResponse],
    ApicadabriBulkHTTPResponse,
):
    """Response class for recursive HTTP API calls."""

    def __init__(
        self,
        apicadabri_args: ApicadabriCallArguments,
        method: Literal["POST", "GET"],
        max_active_calls: int = 20,
        retrier: AsyncRetrier | None = None,
        subtask_creator: SubtaskCreator = lambda client, index, instance_args, result: [],  # noqa: ARG005
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
            kwargs: Additional keyword arguments to pass to the aiohttp get/post method.

        """
        super().__init__(apicadabri_args, method, max_active_calls, retrier, **kwargs)
        self.create_subtasks = subtask_creator

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
        for sub in self.create_subtasks(client, idx, instance_args, result):
            await self.schedule_subtask(client, sub)
        return (idx, result)


def recursive_get(  # noqa: PLR0913
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
        **kwargs,
    )


def recursive_post(  # noqa: PLR0913
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
        **kwargs,
    )


def recursive_call(
    method: Literal["POST", "GET"],
    apicadabri_args: ApicadabriCallArguments,
    max_active_calls: int = 20,
    retrier: AsyncRetrier | None = None,
    subtask_creator: SubtaskCreator = lambda client, index, instance_args, result: [],  # noqa: ARG005
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
        **kwargs,
    )

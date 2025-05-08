"""Main module of apicadabri, containing all top-level members."""

import asyncio
import json
import traceback
from abc import ABC, abstractmethod
from bisect import insort_right
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator, Iterable
from http.cookies import SimpleCookie
from itertools import product, repeat
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeAlias, TypeVar, overload

import aiohttp
import humanize
import yarl
from aiohttp.client_reqrep import ContentDisposition
from aiohttp.connector import Connection
from aiohttp.typedefs import RawHeaders
from multidict import CIMultiDictProxy, MultiDictProxy
from pydantic import BaseModel, Field, ValidationError, model_validator

# source: https://stackoverflow.com/a/76646986
# NOTE: we could use "JSON" instead of Any here to define a recursive type
# however, this won't work with pydantic, so we settle for a shallow representation here
JSON: TypeAlias = dict[str, Any] | list[Any] | str | int | float | bool | None

A = TypeVar("A")


def exception_to_json(e: Exception) -> dict[str, str]:
    """Return a JSON representation of an arbirary exception.

    Keys:
        - "type": The Name of the exception class.
        - "message": The message of the exception.
        - "traceback": The full traceback as string.

    Args:
        e: The exception to capture.

    Return:
        A JSON-serializable dictionary, containing the exception type, message,
        and trackeback.

    """
    return {
        "type": e.__class__.__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }


class ApicadabriCallInstance(BaseModel):
    """Arguments for a single instance of an HTTP API call."""

    url: str
    params: dict[str, str]
    # NOTE we need to use an alias to avoid shadowing the BaseModel field
    json_data: JSON = Field(alias="json")
    headers: dict[str, str]


class ApicadabriCallArguments(BaseModel):
    """A set of arguments to a web API that can be used as an iterator.

    For each of the arguments, you can select whether you want to provide a
    single value or an iterable of multiple values.

    If only one of the arguments is given in list form, this object will
    iterate over that list and provide `ApicadabriCallInstance` with the
    respective values. For more than one argument in list form, the behavior
    is defined by the `mode` parameter:

    - "zip": Assume that all lists have the same length and combine them
        with a call to `zip`, so that the first instance has the values from
        the first element of each list and so on.
    - "product": Iterate over all combinations of arguments. If you give 3 URLs
        and 4 parameter sets, you will end up with 12 calls in total.
    """

    url: str | None = None
    urls: Iterable[str] | None = None
    params: dict[str, str] | None = None
    param_sets: Iterable[dict[str, str]] | None = None
    # NOTE we need to use an alias to avoid shadowing the BaseModel field
    json_data: JSON | None = Field(alias="json", default=None)
    json_sets: Iterable[JSON] | None = None
    headers: dict[str, str] | None = None
    header_sets: Iterable[dict[str, str]] | None = None
    mode: Literal["zip", "product"] = "zip"

    @model_validator(mode="after")
    def validate_not_both_none(self) -> Self:
        """Ensure that either the single or the multi version of a parameter is not None.

        For `params`, `json`, and `headers` the single version is set to an empty dict
        if both are none. For `url`, a validation error is raised.
        """
        if self.url is None and self.urls is None:
            msg = "One of `url` or `urls` must be provided."
            raise ValidationError(msg)
        if self.params is None and self.param_sets is None:
            self.params = {}
        if self.json_data is None and self.json_sets is None:
            self.json_data = {}
        if self.headers is None and self.header_sets is None:
            self.headers = {}
        return self

    @model_validator(mode="after")
    def validate_only_one_provided(self) -> Self:
        """Validate that either the single or the multi version of a parameter remains None."""
        if self.url is not None and self.urls is not None:
            msg = "You cannot specify both `url` and `urls`."
            raise ValidationError(msg)
        if self.params is not None and self.param_sets is not None:
            msg = "You cannot specify both `param` and `param_sets`."
            raise ValidationError(msg)
        if self.json_data is not None and self.json_sets is not None:
            msg = "You cannot specify both `json` and `json_sets`."
            raise ValidationError(msg)
        if self.headers is not None and self.header_sets is not None:
            msg = "You cannot specify both `header` and `header_sets`."
            raise ValidationError(msg)
        return self

    def __iter__(self) -> Generator["ApicadabriCallInstance", None, None]:
        """Iterate over individual call argument instances.

        If multiple parameters are given as a list, the behavior depends
        on `self.mode`:

        - "zip": Assume that all lists have the same length and combine them
            with a call to `zip`, so that the first instance has the values from
            the first element of each list and so on.
        - "product": Iterate over all combinations of arguments. If you give 3 URLs
            and 4 parameter sets, you will end up with 12 calls in total.

        Yields:
            The argument sets to call the API with.

        """
        iterables = (
            self.url_iterable,
            self.params_iterable,
            self.json_iterable,
            self.headers_iterable,
        )
        if self.mode == "zip":
            combined = zip(*iterables, strict=False)
        elif self.mode == "product":
            combined = product(*iterables)
        else:
            msg = f"Mode {self.mode} not implemented."
            raise NotImplementedError(msg)
        return iter(
            ApicadabriCallInstance(url=u, params=p, json=j, headers=h) for u, p, j, h in combined
        )

    def any_iterable(
        self,
        single_val: A | None,
        multi_val: Iterable[A] | None,
    ) -> Iterable[A]:
        """Turn any set of single and multi argument into an iterable.

        Args:
            single_val: The single value version of the argument.
            multi_val: The multi version of the argument.

        Returns:
            An iterable that iterates over all (possibly just one) argument values.

        """
        if single_val is None:
            if multi_val is None:
                msg = "Single and multi val cannot both be null."
                raise ValueError(msg)
            return multi_val
        if self.mode == "zip":
            return repeat(single_val)
        if self.mode == "product":
            return [single_val]
        msg = f"Unrecognized mode {self.mode}"
        raise ValueError(msg)

    @property
    def url_iterable(self) -> Iterable[str]:
        """Iterable version of `url` parameter."""
        return self.any_iterable(self.url, self.urls)

    @property
    def params_iterable(self) -> Iterable[dict[str, str]]:
        """Iterable version of the `params` parameter."""
        return self.any_iterable(self.params, self.param_sets)

    @property
    def json_iterable(self) -> Iterable[JSON]:
        """Iterable version of the `json` parameter."""
        return self.any_iterable(self.json_data, self.json_sets)

    @property
    def headers_iterable(self) -> Iterable[dict[str, str]]:
        """Iterable version of the `headers` parameter."""
        return self.any_iterable(self.headers, self.header_sets)


R = TypeVar("R")
S = TypeVar("S")


class ApicadabriResponse(Generic[R]):
    """Response object that is used for constructing lazy evaluation pipelines.

    The pipeline will only actually be executed once you call one of the
    methods using `self.reduce` to collect the results.

    Args:
        R: The return type that is obtained when evaluating this response.

    """

    @overload
    def map(
        self,
        func: Callable[[R], S],
        on_error: Literal["raise"] | Callable[[R, Exception], S] = "raise",
    ) -> "ApicadabriResponse[S]": ...

    @overload
    def map(
        self,
        func: Callable[[R], S],
        on_error: Literal["return"],
    ) -> "ApicadabriResponse[S | ApicadabriErrorResponse[R]]": ...

    def map(
        self,
        func: Callable[[R], S],
        on_error: Literal["raise", "return"] | Callable[[R, Exception], S] = "raise",
    ) -> "ApicadabriResponse[S] | ApicadabriResponse[S | ApicadabriErrorResponse[R]]":
        """Apply a function to the response.

        Args:
            func: The function to apply to the response value.
            on_error: Whether to just raise errors ("raise"), return an object encapsulating the
                      exception ("return") or use a function to supply a fallback result.

        Returns:
            A response object of the return type of the map function. If `on_error` is
            "return", the response type can also be a special error object.

        """
        if on_error == "raise":
            return ApicadabriMapResponse(self, func)
        if on_error == "return":
            return ApicadabriMaybeMapResponse(self, func)
        return ApicadabriSafeMapResponse(self, func, on_error)

    @abstractmethod
    def call_all(self) -> AsyncGenerator[R, None]:
        """Return an iterator that yields the results of the API calls."""
        ...

    def to_jsonl(self, filename: Path | str, error_value: str | None = None) -> None:
        """Write results directly to a JSONL file.

        As each result is directly appended to the file, this method can be used
        to process results that are too large to fit into memory and to ensure
        that results persist on disk even if the process crashes at some point.

        Args:
            filename: Name of the file to write to.
            error_value: Value to write in case a response object cannot be
                         converted to JSON.

        """
        if error_value is None:
            error_value = "{{}}\n"
        filename_path = Path(filename)
        with filename_path.open("w", encoding="utf-8") as f:
            asyncio.run(
                self.reduce(
                    lambda _, r: f.write(json.dumps(r) + "\n"),
                    start=0,
                    on_error=lambda _, r, e: f.write(
                        error_value.format(result=r, exception=e),
                    ),
                ),
            )

    def to_list(self) -> list[R]:
        """Return a list of all responses."""
        start: list[R] = []

        def appender(lst: list[R], element: R) -> list[R]:
            """Accumulator function that appends elements to a list.

            Essentially, this is a faster version of `lst + [element]`.
            """
            lst.append(element)
            return lst

        return asyncio.run(self.reduce(appender, start=start))

    # TODO should this be async, or should we already use asyncio.run here?
    async def reduce(
        self,
        accumulator: Callable[[A, R], A],
        start: A,
        on_error: Literal["raise"] | Callable[[A, R, Exception], A] = "raise",
    ) -> A:
        """Reduce the pipeline to a single object that collects all results.

        Args:
            accumulator: Accumulator function that takes an intermediary result
                         and adds one response from the pipeline to it.
            start: Initial result object to start with (e.g. empty list).
            on_error: Whether to just raise errors ("raise") or use a function
                      to supply a fallback result.

        """
        accumulated = start
        async for res in self.call_all():
            try:
                accumulated = accumulator(accumulated, res)
            except Exception as e:
                if on_error == "raise":
                    raise e
                accumulated = on_error(accumulated, res, e)
        return accumulated


class ApicadabriMapResponse(ApicadabriResponse[S], Generic[R, S]):
    def __init__(self, base: ApicadabriResponse[R], func: Callable[[R], S]):
        self.func = func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S, None]:
        """Return an iterator that yields the results of the API calls."""
        async for res in self.base.call_all():
            # if this raises an exception, the pipeline will just break
            mapped = self.func(res)
            yield mapped


class ApicadabriSafeMapResponse(ApicadabriResponse[S], Generic[R, S]):
    def __init__(
        self,
        base: ApicadabriResponse[R],
        map_func: Callable[[R], S],
        error_func: Callable[[R, Exception], S],
    ):
        self.map_func = map_func
        self.error_func = error_func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S, None]:
        """Return an iterator that yields the results of the API calls."""
        async for res in self.base.call_all():
            try:
                mapped = self.map_func(res)
                yield mapped
            except Exception as e:  # noqa: BLE001
                yield self.error_func(res, e)


# TODO: Should this really be a pydantic class?
class ApicadabriErrorResponse(BaseModel, Generic[R]):
    type: str
    message: str
    traceback: str
    triggering_input: R

    # need to allow arbitrary types because triggering_input may not be a BaseModel
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_exception(
        cls,
        e: Exception,
        triggering_input: R,
    ) -> "ApicadabriErrorResponse[R]":
        return ApicadabriErrorResponse(
            type=e.__class__.__name__,
            message=str(e),
            traceback=traceback.format_exc(),
            triggering_input=triggering_input,
        )


class ApicadabriMaybeMapResponse(
    ApicadabriResponse[S | ApicadabriErrorResponse[R]],
    Generic[R, S],
):
    def __init__(self, base: ApicadabriResponse[R], func: Callable[[R], S]):
        self.func = func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S | ApicadabriErrorResponse, None]:
        """Return an iterator that yields the results of the API calls."""
        async for res in self.base.call_all():
            # if this raises an exception, the pipeline will just break
            try:
                mapped = self.func(res)
                yield mapped
            except Exception as e:
                yield ApicadabriErrorResponse.from_exception(e, res)


class SyncedClientResponse:
    def __init__(
        self,
        base: aiohttp.ClientResponse,
        body: bytes,
        *,
        is_exception: bool = False,
    ):
        self.base = base
        self.body = body
        self.is_exception = is_exception

    @property
    def version(self) -> aiohttp.HttpVersion | None:
        return self.base.version

    @property
    def status(self) -> int:
        return self.base.status

    @property
    def reason(self) -> str | None:
        return self.base.reason

    @property
    def ok(self) -> bool:
        return self.base.ok

    @property
    def method(self) -> str:
        return self.base.method

    @property
    def url(self) -> yarl.URL:
        return self.base.url

    @property
    def real_url(self) -> yarl.URL:
        return self.base.real_url

    @property
    def connection(self) -> Connection | None:
        return self.base.connection

    @property
    def cookies(self) -> SimpleCookie:
        return self.base.cookies

    @property
    def headers(self) -> CIMultiDictProxy[str]:
        return self.base.headers

    @property
    def raw_headers(self) -> RawHeaders:
        return self.base.raw_headers

    @property
    def links(self) -> MultiDictProxy[MultiDictProxy[str | yarl.URL]]:
        return self.base.links

    @property
    def content_type(self) -> str:
        return self.base.content_type

    @property
    def charset(self) -> str | None:
        return self.base.charset

    @property
    def content_disposition(self) -> ContentDisposition | None:
        return self.base.content_disposition

    @property
    def history(self) -> tuple[aiohttp.ClientResponse, ...]:
        return self.base.history

    def raise_for_status(self) -> None:
        self.base.raise_for_status()

    @property
    def request_info(self) -> aiohttp.RequestInfo:
        return self.base.request_info

    def get_encoding(self) -> str:
        return self.base.get_encoding()

    def text(self, encoding=None) -> str:
        return self.body.decode(encoding or self.get_encoding())

    def json(self) -> Any:
        return json.loads(self.text())

    def read(self) -> bytes:
        return self.body


class ApicadabriRetryError(Exception):
    def __init__(self, i: int):
        super().__init__(f"{humanize.ordinal(i + 1)} retry failed.")


class ApicadabriMaxRetryError(Exception):
    def __init__(self, max_retries: int):
        super().__init__(f"Call failed after {max_retries} retries.")


class AsyncRetrier:
    def __init__(
        self,
        max_retries: int = 10,
        initial_sleep_s: float = 0.01,
        sleep_multiplier: float = 2,
        max_sleep_s: float = 60 * 15,
        should_retry: Callable[[Exception], bool] | None = None,
    ):
        self.max_retries = max_retries
        self.initial_sleep_s = initial_sleep_s
        self.sleep_multiplier = sleep_multiplier
        self.max_sleep_s = max_sleep_s
        self.should_retry = should_retry if should_retry is not None else lambda _: True

    async def retries(self) -> AsyncGenerator[tuple[int, float], None]:
        sleep_s = self.initial_sleep_s
        for i in range(self.max_retries):
            yield i, sleep_s
            await asyncio.sleep(sleep_s)
            sleep_s *= self.sleep_multiplier
            sleep_s = min(self.max_sleep_s, sleep_s)

    async def retry(
        self,
        callable_to_retry: Callable[[], Coroutine[None, None, R]],
    ) -> R:
        last_exception = None
        async for i, _ in self.retries():
            try:
                return await callable_to_retry()
            except Exception as e:
                last_exception = e
                if not self.should_retry(e):
                    raise ApicadabriRetryError(i) from e
        if last_exception is not None:
            raise ApicadabriMaxRetryError(self.max_retries) from last_exception
        msg = "Max retries reached, but no exception stored. This should never happen!"
        raise RuntimeError(msg)


class ApicadabriBulkResponse(ApicadabriResponse[R], Generic[A, R], ABC):
    def __init__(
        self,
        *args,
        max_active_calls: int = 20,
        retrier: AsyncRetrier | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.semaphore = asyncio.Semaphore(max_active_calls)
        self.retrier = AsyncRetrier() if retrier is None else retrier

    async def call_all(self) -> AsyncGenerator[R, None]:
        next_index = 0
        buffer: list[tuple[int, R]] = []
        async with aiohttp.ClientSession() as client:
            for res in asyncio.as_completed(
                [
                    self.call_with_semaphore(client, i, instance)
                    for i, instance in enumerate(self.instances())
                ],
            ):
                current_index, current_res = await res
                insort_right(buffer, (current_index, current_res), key=lambda x: -x[0])
                while current_index == next_index:
                    yield buffer.pop()[1]
                    current_index = buffer[-1][0] if len(buffer) > 0 else -1
                    next_index += 1

    async def call_with_semaphore(
        self,
        client: aiohttp.ClientSession,
        index: int,
        instance_args: A,
    ) -> tuple[int, R]:
        async def call_api_for_retry() -> tuple[int, R]:
            return await self.call_api(client, index, instance_args)

        async with self.semaphore:
            return await self.retrier.retry(call_api_for_retry)

    @abstractmethod
    async def call_api(
        self,
        client: aiohttp.ClientSession,
        index: int,
        instance_args: A,
    ) -> tuple[int, R]:
        """Call the API with the given arguments and return the response.

        The arguments are assumed to be generated by the `instances` method.

        Args:
            client: The aiohttp client session to use for the request.
            index: The index of the instance in the list of instances.
            instance_args: The arguments to pass to the API call.

        Returns:
            A tuple containing the index of the instance (as given in the
            arguments) and the response from the API call.

        """
        ...

    @abstractmethod
    def instances(self) -> Iterable[A]:
        """Generate instances of the API call arguments."""
        ...


class ApicadabriBulkHTTPResponse(
    ApicadabriBulkResponse[ApicadabriCallInstance, SyncedClientResponse],
):
    def __init__(
        self,
        apicadabri_args: ApicadabriCallArguments,
        method: Literal["POST", "GET"],
        max_active_calls: int = 20,
        retrier: AsyncRetrier | None = None,
    ):
        super().__init__(max_active_calls=max_active_calls, retrier=retrier)
        self.apicadabri_args = apicadabri_args
        self.method = method

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        index: int,
        args: ApicadabriCallInstance,
    ) -> tuple[int, SyncedClientResponse]:
        aiohttp_method = session.post if self.method == "POST" else session.get
        async with aiohttp_method(**args.model_dump(by_alias=True)) as resp, resp:
            try:
                return (index, SyncedClientResponse(resp, await resp.read()))
            except Exception as e:  # noqa: BLE001
                return (
                    index,
                    SyncedClientResponse(
                        resp,
                        json.dumps(
                            {
                                "exceptions": [exception_to_json(e)],
                            },
                        ).encode(
                            resp.get_encoding(),
                        ),
                        is_exception=True,
                    ),
                )

    def instances(self) -> Iterable[ApicadabriCallInstance]:
        """Generate instances of the API call arguments."""
        return self.apicadabri_args

    @overload
    def json(
        self,
        on_error: Literal["raise"] | Callable[[SyncedClientResponse, Exception], Any] = "raise",
    ) -> ApicadabriResponse[Any]: ...

    @overload
    def json(
        self,
        on_error: Literal["return"],
    ) -> ApicadabriResponse[Any | ApicadabriErrorResponse[Any]]: ...

    def json(
        self,
        on_error: Literal["raise", "return"]
        | Callable[[SyncedClientResponse, Exception], Any] = "raise",
    ) -> (
        ApicadabriResponse[Any]
        | ApicadabriResponse[Any | ApicadabriErrorResponse[SyncedClientResponse]]
    ):
        return self.map(SyncedClientResponse.json, on_error=on_error)

    @overload
    def text(
        self,
        on_error: Literal["raise"] | Callable[[SyncedClientResponse, Exception], str] = "raise",
    ) -> ApicadabriResponse[str]: ...

    @overload
    def text(
        self,
        on_error: Literal["return"],
    ) -> ApicadabriResponse[str | ApicadabriErrorResponse[SyncedClientResponse]]: ...

    def text(
        self,
        on_error: Literal["raise", "return"]
        | Callable[[SyncedClientResponse, Exception], str] = "raise",
    ) -> (
        ApicadabriResponse[str]
        | ApicadabriResponse[str | ApicadabriErrorResponse[SyncedClientResponse]]
    ):
        return self.map(SyncedClientResponse.text, on_error=on_error)

    def read(self) -> ApicadabriResponse[bytes]:
        # SyncedClientResponse.read just returns an internal variable
        # => there is no way this could raise an exception under normal circumstances
        # => if it does, it is an implementation error and we should just raise it normally
        return self.map(SyncedClientResponse.read, on_error="raise")


def bulk_get(
    url: str | None = None,
    urls: Iterable[str] | None = None,
    params: dict[str, str] | None = None,
    param_sets: Iterable[dict[str, str]] | None = None,
    json: JSON | None = None,
    json_sets: Iterable[JSON] | None = None,
    headers: dict[str, str] | None = None,
    header_sets: Iterable[dict[str, str]] | None = None,
    max_active_calls: int = 20,
    retrier: AsyncRetrier | None = None,
    **kwargs: dict[str, Any],
) -> ApicadabriBulkHTTPResponse:
    if params is None and param_sets is None:
        params = {}
    if json is None and json_sets is None:
        json = {}
    if headers is None and header_sets is None:
        headers = {}
    return bulk_call(
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
            mode="zip",
        ),
        max_active_calls=max_active_calls,
        retrier=retrier,
        **kwargs,
    )


def bulk_call(
    method: Literal["POST", "GET"],
    apicadabri_args: ApicadabriCallArguments,
    max_active_calls: int = 20,
    retrier: AsyncRetrier | None = None,
    # response_type: Literal["bytes", "str", "json", "raw"] = "json",
    **kwargs,
) -> ApicadabriBulkHTTPResponse:
    # TODO allow to pass extra args to aiohttp.ClientSession.get/post
    return ApicadabriBulkHTTPResponse(
        apicadabri_args=apicadabri_args,
        method=method,
        max_active_calls=max_active_calls,
        retrier=retrier,
    )

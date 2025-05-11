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
from typing import Any, Concatenate, Generic, Literal, ParamSpec, Self, TypeAlias, TypeVar, overload

import aiohttp
import humanize
import yarl
from aiohttp.client_reqrep import ContentDisposition
from aiohttp.connector import Connection
from aiohttp.typedefs import RawHeaders
from multidict import CIMultiDictProxy, MultiDictProxy
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class ApicadabriReduceError(Exception, Generic[A, R]):
    """Exception that is raised when a reduce operation fails.

    Args:
        A: The type of the accumulated result.
        R: The type of the result that caused the error.

    """

    def __init__(self, accumulated: A, res: R) -> None:
        """Initialize the exception with the given message.

        Args:
            accumulated: The accumulated result before the error occurred.
            res: The result that caused the error.

        """
        super().__init__(f"Adding {res} to {accumulated} failed.")


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

    # TODO: should this be async, or should we already use asyncio.run here?
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

        Returns:
            The final result object after all responses have been processed.

        Raises:
            ApicadabriReduceError: If the accumulator function raises an exception.

        """
        accumulated = start
        async for res in self.call_all():
            try:
                accumulated = accumulator(accumulated, res)
            except Exception as e:
                if on_error == "raise":
                    raise ApicadabriReduceError(accumulated, res) from e
                accumulated = on_error(accumulated, res, e)
        return accumulated


class ApicadabriMapResponse(ApicadabriResponse[S], Generic[R, S]):
    """Response object that applies a mapping function to the results of the pipeline.

    This version will not catch any exceptions that are raised by the mapping
    function.

    Args:
        R: The return type that this response is based on.
        S: The return type of the mapping function.

    """

    def __init__(self, base: ApicadabriResponse[R], func: Callable[[R], S]) -> None:
        """Initialize the response object.

        Args:
            base: The base response object to map over.
            func: The mapping function to apply to the results of the pipeline.

        """
        self.func = func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S, None]:
        """Return an iterator that yields the results of the map calls."""
        async for res in self.base.call_all():
            # if this raises an exception, the pipeline will just break
            mapped = self.func(res)
            yield mapped


class ApicadabriSafeMapResponse(ApicadabriResponse[S], Generic[R, S]):
    """Response object that applies a mapping function to the results of the pipeline.

    This version calls a function in case of an exception to get a fallback result.

    Args:
        R: The return type that this response is based on.
        S: The return type of the mapping function.

    """

    def __init__(
        self,
        base: ApicadabriResponse[R],
        map_func: Callable[[R], S],
        error_func: Callable[[R, Exception], S],
    ) -> None:
        """Initialize the response object.

        Args:
            base: The base response object to map over.
            map_func: The mapping function to apply to the results of the pipeline.
            error_func: The function to call in case of an error to get a fallback result.

        """
        self.map_func = map_func
        self.error_func = error_func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S, None]:
        """Return an iterator that yields the results of the map calls."""
        async for res in self.base.call_all():
            try:
                mapped = self.map_func(res)
                yield mapped
            except Exception as e:  # noqa: BLE001
                yield self.error_func(res, e)


# TODO: Should this really be a pydantic class?
class ApicadabriErrorResponse(BaseModel, Generic[R]):
    """Response object that encapsulates an exception.

    Args:
        R: The type of the triggering input that caused the exception.

    """

    type: str
    """Exception class name."""
    message: str
    """Exception message."""
    traceback: str
    """Full traceback as string."""
    triggering_input: R
    """The input that caused the exception."""

    # need to allow arbitrary types because triggering_input may not be a Pydantic model
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_exception(
        cls,
        e: Exception,
        triggering_input: R,
    ) -> "ApicadabriErrorResponse[R]":
        """Create an error response from an exception.

        Args:
            e: The exception to capture.
            triggering_input: The input that caused the exception.

        Returns:
            An error response object that contains the exception type, message,
            and traceback.

        """
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
    """Response object that applies a mapping function to the results of the pipeline.

    This version will catch any exceptions that are raised by the mapping function
    and return an error response object instead.

    Args:
        R: The return type that this response is based on.
        S: The return type of the mapping function.

    """

    def __init__(self, base: ApicadabriResponse[R], func: Callable[[R], S]) -> None:
        """Initialize the response object.

        Args:
            base: The base response object to map over.
            func: The mapping function to apply to the results of the pipeline.

        """
        self.func = func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S | ApicadabriErrorResponse, None]:
        """Return an iterator that yields the results of the mapping calls or an error object."""
        async for res in self.base.call_all():
            # if this raises an exception, the pipeline will just break
            try:
                mapped = self.func(res)
                yield mapped
            except Exception as e:  # noqa: BLE001
                yield ApicadabriErrorResponse.from_exception(e, res)

P = ParamSpec("P")
C1 = TypeVar("C1")
C2 = TypeVar("C2")


# source: https://stackoverflow.com/a/68901244
def copy_doc(
    reference: Callable[Concatenate[C1, P], R],
) -> Callable[[Callable[Concatenate[C2, P], R]], Callable[Concatenate[C2, P], R]]:
    """Create a decorator that copies the doc string of the given function to another.

    Args:
        reference: The function to copy the doc string from.

    Returns:
        A decorator that copies the doc string of the given function to another.

    """

    def wrapped(func: Callable[Concatenate[C2, P], R]) -> Callable[Concatenate[C2, P], R]:
        func.__doc__ = reference.__doc__
        return func

    return wrapped


class SyncedClientResponse:
    """Response object that wraps an aiohttp response removing the need for typing `async`.

    Essentially this just delegates all instance variables and methods to an
    `aiohttp.ClientResponse` object. The only exception is the `read`, `text` and `json` methods,
    which are snychronous instead of asynchronous.

    This is achieved by calling `await` before initializing this object and passing the result
    as part of the constructor.
    """

    def __init__(
        self,
        base: aiohttp.ClientResponse,
        body: bytes,
        *,
        is_exception: bool = False,
    ) -> None:
        """Initialize the synced response from an aiotthp response object.

        Args:
            base: The aiohttp response object to wrap.
            body: The body of the response or a JSON-encoded exception.
            is_exception: Whether the response threw an exception when trying to call `base.read()`.
                          In this case, `body` is assumed to contain a JSON-encoded version of the
                          exception.

        """
        self.base = base
        self.body = body
        self.is_exception = is_exception

    @property
    def version(self) -> aiohttp.HttpVersion | None:
        """Response’s version, HttpVersion instance."""  # noqa: RUF002
        # NOTE: docstring copied from aiohttp
        return self.base.version

    @property
    def status(self) -> int:
        """HTTP status code of response, e.g. 200."""
        # NOTE: docstring copied from aiohttp
        return self.base.status

    @property
    def reason(self) -> str | None:
        """HTTP status reason of response, e.g. "OK"."""
        # NOTE: docstring copied from aiohttp
        return self.base.reason

    @property
    def ok(self) -> bool:
        # NOTE: docstring copied from aiohttp
        """Boolean representation of HTTP status code.

        True if status is less than 400; otherwise, False.
        """
        return self.base.ok

    @property
    def method(self) -> str:
        """Request’s method."""  # noqa: RUF002
        # NOTE: docstring copied from aiohttp
        return self.base.method

    @property
    def url(self) -> yarl.URL:
        """URL of request."""
        # NOTE: docstring copied from aiohttp
        return self.base.url

    @property
    def real_url(self) -> yarl.URL:
        """Unmodified URL of request with URL fragment unstripped."""
        # NOTE: docstring copied from aiohttp
        return self.base.real_url

    @property
    def connection(self) -> Connection | None:
        """Connection used for handling response."""
        # NOTE: docstring copied from aiohttp
        return self.base.connection

    @property
    def cookies(self) -> SimpleCookie:
        """HTTP cookies of response (Set-Cookie HTTP header)."""
        return self.base.cookies

    @property
    def headers(self) -> CIMultiDictProxy[str]:
        """A case-insensitive multidict proxy with HTTP headers of response."""
        # NOTE: docstring copied from aiohttp
        return self.base.headers

    @property
    def raw_headers(self) -> RawHeaders:
        """Unmodified HTTP headers of response as unconverted bytes,
        a sequence of (key, value) pairs.
        """  # noqa: D205
        # NOTE: docstring copied from aiohttp
        return self.base.raw_headers

    @property
    def links(self) -> MultiDictProxy[MultiDictProxy[str | yarl.URL]]:
        """Link HTTP header parsed into a MultiDictProxy.

        For each link, key is link param rel when it exists, or link url as
        str otherwise, and value is MultiDictProxy of link params and url at
        key url as URL instance.
        """
        # NOTE: docstring copied from aiohttp
        return self.base.links

    @property
    def content_type(self) -> str:
        """Read-only property with content part of Content-Type header.

        Note: Returns value is 'application/octet-stream' if no Content-Type
        header present in HTTP headers according to RFC 2616. To make sure
        Content-Type header is not present in the server reply, use headers
        or raw_headers, e.g. 'CONTENT-TYPE' not in resp.headers.
        """
        # NOTE: docstring copied from aiohttp
        return self.base.content_type

    @property
    def charset(self) -> str | None:
        """Read-only property that specifies the encoding for the request’s BODY.

        The value is parsed from the Content-Type HTTP header.

        Returns str like 'utf-8' or None if no Content-Type header present in
        HTTP headers or it has no charset information.
        """  # noqa: RUF002
        # NOTE: docstring copied from aiohttp
        return self.base.charset

    @property
    def content_disposition(self) -> ContentDisposition | None:
        """Read-only property that specified the Content-Disposition HTTP header.

        Instance of ContentDisposition or None if no Content-Disposition header present in
        HTTP headers.
        """
        # NOTE: docstring copied from aiohttp
        return self.base.content_disposition

    @property
    def history(self) -> tuple[aiohttp.ClientResponse, ...]:
        """A Sequence of ClientResponse objects of preceding requests (earliest request first)
        if there were redirects, an empty sequence otherwise.
        """  # noqa: D205
        # NOTE: docstring copied from aiohttp
        return self.base.history

    def raise_for_status(self) -> None:
        """Raise an aiohttp.ClientResponseError if the response status is 400 or higher.

        Do nothing for success responses (less than 400).
        """
        # NOTE: docstring copied from aiohttp
        self.base.raise_for_status()

    @property
    def request_info(self) -> aiohttp.RequestInfo:
        """A typing.NamedTuple with request URL and headers from ClientRequest object,
        aiohttp.RequestInfo instance.
        """  # noqa: D205
        # NOTE: docstring copied from aiohttp
        return self.base.request_info

    def get_encoding(self) -> str:
        """Retrieve content encoding using charset info in Content-Type HTTP header.

        If no charset is present or the charset is not understood by Python, the
        fallback_charset_resolver function associated with the ClientSession is called.
        """
        # NOTE: docstring copied from aiohttp
        return self.base.get_encoding()

    def text(self, encoding: str | None = None) -> str:
        """Read response’s body and return decoded str using specified encoding parameter.

        If encoding is None content encoding is determined from the Content-Type header, or using
        the fallback_charset_resolver function.

        Args:
            encoding: ext encoding used for BODY decoding, or None for encoding autodetection.

        Returns:
            decoded BODY

        Raises:
            UnicodeDecodeError if decoding fails. See also get_encoding().

        """  # noqa: RUF002
        # NOTE: docstring copied from aiohttp
        return self.body.decode(encoding or self.get_encoding())

    def json(self) -> JSON:
        """Read response’s body as JSON, return dict using specified encoding and loader.

        If response’s content-type does not match content_type parameter aiohttp.ContentTypeError
        get raised. To disable content type check pass None value.
        """  # noqa: RUF002
        # NOTE: docstring copied from aiohttp
        return json.loads(self.text())

    def read(self) -> bytes:
        """Read the whole response’s body as bytes."""  # noqa: RUF002
        # NOTE: docstring copied from aiohttp
        return self.body


class ApicadabriRetryError(Exception):
    """Exception that is raised when a retry fails."""

    def __init__(self, i: int) -> None:
        """Initialize the exception with the retry number.

        Args:
            i: The number of the retry that failed (0-indexed).

        """
        super().__init__(f"{humanize.ordinal(i + 1)} retry failed.")


class ApicadabriMaxRetryError(Exception):
    """Exception that is raised when the maximum number of retries is reached."""

    def __init__(self, max_retries: int) -> None:
        """Initialize the exception with the maximum number of retries.

        Args:
            max_retries: The maximum number of retries that were attempted.

        """
        super().__init__(f"Call failed after {max_retries} retries.")


class AsyncRetrier:
    """Class that implements a retry mechanism for asynchronous functions.

    This implements an exponential backoff strategy for retrying failed calls.
    """

    def __init__(
        self,
        max_retries: int = 10,
        initial_sleep_s: float = 0.01,
        sleep_multiplier: float = 2,
        max_sleep_s: float = 60 * 15,
        should_retry: Callable[[Exception], bool] | None = None,
    ) -> None:
        """Initialize the retrier with the given parameters.

        Args:
            max_retries: The maximum number of retries to attempt.
            initial_sleep_s: The initial sleep time between retries.
            sleep_multiplier: The multiplier to apply to the sleep time after each retry.
            max_sleep_s: The maximum sleep time between retries.
            should_retry: A function that takes an exception and returns True if the
                          exception should be retried, False otherwise. Defaults to
                          retrying on all exceptions.

        """
        self.max_retries = max_retries
        self.initial_sleep_s = initial_sleep_s
        self.sleep_multiplier = sleep_multiplier
        self.max_sleep_s = max_sleep_s
        self.should_retry = should_retry if should_retry is not None else lambda _: True

    async def retries(self) -> AsyncGenerator[tuple[int, float], None]:
        """Create a Generator that yields the retry number and the sleep time.

        This will already apply the sleep time for the exponential backoff.

        """
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
        """Retry the given callable until it succeeds or the maximum number of retries is reached.

        Args:
            callable_to_retry: The callable to retry. This should be an async function
                               that returns a coroutine.

        Returns:
            The result of the callable once it succeeds.

        """
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
    """Response class for bulk API calls.

    Apart from serving as the base class for all bulk HTTP calls, this class
    also allows to use the Apicadabri functionality in a more generic way.

    You can implement any async task in `call_api` and supply an iterator by
    implementing `instances`. For example, this can be used to utilize
    convenience functions from an already existing Python library instead of
    calling the respective API directly via HTTP.

    Args:
        A: The type of the arguments that are passed to the API call.
        R: The return type of the API call.

    """

    def __init__(
        self,
        *args: tuple,
        max_active_calls: int = 20,
        retrier: AsyncRetrier | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the response object.

        Args:
            max_active_calls: The maximum number of concurrent API calls to make.
            retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                     If None, a new instance will be created with default parameters.
            args: Additional positional arguments to pass to the parent class.
            kwargs: Additional keyword arguments to pass to the parent class.

        """
        super().__init__(*args, **kwargs)
        self.semaphore = asyncio.Semaphore(max_active_calls)
        self.retrier = AsyncRetrier() if retrier is None else retrier

    async def call_all(self) -> AsyncGenerator[R, None]:
        """Return an iterator that yields the results of the API calls.

        This uses a semaphore to limit the number of concurrent API calls.

        It returns results in the same order as the input arguments from
        `instances`. However, it also allows to inspect and process results
        as they arrive.
        """
        next_index = 0
        buffer: list[tuple[int, R]] = []
        # TODO: would it make sense to allow generic types of sessions here instead of aiohttp?
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
        """Call the API with a semaphore (helper function)."""
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
    """Response class for bulk HTTP API calls."""

    def __init__(
        self,
        apicadabri_args: ApicadabriCallArguments,
        method: Literal["POST", "GET"],
        max_active_calls: int = 20,
        retrier: AsyncRetrier | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the response object.

        Args:
            apicadabri_args: The arguments to pass to the API call.
            method: The HTTP method to use for the API call (GET or POST).
            max_active_calls: The maximum number of concurrent API calls to make.
            retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                     If None, a new instance will be created with default parameters.
            kwargs: Additional keyword arguments to pass to the aiohttp get/post method.

        """
        super().__init__(max_active_calls=max_active_calls, retrier=retrier)
        self.apicadabri_args = apicadabri_args
        self.method = method
        self.aiohttp_kwargs = kwargs

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        index: int,
        args: ApicadabriCallInstance,
    ) -> tuple[int, SyncedClientResponse]:
        """Call the API with the given arguments and return the response.

        Args:
            session: The aiohttp client session to use for the request.
            index: The index of the instance in the list of instances.
            args: The arguments to pass to the API call.

        """
        aiohttp_method = session.post if self.method == "POST" else session.get
        method_args = {**args.model_dump(by_alias=True), **self.aiohttp_kwargs}
        async with (
            aiohttp_method(**method_args) as resp,
            resp,
        ):
            try:
                return (index, SyncedClientResponse(resp, await resp.read()))
            except Exception as e:  # noqa: BLE001
                return (
                    index,
                    SyncedClientResponse(
                        resp,
                        json.dumps(
                            exception_to_json(e),
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
        """Parse the response body as JSON.

        Args:
            on_error: Whether to just raise errors ("raise"), return an object encapsulating the
                      exception ("return") or use a function to supply a fallback result.

        Returns:
            A response object containing the JSON conent as dictionary.

        """
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
        """Parse the response body as text.

        Args:
            on_error: Whether to just raise errors ("raise"), return an object encapsulating the
                      exception ("return") or use a function to supply a fallback result.

        Returns:
            A response object containing the text content as string.

        """
        return self.map(SyncedClientResponse.text, on_error=on_error)

    def read(self) -> ApicadabriResponse[bytes]:
        """Read the response body as bytes.

        This cannot raise an exception, as the body is already read in the constructor.
        """
        # SyncedClientResponse.read just returns an internal variable
        # => there is no way this could raise an exception under normal circumstances
        # => if it does, it is an implementation error and we should just raise it normally
        return self.map(SyncedClientResponse.read, on_error="raise")


def bulk_get(  # noqa: PLR0913
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
    **kwargs: dict[str, Any],
) -> ApicadabriBulkHTTPResponse:
    """Make a bulk GET request to the given API endpoint.

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
            mode=mode,
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
    **kwargs: dict[str, Any],
) -> ApicadabriBulkHTTPResponse:
    """Make a bulk API call to the given API endpoint.

    This is a convenience function that wraps the `ApicadabriBulkHTTPResponse` class.

    Args:
        method: The HTTP method to use for the API call (GET or POST).
        apicadabri_args: The arguments to pass to the API call.
        max_active_calls: The maximum number of concurrent API calls to make.
        retrier: An instance of the AsyncRetrier class to use for retrying failed calls.
                 If None, a new instance will be created with default parameters.
        kwargs: Additional keyword arguments to pass to the aiohttp get/post method.

    Returns:
        A response object that can be used for further processing and retrieving the
        API responses.

    """
    # TODO: allow to pass extra args to aiohttp.ClientSession.get/post
    return ApicadabriBulkHTTPResponse(
        apicadabri_args=apicadabri_args,
        method=method,
        max_active_calls=max_active_calls,
        retrier=retrier,
        **kwargs,
    )

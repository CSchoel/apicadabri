"""Main module of apicadabri, containing all top-level members."""

import asyncio
import json
import traceback
from abc import abstractmethod
from bisect import insort_right
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from http.cookies import SimpleCookie
from itertools import product, repeat
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeAlias, TypeVar

import aiohttp
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


def exception_to_json(e: BaseException) -> dict[str, str]:
    return {
        "type": e.__class__.__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }


class ApicadabriErrorResponse(BaseModel):
    type: str
    message: str
    traceback: str

    @classmethod
    def from_exception(cls, e: BaseException) -> "ApicadabriErrorResponse":
        return ApicadabriErrorResponse(
            type=e.__class__.__name__,
            message=str(e),
            traceback=traceback.format_exc(),
        )


class ApicadabriCallInstance(BaseModel):
    url: str
    params: dict[str, str]
    # NOTE we need to use an alias to avoid shadowing the BaseModel field
    json_data: JSON = Field(alias="json")
    headers: dict[str, str]


class ApicadabriCallArguments(BaseModel):
    url: str | None
    urls: Iterable[str] | None
    params: dict[str, str] | None
    param_sets: Iterable[dict[str, str]] | None
    # NOTE we need to use an alias to avoid shadowing the BaseModel field
    json_data: JSON | None = Field(alias="json")
    json_sets: Iterable[JSON] | None
    headers: dict[str, str] | None
    header_sets: Iterable[dict[str, str]] | None
    mode: Literal["zip", "product", "pipeline"]

    @model_validator(mode="after")
    def validate_not_both_none(self):
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
    def validate_only_one_provided(self):
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

    def __iter__(self):
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
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
        return iter(
            ApicadabriCallInstance(url=u, params=p, json=j, headers=h) for u, p, j, h in combined
        )

    def any_iterable(self, single_val: A | None, multi_val: Iterable[A] | None) -> Iterable[A]:
        if single_val is None:
            if multi_val is None:
                msg = "Single and multi val cannot both be null."
                raise ValueError(msg)
            return multi_val
        if self.mode == "zip":
            return repeat(single_val)
        if self.mode == "multiply":
            return [single_val]
        if self.mode == "pipeline":
            msg = "Pipeline mode isn't implemented yet."
            raise NotImplementedError(msg)
        msg = f"Unrecognized mode {self.mode}"
        raise ValueError(msg)

    @property
    def url_iterable(self):
        return self.any_iterable(self.url, self.urls)

    @property
    def params_iterable(self):
        return self.any_iterable(self.params, self.param_sets)

    @property
    def json_iterable(self):
        return self.any_iterable(self.json_data, self.json_sets)

    @property
    def headers_iterable(self):
        return self.any_iterable(self.headers, self.header_sets)


R = TypeVar("R")
S = TypeVar("S")


class ApicadabriResponse(Generic[R]):
    def __init__(self):
        pass

    def map(self, func: Callable[[R], S]) -> "ApicadabriResponse[S]":
        """Apply a function to the response."""
        return ApicadabriMapResponse(self, func)

    @abstractmethod
    def call_all(self) -> AsyncGenerator[R, None]:
        """Return an iterator that yields the results of the API calls."""
        ...

    def to_jsonl(self, filename: Path | str) -> None:
        filename_path = Path(filename)
        with filename_path.open("w", encoding="utf-8") as f:
            asyncio.run(self.reduce(lambda _, r: f.write(json.dumps(r) + "\n"), start=0))

    def to_list(self) -> list[R]:
        start: list[R] = []

        def appender(lst: list[R], element: R):
            lst.append(element)
            return lst

        return asyncio.run(self.reduce(appender, start=start))

    async def reduce(self, accumulator: Callable[[A, R], A], start: A) -> A:
        accumulated = start
        async for res in self.call_all():
            accumulated = accumulator(accumulated, res)
        return accumulated


# TODO: create a map_safe variant that takes a second function to handle errors
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
    def __init__(self, base: ApicadabriResponse[R], map_func: Callable[[R], S], error_func: Callable[[R, BaseException], S]):
        self.map_func = map_func
        self.error_func = error_func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S, None]:
        """Return an iterator that yields the results of the API calls."""
        async for res in self.base.call_all():
            try:
                mapped = self.map_func(res)
                yield mapped
            except BaseException as e:  # noqa: BLE001
                yield self.error_func(res, e)


class SyncedClientResponse:
    def __init__(self, base: aiohttp.ClientResponse, body: bytes, *, is_exception: bool = False):
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


class ApicadabriBulkCallResponse(ApicadabriResponse[SyncedClientResponse]):
    def __init__(
        self,
        apicadabri_args: ApicadabriCallArguments,
        method: Literal["POST", "GET"],
        semaphore: asyncio.Semaphore,
    ):
        self.apicadabri_args = apicadabri_args
        self.method = method
        self.semaphore = semaphore

    async def call_api(
        self,
        args: ApicadabriCallInstance,
        session: aiohttp.ClientSession,
        index: int,
    ) -> tuple[int, SyncedClientResponse]:
        aiohttp_method = session.post if self.method == "POST" else session.get
        async with self.semaphore, aiohttp_method(**args.model_dump(by_alias=True)) as resp, resp:
            try:
                return (index, SyncedClientResponse(resp, await resp.read()))
            except BaseException as e:  # noqa: BLE001
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

    async def call_all(self):
        next_index = 0
        buffer: list[tuple[int, SyncedClientResponse]] = []
        async with aiohttp.ClientSession() as client:
            for res in asyncio.as_completed(
                [
                    self.call_api(instance, client, i)
                    for i, instance in enumerate(self.apicadabri_args)
                ],
            ):
                current_index, current_res = await res
                insort_right(buffer, (current_index, current_res), key=lambda x: -x[0])
                while current_index == next_index:
                    yield buffer.pop()[1]
                    current_index = buffer[-1][0] if len(buffer) > 0 else -1
                    next_index += 1

    def json(self) -> ApicadabriResponse[Any]:
        return self.map(SyncedClientResponse.json)

    def text(self) -> ApicadabriResponse[str]:
        return self.map(SyncedClientResponse.text)

    def read(self) -> ApicadabriResponse[bytes]:
        return self.map(SyncedClientResponse.read)


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
    **kwargs: dict[str, Any],
) -> ApicadabriBulkCallResponse:
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
        **kwargs,
    )


def bulk_call(
    method: Literal["POST", "GET"],
    apicadabri_args: ApicadabriCallArguments,
    max_active_calls: int = 20,
    # response_type: Literal["bytes", "str", "json", "raw"] = "json",
    **kwargs,
) -> ApicadabriBulkCallResponse:
    # TODO allow to pass extra args to aiohttp.ClientSession.get/post
    semaphore = asyncio.Semaphore(max_active_calls)

    return ApicadabriBulkCallResponse(
        apicadabri_args=apicadabri_args,
        method=method,
        semaphore=semaphore,
    )

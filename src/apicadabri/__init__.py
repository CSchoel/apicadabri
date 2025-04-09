"""Main module of apicadabri, containing all top-level members."""

import asyncio
import json
from abc import abstractmethod
from collections.abc import Iterable
from itertools import product, repeat
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Literal, Self, TypeVar

import aiohttp
from pydantic import BaseModel

# TODO limit max tasks committed to memory

# TODO allow different output types: to_jsonl, to_pandas, to_iterable


def bulk_get(
    url: str | None = None,
    urls: Iterable[str] | None = None,
    params: dict[str, str] | None = None,
    param_sets: Iterable[dict[str, str]] | None = None,
    json: dict[str, Any] | None = None,
    json_sets: Iterable[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
    header_sets: Iterable[dict[str, str]] | None = None,
    **kwargs: dict[str, Any],
) -> list[str]:
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
        **kwargs,
    )


A = TypeVar("A")


class ApicadabriCallInstance(BaseModel):
    url: str
    params: dict[str, str]
    json: dict[str, Any]
    headers: dict[str, str]


class ApicadabriCallArguments(BaseModel):
    url: str | None
    urls: Iterable[str] | None
    params: dict[str, str] | None
    param_sets: Iterable[dict[str, str]] | None
    json: dict[str, Any] | None
    json_sets: Iterable[dict[str, Any]] | None
    headers: dict[str, str] | None
    header_sets: Iterable[dict[str, str]] | None
    mode: Literal["zip", "product", "pipeline"]

    # TODO: Validate
    # - At least one not None
    # - Not _both_ list variant and single variant given
    # TODO: Get size if inputs are sized

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
        return self.any_iterable(self.json, self.json_sets)

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


class ApicadabriMapResponse(ApicadabriResponse[S], Generic[R, S]):
    def __init__(self, base: ApicadabriResponse[R], func: Callable[[R], S]):
        self.func = func
        self.base = base

    async def call_all(self) -> AsyncGenerator[S, None]:
        """Return an iterator that yields the results of the API calls."""
        async for res in self.base.call_all():
            yield self.func(res)


def bulk_call(
    method: Literal["POST", "GET"],
    apicadabri_args: ApicadabriCallArguments,
    **kwargs,
):
    semaphore = asyncio.Semaphore(20)
    session = aiohttp

    async def call_api(
        args: ApicadabriCallInstance, session: aiohttp.ClientSession
    ) -> dict[str, Any]:
        aiohttp_method = session.post if method == "POST" else session.get
        async with semaphore, aiohttp_method(**args.model_dump()) as resp:
            # TODO switch expected type based on header args
            return await resp.json()

    async def call_all():
        async with aiohttp.ClientSession() as client:
            # TODO: as_completed only allows generators as input since python 3.12
            for res in asyncio.as_completed(
                [call_api(instance, client) for instance in apicadabri_args],
            ):
                yield await res

    # TODO: buffer results in ordered data structure to return them in original order

    async def test_print():
        lst = []
        async for x in call_all():
            lst.append(x)
            print(x)
        return lst

    return asyncio.run(test_print())

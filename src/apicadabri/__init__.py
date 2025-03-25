import asyncio
from itertools import repeat
from typing import Any, Iterable, Literal, TypeVar

import aiohttp
from pydantic import BaseModel


def hello() -> str:
    return "Hello from apicadabri!"


def bulk_get(
    url: str | None,
    urls: Iterable[str] | None,
    params: dict[str, str],
    param_sets: Iterable[dict[str, str]],
    json: dict[str, Any],
    json_sets: Iterable[dict[str, Any]],
    headers: dict[str, str],
    header_sets: Iterable[dict[str, str]],
    **kwargs: Any,
) -> list[str]:
    return bulk_call(
        ApicadabriCallArguments(
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
        **kwargs
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
    mode: Literal["zip", "multiply", "pipeline"]

    # TODO: Validate
    # - At least one not None
    # - Not _both_ list variant and single variant given
    # TODO: Get size if inputs are sized

    def __iter__(self):
        iterables = (self.url_iterable, self.params_iterable, self.json_iterable, self.headers_iterable)
        if self.method == "zip":
            combined = zip(*iterables, strict=False)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
        return iter(ApicadabriCallInstance(u,p,j,h) for u,p,j,h in combined)

    def any_iterable(self, single_val: A, multi_val: Iterable[A]) -> Iterable[A]:
        if single_val is None:
            return multi_val
        if self.mode == "zip":
            return repeat(single_val)
        elif self.mode == "multiply":
            return [single_val]
        elif self.mode == "pipeline":
            raise NotImplementedError("Pipeline mode isn't implemented yet.")
        else:
            raise ValueError(f"Unrecognized mode {self.mode}")

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


def bulk_call(
    method: Literal["POST", "GET"],
    apicadabri_args: ApicadabriCallArguments,
    **kwargs
):
    semaphore = asyncio.Sempaphore(20)
    session = aiohttp
    async def call_api(args: ApicadabriCallInstance, session: aiohttp.ClientSession):
        aiohttp_method = session.post if method == "POST" else session.get
        async with semaphore, aiohttp_method(**args.model_dump()) as resp:
            # TODO switch expected type based on header args
            return await resp.json()

    async def bulk_call():
        async with aiohttp.ClientSession() as client:
            for instance in ApicadabriCallArguments:
                yield await call_api(instance)
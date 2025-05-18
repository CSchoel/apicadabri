"""Tests for map functions."""

import asyncio
import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import apicadabri


# Source: https://stackoverflow.com/a/59351425
class MockResponse:
    """Mocks aiohttp response object."""

    def __init__(self, text: str, status: int, latency: float | Callable[[], float] = 0.0) -> None:
        """Initialize mock object."""
        self._text = text
        self.status = status
        self.latency = latency
        self.content = MagicMock()

    async def read(self) -> bytes:
        """Get content as bytes."""
        return self._text.encode("utf-8")

    async def maybe_sleep(self) -> None:
        """Sleep if latency has been set."""
        if not isinstance(self.latency, (float, int)):
            await asyncio.sleep(self.latency())
        elif self.latency > 0:
            await asyncio.sleep(self.latency)

    async def text(self) -> str:
        """Get content as string."""
        await self.maybe_sleep()
        return self._text

    async def json(self) -> Any:  # noqa: ANN401
        """Get content as JSON."""
        await self.maybe_sleep()
        return json.loads(self._text)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Allow use in `async with`."""
        return None

    async def __aenter__(self) -> AbstractAsyncContextManager | None:
        """Allow use in `async with`."""
        return self

    def get_encoding(self) -> str:
        """Return encoding of content."""
        return "utf-8"


def test_simple_map(mocker: MockerFixture) -> None:
    """Hypothesis: Simple map method call yields expected result."""
    pokemon = ["bulbasaur", "squirtle", "charmander"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=lambda *args, **kwargs: MockResponse(
            json.dumps({"name": kwargs["url"].split("/")[-1]}),
            200,
        ),
    )
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
        )
        .json()
        .map(lambda res: res["name"])
        .to_list()
    )
    assert data == pokemon


def test_simple_map_error(mocker: MockerFixture) -> None:
    """Hypothesis: With default settings, exceptions thrown in `map` will be re-raised."""
    pokemon = ["bulbasaur", "squirtle", "charmander"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=lambda *args, **kwargs: MockResponse(
            "{}"
            if "squirtle" in kwargs["url"]
            else json.dumps({"name": kwargs["url"].split("/")[-1]}),
            200,
        ),
    )
    with pytest.raises(KeyError):
        _ = (
            apicadabri.bulk_get(
                urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
            )
            .json()
            .map(lambda res: res["name"])
            .to_list()
        )


def test_safe_map_error(mocker: MockerFixture) -> None:
    """Hypothesis: When an error handling function is supplied, no exception will be raised."""
    pokemon = ["bulbasaur", "squirtle", "charmander"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=lambda *args, **kwargs: MockResponse(
            "{}"
            if "squirtle" in kwargs["url"]
            else json.dumps({"name": kwargs["url"].split("/")[-1]}),
            200,
        ),
    )
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
        )
        .json()
        .map(lambda res: res["name"], on_error=lambda _, e: str(e))
        .to_list()
    )
    assert data == ["bulbasaur", "'name'", "charmander"]


def test_map_maybe_error(mocker: MockerFixture) -> None:
    """Hypothesis: With `on_error="return`, an error object is returned."""
    pokemon = ["bulbasaur", "squirtle", "charmander"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=lambda *args, **kwargs: MockResponse(
            "{}"
            if "squirtle" in kwargs["url"]
            else json.dumps({"name": kwargs["url"].split("/")[-1]}),
            200,
        ),
    )
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
        )
        .json()
        .map(lambda res: res["name"], on_error="return")
        .to_list()
    )
    assert data[0] == "bulbasaur"
    assert data[2] == "charmander"
    assert isinstance(data[1], apicadabri.ApicadabriErrorResponse)
    assert data[1].type == "KeyError"

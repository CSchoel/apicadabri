"""Tests for map functions."""

import asyncio
import json
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

import apicadabri


# Source: https://stackoverflow.com/a/59351425
class MockResponse:
    def __init__(self, text, status, latency: float | Callable[[], float] = 0.0):
        self._text = text
        self.status = status
        self.latency = latency
        self.content = MagicMock()

    async def read(self):
        return self._text.encode("utf-8")

    async def maybe_sleep(self):
        if not isinstance(self.latency, (float, int)):
            await asyncio.sleep(self.latency())
        elif self.latency > 0:
            await asyncio.sleep(self.latency)

    async def text(self):
        await self.maybe_sleep()
        return self._text

    async def json(self):
        await self.maybe_sleep()
        return json.loads(self._text)

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

    def get_encoding(self) -> str:
        return "utf-8"


def test_simple_map(mocker):
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


def test_simple_map_error(mocker):
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


def test_safe_map_error(mocker):
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


def test_map_maybe_error(mocker):
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

import asyncio
import json
import random
import time
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


def test_multi_url():
    pokemon = ["bulbasaur", "squirtle", "charmander"]
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
        )
        .json()
        .to_list()
    )
    assert len(data) == len(pokemon)
    assert all(d["name"] in pokemon for d in data)
    assert [d["name"] for d in data] == pokemon


def test_multi_url_mocked(mocker):
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
        .to_list()
    )
    assert len(data) == len(pokemon)
    assert all(d["name"] in pokemon for d in data)
    assert [d["name"] for d in data] == pokemon


def test_multi_url_speed(mocker):
    random.seed(2456567663)
    data = {"answer": 42}
    resp = MockResponse(
        json.dumps(data),
        200,
        latency=lambda: 0.01 if random.random() > 0.01 else 0.1,
    )

    mocker.patch("aiohttp.ClientSession.get", return_value=resp)
    tstamp = time.time()
    lst = (
        apicadabri.bulk_get(
            urls=(str(x) for x in range(1000)),
            max_active_calls=100,
        )
        .json()
        .to_list()
    )
    elapsed = time.time() - tstamp
    # total time = 990 * 0.01 + 10 * 0.1 = 10.9
    # speedup without overhead: 100x (with 100 parallel slots for tasks)
    # => theoretic time = 10.9 / 100 = 0.109
    assert elapsed < 0.3
    assert len(lst) == 1000
    assert lst[0] == {"answer": 42}


@pytest.mark.parametrize(
    ("n", "max_active_calls", "expected_time_s"),
    [
        pytest.param(10_000, 1000, 0.3, id="10k"),
        pytest.param(100_000, 1000, 20, id="100k"),
        pytest.param(
            1_000_000,
            1000,
            200,
            id="1M",
            marks=pytest.mark.skip(
                reason="This test takes >100s to run, so we skip it by default.",
            ),
        ),
    ],
)
def test_task_limit(mocker, n, max_active_calls, expected_time_s):
    data = {"answer": 42}
    resp = MockResponse(json.dumps(data), 200, latency=0)

    mocker.patch("aiohttp.ClientSession.get", return_value=resp)
    tstamp = time.time()
    lst = (
        apicadabri.bulk_get(
            urls=(str(x) for x in range(n)),
            max_active_calls=max_active_calls,
        )
        .json()
        .to_list()
    )
    elapsed = time.time() - tstamp
    assert elapsed < expected_time_s
    assert len(lst) == n
    assert lst[0] == {"answer": 42}

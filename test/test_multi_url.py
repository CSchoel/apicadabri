import asyncio
import json
import random
import time
from typing import Callable

import apicadabri


# Source: https://stackoverflow.com/a/59351425
class MockResponse:
    def __init__(self, text, status, latency: int | Callable[[], float] = 0):
        self._text = text
        self.status = status
        self.latency = latency

    async def maybe_sleep(self):
        if not isinstance(self.latency, int):
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


def test_multi_url():
    pokemon = ["bulbasaur", "squirtle", "charmander"]
    data = apicadabri.bulk_get(
        urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon)
    ).to_list()
    assert len(data) == len(pokemon)
    assert all(d["name"] in pokemon for d in data)
    assert [d["name"] for d in data] == pokemon

def test_multi_url_mocked(mocker):
    pokemon = ["bulbasaur", "squirtle", "charmander"]
    data = apicadabri.bulk_get(
        urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon)
    ).to_list()
    mocker.patch("aiohttp.ClientSession.get", return_value=resp)
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
    lst = apicadabri.bulk_get(urls=(str(x) for x in range(1000)), max_active_calls=100).to_list()
    elapsed = time.time() - tstamp
    # total time = 990 * 0.01 + 10 * 0.1 = 10.9
    # speedup without overhead: 100x (with 100 parallel slots for tasks)
    # => theoretic time = 10.9 / 100 = 0.109
    assert elapsed < 0.3
    assert len(lst) == 1000
    assert lst[0] == {"answer": 42}

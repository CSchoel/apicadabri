import asyncio
import cProfile
import json
from collections.abc import Callable

import aiohttp

import apicadabri


class MockResponse:
    def __init__(self, text, status, latency: float | Callable[[], float] = 0.0):
        self._text = text
        self.status = status
        self.latency = latency

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


def profile_run():
    data = {"answer": 42}
    resp = MockResponse(json.dumps(data), 200, latency=0)
    aiohttp.ClientSession.get = lambda *args, **kwargs: resp  # type: ignore
    _ = apicadabri.bulk_get(
        urls=(str(x) for x in range(100_000)),
        max_active_calls=1000,
    ).to_list()


if __name__ == "__main__":
    cProfile.run("profile_run()", sort="cumtime")

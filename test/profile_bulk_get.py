"""Tests for main functions for bulk calls."""

import asyncio
import cProfile
import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any
from unittest.mock import MagicMock

import aiohttp

import apicadabri


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


def profile_run() -> None:
    """Run a profiling test with mock results."""
    data = {"answer": 42}
    resp = MockResponse(json.dumps(data), 200, latency=0)
    aiohttp.ClientSession.get = lambda *args, **kwargs: resp  # type: ignore
    _ = apicadabri.bulk_get(
        urls=(str(x) for x in range(100_000)),
        max_active_calls=1000,
    ).to_list()


if __name__ == "__main__":
    cProfile.run("profile_run()", sort="cumtime")

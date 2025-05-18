"""Tests for retry functionality."""

import json
import timeit
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import apicadabri


class MockResponse:
    """Initialize mock object."""

    def __init__(self) -> None:
        """Initialize mock object."""
        self._text = '{"result": "success"}'
        self.status = 200
        self.content = MagicMock()

    async def read(self) -> bytes:
        """Get content as bytes."""
        return self._text.encode("utf-8")

    async def text(self) -> str:
        """Get content as string."""
        return self._text

    async def json(self) -> Any:  # noqa: ANN401
        """Get content as JSON."""
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


class Failer:
    """Callable that fails a predefined number of times before it succeeds."""

    def __init__(self, n_fails: int, exception_class: type[Exception] = Exception) -> None:
        """Create new failer."""
        self.n_fails = n_fails
        self.fail_count = 0
        self.exception_class = exception_class

    def __call__(self, *args: list[Any], **kwargs: dict[str, Any]) -> MockResponse:
        """Fail `n_fails` time with an exception of type `self.exception_class`."""
        if self.fail_count < self.n_fails:
            msg = f"Fail {self.fail_count + 1}"
            self.fail_count += 1
            raise self.exception_class(msg)
        return MockResponse()


def test_fail_once(mocker: MockerFixture) -> None:
    """Hypothesis: Default retrier will succeed when a response fails once."""
    pokemon = ["bulbasaur"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(1),
    )
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
        )
        .json()
        .to_list()
    )
    assert len(data) == len(pokemon)
    assert data == [{"result": "success"}]


def test_fail_exactly_max_retries(mocker: MockerFixture) -> None:
    """Hypothesis: Default retrier will succeed when a response fails multiple times."""
    pokemon = ["bulbasaur"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(5),
    )
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
        )
        .json()
        .to_list()
    )
    assert len(data) == len(pokemon)
    assert data == [{"result": "success"}]


def test_fail_over_max_retries(mocker: MockerFixture) -> None:
    """Hypothesis: Retrier will succeed when a response fails `max_retries` times."""
    pokemon = ["bulbasaur"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(3),
    )
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
            retrier=apicadabri.AsyncRetrier(max_retries=3),
        )
        .json()
        .to_list()
    )
    assert len(data) == len(pokemon)
    assert data == [{"result": "success"}]


def test_fail_completely(mocker: MockerFixture) -> None:
    """Hypothesis: Retrier will fail when a response fails `max_retries + 1` times."""
    pokemon = ["bulbasaur"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(4),
    )
    with pytest.raises(apicadabri.ApicadabriMaxRetryError):
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
            retrier=apicadabri.AsyncRetrier(max_retries=3),
        ).json().to_list()


def test_fail_once_filtered(mocker: MockerFixture) -> None:
    """Hypothesis: Retrier succeeds when a selected exception occurs once."""
    pokemon = ["bulbasaur"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(1, ValueError),
    )
    data = (
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
            retrier=apicadabri.AsyncRetrier(should_retry=lambda e: isinstance(e, ValueError)),
        )
        .json()
        .to_list()
    )
    assert len(data) == len(pokemon)
    assert data == [{"result": "success"}]


def test_fail_completely_filtered(mocker: MockerFixture) -> None:
    """Hypothesis: Retrier fails on exception that was not selected for retry."""
    pokemon = ["bulbasaur"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(1, ValueError),
    )
    with pytest.raises(apicadabri.ApicadabriRetryError) as error_info:
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
            retrier=apicadabri.AsyncRetrier(should_retry=lambda e: False),
        ).json().to_list()
    assert isinstance(error_info.value.__cause__, ValueError)
    assert str(error_info.value.__cause__) == "Fail 1"


def test_backoff_three(mocker: MockerFixture) -> None:
    """Hypothesis: Exponential backoff sleeps for expected time on three retries."""
    pokemon = ["bulbasaur"]
    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(3),
    )
    t = timeit.default_timer()
    apicadabri.bulk_get(
        urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
    ).json().to_list()
    t = timeit.default_timer() - t
    assert t > 0.01 + 0.02 + 0.03
    assert t < 0.01 + 0.02 + 0.03 + 0.06


def test_backoff_five(mocker: MockerFixture) -> None:
    """Hypothesis: Exponential backoff sleeps for expected time on five retries."""
    pokemon = ["bulbasaur"]
    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(5),
    )
    t = timeit.default_timer()
    apicadabri.bulk_get(
        urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
    ).json().to_list()
    t = timeit.default_timer() - t
    assert t > 0.01 + 0.02 + 0.03 + 0.06 + 0.12
    assert t < 0.01 + 0.02 + 0.03 + 0.06 + 0.12 + 0.24

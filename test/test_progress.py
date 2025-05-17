"""Tests related to progress bars."""

import json
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from apicadabri import ApicadabriCallArguments, ApicadabriSizeUnknownError, bulk_get


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


class TestArgumentsSize:
    """Tests for determining the size of APicadabriCallArguments."""

    def test_one_sized_arg(self) -> None:
        """Test hypothesis: With a single list input, the size can be determined without hints."""
        args = ApicadabriCallArguments(urls=["foo", "bar", "baz"])
        assert len(args) == 3

    def test_two_sized_arg_zip(self) -> None:
        """Test hypothesis: With two list inputs in zip mode, the size can be determined."""
        args = ApicadabriCallArguments(urls=["foo", "bar", "baz"], json_sets=[{}] * 3)
        assert len(args) == 3

    def test_two_sized_arg_product(self) -> None:
        """Test hypothesis: With two list inputs in product mode, the size can be determined."""
        args = ApicadabriCallArguments(
            urls=["foo", "bar", "baz"],
            json_sets=[{}] * 4,
            mode="product",
        )
        assert len(args) == 12

    def test_iterator_without_hint(self) -> None:
        """Test hypothesis: Determining the size of a non-sized input fails if no hint is given."""
        args = ApicadabriCallArguments(urls=(x for x in ["foo", "bar", "baz"]))
        with pytest.raises(ApicadabriSizeUnknownError):
            len(args)

    def test_iterator_with_hint(self) -> None:
        """Test hypothesis: Determining the size of a non-sized input with hint is possible."""
        args = ApicadabriCallArguments(urls=(x for x in ["foo", "bar", "baz"]), size=3)
        assert len(args) == 3


class TestResponseSize:
    """Tests for determining the size of ApicadabriResponse objects."""

    def test_response(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the size of a bulk call response is possible."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = bulk_get(
            urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
        )
        assert len(data) == 3

    def test_response_iterator_without_hint(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the size of a bulk call fails with iterable and no hint."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
        )
        with pytest.raises(ApicadabriSizeUnknownError):
            len(data)

    def test_response_iterator_with_hint(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the size of a bulk call works with iterable and hint."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
            size=3,
        )
        assert len(data) == 3

    def test_json(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the size of a json response is possible."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = bulk_get(
            urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
        ).json()
        assert len(data) == 3

    def test_map(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the size of a map response is possible."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = (
            bulk_get(
                urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
            )
            .json()
            .map(lambda x: x.keys())
        )
        assert len(data) == 3

    def test_map_return(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the size of a map response with "return" mode works."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = (
            bulk_get(
                urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
            )
            .json()
            .map(lambda x: x.keys(), on_error="return")
        )
        assert len(data) == 3

    def test_map_errorfunc(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the size of a map response with error function works."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = (
            bulk_get(
                urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
            )
            .json()
            .map(lambda x: x.keys(), on_error=lambda e, x: set())
        )
        assert len(data) == 3

    def test_tqdm(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Determining the length of a tqdm response is possible."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = (
            bulk_get(
                urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
            )
            .json()
            .tqdm()
        )
        assert len(data) == len(pokemon)

    def test_progress(self, mocker: MockerFixture) -> None:
        """Test hypothesis: Calling `tqdm` on a sized response is possible."""
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = (
            bulk_get(
                urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
            )
            .json()
            .tqdm()
            .to_list()
        )
        assert len(data) == len(pokemon)

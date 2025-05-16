"""Tests related to progress bars."""

import json
from unittest.mock import MagicMock

import pytest

from apicadabri import ApicadabriCallArguments, ApicadabriSizeUnknownError, bulk_get


class MockResponse:
    def __init__(self):
        self._text = '{"result": "success"}'
        self.status = 200
        self.content = MagicMock()

    async def read(self):
        return self._text.encode("utf-8")

    async def text(self):
        return self._text

    async def json(self):
        return json.loads(self._text)

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

    def get_encoding(self) -> str:
        return "utf-8"


class TestArgumentsSize:
    def test_one_sized_arg(self):
        args = ApicadabriCallArguments(urls=["foo", "bar", "baz"])
        assert len(args) == 3

    def test_two_sized_arg_zip(self):
        args = ApicadabriCallArguments(urls=["foo", "bar", "baz"], json_sets=[{}] * 3)
        assert len(args) == 3

    def test_two_sized_arg_product(self):
        args = ApicadabriCallArguments(
            urls=["foo", "bar", "baz"],
            json_sets=[{}] * 4,
            mode="product",
        )
        assert len(args) == 12

    def test_iterator_without_hint(self):
        args = ApicadabriCallArguments(urls=(x for x in ["foo", "bar", "baz"]))
        with pytest.raises(ApicadabriSizeUnknownError):
            len(args)

    def test_iterator_with_hint(self):
        args = ApicadabriCallArguments(urls=(x for x in ["foo", "bar", "baz"]), size=3)
        assert len(args) == 3

    def test_response(self, mocker):
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = bulk_get(
            urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
        )
        assert len(data) == 3

    def test_json(self, mocker):
        pokemon = ["bulbasaur", "squirtle", "charmander"]

        mocker.patch(
            "aiohttp.ClientSession.get",
            side_effect=lambda *args, **kwargs: MockResponse(),
        )
        data = bulk_get(
            urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
        ).json()
        assert len(data) == 3

    def test_map(self, mocker):
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

    def test_tqdm(self, mocker):
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

    def test_progress(self, mocker):
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

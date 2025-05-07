"""Tests for retry functionality."""

import json
from unittest.mock import MagicMock

import apicadabri


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


class Failer:
    def __init__(self, n_fails: int, exception_class: type[Exception] = Exception):
        self.n_fails = n_fails
        self.fail_count = 0
        self.exception_class = exception_class

    def __call__(self, *args, **kwargs):
        if self.fail_count < self.n_fails:
            msg = f"Fail {self.fail_count + 1}"
            self.fail_count += 1
            raise self.exception_class(msg)
        return MockResponse()


def test_fail_once(mocker):
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

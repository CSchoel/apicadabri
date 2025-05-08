"""Tests for retry functionality."""

import json
import timeit
from unittest.mock import MagicMock

import pytest

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


def test_fail_multiple(mocker):
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


def test_fail_completely(mocker):
    pokemon = ["bulbasaur"]

    mocker.patch(
        "aiohttp.ClientSession.get",
        side_effect=Failer(3),
    )
    with pytest.raises(apicadabri.ApicadabriMaxRetryError):
        apicadabri.bulk_get(
            urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
            retrier=apicadabri.AsyncRetrier(max_retries=3),
        ).json().to_list()


def test_fail_once_filtered(mocker):
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


def test_fail_completely_filtered(mocker):
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


def test_backoff_three(mocker):
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


def test_backoff_five(mocker):
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

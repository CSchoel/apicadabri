"""Tests related to recursive tasks."""

import re
from collections.abc import Iterable
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientSession

from apicadabri import (
    ApicadabriCallArguments,
    ApicadabriCallInstance,
    AsyncRetrier,
    SyncedClientResponse,
)
from apicadabri.recursive import ApicadabriRecursiveResponse, recursive_get


class DummyRR(ApicadabriRecursiveResponse[tuple[int, ...], str]):
    """Dummy class for testing recursive tasks."""

    async def call_api(
        self,
        client: ClientSession,
        index: int,
        instance_args: tuple[int, ...],
    ) -> tuple[int, str]:
        """Dummy api call that just turns input to string and spawns one level of subtasks."""
        if len(instance_args) == 1:
            for i in range(3):
                await self.schedule_subtask(client, (*list(instance_args), i))
        return (index, str(".".join([str(x) for x in instance_args])))

    def instances(self) -> Iterable[tuple[int, ...]]:
        """Returns dummy instances."""
        return [(1,), (2,)]


class TestRecursiveResponse:
    """Tests for base functionality of ApicadabriRecursiveResponse."""

    def test_dummy(self) -> None:
        """Hypothesis: A task that spawns subtasks returns all subtask responses without errors."""
        result = DummyRR()
        res = result.to_list()
        assert {"1", "1.0", "1.1", "1.2", "2", "2.0", "2.1", "2.2"} == set(res)


class TestRecursiveGet:
    """Tests for the top-level `recursive_get` function."""

    def test_wiki(self) -> None:
        """Hypothesis: A task that crawls websites recursively returns all expected results."""
        self.download_counter = 0

        def create_subtask(
            client: aiohttp.ClientSession,
            index: int,
            instance_args: ApicadabriCallInstance,
            result: SyncedClientResponse,
        ) -> Iterable[ApicadabriCallInstance]:
            text = result.text()
            first_link = re.search(pattern=r'href="([^"]+?.html?)"', string=text)
            if first_link is not None and self.download_counter < 4:
                self.download_counter += 1
                url = first_link.group(1)
                absolute_url = urljoin(instance_args.url, url)
                return ApicadabriCallArguments(url=absolute_url, headers=self.headers)
            return []

        self.headers = {
            "User-Agent": (
                "ApicadabriBot/1.0 (https://arbitrary-but-fixed.net/;"
                " apicadabri@arbitrary-but-fixed.org) apicadabri/1.0"
            ),
        }
        res = recursive_get(
            url="https://arbitrary-but-fixed.net/",
            headers=self.headers,
            subtask_creator=create_subtask,
            retrier=AsyncRetrier(),
        ).to_list()
        assert len(res) == 5

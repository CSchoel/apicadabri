"""Tests related to recursive tasks."""

import asyncio
import re
from collections.abc import Iterable
from urllib.parse import urljoin

import aiohttp
import pytest
from aiohttp import ClientSession

from apicadabri import (
    ApicadabriCallArguments,
    ApicadabriCallInstance,
    AsyncRetrier,
    SyncedClientResponse,
)
from apicadabri.recursive import (
    ApicadabriRecursiveResponse,
    SubtaskIndexer,
    recursive_get,
)


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
                tuple_idx = (*await self.indexer.get_tuple_index(index), i)
                await self.schedule_subtask(
                    client,
                    (*list(instance_args), i),
                    tuple_idx,
                )
        return (index, str(".".join([str(x) for x in instance_args])))

    def instances(self) -> Iterable[tuple[int, ...]]:
        """Returns dummy instances."""
        return [(1,), (2,)]


async def get_tuple_indices(indexer: SubtaskIndexer, n: int) -> list[tuple[int, ...]]:
    tuple_indices = [await indexer.get_tuple_index(i) for i in range(n)]
    return tuple_indices


class TestRecursiveResponse:
    """Tests for base functionality of ApicadabriRecursiveResponse."""

    def test_dummy(self) -> None:
        """Hypothesis: A task that spawns subtasks returns all subtask responses without errors."""
        result = DummyRR()
        res = result.to_list()
        assert {"1", "1.0", "1.1", "1.2", "2", "2.0", "2.1", "2.2"} == set(res)

    def test_in_order(self) -> None:
        result = DummyRR()
        _ = result.to_list()
        tuple_indices = asyncio.run(get_tuple_indices(result.indexer, 8))
        assert tuple_indices == [
            (0,),
            (1,),
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ]


class TestRecursiveGet:
    """Tests for the top-level `recursive_get` function."""

    def test_wiki(self) -> None:
        """Hypothesis: A task that crawls websites recursively returns all expected results."""
        self.download_counter = 0

        def create_subtask(
            client: aiohttp.ClientSession,
            index: tuple[int, ...],
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


class TestSubtaskIndexer:
    """Tests for the `SubtaskIndexer` class."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_one(self) -> None:
        """Hypothesis: Adding one elment to the indexer and retrieving it works."""
        indexer = SubtaskIndexer()
        idx = await indexer.add_tuple_index((0,))
        assert idx == 0
        assert await indexer.get_tuple_index(0) == (0,)

    @pytest.mark.asyncio
    async def test_add_multiple(self) -> None:
        """Hypothesis: Adding multiple elements yields successive indices."""
        indexer = SubtaskIndexer()
        for i in range(10):
            idx = await indexer.add_tuple_index((i,))
            assert idx == i

    @pytest.mark.asyncio
    async def test_add_multiple_layers(self) -> None:
        """Hypothesis: After adding elements with multiple layers the can be retrieved."""
        indexer = SubtaskIndexer()
        for i in [(0,), (0, 0), (0, 1), (1,), (1, 0), (1, 1)]:
            idx = await indexer.add_tuple_index(i)
            tuple_idx = await indexer.get_tuple_index(idx)
            assert tuple_idx == i

    @pytest.mark.asyncio
    async def test_expect_in_order(self) -> None:
        """Hypothesis: After adding elements with multiple layers the can be retrieved."""
        indexer = SubtaskIndexer()
        for i in [(1,), (0, 1), (0, 0), (0,), (1, 1), (1, 0)]:
            await indexer.add_tuple_index(i)
        expected_order = [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
        actual_order = [await indexer.next_index(i) for i in range(len(expected_order))]
        assert expected_order == actual_order

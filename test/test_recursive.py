"""Tests related to recursive tasks."""

from collections.abc import Iterable

import aiohttp
from aiohttp import ClientSession

from apicadabri import ApicadabriCallInstance, SyncedClientResponse
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
        def create_subtask(
            client: aiohttp.ClientSession,
            index: int,
            instance_args: ApicadabriCallInstance,
            result: SyncedClientResponse,
        ) -> Iterable[ApicadabriCallInstance]:
            return []

        res = recursive_get(
            url="https://en.wikipedia.org/wiki/Snake", subtask_creator=create_subtask
        )
        assert res == None

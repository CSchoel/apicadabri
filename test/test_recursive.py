"""Tests related to recursive tasks."""

from collections.abc import Iterable

from aiohttp import ClientSession

from apicadabri.recursive import ApicadabriRecursiveResponse


class DummyRR(ApicadabriRecursiveResponse[int, str]):
    async def call_api(self, client: ClientSession, index: int, instance_args: int) -> tuple[int, str]:
        for i in range(3):
            self.task_group.create_task(self.subtask(instance_args, i))
        return (index, str(instance_args))
    async def subtask(self, instance_args: int, subtask_args: int) -> None:
        await self.result_queue.put(f"{instance_args}.{subtask_args}")
    def instances(self) -> Iterable[int]:
        return [1, 2]

class TestRecursiveResponse:
    """Tests for determining the size of APicadabriCallArguments."""

    def test_dummy(self) -> None:
        """Hypothesis: With a single list input, the size can be determined without hints."""
        result = DummyRR()
        res = result.to_list()
        assert {"1", "1.0", "1.1", "1.2", "2", "2.0", "2.1", "2.2"} == set(res)
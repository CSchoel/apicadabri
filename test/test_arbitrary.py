"""Tests for using arbitrary async tasks."""

from typing import Iterable

from aiohttp import ClientSession

from apicadabri import ApicadabriBulkResponse


class TestTask(ApicadabriBulkResponse[str, int]):
    def __init__(self, data: list[str], max_active_calls: int = 10):
        super().__init__(max_active_calls=max_active_calls)
        self.data = data

    async def call_api(
        self, client: ClientSession, index: int, instance_args: str
    ) -> tuple[int, int]:
        return (index, len(instance_args))

    def instances(self) -> Iterable[str]:
        return self.data


def test_arbitrary() -> None:
    """Test hypothesis: We can use an arbitrary task with ApicadabriBulkResponse."""
    data = ["bulbasaur", "squirtle", "charmander"]
    task = TestTask(data)
    result = task.to_list()
    assert len(result) == len(data)
    assert result == [9, 8, 10]

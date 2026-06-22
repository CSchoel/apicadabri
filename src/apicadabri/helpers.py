from abc import abstractmethod
from bisect import insort_right
from typing import Generic, Protocol, Self, TypeVar

R = TypeVar("R")
I = TypeVar("I")


class Ordered(Protocol):
    def __lt__(self, value: Self, /) -> bool: ...
    def __gt__(self, value: Self, /) -> bool: ...


class BufferedOrdererBase(Generic[I, R]):
    def __init__(self):
        self.n_retrieved = 0
        self.buffer: list[tuple[I, R]] = []

    @abstractmethod
    def sorting_key(self, index: I) -> Ordered: ...

    def insert_in_order(self, index: I, result: R):
        insort_right(self.buffer, (index, result), key=lambda x: self.sorting_key(x[0]))

    def current_index(self) -> I | None:
        return self.buffer[-1][0] if len(self.buffer) > 0 else None

    @abstractmethod
    async def next_expected_index(self, n: int) -> I | None: ...

    async def retrieve_next(self) -> list[R]:
        results = []
        while (
            self.current_index() is not None
            and self.current_index() == await self.next_expected_index(self.n_retrieved)
        ):
            results.append(self.buffer.pop()[1])
            self.n_retrieved += 1
        return results


class BufferedOrderer(BufferedOrdererBase[int, R], Generic[R]):
    def sorting_key(self, index: int) -> Ordered:
        return -index

    async def next_expected_index(self, n: int) -> int:
        return n

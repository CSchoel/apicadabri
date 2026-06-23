"""Helper functions and classes."""

from abc import abstractmethod
from bisect import insort_right
from typing import Generic, Protocol, Self, TypeVar

R = TypeVar("R")
Index = TypeVar("Index")


class Ordered(Protocol):
    """Protocol for any type that can be compared against itself."""

    def __lt__(self, value: Self, /) -> bool: ...  # noqa: D105
    def __gt__(self, value: Self, /) -> bool: ...  # noqa: D105


class BufferedOrdererBase(Generic[Index, R]):
    """Saves results of type `R` in a buffer and allows to return them in order.

    Args:
        Index: The type of the index used for ordering. Usually an int, but
               more complex task structures such as recursive calls require
               a more complex type.
        R: Type of the results stored in this buffer.
    """

    def __init__(self) -> None:
        """Create a new orderer."""
        self.n_retrieved = 0
        self.buffer: list[tuple[Index, R]] = []

    @abstractmethod
    def sorting_key(self, index: Index) -> Ordered:
        """Returns the key that is used for sorting the indices.

        Args:
            index: The index from which to generate the key.

        Returns:
            A key used for sorting.
        """

    def insert_in_order(self, index: Index, result: R) -> None:
        """Inserts a new result under a given index.

        This method ensures that the internal buffer stays ordered to reduce
        time complexity.

        Args:
            index: The index to add.
            result: The result to store under this index.
        """
        insort_right(self.buffer, (index, result), key=lambda x: self.sorting_key(x[0]))

    def current_index(self) -> Index | None:
        """Get the "lowest" index currently stored in the buffer or None if the buffer is empty."""
        return self.buffer[-1][0] if len(self.buffer) > 0 else None

    @abstractmethod
    async def next_expected_index(self, n: int) -> Index | None:
        """Computes the index that is the next in order that should be retrieved.

        This is required to ensure that there are no gaps in the retrieved results
        due to a subsequent result being ready while the next one in order has
        not been added to the buffer yet.

        Args:
            n: The number of items already retrieved.

        Returns:
            The (n+1)-th index that should be retrieved next.
        """

    async def retrieve_next_in_line(self) -> list[R]:
        """Retrieves the next results that are ready and next in line.

        This method must be the only one used to retrieve results from the
        buffer. If that is the case, it can guarantee that the nth item
        received will have the index returned by `self.next_expected_index(n-1)`.

        Since more than one element may be retrievable, it loops over them
        until the buffer is empty or the next item in the buffer has a higher
        index than expected.

        Returns:
            Results which are the next in line to retrieve according to
            `self.next_expected_index(n-1)`.
        """
        results = []
        while (
            self.current_index() is not None
            and self.current_index() == await self.next_expected_index(self.n_retrieved)
        ):
            results.append(self.buffer.pop()[1])
            self.n_retrieved += 1
        return results


class BufferedOrderer(BufferedOrdererBase[int, R], Generic[R]):
    """An orderer using `int` as index type."""

    def sorting_key(self, index: int) -> Ordered:
        """Sorts indices by descending natural order.

        Args:
            index: The index from which to generate the key.

        Returns:
            A key used for sorting.
        """
        return -index

    async def next_expected_index(self, n: int) -> int:
        """Returns next expected index.

        It assumes that indices count from zero and increment by one.

        Args:
            n: Number of already retrieved results.

        Returns:
            Next index to retrieve.
        """
        return n

"""Tests for helper methods and classes."""

import pytest

from apicadabri.helpers import BufferedOrderer


class TestBufferedOrderer:
    """Tests the `BufferedOrder` class."""

    @pytest.mark.asyncio
    async def test_empty(self) -> None:
        """Hypothesis: Retrieving from an empty orderer returns an empty list."""
        orderer = BufferedOrderer()
        assert await orderer.retrieve_next() == []

    @pytest.mark.asyncio
    async def test_single_next(self) -> None:
        """Hypothesis: Retrieving the next expected item yields only that item."""
        orderer = BufferedOrderer()
        orderer.insert_in_order(0, "foo")
        assert await orderer.retrieve_next() == ["foo"]

    @pytest.mark.asyncio
    async def test_single_next_nothing_left(self) -> None:
        """Hypothesis: Retrieving again after last element yields empty list."""
        orderer = BufferedOrderer()
        orderer.insert_in_order(0, "foo")
        await orderer.retrieve_next()
        assert await orderer.retrieve_next() == []

    @pytest.mark.asyncio
    async def test_single_not_next(self) -> None:
        """Hypothesis: Retrieving next returns empty list if next is not found."""
        orderer = BufferedOrderer()
        orderer.insert_in_order(1, "foo")
        assert await orderer.retrieve_next() == []

    @pytest.mark.asyncio
    async def test_multiple_next(self) -> None:
        """Hypothesis: Retrieving multiple next expected items works."""
        orderer = BufferedOrderer()
        orderer.insert_in_order(0, "foo")
        orderer.insert_in_order(1, "bar")
        assert await orderer.retrieve_next() == ["foo", "bar"]

    @pytest.mark.asyncio
    async def test_multiple_next_unordered(self) -> None:
        """Hypothesis: Retrieving multiple elements that were inserted out of order works."""
        orderer = BufferedOrderer()
        orderer.insert_in_order(1, "bar")
        orderer.insert_in_order(0, "foo")
        assert await orderer.retrieve_next() == ["foo", "bar"]

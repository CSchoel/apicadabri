"""Tests related to progress bars."""

from apicadabri import ApicadabriCallArguments


class TestArgumentsSize:
    def test_one_sized_arg(self):
        args = ApicadabriCallArguments(urls=["foo", "bar", "baz"])
        assert args.urls == ["foo", "bar", "baz"]
        # assert len(args) == 3

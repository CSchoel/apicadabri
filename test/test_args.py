from apicadabri import ApicadabriCallArguments


class TestApicadabriArgs:
    def test_len_single(self):
        args = ApicadabriCallArguments(url="http://foo.bar")
        assert len(args) == 1
        assert len(list(args)) == 1

import avalign


def test_version_is_string():
    assert isinstance(avalign.__version__, str)
    assert avalign.__version__


def test_version_in_all():
    assert "__version__" in avalign.__all__

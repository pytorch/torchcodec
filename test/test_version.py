import torchcodec


def test_version():
    assert "+cpu" not in torchcodec.__version__
    assert "+cu" not in torchcodec.__version__

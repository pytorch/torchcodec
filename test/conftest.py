import os
import random

import pytest
import torch


def pytest_configure(config):
    # register an additional marker (see pytest_collection_modifyitems)
    config.addinivalue_line(
        "markers", "needs_cuda: mark for tests that rely on a CUDA device"
    )
    config.addinivalue_line(
        "markers", "needs_xpu: mark for tests that rely on a XPU device"
    )


def pytest_collection_modifyitems(items):
    # This hook is called by pytest after it has collected the tests (google its
    # name to check out its doc!). We can ignore some tests as we see fit here,
    # or add marks, such as a skip mark.

    out_items = []
    for item in items:
        # The needs_[cuda|xpu] mark will exist if the test was explicitly decorated
        # with the respective @needs_* decorator. It will also exist if it was
        # parametrized with a parameter that has the mark: for example if a test
        # is parametrized with
        # @pytest.mark.parametrize('device', cpu_and_accelerators())
        # the "instances" of the tests where device == 'cuda' will have the
        # 'needs_cuda' mark, and the ones with device == 'cpu' won't have the
        # mark.
        needs_cuda = item.get_closest_marker("needs_cuda") is not None
        needs_xpu = item.get_closest_marker("needs_xpu") is not None

        if (
            needs_cuda
            and not torch.cuda.is_available()
            and os.environ.get("FAIL_WITHOUT_CUDA") is None
        ):
            # We skip CUDA tests on non-CUDA machines, but only if the
            # FAIL_WITHOUT_CUDA env var wasn't set. If it's set, the test will
            # typically fail with a "Unsupported device: cuda" error. This is
            # normal and desirable: this env var is set on CI jobs that are
            # supposed to run the CUDA tests, so if CUDA isn't available on
            # those for whatever reason, we need to know.
            item.add_marker(pytest.mark.skip(reason="CUDA not available."))

        if (
            needs_xpu
            and not torch.xpu.is_available()
            and os.environ.get("FAIL_WITHOUT_XPU") is None
        ):
            item.add_marker(pytest.mark.skip(reason="XPU not available."))

        out_items.append(item)

    items[:] = out_items


@pytest.fixture(autouse=True)
def prevent_leaking_rng():
    # Prevent each test from leaking the rng to all other test when they call
    # torch.manual_seed() or random.seed().

    torch_rng_state = torch.get_rng_state()
    builtin_rng_state = random.getstate()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    if torch.xpu.is_available():
        xpu_rng_state = torch.xpu.get_rng_state()

    yield

    torch.set_rng_state(torch_rng_state)
    random.setstate(builtin_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    if torch.xpu.is_available():
        torch.xpu.set_rng_state(xpu_rng_state)

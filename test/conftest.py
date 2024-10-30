import random

import pytest
import torch


@pytest.fixture(autouse=True)
def prevent_leaking_rng():
    # Prevent each test from leaking the rng to all other test when they call
    # torch.manual_seed() or random.seed().

    torch_rng_state = torch.get_rng_state()
    builtin_rng_state = random.getstate()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()

    yield

    torch.set_rng_state(torch_rng_state)
    random.setstate(builtin_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)


def pytest_configure(config):
    # register an additional marker (see pytest_collection_modifyitems)
    config.addinivalue_line(
        "markers", "needs_cuda: mark for tests that rely on a CUDA device"
    )

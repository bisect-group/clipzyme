import pytest
import torch

REAL_WEIGHTS_TESTS = {"test_cache_vectors_equal_in_loop_esm2_on_real_sequences"}


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device; run by verify_equivalence.sh --gpu")
    config.addinivalue_line(
        "markers", "real_weights: needs the downloaded ESM-2 650M weights; not run by verify_equivalence.sh"
    )


def pytest_collection_modifyitems(config, items):
    no_cuda = not torch.cuda.is_available()
    for item in items:
        if item.name in REAL_WEIGHTS_TESTS:
            item.add_marker(pytest.mark.real_weights)
        if no_cuda and "gpu" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="no CUDA device visible"))

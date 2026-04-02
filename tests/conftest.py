"""Shared pytest configuration for the akatsuki test suite."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")
    config.addinivalue_line("markers", "slow: long-running integration test")
    config.addinivalue_line("markers", "vlm: test specific to a vision-language model")
    config.addinivalue_line("markers", "moe: test specific to a MoE model")

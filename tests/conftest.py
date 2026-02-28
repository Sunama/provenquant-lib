"""Pytest configuration to import the package without installation."""
from __future__ import annotations
import multiprocessing as mp

def pytest_configure(config):
    """Configure multiprocessing to use 'spawn' to avoid fork() related issues."""
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # In case the start method was already set elsewhere
        pass
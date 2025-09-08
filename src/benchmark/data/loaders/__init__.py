"""
Data loaders for the LLM Cybersecurity Benchmark system.

This package contains data loaders for various data sources including
local files, remote datasets, and streaming data.
"""

from .base_loader import DataLoader
from .local_loader import LocalFileDataLoader

__all__ = [
    "DataLoader",
    "LocalFileDataLoader",
]

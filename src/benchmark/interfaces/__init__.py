"""
Interfaces package for the LLM Cybersecurity Benchmark system.

This package contains abstract interfaces and data models used throughout
the benchmarking framework.
"""

from .model_interfaces import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    LoadedModel,
    ModelConfig,
    ModelInfo,
    ModelPlugin,
    ModelResourceMonitor,
    PerformanceMetrics,
    PluginRegistry,
    Prediction,
)

__all__ = [
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "LoadedModel",
    "ModelConfig",
    "ModelInfo",
    "ModelPlugin",
    "ModelResourceMonitor",
    "PerformanceMetrics",
    "PluginRegistry",
    "Prediction",
]

"""
Model plugins for the LLM Cybersecurity Benchmark system.

This module provides plugin implementations for different model types
and inference backends.
"""

from benchmark.models.plugins.anthropic_api import AnthropicModelPlugin
from benchmark.models.plugins.mlx_local import MLXModelPlugin
from benchmark.models.plugins.ollama_local import OllamaModelPlugin
from benchmark.models.plugins.openai_api import OpenAIModelPlugin

__all__ = [
    "AnthropicModelPlugin",
    "MLXModelPlugin",
    "OllamaModelPlugin",
    "OpenAIModelPlugin",
]

"""
Evaluation module for benchmark evaluation results.

This module provides comprehensive evaluation result storage, querying,
and analysis capabilities with SQLite backend storage.
"""

from .comparison_engine import ModelComparisonEngine
from .result_models import EvaluationResult, ModelPerformanceHistory, ResultsQuery
from .results_analyzer import ResultsAnalyzer
from .results_storage import DataIntegrityError, ResultsStorage, StorageError

__all__ = [
    "EvaluationResult",
    "ModelPerformanceHistory",
    "ResultsQuery",
    "ResultsStorage",
    "StorageError",
    "DataIntegrityError",
    "ResultsAnalyzer",
    "ModelComparisonEngine",
]

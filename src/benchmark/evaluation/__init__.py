"""
Evaluation module for benchmark evaluation results.

This module provides comprehensive evaluation result storage, querying,
analysis capabilities, and clean API interfaces for external integration.
"""

from .api_models import (
    APIError,
    APIResponse,
    AvailableEvaluatorsResponse,
    EvaluationConfig,
    EvaluationProgress,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultsResponse,
    EvaluationStatus,
    EvaluationStatusResponse,
    EvaluationType,
    ValidationError,
    ValidationLevel,
)
from .api_models import (
    ResourceLimits as APIResourceLimits,
)
from .batch_evaluator import (
    BatchEvaluator,
    BatchResult,
    BatchStatus,
    ResourceLimits,
    ResourceMonitor,
)
from .comparison_engine import ModelComparisonEngine
from .evaluation_api import EvaluationAPI
from .evaluation_workflow import EvaluationWorkflow, WorkflowProgress, WorkflowState, WorkflowStep
from .result_models import EvaluationResult, ModelPerformanceHistory, ResultsQuery
from .results_analyzer import ResultsAnalyzer
from .results_storage import DataIntegrityError, ResultsStorage, StorageError

__all__ = [
    # Core evaluation components
    "EvaluationResult",
    "ModelPerformanceHistory",
    "ResultsQuery",
    "ResultsStorage",
    "StorageError",
    "DataIntegrityError",
    "ResultsAnalyzer",
    "ModelComparisonEngine",
    # Workflow and batch processing
    "EvaluationWorkflow",
    "WorkflowState",
    "WorkflowStep",
    "WorkflowProgress",
    "BatchEvaluator",
    "BatchResult",
    "BatchStatus",
    "ResourceLimits",
    "ResourceMonitor",
    # API interface and models
    "EvaluationAPI",
    "EvaluationRequest",
    "EvaluationResponse",
    "EvaluationStatusResponse",
    "EvaluationResultsResponse",
    "AvailableEvaluatorsResponse",
    "APIError",
    "APIResponse",
    "EvaluationStatus",
    "EvaluationType",
    "EvaluationConfig",
    "EvaluationProgress",
    "APIResourceLimits",
    "ValidationError",
    "ValidationLevel",
]

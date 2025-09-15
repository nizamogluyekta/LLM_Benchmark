"""
Interfaces and data structures for the evaluation service.

This module provides the core interfaces and data structures for evaluating
model predictions using various metrics through a plugin architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Enumeration of available evaluation metric types."""

    ACCURACY = "accuracy"
    EXPLAINABILITY = "explainability"
    PERFORMANCE = "performance"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    CONFUSION_MATRIX = "confusion_matrix"


@dataclass
class EvaluationRequest:
    """Request for evaluating model predictions against ground truth."""

    experiment_id: str
    model_id: str
    dataset_id: str
    predictions: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]]
    metrics: list[MetricType]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate the evaluation request after initialization."""
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if not self.predictions:
            raise ValueError("predictions cannot be empty")
        if not self.ground_truth:
            raise ValueError("ground_truth cannot be empty")
        if not self.metrics:
            raise ValueError("metrics list cannot be empty")


@dataclass
class EvaluationResult:
    """Result of an evaluation operation."""

    experiment_id: str
    model_id: str
    dataset_id: str
    metrics: dict[str, float]
    detailed_results: dict[str, Any]
    execution_time_seconds: float
    timestamp: str
    metadata: dict[str, Any]
    success: bool = True
    error_message: str | None = None

    def get_metric_value(self, metric_name: str) -> float | None:
        """Get a specific metric value by name."""
        return self.metrics.get(metric_name)

    def get_detailed_result(self, metric_type: str) -> dict[str, Any] | None:
        """Get detailed results for a specific metric type."""
        return self.detailed_results.get(metric_type)


@dataclass
class MetricConfiguration:
    """Configuration for a specific metric evaluator."""

    metric_type: MetricType
    parameters: dict[str, Any]
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a dictionary")


class MetricEvaluator(ABC):
    """Base interface for all metric evaluators."""

    @abstractmethod
    async def evaluate(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            Dictionary mapping metric names to values

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If evaluation fails
        """
        pass

    @abstractmethod
    def get_metric_names(self) -> list[str]:
        """
        Get list of metrics this evaluator produces.

        Returns:
            List of metric names as strings
        """
        pass

    @abstractmethod
    def get_required_prediction_fields(self) -> list[str]:
        """
        Get required fields in prediction data.

        Returns:
            List of required field names in prediction dictionaries
        """
        pass

    @abstractmethod
    def get_required_ground_truth_fields(self) -> list[str]:
        """
        Get required fields in ground truth data.

        Returns:
            List of required field names in ground truth dictionaries
        """
        pass

    @abstractmethod
    def get_metric_type(self) -> MetricType:
        """
        Get the metric type this evaluator handles.

        Returns:
            MetricType enum value
        """
        pass

    def validate_data_compatibility(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> bool:
        """
        Validate that prediction and ground truth data are compatible.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            True if data is compatible, False otherwise
        """
        if len(predictions) != len(ground_truth):
            return False

        # Check required fields in predictions
        pred_fields = self.get_required_prediction_fields()
        for pred in predictions:
            if not all(field in pred for field in pred_fields):
                return False

        # Check required fields in ground truth
        gt_fields = self.get_required_ground_truth_fields()
        return all(all(field in gt for field in gt_fields) for gt in ground_truth)

    def get_evaluator_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about this evaluator.

        Returns:
            Dictionary with evaluator metadata
        """
        return {
            "metric_type": self.get_metric_type().value,
            "metric_names": self.get_metric_names(),
            "required_prediction_fields": self.get_required_prediction_fields(),
            "required_ground_truth_fields": self.get_required_ground_truth_fields(),
            "evaluator_class": self.__class__.__name__,
        }


class EvaluationProgressCallback(ABC):
    """Interface for tracking evaluation progress."""

    @abstractmethod
    async def on_evaluation_started(self, request: EvaluationRequest) -> None:
        """Called when evaluation starts."""
        pass

    @abstractmethod
    async def on_metric_completed(self, metric_type: MetricType, result: dict[str, float]) -> None:
        """Called when a single metric evaluation completes."""
        pass

    @abstractmethod
    async def on_evaluation_completed(self, result: EvaluationResult) -> None:
        """Called when entire evaluation completes."""
        pass

    @abstractmethod
    async def on_evaluation_error(self, error: Exception) -> None:
        """Called when evaluation encounters an error."""
        pass


class EvaluationFilter(ABC):
    """Interface for filtering evaluation results."""

    @abstractmethod
    def applies_to(self, result: EvaluationResult) -> bool:
        """Check if this filter applies to the given result."""
        pass

    @abstractmethod
    def filter_result(self, result: EvaluationResult) -> EvaluationResult:
        """Apply filter transformations to the result."""
        pass


@dataclass
class EvaluationSummary:
    """Summary statistics for a collection of evaluation results."""

    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    average_execution_time: float
    metric_summaries: dict[str, dict[str, float]]
    time_range: dict[str, str]
    models_evaluated: list[str]
    datasets_evaluated: list[str]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_evaluations == 0:
            return 0.0
        return (self.successful_evaluations / self.total_evaluations) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as a percentage."""
        return 100.0 - self.success_rate

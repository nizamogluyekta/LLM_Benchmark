"""
Data models for evaluation results.

This module defines the core data structures for storing and managing
evaluation results with comprehensive metadata and validation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EvaluationResult:
    """
    Comprehensive evaluation result with metadata and raw responses.

    This dataclass encapsulates all information from an evaluation run,
    including metrics, configuration, timing, and raw model responses.
    """

    evaluation_id: str
    model_name: str
    task_type: str
    dataset_name: str
    metrics: dict[str, float]
    timestamp: datetime
    configuration: dict[str, Any]
    raw_responses: list[dict[str, Any]]
    processing_time: float

    # Optional metadata fields
    model_version: str | None = None
    dataset_version: str | None = None
    experiment_name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    error_count: int = 0
    success_rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate evaluation result data integrity."""
        if not self.evaluation_id:
            raise ValueError("evaluation_id cannot be empty")

        if not self.model_name:
            raise ValueError("model_name cannot be empty")

        if not self.task_type:
            raise ValueError("task_type cannot be empty")

        if not self.dataset_name:
            raise ValueError("dataset_name cannot be empty")

        if self.processing_time < 0:
            raise ValueError("processing_time cannot be negative")

        if not isinstance(self.metrics, dict):
            raise ValueError("metrics must be a dictionary")

        if not isinstance(self.raw_responses, list):
            raise ValueError("raw_responses must be a list")

        if not isinstance(self.configuration, dict):
            raise ValueError("configuration must be a dictionary")

        # Validate success rate is between 0 and 1
        if not 0 <= self.success_rate <= 1:
            raise ValueError("success_rate must be between 0 and 1")

        # Validate error count is non-negative
        if self.error_count < 0:
            raise ValueError("error_count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "configuration": self.configuration,
            "raw_responses": self.raw_responses,
            "processing_time": self.processing_time,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "experiment_name": self.experiment_name,
            "tags": self.tags,
            "notes": self.notes,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        """Create EvaluationResult from dictionary."""
        # Parse timestamp if it's a string
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            evaluation_id=data["evaluation_id"],
            model_name=data["model_name"],
            task_type=data["task_type"],
            dataset_name=data["dataset_name"],
            metrics=data["metrics"],
            timestamp=timestamp,
            configuration=data["configuration"],
            raw_responses=data["raw_responses"],
            processing_time=data["processing_time"],
            model_version=data.get("model_version"),
            dataset_version=data.get("dataset_version"),
            experiment_name=data.get("experiment_name"),
            tags=data.get("tags", []),
            notes=data.get("notes"),
            error_count=data.get("error_count", 0),
            success_rate=data.get("success_rate", 1.0),
        )

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationResult":
        """Create EvaluationResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_primary_metric(self) -> float | None:
        """Get the primary metric value for this evaluation."""
        # Look for common primary metrics in order of preference
        primary_metrics = [
            "accuracy",
            "f1_score",
            "f1",
            "auc",
            "rouge_l",
            "bleu",
            "exact_match",
            "precision",
            "recall",
        ]

        for metric in primary_metrics:
            if metric in self.metrics:
                return self.metrics[metric]

        # Return first metric if no primary metric found
        if self.metrics:
            return next(iter(self.metrics.values()))

        return None

    def add_tag(self, tag: str) -> None:
        """Add a tag to this evaluation result."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this evaluation result."""
        if tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if this evaluation result has a specific tag."""
        return tag in self.tags


@dataclass
class ModelPerformanceHistory:
    """Performance history for a specific model across evaluations."""

    model_name: str
    evaluations: list[EvaluationResult] = field(default_factory=list)

    def add_evaluation(self, result: EvaluationResult) -> None:
        """Add an evaluation result to this model's history."""
        if result.model_name != self.model_name:
            raise ValueError(
                f"Evaluation model_name '{result.model_name}' does not match history model_name '{self.model_name}'"
            )

        self.evaluations.append(result)
        # Sort by timestamp to maintain chronological order
        self.evaluations.sort(key=lambda x: x.timestamp)

    def get_latest_result(self) -> EvaluationResult | None:
        """Get the most recent evaluation result."""
        return self.evaluations[-1] if self.evaluations else None

    def get_best_result(self, metric_name: str = "accuracy") -> EvaluationResult | None:
        """Get the evaluation with the best performance for a given metric."""
        if not self.evaluations:
            return None

        valid_evaluations = [e for e in self.evaluations if metric_name in e.metrics]
        if not valid_evaluations:
            return None

        return max(valid_evaluations, key=lambda x: x.metrics[metric_name])

    def get_average_metric(self, metric_name: str) -> float | None:
        """Get the average value for a specific metric across all evaluations."""
        values = [e.metrics[metric_name] for e in self.evaluations if metric_name in e.metrics]
        return sum(values) / len(values) if values else None

    def get_metric_trend(self, metric_name: str) -> list[tuple[datetime, float]]:
        """Get metric values over time for trend analysis."""
        return [
            (e.timestamp, e.metrics[metric_name])
            for e in self.evaluations
            if metric_name in e.metrics
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert performance history to dictionary."""
        latest_result = self.get_latest_result()
        return {
            "model_name": self.model_name,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "evaluation_count": len(self.evaluations),
            "latest_evaluation": latest_result.timestamp.isoformat() if latest_result else None,
        }


@dataclass
class ResultsQuery:
    """Query parameters for filtering evaluation results."""

    evaluation_id: str | None = None
    model_name: str | None = None
    task_type: str | None = None
    dataset_name: str | None = None
    experiment_name: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    tags: list[str] = field(default_factory=list)
    min_success_rate: float | None = None
    metric_filters: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )  # metric_name: (min_value, max_value)
    limit: int | None = None
    offset: int = 0
    sort_by: str = "timestamp"  # timestamp, model_name, success_rate, or metric name
    sort_order: str = "desc"  # asc or desc

    def to_dict(self) -> dict[str, Any]:
        """Convert query to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "experiment_name": self.experiment_name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "tags": self.tags,
            "min_success_rate": self.min_success_rate,
            "metric_filters": self.metric_filters,
            "limit": self.limit,
            "offset": self.offset,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
        }

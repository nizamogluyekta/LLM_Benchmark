"""
API data models for evaluation service integration.

This module provides comprehensive request/response models for the evaluation API,
including validation, serialization, and documentation structures.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .result_models import EvaluationResult


class EvaluationStatus(Enum):
    """Status of an evaluation request."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class EvaluationType(Enum):
    """Types of evaluations that can be performed."""

    SINGLE_MODEL = "single_model"
    MODEL_COMPARISON = "model_comparison"
    BASELINE_EVALUATION = "baseline_evaluation"
    COMPREHENSIVE = "comprehensive"
    BATCH_PROCESSING = "batch_processing"


class ValidationLevel(Enum):
    """Levels of input validation."""

    STRICT = "strict"  # Full validation with all checks
    MODERATE = "moderate"  # Essential validation only
    PERMISSIVE = "permissive"  # Minimal validation


@dataclass
class EvaluationConfig:
    """Configuration parameters for evaluation execution."""

    batch_size: int = 32
    max_length: int | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    timeout_seconds: int | None = None
    validation_level: ValidationLevel = ValidationLevel.MODERATE

    def validate(self) -> list[str]:
        """Validate configuration parameters."""
        errors = []

        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.max_length is not None and self.max_length <= 0:
            errors.append("max_length must be positive if specified")
        if not 0.0 <= self.temperature <= 2.0:
            errors.append("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            errors.append("top_p must be between 0.0 and 1.0")
        if self.top_k is not None and self.top_k <= 0:
            errors.append("top_k must be positive if specified")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive if specified")

        return errors


@dataclass
class ResourceLimits:
    """Resource limits for evaluation execution."""

    max_memory_mb: int | None = None
    max_cpu_percent: float | None = None
    max_gpu_memory_mb: int | None = None
    max_concurrent_evaluations: int = 4
    max_execution_time_seconds: int | None = None

    def validate(self) -> list[str]:
        """Validate resource limits."""
        errors = []

        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive if specified")
        if self.max_cpu_percent is not None and not 0.0 < self.max_cpu_percent <= 100.0:
            errors.append("max_cpu_percent must be between 0.0 and 100.0")
        if self.max_gpu_memory_mb is not None and self.max_gpu_memory_mb <= 0:
            errors.append("max_gpu_memory_mb must be positive if specified")
        if self.max_concurrent_evaluations <= 0:
            errors.append("max_concurrent_evaluations must be positive")
        if self.max_execution_time_seconds is not None and self.max_execution_time_seconds <= 0:
            errors.append("max_execution_time_seconds must be positive if specified")

        return errors


@dataclass
class EvaluationRequest:
    """Request to start a new evaluation."""

    # Required fields
    model_names: list[str]
    task_types: list[str]

    # Optional fields with defaults
    dataset_names: list[str] | None = None
    evaluation_type: EvaluationType = EvaluationType.SINGLE_MODEL
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    requested_by: str | None = None
    priority: int = 0  # Higher values = higher priority
    tags: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate the evaluation request."""
        errors = []

        # Validate required fields
        if not self.model_names:
            errors.append("model_names cannot be empty")
        if not self.task_types:
            errors.append("task_types cannot be empty")

        # Validate model names
        for model_name in self.model_names:
            if not model_name or not isinstance(model_name, str):
                errors.append(f"Invalid model name: {model_name}")

        # Validate task types
        for task_type in self.task_types:
            if not task_type or not isinstance(task_type, str):
                errors.append(f"Invalid task type: {task_type}")

        # Validate dataset names if provided
        if self.dataset_names is not None:
            for dataset_name in self.dataset_names:
                if not dataset_name or not isinstance(dataset_name, str):
                    errors.append(f"Invalid dataset name: {dataset_name}")

        # Validate evaluation type compatibility
        if self.evaluation_type == EvaluationType.MODEL_COMPARISON and len(self.model_names) < 2:
            errors.append("MODEL_COMPARISON requires at least 2 models")
        if self.evaluation_type == EvaluationType.BASELINE_EVALUATION and len(self.model_names) < 2:
            errors.append("BASELINE_EVALUATION requires at least 2 models")

        # Validate priority
        if not isinstance(self.priority, int) or self.priority < 0:
            errors.append("priority must be a non-negative integer")

        # Validate nested configurations
        errors.extend(self.evaluation_config.validate())
        errors.extend(self.resource_limits.validate())

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert request to dictionary representation."""
        return {
            "request_id": self.request_id,
            "model_names": self.model_names,
            "task_types": self.task_types,
            "dataset_names": self.dataset_names,
            "evaluation_type": self.evaluation_type.value,
            "evaluation_config": {
                "batch_size": self.evaluation_config.batch_size,
                "max_length": self.evaluation_config.max_length,
                "temperature": self.evaluation_config.temperature,
                "top_p": self.evaluation_config.top_p,
                "top_k": self.evaluation_config.top_k,
                "seed": self.evaluation_config.seed,
                "timeout_seconds": self.evaluation_config.timeout_seconds,
                "validation_level": self.evaluation_config.validation_level.value,
            },
            "resource_limits": {
                "max_memory_mb": self.resource_limits.max_memory_mb,
                "max_cpu_percent": self.resource_limits.max_cpu_percent,
                "max_gpu_memory_mb": self.resource_limits.max_gpu_memory_mb,
                "max_concurrent_evaluations": self.resource_limits.max_concurrent_evaluations,
                "max_execution_time_seconds": self.resource_limits.max_execution_time_seconds,
            },
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "requested_by": self.requested_by,
            "priority": self.priority,
            "tags": self.tags,
        }


@dataclass
class EvaluationProgress:
    """Progress information for running evaluations."""

    total_steps: int
    completed_steps: int
    current_step_description: str
    estimated_completion_time: datetime | None = None
    progress_percentage: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate progress percentage."""
        if self.total_steps > 0:
            self.progress_percentage = (self.completed_steps / self.total_steps) * 100.0
        else:
            self.progress_percentage = 0.0


@dataclass
class EvaluationResponse:
    """Response from starting an evaluation."""

    evaluation_id: str
    request_id: str
    status: EvaluationStatus
    message: str
    created_at: datetime = field(default_factory=datetime.now)
    estimated_completion_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary representation."""
        return {
            "evaluation_id": self.evaluation_id,
            "request_id": self.request_id,
            "status": self.status.value,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "estimated_completion_time": (
                self.estimated_completion_time.isoformat()
                if self.estimated_completion_time
                else None
            ),
        }


@dataclass
class EvaluationStatusResponse:
    """Response containing evaluation status information."""

    evaluation_id: str
    status: EvaluationStatus
    progress: EvaluationProgress | None
    start_time: datetime | None
    end_time: datetime | None
    duration_seconds: float | None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert status response to dictionary."""
        result: dict[str, Any] = {
            "evaluation_id": self.evaluation_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }

        if self.progress:
            result["progress"] = {
                "total_steps": self.progress.total_steps,
                "completed_steps": self.progress.completed_steps,
                "current_step_description": self.progress.current_step_description,
                "progress_percentage": self.progress.progress_percentage,
                "estimated_completion_time": (
                    self.progress.estimated_completion_time.isoformat()
                    if self.progress.estimated_completion_time
                    else None
                ),
            }

        return result


@dataclass
class EvaluationResultsResponse:
    """Response containing evaluation results."""

    evaluation_id: str
    status: EvaluationStatus
    results: list[EvaluationResult]
    summary: dict[str, Any]
    analysis: dict[str, Any] | None = None
    comparison_results: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert results response to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "status": self.status.value,
            "results": [
                {
                    "evaluation_id": result.evaluation_id,
                    "model_name": result.model_name,
                    "task_type": result.task_type,
                    "dataset_name": result.dataset_name,
                    "metrics": result.metrics,
                    "timestamp": result.timestamp.isoformat(),
                    "processing_time": result.processing_time,
                    "experiment_name": result.experiment_name,
                    "tags": result.tags,
                }
                for result in self.results
            ],
            "summary": self.summary,
            "analysis": self.analysis,
            "comparison_results": self.comparison_results,
            "metadata": self.metadata,
        }


@dataclass
class AvailableEvaluatorsResponse:
    """Response listing available evaluation types and capabilities."""

    evaluation_types: list[str]
    supported_tasks: dict[str, list[str]]  # task_type -> list of available evaluators
    supported_metrics: list[str]
    model_requirements: dict[str, Any]  # Requirements for each evaluation type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "evaluation_types": self.evaluation_types,
            "supported_tasks": self.supported_tasks,
            "supported_metrics": self.supported_metrics,
            "model_requirements": self.model_requirements,
        }


@dataclass
class ValidationError:
    """Represents a validation error."""

    field: str
    message: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field": self.field,
            "message": self.message,
            "value": self.value,
        }


@dataclass
class APIError:
    """Represents an API error response."""

    error_code: str
    error_message: str
    details: dict[str, Any] | None = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "details": self.details,
            "validation_errors": [error.to_dict() for error in self.validation_errors],
            "timestamp": self.timestamp.isoformat(),
        }


# Type aliases for better readability
EvaluationID = str
RequestID = str
ModelName = str
TaskType = str
DatasetName = str

# Response type unions
APIResponse = (
    EvaluationResponse
    | EvaluationStatusResponse
    | EvaluationResultsResponse
    | AvailableEvaluatorsResponse
    | APIError
)

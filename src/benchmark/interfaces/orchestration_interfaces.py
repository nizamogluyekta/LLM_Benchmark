"""Interfaces for orchestration service."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExperimentStatus(Enum):
    """Status of experiment execution."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentProgress:
    """Progress information for an experiment."""

    experiment_id: str
    status: ExperimentStatus
    current_step: str
    total_steps: int
    completed_steps: int
    percentage: float
    elapsed_time_seconds: float
    estimated_remaining_seconds: float | None
    error_message: str | None = None


@dataclass
class ExperimentResult:
    """Complete result of an experiment."""

    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    started_at: str
    completed_at: str | None
    total_duration_seconds: float
    models_evaluated: list[str]
    datasets_used: list[str]
    evaluation_results: dict[str, Any]
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """Context passed between workflow steps."""

    experiment_id: str
    config: dict[str, Any]
    services: dict[str, Any]
    step_results: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False


class WorkflowStep(ABC):
    """Abstract base class for workflow steps."""

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute the workflow step.

        Args:
            context: Workflow context containing config, services, and results

        Returns:
            Dictionary with step execution results
        """
        pass

    @abstractmethod
    def get_step_name(self) -> str:
        """Get human-readable step name."""
        pass

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Get list of required service dependencies."""
        pass

    def get_estimated_duration_seconds(self) -> float | None:
        """Get estimated duration for this step in seconds."""
        return None


class OrchestrationInterface(ABC):
    """Interface for orchestration service."""

    @abstractmethod
    async def create_experiment(self, config_path: str, experiment_name: str | None = None) -> str:
        """Create a new experiment.

        Args:
            config_path: Path to experiment configuration
            experiment_name: Optional experiment name

        Returns:
            Experiment ID
        """
        pass

    @abstractmethod
    async def start_experiment(self, experiment_id: str, background: bool = True) -> dict[str, Any]:
        """Start experiment execution.

        Args:
            experiment_id: ID of experiment to start
            background: Whether to run in background

        Returns:
            Execution status and metadata
        """
        pass

    @abstractmethod
    async def get_experiment_progress(self, experiment_id: str) -> ExperimentProgress:
        """Get current experiment progress.

        Args:
            experiment_id: ID of experiment

        Returns:
            Current progress information
        """
        pass

    @abstractmethod
    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment.

        Args:
            experiment_id: ID of experiment to cancel

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    async def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """Get experiment result.

        Args:
            experiment_id: ID of experiment

        Returns:
            Complete experiment result
        """
        pass

    @abstractmethod
    async def list_experiments(
        self, status_filter: ExperimentStatus | None = None
    ) -> list[dict[str, Any]]:
        """List experiments with optional status filter.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of experiment summaries
        """
        pass

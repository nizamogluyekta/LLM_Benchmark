"""
Model interfaces for the LLM Cybersecurity Benchmark system.

This module defines the abstract interfaces for model plugins and related
components in the benchmarking framework.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from benchmark.core.base import ServiceResponse


class Prediction(BaseModel):
    """A prediction result from a model with comprehensive metadata."""

    sample_id: str = Field(..., description="Unique identifier for the input sample")
    input_text: str = Field(..., description="Original input text that was processed")
    prediction: str = Field(..., description="Model prediction: 'ATTACK' or 'BENIGN'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    attack_type: str | None = Field(
        None, description="Specific attack type if prediction is ATTACK"
    )
    explanation: str | None = Field(None, description="Model explanation for the prediction")
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Time taken for inference in milliseconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional prediction metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    model_version: str | None = Field(None, description="Model version used for prediction")

    model_config = {"use_enum_values": True}


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (mlx, api, ollama, etc.)")
    version: str | None = Field(None, description="Model version")
    description: str | None = Field(None, description="Model description")
    capabilities: list[str] = Field(default_factory=list, description="Model capabilities")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Current memory usage in MB")
    status: str = Field(default="loaded", description="Model status")
    loaded_at: datetime = Field(default_factory=datetime.now, description="Model load timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional model metadata")

    model_config = {"use_enum_values": True}


class PerformanceMetrics(BaseModel):
    """Performance metrics for a model."""

    model_id: str = Field(..., description="Model identifier")
    total_predictions: int = Field(default=0, ge=0, description="Total number of predictions made")
    total_inference_time_ms: float = Field(default=0.0, ge=0.0, description="Total inference time")
    average_inference_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average inference time"
    )
    predictions_per_second: float = Field(default=0.0, ge=0.0, description="Predictions per second")
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Current memory usage")
    peak_memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Peak memory usage")
    error_count: int = Field(default=0, ge=0, description="Number of inference errors")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate")
    last_prediction_at: datetime | None = Field(None, description="Last prediction timestamp")
    metrics_collected_at: datetime = Field(
        default_factory=datetime.now, description="Metrics collection time"
    )

    model_config = {"use_enum_values": True}


class LoadedModel(BaseModel):
    """Information about a loaded model instance."""

    model_id: str = Field(..., description="Unique model identifier")
    plugin: Any = Field(..., description="Model plugin instance")
    config: dict[str, Any] = Field(..., description="Model configuration")
    info: ModelInfo = Field(..., description="Model information")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_accessed_at: datetime = Field(
        default_factory=datetime.now, description="Last access timestamp"
    )

    model_config = {"arbitrary_types_allowed": True, "use_enum_values": True}


class ModelPlugin(ABC):
    """Abstract base class for model plugins."""

    @abstractmethod
    async def initialize(self, config: dict[str, Any]) -> ServiceResponse:
        """
        Initialize the model plugin with the given configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            ServiceResponse indicating success or failure
        """
        pass

    @abstractmethod
    async def predict(self, samples: list[str]) -> list[Prediction]:
        """
        Make predictions on a batch of samples.

        Args:
            samples: List of input text samples

        Returns:
            List of prediction results
        """
        pass

    @abstractmethod
    async def explain(self, sample: str) -> str:
        """
        Generate an explanation for a prediction on a single sample.

        Args:
            sample: Input text sample

        Returns:
            Explanation string
        """
        pass

    @abstractmethod
    async def get_model_info(self) -> ModelInfo:
        """
        Get information about the model.

        Returns:
            ModelInfo object with model details
        """
        pass

    @abstractmethod
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics for the model.

        Returns:
            PerformanceMetrics object
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the model.

        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        pass


class ModelResourceMonitor:
    """Monitor model resource usage."""

    def __init__(self) -> None:
        self.model_metrics: dict[str, PerformanceMetrics] = {}
        self.system_metrics: dict[str, Any] = {}

    async def track_inference(
        self, model_id: str, inference_time_ms: float, success: bool = True
    ) -> None:
        """Track an inference operation."""
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = PerformanceMetrics(
                model_id=model_id, last_prediction_at=None
            )

        metrics = self.model_metrics[model_id]
        metrics.total_predictions += 1
        metrics.total_inference_time_ms += inference_time_ms
        metrics.average_inference_time_ms = (
            metrics.total_inference_time_ms / metrics.total_predictions
        )
        metrics.predictions_per_second = (
            1000.0 / metrics.average_inference_time_ms
            if metrics.average_inference_time_ms > 0
            else 0.0
        )
        metrics.last_prediction_at = datetime.now()

        if not success:
            metrics.error_count += 1

        metrics.success_rate = (
            metrics.total_predictions - metrics.error_count
        ) / metrics.total_predictions

    async def update_memory_usage(self, model_id: str, memory_mb: float) -> None:
        """Update memory usage for a model."""
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = PerformanceMetrics(
                model_id=model_id, last_prediction_at=None
            )

        metrics = self.model_metrics[model_id]
        metrics.memory_usage_mb = memory_mb
        if memory_mb > metrics.peak_memory_usage_mb:
            metrics.peak_memory_usage_mb = memory_mb

    async def get_metrics(self, model_id: str) -> PerformanceMetrics | None:
        """Get metrics for a specific model."""
        return self.model_metrics.get(model_id)

    async def get_all_metrics(self) -> dict[str, PerformanceMetrics]:
        """Get metrics for all models."""
        return self.model_metrics.copy()

    async def reset_metrics(self, model_id: str) -> None:
        """Reset metrics for a specific model."""
        if model_id in self.model_metrics:
            del self.model_metrics[model_id]

    async def cleanup(self) -> None:
        """Clean up the resource monitor."""
        self.model_metrics.clear()
        self.system_metrics.clear()


class BatchInferenceRequest(BaseModel):
    """Request for batch inference."""

    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request ID")
    model_id: str = Field(..., description="Target model ID")
    samples: list[str] = Field(..., description="Input samples for inference")
    batch_size: int = Field(default=32, ge=1, description="Batch processing size")
    include_explanations: bool = Field(default=False, description="Include explanations in results")
    timeout_seconds: float | None = Field(None, ge=0.0, description="Request timeout")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Request metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Request creation time")

    model_config = {"use_enum_values": True}


class BatchInferenceResponse(BaseModel):
    """Response for batch inference."""

    request_id: str = Field(..., description="Original request ID")
    model_id: str = Field(..., description="Model used for inference")
    predictions: list[Prediction] = Field(..., description="Prediction results")
    total_samples: int = Field(..., description="Total number of samples processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(default=0, description="Number of failed predictions")
    total_inference_time_ms: float = Field(..., description="Total processing time")
    average_inference_time_ms: float = Field(..., description="Average time per sample")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    completed_at: datetime = Field(default_factory=datetime.now, description="Completion timestamp")

    model_config = {"use_enum_values": True}


# Type aliases for convenience
ModelConfig = dict[str, Any]
PluginRegistry = dict[str, type]

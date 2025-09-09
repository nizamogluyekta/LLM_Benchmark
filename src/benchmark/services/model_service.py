"""
Model service for the LLM Cybersecurity Benchmark system.

This module provides a comprehensive service for managing different types of LLM models
through a plugin architecture with resource monitoring and performance tracking.
"""

import asyncio
import contextlib
import time
from datetime import datetime
from typing import Any, cast
from uuid import uuid4

import psutil

from benchmark.core.base import BaseService, HealthCheck, ServiceResponse, ServiceStatus
from benchmark.core.exceptions import (
    BenchmarkError,
    ConfigurationError,
    ErrorCode,
    ServiceUnavailableError,
    model_loading_error,
)
from benchmark.core.logging import get_logger
from benchmark.interfaces.model_interfaces import (
    BatchInferenceResponse,
    LoadedModel,
    ModelInfo,
    ModelPlugin,
    ModelResourceMonitor,
    PerformanceMetrics,
    PluginRegistry,
    Prediction,
)


class ModelPerformanceMonitor:
    """Enhanced performance monitor for model operations."""

    def __init__(self) -> None:
        self.resource_monitor = ModelResourceMonitor()
        self.inference_history: dict[str, list[dict[str, Any]]] = {}
        self.system_stats: dict[str, Any] = {}

    async def track_batch_inference(
        self, model_id: str, batch_size: int, total_time_ms: float, successful_count: int
    ) -> None:
        """Track batch inference performance."""
        await self.resource_monitor.track_inference(
            model_id,
            total_time_ms / batch_size if batch_size > 0 else 0,
            successful_count == batch_size,
        )

        # Store detailed inference history
        if model_id not in self.inference_history:
            self.inference_history[model_id] = []

        self.inference_history[model_id].append(
            {
                "timestamp": datetime.now(),
                "batch_size": batch_size,
                "total_time_ms": total_time_ms,
                "successful_count": successful_count,
                "throughput": (successful_count / total_time_ms * 1000) if total_time_ms > 0 else 0,
            }
        )

        # Keep only last 100 entries
        if len(self.inference_history[model_id]) > 100:
            self.inference_history[model_id] = self.inference_history[model_id][-100:]

    async def get_performance_summary(self, model_id: str) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = await self.resource_monitor.get_metrics(model_id)
        if not metrics:
            return {}

        history = self.inference_history.get(model_id, [])
        recent_history = history[-10:] if history else []

        return {
            "basic_metrics": metrics.model_dump(),
            "recent_throughput": [h["throughput"] for h in recent_history],
            "recent_batch_sizes": [h["batch_size"] for h in recent_history],
            "total_batches_processed": len(history),
            "average_batch_throughput": sum(h["throughput"] for h in recent_history)
            / len(recent_history)
            if recent_history
            else 0,
        }


class ModelService(BaseService):
    """
    Service for managing LLM models through a plugin architecture.

    Provides model loading, inference, resource monitoring, and performance tracking
    with support for different model types through plugins.
    """

    def __init__(
        self,
        max_models: int = 10,
        max_memory_mb: int = 8192,
        cleanup_interval_seconds: int = 300,
        enable_performance_monitoring: bool = True,
    ):
        """
        Initialize the model service.

        Args:
            max_models: Maximum number of models to keep loaded
            max_memory_mb: Maximum memory usage threshold in MB
            cleanup_interval_seconds: Interval for automatic cleanup
            enable_performance_monitoring: Enable detailed performance monitoring
        """
        super().__init__("model_service")

        # Core components
        self.plugins: PluginRegistry = {}
        self.loaded_models: dict[str, LoadedModel] = {}

        # Configuration
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_performance_monitoring = enable_performance_monitoring

        # Performance monitoring
        self.performance_monitor = ModelPerformanceMonitor()

        # Async management
        self.cleanup_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        self.logger = get_logger("model_service")

    async def initialize(self) -> ServiceResponse:
        """Initialize the model service."""
        try:
            self.logger.info("Initializing Model Service")

            # Start background cleanup task
            if self.cleanup_interval_seconds > 0:
                self.cleanup_task = asyncio.create_task(self._background_cleanup())

            self._set_status(ServiceStatus.HEALTHY)
            self._mark_initialized()
            self.logger.info("Model Service initialized successfully")

            return ServiceResponse(
                success=True,
                message="Model service initialized",
                data={
                    "max_models": self.max_models,
                    "max_memory_mb": self.max_memory_mb,
                    "performance_monitoring": self.enable_performance_monitoring,
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize model service: {e}")
            self._set_status(ServiceStatus.ERROR)
            raise ServiceUnavailableError(f"Model service initialization failed: {e}") from e

    async def health_check(self) -> HealthCheck:
        """Check the health of the model service."""
        try:
            checks: dict[str, Any] = {}

            # Check service status
            checks["service_status"] = self.status.value
            checks["loaded_models"] = len(self.loaded_models)
            checks["registered_plugins"] = len(self.plugins)

            # Check memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            checks["memory_usage_mb"] = memory_mb
            checks["memory_within_limits"] = memory_mb < self.max_memory_mb

            # Check individual model health
            model_health = {}
            for model_id, loaded_model in self.loaded_models.items():
                try:
                    model_health_status = await loaded_model.plugin.health_check()
                    model_health[model_id] = model_health_status.get("status", "unknown")
                except Exception as e:
                    model_health[model_id] = f"error: {str(e)}"

            checks["model_health"] = model_health

            # Overall health status
            is_healthy = (
                self.status in [ServiceStatus.HEALTHY]
                and len(self.loaded_models) <= self.max_models
                and checks["memory_within_limits"]
                and all(status not in ["error", "failed"] for status in model_health.values())
            )

            status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.DEGRADED

            return HealthCheck(
                status=status,
                message=f"Model service with {len(self.loaded_models)} loaded models",
                checks=checks,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return HealthCheck(
                status=ServiceStatus.ERROR,
                message=f"Health check error: {e}",
                checks={"error": str(e)},
                timestamp=datetime.now().isoformat(),
            )

    async def shutdown(self) -> ServiceResponse:
        """Shutdown the model service and clean up resources."""
        try:
            self.logger.info("Shutting down Model Service")

            # Signal shutdown to background tasks
            self._shutdown_event.set()

            # Cancel cleanup task
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.cleanup_task

            # Clean up all loaded models
            for model_id in list(self.loaded_models.keys()):
                await self.cleanup_model(model_id)

            # Clean up performance monitor
            await self.performance_monitor.resource_monitor.cleanup()

            self._set_status(ServiceStatus.STOPPED)
            self._mark_uninitialized()
            self.logger.info("Model Service shutdown complete")

            return ServiceResponse(
                success=True,
                message="Model service shutdown complete",
                data={"cleaned_up_models": len(self.loaded_models)},
            )

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise ServiceUnavailableError(f"Model service shutdown failed: {e}") from e

    async def register_plugin(self, model_type: str, plugin_class: type) -> ServiceResponse:
        """
        Register a model plugin for a specific model type.

        Args:
            model_type: Type identifier for the model (e.g., 'mlx', 'api', 'ollama')
            plugin_class: Plugin class implementing ModelPlugin interface

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            if not issubclass(plugin_class, ModelPlugin):
                raise ConfigurationError("Plugin class must implement ModelPlugin interface")

            self.plugins[model_type] = plugin_class
            self.logger.info(f"Registered plugin for model type: {model_type}")

            return ServiceResponse(
                success=True,
                message=f"Plugin registered for {model_type}",
                data={"model_type": model_type, "plugin_class": plugin_class.__name__},
            )

        except Exception as e:
            self.logger.error(f"Failed to register plugin for {model_type}: {e}")
            raise ConfigurationError(f"Plugin registration failed: {e}") from e

    async def load_model(self, model_config: dict[str, Any]) -> str:
        """
        Load a model with the given configuration.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Model ID for the loaded model
        """
        model_id = None
        try:
            # Validate configuration
            if "type" not in model_config:
                raise ConfigurationError("Model configuration must include 'type' field")

            model_type = model_config["type"]
            if model_type not in self.plugins:
                raise ConfigurationError(f"No plugin registered for model type: {model_type}")

            # Check resource limits
            if len(self.loaded_models) >= self.max_models:
                await self._cleanup_least_recently_used()

            # Generate model ID
            model_id = f"{model_type}_{uuid4().hex[:8]}"
            model_config["model_id"] = model_id

            # Create plugin instance
            plugin_class = self.plugins[model_type]
            plugin = plugin_class()

            self.logger.info(f"Loading model {model_id} with type {model_type}")

            # Initialize plugin
            init_response = await plugin.initialize(model_config)
            if not init_response.success:
                raise model_loading_error(
                    model_id, f"Plugin initialization failed: {init_response.message}"
                )

            # Get model info
            model_info = await plugin.get_model_info()
            model_info.model_id = model_id

            # Create performance metrics
            performance_metrics = PerformanceMetrics(model_id=model_id, last_prediction_at=None)

            # Create loaded model entry
            loaded_model = LoadedModel(
                model_id=model_id,
                plugin=plugin,
                config=model_config,
                info=model_info,
                performance_metrics=performance_metrics,
            )

            self.loaded_models[model_id] = loaded_model

            # Update memory tracking
            await self.performance_monitor.resource_monitor.update_memory_usage(
                model_id, model_info.memory_usage_mb
            )

            self.logger.info(f"Successfully loaded model {model_id}")

            return model_id

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            if model_id:
                # Clean up partial loading
                await self.cleanup_model(model_id)
            raise model_loading_error(model_config.get("name", "unknown"), str(e)) from e

    async def predict_batch(
        self,
        model_id: str,
        samples: list[str],
        include_explanations: bool = False,
        batch_size: int = 32,
    ) -> BatchInferenceResponse:
        """
        Make batch predictions using the specified model.

        Args:
            model_id: ID of the model to use
            samples: List of input samples
            include_explanations: Whether to include explanations
            batch_size: Batch size for processing

        Returns:
            BatchInferenceResponse with prediction results
        """
        request_id = str(uuid4())
        start_time = time.time()

        try:
            if model_id not in self.loaded_models:
                raise BenchmarkError(
                    f"Model {model_id} not found", error_code=ErrorCode.MODEL_NOT_FOUND
                )

            loaded_model = self.loaded_models[model_id]
            loaded_model.last_accessed_at = datetime.now()

            self.logger.info(
                f"Processing batch inference for {len(samples)} samples with model {model_id}"
            )

            # Process samples in batches
            all_predictions = []
            successful_count = 0
            failed_count = 0

            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]

                try:
                    # Make predictions
                    batch_predictions = await loaded_model.plugin.predict(batch)

                    # Add explanations if requested
                    if include_explanations:
                        for prediction in batch_predictions:
                            if not prediction.explanation:
                                try:
                                    explanation = await loaded_model.plugin.explain(
                                        prediction.input_text
                                    )
                                    prediction.explanation = explanation
                                except Exception as e:
                                    self.logger.warning(f"Failed to generate explanation: {e}")
                                    prediction.explanation = "Explanation not available"

                    all_predictions.extend(batch_predictions)
                    successful_count += len(batch_predictions)

                except Exception as e:
                    self.logger.error(f"Batch prediction failed: {e}")
                    failed_count += len(batch)

                    # Create error predictions for failed samples
                    for j, sample in enumerate(batch):
                        error_prediction = Prediction(
                            sample_id=f"error_{i+j}",
                            input_text=sample,
                            prediction="ERROR",
                            confidence=0.0,
                            attack_type=None,
                            explanation=f"Prediction failed: {str(e)}",
                            inference_time_ms=0.0,
                            metadata={"error": str(e)},
                            model_version=None,
                        )
                        all_predictions.append(error_prediction)

            total_time_ms = (time.time() - start_time) * 1000

            # Update performance metrics
            if self.enable_performance_monitoring:
                await self.performance_monitor.track_batch_inference(
                    model_id, len(samples), total_time_ms, successful_count
                )

            response = BatchInferenceResponse(
                request_id=request_id,
                model_id=model_id,
                predictions=all_predictions,
                total_samples=len(samples),
                successful_predictions=successful_count,
                failed_predictions=failed_count,
                total_inference_time_ms=total_time_ms,
                average_inference_time_ms=total_time_ms / len(samples) if samples else 0.0,
                metadata={"batch_size": batch_size, "include_explanations": include_explanations},
            )

            self.logger.info(
                f"Batch inference completed: {successful_count}/{len(samples)} successful"
            )
            return response

        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            total_time_ms = (time.time() - start_time) * 1000

            return BatchInferenceResponse(
                request_id=request_id,
                model_id=model_id,
                predictions=[],
                total_samples=len(samples),
                successful_predictions=0,
                failed_predictions=len(samples),
                total_inference_time_ms=total_time_ms,
                average_inference_time_ms=0.0,
                metadata={"error": str(e)},
            )

    async def explain_prediction(self, model_id: str, sample: str) -> str:
        """
        Get an explanation for a prediction on a single sample.

        Args:
            model_id: ID of the model to use
            sample: Input sample

        Returns:
            Explanation string
        """
        try:
            if model_id not in self.loaded_models:
                raise BenchmarkError(
                    f"Model {model_id} not found", error_code=ErrorCode.MODEL_NOT_FOUND
                )

            loaded_model = self.loaded_models[model_id]
            loaded_model.last_accessed_at = datetime.now()

            explanation = await loaded_model.plugin.explain(sample)
            return str(explanation)

        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return f"Explanation not available: {str(e)}"

    async def get_model_info(self, model_id: str) -> ModelInfo:
        """
        Get information about a loaded model.

        Args:
            model_id: ID of the model

        Returns:
            ModelInfo object
        """
        if model_id not in self.loaded_models:
            raise BenchmarkError(
                f"Model {model_id} not found", error_code=ErrorCode.MODEL_NOT_FOUND
            )

        loaded_model = self.loaded_models[model_id]
        model_info = await loaded_model.plugin.get_model_info()
        return cast(ModelInfo, model_info)

    async def get_model_performance(self, model_id: str) -> dict[str, Any]:
        """
        Get performance metrics for a model.

        Args:
            model_id: ID of the model

        Returns:
            Performance metrics dictionary
        """
        if model_id not in self.loaded_models:
            raise BenchmarkError(
                f"Model {model_id} not found", error_code=ErrorCode.MODEL_NOT_FOUND
            )

        return await self.performance_monitor.get_performance_summary(model_id)

    async def get_all_models(self) -> dict[str, ModelInfo]:
        """Get information about all loaded models."""
        models = {}
        for model_id, loaded_model in self.loaded_models.items():
            try:
                models[model_id] = await loaded_model.plugin.get_model_info()
            except Exception as e:
                self.logger.warning(f"Failed to get info for model {model_id}: {e}")
        return models

    async def cleanup_model(self, model_id: str) -> ServiceResponse:
        """
        Clean up a loaded model and free resources.

        Args:
            model_id: ID of the model to clean up

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            if model_id not in self.loaded_models:
                return ServiceResponse(
                    success=True,
                    message=f"Model {model_id} not found (already cleaned up?)",
                    data={"model_id": model_id},
                )

            loaded_model = self.loaded_models[model_id]

            # Clean up the plugin
            await loaded_model.plugin.cleanup()

            # Remove from loaded models
            del self.loaded_models[model_id]

            # Clean up performance metrics
            await self.performance_monitor.resource_monitor.reset_metrics(model_id)

            self.logger.info(f"Cleaned up model {model_id}")

            return ServiceResponse(
                success=True,
                message=f"Model {model_id} cleaned up successfully",
                data={"model_id": model_id},
            )

        except Exception as e:
            self.logger.error(f"Failed to cleanup model {model_id}: {e}")
            return ServiceResponse(
                success=False,
                message=f"Model cleanup failed: {e}",
                data={"model_id": model_id, "error": str(e)},
            )

    async def get_service_stats(self) -> dict[str, Any]:
        """Get comprehensive service statistics."""
        stats: dict[str, Any] = {
            "service_status": self.status.value,
            "loaded_models": len(self.loaded_models),
            "registered_plugins": list(self.plugins.keys()),
            "max_models": self.max_models,
            "max_memory_mb": self.max_memory_mb,
        }

        # Add memory usage
        try:
            process = psutil.Process()
            stats["current_memory_mb"] = process.memory_info().rss / 1024 / 1024
        except Exception:
            stats["current_memory_mb"] = 0

        # Add per-model statistics
        model_stats: dict[str, dict[str, Any]] = {}
        for model_id, loaded_model in self.loaded_models.items():
            try:
                performance = await self.performance_monitor.get_performance_summary(model_id)
                model_stats[model_id] = {
                    "type": loaded_model.config.get("type", "unknown"),
                    "loaded_at": loaded_model.created_at.isoformat(),
                    "last_accessed": loaded_model.last_accessed_at.isoformat(),
                    "performance": performance,
                }
            except Exception as e:
                model_stats[model_id] = {"error": str(e)}

        stats["model_statistics"] = model_stats
        return stats

    async def _background_cleanup(self) -> None:
        """Background task for periodic cleanup."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)

                if self._shutdown_event.is_set():
                    break

                # Check memory usage and clean up if needed
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024

                if memory_mb > self.max_memory_mb:
                    self.logger.warning(
                        f"Memory usage {memory_mb:.2f}MB exceeds limit {self.max_memory_mb}MB"
                    )
                    await self._cleanup_least_recently_used()

                # Clean up models that haven't been accessed recently
                cutoff_time = datetime.now().timestamp() - (self.cleanup_interval_seconds * 2)
                models_to_cleanup = []

                for model_id, loaded_model in self.loaded_models.items():
                    if loaded_model.last_accessed_at.timestamp() < cutoff_time:
                        models_to_cleanup.append(model_id)

                for model_id in models_to_cleanup:
                    await self.cleanup_model(model_id)

            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")

    async def _cleanup_least_recently_used(self) -> None:
        """Clean up the least recently used model."""
        if not self.loaded_models:
            return

        # Find the least recently used model
        lru_model_id = min(
            self.loaded_models.keys(), key=lambda mid: self.loaded_models[mid].last_accessed_at
        )

        self.logger.info(f"Cleaning up least recently used model: {lru_model_id}")
        await self.cleanup_model(lru_model_id)

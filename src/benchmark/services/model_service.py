"""
Model service for the LLM Cybersecurity Benchmark system.

This module provides a comprehensive service for managing different types of LLM models
through a plugin architecture with resource monitoring and performance tracking.
"""

import asyncio
import contextlib
import statistics
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
    CostEstimate,
    EnhancedModelInfo,
    LoadedModel,
    LoadingStrategy,
    ModelDiscoveryResult,
    ModelInfo,
    ModelPlugin,
    ModelResourceMonitor,
    PerformanceComparison,
    PerformanceMetrics,
    PluginRegistry,
    Prediction,
)
from benchmark.models.plugins import (
    AnthropicModelPlugin,
    MLXModelPlugin,
    OllamaModelPlugin,
    OpenAIModelPlugin,
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
        """Initialize the model service and register plugins."""
        try:
            self.logger.info("Initializing Model Service")

            # Register all available model plugins
            await self._register_default_plugins()

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
                    "registered_plugins": list(self.plugins.keys()),
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

        except BenchmarkError:
            # Re-raise BenchmarkErrors to allow proper error handling in tests
            raise
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

    # Unified Model Management Methods

    async def _register_default_plugins(self) -> None:
        """Register all available model plugins."""
        try:
            # Local plugins
            self.logger.info("Registering local model plugins")
            self.plugins["mlx_local"] = MLXModelPlugin
            self.plugins["ollama"] = OllamaModelPlugin

            # API plugins
            self.logger.info("Registering API model plugins")
            self.plugins["openai_api"] = OpenAIModelPlugin
            self.plugins["anthropic_api"] = AnthropicModelPlugin

            self.logger.info(f"Successfully registered {len(self.plugins)} model plugins")

        except Exception as e:
            self.logger.error(f"Failed to register default plugins: {e}")
            raise ConfigurationError(f"Plugin registration failed: {e}") from e

    async def discover_available_models(self) -> ModelDiscoveryResult:
        """Discover all available models from all registered plugins."""
        start_time = time.time()

        try:
            available_models: list[EnhancedModelInfo] = []
            models_by_plugin: dict[str, list[EnhancedModelInfo]] = {}
            plugin_status: dict[str, dict[str, Any]] = {}
            errors: list[str] = []

            for plugin_name, plugin_class in self.plugins.items():
                try:
                    self.logger.info(f"Discovering models for plugin: {plugin_name}")
                    plugin_models = await self._discover_plugin_models(plugin_name, plugin_class)

                    available_models.extend(plugin_models)
                    models_by_plugin[plugin_name] = plugin_models

                    plugin_status[plugin_name] = {
                        "status": "active",
                        "models_found": len(plugin_models),
                        "last_discovery": datetime.now().isoformat(),
                    }

                except Exception as e:
                    error_msg = f"Failed to discover models for {plugin_name}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)

                    plugin_status[plugin_name] = {
                        "status": "error",
                        "error": str(e),
                        "models_found": 0,
                        "last_discovery": datetime.now().isoformat(),
                    }

                    models_by_plugin[plugin_name] = []

            discovery_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"Model discovery completed: {len(available_models)} models found in {discovery_time_ms:.2f}ms"
            )

            return ModelDiscoveryResult(
                available_models=available_models,
                models_by_plugin=models_by_plugin,
                total_models=len(available_models),
                plugin_status=plugin_status,
                discovery_time_ms=discovery_time_ms,
                errors=errors,
            )

        except Exception as e:
            self.logger.error(f"Model discovery failed: {e}")
            discovery_time_ms = (time.time() - start_time) * 1000

            return ModelDiscoveryResult(
                available_models=[],
                models_by_plugin={},
                total_models=0,
                plugin_status={},
                discovery_time_ms=discovery_time_ms,
                errors=[f"Discovery failed: {str(e)}"],
            )

    async def _discover_plugin_models(
        self, plugin_name: str, plugin_class: type
    ) -> list[EnhancedModelInfo]:
        """Discover models available for a specific plugin."""
        try:
            # Create temporary plugin instance for discovery
            temp_plugin = plugin_class()

            # Get supported models if plugin has this method
            if hasattr(temp_plugin, "get_supported_models"):
                supported_models = temp_plugin.get_supported_models()
            else:
                # Fallback: try to determine from plugin type
                supported_models = self._get_fallback_models(plugin_name)

            models: list[EnhancedModelInfo] = []

            for model_name in supported_models:
                try:
                    model_info = await self._create_enhanced_model_info(
                        plugin_name, model_name, temp_plugin
                    )
                    models.append(model_info)
                except Exception as e:
                    self.logger.warning(f"Failed to create model info for {model_name}: {e}")

            # Cleanup temporary plugin
            if hasattr(temp_plugin, "cleanup"):
                await temp_plugin.cleanup()

            return models

        except Exception as e:
            self.logger.error(f"Failed to discover models for plugin {plugin_name}: {e}")
            return []

    def _get_fallback_models(self, plugin_name: str) -> list[str]:
        """Get fallback model list when plugin doesn't support discovery."""
        fallback_models = {
            "mlx_local": [
                "mlx-community/Llama-2-7b-chat-hf-4bit",
                "mlx-community/CodeLlama-7b-Instruct-hf",
            ],
            "ollama": ["llama2", "codellama", "mistral"],
            "openai_api": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "anthropic_api": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        }
        return fallback_models.get(plugin_name, [])

    async def _create_enhanced_model_info(
        self, plugin_name: str, model_name: str, plugin_instance: Any
    ) -> EnhancedModelInfo:
        """Create enhanced model info from plugin data."""

        # Get model specifications if available
        specs: dict[str, Any] = {}
        if hasattr(plugin_instance, "get_model_specs"):
            specs = plugin_instance.get_model_specs(model_name) or {}

        # Determine deployment type and vendor
        deployment_type = "api" if "api" in plugin_name else "local"
        vendor = None

        if "openai" in plugin_name.lower():
            vendor = "OpenAI"
        elif "anthropic" in plugin_name.lower():
            vendor = "Anthropic"
        elif "ollama" in plugin_name.lower():
            vendor = "Ollama"
        elif "mlx" in plugin_name.lower():
            vendor = "MLX"

        # Extract cost information for API models
        cost_per_1k_tokens = None
        if hasattr(plugin_instance, "cost_tracker") and hasattr(
            plugin_instance.cost_tracker, "pricing"
        ):
            pricing = plugin_instance.cost_tracker.pricing.get(model_name, {})
            if pricing:
                # Average of input and output costs
                input_cost = pricing.get("input", 0)
                output_cost = pricing.get("output", 0)
                cost_per_1k_tokens = (input_cost + output_cost) / 2

        # Determine performance tier
        performance_tier = "standard"
        if any(tier in model_name.lower() for tier in ["mini", "haiku", "small"]):
            performance_tier = "fast"
        elif any(tier in model_name.lower() for tier in ["gpt-4", "opus", "large"]):
            performance_tier = "premium"

        return EnhancedModelInfo(
            plugin_type=plugin_name,
            model_id=f"{plugin_name}::{model_name}",
            model_name=model_name,
            parameters=self._estimate_parameters(model_name),
            memory_requirement_gb=specs.get(
                "memory_gb", self._estimate_memory_requirement(model_name)
            ),
            cost_per_1k_tokens=cost_per_1k_tokens,
            supports_batching=True,  # Most models support batching
            supports_explanations=True,  # Most models support explanations
            supports_streaming="api" in plugin_name,  # API models typically support streaming
            max_context_length=specs.get("context_window"),
            recommended_batch_size=self._get_recommended_batch_size(plugin_name, model_name),
            deployment_type=deployment_type,
            vendor=vendor,
            version=None,  # Add missing required field
            description=f"{vendor} {model_name}" if vendor else model_name,
            capabilities=["text-generation", "cybersecurity-analysis"],
            tags=self._generate_model_tags(plugin_name, model_name),
            performance_tier=performance_tier,
        )

    def _estimate_parameters(self, model_name: str) -> int | None:
        """Estimate model parameters from model name."""
        name_lower = model_name.lower()

        # Extract common parameter counts from model names
        if "7b" in name_lower:
            return 7_000_000_000
        elif "13b" in name_lower:
            return 13_000_000_000
        elif "30b" in name_lower or "33b" in name_lower:
            return 30_000_000_000
        elif "70b" in name_lower:
            return 70_000_000_000
        elif "175b" in name_lower:
            return 175_000_000_000
        elif "gpt-4" in name_lower:
            return 1_000_000_000_000  # Estimated for GPT-4
        elif "gpt-3.5" in name_lower:
            return 175_000_000_000  # GPT-3.5 estimate
        elif "claude" in name_lower:
            return 500_000_000_000  # Claude estimate

        return None

    def _estimate_memory_requirement(self, model_name: str) -> float:
        """Estimate memory requirement in GB."""
        params = self._estimate_parameters(model_name)
        if params:
            # Rough estimate: ~2-4 bytes per parameter for inference
            return (params * 3) / 1_000_000_000

        # Fallback estimates based on model type
        name_lower = model_name.lower()
        if any(size in name_lower for size in ["mini", "small", "7b"]):
            return 8.0
        elif any(size in name_lower for size in ["medium", "13b"]):
            return 16.0
        elif any(size in name_lower for size in ["large", "30b", "33b"]):
            return 32.0
        else:
            return 4.0  # Default for API models

    def _get_recommended_batch_size(self, plugin_name: str, model_name: str) -> int:
        """Get recommended batch size for model."""
        if "api" in plugin_name:
            return 8  # Smaller batches for API models due to rate limits
        elif "7b" in model_name.lower():
            return 16
        elif any(size in model_name.lower() for size in ["13b", "medium"]):
            return 8
        elif any(size in model_name.lower() for size in ["30b", "33b", "large"]):
            return 4
        else:
            return 32  # Default

    def _generate_model_tags(self, plugin_name: str, model_name: str) -> list[str]:
        """Generate tags for model categorization."""
        tags = [plugin_name.replace("_", "-")]

        name_lower = model_name.lower()

        # Size tags
        if any(size in name_lower for size in ["mini", "small", "7b"]):
            tags.append("small")
        elif any(size in name_lower for size in ["medium", "13b"]):
            tags.append("medium")
        elif any(size in name_lower for size in ["large", "30b", "33b", "70b"]):
            tags.append("large")

        # Capability tags
        if "chat" in name_lower:
            tags.append("conversational")
        if "code" in name_lower:
            tags.append("code-generation")
        if "instruct" in name_lower:
            tags.append("instruction-following")

        # Performance tags
        if "turbo" in name_lower:
            tags.append("fast")
        if "gpt-4" in name_lower or "opus" in name_lower:
            tags.append("high-quality")

        return tags

    async def list_available_models(self) -> list[EnhancedModelInfo]:
        """List all available models from all plugins."""
        discovery_result = await self.discover_available_models()
        return discovery_result.available_models

    async def compare_model_performance(self, model_ids: list[str]) -> PerformanceComparison:
        """Compare performance metrics across multiple models."""
        try:
            if len(model_ids) < 2:
                raise BenchmarkError(
                    "At least 2 models are required for comparison",
                    error_code=ErrorCode.INVALID_PARAMETER,
                )

            metrics: dict[str, dict[str, Any]] = {}
            valid_models = []

            # Collect metrics for each model
            for model_id in model_ids:
                try:
                    if model_id in self.loaded_models:
                        model_metrics = await self.get_model_performance(model_id)
                        metrics[model_id] = model_metrics
                        valid_models.append(model_id)
                    else:
                        self.logger.warning(f"Model {model_id} not loaded, skipping comparison")

                except Exception as e:
                    self.logger.error(f"Failed to get metrics for model {model_id}: {e}")

            if len(valid_models) < 2:
                raise BenchmarkError(
                    "Insufficient models with valid metrics for comparison",
                    error_code=ErrorCode.INSUFFICIENT_DATA,
                )

            # Calculate rankings for each metric
            rankings = self._calculate_performance_rankings(metrics)

            # Generate comparison summary
            summary = self._generate_comparison_summary(metrics, rankings)

            return PerformanceComparison(
                model_ids=valid_models,
                metrics=metrics,
                rankings=rankings,
                summary=summary,
            )

        except BenchmarkError:
            # Re-raise BenchmarkErrors to preserve original error codes
            raise
        except Exception as e:
            self.logger.error(f"Performance comparison failed: {e}")
            raise BenchmarkError(
                f"Performance comparison failed: {e}", ErrorCode.INTERNAL_ERROR
            ) from e

    def _calculate_performance_rankings(
        self, metrics: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        """Calculate performance rankings for each metric."""
        rankings: dict[str, dict[str, int]] = {}

        # Define metrics to rank (higher is better vs lower is better)
        higher_better = ["predictions_per_second", "success_rate", "average_batch_throughput"]

        lower_better = ["average_inference_time_ms", "memory_usage_mb", "error_count"]

        all_metrics = higher_better + lower_better

        for metric_name in all_metrics:
            metric_values = []

            # Extract metric values from all models
            for model_id, model_metrics in metrics.items():
                basic_metrics = model_metrics.get("basic_metrics", {})
                if metric_name in basic_metrics:
                    metric_values.append((model_id, basic_metrics[metric_name]))
                elif metric_name in model_metrics:
                    metric_values.append((model_id, model_metrics[metric_name]))

            if not metric_values:
                continue

            # Sort based on whether higher or lower is better
            reverse = metric_name in higher_better
            sorted_values = sorted(metric_values, key=lambda x: x[1], reverse=reverse)

            # Assign rankings (1 = best)
            model_rankings = {}
            for rank, (model_id, _) in enumerate(sorted_values, 1):
                model_rankings[model_id] = rank

            rankings[metric_name] = model_rankings

        return rankings

    def _generate_comparison_summary(
        self, metrics: dict[str, dict[str, Any]], rankings: dict[str, dict[str, int]]
    ) -> dict[str, Any]:
        """Generate a summary of the performance comparison."""

        # Calculate average rankings for overall performance
        overall_rankings = {}
        for model_id in metrics:
            model_ranks = []
            for metric_rankings in rankings.values():
                if model_id in metric_rankings:
                    model_ranks.append(metric_rankings[model_id])

            if model_ranks:
                overall_rankings[model_id] = statistics.mean(model_ranks)

        # Sort by average ranking (lower is better)
        sorted_overall = sorted(overall_rankings.items(), key=lambda x: x[1])

        # Generate insights
        insights = []
        if sorted_overall:
            best_model = sorted_overall[0][0]
            insights.append(f"Best overall performer: {best_model}")

            # Find best in specific categories
            for metric, model_rankings in rankings.items():
                if model_rankings:
                    best_in_category = min(model_rankings.items(), key=lambda x: x[1])
                    insights.append(f"Best {metric}: {best_in_category[0]}")

        return {
            "overall_rankings": dict(sorted_overall),
            "best_performer": sorted_overall[0][0] if sorted_overall else None,
            "insights": insights,
            "comparison_date": datetime.now().isoformat(),
            "models_compared": len(metrics),
        }

    async def optimize_model_loading(self, configs: list[dict[str, Any]]) -> LoadingStrategy:
        """Optimize model loading order and resource usage."""
        try:
            if not configs:
                raise BenchmarkError(
                    "No model configurations provided", ErrorCode.INVALID_PARAMETER
                )

            # Analyze resource requirements for each model
            resource_analysis = []
            for config in configs:
                analysis = await self._analyze_model_resources(config)
                resource_analysis.append(analysis)

            # Generate optimal loading order
            loading_order = self._calculate_optimal_loading_order(resource_analysis)

            # Plan resource allocation
            resource_allocation = self._plan_resource_allocation(resource_analysis)

            # Identify parallel loading opportunities
            parallel_groups = self._identify_parallel_loading_groups(resource_analysis)

            # Calculate estimates
            total_memory = sum(analysis["memory_requirement_mb"] for analysis in resource_analysis)
            estimated_loading_time = self._estimate_loading_time(resource_analysis, parallel_groups)

            # Generate optimization notes
            optimization_notes = self._generate_optimization_notes(
                resource_analysis, parallel_groups, total_memory
            )

            return LoadingStrategy(
                model_configs=configs,
                loading_order=loading_order,
                resource_allocation=resource_allocation,
                parallel_loading_groups=parallel_groups,
                estimated_total_memory_mb=total_memory,
                estimated_loading_time_seconds=estimated_loading_time,
                optimization_notes=optimization_notes,
            )

        except BenchmarkError:
            # Re-raise BenchmarkErrors to preserve original error codes
            raise
        except Exception as e:
            self.logger.error(f"Loading optimization failed: {e}")
            raise BenchmarkError(
                f"Loading optimization failed: {e}", ErrorCode.INTERNAL_ERROR
            ) from e

    async def _analyze_model_resources(self, config: dict[str, Any]) -> dict[str, Any]:
        """Analyze resource requirements for a model configuration."""
        model_type = config.get("type", "unknown")
        model_name = config.get("model_name", config.get("name", "unknown"))

        # Estimate memory requirement
        memory_mb = self._estimate_memory_requirement(model_name) * 1024  # Convert GB to MB

        # Estimate loading time based on model type and size
        loading_time_seconds = 10.0  # Base loading time
        if "api" in model_type:
            loading_time_seconds = 2.0  # API models load faster
        elif "70b" in model_name.lower():
            loading_time_seconds = 60.0  # Large models take longer
        elif "30b" in model_name.lower():
            loading_time_seconds = 30.0
        elif "13b" in model_name.lower():
            loading_time_seconds = 15.0

        return {
            "config": config,
            "model_name": model_name,
            "model_type": model_type,
            "memory_requirement_mb": memory_mb,
            "loading_time_seconds": loading_time_seconds,
            "is_api_model": "api" in model_type,
            "priority": self._calculate_model_priority(config),
        }

    def _calculate_model_priority(self, config: dict[str, Any]) -> int:
        """Calculate loading priority for a model (1 = highest priority)."""
        model_name = config.get("model_name", "").lower()
        model_type = config.get("type", "").lower()

        # API models get higher priority (load faster)
        if "api" in model_type:
            return 1

        # Smaller models get higher priority
        if "7b" in model_name or "mini" in model_name or "small" in model_name:
            return 2
        elif "13b" in model_name or "medium" in model_name:
            return 3
        elif "30b" in model_name or "large" in model_name:
            return 4
        else:
            return 5

    def _calculate_optimal_loading_order(
        self, resource_analysis: list[dict[str, Any]]
    ) -> list[str]:
        """Calculate optimal loading order based on resource analysis."""
        # Sort by priority first, then by memory requirement
        sorted_analysis = sorted(
            resource_analysis, key=lambda x: (x["priority"], x["memory_requirement_mb"])
        )

        return [analysis["model_name"] for analysis in sorted_analysis]

    def _plan_resource_allocation(
        self, resource_analysis: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Plan resource allocation for each model."""
        allocation = {}

        for analysis in resource_analysis:
            model_name = analysis["model_name"]
            allocation[model_name] = {
                "memory_mb": analysis["memory_requirement_mb"],
                "cpu_cores": 2 if analysis["is_api_model"] else 4,
                "gpu_memory_mb": 0
                if analysis["is_api_model"]
                else analysis["memory_requirement_mb"] * 0.8,
                "loading_time_seconds": analysis["loading_time_seconds"],
            }

        return allocation

    def _identify_parallel_loading_groups(
        self, resource_analysis: list[dict[str, Any]]
    ) -> list[list[str]]:
        """Identify models that can be loaded in parallel."""
        api_models = []
        small_local_models = []
        large_local_models = []

        for analysis in resource_analysis:
            model_name = analysis["model_name"]

            if analysis["is_api_model"]:
                api_models.append(model_name)
            elif analysis["memory_requirement_mb"] < 16000:  # Less than 16GB
                small_local_models.append(model_name)
            else:
                large_local_models.append(model_name)

        parallel_groups = []

        # API models can all load in parallel
        if api_models:
            parallel_groups.append(api_models)

        # Small local models can load in pairs
        for i in range(0, len(small_local_models), 2):
            group = small_local_models[i : i + 2]
            parallel_groups.append(group)

        # Large local models load sequentially
        for model in large_local_models:
            parallel_groups.append([model])

        return parallel_groups

    def _estimate_loading_time(
        self, resource_analysis: list[dict[str, Any]], parallel_groups: list[list[str]]
    ) -> float:
        """Estimate total loading time considering parallel loading."""
        total_time = 0.0

        for group in parallel_groups:
            # Find the maximum loading time in this parallel group
            group_loading_times = []
            for model_name in group:
                for analysis in resource_analysis:
                    if analysis["model_name"] == model_name:
                        group_loading_times.append(analysis["loading_time_seconds"])
                        break

            if group_loading_times:
                total_time += max(group_loading_times)

        return total_time

    def _generate_optimization_notes(
        self,
        resource_analysis: list[dict[str, Any]],
        parallel_groups: list[list[str]],
        total_memory_mb: float,
    ) -> list[str]:
        """Generate optimization explanations."""
        notes = []

        # Memory optimization
        if total_memory_mb > self.max_memory_mb:
            notes.append(
                f"⚠️ Total memory requirement ({total_memory_mb:.1f}MB) exceeds limit ({self.max_memory_mb}MB)"
            )
            notes.append("💡 Consider loading models sequentially or increasing memory limit")

        # Parallel loading opportunities
        parallel_count = sum(len(group) for group in parallel_groups if len(group) > 1)
        if parallel_count > 0:
            notes.append(
                f"🚀 {parallel_count} models can be loaded in parallel for faster initialization"
            )

        # API vs local model optimization
        api_count = sum(1 for analysis in resource_analysis if analysis["is_api_model"])
        local_count = len(resource_analysis) - api_count

        if api_count > 0 and local_count > 0:
            notes.append(
                f"⚡ {api_count} API models will load quickly, {local_count} local models need more resources"
            )

        # Resource recommendations
        large_models = [a for a in resource_analysis if a["memory_requirement_mb"] > 32000]
        if large_models:
            notes.append(
                f"🏋️ {len(large_models)} large models detected - consider GPU acceleration if available"
            )

        return notes

    async def get_cost_estimates(
        self, model_configs: list[dict[str, Any]], estimated_samples: int
    ) -> CostEstimate:
        """Estimate costs for running evaluation with given models."""
        try:
            if not model_configs:
                raise BenchmarkError(
                    "No model configurations provided", ErrorCode.INVALID_PARAMETER
                )

            if estimated_samples <= 0:
                raise BenchmarkError(
                    "Estimated samples must be positive", ErrorCode.INVALID_PARAMETER
                )

            cost_by_model: dict[str, float] = {}
            api_costs = 0.0
            local_compute_costs = 0.0
            recommendations: list[str] = []
            assumptions: dict[str, Any] = {
                "estimated_samples": estimated_samples,
                "average_tokens_per_sample": 200,  # Assumption for cybersecurity samples
                "electricity_cost_per_kwh": 0.12,  # USD per kWh
                "gpu_power_consumption_watts": 300,  # Typical GPU power usage
            }

            for config in model_configs:
                model_type = config.get("type", "unknown")
                model_name = config.get("model_name", config.get("name", "unknown"))

                model_cost = await self._estimate_model_cost(
                    model_type, model_name, estimated_samples, assumptions
                )

                cost_by_model[model_name] = model_cost

                if "api" in model_type:
                    api_costs += model_cost
                else:
                    local_compute_costs += model_cost

            total_cost = sum(cost_by_model.values())

            # Generate cost optimization recommendations
            recommendations = self._generate_cost_recommendations(
                model_configs, cost_by_model, total_cost, estimated_samples
            )

            return CostEstimate(
                model_configs=model_configs,
                estimated_samples=estimated_samples,
                total_estimated_cost_usd=total_cost,
                cost_by_model=cost_by_model,
                api_costs=api_costs,
                local_compute_costs=local_compute_costs,
                recommendations=recommendations,
                assumptions=assumptions,
            )

        except BenchmarkError:
            # Re-raise BenchmarkErrors to preserve original error codes
            raise
        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            raise BenchmarkError(f"Cost estimation failed: {e}", ErrorCode.INTERNAL_ERROR) from e

    async def _estimate_model_cost(
        self, model_type: str, model_name: str, samples: int, assumptions: dict[str, Any]
    ) -> float:
        """Estimate cost for a specific model."""

        if "api" in model_type:
            # API model cost estimation
            return self._estimate_api_cost(model_type, model_name, samples, assumptions)
        else:
            # Local model cost estimation (electricity)
            return self._estimate_local_cost(model_name, samples, assumptions)

    def _estimate_api_cost(
        self, model_type: str, model_name: str, samples: int, assumptions: dict[str, Any]
    ) -> float:
        """Estimate API cost based on token usage."""

        # Get cost per 1k tokens from model pricing
        cost_per_1k = None

        try:
            if "openai" in model_type:
                from benchmark.models.plugins.openai_api import CostTracker

                openai_tracker = CostTracker()
                pricing = openai_tracker.pricing.get(model_name, {})
                if pricing:
                    cost_per_1k = (pricing.get("input", 0) + pricing.get("output", 0)) / 2

            elif "anthropic" in model_type:
                from benchmark.models.plugins.anthropic_api import AnthropicCostTracker

                anthropic_tracker: Any = AnthropicCostTracker()
                pricing = anthropic_tracker.pricing.get(model_name, {})
                if pricing:
                    cost_per_1k = (pricing.get("input", 0) + pricing.get("output", 0)) / 2

        except Exception as e:
            self.logger.warning(f"Failed to get pricing for {model_name}: {e}")

        # Fallback pricing if not found
        if cost_per_1k is None:
            if "gpt-4" in model_name.lower():
                cost_per_1k = 0.045  # Average GPT-4 pricing
            elif "gpt-3.5" in model_name.lower() or "mini" in model_name.lower():
                cost_per_1k = 0.001  # GPT-3.5/mini pricing
            elif "claude" in model_name.lower():
                cost_per_1k = 0.008  # Average Claude pricing
            else:
                cost_per_1k = 0.002  # Generic API model

        # Calculate total cost
        avg_tokens: float = float(assumptions["average_tokens_per_sample"])
        total_tokens = samples * avg_tokens
        total_cost = (total_tokens / 1000) * cost_per_1k

        return float(total_cost)

    def _estimate_local_cost(
        self, model_name: str, samples: int, assumptions: dict[str, Any]
    ) -> float:
        """Estimate local compute cost (electricity)."""

        # Estimate inference time per sample based on model size
        if "70b" in model_name.lower():
            seconds_per_sample = 10.0
        elif "30b" in model_name.lower():
            seconds_per_sample = 5.0
        elif "13b" in model_name.lower():
            seconds_per_sample = 2.0
        elif "7b" in model_name.lower():
            seconds_per_sample = 1.0
        else:
            seconds_per_sample = 3.0  # Default

        # Calculate total compute time
        total_seconds = samples * seconds_per_sample
        total_hours = total_seconds / 3600

        # Calculate electricity cost
        power_kw = float(assumptions["gpu_power_consumption_watts"]) / 1000
        electricity_cost_per_hour = power_kw * float(assumptions["electricity_cost_per_kwh"])
        total_electricity_cost = total_hours * electricity_cost_per_hour

        return float(total_electricity_cost)

    def _generate_cost_recommendations(
        self,
        model_configs: list[dict[str, Any]],
        cost_by_model: dict[str, float],
        total_cost: float,
        samples: int,
    ) -> list[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Find most expensive model
        if cost_by_model:
            most_expensive = max(cost_by_model.items(), key=lambda x: x[1])
            if most_expensive[1] > total_cost * 0.5:
                recommendations.append(f"💰 {most_expensive[0]} accounts for over 50% of costs")

        # API vs local cost comparison
        api_models = [c for c in model_configs if "api" in c.get("type", "")]
        local_models = [c for c in model_configs if "api" not in c.get("type", "")]

        if api_models and local_models:
            api_cost = sum(cost_by_model.get(c.get("model_name", ""), 0) for c in api_models)
            local_cost = sum(cost_by_model.get(c.get("model_name", ""), 0) for c in local_models)

            if api_cost > local_cost * 2:
                recommendations.append("🏠 Consider using more local models to reduce API costs")
            elif local_cost > api_cost * 2:
                recommendations.append(
                    "☁️ API models may be more cost-effective for this evaluation"
                )

        # Sample size recommendations
        if total_cost > 100:  # Over $100
            smaller_sample_cost = (total_cost / samples) * min(samples, 1000)
            recommendations.append(
                f"📊 Consider testing with 1,000 samples first (${smaller_sample_cost:.2f})"
            )

        # Batch processing recommendation
        recommendations.append("⚡ Use larger batch sizes when possible to improve efficiency")

        return recommendations

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

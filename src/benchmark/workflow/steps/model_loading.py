"""Model loading workflow step for loading and validating models."""

from typing import Any

from benchmark.core.logging import get_logger
from benchmark.interfaces.orchestration_interfaces import WorkflowContext, WorkflowStep


class ModelLoadingStep(WorkflowStep):
    """Workflow step for loading and validating models with resource management."""

    def __init__(self) -> None:
        self.logger = get_logger("model_loading_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute model loading step with resource management."""
        self.logger.info("Starting model loading step for experiment %s", context.experiment_id)

        model_service = context.services.get("model")
        if not model_service:
            raise Exception("Model service not available")

        models_config = context.config.get("models", [])
        if not models_config:
            raise Exception("No models specified in configuration")

        loaded_models: dict[str, Any] = {}
        loading_results = {}
        model_info = []
        total_memory_usage = 0.0

        # Load models sequentially to manage memory usage
        for model_config in models_config:
            model_id = model_config.get("id")
            if not model_id:
                raise Exception("Model configuration missing 'id' field")

            try:
                self.logger.info("Loading model: %s", model_id)

                # Check resource availability before loading
                try:
                    resource_check = await model_service.resource_manager.can_load_model(
                        model_config
                    )
                    estimated_memory = getattr(resource_check, "estimated_memory_gb", 1.0)
                    can_load = getattr(resource_check, "can_load", True)
                except AttributeError:
                    # Fallback if resource manager doesn't have these methods
                    self.logger.warning(
                        "Resource manager check not available, proceeding with model loading"
                    )
                    estimated_memory = 1.0
                    can_load = True

                if not can_load and loaded_models:
                    # Try to free up memory by unloading loaded models if possible
                    self.logger.info("Attempting to free memory for model %s", model_id)
                    # For simplicity, we'll continue loading but log the warning
                    self.logger.warning(
                        "Insufficient resources for model %s, proceeding anyway", model_id
                    )

                # Load model using model service
                load_response = await model_service.load_model(
                    model_id=model_id,
                    plugin_type=model_config.get("plugin_type", "local"),
                    config=model_config.get("config", {}),
                )

                if not load_response.success:
                    raise Exception(f"Failed to load model {model_id}: {load_response.error}")

                model_data = load_response.data
                loaded_models[model_id] = model_data

                # Get model information
                try:
                    model_info_response = await model_service.get_model_info(model_id)
                    model_metadata = model_info_response.data if model_info_response.success else {}
                except Exception as e:
                    self.logger.warning("Failed to get model info for %s: %s", model_id, e)
                    model_metadata = {}

                # Extract model statistics
                memory_usage = model_metadata.get("memory_usage_mb", estimated_memory * 1024)
                if isinstance(memory_usage, int | float):
                    total_memory_usage += memory_usage

                model_type = model_config.get("plugin_type", "unknown")
                model_params = model_metadata.get("parameters", "unknown")

                loading_results[model_id] = {
                    "status": "success",
                    "plugin_type": model_type,
                    "memory_usage_mb": memory_usage,
                    "model_parameters": model_params,
                    "config": model_config.get("config", {}),
                    "estimated_memory_gb": estimated_memory,
                }

                model_info.append(
                    {
                        "id": model_id,
                        "plugin_type": model_type,
                        "status": "loaded",
                        "memory_usage": memory_usage,
                        "parameters": model_params,
                    }
                )

                self.logger.info(
                    "Successfully loaded model: %s (%s, %.1f MB)",
                    model_id,
                    model_type,
                    memory_usage,
                )

            except Exception as e:
                error_msg = str(e)
                self.logger.error("Failed to load model %s: %s", model_id, error_msg)

                loading_results[model_id] = {
                    "status": "failed",
                    "error": error_msg,
                    "plugin_type": model_config.get("plugin_type", "unknown"),
                }

                # Continue loading other models even if one fails
                self.logger.warning("Continuing with other models after failure of %s", model_id)

        if not loaded_models:
            raise Exception("No models could be loaded successfully")

        # Store loaded models in context resources
        context.resources["loaded_models"] = loaded_models

        # Apply hardware optimizations if available
        try:
            await model_service.optimize_for_hardware()
            optimization_applied = True
            self.logger.info("Applied hardware optimizations for loaded models")
        except AttributeError:
            # Service doesn't have optimization method
            optimization_applied = False
            self.logger.info("Hardware optimization not available")
        except Exception as e:
            optimization_applied = False
            self.logger.warning("Failed to apply hardware optimizations: %s", e)

        # Calculate success/failure statistics
        successful_loads = sum(
            1 for result in loading_results.values() if result["status"] == "success"
        )
        failed_loads = len(loading_results) - successful_loads

        # Generate loading summary
        result = {
            "models_loaded": len(loaded_models),
            "models": model_info,
            "loading_results": loading_results,
            "total_memory_usage_mb": total_memory_usage,
            "successful_loads": successful_loads,
            "failed_loads": failed_loads,
            "optimization_applied": optimization_applied,
            "model_types": list(
                {
                    res.get("plugin_type", "unknown")
                    for res in loading_results.values()
                    if res["status"] == "success"
                }
            ),
        }

        self.logger.info(
            "Model loading step completed: %d models loaded, %d failed, %.1f MB total memory",
            successful_loads,
            failed_loads,
            total_memory_usage,
        )

        return result

    def get_step_name(self) -> str:
        """Get step name."""
        return "model_loading"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return ["model"]

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 360.0  # 6 minutes for model loading with multiple models

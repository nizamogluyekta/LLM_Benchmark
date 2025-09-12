"""
Intelligent resource management for multiple models on M4 Pro hardware.

This module provides sophisticated resource management capabilities for efficiently
handling multiple LLM models within the memory constraints of MacBook Pro M4 Pro,
taking advantage of Apple Silicon's unified memory architecture.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

from benchmark.core.config import ModelConfig


class ModelPriority(Enum):
    """Model priority levels for resource allocation."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class UnloadReason(Enum):
    """Reasons for model unloading."""

    MEMORY_PRESSURE = "memory_pressure"
    IDLE_TIMEOUT = "idle_timeout"
    USER_REQUEST = "user_request"
    OPTIMIZATION = "optimization"


@dataclass
class LoadedModelInfo:
    """Information about a loaded model."""

    model_id: str
    config: ModelConfig
    memory_usage_gb: float
    load_time: datetime
    last_used: datetime
    access_count: int
    plugin_type: str
    priority: ModelPriority = ModelPriority.NORMAL
    estimated_memory_gb: float = 0.0


@dataclass
class ResourceCheckResult:
    """Result of resource availability check."""

    can_load: bool
    estimated_memory_gb: float
    current_usage_gb: float
    available_memory_gb: float
    recommendations: list[str]
    required_unloads: list[str] = None
    optimization_suggestions: list[str] = None


@dataclass
class SystemMemoryInfo:
    """System memory information."""

    total_gb: float
    available_gb: float
    used_gb: float
    percentage: float
    swap_used_gb: float = 0.0
    apple_silicon_unified: bool = True


@dataclass
class ModelEstimate:
    """Model memory usage estimate."""

    base_memory_gb: float
    context_memory_gb: float
    overhead_memory_gb: float
    total_estimated_gb: float
    confidence: float  # 0.0 to 1.0


class MemoryMonitor:
    """Monitor system and process memory usage."""

    def __init__(self):
        self.process = psutil.Process()
        self.logger = logging.getLogger(__name__)
        self._baseline_memory = None

    async def get_current_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024**3)  # Convert to GB
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return 0.0

    async def get_system_memory_info(self) -> SystemMemoryInfo:
        """Get comprehensive system memory information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory() if hasattr(psutil, "swap_memory") else None

            return SystemMemoryInfo(
                total_gb=memory.total / (1024**3),
                available_gb=memory.available / (1024**3),
                used_gb=memory.used / (1024**3),
                percentage=memory.percent,
                swap_used_gb=swap.used / (1024**3) if swap else 0.0,
                apple_silicon_unified=self._detect_apple_silicon(),
            )
        except Exception as e:
            self.logger.error(f"Failed to get system memory info: {e}")
            return SystemMemoryInfo(16.0, 8.0, 8.0, 50.0)

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        try:
            import platform

            return platform.machine() == "arm64" and platform.system() == "Darwin"
        except Exception:
            return False

    async def set_baseline(self) -> None:
        """Set baseline memory usage."""
        self._baseline_memory = await self.get_current_usage()
        self.logger.info(f"Memory baseline set: {self._baseline_memory:.2f}GB")

    async def get_memory_delta(self) -> float:
        """Get memory usage delta from baseline."""
        if self._baseline_memory is None:
            await self.set_baseline()
            return 0.0

        current = await self.get_current_usage()
        return current - self._baseline_memory


class ModelResourceManager:
    """Intelligent resource manager for multiple models on M4 Pro hardware."""

    def __init__(self, max_memory_gb: float = 32.0, cache_dir: Path | None = None):
        """
        Initialize resource manager.

        Args:
            max_memory_gb: Maximum memory to use (conservative for M4 Pro)
            cache_dir: Directory for model caching
        """
        self.max_memory_gb = max_memory_gb
        self.cache_dir = cache_dir or Path.home() / ".benchmark_cache" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_models: dict[str, LoadedModelInfo] = {}
        self.memory_monitor = MemoryMonitor()
        self.logger = logging.getLogger(__name__)

        # Resource management parameters
        self.memory_pressure_threshold = 0.85  # 85% memory usage
        self.idle_timeout_minutes = 30
        self.cleanup_interval_seconds = 300  # 5 minutes

        # Model size estimates (in GB) - conservative estimates
        self.model_size_estimates = {
            "gpt-4o-mini": ModelEstimate(0.5, 0.2, 0.3, 1.0, 0.8),
            "gpt-4": ModelEstimate(1.0, 0.5, 0.5, 2.0, 0.7),
            "claude-3-haiku": ModelEstimate(0.8, 0.3, 0.4, 1.5, 0.8),
            "claude-3-sonnet": ModelEstimate(2.0, 0.8, 0.7, 3.5, 0.7),
            "llama2-7b": ModelEstimate(7.0, 1.0, 1.0, 9.0, 0.9),
            "llama2-13b": ModelEstimate(13.0, 2.0, 1.5, 16.5, 0.9),
            "llama2-70b": ModelEstimate(70.0, 8.0, 7.0, 85.0, 0.8),
            "mixtral-8x7b": ModelEstimate(45.0, 6.0, 4.0, 55.0, 0.7),
        }

        # Start background cleanup task
        self._cleanup_task = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize the resource manager."""
        await self.memory_monitor.set_baseline()
        self._running = True
        self._cleanup_task = asyncio.create_task(self._background_cleanup())
        self.logger.info("Resource manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the resource manager."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
        self.logger.info("Resource manager shutdown")

    async def can_load_model(self, model_config: ModelConfig) -> ResourceCheckResult:
        """
        Check if model can be loaded within resource constraints.

        Args:
            model_config: Configuration of model to check

        Returns:
            ResourceCheckResult with load feasibility and recommendations
        """
        try:
            # Get current memory status
            current_usage = await self.memory_monitor.get_current_usage()
            system_info = await self.memory_monitor.get_system_memory_info()

            # Estimate model memory requirements
            estimated_memory = await self._estimate_model_memory(model_config)

            # Calculate if we can load
            total_after_load = current_usage + estimated_memory.total_estimated_gb
            can_load = total_after_load <= self.max_memory_gb

            # Generate recommendations
            recommendations = []
            required_unloads = []
            optimization_suggestions = []

            if not can_load:
                # Calculate how much memory we need to free
                memory_needed = total_after_load - self.max_memory_gb

                # Find models to unload
                candidates = await self.suggest_model_unload_candidates(memory_needed)
                required_unloads = candidates

                recommendations.append(f"Need to free {memory_needed:.2f}GB memory")
                if candidates:
                    recommendations.append(f"Suggest unloading: {', '.join(candidates)}")
                else:
                    recommendations.append(
                        "No suitable models to unload - consider increasing memory limit"
                    )

            # Add optimization suggestions
            optimization_suggestions.extend(
                await self._get_optimization_recommendations(model_config)
            )

            return ResourceCheckResult(
                can_load=can_load,
                estimated_memory_gb=estimated_memory.total_estimated_gb,
                current_usage_gb=current_usage,
                available_memory_gb=system_info.available_gb,
                recommendations=recommendations,
                required_unloads=required_unloads,
                optimization_suggestions=optimization_suggestions,
            )

        except Exception as e:
            self.logger.error(f"Error checking resource availability: {e}")
            return ResourceCheckResult(
                can_load=False,
                estimated_memory_gb=0.0,
                current_usage_gb=0.0,
                available_memory_gb=0.0,
                recommendations=[f"Error checking resources: {e}"],
            )

    async def register_model_load(
        self, model_id: str, config: ModelConfig, actual_memory_gb: float, plugin_type: str
    ) -> None:
        """
        Register a successfully loaded model.

        Args:
            model_id: Unique identifier for the model
            config: Model configuration
            actual_memory_gb: Actual memory usage in GB
            plugin_type: Type of model plugin
        """
        now = datetime.now()
        estimated_memory = await self._estimate_model_memory(config)

        self.loaded_models[model_id] = LoadedModelInfo(
            model_id=model_id,
            config=config,
            memory_usage_gb=actual_memory_gb,
            load_time=now,
            last_used=now,
            access_count=1,
            plugin_type=plugin_type,
            priority=ModelPriority.NORMAL,
            estimated_memory_gb=estimated_memory.total_estimated_gb,
        )

        self.logger.info(f"Registered model {model_id}: {actual_memory_gb:.2f}GB")

    async def register_model_access(self, model_id: str) -> None:
        """Register model access for usage tracking."""
        if model_id in self.loaded_models:
            model_info = self.loaded_models[model_id]
            model_info.last_used = datetime.now()
            model_info.access_count += 1

    async def unregister_model(
        self, model_id: str, reason: UnloadReason = UnloadReason.USER_REQUEST
    ) -> None:
        """
        Unregister a model that has been unloaded.

        Args:
            model_id: Model to unregister
            reason: Reason for unloading
        """
        if model_id in self.loaded_models:
            model_info = self.loaded_models.pop(model_id)
            self.logger.info(
                f"Unregistered model {model_id} (reason: {reason.value}): "
                f"freed {model_info.memory_usage_gb:.2f}GB"
            )

    async def optimize_model_loading_order(
        self, model_configs: list[ModelConfig]
    ) -> list[ModelConfig]:
        """
        Optimize the order of model loading for best resource utilization.

        Args:
            model_configs: List of model configurations to load

        Returns:
            Optimized loading order
        """
        try:
            # Create model estimates with priorities
            model_estimates = []
            for config in model_configs:
                estimate = await self._estimate_model_memory(config)
                priority_score = self._calculate_priority_score(config, estimate)
                model_estimates.append((config, estimate, priority_score))

            # Sort by priority score (higher is better) and memory efficiency
            optimized = sorted(
                model_estimates,
                key=lambda x: (x[2], -x[1].total_estimated_gb),  # High priority, low memory first
            )

            return [config for config, _, _ in optimized]

        except Exception as e:
            self.logger.error(f"Error optimizing loading order: {e}")
            return model_configs  # Return original order on error

    async def suggest_model_unload_candidates(self, required_memory_gb: float) -> list[str]:
        """
        Suggest which models to unload to free up required memory.

        Args:
            required_memory_gb: Amount of memory that needs to be freed

        Returns:
            List of model IDs to unload
        """
        if not self.loaded_models:
            return []

        # Score models for unloading (lower score = better candidate)
        candidates = []
        for model_id, model_info in self.loaded_models.items():
            score = self._calculate_unload_score(model_info)
            candidates.append((model_id, model_info.memory_usage_gb, score))

        # Sort by unload score (best candidates first)
        candidates.sort(key=lambda x: x[2])

        # Select models to unload
        unload_candidates = []
        freed_memory = 0.0

        for model_id, memory_usage, _ in candidates:
            if freed_memory >= required_memory_gb:
                break

            unload_candidates.append(model_id)
            freed_memory += memory_usage

        return unload_candidates

    async def get_resource_statistics(self) -> dict[str, Any]:
        """Get comprehensive resource usage statistics."""
        try:
            current_usage = await self.memory_monitor.get_current_usage()
            system_info = await self.memory_monitor.get_system_memory_info()

            # Calculate model statistics
            total_model_memory = sum(info.memory_usage_gb for info in self.loaded_models.values())
            model_count = len(self.loaded_models)

            # Usage efficiency
            efficiency = (total_model_memory / current_usage * 100) if current_usage > 0 else 0

            return {
                "memory": {
                    "current_usage_gb": current_usage,
                    "max_allowed_gb": self.max_memory_gb,
                    "utilization_percent": (current_usage / self.max_memory_gb) * 100,
                    "total_model_memory_gb": total_model_memory,
                    "system_total_gb": system_info.total_gb,
                    "system_available_gb": system_info.available_gb,
                    "apple_silicon": system_info.apple_silicon_unified,
                },
                "models": {
                    "loaded_count": model_count,
                    "total_memory_gb": total_model_memory,
                    "average_memory_gb": total_model_memory / model_count if model_count > 0 else 0,
                    "efficiency_percent": efficiency,
                },
                "performance": {
                    "memory_pressure": current_usage / self.max_memory_gb
                    > self.memory_pressure_threshold,
                    "idle_models": len(
                        [m for m in self.loaded_models.values() if self._is_model_idle(m)]
                    ),
                    "last_cleanup": getattr(self, "_last_cleanup", None),
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting resource statistics: {e}")
            return {"error": str(e)}

    async def force_cleanup(self) -> dict[str, Any]:
        """Force immediate cleanup of idle models."""
        cleanup_results = {"unloaded_models": [], "freed_memory_gb": 0.0}

        try:
            idle_models = [
                (model_id, info)
                for model_id, info in self.loaded_models.items()
                if self._is_model_idle(info)
            ]

            for model_id, model_info in idle_models:
                cleanup_results["unloaded_models"].append(model_id)
                cleanup_results["freed_memory_gb"] += model_info.memory_usage_gb
                await self.unregister_model(model_id, UnloadReason.IDLE_TIMEOUT)

            self.logger.info(
                f"Force cleanup freed {cleanup_results['freed_memory_gb']:.2f}GB "
                f"from {len(cleanup_results['unloaded_models'])} models"
            )

        except Exception as e:
            self.logger.error(f"Error during force cleanup: {e}")
            cleanup_results["error"] = str(e)

        return cleanup_results

    async def _estimate_model_memory(self, config: ModelConfig) -> ModelEstimate:
        """Estimate memory requirements for a model."""
        model_name = getattr(config, "model_name", "unknown")
        model_type = getattr(config, "type", "unknown")

        # Check if we have a specific estimate
        for pattern, estimate in self.model_size_estimates.items():
            if pattern.lower() in model_name.lower():
                return estimate

        # Default estimates based on model type
        if model_type == "openai_api" or model_type == "anthropic_api":
            # API models have minimal local memory footprint
            return ModelEstimate(0.2, 0.1, 0.2, 0.5, 0.9)
        elif "llama" in model_name.lower():
            if "70b" in model_name.lower():
                return ModelEstimate(70.0, 8.0, 7.0, 85.0, 0.8)
            elif "13b" in model_name.lower():
                return ModelEstimate(13.0, 2.0, 1.5, 16.5, 0.9)
            else:  # Assume 7B
                return ModelEstimate(7.0, 1.0, 1.0, 9.0, 0.9)
        else:
            # Conservative default for unknown models
            return ModelEstimate(5.0, 1.0, 1.0, 7.0, 0.5)

    def _calculate_priority_score(self, config: ModelConfig, estimate: ModelEstimate) -> float:
        """Calculate priority score for model loading order."""
        score = 0.0

        model_type = getattr(config, "type", "")

        # API models get higher priority (lower resource usage)
        if model_type in ["openai_api", "anthropic_api"]:
            score += 10.0

        # Smaller models get higher priority
        if estimate.total_estimated_gb < 2.0:
            score += 5.0
        elif estimate.total_estimated_gb < 10.0:
            score += 3.0

        # High confidence estimates get higher priority
        score += estimate.confidence * 2.0

        return score

    def _calculate_unload_score(self, model_info: LoadedModelInfo) -> float:
        """Calculate score for model unloading (lower = better candidate)."""
        now = datetime.now()
        time_since_use = (now - model_info.last_used).total_seconds() / 3600  # Hours

        score = 0.0

        # Time since last use (higher = better candidate)
        score += time_since_use * 2.0

        # Memory usage (higher = better candidate for large memory needs)
        score += model_info.memory_usage_gb * 0.5

        # Access frequency (lower = better candidate)
        hours_loaded = (now - model_info.load_time).total_seconds() / 3600
        access_rate = model_info.access_count / max(hours_loaded, 1.0)
        score -= access_rate * 3.0

        # Priority (lower priority = better candidate)
        priority_penalty = {
            ModelPriority.CRITICAL: 100.0,
            ModelPriority.HIGH: 50.0,
            ModelPriority.NORMAL: 0.0,
            ModelPriority.LOW: -10.0,
        }
        score += priority_penalty.get(model_info.priority, 0.0)

        return score

    def _is_model_idle(self, model_info: LoadedModelInfo) -> bool:
        """Check if a model is considered idle."""
        idle_time = datetime.now() - model_info.last_used
        return idle_time > timedelta(minutes=self.idle_timeout_minutes)

    async def _get_optimization_recommendations(self, config: ModelConfig) -> list[str]:
        """Get optimization recommendations for model loading."""
        recommendations = []

        model_name = getattr(config, "model_name", "")
        model_type = getattr(config, "type", "")

        # API model recommendations
        if model_type in ["openai_api", "anthropic_api"]:
            recommendations.append("API model: Consider batch processing for efficiency")
            recommendations.append("API model: Minimal local memory usage")

        # Local model recommendations
        elif model_type in ["mlx_local", "ollama_local"]:
            if "70b" in model_name.lower():
                recommendations.append("Large model: Consider model sharding or quantization")
                recommendations.append(
                    "Large model: Ensure sufficient system memory (64GB+ recommended)"
                )
            elif "13b" in model_name.lower():
                recommendations.append("Medium model: Good balance for M4 Pro (32GB RAM)")
            else:
                recommendations.append("Small model: Optimal for M4 Pro hardware")

            recommendations.append("Local model: Benefits from Apple Silicon optimization")

        # Memory-specific recommendations
        estimate = await self._estimate_model_memory(config)
        if estimate.total_estimated_gb > 16.0:
            recommendations.append("High memory usage: Monitor system performance")

        return recommendations

    async def _background_cleanup(self) -> None:
        """Background task for cleaning up idle models."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)

                if not self._running:
                    break

                # Check for memory pressure
                current_usage = await self.memory_monitor.get_current_usage()
                memory_pressure = current_usage / self.max_memory_gb

                if memory_pressure > self.memory_pressure_threshold:
                    self.logger.warning(f"Memory pressure detected: {memory_pressure:.1%}")
                    await self.force_cleanup()

                # Clean up idle models
                idle_models = [
                    model_id
                    for model_id, info in self.loaded_models.items()
                    if self._is_model_idle(info)
                ]

                for model_id in idle_models:
                    await self.unregister_model(model_id, UnloadReason.IDLE_TIMEOUT)

                self._last_cleanup = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {e}")

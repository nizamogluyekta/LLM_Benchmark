"""
Model configuration validation and compatibility checking.

This module provides comprehensive validation for model configurations, hardware
compatibility checking, API access validation, and optimization recommendations.
"""

import os
import platform
import subprocess
from typing import Any

import aiohttp
import psutil
from pydantic import BaseModel, Field

from benchmark.core.logging import get_logger


class HardwareInfo(BaseModel):
    """System hardware information for compatibility checking."""

    cpu_cores: int = Field(..., ge=1, description="Number of CPU cores")
    memory_gb: float = Field(..., ge=0.0, description="Total system memory in GB")
    gpu_memory_gb: float | None = Field(None, ge=0.0, description="GPU memory in GB")
    neural_engine_available: bool = Field(default=False, description="Neural Engine availability")
    apple_silicon: bool = Field(default=False, description="Apple Silicon processor")
    platform: str = Field(..., description="Operating system platform")
    architecture: str = Field(..., description="CPU architecture")
    disk_space_gb: float = Field(..., ge=0.0, description="Available disk space in GB")

    model_config = {"use_enum_values": True}


class ValidationIssue(BaseModel):
    """Represents a validation issue."""

    severity: str = Field(..., description="Issue severity: error, warning, info")
    category: str = Field(..., description="Issue category")
    message: str = Field(..., description="Human-readable issue description")
    suggestion: str | None = Field(None, description="Suggested fix")
    field: str | None = Field(None, description="Configuration field causing issue")

    model_config = {"use_enum_values": True}


class ValidationResult(BaseModel):
    """Result of model configuration validation."""

    valid: bool = Field(..., description="Whether configuration is valid")
    issues: list[ValidationIssue] = Field(default_factory=list, description="Validation issues")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    recommendations: list[str] = Field(
        default_factory=list, description="Optimization recommendations"
    )
    estimated_memory_mb: float | None = Field(None, description="Estimated memory usage")
    compatibility_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Hardware compatibility score"
    )

    model_config = {"use_enum_values": True}


class HardwareCompatibility(BaseModel):
    """Hardware compatibility assessment."""

    compatible: bool = Field(..., description="Whether hardware is compatible")
    memory_sufficient: bool = Field(..., description="Whether memory is sufficient")
    gpu_compatible: bool = Field(default=True, description="GPU compatibility")
    neural_engine_supported: bool = Field(default=False, description="Neural Engine support")
    estimated_load_time_s: float | None = Field(None, ge=0.0, description="Estimated load time")
    performance_tier: str = Field(..., description="Expected performance tier")
    bottlenecks: list[str] = Field(default_factory=list, description="Identified bottlenecks")
    recommendations: list[str] = Field(default_factory=list, description="Hardware recommendations")

    model_config = {"use_enum_values": True}


class ModelRecommendations(BaseModel):
    """Model optimization recommendations."""

    optimal_batch_size: int = Field(..., ge=1, description="Recommended batch size")
    recommended_quantization: str | None = Field(None, description="Suggested quantization")
    memory_optimization_tips: list[str] = Field(
        default_factory=list, description="Memory optimization tips"
    )
    performance_expectations: dict[str, str] = Field(
        default_factory=dict, description="Performance expectations"
    )
    alternative_configs: list[dict[str, Any]] = Field(
        default_factory=list, description="Alternative configurations"
    )
    resource_allocation: dict[str, Any] = Field(
        default_factory=dict, description="Optimal resource allocation"
    )

    model_config = {"use_enum_values": True}


class CompatibilityReport(BaseModel):
    """Report for multiple model compatibility."""

    compatible: bool = Field(..., description="Whether all models can run together")
    total_memory_required_gb: float = Field(..., ge=0.0, description="Total memory requirement")
    memory_available_gb: float = Field(..., ge=0.0, description="Available memory")
    conflicts: list[str] = Field(default_factory=list, description="Model conflicts")
    scheduling_recommendations: list[str] = Field(
        default_factory=list, description="Scheduling suggestions"
    )
    model_priorities: dict[str, int] = Field(
        default_factory=dict, description="Recommended loading order"
    )
    resource_sharing_strategy: str = Field(
        default="sequential", description="Resource sharing strategy"
    )

    model_config = {"use_enum_values": True}


class ModelValidator:
    """
    Comprehensive model configuration validator and compatibility checker.

    Provides validation for model configurations, hardware compatibility assessment,
    API access validation, and optimization recommendations.
    """

    def __init__(self, hardware_info: HardwareInfo | None = None):
        """
        Initialize the model validator.

        Args:
            hardware_info: System hardware information. If None, will auto-detect.
        """
        self.logger = get_logger("model_validator")
        self.hardware_info = hardware_info or self._detect_hardware()

        # Model type requirements (estimated)
        self.model_requirements: dict[str, dict[str, Any]] = {
            "mlx": {
                "min_memory_gb": 8.0,
                "apple_silicon_required": True,
                "gpu_acceleration": True,
                "typical_models": {
                    "7b": {"memory_gb": 4.0, "batch_size": 32},
                    "13b": {"memory_gb": 8.0, "batch_size": 16},
                    "30b": {"memory_gb": 16.0, "batch_size": 8},
                    "70b": {"memory_gb": 40.0, "batch_size": 4},
                },
            },
            "ollama": {
                "min_memory_gb": 4.0,
                "apple_silicon_required": False,
                "gpu_acceleration": False,
                "typical_models": {
                    "7b": {"memory_gb": 4.0, "batch_size": 16},
                    "13b": {"memory_gb": 8.0, "batch_size": 8},
                    "30b": {"memory_gb": 16.0, "batch_size": 4},
                },
            },
            "api": {
                "min_memory_gb": 0.1,
                "apple_silicon_required": False,
                "gpu_acceleration": False,
                "requires_network": True,
                "requires_credentials": True,
            },
            "transformers": {
                "min_memory_gb": 2.0,
                "apple_silicon_required": False,
                "gpu_acceleration": True,
                "typical_models": {
                    "small": {"memory_gb": 1.0, "batch_size": 64},
                    "base": {"memory_gb": 2.0, "batch_size": 32},
                    "large": {"memory_gb": 4.0, "batch_size": 16},
                },
            },
        }

        # API endpoints for validation
        self.api_endpoints: dict[str, str] = {
            "openai": "https://api.openai.com/v1/models",
            "anthropic": "https://api.anthropic.com/v1/models",
            "cohere": "https://api.cohere.ai/v1/models",
            "huggingface": "https://huggingface.co/api/models",
        }

    def _detect_hardware(self) -> HardwareInfo:
        """Auto-detect system hardware information."""
        try:
            # Basic system info
            memory_bytes = psutil.virtual_memory().total
            memory_gb = memory_bytes / (1024**3)
            cpu_cores = psutil.cpu_count(logical=True) or 1

            # Disk space
            disk_usage = psutil.disk_usage("/")
            disk_space_gb = disk_usage.free / (1024**3)

            # Platform info
            system_platform = platform.system()
            architecture = platform.machine()

            # Apple Silicon detection
            apple_silicon = system_platform == "Darwin" and "arm64" in architecture.lower()

            # Neural Engine detection (M1/M2/M3/M4 chips)
            neural_engine_available = apple_silicon

            # GPU memory detection (simplified)
            gpu_memory_gb = None
            if apple_silicon:
                # On Apple Silicon, GPU shares system memory
                gpu_memory_gb = memory_gb * 0.6  # Rough estimate
            elif system_platform == "Linux":
                try:
                    # Try to detect NVIDIA GPU
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        gpu_memory_mb = float(result.stdout.strip())
                        gpu_memory_gb = gpu_memory_mb / 1024
                except (
                    subprocess.TimeoutExpired,
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                ):
                    pass

            return HardwareInfo(
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                gpu_memory_gb=gpu_memory_gb,
                neural_engine_available=neural_engine_available,
                apple_silicon=apple_silicon,
                platform=system_platform,
                architecture=architecture,
                disk_space_gb=disk_space_gb,
            )

        except Exception as e:
            self.logger.warning(f"Failed to detect hardware: {e}")
            # Return conservative defaults
            return HardwareInfo(
                cpu_cores=4,
                memory_gb=8.0,
                gpu_memory_gb=None,
                neural_engine_available=False,
                apple_silicon=False,
                platform=platform.system(),
                architecture=platform.machine(),
                disk_space_gb=10.0,
            )

    async def validate_model_config(self, config: dict[str, Any]) -> ValidationResult:
        """
        Validate individual model configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            ValidationResult with validation details
        """
        issues: list[ValidationIssue] = []
        warnings: list[str] = []
        recommendations: list[str] = []
        estimated_memory_mb: float | None = None
        compatibility_score = 1.0

        try:
            # Check required fields
            required_fields = ["type", "name"]
            for field in required_fields:
                if field not in config:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="configuration",
                            message=f"Missing required field: {field}",
                            suggestion=f"Add '{field}' field to model configuration",
                            field=field,
                        )
                    )

            if not config.get("type"):
                return ValidationResult(
                    valid=False,
                    issues=issues,
                    warnings=warnings,
                    recommendations=recommendations,
                    estimated_memory_mb=None,
                )

            model_type = config["type"]

            # Validate model type
            if model_type not in self.model_requirements:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="model_type",
                        message=f"Unsupported model type: {model_type}",
                        suggestion=f"Use one of: {', '.join(self.model_requirements.keys())}",
                        field="type",
                    )
                )
                return ValidationResult(
                    valid=False,
                    issues=issues,
                    warnings=warnings,
                    recommendations=recommendations,
                    estimated_memory_mb=None,
                )

            requirements = self.model_requirements[model_type]

            # Check Apple Silicon requirement
            if (
                requirements.get("apple_silicon_required", False)
                and not self.hardware_info.apple_silicon
            ):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="hardware",
                        message=f"Model type '{model_type}' requires Apple Silicon",
                        suggestion="Use a different model type or run on Apple Silicon hardware",
                        field="type",
                    )
                )
                compatibility_score *= 0.0

            # Estimate memory requirements
            estimated_memory_mb = self._estimate_memory_usage(config)
            if estimated_memory_mb:
                memory_gb = estimated_memory_mb / 1024
                if memory_gb > self.hardware_info.memory_gb:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="memory",
                            message=f"Model requires ~{memory_gb:.1f}GB RAM, only {self.hardware_info.memory_gb:.1f}GB available",
                            suggestion="Use a smaller model or add more RAM",
                            field="model",
                        )
                    )
                    compatibility_score *= 0.2
                elif memory_gb > self.hardware_info.memory_gb * 0.8:
                    warnings.append(
                        f"Model will use {memory_gb:.1f}GB of {self.hardware_info.memory_gb:.1f}GB available RAM"
                    )
                    compatibility_score *= 0.8

            # API-specific validation
            if model_type == "api":
                api_valid = await self._validate_api_config(config)
                if not api_valid:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="api",
                            message="Invalid API configuration or credentials",
                            suggestion="Check API endpoint, credentials, and network connectivity",
                            field="api",
                        )
                    )
                    compatibility_score *= 0.3

            # Model-specific validation
            await self._validate_model_specific_config(config, issues, warnings, recommendations)

            # Generate recommendations
            if not recommendations:
                recommendations.extend(await self._generate_basic_recommendations(config))

            is_valid = not any(issue.severity == "error" for issue in issues)

            return ValidationResult(
                valid=is_valid,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                estimated_memory_mb=estimated_memory_mb,
                compatibility_score=compatibility_score,
            )

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="internal",
                    message=f"Validation failed: {str(e)}",
                    suggestion="Check configuration format and try again",
                    field=None,
                )
            )

            return ValidationResult(
                valid=False,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                estimated_memory_mb=None,
            )

    def _estimate_memory_usage(self, config: dict[str, Any]) -> float | None:
        """Estimate memory usage for a model configuration."""
        try:
            model_type = config.get("type", "")
            requirements = self.model_requirements.get(model_type, {})

            # Base memory for model type
            base_memory = requirements.get("min_memory_gb", 1.0) * 1024  # Convert to MB

            # Model size estimation
            model_name = config.get("name", "").lower()
            model_path = config.get("model_path", "").lower()

            # Try to extract model size from name/path
            size_multipliers = {
                "7b": 4.0,
                "13b": 8.0,
                "30b": 16.0,
                "70b": 40.0,
                "small": 1.0,
                "base": 2.0,
                "large": 4.0,
                "xl": 8.0,
                "mini": 0.5,
                "tiny": 0.25,
            }

            multiplier = 1.0
            for size_key, mult in size_multipliers.items():
                if size_key in model_name or size_key in model_path:
                    multiplier = mult
                    break

            estimated_mb = base_memory * multiplier

            # Batch size adjustment
            batch_size = config.get("batch_size", 1)
            if batch_size > 1:
                estimated_mb *= 1 + batch_size * 0.1  # Rough batch size impact

            return float(estimated_mb)

        except Exception:
            return None

    async def _validate_api_config(self, config: dict[str, Any]) -> bool:
        """Validate API configuration and credentials."""
        try:
            api_provider = config.get("provider", "").lower()
            api_key = config.get("api_key") or os.getenv(f"{api_provider.upper()}_API_KEY")
            endpoint = config.get("endpoint")

            if not api_key:
                return False

            if not endpoint and api_provider in self.api_endpoints:
                endpoint = self.api_endpoints[api_provider]

            if not endpoint:
                return False

            # Simple connectivity check
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    async with session.get(endpoint, headers=headers) as response:
                        return response.status in [
                            200,
                            401,
                            403,
                        ]  # 401/403 means endpoint is reachable
            except Exception:
                return False

        except Exception:
            return False

    async def _validate_model_specific_config(
        self,
        config: dict[str, Any],
        issues: list[ValidationIssue],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate model-type-specific configuration."""
        model_type = config.get("type", "")

        if model_type == "mlx":
            # MLX-specific validation
            if not config.get("model_path"):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="configuration",
                        message="MLX models require 'model_path' field",
                        suggestion="Specify path to MLX model files",
                        field="model_path",
                    )
                )

            if config.get("quantization") and config["quantization"] not in ["4bit", "8bit"]:
                warnings.append("Unsupported quantization setting for MLX")

        elif model_type == "ollama":
            # Ollama-specific validation
            if not config.get("model_name"):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="configuration",
                        message="Ollama models require 'model_name' field",
                        suggestion="Specify Ollama model name (e.g., 'llama2', 'mistral')",
                        field="model_name",
                    )
                )

        elif model_type == "api":
            # API-specific validation
            required_api_fields = ["provider", "model_name"]
            for field in required_api_fields:
                if not config.get(field):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="configuration",
                            message=f"API models require '{field}' field",
                            suggestion=f"Specify {field} for API model",
                            field=field,
                        )
                    )

    async def _generate_basic_recommendations(self, config: dict[str, Any]) -> list[str]:
        """Generate basic optimization recommendations."""
        recommendations = []
        model_type = config.get("type", "")

        # Memory optimization
        if self.hardware_info.memory_gb < 16:
            recommendations.append("Consider using quantized models to reduce memory usage")

        # Batch size recommendations
        if not config.get("batch_size"):
            if self.hardware_info.memory_gb >= 32:
                recommendations.append("Consider setting batch_size to 32 for better throughput")
            elif self.hardware_info.memory_gb >= 16:
                recommendations.append("Consider setting batch_size to 16 for balanced performance")
            else:
                recommendations.append("Consider setting batch_size to 8 or lower for stability")

        # Hardware-specific recommendations
        if model_type == "mlx" and self.hardware_info.apple_silicon:
            recommendations.append("MLX models are optimized for your Apple Silicon hardware")

        if self.hardware_info.neural_engine_available:
            recommendations.append("Your system supports Neural Engine acceleration")

        return recommendations

    async def validate_model_compatibility(
        self, configs: list[dict[str, Any]]
    ) -> CompatibilityReport:
        """
        Check if multiple models can be run together.

        Args:
            configs: List of model configurations

        Returns:
            CompatibilityReport with compatibility assessment
        """
        try:
            total_memory_required = 0.0
            conflicts = []
            model_priorities = {}

            # Analyze each model
            for i, config in enumerate(configs):
                model_id = config.get("name", f"model_{i}")

                # Estimate memory for each model
                memory_mb = self._estimate_memory_usage(config)
                if memory_mb:
                    total_memory_required += memory_mb / 1024  # Convert to GB

                # Check for type conflicts
                model_type = config.get("type", "")
                if model_type == "mlx" and not self.hardware_info.apple_silicon:
                    conflicts.append(
                        f"Model {model_id} requires Apple Silicon but hardware is incompatible"
                    )

                # Set priorities (lower number = higher priority)
                if model_type == "api":
                    model_priorities[model_id] = 1  # APIs load fastest
                elif model_type == "ollama":
                    model_priorities[model_id] = 2
                else:
                    model_priorities[model_id] = 3

            memory_available = self.hardware_info.memory_gb * 0.8  # Reserve 20% for system
            compatible = total_memory_required <= memory_available and not conflicts

            # Generate scheduling recommendations
            scheduling_recommendations = []
            if total_memory_required > memory_available:
                scheduling_recommendations.append(
                    "Sequential loading recommended due to memory constraints"
                )
                scheduling_recommendations.append("Consider unloading models between evaluations")
            else:
                scheduling_recommendations.append(
                    "Concurrent loading possible with sufficient memory"
                )

            if len(configs) > 3:
                scheduling_recommendations.append(
                    "Large number of models may benefit from batch processing"
                )

            # Resource sharing strategy
            if total_memory_required > memory_available:
                resource_sharing = "sequential"
            elif total_memory_required > memory_available * 0.7:
                resource_sharing = "time-shared"
            else:
                resource_sharing = "concurrent"

            return CompatibilityReport(
                compatible=compatible,
                total_memory_required_gb=total_memory_required,
                memory_available_gb=memory_available,
                conflicts=conflicts,
                scheduling_recommendations=scheduling_recommendations,
                model_priorities=model_priorities,
                resource_sharing_strategy=resource_sharing,
            )

        except Exception as e:
            self.logger.error(f"Compatibility check failed: {e}")
            return CompatibilityReport(
                compatible=False,
                total_memory_required_gb=0.0,
                memory_available_gb=self.hardware_info.memory_gb,
                conflicts=[f"Compatibility check failed: {str(e)}"],
            )

    async def check_hardware_requirements(self, config: dict[str, Any]) -> HardwareCompatibility:
        """
        Check if model fits hardware constraints.

        Args:
            config: Model configuration

        Returns:
            HardwareCompatibility assessment
        """
        try:
            model_type = config.get("type", "")
            requirements = self.model_requirements.get(model_type, {})

            # Memory check
            estimated_memory_mb = self._estimate_memory_usage(config)
            memory_gb = (estimated_memory_mb or 0) / 1024
            memory_sufficient = memory_gb <= self.hardware_info.memory_gb * 0.8

            # Apple Silicon check
            apple_silicon_required = requirements.get("apple_silicon_required", False)
            apple_silicon_compatible = (
                not apple_silicon_required or self.hardware_info.apple_silicon
            )

            # GPU compatibility
            gpu_compatible = True
            if requirements.get("gpu_acceleration") and not self.hardware_info.gpu_memory_gb:
                gpu_compatible = False

            # Neural Engine support
            neural_engine_supported = self.hardware_info.neural_engine_available and model_type in [
                "mlx",
                "coreml",
            ]

            # Overall compatibility
            compatible = memory_sufficient and apple_silicon_compatible

            # Performance tier estimation
            if self.hardware_info.apple_silicon and self.hardware_info.memory_gb >= 64:
                performance_tier = "high"
            elif self.hardware_info.memory_gb >= 32:
                performance_tier = "medium"
            elif self.hardware_info.memory_gb >= 16:
                performance_tier = "low"
            else:
                performance_tier = "minimal"

            # Estimated load time
            base_load_time = 10.0  # Base load time in seconds
            if memory_gb > 10:
                base_load_time += (memory_gb - 10) * 2  # 2s per GB above 10GB
            if not self.hardware_info.apple_silicon and model_type == "transformers":
                base_load_time *= 1.5  # Slower loading without optimized hardware

            # Identify bottlenecks
            bottlenecks = []
            if not memory_sufficient:
                bottlenecks.append(
                    f"Insufficient memory: need {memory_gb:.1f}GB, have {self.hardware_info.memory_gb:.1f}GB"
                )
            if apple_silicon_required and not self.hardware_info.apple_silicon:
                bottlenecks.append("Requires Apple Silicon processor")
            if self.hardware_info.cpu_cores < 4:
                bottlenecks.append("Low CPU core count may affect performance")

            # Hardware recommendations
            hardware_recommendations = []
            if not memory_sufficient:
                recommended_memory = max(16, int(memory_gb * 1.5))
                hardware_recommendations.append(f"Upgrade to {recommended_memory}GB+ RAM")
            if model_type == "mlx" and not self.hardware_info.apple_silicon:
                hardware_recommendations.append("Use Apple Silicon Mac for optimal MLX performance")
            if requirements.get("gpu_acceleration") and not self.hardware_info.gpu_memory_gb:
                hardware_recommendations.append("Add dedicated GPU for acceleration")

            return HardwareCompatibility(
                compatible=compatible,
                memory_sufficient=memory_sufficient,
                gpu_compatible=gpu_compatible,
                neural_engine_supported=neural_engine_supported,
                estimated_load_time_s=base_load_time,
                performance_tier=performance_tier,
                bottlenecks=bottlenecks,
                recommendations=hardware_recommendations,
            )

        except Exception as e:
            self.logger.error(f"Hardware compatibility check failed: {e}")
            return HardwareCompatibility(
                compatible=False,
                memory_sufficient=False,
                performance_tier="unknown",
                bottlenecks=[f"Compatibility check failed: {str(e)}"],
                estimated_load_time_s=None,
            )

    async def validate_api_access(self, config: dict[str, Any]) -> bool:
        """
        Validate API access for cloud models.

        Args:
            config: Model configuration with API details

        Returns:
            True if API access is valid
        """
        return await self._validate_api_config(config)

    async def recommend_model_settings(self, config: dict[str, Any]) -> ModelRecommendations:
        """
        Provide optimized settings for model.

        Args:
            config: Model configuration

        Returns:
            ModelRecommendations with optimization suggestions
        """
        try:
            model_type = config.get("type", "")

            # Optimal batch size based on memory
            if self.hardware_info.memory_gb >= 64:
                optimal_batch_size = 64
            elif self.hardware_info.memory_gb >= 32:
                optimal_batch_size = 32
            elif self.hardware_info.memory_gb >= 16:
                optimal_batch_size = 16
            else:
                optimal_batch_size = 8

            # Adjust for model type
            if model_type == "mlx":
                optimal_batch_size = min(
                    optimal_batch_size, 32
                )  # MLX works well with smaller batches
            elif model_type == "api":
                optimal_batch_size = min(optimal_batch_size, 20)  # API rate limits

            # Quantization recommendation
            recommended_quantization = None
            if self.hardware_info.memory_gb < 16:
                recommended_quantization = "4bit"
            elif self.hardware_info.memory_gb < 32:
                recommended_quantization = "8bit"

            # Memory optimization tips
            memory_tips = []
            if self.hardware_info.memory_gb < 32:
                memory_tips.extend(
                    [
                        "Use gradient checkpointing if available",
                        "Consider smaller context windows",
                        "Enable memory mapping for large models",
                    ]
                )

            if model_type == "transformers":
                memory_tips.append("Use torch.compile() for memory efficiency")

            # Performance expectations
            performance_expectations = {
                "inference_speed": self._estimate_inference_speed(config),
                "memory_usage": f"{self._estimate_memory_usage(config) or 0 / 1024:.1f}GB",
                "throughput": f"{optimal_batch_size} samples/batch",
            }

            # Alternative configurations
            alternative_configs = []
            if model_type == "mlx":
                alternative_configs.append(
                    {
                        "type": "ollama",
                        "description": "Fallback for non-Apple Silicon systems",
                        "trade_offs": "May be slower but more compatible",
                    }
                )
            elif model_type == "transformers":
                alternative_configs.append(
                    {
                        "type": "api",
                        "description": "Cloud-based alternative",
                        "trade_offs": "Requires internet but uses no local resources",
                    }
                )

            # Resource allocation
            resource_allocation = {
                "cpu_threads": min(self.hardware_info.cpu_cores, 8),
                "memory_limit_gb": self.hardware_info.memory_gb * 0.8,
                "batch_size": optimal_batch_size,
            }

            if self.hardware_info.gpu_memory_gb:
                resource_allocation["gpu_memory_gb"] = self.hardware_info.gpu_memory_gb * 0.9

            return ModelRecommendations(
                optimal_batch_size=optimal_batch_size,
                recommended_quantization=recommended_quantization,
                memory_optimization_tips=memory_tips,
                performance_expectations=performance_expectations,
                alternative_configs=alternative_configs,
                resource_allocation=resource_allocation,
            )

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return ModelRecommendations(
                optimal_batch_size=8,
                recommended_quantization=None,
                memory_optimization_tips=[f"Recommendation generation failed: {str(e)}"],
                performance_expectations={"status": "unknown"},
            )

    def _estimate_inference_speed(self, config: dict[str, Any]) -> str:
        """Estimate inference speed category."""
        model_type = config.get("type", "")

        if model_type == "api":
            return "fast (cloud processing)"
        elif model_type == "mlx" and self.hardware_info.apple_silicon:
            return "very fast (optimized)"
        elif self.hardware_info.memory_gb >= 32:
            return "fast"
        elif self.hardware_info.memory_gb >= 16:
            return "moderate"
        else:
            return "slow"

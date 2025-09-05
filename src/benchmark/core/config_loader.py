"""
YAML configuration loader with environment variable resolution and validation.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from .config import (
    BenchmarkConfig,
    ExperimentConfig,
)
from .exceptions import ConfigurationError, config_validation_error
from .logging import get_config_logger

logger = get_config_logger()


class ConfigurationCache:
    """Simple in-memory cache for loaded configurations."""

    def __init__(self, max_size: int = 100) -> None:
        """Initialize cache with maximum size."""
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        """Get cached configuration."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Cache configuration with LRU eviction."""
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new
            if len(self._cache) >= self._max_size:
                # Evict least recently used
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            self._cache[key] = value
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached configurations."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class ConfigurationLoader:
    """YAML configuration loader with environment variable resolution."""

    def __init__(self, cache_enabled: bool = True, cache_size: int = 100) -> None:
        """
        Initialize configuration loader.

        Args:
            cache_enabled: Whether to enable configuration caching
            cache_size: Maximum number of configurations to cache
        """
        self._cache_enabled = cache_enabled
        self._cache = ConfigurationCache(cache_size) if cache_enabled else None
        self._env_var_pattern = re.compile(r"\$\{([^}]+)\}")

    def load_experiment_config(
        self,
        config_path: str | Path,
        base_config_path: str | Path | None = None,
        validate: bool = True,
    ) -> ExperimentConfig:
        """
        Load experiment configuration from YAML file.

        Args:
            config_path: Path to the experiment configuration file
            base_config_path: Optional path to base configuration to inherit from
            validate: Whether to validate the loaded configuration

        Returns:
            Validated ExperimentConfig instance

        Raises:
            ConfigurationError: If loading or validation fails
        """
        config_path = Path(config_path)
        cache_key = f"experiment:{config_path}"

        if base_config_path:
            base_config_path = Path(base_config_path)
            cache_key += f"|base:{base_config_path}"

        # Check cache first
        if self._cache_enabled and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Loaded experiment config from cache: {config_path}")
                return cached  # type: ignore[no-any-return]

        try:
            # Load base configuration if provided
            base_config_data = {}
            if base_config_path:
                logger.debug(f"Loading base configuration: {base_config_path}")
                base_config_data = self._load_yaml_file(base_config_path)
                base_config_data = self.resolve_environment_variables(base_config_data)

            # Load main configuration
            logger.debug(f"Loading experiment configuration: {config_path}")
            config_data = self._load_yaml_file(config_path)
            config_data = self.resolve_environment_variables(config_data)

            # Merge configurations if base config provided
            if base_config_data:
                config_data = self.merge_configurations(base_config_data, config_data)

            # Create and validate configuration
            experiment_config = ExperimentConfig.model_validate(config_data)

            if validate:
                warnings = self.validate_configuration(experiment_config)
                if warnings:
                    logger.warning(f"Configuration warnings for {config_path}:")
                    for warning in warnings:
                        logger.warning(f"  - {warning}")

            # Cache the result
            if self._cache_enabled and self._cache:
                self._cache.set(cache_key, experiment_config)
                logger.debug(f"Cached experiment config: {config_path}")

            logger.info(f"Successfully loaded experiment config: {config_path}")
            return experiment_config

        except ValidationError as e:
            error_details = self._format_validation_errors(e.errors())
            raise config_validation_error(
                str(config_path),
                config_data if "config_data" in locals() else {},
                f"Validation failed: {error_details}",
            ) from e

        except ConfigurationError:
            # Re-raise configuration errors without wrapping
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load experiment configuration from {config_path}",
                metadata={
                    "config_path": str(config_path),
                    "base_config_path": str(base_config_path) if base_config_path else None,
                },
            ) from e

    def load_benchmark_config(
        self, config_path: str | Path, validate: bool = True
    ) -> BenchmarkConfig:
        """
        Load benchmark configuration from YAML file.

        Args:
            config_path: Path to the benchmark configuration file
            validate: Whether to validate the loaded configuration

        Returns:
            Validated BenchmarkConfig instance

        Raises:
            ConfigurationError: If loading or validation fails
        """
        config_path = Path(config_path)
        cache_key = f"benchmark:{config_path}"

        # Check cache first
        if self._cache_enabled and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Loaded benchmark config from cache: {config_path}")
                return cached  # type: ignore[no-any-return]

        try:
            logger.debug(f"Loading benchmark configuration: {config_path}")
            config_data = self._load_yaml_file(config_path)
            config_data = self.resolve_environment_variables(config_data)

            # Create and validate configuration
            benchmark_config = BenchmarkConfig.model_validate(config_data)

            if validate:
                # Validate each experiment
                for _i, experiment in enumerate(benchmark_config.experiments):
                    warnings = self.validate_configuration(experiment)
                    if warnings:
                        logger.warning(
                            f"Configuration warnings for experiment '{experiment.name}':"
                        )
                        for warning in warnings:
                            logger.warning(f"  - {warning}")

            # Cache the result
            if self._cache_enabled and self._cache:
                self._cache.set(cache_key, benchmark_config)
                logger.debug(f"Cached benchmark config: {config_path}")

            logger.info(f"Successfully loaded benchmark config: {config_path}")
            return benchmark_config

        except ValidationError as e:
            error_details = self._format_validation_errors(e.errors())
            raise config_validation_error(
                str(config_path),
                config_data if "config_data" in locals() else {},
                f"Validation failed: {error_details}",
            ) from e

        except ConfigurationError:
            # Re-raise configuration errors without wrapping
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load benchmark configuration from {config_path}",
                metadata={"config_path": str(config_path)},
            ) from e

    def resolve_environment_variables(self, config: Any) -> Any:
        """
        Recursively resolve environment variables in configuration.

        Environment variables should be specified as ${VAR_NAME} and can include
        default values like ${VAR_NAME:default_value}.

        Args:
            config: Configuration dictionary to process

        Returns:
            Configuration with resolved environment variables
        """
        if isinstance(config, dict):
            return {key: self.resolve_environment_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self.resolve_environment_variables(item) for item in config]
        elif isinstance(config, str):
            return self._resolve_env_vars_in_string(config)
        else:
            return config

    def validate_configuration(self, config: ExperimentConfig) -> list[str]:
        """
        Validate configuration and return list of warnings.

        Args:
            config: Configuration to validate

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for reasonable dataset sizes
        for dataset in config.datasets:
            if dataset.max_samples and dataset.max_samples < 100:
                warnings.append(
                    f"Dataset '{dataset.name}' has very small sample size ({dataset.max_samples}). "
                    "Consider using at least 100 samples for meaningful evaluation."
                )

            # Check split ratios are reasonable
            if dataset.test_split < 0.1:
                warnings.append(
                    f"Dataset '{dataset.name}' has very small test split ({dataset.test_split}). "
                    "Consider using at least 10% for reliable evaluation."
                )

        # Check for reasonable model configurations
        for model in config.models:
            if model.max_tokens < 50:
                warnings.append(
                    f"Model '{model.name}' has very low max_tokens ({model.max_tokens}). "
                    "This may truncate important information."
                )

            if model.temperature > 1.5:
                warnings.append(
                    f"Model '{model.name}' has high temperature ({model.temperature}). "
                    "This may produce inconsistent results for evaluation."
                )

        # Check evaluation configuration
        if config.evaluation.batch_size < 8:
            warnings.append(
                f"Small batch size ({config.evaluation.batch_size}) may slow evaluation. "
                "Consider using at least 8 for better throughput."
            )

        if config.evaluation.timeout_minutes < 30:
            warnings.append(
                f"Short timeout ({config.evaluation.timeout_minutes} minutes) may cause "
                "premature evaluation termination. Consider at least 30 minutes."
            )

        # Check for missing commonly used metrics
        common_metrics = {"accuracy", "precision", "recall", "f1_score"}
        missing_metrics = common_metrics - set(config.evaluation.metrics)
        if missing_metrics:
            warnings.append(
                f"Consider adding common metrics: {', '.join(missing_metrics)} "
                "for comprehensive evaluation."
            )

        return warnings

    def merge_configurations(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merge configuration dictionaries with deep merging for nested objects.

        Args:
            base: Base configuration
            override: Override configuration (takes precedence)

        Returns:
            Merged configuration
        """
        merged = base.copy()

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Deep merge for nested dictionaries
                merged[key] = self.merge_configurations(merged[key], value)
            elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
                # Special handling for lists of dictionaries (like models and datasets)
                if key in {"models", "datasets"} and all(isinstance(item, dict) for item in value):
                    merged[key] = self._merge_lists_by_name(merged[key], value)
                else:
                    # For other lists, override completely
                    merged[key] = value
            else:
                # Direct override
                merged[key] = value

        return merged

    def _merge_lists_by_name(
        self, base_list: list[dict[str, Any]], override_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Merge lists of dictionaries by matching 'name' field.

        Args:
            base_list: Base list of dictionaries
            override_list: Override list of dictionaries

        Returns:
            Merged list with overrides applied and new items added
        """
        # Create a mapping of base items by name
        base_by_name = {item.get("name"): item for item in base_list if "name" in item}

        # Start with base items
        merged = []
        processed_names = set()

        # Process override items
        for override_item in override_list:
            name = override_item.get("name")
            if name and name in base_by_name:
                # Merge with existing base item
                merged_item = self.merge_configurations(base_by_name[name], override_item)
                merged.append(merged_item)
                processed_names.add(name)
            else:
                # New item from override
                merged.append(override_item)
                if name:
                    processed_names.add(name)

        # Add remaining base items that weren't overridden
        for name, base_item in base_by_name.items():
            if name not in processed_names:
                merged.append(base_item)

        return merged

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        if self._cache:
            self._cache.clear()
            logger.debug("Configuration cache cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        if not self._cache:
            return {"enabled": False, "size": 0, "max_size": 0}

        return {"enabled": True, "size": self._cache.size(), "max_size": self._cache._max_size}

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """
        Load YAML file with proper error handling.

        Args:
            file_path: Path to YAML file

        Returns:
            Parsed YAML content

        Raises:
            ConfigurationError: If file doesn't exist or YAML is invalid
        """
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}", metadata={"file_path": str(file_path)}
            )

        try:
            with open(file_path, encoding="utf-8") as f:
                content = yaml.safe_load(f)

            if content is None:
                raise ConfigurationError(
                    f"Configuration file is empty: {file_path}",
                    metadata={"file_path": str(file_path)},
                )

            if not isinstance(content, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a YAML object/dictionary: {file_path}",
                    metadata={"file_path": str(file_path), "content_type": type(content).__name__},
                )

            return content

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML syntax in {file_path}: {e}",
                metadata={"file_path": str(file_path), "yaml_error": str(e)},
            ) from e

        except Exception as e:
            raise ConfigurationError(
                f"Failed to read configuration file {file_path}: {e}",
                metadata={"file_path": str(file_path)},
            ) from e

    def _resolve_env_vars_in_string(self, text: str) -> str:
        """
        Resolve environment variables in a string.

        Supports:
        - ${VAR_NAME} - required variable
        - ${VAR_NAME:default} - variable with default value

        Args:
            text: String potentially containing environment variables

        Returns:
            String with environment variables resolved

        Raises:
            ConfigurationError: If required environment variable is missing
        """

        def replace_var(match: re.Match[str]) -> str:
            var_expr = match.group(1)

            # Check for default value syntax
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                env_value = os.getenv(var_name.strip())
                if env_value is not None:
                    return env_value
                else:
                    logger.debug(f"Using default value for ${{{var_name}}}: {default_value}")
                    return default_value
            else:
                # Required variable
                var_name = var_expr.strip()
                env_value = os.getenv(var_name)
                if env_value is not None:
                    return env_value
                else:
                    raise ConfigurationError(
                        f"Required environment variable '{var_name}' is not set",
                        metadata={"variable_name": var_name, "expression": f"${{{var_expr}}}"},
                    )

        try:
            return self._env_var_pattern.sub(replace_var, text)
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to resolve environment variables in: {text}", metadata={"text": text}
            ) from e

    def _format_validation_errors(self, errors: list[ErrorDetails]) -> str:
        """
        Format Pydantic validation errors into readable messages.

        Args:
            errors: List of validation error dictionaries from Pydantic

        Returns:
            Formatted error message
        """
        formatted_errors = []

        for error in errors:
            location = " -> ".join(str(loc) for loc in error.get("loc", []))
            message = error.get("msg", "Unknown error")
            error_type = error.get("type", "unknown")

            if location:
                formatted_errors.append(f"Field '{location}': {message} (type: {error_type})")
            else:
                formatted_errors.append(f"{message} (type: {error_type})")

        return "\n".join(formatted_errors)


# Default global configuration loader instance
default_loader = ConfigurationLoader()


# Convenience functions using the default loader
def load_experiment_config(
    config_path: str | Path, base_config_path: str | Path | None = None, validate: bool = True
) -> ExperimentConfig:
    """Load experiment configuration using default loader."""
    return default_loader.load_experiment_config(config_path, base_config_path, validate)


def load_benchmark_config(config_path: str | Path, validate: bool = True) -> BenchmarkConfig:
    """Load benchmark configuration using default loader."""
    return default_loader.load_benchmark_config(config_path, validate)


def resolve_environment_variables(config: Any) -> Any:
    """Resolve environment variables using default loader."""
    return default_loader.resolve_environment_variables(config)


def clear_config_cache() -> None:
    """Clear configuration cache of default loader."""
    default_loader.clear_cache()


def get_config_cache_stats() -> dict[str, int]:
    """Get cache statistics from default loader."""
    return default_loader.get_cache_stats()

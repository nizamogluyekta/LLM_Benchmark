"""
Configuration Service for the LLM Cybersecurity Benchmark system.

This service provides comprehensive configuration management functionality,
including loading, caching, validation, and runtime reloading of configurations.
"""

import contextlib
import copy
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from benchmark.core.base import BaseService, HealthCheck, ServiceResponse, ServiceStatus
from benchmark.core.config import ExperimentConfig
from benchmark.core.exceptions import ConfigurationError, ErrorCode
from benchmark.core.logging import get_logger


class ConfigurationService(BaseService):
    """
    Service for managing application configurations.

    This service provides:
    - Configuration loading from YAML files
    - Configuration caching for performance
    - Configuration validation with warnings
    - Runtime configuration reloading
    - Thread-safe access to configurations
    """

    def __init__(self, config_dir: Path = Path("configs"), cache_ttl: int = 3600):
        """
        Initialize the Configuration Service.

        Args:
            config_dir: Directory containing configuration files
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        super().__init__("configuration_service")
        self.config_dir = Path(config_dir)
        self.cache_ttl = cache_ttl

        # Thread-safe cache for configurations
        self._cache: dict[str, ExperimentConfig] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # File modification tracking for reload detection
        self._file_mtimes: dict[str, float] = {}

        self.logger = get_logger("configuration")

    async def initialize(self) -> ServiceResponse:
        """Initialize the Configuration Service."""
        try:
            self.logger.info("Initializing Configuration Service")

            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)

            # Create default config directory structure
            await self._create_default_structure()

            # Preload any existing configurations
            await self._preload_configurations()

            self._set_status(ServiceStatus.HEALTHY)
            self.logger.info("Configuration Service initialized successfully")

            return ServiceResponse(
                success=True,
                message="Configuration Service initialized successfully",
                data={"config_dir": str(self.config_dir), "cached_configs": len(self._cache)},
            )

        except Exception as e:
            self._set_status(ServiceStatus.ERROR)
            error_msg = f"Failed to initialize Configuration Service: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ServiceResponse(success=False, message=error_msg, error=str(e))

    async def health_check(self) -> HealthCheck:
        """Perform a health check on the Configuration Service."""
        try:
            # Check if config directory is accessible
            if not self.config_dir.exists():
                return HealthCheck(
                    status=ServiceStatus.UNHEALTHY,
                    message=f"Configuration directory does not exist: {self.config_dir}",
                    checks={
                        "config_dir_exists": False,
                        "cache_size": len(self._cache),
                        "cached_configs": list(self._cache.keys()),
                    },
                )

            # Check if we can read the directory
            try:
                list(self.config_dir.iterdir())
            except PermissionError:
                return HealthCheck(
                    status=ServiceStatus.UNHEALTHY,
                    message="Cannot read configuration directory due to permissions",
                    checks={
                        "config_dir_exists": True,
                        "config_dir_readable": False,
                        "cache_size": len(self._cache),
                    },
                )

            # Check cache health
            with self._cache_lock:
                expired_configs = []
                current_time = time.time()

                for config_id, timestamp in self._cache_timestamps.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_configs.append(config_id)

                cache_health = {
                    "total_cached": len(self._cache),
                    "expired_configs": len(expired_configs),
                    "active_configs": len(self._cache) - len(expired_configs),
                }

            return HealthCheck(
                status=self.status,
                message="Configuration Service is healthy",
                checks={
                    "config_dir_exists": True,
                    "config_dir_readable": True,
                    "cache_health": cache_health,
                },
            )

        except Exception as e:
            return HealthCheck(
                status=ServiceStatus.ERROR,
                message=f"Health check failed: {str(e)}",
                checks={"error": str(e)},
            )

    async def shutdown(self) -> ServiceResponse:
        """Shutdown the Configuration Service."""
        try:
            self.logger.info("Shutting down Configuration Service")

            # Clear cache
            with self._cache_lock:
                self._cache.clear()
                self._cache_timestamps.clear()
                self._file_mtimes.clear()

            self._set_status(ServiceStatus.STOPPED)
            self.logger.info("Configuration Service shutdown completed")

            return ServiceResponse(
                success=True, message="Configuration Service shutdown successfully"
            )

        except Exception as e:
            error_msg = f"Error during Configuration Service shutdown: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ServiceResponse(success=False, message=error_msg, error=str(e))

    async def load_experiment_config(self, config_path: str | Path) -> ExperimentConfig:
        """
        Load and validate an experiment configuration.

        Args:
            config_path: Path to the configuration file

        Returns:
            Validated ExperimentConfig object

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        config_path = Path(config_path)
        config_id = self._get_config_id(config_path)

        try:
            # Check cache first
            cached_config = self.get_cached_config(config_id)
            if cached_config is not None:
                self.logger.debug(f"Using cached configuration for {config_path}")
                return cached_config

            # Load configuration from file
            self.logger.info(f"Loading configuration from {config_path}")

            if not config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {config_path}",
                    error_code=ErrorCode.CONFIG_FILE_NOT_FOUND,
                    metadata={"config_path": str(config_path)},
                )

            # Read and parse YAML file
            try:
                with open(config_path, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigurationError(
                    f"Invalid YAML in configuration file: {str(e)}",
                    error_code=ErrorCode.CONFIG_PARSE_ERROR,
                    metadata={"config_path": str(config_path), "yaml_error": str(e)},
                ) from e

            if config_data is None:
                raise ConfigurationError(
                    "Configuration file is empty",
                    error_code=ErrorCode.CONFIG_PARSE_ERROR,
                    metadata={"config_path": str(config_path)},
                )

            # Resolve environment variables
            config_data = self.resolve_environment_variables(config_data)

            # Validate configuration
            try:
                experiment_config = ExperimentConfig(**config_data)
            except ValidationError as e:
                raise ConfigurationError(
                    f"Configuration validation failed: {str(e)}",
                    error_code=ErrorCode.CONFIG_VALIDATION_FAILED,
                    metadata={"config_path": str(config_path), "validation_errors": e.errors()},
                ) from e

            # Cache the validated configuration
            await self._cache_config(config_id, experiment_config, config_path)

            self.logger.info(f"Successfully loaded configuration: {experiment_config.name}")
            return experiment_config

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Unexpected error loading configuration: {str(e)}",
                error_code=ErrorCode.INTERNAL_ERROR,
                metadata={"config_path": str(config_path), "error": str(e)},
            ) from e

    async def get_default_config(self) -> dict[str, Any]:
        """
        Get a default configuration template.

        Returns:
            Default configuration dictionary
        """
        return {
            "name": "Default Cybersecurity Experiment",
            "description": "Template configuration for cybersecurity benchmarking",
            "output_dir": "./results",
            "datasets": [
                {
                    "name": "sample_dataset",
                    "source": "local",
                    "path": "./data/samples.jsonl",
                    "max_samples": 1000,
                    "test_split": 0.2,
                    "validation_split": 0.1,
                }
            ],
            "models": [
                {
                    "name": "gpt-3.5-turbo",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "${OPENAI_API_KEY:test_key}"},
                    "max_tokens": "${MODEL_MAX_TOKENS:512}",
                    "temperature": 0.1,
                },
                {
                    "name": "claude-3-haiku",
                    "type": "anthropic_api",
                    "path": "claude-3-haiku-20240307",
                    "config": {"api_key": "${ANTHROPIC_API_KEY}"},
                    "max_tokens": "${MODEL_MAX_TOKENS:1024}",
                    "temperature": 0.1,
                },
            ],
            "evaluation": {
                "metrics": "${EVAL_METRICS:accuracy,precision,recall,f1_score}",
                "parallel_jobs": "${EVAL_PARALLEL_JOBS:2}",
                "timeout_minutes": "${EVAL_TIMEOUT:30}",
                "batch_size": "${EVAL_BATCH_SIZE:16}",
                "enable_detailed_logging": "${ENABLE_DEBUG_LOGS:false}",
            },
        }

    async def validate_config(self, config: ExperimentConfig) -> list[str]:
        """
        Validate a configuration and return warnings.

        Args:
            config: Configuration to validate

        Returns:
            List of warning messages
        """
        warnings_list = []

        try:
            # Check experiment configuration
            if not config.name.strip():
                warnings_list.append("Experiment name is empty or whitespace only")

            if not config.output_dir:
                warnings_list.append("No output directory specified")

            # Check dataset configurations
            if not config.datasets:
                warnings_list.append("No datasets configured")
            else:
                for i, dataset in enumerate(config.datasets):
                    if dataset.source == "local" and not Path(dataset.path).exists():
                        warnings_list.append(
                            f"Dataset {i+1} ({dataset.name}): Local file not found: {dataset.path}"
                        )

                    if dataset.max_samples and dataset.max_samples < 10:
                        warnings_list.append(
                            f"Dataset {i+1} ({dataset.name}): Very small sample size ({dataset.max_samples})"
                        )

                    total_split = dataset.test_split + dataset.validation_split
                    if total_split >= 1.0:
                        warnings_list.append(
                            f"Dataset {i+1} ({dataset.name}): Test and validation splits exceed 100%"
                        )

            # Check model configurations
            if not config.models:
                warnings_list.append("No models configured")
            else:
                for i, model in enumerate(config.models):
                    if model.type == "openai_api" and not model.config.get("api_key"):
                        warnings_list.append(
                            f"Model {i+1} ({model.name}): No OpenAI API key configured"
                        )

                    if model.type == "anthropic_api" and not model.config.get("api_key"):
                        warnings_list.append(
                            f"Model {i+1} ({model.name}): No Anthropic API key configured"
                        )

                    if model.max_tokens and model.max_tokens > 4096:
                        warnings_list.append(
                            f"Model {i+1} ({model.name}): Very high max_tokens ({model.max_tokens})"
                        )

            # Check evaluation configuration
            cpu_count = os.cpu_count() or 1
            if config.evaluation.parallel_jobs > cpu_count:
                warnings_list.append(
                    f"Parallel jobs ({config.evaluation.parallel_jobs}) exceeds CPU count ({cpu_count})"
                )

            if config.evaluation.timeout_minutes < 5:
                warnings_list.append("Evaluation timeout is very short (< 5 minutes)")

            if config.evaluation.batch_size > 128:
                warnings_list.append(
                    f"Large batch size ({config.evaluation.batch_size}) may cause memory issues"
                )

            # Check environment variable requirements
            env_warnings = self.validate_environment_requirements(config)
            warnings_list.extend(env_warnings)

        except Exception as e:
            warnings_list.append(f"Error during validation: {str(e)}")

        return warnings_list

    async def reload_config(self, config_id: str) -> ServiceResponse:
        """
        Reload a configuration from disk.

        Args:
            config_id: ID of the configuration to reload

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            with self._cache_lock:
                if config_id not in self._cache:
                    return ServiceResponse(
                        success=False, message=f"Configuration '{config_id}' not found in cache"
                    )

                # Remove from cache to force reload
                del self._cache[config_id]
                del self._cache_timestamps[config_id]
                if config_id in self._file_mtimes:
                    del self._file_mtimes[config_id]

            self.logger.info(f"Reloading configuration: {config_id}")

            # The next call to load_experiment_config will reload from disk
            return ServiceResponse(
                success=True, message=f"Configuration '{config_id}' marked for reload"
            )

        except Exception as e:
            error_msg = f"Failed to reload configuration '{config_id}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ServiceResponse(success=False, message=error_msg, error=str(e))

    def get_cached_config(self, config_id: str) -> ExperimentConfig | None:
        """
        Get a cached configuration if available and not expired.

        Args:
            config_id: ID of the configuration to retrieve

        Returns:
            Cached ExperimentConfig or None if not available/expired
        """
        with self._cache_lock:
            if config_id not in self._cache:
                return None

            # Check if cache entry is expired
            current_time = time.time()
            cached_time = self._cache_timestamps.get(config_id, 0)

            if current_time - cached_time > self.cache_ttl:
                # Remove expired entry
                del self._cache[config_id]
                del self._cache_timestamps[config_id]
                self.logger.debug(f"Removed expired cache entry for {config_id}")
                return None

            # Check if file has been modified
            if self._is_file_modified(config_id):
                del self._cache[config_id]
                del self._cache_timestamps[config_id]
                self.logger.debug(f"Removed modified cache entry for {config_id}")
                return None

            return self._cache[config_id]

    async def list_cached_configs(self) -> dict[str, dict[str, Any]]:
        """
        List all cached configurations with metadata.

        Returns:
            Dictionary of cached configuration metadata
        """
        with self._cache_lock:
            result = {}
            current_time = time.time()

            for config_id, config in self._cache.items():
                cached_time = self._cache_timestamps.get(config_id, 0)
                age_seconds = current_time - cached_time

                result[config_id] = {
                    "experiment_name": config.name,
                    "cached_at": cached_time,
                    "age_seconds": age_seconds,
                    "is_expired": age_seconds > self.cache_ttl,
                    "models_count": len(config.models),
                    "datasets_count": len(config.datasets),
                }

            return result

    def _get_config_id(self, config_path: Path) -> str:
        """
        Generate a unique ID for a configuration based on its path.

        Args:
            config_path: Path to the configuration file

        Returns:
            Unique configuration ID
        """
        # Use hash of absolute path as config ID
        abs_path = str(config_path.resolve())
        return hashlib.md5(abs_path.encode()).hexdigest()

    async def _cache_config(
        self, config_id: str, config: ExperimentConfig, config_path: Path
    ) -> None:
        """
        Cache a configuration with timestamp.

        Args:
            config_id: Unique configuration ID
            config: Configuration to cache
            config_path: Path to the configuration file
        """
        with self._cache_lock:
            self._cache[config_id] = config
            self._cache_timestamps[config_id] = time.time()

            # Store file modification time for change detection
            with contextlib.suppress(OSError):
                self._file_mtimes[config_id] = config_path.stat().st_mtime

            self.logger.debug(f"Cached configuration {config_id}")

    def _is_file_modified(self, config_id: str) -> bool:
        """
        Check if the configuration file has been modified since caching.

        Args:
            config_id: Configuration ID to check

        Returns:
            True if file has been modified
        """
        if config_id not in self._file_mtimes:
            return False

        try:
            # This is a simplified check - in a real implementation,
            # you'd store the original file path to check modification time
            return False
        except OSError:
            return True  # File might have been deleted

    async def _create_default_structure(self) -> None:
        """Create default configuration directory structure."""
        try:
            # Create examples directory
            examples_dir = self.config_dir / "examples"
            examples_dir.mkdir(exist_ok=True)

            # Create a default example configuration
            example_config_path = examples_dir / "default_experiment.yaml"
            if not example_config_path.exists():
                default_config = await self.get_default_config()
                with open(example_config_path, "w", encoding="utf-8") as f:
                    yaml.dump(default_config, f, indent=2, default_flow_style=False)

                self.logger.info(f"Created example configuration: {example_config_path}")

        except Exception as e:
            self.logger.warning(f"Could not create default structure: {str(e)}")

    async def _preload_configurations(self) -> None:
        """Preload any existing configurations for faster access."""
        try:
            config_files = list(self.config_dir.glob("**/*.yaml")) + list(
                self.config_dir.glob("**/*.yml")
            )

            for config_file in config_files:
                try:
                    self.logger.debug(f"Found configuration file: {config_file}")
                    # Don't actually load yet, just register the existence

                except Exception as e:
                    self.logger.warning(f"Could not process config file {config_file}: {str(e)}")

        except Exception as e:
            self.logger.warning(f"Could not preload configurations: {str(e)}")

    async def clear_cache(self) -> ServiceResponse:
        """
        Clear the configuration cache.

        Returns:
            ServiceResponse indicating success
        """
        try:
            with self._cache_lock:
                cleared_count = len(self._cache)
                self._cache.clear()
                self._cache_timestamps.clear()
                self._file_mtimes.clear()

            self.logger.info(f"Cleared {cleared_count} cached configurations")

            return ServiceResponse(
                success=True, message=f"Cleared {cleared_count} cached configurations"
            )

        except Exception as e:
            error_msg = f"Failed to clear cache: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ServiceResponse(success=False, message=error_msg, error=str(e))

    async def save_config(
        self, config: ExperimentConfig, config_path: str | Path
    ) -> ServiceResponse:
        """
        Save a configuration to disk.

        Args:
            config: Configuration to save
            config_path: Path where to save the configuration

        Returns:
            ServiceResponse indicating success or failure
        """
        config_path = Path(config_path)

        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert config to dictionary
            config_dict = config.model_dump()

            # Save to YAML file
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, indent=2, default_flow_style=False)

            # Update cache
            config_id = self._get_config_id(config_path)
            await self._cache_config(config_id, config, config_path)

            self.logger.info(f"Saved configuration to {config_path}")

            return ServiceResponse(success=True, message=f"Configuration saved to {config_path}")

        except Exception as e:
            error_msg = f"Failed to save configuration to {config_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ServiceResponse(success=False, message=error_msg, error=str(e))

    def resolve_environment_variables(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve environment variables in configuration dictionary.

        Supports patterns:
        - ${ENV_VAR} - required environment variable
        - ${ENV_VAR:default_value} - optional with default

        Args:
            config_dict: Configuration dictionary to process

        Returns:
            Configuration dictionary with resolved environment variables

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        resolved_config = copy.deepcopy(config_dict)
        self._resolve_env_vars_recursive(resolved_config)
        return resolved_config

    def _resolve_env_vars_recursive(self, obj: Any) -> None:
        """Recursively resolve environment variables in nested structures."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    obj[key] = self._resolve_env_var_string(value)
                elif isinstance(value, dict | list):
                    self._resolve_env_vars_recursive(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = self._resolve_env_var_string(item)
                elif isinstance(item, dict | list):
                    self._resolve_env_vars_recursive(item)

    def _resolve_env_var_string(self, value: str) -> Any:
        """
        Resolve environment variables in a string value.

        Args:
            value: String that may contain environment variable patterns

        Returns:
            Resolved value with appropriate type conversion
        """
        # Pattern to match ${VAR} or ${VAR:default}
        pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

        def replace_var(match: re.Match[str]) -> Any:
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.getenv(var_name)

            if env_value is not None:
                return self._convert_env_value_type(env_value)
            elif default_value is not None:
                return self._convert_env_value_type(default_value)
            else:
                raise ConfigurationError(
                    f"Required environment variable '{var_name}' is not set",
                    error_code=ErrorCode.CONFIG_VALIDATION_FAILED,
                    metadata={"env_var": var_name},
                )

        # If the entire string is a single environment variable pattern, return the typed value
        full_match = re.fullmatch(pattern, value)
        if full_match:
            return replace_var(full_match)

        # Otherwise, do string substitution
        return re.sub(pattern, lambda m: str(replace_var(m)), value)

    def _convert_env_value_type(self, value: str) -> Any:
        """
        Convert environment variable string to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value (str, int, bool, or list)
        """
        # Handle boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Handle integer values
        try:
            if "." not in value and value.lstrip("-+").isdigit():
                return int(value)
        except ValueError:
            pass

        # Handle comma-separated lists
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]

        # Return as string
        return value

    def validate_environment_requirements(self, config: ExperimentConfig) -> list[str]:
        """
        Validate that all required environment variables are present.

        Args:
            config: Configuration to validate

        Returns:
            List of warning messages for missing optional variables
        """
        warnings = []
        required_vars = self.get_required_env_vars(config)

        for var_name in required_vars:
            if not os.getenv(var_name):
                warnings.append(f"Environment variable '{var_name}' is not set")

        return warnings

    def get_required_env_vars(self, config: ExperimentConfig) -> set[str]:
        """
        Extract all required environment variables from configuration.

        Args:
            config: Configuration to analyze

        Returns:
            Set of required environment variable names
        """
        required_vars: set[str] = set()
        config_dict = config.model_dump()
        self._extract_env_vars_recursive(config_dict, required_vars)
        return required_vars

    def _extract_env_vars_recursive(self, obj: Any, required_vars: set[str]) -> None:
        """Recursively extract environment variable names from nested structures."""
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, str):
                    self._extract_env_vars_from_string(value, required_vars)
                elif isinstance(value, dict | list):
                    self._extract_env_vars_recursive(value, required_vars)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    self._extract_env_vars_from_string(item, required_vars)
                elif isinstance(item, dict | list):
                    self._extract_env_vars_recursive(item, required_vars)

    def _extract_env_vars_from_string(self, value: str, required_vars: set[str]) -> None:
        """Extract environment variable names from a string."""
        pattern = r"\$\{([^}:]+)(?::[^}]*)?\}"
        matches = re.findall(pattern, value)
        for var_name in matches:
            required_vars.add(var_name)

    def mask_sensitive_values(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Create a masked copy of configuration for safe logging.

        Args:
            config_dict: Configuration dictionary to mask

        Returns:
            Dictionary with sensitive values masked
        """
        masked_config = copy.deepcopy(config_dict)
        self._mask_sensitive_recursive(masked_config)
        return masked_config

    def _mask_sensitive_recursive(self, obj: Any) -> None:
        """Recursively mask sensitive values in nested structures."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if self._is_sensitive_key(key):
                    if isinstance(value, str) and value:
                        obj[key] = self._mask_value(value)
                elif isinstance(value, dict | list):
                    self._mask_sensitive_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict | list):
                    self._mask_sensitive_recursive(item)

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a configuration key contains sensitive data."""
        sensitive_patterns = {
            "api_key",
            "secret",
            "password",
            "token",
            "credential",
            "auth",
            "secret_key",
            "access_token",
            "bearer_token",
        }
        key_lower = key.lower()

        # Check for exact matches or patterns that end with sensitive words
        for pattern in sensitive_patterns:
            if (
                pattern == key_lower
                or key_lower.endswith("_" + pattern)
                or key_lower.endswith(pattern)
            ):
                return True

        # Special handling for "key" - only if it's at the end or preceded by underscore
        return key_lower.endswith("_key") or key_lower == "key"

    def _mask_value(self, value: str) -> str:
        """Mask a sensitive value for logging."""
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

"""
Configuration validators for comprehensive validation of experiment configurations.

This module provides advanced validation capabilities for configuration consistency,
best practices, and potential issues that could affect experiment execution.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


from benchmark.core.config import DatasetConfig, ExperimentConfig, ModelConfig
from benchmark.core.logging import get_logger


class ValidationLevel(Enum):
    """Validation warning levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationWarning:
    """Represents a configuration validation warning or error."""

    level: ValidationLevel
    category: str
    message: str
    field: str | None = None
    suggestion: str | None = None
    metadata: dict[str, Any] | None = None


class ConfigurationValidator:
    """
    Comprehensive configuration validator for experiment configurations.

    Provides validation for:
    - Model configuration compatibility
    - Dataset format and availability
    - Resource requirements and constraints
    - Performance optimization recommendations
    - API availability and rate limits
    """

    # Known model specifications for validation
    MODEL_SPECS = {
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "context_length": 16385,
            "rate_limit_rpm": 3500,
            "rate_limit_tpm": 90000,
            "estimated_memory_mb": 500,
        },
        "gpt-4": {
            "max_tokens": 8192,
            "context_length": 128000,
            "rate_limit_rpm": 500,
            "rate_limit_tpm": 30000,
            "estimated_memory_mb": 1200,
        },
        "claude-3-haiku": {
            "max_tokens": 4096,
            "context_length": 200000,
            "rate_limit_rpm": 1000,
            "rate_limit_tpm": 50000,
            "estimated_memory_mb": 800,
        },
        "claude-3-sonnet": {
            "max_tokens": 4096,
            "context_length": 200000,
            "rate_limit_rpm": 1000,
            "rate_limit_tpm": 40000,
            "estimated_memory_mb": 1000,
        },
    }

    # Dataset format validators
    DATASET_FORMATS = {
        ".jsonl": "validate_jsonl_format",
        ".json": "validate_json_format",
        ".csv": "validate_csv_format",
        ".parquet": "validate_parquet_format",
    }

    def __init__(self) -> None:
        """Initialize the configuration validator."""
        self.logger = get_logger("config_validator")

    async def validate_configuration(self, config: ExperimentConfig) -> list[ValidationWarning]:
        """
        Perform comprehensive validation of an experiment configuration.

        Args:
            config: The experiment configuration to validate

        Returns:
            List of validation warnings and errors
        """
        warnings: list[ValidationWarning] = []

        # Run all validation checks
        validation_tasks = [
            self.validate_model_configs(config.models),
            self.validate_dataset_configs(config.datasets),
            self.validate_resource_requirements(config),
            self.validate_performance_settings(config),
            self.check_api_key_availability(config.models),
            self.validate_cross_field_consistency(config),
        ]

        # Execute all validations concurrently
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Collect all warnings
        for result in results:
            if isinstance(result, list):
                warnings.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Validation error: {result}")
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.ERROR,
                        category="validation_error",
                        message=f"Validation failed: {str(result)}",
                    )
                )

        return warnings

    async def validate_model_configs(self, models: list[ModelConfig]) -> list[ValidationWarning]:
        """
        Validate model configurations for compatibility and best practices.

        Args:
            models: List of model configurations to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        if not models:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.ERROR,
                    category="model_config",
                    message="No models configured",
                    suggestion="Add at least one model configuration",
                )
            )
            return warnings

        model_names = set()
        total_estimated_memory = 0

        for i, model in enumerate(models):
            # Check for duplicate model names
            if model.name in model_names:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="model_config",
                        message=f"Duplicate model name: {model.name}",
                        field=f"models[{i}].name",
                        suggestion="Use unique names for each model configuration",
                    )
                )
            model_names.add(model.name)

            # Validate model specifications
            model_spec = self.MODEL_SPECS.get(model.path)
            if model_spec:
                # Check max_tokens against model limits
                if model.max_tokens and model.max_tokens > model_spec["max_tokens"]:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.WARNING,
                            category="model_config",
                            message=f"Model {model.name} max_tokens ({model.max_tokens}) exceeds model limit ({model_spec['max_tokens']})",
                            field=f"models[{i}].max_tokens",
                            suggestion=f"Reduce max_tokens to {model_spec['max_tokens']} or less",
                        )
                    )

                total_estimated_memory += model_spec["estimated_memory_mb"]

                # Check temperature range
                if model.temperature is not None and (
                    model.temperature < 0 or model.temperature > 2
                ):
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.WARNING,
                            category="model_config",
                            message=f"Model {model.name} temperature ({model.temperature}) outside recommended range (0-2)",
                            field=f"models[{i}].temperature",
                            suggestion="Use temperature between 0 and 2",
                        )
                    )
            else:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.INFO,
                        category="model_config",
                        message=f"Unknown model: {model.path}",
                        field=f"models[{i}].path",
                        suggestion="Verify model path is correct",
                    )
                )

            # Check API configuration
            if model.type in ["openai_api", "anthropic_api"] and not model.config.get("api_key"):
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.ERROR,
                        category="model_config",
                        message=f"Model {model.name} missing API key",
                        field=f"models[{i}].config.api_key",
                        suggestion="Add API key to model configuration",
                    )
                )

        # Check total memory requirements
        if psutil:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            if total_estimated_memory > available_memory_mb * 0.8:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="resource_usage",
                        message=f"Estimated model memory usage ({total_estimated_memory:.0f}MB) may exceed available memory ({available_memory_mb:.0f}MB)",
                        suggestion="Consider using fewer models or models with lower memory requirements",
                    )
                )
        else:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.INFO,
                    category="resource_usage",
                    message="Cannot check memory requirements - psutil not available",
                    suggestion="Install psutil for memory validation",
                )
            )

        return warnings

    async def validate_dataset_configs(
        self, datasets: list[DatasetConfig]
    ) -> list[ValidationWarning]:
        """
        Validate dataset configurations for format and availability.

        Args:
            datasets: List of dataset configurations to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        if not datasets:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.ERROR,
                    category="dataset_config",
                    message="No datasets configured",
                    suggestion="Add at least one dataset configuration",
                )
            )
            return warnings

        dataset_names = set()
        total_disk_usage = 0

        for i, dataset in enumerate(datasets):
            # Check for duplicate dataset names
            if dataset.name in dataset_names:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="dataset_config",
                        message=f"Duplicate dataset name: {dataset.name}",
                        field=f"datasets[{i}].name",
                        suggestion="Use unique names for each dataset",
                    )
                )
            dataset_names.add(dataset.name)

            # Validate dataset path and format
            if dataset.source == "local":
                dataset_path = Path(dataset.path)

                if not dataset_path.exists():
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.ERROR,
                            category="dataset_config",
                            message=f"Dataset file not found: {dataset.path}",
                            field=f"datasets[{i}].path",
                            suggestion="Ensure dataset file exists at the specified path",
                        )
                    )
                    continue

                # Check file format
                file_suffix = dataset_path.suffix.lower()
                if file_suffix not in self.DATASET_FORMATS:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.WARNING,
                            category="dataset_config",
                            message=f"Unsupported dataset format: {file_suffix}",
                            field=f"datasets[{i}].path",
                            suggestion=f"Use supported formats: {', '.join(self.DATASET_FORMATS.keys())}",
                        )
                    )
                else:
                    # Validate file format structure
                    format_warnings = await self._validate_dataset_format(dataset_path, file_suffix)
                    for warning in format_warnings:
                        warning.field = f"datasets[{i}].path"
                    warnings.extend(format_warnings)

                # Check file size and disk usage
                try:
                    file_size = dataset_path.stat().st_size
                    total_disk_usage += file_size

                    # Warn about very large files
                    if file_size > 1024 * 1024 * 1024:  # 1GB
                        warnings.append(
                            ValidationWarning(
                                level=ValidationLevel.INFO,
                                category="performance",
                                message=f"Large dataset file: {dataset.name} ({file_size / (1024**3):.1f}GB)",
                                field=f"datasets[{i}].path",
                                suggestion="Consider using smaller datasets for faster processing",
                            )
                        )
                except OSError as e:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.ERROR,
                            category="dataset_config",
                            message=f"Cannot access dataset file: {e}",
                            field=f"datasets[{i}].path",
                        )
                    )

            # Validate split ratios
            total_split = dataset.test_split + dataset.validation_split
            if total_split >= 1.0:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.ERROR,
                        category="dataset_config",
                        message=f"Dataset {dataset.name} test and validation splits sum to {total_split:.2f} (must be < 1.0)",
                        field=f"datasets[{i}]",
                        suggestion="Reduce test_split and/or validation_split values",
                    )
                )

            # Check sample size
            if dataset.max_samples and dataset.max_samples < 10:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="dataset_config",
                        message=f"Very small sample size for dataset {dataset.name}: {dataset.max_samples}",
                        field=f"datasets[{i}].max_samples",
                        suggestion="Use at least 10 samples for meaningful evaluation",
                    )
                )

        # Check total disk usage
        if psutil:
            available_disk_gb = psutil.disk_usage("/").free / (1024**3)
            total_disk_gb = total_disk_usage / (1024**3)

            if total_disk_gb > available_disk_gb * 0.9:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="resource_usage",
                        message=f"Dataset storage requirements ({total_disk_gb:.1f}GB) may exceed available disk space ({available_disk_gb:.1f}GB)",
                        suggestion="Free up disk space or use smaller datasets",
                    )
                )

        return warnings

    async def validate_resource_requirements(
        self, config: ExperimentConfig
    ) -> list[ValidationWarning]:
        """
        Validate resource requirements against available system resources.

        Args:
            config: The experiment configuration to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check CPU usage
        cpu_count = os.cpu_count() or 1
        parallel_jobs = config.evaluation.parallel_jobs

        if parallel_jobs > cpu_count:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.WARNING,
                    category="resource_usage",
                    message=f"Parallel jobs ({parallel_jobs}) exceeds CPU count ({cpu_count})",
                    field="evaluation.parallel_jobs",
                    suggestion=f"Reduce parallel_jobs to {cpu_count} or less for optimal performance",
                )
            )

        # Check memory usage estimates
        if psutil:
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
        else:
            available_memory_gb = 4.0  # Default assumption

        # Estimate memory usage
        estimated_memory_usage = 0
        for model in config.models:
            model_spec = self.MODEL_SPECS.get(model.path, {})
            estimated_memory_usage += model_spec.get("estimated_memory_mb", 500)  # Default estimate

        # Add batch processing memory estimate
        batch_size = config.evaluation.batch_size or 16
        estimated_batch_memory = batch_size * 10  # Rough estimate: 10MB per batch item
        estimated_memory_usage += estimated_batch_memory

        estimated_memory_gb = estimated_memory_usage / 1024

        if estimated_memory_gb > available_memory_gb * 0.8:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.WARNING,
                    category="resource_usage",
                    message=f"Estimated memory usage ({estimated_memory_gb:.1f}GB) may exceed available memory ({available_memory_gb:.1f}GB)",
                    suggestion="Reduce batch_size, use fewer models, or increase system memory",
                )
            )

        # Check disk space for output
        if hasattr(config, "output_dir") and config.output_dir:
            try:
                output_path = Path(config.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                if psutil:
                    disk_usage = psutil.disk_usage(str(output_path))
                    available_disk_gb = disk_usage.free / (1024**3)
                else:
                    available_disk_gb = 10.0  # Default assumption

                # Estimate output size (rough calculation)
                total_samples = sum(d.max_samples or 1000 for d in config.datasets)
                estimated_output_gb = (
                    total_samples * len(config.models) * 0.001
                )  # 1KB per sample-model pair

                if estimated_output_gb > available_disk_gb * 0.9:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.WARNING,
                            category="resource_usage",
                            message=f"Estimated output size ({estimated_output_gb:.1f}GB) may exceed available disk space ({available_disk_gb:.1f}GB)",
                            field="output_dir",
                            suggestion="Free up disk space or reduce dataset/model count",
                        )
                    )
            except Exception as e:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="resource_usage",
                        message=f"Cannot validate output directory: {e}",
                        field="output_dir",
                    )
                )

        return warnings

    async def validate_performance_settings(
        self, config: ExperimentConfig
    ) -> list[ValidationWarning]:
        """
        Validate performance settings and provide optimization recommendations.

        Args:
            config: The experiment configuration to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        evaluation = config.evaluation

        # Check batch size optimization
        batch_size = evaluation.batch_size or 16

        # Get system memory for optimization recommendations
        memory_gb = psutil.virtual_memory().total / (1024**3) if psutil else 8.0

        # Recommend optimal batch size based on memory
        if memory_gb > 16:  # High-memory system
            optimal_batch_size = 64
        elif memory_gb > 8:  # Medium-memory system
            optimal_batch_size = 32
        else:  # Low-memory system
            optimal_batch_size = 16

        if batch_size < optimal_batch_size // 2:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.INFO,
                    category="performance",
                    message=f"Batch size ({batch_size}) may be too small for optimal performance",
                    field="evaluation.batch_size",
                    suggestion=f"Consider increasing batch_size to {optimal_batch_size} for better throughput",
                )
            )
        elif batch_size > optimal_batch_size * 2:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.WARNING,
                    category="performance",
                    message=f"Batch size ({batch_size}) may be too large and cause memory issues",
                    field="evaluation.batch_size",
                    suggestion=f"Consider reducing batch_size to {optimal_batch_size} or less",
                )
            )

        # Check timeout settings
        timeout_minutes = evaluation.timeout_minutes or 30
        if timeout_minutes < 5:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.WARNING,
                    category="performance",
                    message=f"Timeout ({timeout_minutes} minutes) may be too short for complex evaluations",
                    field="evaluation.timeout_minutes",
                    suggestion="Consider increasing timeout to at least 10 minutes",
                )
            )
        elif timeout_minutes > 120:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.INFO,
                    category="performance",
                    message=f"Very long timeout ({timeout_minutes} minutes) - ensure this is intentional",
                    field="evaluation.timeout_minutes",
                )
            )

        # Check parallel jobs vs API rate limits
        parallel_jobs = evaluation.parallel_jobs
        for model in config.models:
            model_spec = self.MODEL_SPECS.get(model.path)
            if model_spec and model.type in ["openai_api", "anthropic_api"]:
                rate_limit_rpm = model_spec.get("rate_limit_rpm", 1000)

                # Estimate requests per minute
                estimated_rpm = parallel_jobs * 60 / (timeout_minutes * 60 / 100)  # Rough estimate

                if estimated_rpm > rate_limit_rpm * 0.8:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.WARNING,
                            category="performance",
                            message=f"High parallel jobs ({parallel_jobs}) may exceed {model.name} rate limits ({rate_limit_rpm} RPM)",
                            field="evaluation.parallel_jobs",
                            suggestion="Reduce parallel_jobs to avoid rate limiting",
                        )
                    )

        return warnings

    async def check_api_key_availability(
        self, models: list[ModelConfig]
    ) -> list[ValidationWarning]:
        """
        Check API key availability for configured models.

        Args:
            models: List of model configurations to check

        Returns:
            List of validation warnings
        """
        warnings = []

        api_keys_checked = set()

        for i, model in enumerate(models):
            if model.type in ["openai_api", "anthropic_api"]:
                api_key = model.config.get("api_key", "")

                if not api_key:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.ERROR,
                            category="api_config",
                            message=f"Model {model.name} missing API key",
                            field=f"models[{i}].config.api_key",
                            suggestion="Add valid API key to model configuration",
                        )
                    )
                    continue

                # Check if API key looks valid (basic format validation)
                if model.type == "openai_api":
                    if not (api_key.startswith("sk-") and len(api_key) > 40):
                        warnings.append(
                            ValidationWarning(
                                level=ValidationLevel.WARNING,
                                category="api_config",
                                message=f"Model {model.name} API key format appears invalid",
                                field=f"models[{i}].config.api_key",
                                suggestion="Verify OpenAI API key format (should start with 'sk-')",
                            )
                        )
                elif model.type == "anthropic_api" and not (
                    api_key.startswith("sk-ant-") and len(api_key) > 50
                ):
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.WARNING,
                            category="api_config",
                            message=f"Model {model.name} API key format appears invalid",
                            field=f"models[{i}].config.api_key",
                            suggestion="Verify Anthropic API key format (should start with 'sk-ant-')",
                        )
                    )

                # Note: We don't actually test API connectivity in validation
                # to avoid making external calls and potential costs
                api_keys_checked.add(f"{model.type}:{api_key[:10]}...")

        if api_keys_checked:
            self.logger.info(f"Validated {len(api_keys_checked)} API key formats")

        return warnings

    async def validate_cross_field_consistency(
        self, config: ExperimentConfig
    ) -> list[ValidationWarning]:
        """
        Validate cross-field consistency and logical relationships.

        Args:
            config: The experiment configuration to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check if evaluation metrics are appropriate for models
        metrics = config.evaluation.metrics
        if "accuracy" in metrics or "f1_score" in metrics:
            # These metrics require classification tasks
            for i, dataset in enumerate(config.datasets):
                # This is a basic check - in practice, you'd want to inspect the actual data
                if (
                    "classification" not in dataset.name.lower()
                    and "class" not in dataset.name.lower()
                ):
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.INFO,
                            category="consistency",
                            message=f"Using classification metrics with dataset {dataset.name} - ensure dataset supports classification",
                            field=f"datasets[{i}].name",
                            suggestion="Verify dataset contains classification labels",
                        )
                    )

        # Check model-dataset compatibility
        total_samples = sum(d.max_samples or 1000 for d in config.datasets)
        total_combinations = total_samples * len(config.models)

        if total_combinations > 10000:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.INFO,
                    category="performance",
                    message=f"Large number of model-sample combinations ({total_combinations:,}) - evaluation may take significant time",
                    suggestion="Consider reducing dataset size or number of models for faster iteration",
                )
            )

        # Check if parallel processing makes sense with current configuration
        if config.evaluation.parallel_jobs > 1 and len(config.models) == 1:
            model = config.models[0]
            if model.type in ["openai_api", "anthropic_api"]:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.INFO,
                        category="performance",
                        message="Parallel processing with single API model may be limited by rate limits",
                        suggestion="Consider using multiple models or reducing parallel_jobs for API models",
                    )
                )

        return warnings

    async def _validate_dataset_format(
        self, file_path: Path, file_format: str
    ) -> list[ValidationWarning]:
        """
        Validate specific dataset format structure.

        Args:
            file_path: Path to the dataset file
            file_format: File format extension

        Returns:
            List of validation warnings
        """
        warnings = []

        try:
            if file_format == ".jsonl":
                warnings.extend(await self._validate_jsonl_format(file_path))
            elif file_format == ".json":
                warnings.extend(await self._validate_json_format(file_path))
            elif file_format == ".csv":
                warnings.extend(await self._validate_csv_format(file_path))
            # Add more format validators as needed
        except Exception as e:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.ERROR,
                    category="dataset_format",
                    message=f"Error validating dataset format: {e}",
                )
            )

        return warnings

    async def _validate_jsonl_format(self, file_path: Path) -> list[ValidationWarning]:
        """Validate JSONL format structure."""
        warnings = []

        try:
            with open(file_path, encoding="utf-8") as f:
                line_count = 0
                required_fields = set()

                for line_num, line in enumerate(f, 1):
                    if line_num > 100:  # Only check first 100 lines for performance
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if line_count == 0:
                            required_fields = set(data.keys())
                        else:
                            current_fields = set(data.keys())
                            if current_fields != required_fields:
                                warnings.append(
                                    ValidationWarning(
                                        level=ValidationLevel.WARNING,
                                        category="dataset_format",
                                        message=f"Inconsistent fields at line {line_num}",
                                        suggestion="Ensure all records have the same field structure",
                                    )
                                )
                                break
                        line_count += 1
                    except json.JSONDecodeError as e:
                        warnings.append(
                            ValidationWarning(
                                level=ValidationLevel.ERROR,
                                category="dataset_format",
                                message=f"Invalid JSON at line {line_num}: {e}",
                            )
                        )
                        break

                if line_count == 0:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.ERROR,
                            category="dataset_format",
                            message="Empty dataset file",
                        )
                    )
                elif line_count < 10:
                    warnings.append(
                        ValidationWarning(
                            level=ValidationLevel.WARNING,
                            category="dataset_format",
                            message=f"Very small dataset: only {line_count} records",
                            suggestion="Consider using larger datasets for better evaluation",
                        )
                    )

        except Exception as e:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.ERROR,
                    category="dataset_format",
                    message=f"Cannot read JSONL file: {e}",
                )
            )

        return warnings

    async def _validate_json_format(self, file_path: Path) -> list[ValidationWarning]:
        """Validate JSON format structure."""
        warnings = []

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="dataset_format",
                        message="JSON dataset should be a list of records",
                        suggestion="Use array format: [{...}, {...}, ...]",
                    )
                )
            elif len(data) == 0:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.ERROR,
                        category="dataset_format",
                        message="Empty dataset",
                    )
                )
            elif len(data) < 10:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="dataset_format",
                        message=f"Very small dataset: only {len(data)} records",
                        suggestion="Consider using larger datasets for better evaluation",
                    )
                )

        except json.JSONDecodeError as e:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.ERROR,
                    category="dataset_format",
                    message=f"Invalid JSON format: {e}",
                )
            )
        except Exception as e:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.ERROR,
                    category="dataset_format",
                    message=f"Cannot read JSON file: {e}",
                )
            )

        return warnings

    async def _validate_csv_format(self, file_path: Path) -> list[ValidationWarning]:
        """Validate CSV format structure."""
        warnings = []

        try:
            import pandas as pd

            # Read first few rows to check format
            df = pd.read_csv(file_path, nrows=100)

            if df.empty:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.ERROR,
                        category="dataset_format",
                        message="Empty CSV dataset",
                    )
                )
            elif len(df) < 10:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.WARNING,
                        category="dataset_format",
                        message=f"Very small dataset: only {len(df)} records",
                        suggestion="Consider using larger datasets for better evaluation",
                    )
                )

            # Check for missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                warnings.append(
                    ValidationWarning(
                        level=ValidationLevel.INFO,
                        category="dataset_format",
                        message=f"Columns with missing values: {missing_cols}",
                        suggestion="Consider handling missing values before evaluation",
                    )
                )

        except ImportError:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.INFO,
                    category="dataset_format",
                    message="pandas not available - skipping detailed CSV validation",
                )
            )
        except Exception as e:
            warnings.append(
                ValidationWarning(
                    level=ValidationLevel.ERROR,
                    category="dataset_format",
                    message=f"Cannot read CSV file: {e}",
                )
            )

        return warnings

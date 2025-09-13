"""
Configuration data models for the LLM Cybersecurity Benchmark system.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and processing."""

    name: str = Field(..., min_length=1, description="Dataset name identifier")
    source: str = Field(..., description="Data source type")
    path: str = Field(..., min_length=1, description="Path to dataset")
    max_samples: int | None = Field(None, gt=0, description="Maximum samples to load")
    test_split: float = Field(0.2, ge=0.0, le=0.8, description="Test set proportion")
    validation_split: float = Field(0.1, ge=0.0, le=0.5, description="Validation set proportion")
    preprocessing: list[str] = Field(default_factory=list, description="Preprocessing steps")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate dataset source type."""
        valid_sources = {"kaggle", "huggingface", "local", "remote", "synthetic"}
        if v.lower() not in valid_sources:
            raise ValueError(f"Invalid source '{v}'. Must be one of: {', '.join(valid_sources)}")
        return v.lower()

    @model_validator(mode="after")
    def validate_splits(self) -> "DatasetConfig":
        """Validate that test and validation splits don't exceed 100%."""
        total_split = self.test_split + self.validation_split
        if total_split >= 1.0:
            raise ValueError(
                f"Combined test_split ({self.test_split}) and validation_split "
                f"({self.validation_split}) must be less than 1.0. Current total: {total_split}"
            )
        return self

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ModelConfig(BaseModel):
    """Configuration for model loading and inference."""

    name: str = Field(..., min_length=1, description="Model name identifier")
    type: str = Field(..., description="Model provider/type")
    path: str = Field(..., min_length=1, description="Path or endpoint to model")
    config: dict[str, Any] = Field(default_factory=dict, description="Model-specific configuration")
    max_tokens: int = Field(512, gt=0, le=4096, description="Maximum tokens for generation")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate model type."""
        valid_types = {
            "mlx_local",
            "openai_api",
            "anthropic_api",
            "huggingface_local",
            "huggingface_api",
            "ollama_local",
            "custom",
        }
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid model type '{v}'. Must be one of: {', '.join(valid_types)}")
        return v.lower()

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate model configuration dictionary."""
        if not isinstance(v, dict):
            raise ValueError("Config must be a dictionary")
        return v

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation process."""

    metrics: list[str] = Field(..., min_length=1, description="Evaluation metrics to calculate")
    parallel_jobs: int = Field(1, ge=1, le=8, description="Number of parallel evaluation jobs")
    timeout_minutes: int = Field(60, gt=0, le=1440, description="Evaluation timeout in minutes")
    batch_size: int = Field(32, gt=0, le=128, description="Batch size for evaluation")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate evaluation metrics."""
        valid_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "confusion_matrix",
            "detection_rate",
            "false_positive_rate",
            "response_time",
            "explainability_score",
        }

        invalid_metrics = [metric for metric in v if metric.lower() not in valid_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {', '.join(invalid_metrics)}. "
                f"Valid metrics: {', '.join(valid_metrics)}"
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_metrics = []
        for metric in v:
            metric_lower = metric.lower()
            if metric_lower not in seen:
                seen.add(metric_lower)
                unique_metrics.append(metric_lower)

        return unique_metrics

    model_config = ConfigDict(
        validate_assignment=True,
    )


class ExperimentConfig(BaseModel):
    """Configuration for complete experiment setup."""

    name: str = Field(..., min_length=1, description="Experiment name")
    description: str | None = Field(None, description="Experiment description")
    output_dir: str = Field("./results", description="Output directory for results")
    datasets: list[DatasetConfig] = Field(..., min_length=1, description="Dataset configurations")
    models: list[ModelConfig] = Field(..., min_length=1, description="Model configurations")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration")

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Validate output directory path."""
        from pathlib import Path

        # Ensure path is not empty and convert to Path for validation
        if not v.strip():
            raise ValueError("Output directory cannot be empty")

        try:
            # Validate the path without converting to absolute
            # This preserves relative paths as expected by users
            Path(v)  # Just validate that it's a valid path
            return v.strip()  # Return original path, just stripped
        except Exception as e:
            raise ValueError(f"Invalid output directory path '{v}': {e}") from e

    @model_validator(mode="after")
    def validate_unique_names(self) -> "ExperimentConfig":
        """Validate that dataset and model names are unique within the experiment."""
        # Check dataset name uniqueness
        dataset_names = [ds.name for ds in self.datasets]
        if len(dataset_names) != len(set(dataset_names)):
            duplicates = [name for name in dataset_names if dataset_names.count(name) > 1]
            raise ValueError(f"Duplicate dataset names found: {', '.join(set(duplicates))}")

        # Check model name uniqueness
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            duplicates = [name for name in model_names if model_names.count(name) > 1]
            raise ValueError(f"Duplicate model names found: {', '.join(set(duplicates))}")

        return self

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class BenchmarkConfig(BaseModel):
    """Root configuration for the entire benchmark system."""

    version: str = Field("1.0", description="Configuration version")
    experiments: list[ExperimentConfig] = Field(
        ..., min_length=1, description="Experiment configurations"
    )
    global_settings: dict[str, Any] = Field(
        default_factory=dict, description="Global benchmark settings"
    )
    logging_level: str = Field("INFO", description="Global logging level")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate configuration version format."""
        import re

        if not re.match(r"^\d+\.\d+(\.\d+)?$", v):
            raise ValueError(f"Invalid version format '{v}'. Expected format: X.Y or X.Y.Z")
        return v

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(
                f"Invalid logging level '{v}'. Must be one of: {', '.join(valid_levels)}"
            )
        return v.upper()

    @model_validator(mode="after")
    def validate_experiment_names(self) -> "BenchmarkConfig":
        """Validate that experiment names are unique across the benchmark."""
        experiment_names = [exp.name for exp in self.experiments]
        if len(experiment_names) != len(set(experiment_names)):
            duplicates = [name for name in experiment_names if experiment_names.count(name) > 1]
            raise ValueError(f"Duplicate experiment names found: {', '.join(set(duplicates))}")
        return self

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# Convenience functions for creating common configurations


def create_local_dataset_config(
    name: str,
    path: str,
    test_split: float = 0.2,
    validation_split: float = 0.1,
    max_samples: int | None = None,
    preprocessing: list[str] | None = None,
) -> DatasetConfig:
    """Create configuration for local dataset."""
    return DatasetConfig(
        name=name,
        source="local",
        path=path,
        test_split=test_split,
        validation_split=validation_split,
        max_samples=max_samples,
        preprocessing=preprocessing or [],
    )


def create_mlx_model_config(
    name: str,
    path: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    config: dict[str, Any] | None = None,
) -> ModelConfig:
    """Create configuration for MLX local model."""
    return ModelConfig(
        name=name,
        type="mlx_local",
        path=path,
        max_tokens=max_tokens,
        temperature=temperature,
        config=config or {},
    )


def create_api_model_config(
    name: str,
    provider: str,
    model_name: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    config: dict[str, Any] | None = None,
) -> ModelConfig:
    """Create configuration for API-based model."""
    provider_types = {
        "openai": "openai_api",
        "anthropic": "anthropic_api",
        "huggingface": "huggingface_api",
    }

    if provider.lower() not in provider_types:
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported: {', '.join(provider_types.keys())}"
        )

    return ModelConfig(
        name=name,
        type=provider_types[provider.lower()],
        path=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        config=config or {},
    )


def create_standard_evaluation_config(
    metrics: list[str] | None = None,
    parallel_jobs: int = 1,
    timeout_minutes: int = 60,
    batch_size: int = 32,
) -> EvaluationConfig:
    """Create standard evaluation configuration."""
    default_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "detection_rate",
        "false_positive_rate",
        "response_time",
    ]

    return EvaluationConfig(
        metrics=metrics or default_metrics,
        parallel_jobs=parallel_jobs,
        timeout_minutes=timeout_minutes,
        batch_size=batch_size,
    )


def load_config_from_file(file_path: str) -> BenchmarkConfig:
    """Load configuration from YAML or JSON file."""
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            with open(path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        return BenchmarkConfig.model_validate(config_data)

    except Exception as e:
        raise ValueError(f"Failed to load configuration from {file_path}: {e}") from e


def save_config_to_file(config: BenchmarkConfig, file_path: str, format: str = "yaml") -> None:
    """Save configuration to YAML or JSON file."""
    from pathlib import Path

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_dict = config.model_dump()

        if format.lower() in {"yaml", "yml"}:
            import yaml

            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == "json":
            import json

            with open(path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'yaml' or 'json'")

    except Exception as e:
        raise ValueError(f"Failed to save configuration to {file_path}: {e}") from e

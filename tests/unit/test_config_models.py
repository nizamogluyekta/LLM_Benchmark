"""
Unit tests for configuration data models.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.core.config import (
    BenchmarkConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    create_api_model_config,
    create_local_dataset_config,
    create_mlx_model_config,
    create_standard_evaluation_config,
    load_config_from_file,
    save_config_to_file,
)


class TestDatasetConfig:
    """Test DatasetConfig model."""

    def test_valid_dataset_config(self):
        """Test creating valid dataset configuration."""
        config = DatasetConfig(
            name="test-dataset",
            source="local",
            path="/data/test.csv",
            max_samples=1000,
            test_split=0.2,
            validation_split=0.1,
            preprocessing=["normalize", "tokenize"],
        )

        assert config.name == "test-dataset"
        assert config.source == "local"
        assert config.path == "/data/test.csv"
        assert config.max_samples == 1000
        assert config.test_split == 0.2
        assert config.validation_split == 0.1
        assert config.preprocessing == ["normalize", "tokenize"]

    def test_dataset_config_defaults(self):
        """Test dataset configuration with default values."""
        config = DatasetConfig(
            name="minimal-dataset",
            source="kaggle",
            path="/data/minimal.csv",
        )

        assert config.max_samples is None
        assert config.test_split == 0.2
        assert config.validation_split == 0.1
        assert config.preprocessing == []

    def test_dataset_config_source_validation(self):
        """Test dataset source validation."""
        # Valid sources
        for source in ["local", "kaggle", "huggingface", "remote", "synthetic"]:
            config = DatasetConfig(name="test", source=source, path="/data/test.csv")
            assert config.source == source.lower()

        # Invalid source
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(name="test", source="invalid", path="/data/test.csv")

        error = exc_info.value.errors()[0]
        assert "Invalid source" in error["msg"]

    def test_dataset_config_split_validation(self):
        """Test dataset split validation."""
        # Valid splits
        DatasetConfig(
            name="test", source="local", path="/data/test.csv", test_split=0.3, validation_split=0.2
        )

        # Invalid splits (sum >= 1.0)
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(
                name="test",
                source="local",
                path="/data/test.csv",
                test_split=0.8,
                validation_split=0.3,
            )

        error = exc_info.value.errors()[0]
        assert "must be less than 1.0" in error["msg"]

    def test_dataset_config_field_constraints(self):
        """Test field constraints validation."""
        # Empty name
        with pytest.raises(ValidationError):
            DatasetConfig(name="", source="local", path="/data/test.csv")

        # Empty path
        with pytest.raises(ValidationError):
            DatasetConfig(name="test", source="local", path="")

        # Invalid max_samples (zero or negative)
        with pytest.raises(ValidationError):
            DatasetConfig(name="test", source="local", path="/data/test.csv", max_samples=0)

        # Invalid test_split (negative or > 0.8)
        with pytest.raises(ValidationError):
            DatasetConfig(name="test", source="local", path="/data/test.csv", test_split=-0.1)

        with pytest.raises(ValidationError):
            DatasetConfig(name="test", source="local", path="/data/test.csv", test_split=0.9)


class TestModelConfig:
    """Test ModelConfig model."""

    def test_valid_model_config(self):
        """Test creating valid model configuration."""
        config = ModelConfig(
            name="test-model",
            type="mlx_local",
            path="/models/test.mlx",
            config={"param1": "value1"},
            max_tokens=1024,
            temperature=0.7,
        )

        assert config.name == "test-model"
        assert config.type == "mlx_local"
        assert config.path == "/models/test.mlx"
        assert config.config == {"param1": "value1"}
        assert config.max_tokens == 1024
        assert config.temperature == 0.7

    def test_model_config_defaults(self):
        """Test model configuration with default values."""
        config = ModelConfig(
            name="minimal-model",
            type="openai_api",
            path="gpt-3.5-turbo",
        )

        assert config.config == {}
        assert config.max_tokens == 512
        assert config.temperature == 0.1

    def test_model_config_type_validation(self):
        """Test model type validation."""
        # Valid types
        valid_types = [
            "mlx_local",
            "openai_api",
            "anthropic_api",
            "huggingface_local",
            "huggingface_api",
            "ollama_local",
            "custom",
        ]
        for model_type in valid_types:
            config = ModelConfig(name="test", type=model_type, path="test-path")
            assert config.type == model_type.lower()

        # Invalid type
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(name="test", type="invalid_type", path="test-path")

        error = exc_info.value.errors()[0]
        assert "Invalid model type" in error["msg"]

    def test_model_config_constraints(self):
        """Test model configuration constraints."""
        # Valid constraints
        ModelConfig(name="test", type="mlx_local", path="test", max_tokens=4096, temperature=2.0)

        # Invalid max_tokens (too high)
        with pytest.raises(ValidationError):
            ModelConfig(name="test", type="mlx_local", path="test", max_tokens=5000)

        # Invalid max_tokens (zero)
        with pytest.raises(ValidationError):
            ModelConfig(name="test", type="mlx_local", path="test", max_tokens=0)

        # Invalid temperature (negative)
        with pytest.raises(ValidationError):
            ModelConfig(name="test", type="mlx_local", path="test", temperature=-0.1)

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            ModelConfig(name="test", type="mlx_local", path="test", temperature=2.1)


class TestEvaluationConfig:
    """Test EvaluationConfig model."""

    def test_valid_evaluation_config(self):
        """Test creating valid evaluation configuration."""
        config = EvaluationConfig(
            metrics=["accuracy", "precision", "recall"],
            parallel_jobs=4,
            timeout_minutes=120,
            batch_size=64,
        )

        assert config.metrics == ["accuracy", "precision", "recall"]
        assert config.parallel_jobs == 4
        assert config.timeout_minutes == 120
        assert config.batch_size == 64

    def test_evaluation_config_defaults(self):
        """Test evaluation configuration with default values."""
        config = EvaluationConfig(metrics=["accuracy"])

        assert config.parallel_jobs == 1
        assert config.timeout_minutes == 60
        assert config.batch_size == 32

    def test_evaluation_metrics_validation(self):
        """Test evaluation metrics validation."""
        # Valid metrics
        valid_metrics = [
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
        ]

        config = EvaluationConfig(metrics=valid_metrics)
        assert len(config.metrics) == len(valid_metrics)

        # Invalid metric
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(metrics=["invalid_metric"])

        error = exc_info.value.errors()[0]
        assert "Invalid metrics" in error["msg"]

        # Mixed valid/invalid metrics
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(metrics=["accuracy", "invalid_metric", "precision"])

        error = exc_info.value.errors()[0]
        assert "invalid_metric" in error["msg"]

    def test_evaluation_metrics_deduplication(self):
        """Test evaluation metrics deduplication."""
        config = EvaluationConfig(metrics=["accuracy", "ACCURACY", "precision", "accuracy"])

        # Should deduplicate while preserving order
        assert config.metrics == ["accuracy", "precision"]

    def test_evaluation_config_constraints(self):
        """Test evaluation configuration constraints."""
        # Valid constraints
        EvaluationConfig(
            metrics=["accuracy"], parallel_jobs=8, timeout_minutes=1440, batch_size=128
        )

        # Invalid parallel_jobs (too high)
        with pytest.raises(ValidationError):
            EvaluationConfig(metrics=["accuracy"], parallel_jobs=10)

        # Invalid timeout_minutes (zero)
        with pytest.raises(ValidationError):
            EvaluationConfig(metrics=["accuracy"], timeout_minutes=0)

        # Invalid batch_size (too high)
        with pytest.raises(ValidationError):
            EvaluationConfig(metrics=["accuracy"], batch_size=200)

        # Empty metrics list
        with pytest.raises(ValidationError):
            EvaluationConfig(metrics=[])


class TestExperimentConfig:
    """Test ExperimentConfig model."""

    def test_valid_experiment_config(self):
        """Test creating valid experiment configuration."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])

        config = ExperimentConfig(
            name="test-experiment",
            description="Test experiment description",
            output_dir="./custom_results",
            datasets=[dataset],
            models=[model],
            evaluation=evaluation,
        )

        assert config.name == "test-experiment"
        assert config.description == "Test experiment description"
        assert "custom_results" in config.output_dir  # May be converted to absolute
        assert len(config.datasets) == 1
        assert len(config.models) == 1

    def test_experiment_config_defaults(self):
        """Test experiment configuration with default values."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])

        config = ExperimentConfig(
            name="minimal-experiment",
            datasets=[dataset],
            models=[model],
            evaluation=evaluation,
        )

        assert config.description is None
        assert "results" in config.output_dir  # Default output directory

    def test_experiment_output_dir_validation(self):
        """Test output directory validation."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])

        # Valid relative path
        config = ExperimentConfig(
            name="test",
            datasets=[dataset],
            models=[model],
            evaluation=evaluation,
            output_dir="./results",
        )
        assert "results" in config.output_dir

        # Empty output directory
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                name="test",
                datasets=[dataset],
                models=[model],
                evaluation=evaluation,
                output_dir="",
            )

        error = exc_info.value.errors()[0]
        assert "cannot be empty" in error["msg"]

    def test_experiment_unique_names_validation(self):
        """Test unique name validation."""
        dataset1 = DatasetConfig(name="duplicate", source="local", path="/data/test1.csv")
        dataset2 = DatasetConfig(name="duplicate", source="local", path="/data/test2.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])

        # Duplicate dataset names
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                name="test",
                datasets=[dataset1, dataset2],
                models=[model],
                evaluation=evaluation,
            )

        error = exc_info.value.errors()[0]
        assert "Duplicate dataset names" in error["msg"]

        # Duplicate model names
        model1 = ModelConfig(name="duplicate", type="mlx_local", path="/models/test1.mlx")
        model2 = ModelConfig(name="duplicate", type="openai_api", path="gpt-3.5-turbo")
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")

        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                name="test",
                datasets=[dataset],
                models=[model1, model2],
                evaluation=evaluation,
            )

        error = exc_info.value.errors()[0]
        assert "Duplicate model names" in error["msg"]


class TestBenchmarkConfig:
    """Test BenchmarkConfig model."""

    def test_valid_benchmark_config(self):
        """Test creating valid benchmark configuration."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])
        experiment = ExperimentConfig(
            name="test-experiment",
            datasets=[dataset],
            models=[model],
            evaluation=evaluation,
        )

        config = BenchmarkConfig(
            version="1.0.0",
            experiments=[experiment],
            global_settings={"debug": True},
            logging_level="DEBUG",
        )

        assert config.version == "1.0.0"
        assert len(config.experiments) == 1
        assert config.global_settings == {"debug": True}
        assert config.logging_level == "DEBUG"

    def test_benchmark_config_defaults(self):
        """Test benchmark configuration with default values."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])
        experiment = ExperimentConfig(
            name="test-experiment",
            datasets=[dataset],
            models=[model],
            evaluation=evaluation,
        )

        config = BenchmarkConfig(experiments=[experiment])

        assert config.version == "1.0"
        assert config.global_settings == {}
        assert config.logging_level == "INFO"

    def test_benchmark_version_validation(self):
        """Test version format validation."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])
        experiment = ExperimentConfig(
            name="test", datasets=[dataset], models=[model], evaluation=evaluation
        )

        # Valid versions
        for version in ["1.0", "2.1.3", "10.0.0"]:
            config = BenchmarkConfig(experiments=[experiment], version=version)
            assert config.version == version

        # Invalid versions
        for invalid_version in ["1", "v1.0", "1.0.0.0", "abc"]:
            with pytest.raises(ValidationError) as exc_info:
                BenchmarkConfig(experiments=[experiment], version=invalid_version)

            error = exc_info.value.errors()[0]
            assert "Invalid version format" in error["msg"]

    def test_benchmark_logging_level_validation(self):
        """Test logging level validation."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])
        experiment = ExperimentConfig(
            name="test", datasets=[dataset], models=[model], evaluation=evaluation
        )

        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = BenchmarkConfig(experiments=[experiment], logging_level=level)
            assert config.logging_level == level

        # Case insensitive
        config = BenchmarkConfig(experiments=[experiment], logging_level="debug")
        assert config.logging_level == "DEBUG"

        # Invalid level
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig(experiments=[experiment], logging_level="INVALID")

        error = exc_info.value.errors()[0]
        assert "Invalid logging level" in error["msg"]

    def test_benchmark_unique_experiment_names(self):
        """Test unique experiment names validation."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])

        exp1 = ExperimentConfig(
            name="duplicate", datasets=[dataset], models=[model], evaluation=evaluation
        )
        exp2 = ExperimentConfig(
            name="duplicate", datasets=[dataset], models=[model], evaluation=evaluation
        )

        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig(experiments=[exp1, exp2])

        error = exc_info.value.errors()[0]
        assert "Duplicate experiment names" in error["msg"]


class TestConvenienceFunctions:
    """Test convenience functions for creating configurations."""

    def test_create_local_dataset_config(self):
        """Test creating local dataset configuration."""
        config = create_local_dataset_config(
            name="test-local",
            path="/data/local.csv",
            test_split=0.3,
            max_samples=500,
            preprocessing=["clean", "normalize"],
        )

        assert config.name == "test-local"
        assert config.source == "local"
        assert config.path == "/data/local.csv"
        assert config.test_split == 0.3
        assert config.max_samples == 500
        assert config.preprocessing == ["clean", "normalize"]

    def test_create_mlx_model_config(self):
        """Test creating MLX model configuration."""
        config = create_mlx_model_config(
            name="test-mlx",
            path="/models/test.mlx",
            max_tokens=1024,
            temperature=0.5,
            config={"device": "gpu"},
        )

        assert config.name == "test-mlx"
        assert config.type == "mlx_local"
        assert config.path == "/models/test.mlx"
        assert config.max_tokens == 1024
        assert config.temperature == 0.5
        assert config.config == {"device": "gpu"}

    def test_create_api_model_config(self):
        """Test creating API model configuration."""
        # OpenAI config
        config = create_api_model_config(
            name="gpt-test",
            provider="openai",
            model_name="gpt-3.5-turbo",
            max_tokens=1000,
        )

        assert config.name == "gpt-test"
        assert config.type == "openai_api"
        assert config.path == "gpt-3.5-turbo"
        assert config.max_tokens == 1000

        # Anthropic config
        config = create_api_model_config(
            name="claude-test",
            provider="anthropic",
            model_name="claude-3-sonnet-20240229",
        )

        assert config.name == "claude-test"
        assert config.type == "anthropic_api"
        assert config.path == "claude-3-sonnet-20240229"

        # Invalid provider
        with pytest.raises(ValueError) as exc_info:
            create_api_model_config("test", "invalid_provider", "model")

        assert "Unsupported provider" in str(exc_info.value)

    def test_create_standard_evaluation_config(self):
        """Test creating standard evaluation configuration."""
        # Default configuration
        config = create_standard_evaluation_config()

        expected_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "detection_rate",
            "false_positive_rate",
            "response_time",
        ]
        assert config.metrics == expected_metrics
        assert config.parallel_jobs == 1

        # Custom configuration
        config = create_standard_evaluation_config(
            metrics=["accuracy", "precision"],
            parallel_jobs=4,
            batch_size=16,
        )

        assert config.metrics == ["accuracy", "precision"]
        assert config.parallel_jobs == 4
        assert config.batch_size == 16


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_model_serialization(self):
        """Test model serialization to dictionary."""
        config = ModelConfig(
            name="test-model",
            type="mlx_local",
            path="/models/test.mlx",
            max_tokens=1024,
            config={"param": "value"},
        )

        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test-model"
        assert config_dict["type"] == "mlx_local"
        assert config_dict["config"] == {"param": "value"}

    def test_model_deserialization(self):
        """Test model creation from dictionary."""
        config_dict = {
            "name": "test-model",
            "type": "openai_api",
            "path": "gpt-3.5-turbo",
            "max_tokens": 512,
            "temperature": 0.1,
            "config": {},
        }

        config = ModelConfig.model_validate(config_dict)

        assert config.name == "test-model"
        assert config.type == "openai_api"
        assert config.path == "gpt-3.5-turbo"

    def test_json_serialization(self):
        """Test JSON serialization of complete configuration."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy", "precision"])
        experiment = ExperimentConfig(
            name="test-experiment",
            datasets=[dataset],
            models=[model],
            evaluation=evaluation,
        )
        config = BenchmarkConfig(experiments=[experiment])

        # Should be JSON serializable
        config_dict = config.model_dump()
        json_str = json.dumps(config_dict)
        assert isinstance(json_str, str)

        # Should be deserializable
        recovered_dict = json.loads(json_str)
        recovered_config = BenchmarkConfig.model_validate(recovered_dict)

        assert recovered_config.experiments[0].name == "test-experiment"
        assert len(recovered_config.experiments[0].datasets) == 1
        assert len(recovered_config.experiments[0].models) == 1


class TestConfigFileOperations:
    """Test configuration file loading and saving."""

    def test_save_and_load_json_config(self):
        """Test saving and loading configuration from JSON file."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])
        experiment = ExperimentConfig(
            name="test", datasets=[dataset], models=[model], evaluation=evaluation
        )
        original_config = BenchmarkConfig(experiments=[experiment])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save configuration
            save_config_to_file(original_config, temp_path, format="json")

            # Load configuration
            loaded_config = load_config_from_file(temp_path)

            assert loaded_config.experiments[0].name == original_config.experiments[0].name
            assert len(loaded_config.experiments) == len(original_config.experiments)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file("/nonexistent/path.json")

    def test_save_config_invalid_format(self):
        """Test saving configuration with invalid format."""
        dataset = DatasetConfig(name="test-data", source="local", path="/data/test.csv")
        model = ModelConfig(name="test-model", type="mlx_local", path="/models/test.mlx")
        evaluation = EvaluationConfig(metrics=["accuracy"])
        experiment = ExperimentConfig(
            name="test", datasets=[dataset], models=[model], evaluation=evaluation
        )
        config = BenchmarkConfig(experiments=[experiment])

        with pytest.raises(ValueError) as exc_info:
            save_config_to_file(config, "test.txt", format="invalid")

        assert "Unsupported format" in str(exc_info.value)


class TestComplexConfigurations:
    """Test complex, realistic configuration scenarios."""

    def test_multi_model_multi_dataset_experiment(self):
        """Test experiment with multiple models and datasets."""
        datasets = [
            DatasetConfig(name="unsw-nb15", source="kaggle", path="/data/unsw_nb15.csv"),
            DatasetConfig(
                name="kdd-cup", source="local", path="/data/kdd_cup.csv", max_samples=10000
            ),
        ]

        models = [
            ModelConfig(name="gpt-3.5", type="openai_api", path="gpt-3.5-turbo"),
            ModelConfig(name="claude", type="anthropic_api", path="claude-3-sonnet-20240229"),
            ModelConfig(name="local-llama", type="mlx_local", path="/models/llama-7b.mlx"),
        ]

        evaluation = EvaluationConfig(
            metrics=["accuracy", "precision", "recall", "f1_score", "response_time"],
            parallel_jobs=2,
            batch_size=16,
        )

        experiment = ExperimentConfig(
            name="comprehensive-eval",
            description="Multi-model evaluation on cybersecurity datasets",
            datasets=datasets,
            models=models,
            evaluation=evaluation,
        )

        config = BenchmarkConfig(
            experiments=[experiment],
            global_settings={"hardware_acceleration": True, "cache_models": True},
            logging_level="INFO",
        )

        # Should validate successfully
        assert len(config.experiments[0].datasets) == 2
        assert len(config.experiments[0].models) == 3
        assert len(config.experiments[0].evaluation.metrics) == 5

    def test_configuration_edge_cases(self):
        """Test configuration edge cases and boundary conditions."""
        # Maximum splits (just under 1.0)
        dataset = DatasetConfig(
            name="max-split",
            source="local",
            path="/data/test.csv",
            test_split=0.7,
            validation_split=0.29,  # 0.7 + 0.29 = 0.99 < 1.0
        )
        assert dataset.test_split + dataset.validation_split < 1.0

        # Maximum configuration values
        model = ModelConfig(
            name="max-model",
            type="custom",
            path="/models/custom.bin",
            max_tokens=4096,
            temperature=2.0,
        )
        assert model.max_tokens == 4096
        assert model.temperature == 2.0

        # Maximum evaluation settings
        evaluation = EvaluationConfig(
            metrics=["accuracy"],
            parallel_jobs=8,
            timeout_minutes=1440,  # 24 hours
            batch_size=128,
        )
        assert evaluation.parallel_jobs == 8
        assert evaluation.timeout_minutes == 1440

    def test_validation_error_details(self):
        """Test that validation errors provide useful details."""
        # Test invalid split combination
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(
                name="invalid-splits",
                source="local",
                path="/data/test.csv",
                test_split=0.6,
                validation_split=0.5,  # Total = 1.1 > 1.0
            )

        error = exc_info.value.errors()[0]
        assert "1.1" in error["msg"]  # Should show actual total
        assert "test_split" in error["msg"] or "validation_split" in error["msg"]

        # Test invalid metrics
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(metrics=["accuracy", "invalid_metric", "precision"])

        error = exc_info.value.errors()[0]
        assert "invalid_metric" in error["msg"]
        assert "Valid metrics" in error["msg"]


class TestConfigurationIntegration:
    """Test integration between configuration components."""

    def test_complete_configuration_workflow(self):
        """Test complete configuration creation and validation workflow."""
        # Create realistic cybersecurity benchmark configuration
        datasets = [
            create_local_dataset_config(
                name="malware-detection",
                path="/data/malware_samples.csv",
                test_split=0.25,
                validation_split=0.15,
                preprocessing=["tokenize", "normalize", "feature_extract"],
            ),
            DatasetConfig(
                name="network-intrusion",
                source="kaggle",
                path="unsw-nb15/UNSW-NB15.csv",
                max_samples=50000,
            ),
        ]

        models = [
            create_mlx_model_config(
                name="llama-cybersec",
                path="/models/llama-7b-cybersec-ft.mlx",
                max_tokens=1024,
                temperature=0.3,
                config={"device": "mps", "precision": "float16"},
            ),
            create_api_model_config(
                name="gpt-4-analysis",
                provider="openai",
                model_name="gpt-4-0125-preview",
                max_tokens=2048,
                temperature=0.1,
            ),
        ]

        evaluation = create_standard_evaluation_config(
            metrics=[
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "detection_rate",
                "explainability_score",
            ],
            parallel_jobs=4,
            timeout_minutes=90,
        )

        experiment = ExperimentConfig(
            name="cybersec-llm-benchmark",
            description="Comprehensive evaluation of LLMs for cybersecurity tasks",
            output_dir="./results/cybersec_eval",
            datasets=datasets,
            models=models,
            evaluation=evaluation,
        )

        config = BenchmarkConfig(
            version="1.2.0",
            experiments=[experiment],
            global_settings={
                "hardware_optimization": True,
                "result_caching": True,
                "detailed_logging": True,
            },
            logging_level="INFO",
        )

        # Validate complete configuration
        assert config.version == "1.2.0"
        assert len(config.experiments) == 1
        assert len(config.experiments[0].datasets) == 2
        assert len(config.experiments[0].models) == 2
        assert "explainability_score" in config.experiments[0].evaluation.metrics

        # Test serialization
        config_dict = config.model_dump()
        json.dumps(config_dict)  # Should not raise

        # Test roundtrip
        recovered_config = BenchmarkConfig.model_validate(config_dict)
        assert recovered_config.experiments[0].name == experiment.name

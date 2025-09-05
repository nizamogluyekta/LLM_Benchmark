"""
Integration tests for sample configuration files.

Tests that all sample configuration files:
1. Load successfully without validation errors
2. Have properly formatted environment variables
3. Contain valid configuration structures
4. Pass configuration inheritance correctly
5. Meet minimum requirements for experiments
"""

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.core.config import BenchmarkConfig, ExperimentConfig
from benchmark.core.config_loader import ConfigurationLoader, load_experiment_config
from benchmark.core.exceptions import ConfigurationError


# Global fixtures
@pytest.fixture(scope="session")
def configs_dir():
    """Get the configs directory path."""
    return Path(__file__).parent.parent.parent / "configs"


@pytest.fixture
def config_loader():
    """Create a configuration loader for testing."""
    return ConfigurationLoader(cache_enabled=False)


@pytest.fixture(autouse=True)
def setup_environment():
    """Set up test environment variables."""
    test_env = {
        # API Keys (fake for testing)
        "OPENAI_API_KEY": "test-key-openai",
        "ANTHROPIC_API_KEY": "test-key-anthropic",
        "KAGGLE_USERNAME": "test-user",
        "KAGGLE_KEY": "test-key-kaggle",
        "HF_TOKEN": "test-token-hf",
        "OPENAI_ORG_ID": "test-org-id",
        # Directories
        "RESULTS_DIR": "./test_results",
        "DATA_DIR": "./test_data",
        "MODEL_DIR": "./test_models",
        "TEMP_DIR": "/tmp/test",
        # Other settings
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "INFO",
        "DEBUG": "false",
        "MLX_DEVICE": "cpu",  # Use CPU for testing
        "WANDB_ENTITY": "test-entity",  # For wandb integration
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",  # For slack integration
        "MLFLOW_TRACKING_URI": "http://localhost:5000",  # For MLflow
        "SMTP_SERVER": "smtp.gmail.com",  # For email notifications
        "SMTP_USERNAME": "test@example.com",  # For email notifications
        "SMTP_PASSWORD": "test-password",  # For email notifications
        "SMTP_PORT": "587",  # For SMTP port
        "AWS_S3_BUCKET": "test-bucket",  # For AWS S3
        "AWS_REGION": "us-east-1",  # For AWS
        "GCP_STORAGE_BUCKET": "test-gcp-bucket",  # For GCP
        "GCP_PROJECT": "test-project",  # For GCP
    }

    with patch.dict(os.environ, test_env, clear=False):
        yield


class TestSampleConfigurations:
    """Test suite for sample configuration files."""

    def test_basic_evaluation_config_loads(self, config_loader, configs_dir):
        """Test that basic_evaluation.yaml loads successfully."""
        config_path = configs_dir / "experiments" / "basic_evaluation.yaml"

        # Should load without errors
        config = config_loader.load_experiment_config(config_path)

        # Verify it's a valid ExperimentConfig
        assert isinstance(config, ExperimentConfig)
        assert config.name == "cybersec-basic-evaluation"
        assert config.description

        # Check required components
        assert len(config.datasets) >= 1
        assert len(config.models) >= 1
        assert config.evaluation is not None

        # Verify environment variable resolution
        assert "test_results" in config.output_dir

        # Check dataset configuration
        dataset = config.datasets[0]
        assert dataset.name == "malware-detection-basic"
        assert dataset.source == "local"
        assert "test_data" in dataset.path

        # Check model configuration
        model = config.models[0]
        assert model.name == "gpt-3.5-cybersec"
        assert model.type == "openai_api"
        assert model.config["api_key"] == "test-key-openai"

        # Check evaluation settings
        assert config.evaluation.batch_size == 8
        assert "accuracy" in config.evaluation.metrics

    def test_model_comparison_config_loads(self, config_loader, configs_dir):
        """Test that model_comparison.yaml loads successfully."""
        config_path = configs_dir / "experiments" / "model_comparison.yaml"

        config = config_loader.load_experiment_config(config_path)

        assert isinstance(config, ExperimentConfig)
        assert config.name == "cybersec-model-comparison"

        # Should have multiple models for comparison
        assert len(config.models) >= 3

        model_names = [m.name for m in config.models]
        expected_models = ["gpt-4-cybersec", "gpt-3.5-cybersec", "claude-3-sonnet-cybersec"]

        for expected in expected_models:
            assert expected in model_names

        # Check that all models have consistent temperature (0.0 for comparison)
        for model in config.models:
            if model.type in ["openai_api", "anthropic_api"]:
                assert model.temperature == 0.0

        # Verify comprehensive metrics for comparison
        metrics = config.evaluation.metrics
        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in expected_metrics:
            assert metric in metrics

    def test_local_models_config_loads(self, config_loader, configs_dir):
        """Test that local_models.yaml loads successfully."""
        config_path = configs_dir / "models" / "local_models.yaml"

        config = config_loader.load_experiment_config(config_path)

        assert isinstance(config, ExperimentConfig)
        assert config.name == "local-mlx-models"

        # Check that all models are MLX type
        for model in config.models:
            assert model.type == "mlx_local"
            assert "test_models" in model.path
            assert model.config["device"] == "cpu"  # From test environment

        # Verify MLX-specific configurations
        model = config.models[0]  # Check first model
        assert "precision" in model.config
        assert "quantization" in model.config
        assert model.config["precision"] == "float16"

    def test_public_datasets_config_loads(self, config_loader, configs_dir):
        """Test that public_datasets.yaml loads successfully."""
        config_path = configs_dir / "datasets" / "public_datasets.yaml"

        config = config_loader.load_experiment_config(config_path)

        assert isinstance(config, ExperimentConfig)
        assert config.name == "public-cybersecurity-datasets"

        # Check various dataset sources
        sources = {ds.source for ds in config.datasets}
        expected_sources = {"kaggle", "huggingface", "remote", "local", "synthetic"}

        # Should have at least some of the expected sources
        assert len(sources.intersection(expected_sources)) >= 2

        # Verify Kaggle dataset configuration
        kaggle_datasets = [ds for ds in config.datasets if ds.source == "kaggle"]
        if kaggle_datasets:
            kaggle_ds = kaggle_datasets[0]
            assert kaggle_ds.name
            assert kaggle_ds.path

    def test_default_config_loads(self, config_loader, configs_dir):
        """Test that default.yaml loads successfully."""
        config_path = configs_dir / "default.yaml"

        # Note: default.yaml is not a complete experiment config
        # It's a template/defaults file, so we test loading it as raw YAML
        yaml_content = config_loader._load_yaml_file(config_path)

        assert yaml_content["name"] == "llm-cybersec-benchmark-defaults"
        assert "global_settings" in yaml_content
        assert "default_dataset" in yaml_content
        assert "default_model" in yaml_content
        assert "evaluation" in yaml_content

    def test_configuration_inheritance(self, config_loader, configs_dir):
        """Test configuration inheritance works with sample configs."""
        base_path = configs_dir / "default.yaml"
        override_path = configs_dir / "experiments" / "basic_evaluation.yaml"

        # Load with inheritance
        config = config_loader.load_experiment_config(override_path, base_config_path=base_path)

        assert isinstance(config, ExperimentConfig)

        # Should have experiment-specific values
        assert config.name == "cybersec-basic-evaluation"

        # Should inherit evaluation metrics from defaults if not overridden
        assert config.evaluation is not None

    def test_environment_variable_patterns(self, configs_dir):
        """Test that environment variables are correctly formatted."""
        env_var_pattern = re.compile(r"\$\{([^}]+)\}")

        config_files = [
            configs_dir / "experiments" / "basic_evaluation.yaml",
            configs_dir / "experiments" / "model_comparison.yaml",
            configs_dir / "models" / "local_models.yaml",
            configs_dir / "datasets" / "public_datasets.yaml",
            configs_dir / "default.yaml",
        ]

        for config_file in config_files:
            content = config_file.read_text()

            # Find all environment variable references
            env_vars = env_var_pattern.findall(content)

            for env_var in env_vars:
                # Check format: either VAR_NAME or VAR_NAME:default
                assert (
                    ":" in env_var or env_var.replace("_", "").isalnum()
                ), f"Invalid env var format in {config_file}: ${{{env_var}}}"

                if ":" in env_var:
                    var_name, default = env_var.split(":", 1)
                    assert var_name.replace(
                        "_", ""
                    ).isalnum(), f"Invalid env var name in {config_file}: {var_name}"

    def test_all_configs_have_required_fields(self, config_loader, configs_dir):
        """Test that all experiment configs have required fields."""
        experiment_configs = [
            configs_dir / "experiments" / "basic_evaluation.yaml",
            configs_dir / "experiments" / "model_comparison.yaml",
        ]

        for config_path in experiment_configs:
            config = config_loader.load_experiment_config(config_path)

            # Required fields for experiments
            assert config.name
            assert config.description
            assert len(config.datasets) >= 1
            assert len(config.models) >= 1
            assert config.evaluation is not None

            # Each dataset should have required fields
            for dataset in config.datasets:
                assert dataset.name
                assert dataset.source
                assert dataset.path

            # Each model should have required fields
            for model in config.models:
                assert model.name
                assert model.type
                assert model.path
                assert model.max_tokens > 0

    def test_config_validation_warnings(self, config_loader, configs_dir):
        """Test that configs generate appropriate validation warnings."""
        config_path = configs_dir / "experiments" / "basic_evaluation.yaml"
        config = config_loader.load_experiment_config(config_path)

        # Run validation and check for warnings
        warnings = config_loader.validate_configuration(config)

        # Should not have critical warnings for sample configs
        warning_text = " ".join(warnings) if warnings else ""

        # Check that we don't have major issues
        assert "very small sample size" not in warning_text
        assert "very small test split" not in warning_text

    def test_benchmark_config_creation(self, configs_dir):
        """Test creating a benchmark config with multiple experiments."""
        # Create a temporary benchmark config that references our sample experiments
        benchmark_config = {
            "version": "1.2.0",
            "logging_level": "INFO",
            "global_settings": {"hardware_optimization": True},
            "experiments": [
                # Load the basic evaluation as an experiment
                load_experiment_config(configs_dir / "experiments" / "basic_evaluation.yaml")
            ],
        }

        # Convert to BenchmarkConfig (this tests the structure)
        config = BenchmarkConfig.model_validate(
            {
                "version": "1.2.0",
                "logging_level": "INFO",
                "global_settings": {"hardware_optimization": True},
                "experiments": [benchmark_config["experiments"][0].model_dump()],
            }
        )

        assert config.version == "1.2.0"
        assert len(config.experiments) == 1
        assert config.experiments[0].name == "cybersec-basic-evaluation"


class TestEnvironmentVariableTemplates:
    """Test environment variable template functionality."""

    def test_env_example_file_exists(self):
        """Test that .env.example exists and has correct structure."""
        env_example_path = Path(__file__).parent.parent.parent / ".env.example"
        assert env_example_path.exists()

        content = env_example_path.read_text()

        # Check for key sections
        assert "OPENAI_API_KEY" in content
        assert "RESULTS_DIR" in content
        assert "DATA_DIR" in content
        assert "MODEL_DIR" in content

        # Check for documentation
        assert "# " in content  # Has comments
        assert "Example:" in content  # Has usage examples

    def test_environment_variables_with_defaults(self, configs_dir):
        """Test environment variable resolution with default values."""
        config_loader = ConfigurationLoader()

        # Test with minimal environment (no optional vars set)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            config_path = configs_dir / "experiments" / "basic_evaluation.yaml"
            config = config_loader.load_experiment_config(config_path)

            # Should use default values
            assert "results" in config.output_dir  # Default RESULTS_DIR
            dataset = config.datasets[0]
            assert "data" in dataset.path  # Default DATA_DIR

    def test_missing_required_environment_variables(self, config_loader, configs_dir):
        """Test handling of missing required environment variables."""
        # Clear environment of API keys
        with patch.dict(os.environ, {}, clear=True):
            config_path = configs_dir / "experiments" / "basic_evaluation.yaml"

            # Should fail when trying to resolve required API key
            with pytest.raises(ConfigurationError) as exc_info:
                config_loader.load_experiment_config(config_path)

            assert "OPENAI_API_KEY" in str(exc_info.value)


class TestConfigurationUseCases:
    """Test common configuration use cases."""

    def test_quick_development_setup(self, configs_dir):
        """Test a typical development setup scenario."""
        test_env = {
            "OPENAI_API_KEY": "test-key",
            "RESULTS_DIR": "./dev_results",
            "LOG_LEVEL": "DEBUG",
            "DEBUG": "true",
            "ENVIRONMENT": "development",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config_loader = ConfigurationLoader()
            config_path = configs_dir / "experiments" / "basic_evaluation.yaml"

            config = config_loader.load_experiment_config(config_path)

            # Should work with minimal setup
            assert config.name == "cybersec-basic-evaluation"
            assert "dev_results" in config.output_dir

    def test_production_deployment_setup(self, configs_dir):
        """Test a production deployment scenario."""
        test_env = {
            "OPENAI_API_KEY": "prod-key",
            "ANTHROPIC_API_KEY": "prod-anthropic-key",
            "RESULTS_DIR": "/var/benchmark/results",
            "DATA_DIR": "/var/benchmark/data",
            "MODEL_DIR": "/var/benchmark/models",
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "WARNING",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config_loader = ConfigurationLoader()
            config_path = configs_dir / "experiments" / "model_comparison.yaml"

            config = config_loader.load_experiment_config(config_path)

            # Should work with production paths
            assert "/var/benchmark/results" in config.output_dir

            # Multiple models should all have their API keys resolved
            for model in config.models:
                if model.type == "openai_api":
                    assert model.config["api_key"] == "prod-key"
                elif model.type == "anthropic_api":
                    assert model.config["api_key"] == "prod-anthropic-key"

    def test_local_only_setup(self, configs_dir):
        """Test a local-only setup with MLX models."""
        test_env = {
            "MODEL_DIR": "./local_models",
            "MLX_DEVICE": "mps",
            "RESULTS_DIR": "./local_results",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config_loader = ConfigurationLoader()
            config_path = configs_dir / "models" / "local_models.yaml"

            config = config_loader.load_experiment_config(config_path)

            # Should work without API keys
            assert config.name == "local-mlx-models"

            # All models should be local
            for model in config.models:
                assert model.type == "mlx_local"
                assert "local_models" in model.path
                assert model.config["device"] == "mps"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

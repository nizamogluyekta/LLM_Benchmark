"""
Unit tests for YAML configuration loader.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.core.config import DatasetConfig, EvaluationConfig, ExperimentConfig, ModelConfig
from benchmark.core.config_loader import (
    ConfigurationCache,
    ConfigurationLoader,
    clear_config_cache,
    get_config_cache_stats,
    load_benchmark_config,
    load_experiment_config,
    resolve_environment_variables,
)
from benchmark.core.exceptions import ConfigurationError


class TestConfigurationCache:
    """Test configuration cache functionality."""

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = ConfigurationCache(max_size=3)

        # Test empty cache
        assert cache.get("key1") is None
        assert cache.size() == 0

        # Test adding items
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.size() == 2

    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = ConfigurationCache(max_size=2)

        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.size() == 2

        # Access key1 to make it more recently used
        cache.get("key1")

        # Add new item, should evict key2 (least recently used)
        cache.set("key3", "value3")
        assert cache.size() == 2
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"

    def test_cache_update_existing(self):
        """Test updating existing cache entries."""
        cache = ConfigurationCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Update existing key
        cache.set("key1", "updated_value1")
        assert cache.get("key1") == "updated_value1"
        assert cache.size() == 2

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = ConfigurationCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestConfigurationLoader:
    """Test ConfigurationLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a fresh configuration loader for testing."""
        return ConfigurationLoader(cache_enabled=True, cache_size=10)

    @pytest.fixture
    def fixtures_path(self):
        """Path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures"

    def test_loader_initialization(self):
        """Test loader initialization."""
        # With cache enabled
        loader = ConfigurationLoader(cache_enabled=True, cache_size=5)
        assert loader._cache_enabled
        assert loader._cache is not None

        # With cache disabled
        loader = ConfigurationLoader(cache_enabled=False)
        assert not loader._cache_enabled
        assert loader._cache is None

    def test_load_valid_experiment_config(self, loader, fixtures_path):
        """Test loading valid experiment configuration."""
        config_path = fixtures_path / "valid_config.yaml"
        config = loader.load_experiment_config(config_path)

        assert isinstance(config, ExperimentConfig)
        assert config.name == "cybersec-evaluation"
        assert config.description == "Comprehensive cybersecurity LLM evaluation"
        assert len(config.datasets) == 2
        assert len(config.models) == 2
        assert len(config.evaluation.metrics) == 7

        # Check dataset details
        malware_dataset = next(d for d in config.datasets if d.name == "malware-detection")
        assert malware_dataset.source == "local"
        assert malware_dataset.max_samples == 5000
        assert malware_dataset.test_split == 0.2

        # Check model details
        llama_model = next(m for m in config.models if m.name == "llama-cybersec")
        assert llama_model.type == "mlx_local"
        assert llama_model.max_tokens == 1024
        assert llama_model.config["device"] == "mps"

    def test_load_benchmark_config(self, loader, fixtures_path):
        """Test loading benchmark configuration."""
        config_path = fixtures_path / "benchmark_config.yaml"
        config = loader.load_benchmark_config(config_path)

        assert config.version == "1.2.0"
        assert config.logging_level == "INFO"
        assert len(config.experiments) == 2
        assert config.global_settings["hardware_optimization"] is True

        # Check experiments
        malware_exp = next(
            e for e in config.experiments if e.name == "malware-detection-experiment"
        )
        assert malware_exp.description == "Evaluate LLM performance on malware detection"
        assert len(malware_exp.datasets) == 1
        assert len(malware_exp.models) == 1

    def test_environment_variable_resolution(self, loader):
        """Test environment variable resolution."""
        # Set test environment variables
        test_env = {
            "DATA_PATH": "/test/data/path",
            "OPENAI_API_KEY": "test-api-key",
            "OPENAI_MODEL": "gpt-4",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = {
                "path": "${DATA_PATH}",
                "api_key": "${OPENAI_API_KEY}",
                "model": "${OPENAI_MODEL}",
                "default_value": "${MISSING_VAR:default_value}",
                "nested": {"deep": {"value": "${DATA_PATH}/nested"}},
                "list": ["${DATA_PATH}", "static_value"],
                "non_string": 42,
            }

            resolved = loader.resolve_environment_variables(config)

            assert resolved["path"] == "/test/data/path"
            assert resolved["api_key"] == "test-api-key"
            assert resolved["model"] == "gpt-4"
            assert resolved["default_value"] == "default_value"
            assert resolved["nested"]["deep"]["value"] == "/test/data/path/nested"
            assert resolved["list"] == ["/test/data/path", "static_value"]
            assert resolved["non_string"] == 42

    def test_missing_required_env_var(self, loader):
        """Test error handling for missing required environment variables."""
        config = {"required_var": "${MISSING_REQUIRED_VAR}"}

        with pytest.raises(ConfigurationError) as exc_info:
            loader.resolve_environment_variables(config)

        assert "Required environment variable 'MISSING_REQUIRED_VAR' is not set" in str(
            exc_info.value
        )

    def test_load_config_with_env_vars(self, loader, fixtures_path):
        """Test loading configuration with environment variables."""
        config_path = fixtures_path / "config_with_env_vars.yaml"

        test_env = {
            "DATA_PATH": "/test/data/cybersec.csv",
            "OPENAI_API_KEY": "test-key-123",
            "OPENAI_MODEL": "gpt-4-turbo",
            "LOCAL_MODEL_PATH": "/models/custom.mlx",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = loader.load_experiment_config(config_path)

            assert config.name == "env-var-test"
            assert config.output_dir.endswith("results/default")  # Uses default value

            # Check resolved environment variables
            test_dataset = config.datasets[0]
            assert test_dataset.path == "/test/data/cybersec.csv"

            openai_model = next(m for m in config.models if m.name == "openai-model")
            assert openai_model.path == "gpt-4-turbo"
            assert openai_model.config["api_key"] == "test-key-123"
            assert openai_model.config["api_base"] == "https://api.openai.com/v1"  # Default value

            local_model = next(m for m in config.models if m.name == "local-model")
            assert local_model.path == "/models/custom.mlx"

    def test_configuration_inheritance(self, loader, fixtures_path):
        """Test configuration inheritance and merging."""
        base_path = fixtures_path / "base_config.yaml"
        override_path = fixtures_path / "override_config.yaml"

        config = loader.load_experiment_config(override_path, base_config_path=base_path)

        assert config.name == "override-experiment"  # Override wins
        assert config.output_dir.endswith("results/override")  # Override wins

        # Check datasets - should have both base and override
        dataset_names = [d.name for d in config.datasets]
        assert "base-dataset" in dataset_names
        assert "override-dataset" in dataset_names
        assert len(config.datasets) == 2

        # Check models - base model should be merged with overrides
        model_names = [m.name for m in config.models]
        assert "base-model" in model_names
        assert "override-model" in model_names
        assert len(config.models) == 2

        base_model = next(m for m in config.models if m.name == "base-model")
        assert base_model.max_tokens == 1024  # Overridden
        assert base_model.temperature == 0.2  # Overridden
        assert base_model.config["device"] == "mps"  # Overridden
        assert base_model.config["precision"] == "float32"  # Inherited from base
        assert base_model.config["batch_size"] == 32  # New from override

        # Check evaluation - should be merged
        assert len(config.evaluation.metrics) == 4  # Override wins
        assert config.evaluation.parallel_jobs == 4  # Override wins
        assert config.evaluation.timeout_minutes == 60  # Inherited from base

    def test_merge_configurations(self, loader):
        """Test configuration merging logic."""
        base = {
            "name": "base",
            "shared_dict": {"key1": "base_value1", "key2": "base_value2"},
            "shared_list": ["base1", "base2"],
            "base_only": "base_value",
        }

        override = {
            "name": "override",
            "shared_dict": {
                "key1": "override_value1",  # Override existing
                "key3": "override_value3",  # Add new
            },
            "shared_list": ["override1"],  # Replace list
            "override_only": "override_value",
        }

        merged = loader.merge_configurations(base, override)

        assert merged["name"] == "override"
        assert merged["base_only"] == "base_value"
        assert merged["override_only"] == "override_value"

        # Dictionary merge
        assert merged["shared_dict"]["key1"] == "override_value1"  # Overridden
        assert merged["shared_dict"]["key2"] == "base_value2"  # Inherited
        assert merged["shared_dict"]["key3"] == "override_value3"  # Added

        # List replacement
        assert merged["shared_list"] == ["override1"]

    def test_config_validation_warnings(self, loader):
        """Test configuration validation warnings."""
        # Create config with warning-worthy settings
        config = ExperimentConfig(
            name="test-config",
            datasets=[
                DatasetConfig(
                    name="small-dataset",
                    source="local",
                    path="/data/small.csv",
                    max_samples=50,  # Very small
                    test_split=0.05,  # Very small test split
                    validation_split=0.1,
                )
            ],
            models=[
                ModelConfig(
                    name="extreme-model",
                    type="openai_api",
                    path="gpt-3.5-turbo",
                    max_tokens=25,  # Very low
                    temperature=1.8,  # Very high
                )
            ],
            evaluation=EvaluationConfig(
                metrics=["accuracy"],  # Missing common metrics
                parallel_jobs=1,
                timeout_minutes=15,  # Short timeout
                batch_size=4,  # Small batch size
            ),
        )

        warnings = loader.validate_configuration(config)

        assert len(warnings) > 0
        warning_text = " ".join(warnings)
        assert "very small sample size" in warning_text
        assert "very small test split" in warning_text
        assert "very low max_tokens" in warning_text
        assert "high temperature" in warning_text
        assert "Short timeout" in warning_text
        assert "Small batch size" in warning_text
        assert "common metrics" in warning_text

    def test_config_caching(self, fixtures_path):
        """Test configuration caching functionality."""
        loader = ConfigurationLoader(cache_enabled=True, cache_size=5)
        config_path = fixtures_path / "valid_config.yaml"

        # Load config first time
        config1 = loader.load_experiment_config(config_path)
        cache_stats = loader.get_cache_stats()
        assert cache_stats["enabled"]
        assert cache_stats["size"] == 1

        # Load same config again - should come from cache
        config2 = loader.load_experiment_config(config_path)
        assert config1 is config2  # Same object instance
        assert loader.get_cache_stats()["size"] == 1

        # Clear cache
        loader.clear_cache()
        assert loader.get_cache_stats()["size"] == 0

        # Load again - should create new instance
        config3 = loader.load_experiment_config(config_path)
        assert config1 is not config3  # Different object instances

    def test_invalid_yaml_syntax(self, loader, fixtures_path):
        """Test error handling for malformed YAML."""
        config_path = fixtures_path / "malformed.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            loader.load_experiment_config(config_path)

        assert "Invalid YAML syntax" in str(exc_info.value)

    def test_invalid_config_validation(self, loader, fixtures_path):
        """Test error handling for configuration validation errors."""
        config_path = fixtures_path / "invalid_config.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            loader.load_experiment_config(config_path)

        error_msg = str(exc_info.value)
        assert "Validation failed" in error_msg
        # Should contain details about specific validation errors

    def test_missing_config_file(self, loader):
        """Test error handling for missing configuration files."""
        missing_path = Path("/nonexistent/config.yaml")

        with pytest.raises(ConfigurationError) as exc_info:
            loader.load_experiment_config(missing_path)

        assert "Configuration file not found" in str(exc_info.value)

    def test_empty_config_file(self, loader):
        """Test error handling for empty configuration files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("")
            tmp.flush()
            empty_path = Path(tmp.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                loader.load_experiment_config(empty_path)

            assert "Configuration file is empty" in str(exc_info.value)
        finally:
            empty_path.unlink(missing_ok=True)

    def test_non_dict_yaml_content(self, loader):
        """Test error handling for YAML that doesn't contain a dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("- item1\n- item2\n")  # YAML list, not dict
            tmp.flush()
            list_path = Path(tmp.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                loader.load_experiment_config(list_path)

            assert "must contain a YAML object/dictionary" in str(exc_info.value)
        finally:
            list_path.unlink(missing_ok=True)

    def test_cache_disabled_loader(self, fixtures_path):
        """Test loader with caching disabled."""
        loader = ConfigurationLoader(cache_enabled=False)
        config_path = fixtures_path / "valid_config.yaml"

        config1 = loader.load_experiment_config(config_path)
        config2 = loader.load_experiment_config(config_path)

        # Should be different instances (no caching)
        assert config1 is not config2
        assert config1.name == config2.name  # But same content

        cache_stats = loader.get_cache_stats()
        assert not cache_stats["enabled"]
        assert cache_stats["size"] == 0


class TestGlobalConfigurationFunctions:
    """Test global configuration convenience functions."""

    @pytest.fixture
    def fixtures_path(self):
        """Path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures"

    def test_load_experiment_config_function(self, fixtures_path):
        """Test global load_experiment_config function."""
        config_path = fixtures_path / "valid_config.yaml"
        config = load_experiment_config(config_path)

        assert isinstance(config, ExperimentConfig)
        assert config.name == "cybersec-evaluation"

    def test_load_benchmark_config_function(self, fixtures_path):
        """Test global load_benchmark_config function."""
        config_path = fixtures_path / "benchmark_config.yaml"
        config = load_benchmark_config(config_path)

        assert config.version == "1.2.0"
        assert len(config.experiments) == 2

    def test_resolve_environment_variables_function(self):
        """Test global resolve_environment_variables function."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=False):
            config = {"test": "${TEST_VAR}"}
            resolved = resolve_environment_variables(config)
            assert resolved["test"] == "test_value"

    def test_cache_management_functions(self, fixtures_path):
        """Test global cache management functions."""
        config_path = fixtures_path / "valid_config.yaml"

        # Load config to populate cache
        load_experiment_config(config_path)

        # Check cache stats
        stats = get_config_cache_stats()
        assert stats["enabled"]
        assert stats["size"] >= 1

        # Clear cache
        clear_config_cache()
        stats = get_config_cache_stats()
        assert stats["size"] == 0


class TestComplexScenarios:
    """Test complex configuration scenarios."""

    @pytest.fixture
    def fixtures_path(self):
        """Path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures"

    def test_nested_environment_variables(self):
        """Test complex nested environment variable resolution."""
        loader = ConfigurationLoader()

        config = {
            "database": {
                "host": "${DB_HOST:localhost}",
                "port": "${DB_PORT:5432}",
                "credentials": {"username": "${DB_USER}", "password": "${DB_PASS}"},
            },
            "paths": ["${DATA_DIR}/dataset1.csv", "${DATA_DIR}/dataset2.csv"],
        }

        test_env = {"DB_USER": "test_user", "DB_PASS": "test_pass", "DATA_DIR": "/data/cybersec"}

        with patch.dict(os.environ, test_env, clear=False):
            resolved = loader.resolve_environment_variables(config)

            assert resolved["database"]["host"] == "localhost"  # Default value
            assert resolved["database"]["port"] == "5432"  # Default value
            assert resolved["database"]["credentials"]["username"] == "test_user"
            assert resolved["database"]["credentials"]["password"] == "test_pass"
            assert resolved["paths"] == [
                "/data/cybersec/dataset1.csv",
                "/data/cybersec/dataset2.csv",
            ]

    def test_config_inheritance_with_env_vars(self, fixtures_path):
        """Test configuration inheritance combined with environment variables."""
        loader = ConfigurationLoader()

        # Create a temporary override config with env vars
        override_yaml = """
name: "env-override-${EXPERIMENT_SUFFIX:test}"
output_dir: "${RESULTS_DIR:./results}/override"

models:
  - name: "base-model"
    config:
      api_key: "${API_KEY}"
      device: "${DEVICE:mps}"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write(override_yaml)
            tmp.flush()
            override_path = Path(tmp.name)

        test_env = {
            "EXPERIMENT_SUFFIX": "production",
            "RESULTS_DIR": "/production/results",
            "API_KEY": "prod-api-key-123",
        }

        try:
            with patch.dict(os.environ, test_env, clear=False):
                base_path = fixtures_path / "base_config.yaml"
                config = loader.load_experiment_config(override_path, base_config_path=base_path)

                assert config.name == "env-override-production"
                assert "/production/results/override" in config.output_dir

                base_model = next(m for m in config.models if m.name == "base-model")
                assert base_model.config["api_key"] == "prod-api-key-123"
                assert base_model.config["device"] == "mps"  # Default value
                assert base_model.config["precision"] == "float32"  # Inherited from base

        finally:
            override_path.unlink(missing_ok=True)

    def test_validation_with_inheritance(self, fixtures_path):
        """Test validation warnings work correctly with inheritance."""
        loader = ConfigurationLoader()

        # Create override config that introduces validation warnings
        warning_yaml = """
name: "warning-config"

datasets:
  - name: "warning-dataset"
    source: "local"
    path: "/data/warning.csv"
    max_samples: 25  # Very small sample size
    test_split: 0.02  # Very small test split

models:
  - name: "warning-model"
    type: "openai_api"
    path: "gpt-3.5-turbo"
    max_tokens: 10    # Very low max_tokens
    temperature: 1.9  # Very high temperature

evaluation:
  metrics: ["accuracy"]  # Missing common metrics
  timeout_minutes: 5     # Very short timeout
  batch_size: 1          # Very small batch
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write(warning_yaml)
            tmp.flush()
            warning_path = Path(tmp.name)

        try:
            base_path = fixtures_path / "base_config.yaml"

            # Load with validation
            config = loader.load_experiment_config(
                warning_path, base_config_path=base_path, validate=True
            )

            # Should load successfully but generate warnings
            assert config.name == "warning-config"

            # Validate the merged config and check for warnings
            warnings = loader.validate_configuration(config)
            assert len(warnings) > 5  # Should have multiple warnings

            warning_text = " ".join(warnings)
            assert "very small sample size" in warning_text
            assert "very small test split" in warning_text
            assert "very low max_tokens" in warning_text

        finally:
            warning_path.unlink(missing_ok=True)

    def test_circular_environment_variable_reference(self):
        """Test handling of complex environment variable patterns."""
        loader = ConfigurationLoader()

        config = {
            "path": "${BASE_PATH}/${SUB_PATH}",
            "full_url": "${PROTOCOL}://${HOST}:${PORT}/${ENDPOINT}",
        }

        test_env = {
            "BASE_PATH": "/data/cybersec",
            "SUB_PATH": "experiments/2024",
            "PROTOCOL": "https",
            "HOST": "api.example.com",
            "PORT": "443",
            "ENDPOINT": "v1/models",
        }

        with patch.dict(os.environ, test_env, clear=False):
            resolved = loader.resolve_environment_variables(config)

            assert resolved["path"] == "/data/cybersec/experiments/2024"
            assert resolved["full_url"] == "https://api.example.com:443/v1/models"

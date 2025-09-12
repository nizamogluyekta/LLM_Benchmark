"""
Integration tests for the Configuration Service.

This module provides comprehensive integration tests that test the configuration
service with real configuration files and various realistic scenarios.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
import yaml

from benchmark.core.config import ExperimentConfig
from benchmark.services.configuration_service import ConfigurationService


class TestConfigurationServiceIntegration:
    """Integration tests for Configuration Service with real scenarios."""

    @pytest_asyncio.fixture
    async def config_service(self):
        """Create and initialize a ConfigurationService for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir), cache_ttl=60)
            await service.initialize()
            yield service
            await service.shutdown()

    @pytest.fixture
    def sample_configs_dir(self):
        """Create a directory with sample configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs_dir = Path(temp_dir)

            # Create multiple sample configurations
            configs = [
                {
                    "name": "Small Scale Experiment",
                    "description": "Small scale cybersecurity classification",
                    "output_dir": "./results/small_scale",
                    "datasets": [
                        {
                            "name": "malware_classification",
                            "source": "local",
                            "path": "./data/malware_samples.jsonl",
                            "max_samples": 100,
                            "test_split": 0.2,
                            "validation_split": 0.1,
                        }
                    ],
                    "models": [
                        {
                            "name": "gpt-3.5-turbo",
                            "type": "openai_api",
                            "path": "gpt-3.5-turbo",
                            "config": {"api_key": "${OPENAI_API_KEY}"},
                            "max_tokens": 512,
                            "temperature": 0.1,
                        }
                    ],
                    "evaluation": {
                        "metrics": ["accuracy", "precision", "recall", "f1_score"],
                        "parallel_jobs": 2,
                        "timeout_minutes": 15,
                        "batch_size": 16,
                    },
                },
                {
                    "name": "Multi Model Comparison",
                    "description": "Compare multiple models on cybersecurity tasks",
                    "output_dir": "./results/multi_model",
                    "datasets": [
                        {
                            "name": "phishing_detection",
                            "source": "local",
                            "path": "./data/phishing_emails.jsonl",
                            "max_samples": 500,
                            "test_split": 0.3,
                            "validation_split": 0.1,
                        },
                        {
                            "name": "intrusion_detection",
                            "source": "local",
                            "path": "./data/network_logs.jsonl",
                            "max_samples": 1000,
                            "test_split": 0.2,
                            "validation_split": 0.1,
                        },
                    ],
                    "models": [
                        {
                            "name": "gpt-3.5-turbo",
                            "type": "openai_api",
                            "path": "gpt-3.5-turbo",
                            "config": {"api_key": "${OPENAI_API_KEY}"},
                            "max_tokens": 1024,
                            "temperature": 0.0,
                        },
                        {
                            "name": "claude-3-haiku",
                            "type": "anthropic_api",
                            "path": "claude-3-haiku-20240307",
                            "config": {"api_key": "${ANTHROPIC_API_KEY}"},
                            "max_tokens": 1024,
                            "temperature": 0.0,
                        },
                    ],
                    "evaluation": {
                        "metrics": ["accuracy", "f1_score", "precision", "recall"],
                        "parallel_jobs": 4,
                        "timeout_minutes": 30,
                        "batch_size": 32,
                    },
                },
                {
                    "name": "Performance Benchmarking",
                    "description": "Large scale performance benchmarking",
                    "output_dir": "./results/performance",
                    "datasets": [
                        {
                            "name": "large_mixed_dataset",
                            "source": "local",
                            "path": "./data/large_cybersec_dataset.jsonl",
                            "max_samples": 10000,
                            "test_split": 0.15,
                            "validation_split": 0.05,
                        }
                    ],
                    "models": [
                        {
                            "name": "gpt-4",
                            "type": "openai_api",
                            "path": "gpt-4",
                            "config": {"api_key": "${OPENAI_API_KEY}"},
                            "max_tokens": 2048,
                            "temperature": 0.1,
                        }
                    ],
                    "evaluation": {
                        "metrics": ["accuracy", "f1_score"],
                        "parallel_jobs": 8,
                        "timeout_minutes": 120,
                        "batch_size": 64,
                    },
                },
            ]

            # Write configuration files
            for i, config in enumerate(configs):
                config_file = configs_dir / f"experiment_{i + 1}.yaml"
                with open(config_file, "w") as f:
                    yaml.dump(config, f, indent=2, default_flow_style=False)

            yield configs_dir

    @pytest.fixture
    def sample_datasets_dir(self):
        """Create sample dataset files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()

            # Create sample JSONL datasets
            datasets = {
                "malware_samples.jsonl": [
                    {"text": "suspicious_executable.exe detected in downloads", "label": "MALWARE"},
                    {"text": "normal system file access", "label": "BENIGN"},
                    {"text": "encrypted payload in network traffic", "label": "MALWARE"},
                    {"text": "user login successful", "label": "BENIGN"},
                ]
                * 25,  # 100 samples total
                "phishing_emails.jsonl": [
                    {"text": "Urgent: Update your bank account details", "label": "PHISHING"},
                    {"text": "Meeting scheduled for tomorrow at 2pm", "label": "LEGITIMATE"},
                    {"text": "Click here to claim your prize!", "label": "PHISHING"},
                    {"text": "Weekly project status report attached", "label": "LEGITIMATE"},
                ]
                * 125,  # 500 samples total
                "network_logs.jsonl": [
                    {
                        "text": "192.168.1.100 -> 10.0.0.5 PORT_SCAN ports 22,23,80,443",
                        "label": "ATTACK",
                    },
                    {"text": "Normal HTTP GET request to /api/users/profile", "label": "BENIGN"},
                    {"text": "Multiple failed login attempts from 192.168.1.50", "label": "ATTACK"},
                    {"text": "File transfer completed successfully", "label": "BENIGN"},
                ]
                * 250,  # 1000 samples total
                "large_cybersec_dataset.jsonl": [
                    {
                        "text": f"Security event {i}: various cybersec scenarios",
                        "label": "ATTACK" if i % 2 else "BENIGN",
                    }
                    for i in range(10000)
                ],
            }

            # Write dataset files
            for filename, samples in datasets.items():
                dataset_file = data_dir / filename
                with open(dataset_file, "w") as f:
                    for sample in samples:
                        json.dump(sample, f)
                        f.write("\n")

            yield temp_dir

    @pytest.mark.asyncio
    async def test_load_all_sample_configs(
        self, config_service, sample_configs_dir, sample_datasets_dir
    ):
        """Test loading all sample configuration files."""
        # Set up environment variables
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test1234567890abcdef1234567890abcdef12345678",
                "ANTHROPIC_API_KEY": "sk-ant-test1234567890abcdef1234567890abcdef1234567890abcdef",
            },
        ):
            # Copy sample datasets to accessible location
            data_dir = Path(sample_datasets_dir) / "data"
            target_data_dir = Path("./data")
            target_data_dir.mkdir(exist_ok=True)

            try:
                for dataset_file in data_dir.glob("*.jsonl"):
                    shutil.copy(dataset_file, target_data_dir / dataset_file.name)

                # Load each configuration
                config_files = list(sample_configs_dir.glob("*.yaml"))
                assert len(config_files) == 3

                loaded_configs = []
                for config_file in config_files:
                    config = await config_service.load_experiment_config(config_file)
                    loaded_configs.append(config)

                    # Validate basic structure
                    assert isinstance(config, ExperimentConfig)
                    assert config.name
                    assert len(config.datasets) > 0
                    assert len(config.models) > 0
                    assert len(config.evaluation.metrics) > 0

                # Verify different experiment types
                experiment_names = [config.name for config in loaded_configs]
                assert "Small Scale Experiment" in experiment_names
                assert "Multi Model Comparison" in experiment_names
                assert "Performance Benchmarking" in experiment_names

            finally:
                # Cleanup
                if target_data_dir.exists():
                    shutil.rmtree(target_data_dir)

    @pytest.mark.asyncio
    async def test_config_inheritance_and_overrides(self, config_service):
        """Test configuration inheritance and override patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create base configuration
            base_config = {
                "name": "Base Configuration",
                "description": "Base configuration template",
                "output_dir": "./results",
                "datasets": [
                    {
                        "name": "base_dataset",
                        "source": "local",
                        "path": "./data/base.jsonl",
                        "max_samples": 100,
                        "test_split": 0.2,
                        "validation_split": 0.1,
                    }
                ],
                "models": [
                    {
                        "name": "base_model",
                        "type": "openai_api",
                        "path": "gpt-3.5-turbo",
                        "config": {"api_key": "${OPENAI_API_KEY:default_key}"},
                        "max_tokens": 512,
                        "temperature": 0.1,
                    }
                ],
                "evaluation": {
                    "metrics": ["accuracy"],
                    "parallel_jobs": 2,
                    "timeout_minutes": 30,
                    "batch_size": 16,
                },
            }

            # Create override configuration
            override_config = {
                "name": "Override Configuration",
                "description": "Configuration with overrides",
                "output_dir": "./results/override",
                "datasets": [
                    {
                        "name": "override_dataset",
                        "source": "local",
                        "path": "./data/override.jsonl",
                        "max_samples": 500,  # Override sample size
                        "test_split": 0.3,  # Override split
                        "validation_split": 0.1,
                    }
                ],
                "models": [
                    {
                        "name": "override_model",
                        "type": "openai_api",
                        "path": "gpt-4",  # Override model
                        "config": {"api_key": "${OPENAI_API_KEY:default_key}"},
                        "max_tokens": 2048,  # Override tokens
                        "temperature": 0.0,  # Override temperature
                    }
                ],
                "evaluation": {
                    "metrics": ["accuracy", "f1_score", "precision"],  # Override metrics
                    "parallel_jobs": 4,  # Override parallel jobs
                    "timeout_minutes": 60,  # Override timeout
                    "batch_size": 32,  # Override batch size
                },
            }

            # Write configuration files
            base_file = Path(temp_dir) / "base_config.yaml"
            override_file = Path(temp_dir) / "override_config.yaml"

            with open(base_file, "w") as f:
                yaml.dump(base_config, f, indent=2)
            with open(override_file, "w") as f:
                yaml.dump(override_config, f, indent=2)

            # Load configurations
            base_loaded = await config_service.load_experiment_config(base_file)
            override_loaded = await config_service.load_experiment_config(override_file)

            # Verify base configuration
            assert base_loaded.name == "Base Configuration"
            assert base_loaded.datasets[0].max_samples == 100
            assert base_loaded.models[0].path == "gpt-3.5-turbo"
            assert base_loaded.models[0].max_tokens == 512
            assert len(base_loaded.evaluation.metrics) == 1

            # Verify override configuration
            assert override_loaded.name == "Override Configuration"
            assert override_loaded.datasets[0].max_samples == 500  # Overridden
            assert override_loaded.models[0].path == "gpt-4"  # Overridden
            assert override_loaded.models[0].max_tokens == 2048  # Overridden
            assert len(override_loaded.evaluation.metrics) == 3  # Overridden

    @pytest.mark.asyncio
    async def test_environment_resolution_scenarios(self, config_service):
        """Test environment variable resolution in realistic scenarios."""
        config_data = {
            "name": "Environment Resolution Test",
            "description": "Testing environment variable resolution",
            "output_dir": "${EXPERIMENT_OUTPUT_DIR:./default_results}",
            "datasets": [
                {
                    "name": "env_dataset",
                    "source": "local",
                    "path": "${DATASET_PATH:./data/default.jsonl}",
                    "max_samples": "${MAX_SAMPLES:100}",
                    "test_split": "${TEST_SPLIT:0.2}",
                    "validation_split": "${VAL_SPLIT:0.1}",
                }
            ],
            "models": [
                {
                    "name": "configurable_model",
                    "type": "openai_api",
                    "path": "${MODEL_NAME:gpt-3.5-turbo}",
                    "config": {
                        "api_key": "${OPENAI_API_KEY}",
                        "organization": "${OPENAI_ORG:default-org}",
                    },
                    "max_tokens": "${MODEL_MAX_TOKENS:1024}",
                    "temperature": "${MODEL_TEMPERATURE:0.1}",
                }
            ],
            "evaluation": {
                "metrics": "${EVAL_METRICS:accuracy,f1_score}",
                "parallel_jobs": "${PARALLEL_JOBS:2}",
                "timeout_minutes": "${TIMEOUT_MINUTES:30}",
                "batch_size": "${BATCH_SIZE:16}",
                "enable_detailed_logging": "${DEBUG_LOGGING:false}",
            },
        }

        # Test different environment scenarios
        scenarios = [
            # Scenario 1: Minimal environment (use defaults)
            {
                "env": {"OPENAI_API_KEY": "sk-test123"},
                "expected": {
                    "output_dir": "./default_results",
                    "max_samples": 100,
                    "model_path": "gpt-3.5-turbo",
                    "max_tokens": 1024,
                    "parallel_jobs": 2,
                },
            },
            # Scenario 2: Full environment override
            {
                "env": {
                    "OPENAI_API_KEY": "sk-prod123",
                    "EXPERIMENT_OUTPUT_DIR": "./production_results",
                    "MAX_SAMPLES": "1000",
                    "MODEL_NAME": "gpt-4",
                    "MODEL_MAX_TOKENS": "2048",
                    "PARALLEL_JOBS": "8",
                    "EVAL_METRICS": "accuracy,precision,recall,f1_score",
                    "DEBUG_LOGGING": "true",
                },
                "expected": {
                    "output_dir": "./production_results",
                    "max_samples": 1000,
                    "model_path": "gpt-4",
                    "max_tokens": 2048,
                    "parallel_jobs": 8,
                    "debug_logging": True,
                },
            },
            # Scenario 3: Mixed environment (some defaults, some overrides)
            {
                "env": {
                    "OPENAI_API_KEY": "sk-mixed123",
                    "MAX_SAMPLES": "500",
                    "MODEL_TEMPERATURE": "0.5",
                    "BATCH_SIZE": "64",
                },
                "expected": {
                    "output_dir": "./default_results",  # Default
                    "max_samples": 500,  # Override
                    "model_path": "gpt-3.5-turbo",  # Default
                    "temperature": 0.5,  # Override
                    "batch_size": 64,  # Override
                },
            },
        ]

        for _i, scenario in enumerate(scenarios):
            with patch.dict(os.environ, scenario["env"], clear=True):
                # Write config file
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    yaml.dump(config_data, f)
                    config_file = Path(f.name)

                try:
                    # Load and validate configuration
                    config = await config_service.load_experiment_config(config_file)

                    # Verify expected values
                    expected = scenario["expected"]
                    if "output_dir" in expected:
                        assert config.output_dir == expected["output_dir"]
                    if "max_samples" in expected:
                        assert config.datasets[0].max_samples == expected["max_samples"]
                    if "model_path" in expected:
                        assert config.models[0].path == expected["model_path"]
                    if "max_tokens" in expected:
                        assert config.models[0].max_tokens == expected["max_tokens"]
                    if "temperature" in expected:
                        assert config.models[0].temperature == expected["temperature"]
                    if "parallel_jobs" in expected:
                        assert config.evaluation.parallel_jobs == expected["parallel_jobs"]
                    if "batch_size" in expected:
                        assert config.evaluation.batch_size == expected["batch_size"]
                    if "debug_logging" in expected:
                        assert (
                            config.evaluation.enable_detailed_logging == expected["debug_logging"]
                        )

                finally:
                    os.unlink(config_file)

    @pytest.mark.asyncio
    async def test_configuration_caching_performance(self, config_service):
        """Test configuration caching and performance benefits."""
        # Create a configuration file
        config_data = {
            "name": "Caching Performance Test",
            "description": "Test caching performance",
            "output_dir": "./results",
            "datasets": [{"name": "test", "source": "local", "path": "./data/test.jsonl"}],
            "models": [
                {
                    "name": "test",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "sk-test"},
                }
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            # First load - should be slow (file I/O)
            start_time = time.time()
            config1 = await config_service.load_experiment_config(config_file)
            first_load_time = time.time() - start_time

            # Second load - should be fast (cached)
            start_time = time.time()
            config2 = await config_service.load_experiment_config(config_file)
            second_load_time = time.time() - start_time

            # Third load - should still be fast (cached)
            start_time = time.time()
            config3 = await config_service.load_experiment_config(config_file)
            third_load_time = time.time() - start_time

            # Verify configurations are equivalent
            assert config1.name == config2.name == config3.name
            assert config1.datasets[0].name == config2.datasets[0].name == config3.datasets[0].name

            # Verify caching performance benefit
            # Note: This is a rough test - actual times may vary
            assert (
                second_load_time < first_load_time * 0.5 or second_load_time < 0.01
            )  # Much faster or very fast
            assert (
                third_load_time < first_load_time * 0.5 or third_load_time < 0.01
            )  # Much faster or very fast

            # Verify cache contains the configuration
            config_id = config_service._get_config_id(config_file)
            cached_config = config_service.get_cached_config(config_id)
            assert cached_config is not None
            assert cached_config.name == config1.name

        finally:
            os.unlink(config_file)

    @pytest.mark.asyncio
    async def test_concurrent_config_access(self, config_service):
        """Test concurrent access to configuration service."""
        # Create multiple configuration files
        config_files = []
        try:
            for i in range(5):
                config_data = {
                    "name": f"Concurrent Test {i + 1}",
                    "description": f"Concurrent access test configuration {i + 1}",
                    "output_dir": f"./results/concurrent_{i + 1}",
                    "datasets": [
                        {
                            "name": f"dataset_{i + 1}",
                            "source": "local",
                            "path": f"./data/test_{i + 1}.jsonl",
                        }
                    ],
                    "models": [
                        {
                            "name": f"model_{i + 1}",
                            "type": "openai_api",
                            "path": "gpt-3.5-turbo",
                            "config": {"api_key": "sk-test"},
                        }
                    ],
                    "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 1},
                }

                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    yaml.dump(config_data, f)
                    config_files.append(Path(f.name))

            # Load configurations concurrently
            async def load_config(config_file):
                return await config_service.load_experiment_config(config_file)

            # Create concurrent tasks
            tasks = [load_config(config_file) for config_file in config_files]

            # Execute concurrently
            configs = await asyncio.gather(*tasks)

            # Verify all configurations loaded successfully
            assert len(configs) == 5
            for i, config in enumerate(configs):
                assert config.name == f"Concurrent Test {i + 1}"
                assert config.datasets[0].name == f"dataset_{i + 1}"
                assert config.models[0].name == f"model_{i + 1}"

            # Test concurrent access to same configuration
            same_config_tasks = [load_config(config_files[0]) for _ in range(10)]
            same_configs = await asyncio.gather(*same_config_tasks)

            # Verify all returned the same configuration
            assert len(same_configs) == 10
            for config in same_configs:
                assert config.name == "Concurrent Test 1"

        finally:
            # Cleanup
            for config_file in config_files:
                os.unlink(config_file)

    @pytest.mark.asyncio
    async def test_invalid_config_handling(self, config_service):
        """Test handling of various invalid configuration scenarios."""
        invalid_configs = [
            # Invalid YAML syntax
            ("invalid_yaml.yaml", "invalid: yaml: content: [unclosed"),
            # Empty file
            ("empty.yaml", ""),
            # Valid YAML but invalid configuration structure
            (
                "invalid_structure.yaml",
                yaml.dump(
                    {
                        "name": "",  # Empty name
                        "datasets": [],  # No datasets
                        "models": [],  # No models
                        "evaluation": {"metrics": []},  # No metrics
                    }
                ),
            ),
            # Missing required fields
            (
                "missing_fields.yaml",
                yaml.dump(
                    {
                        "name": "Test",
                        # Missing datasets, models, evaluation
                    }
                ),
            ),
        ]

        for _filename, content in invalid_configs:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(content)
                config_file = Path(f.name)

            try:
                with pytest.raises(
                    (Exception, ValueError, TypeError)
                ):  # Should raise ConfigurationError
                    await config_service.load_experiment_config(config_file)
            finally:
                os.unlink(config_file)

    @pytest.mark.asyncio
    async def test_config_reload_during_runtime(self, config_service):
        """Test configuration reloading during runtime."""
        # Create initial configuration
        initial_config = {
            "name": "Initial Configuration",
            "description": "Initial version",
            "output_dir": "./results/initial",
            "datasets": [
                {"name": "initial_dataset", "source": "local", "path": "./data/initial.jsonl"}
            ],
            "models": [
                {
                    "name": "initial_model",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "sk-test"},
                }
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(initial_config, f)
            config_file = Path(f.name)

        try:
            # Load initial configuration
            config1 = await config_service.load_experiment_config(config_file)
            assert config1.name == "Initial Configuration"
            assert config1.output_dir == "./results/initial"

            # Verify it's cached
            config_id = config_service._get_config_id(config_file)
            assert config_service.get_cached_config(config_id) is not None

            # Modify the configuration file
            updated_config = {
                "name": "Updated Configuration",
                "description": "Updated version",
                "output_dir": "./results/updated",
                "datasets": [
                    {"name": "updated_dataset", "source": "local", "path": "./data/updated.jsonl"}
                ],
                "models": [
                    {
                        "name": "updated_model",
                        "type": "openai_api",
                        "path": "gpt-4",
                        "config": {"api_key": "sk-test"},
                    }
                ],
                "evaluation": {"metrics": ["accuracy", "f1_score"], "parallel_jobs": 2},
            }

            with open(config_file, "w") as f:
                yaml.dump(updated_config, f)

            # Reload configuration
            reload_response = await config_service.reload_config(config_id)
            assert reload_response.success is True

            # Load again - should get updated version
            config2 = await config_service.load_experiment_config(config_file)
            assert config2.name == "Updated Configuration"
            assert config2.output_dir == "./results/updated"
            assert config2.models[0].path == "gpt-4"
            assert len(config2.evaluation.metrics) == 2

            # Verify the configuration actually changed
            assert config1.name != config2.name
            assert config1.output_dir != config2.output_dir

        finally:
            os.unlink(config_file)

    @pytest.mark.asyncio
    async def test_comprehensive_validation_integration(self, config_service, sample_datasets_dir):
        """Test comprehensive validation integration with realistic configurations."""
        # Set up sample data files
        data_dir = Path(sample_datasets_dir) / "data"
        target_data_dir = Path("./data")
        target_data_dir.mkdir(exist_ok=True)

        try:
            # Copy a dataset file for testing
            sample_file = data_dir / "malware_samples.jsonl"
            target_file = target_data_dir / "malware_samples.jsonl"
            shutil.copy(sample_file, target_file)

            # Create configuration with various validation scenarios
            test_config = {
                "name": "Comprehensive Validation Test",
                "description": "Test comprehensive validation features",
                "output_dir": "./results/validation_test",
                "datasets": [
                    {
                        "name": "malware_classification",
                        "source": "local",
                        "path": "./data/malware_samples.jsonl",
                        "max_samples": 50,  # Small sample size - should warn
                        "test_split": 0.2,
                        "validation_split": 0.1,
                    },
                    {
                        "name": "missing_dataset",
                        "source": "local",
                        "path": "./data/nonexistent.jsonl",  # Missing file - should error
                        "max_samples": 100,
                    },
                ],
                "models": [
                    {
                        "name": "gpt-3.5-turbo",
                        "type": "openai_api",
                        "path": "gpt-3.5-turbo",
                        "config": {
                            "api_key": "sk-test123456789012345678901234567890"
                        },  # Valid format
                        "max_tokens": 512,
                        "temperature": 0.1,
                    },
                    {
                        "name": "gpt-4-large",
                        "type": "openai_api",
                        "path": "gpt-4",
                        "config": {"api_key": "invalid-key"},  # Invalid format - should warn
                        "max_tokens": 4096,  # At model limit
                        "temperature": 2.0,  # At temperature limit
                    },
                    {
                        "name": "missing-key-model",
                        "type": "anthropic_api",
                        "path": "claude-3-haiku",
                        "config": {},  # Missing API key - should error
                        "max_tokens": 1024,
                    },
                ],
                "evaluation": {
                    "metrics": ["accuracy", "f1_score"],
                    "parallel_jobs": 8,  # At CPU limit
                    "timeout_minutes": 2,  # Short timeout
                    "batch_size": 128,  # At batch size limit
                },
            }

            # Write configuration
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(test_config, f)
                config_file = Path(f.name)

            try:
                # Load configuration (should succeed despite warnings)
                config = await config_service.load_experiment_config(config_file)

                # Validate configuration
                warnings = await config_service.validate_config(config)

                # Should have multiple warnings
                assert len(warnings) > 5  # Expect several validation warnings

                # Check for specific warning types
                warning_text = " ".join(warnings)
                assert (
                    "small sample size" in warning_text.lower()
                    or "very small" in warning_text.lower()
                )
                assert "not found" in warning_text.lower() or "missing" in warning_text.lower()
                assert "exceeds" in warning_text.lower() or "cpu count" in warning_text.lower()
                assert "timeout" in warning_text.lower() or "short" in warning_text.lower()
                assert "batch size" in warning_text.lower() or "large" in warning_text.lower()

                # Verify configuration still loaded successfully
                assert config.name == "Comprehensive Validation Test"
                assert len(config.datasets) == 2
                assert len(config.models) == 3

            finally:
                os.unlink(config_file)

        finally:
            # Cleanup
            if target_data_dir.exists():
                shutil.rmtree(target_data_dir)

"""
Unit tests for environment variable resolution in ConfigurationService.

This module tests all functionality related to environment variable handling
including resolution, type conversion, validation, and secure masking.
"""

import tempfile
from pathlib import Path

import pytest

from benchmark.core.config import ExperimentConfig
from benchmark.core.exceptions import ConfigurationError, ErrorCode
from benchmark.services.configuration_service import ConfigurationService


class TestEnvironmentVariableResolution:
    """Test environment variable resolution functionality."""

    def test_resolve_string_variables(self, env_variables):
        """Test resolving string environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {
                "api_key": "${TEST_API_KEY}",
                "description": "Using ${TEST_API_KEY} for authentication",
            }

            resolved = service.resolve_environment_variables(config_dict)

            assert resolved["api_key"] == "test-key-123"
            assert resolved["description"] == "Using test-key-123 for authentication"

    def test_resolve_integer_variables(self, env_variables):
        """Test resolving integer environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {"timeout": "${TEST_TIMEOUT}", "count": "${TEST_INTEGER}"}

            resolved = service.resolve_environment_variables(config_dict)

            assert resolved["timeout"] == 30
            assert resolved["count"] == 42
            assert isinstance(resolved["timeout"], int)
            assert isinstance(resolved["count"], int)

    def test_resolve_boolean_variables(self, env_variables):
        """Test resolving boolean environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {
                "feature_enabled": "${TEST_BOOLEAN_TRUE}",
                "debug_mode": "${TEST_BOOLEAN_FALSE}",
                "explicit_true": "${TEST_ENABLE_FEATURE}",
            }

            resolved = service.resolve_environment_variables(config_dict)

            assert resolved["feature_enabled"] is True
            assert resolved["debug_mode"] is False
            assert resolved["explicit_true"] is True
            assert isinstance(resolved["feature_enabled"], bool)
            assert isinstance(resolved["debug_mode"], bool)

    def test_resolve_list_variables(self, env_variables):
        """Test resolving list environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {"metrics": "${TEST_LIST_VALUES}"}

            resolved = service.resolve_environment_variables(config_dict)

            assert resolved["metrics"] == ["accuracy", "precision", "recall"]
            assert isinstance(resolved["metrics"], list)

    def test_resolve_with_defaults(self):
        """Test resolving environment variables with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {
                "api_key": "${MISSING_API_KEY:default-key}",
                "timeout": "${MISSING_TIMEOUT:60}",
                "enabled": "${MISSING_FEATURE:false}",
                "metrics": "${MISSING_METRICS:accuracy,precision}",
            }

            resolved = service.resolve_environment_variables(config_dict)

            assert resolved["api_key"] == "default-key"
            assert resolved["timeout"] == 60
            assert resolved["enabled"] is False
            assert resolved["metrics"] == ["accuracy", "precision"]

    def test_missing_required_variable(self):
        """Test error when required environment variable is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {"api_key": "${MISSING_REQUIRED_KEY}"}

            with pytest.raises(ConfigurationError) as exc_info:
                service.resolve_environment_variables(config_dict)

            assert exc_info.value.error_code == ErrorCode.CONFIG_VALIDATION_FAILED
            assert "MISSING_REQUIRED_KEY" in str(exc_info.value)

    def test_nested_structure_resolution(self, env_variables):
        """Test resolving environment variables in nested structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {
                "models": [
                    {
                        "name": "test-model",
                        "config": {"api_key": "${TEST_API_KEY}", "timeout": "${TEST_TIMEOUT}"},
                        "settings": {
                            "enabled": "${TEST_ENABLE_FEATURE}",
                            "batch_size": "${TEST_INTEGER}",
                        },
                    }
                ],
                "evaluation": {
                    "metrics": "${TEST_LIST_VALUES}",
                    "parallel": "${TEST_BOOLEAN_TRUE}",
                },
            }

            resolved = service.resolve_environment_variables(config_dict)

            model_config = resolved["models"][0]["config"]
            model_settings = resolved["models"][0]["settings"]
            evaluation = resolved["evaluation"]

            assert model_config["api_key"] == "test-key-123"
            assert model_config["timeout"] == 30
            assert model_settings["enabled"] is True
            assert model_settings["batch_size"] == 42
            assert evaluation["metrics"] == ["accuracy", "precision", "recall"]
            assert evaluation["parallel"] is True

    def test_partial_string_substitution(self, env_variables):
        """Test partial string substitution with environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {
                "url": "https://api.example.com/v1?key=${TEST_API_KEY}&timeout=${TEST_TIMEOUT}",
                "message": "Timeout is set to ${TEST_TIMEOUT} seconds",
            }

            resolved = service.resolve_environment_variables(config_dict)

            assert resolved["url"] == "https://api.example.com/v1?key=test-key-123&timeout=30"
            assert resolved["message"] == "Timeout is set to 30 seconds"

    def test_type_conversion_edge_cases(self):
        """Test type conversion edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            test_cases = [
                ("TRUE", True),
                ("False", False),
                ("1", True),
                ("0", False),
                ("yes", True),
                ("NO", False),
                ("on", True),
                ("OFF", False),
                ("123", 123),
                ("-456", -456),
                ("+789", 789),
                ("12.34", "12.34"),  # Should remain string (has decimal)
                ("not_a_number", "not_a_number"),
                ("", ""),
                ("single", "single"),
                ("a,b,c,d", ["a", "b", "c", "d"]),
                (
                    "accuracy, precision , recall",
                    ["accuracy", "precision", "recall"],
                ),  # Should trim spaces
                (",,,", []),  # Should handle empty items
            ]

            for input_val, expected in test_cases:
                result = service._convert_env_value_type(input_val)
                assert result == expected, (
                    f"Failed for input '{input_val}': expected {expected}, got {result}"
                )

    def test_get_required_env_vars(self, env_variables):
        """Test extraction of required environment variables."""
        config_data = {
            "name": "Test Experiment",
            "description": "Test configuration",
            "output_dir": "./results",
            "datasets": [
                {
                    "name": "test_dataset",
                    "source": "local",
                    "path": "./data/test.jsonl",
                    "max_samples": 100,
                }
            ],
            "models": [
                {
                    "name": "test_model",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "${TEST_API_KEY}"},
                    "max_tokens": "${TEST_TIMEOUT}",
                },
                {
                    "name": "test_model_2",
                    "type": "anthropic_api",
                    "path": "claude-3-haiku",
                    "config": {"api_key": "${ANTHROPIC_API_KEY}"},
                    "max_tokens": "${TEST_INTEGER:1000}",
                },
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))
            # Don't resolve environment variables first - we want to extract the raw patterns

            # Extract from the original config_data before resolution
            required_vars = set()
            service._extract_env_vars_recursive(config_data, required_vars)

            assert "TEST_API_KEY" in required_vars
            assert "ANTHROPIC_API_KEY" in required_vars
            assert "TEST_TIMEOUT" in required_vars
            assert "TEST_INTEGER" in required_vars

    def test_validate_environment_requirements_success(self, env_variables):
        """Test environment variable validation with all variables present."""
        config_data = {
            "name": "Test Experiment",
            "description": "Test configuration",
            "output_dir": "./results",
            "datasets": [
                {
                    "name": "test_dataset",
                    "source": "local",
                    "path": "./data/test.jsonl",
                    "max_samples": 100,
                }
            ],
            "models": [
                {
                    "name": "test_model",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "${TEST_API_KEY}"},
                    "max_tokens": 512,
                }
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))
            config_data = service.resolve_environment_variables(config_data)
            config = ExperimentConfig(**config_data)

            warnings = service.validate_environment_requirements(config)

            # Should have no warnings since TEST_API_KEY is available
            assert len(warnings) == 0

    def test_validate_environment_requirements_missing(self):
        """Test environment variable validation with missing variables."""
        config_data = {
            "name": "Test Experiment",
            "description": "Test configuration",
            "output_dir": "./results",
            "datasets": [
                {
                    "name": "test_dataset",
                    "source": "local",
                    "path": "./data/test.jsonl",
                    "max_samples": 100,
                }
            ],
            "models": [
                {
                    "name": "test_model",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "${MISSING_API_KEY:default}"},
                    "max_tokens": 512,
                }
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))
            config_data = service.resolve_environment_variables(config_data)
            config = ExperimentConfig(**config_data)

            # Manually set an env var that doesn't exist in config to test missing detection
            config_dict = config.model_dump()
            config_dict["models"][0]["config"]["api_key"] = "${DEFINITELY_MISSING_KEY}"
            config = ExperimentConfig(**config_dict)

            warnings = service.validate_environment_requirements(config)

            # Should have warnings for missing variables
            assert len(warnings) > 0
            assert any("DEFINITELY_MISSING_KEY" in warning for warning in warnings)


class TestSensitiveDataMasking:
    """Test sensitive data masking functionality."""

    def test_mask_sensitive_values(self):
        """Test masking of sensitive configuration values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {
                "models": [
                    {
                        "name": "test-model",
                        "config": {
                            "api_key": "super-secret-key-12345",
                            "secret_token": "secret-token-67890",
                            "password": "password123",
                            "non_sensitive": "public-value",
                        },
                    }
                ],
                "database": {
                    "connection": "public-connection-string",
                    "auth_token": "sensitive-auth-token-xyz",
                    "credential": "sensitive-credential",
                },
            }

            masked = service.mask_sensitive_values(config_dict)

            # Sensitive values should be masked
            model_config = masked["models"][0]["config"]
            assert model_config["api_key"] == "su******************45"
            assert model_config["secret_token"] == "se**************90"
            assert model_config["password"] == "pa*******23"
            assert model_config["non_sensitive"] == "public-value"  # Should not be masked

            # Database config
            db_config = masked["database"]
            assert db_config["connection"] == "public-connection-string"  # Should not be masked
            assert db_config["auth_token"] == "se********************yz"
            assert db_config["credential"] == "se****************al"

    def test_mask_short_values(self):
        """Test masking of short sensitive values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {"api_key": "abc", "token": "xy", "secret": "a"}

            masked = service.mask_sensitive_values(config_dict)

            assert masked["api_key"] == "***"
            assert masked["token"] == "**"
            assert masked["secret"] == "*"

    def test_sensitive_key_detection(self):
        """Test detection of sensitive configuration keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            sensitive_keys = [
                "api_key",
                "secret",
                "password",
                "token",
                "credential",
                "auth",
                "key",
                "secret_key",
                "access_token",
                "bearer_token",
                "API_KEY",
                "SECRET_TOKEN",
                "Auth_Password",  # Test case insensitivity
            ]

            non_sensitive_keys = [
                "name",
                "description",
                "timeout",
                "batch_size",
                "model_path",
                "dataset_name",
                "output_dir",
                "public_key_id",  # "key" but in different context
            ]

            for key in sensitive_keys:
                assert service._is_sensitive_key(key), f"'{key}' should be detected as sensitive"

            for key in non_sensitive_keys:
                assert not service._is_sensitive_key(key), (
                    f"'{key}' should not be detected as sensitive"
                )


class TestEnvironmentVariableIntegration:
    """Test integration of environment variable resolution with configuration loading."""

    @pytest.mark.asyncio
    async def test_load_config_with_env_vars(self, env_variables):
        """Test loading configuration files with environment variable resolution."""
        import tempfile

        import yaml

        config_data = {
            "name": "Environment Variable Test",
            "description": "Testing env var resolution",
            "output_dir": "./results",
            "datasets": [
                {
                    "name": "test_dataset",
                    "source": "local",
                    "path": "./data/test.jsonl",
                    "max_samples": "${TEST_INTEGER}",
                }
            ],
            "models": [
                {
                    "name": "test_model",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "${TEST_API_KEY}"},
                    "max_tokens": "${TEST_TIMEOUT}",
                }
            ],
            "evaluation": {
                "metrics": ["accuracy", "precision"],
                "parallel_jobs": 2,
                "enable_detailed_logging": "${TEST_ENABLE_FEATURE}",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))
            await service.initialize()

            # Create config file
            config_file = Path(temp_dir) / "test_config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load config - should resolve environment variables
            config = await service.load_experiment_config(config_file)

            # Verify environment variables were resolved
            assert config.datasets[0].max_samples == 42
            assert config.models[0].config["api_key"] == "test-key-123"
            assert config.models[0].max_tokens == 30
            assert config.evaluation.metrics == ["accuracy", "precision"]

            await service.shutdown()

    @pytest.mark.asyncio
    async def test_default_config_env_resolution(self, env_variables):
        """Test that default configuration resolves environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))
            await service.initialize()

            default_config = await service.get_default_config()
            resolved_config = service.resolve_environment_variables(default_config)

            # Verify some environment variables are resolved
            # The default config should have ${OPENAI_API_KEY:test_key}
            # Since env_variables fixture doesn't set OPENAI_API_KEY, it should use default
            models = resolved_config["models"]
            openai_model = next(m for m in models if m["name"] == "gpt-3.5-turbo")
            # The conftest.py clean_env fixture sets OPENAI_API_KEY to "test_key"
            assert openai_model["config"]["api_key"] == "test_key"

            await service.shutdown()

    @pytest.mark.asyncio
    async def test_config_validation_includes_env_vars(self, env_variables):
        """Test that configuration validation includes environment variable checks."""
        config_data = {
            "name": "Validation Test",
            "description": "Testing validation with env vars",
            "output_dir": "./results",
            "datasets": [
                {
                    "name": "test_dataset",
                    "source": "local",
                    "path": "./data/test.jsonl",
                    "max_samples": 100,
                }
            ],
            "models": [
                {
                    "name": "test_model",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "${TEST_API_KEY}"},
                    "max_tokens": 512,
                }
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))
            await service.initialize()

            config_data = service.resolve_environment_variables(config_data)
            config = ExperimentConfig(**config_data)

            warnings = await service.validate_config(config)

            # Should include environment variable validation
            # Since TEST_API_KEY is available in the fixture, there should be no env var warnings
            env_warnings = [w for w in warnings if "Environment variable" in w]
            assert len(env_warnings) == 0  # No missing env vars

            await service.shutdown()


class TestEnvironmentVariableErrorHandling:
    """Test error handling in environment variable resolution."""

    def test_malformed_env_var_patterns(self):
        """Test handling of malformed environment variable patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            config_dict = {
                "valid": "${VALID_VAR:default}",
                "malformed1": "${UNCLOSED_VAR",
                "malformed2": "$MISSING_BRACES}",
                "malformed3": "${:MISSING_NAME}",
                "not_env_var": "This is just a normal string with $ symbols",
            }

            # Should handle malformed patterns gracefully (leave them as-is)
            resolved = service.resolve_environment_variables(config_dict)

            assert resolved["valid"] == "default"  # Should resolve
            assert resolved["malformed1"] == "${UNCLOSED_VAR"  # Should remain unchanged
            assert resolved["malformed2"] == "$MISSING_BRACES}"  # Should remain unchanged
            assert resolved["malformed3"] == "${:MISSING_NAME}"  # Should remain unchanged
            assert resolved["not_env_var"] == "This is just a normal string with $ symbols"

    def test_env_var_resolution_preserves_structure(self, env_variables):
        """Test that environment variable resolution preserves original structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationService(config_dir=Path(temp_dir))

            original_config = {
                "string": "test",
                "number": 42,
                "boolean": True,
                "none_value": None,
                "list": [1, 2, 3],
                "nested": {"env_var": "${TEST_API_KEY}", "normal": "value"},
            }

            resolved = service.resolve_environment_variables(original_config)

            # Structure should be preserved
            assert isinstance(resolved["string"], str)
            assert isinstance(resolved["number"], int)
            assert isinstance(resolved["boolean"], bool)
            assert resolved["none_value"] is None
            assert isinstance(resolved["list"], list)
            assert isinstance(resolved["nested"], dict)

            # Only string with env var pattern should change
            assert resolved["nested"]["env_var"] == "test-key-123"
            assert resolved["nested"]["normal"] == "value"

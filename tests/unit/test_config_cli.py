"""
Unit tests for CLI configuration commands.

This module tests all CLI commands for configuration management including
validation, generation, display, and environment variable checking.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from click.testing import CliRunner

from benchmark.cli.config_commands import config
from benchmark.core.config import DatasetConfig, EvaluationConfig, ExperimentConfig, ModelConfig


class TestConfigValidateCommand:
    """Test the config validate command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            "name": "Test Configuration",
            "description": "Test configuration for CLI",
            "output_dir": "./test_results",
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
                    "config": {"api_key": "sk-test123456789012345678901234567890"},
                    "max_tokens": 512,
                }
            ],
            "evaluation": {
                "metrics": ["accuracy"],
                "parallel_jobs": 2,
                "timeout_minutes": 30,
                "batch_size": 16,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        os.unlink(f.name)

    @pytest.fixture
    def invalid_config_file(self):
        """Create a temporary invalid config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            yield f.name

        os.unlink(f.name)

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_validate_command_success(self, mock_service_class, runner, sample_config_file):
        """Test successful configuration validation."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock successful config loading
        mock_config = ExperimentConfig(
            name="Test Configuration",
            description="Test",
            output_dir="./test_results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(
                    name="test",
                    type="openai_api",
                    path="gpt-3.5-turbo",
                    config={"api_key": "sk-test"},
                )
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config

        # Mock validation with no warnings
        mock_service.validate_config.return_value = []

        # Run command
        result = runner.invoke(config, ["validate", sample_config_file])

        # Assertions
        assert result.exit_code == 0
        assert "Configuration validation passed!" in result.output
        mock_service.initialize.assert_called_once()
        mock_service.shutdown.assert_called_once()

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_validate_command_with_warnings(self, mock_service_class, runner, sample_config_file):
        """Test validation with warnings."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock config loading
        mock_config = ExperimentConfig(
            name="Test Configuration",
            description="Test",
            output_dir="./test_results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(
                    name="test",
                    type="openai_api",
                    path="gpt-3.5-turbo",
                    config={"api_key": "sk-test"},
                )
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config

        # Mock validation with warnings
        mock_service.validate_config.return_value = [
            "[WARNING] Batch size (16) may be too small for optimal performance",
            "[INFO] Consider increasing batch_size to 32 for better throughput",
        ]

        # Run command
        result = runner.invoke(config, ["validate", sample_config_file])

        # Assertions
        assert result.exit_code == 0
        assert "Validation Summary:" in result.output
        assert "Warnings:" in result.output
        assert "Recommendations:" in result.output

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_validate_command_with_errors(self, mock_service_class, runner, sample_config_file):
        """Test validation with errors."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock config loading
        mock_config = ExperimentConfig(
            name="Test Configuration",
            description="Test",
            output_dir="./test_results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(
                    name="test",
                    type="openai_api",
                    path="gpt-3.5-turbo",
                    config={"api_key": "sk-test"},
                )
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config

        # Mock validation with errors
        mock_service.validate_config.return_value = [
            "[ERROR] Dataset file not found: ./data/test.jsonl",
            "[ERROR] Model test_model missing API key",
        ]

        # Run command
        result = runner.invoke(config, ["validate", sample_config_file])

        # Assertions
        assert result.exit_code == 1
        assert "Errors (2):" in result.output
        assert "Configuration has 2 issue(s) that need attention" in result.output

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_validate_command_json_output(self, mock_service_class, runner, sample_config_file):
        """Test validation with JSON output format."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock config loading
        mock_config = ExperimentConfig(
            name="Test Configuration",
            description="Test",
            output_dir="./test_results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(
                    name="test",
                    type="openai_api",
                    path="gpt-3.5-turbo",
                    config={"api_key": "sk-test"},
                )
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config
        mock_service.validate_config.return_value = ["[WARNING] Test warning"]

        # Run command
        result = runner.invoke(config, ["validate", sample_config_file, "--json-output"])

        # Assertions
        assert result.exit_code == 0

        # Parse JSON output
        output_data = json.loads(result.output)
        assert output_data["success"] is True
        assert output_data["config_name"] == "Test Configuration"
        assert output_data["total_warnings"] == 1
        assert len(output_data["warnings"]) == 1

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_validate_command_quiet_mode(self, mock_service_class, runner, sample_config_file):
        """Test validation in quiet mode."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock config loading
        mock_config = ExperimentConfig(
            name="Test Configuration",
            description="Test",
            output_dir="./test_results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(
                    name="test",
                    type="openai_api",
                    path="gpt-3.5-turbo",
                    config={"api_key": "sk-test"},
                )
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config
        mock_service.validate_config.return_value = [
            "[INFO] Consider increasing batch_size to 32 for better throughput"
        ]

        # Run command
        result = runner.invoke(config, ["validate", sample_config_file, "--quiet"])

        # Assertions
        assert result.exit_code == 0
        assert "Recommendations" not in result.output  # Info messages should be hidden

    def test_validate_command_file_not_found(self, runner):
        """Test validation with non-existent file."""
        result = runner.invoke(config, ["validate", "/nonexistent/config.yaml"])

        assert result.exit_code == 2  # Click error code for missing file
        assert "does not exist" in result.output.lower()

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_validate_command_service_error(self, mock_service_class, runner, sample_config_file):
        """Test validation when service throws error."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock service initialization failure
        mock_service.initialize.side_effect = Exception("Service initialization failed")

        # Run command
        result = runner.invoke(config, ["validate", sample_config_file])

        # Assertions
        assert result.exit_code == 1
        assert "Unexpected error" in result.output


class TestConfigGenerateCommand:
    """Test the config generate command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_generate_command_default(self, mock_service_class, runner):
        """Test default configuration generation."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock default config
        default_config = {
            "name": "Default Configuration",
            "description": "Default configuration template",
            "output_dir": "./results",
            "datasets": [{"name": "sample", "source": "local", "path": "./data/sample.jsonl"}],
            "models": [{"name": "gpt-3.5", "type": "openai_api", "path": "gpt-3.5-turbo"}],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2},
        }
        mock_service.get_default_config.return_value = default_config

        # Run command
        with runner.isolated_filesystem():
            result = runner.invoke(config, ["generate"])

            # Assertions
            assert result.exit_code == 0
            assert "Configuration generated successfully!" in result.output
            assert Path("config.yaml").exists()

            # Check file content
            with open("config.yaml") as f:
                generated_config = yaml.safe_load(f)
            assert generated_config["name"] == "Default Configuration"

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_generate_command_custom_output(self, mock_service_class, runner):
        """Test generation with custom output file."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        mock_service.get_default_config.return_value = {"name": "Test"}

        # Run command
        with runner.isolated_filesystem():
            result = runner.invoke(config, ["generate", "--output", "custom.yaml"])

            assert result.exit_code == 0
            assert Path("custom.yaml").exists()

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_generate_command_json_format(self, mock_service_class, runner):
        """Test generation in JSON format."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        default_config = {"name": "Test", "description": "Test config"}
        mock_service.get_default_config.return_value = default_config

        # Run command
        with runner.isolated_filesystem():
            result = runner.invoke(
                config, ["generate", "--format", "json", "--output", "test.json"]
            )

            assert result.exit_code == 0
            assert Path("test.json").exists()

            # Check JSON content
            with open("test.json") as f:
                generated_config = json.load(f)
            assert generated_config["name"] == "Test"

    @patch("benchmark.cli.config_commands.ConfigurationService")
    @patch("benchmark.cli.config_commands.Prompt")
    @patch("benchmark.cli.config_commands.Confirm")
    def test_generate_command_interactive_mode(
        self, mock_confirm, mock_prompt, mock_service_class, runner
    ):
        """Test interactive generation mode."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        default_config = {
            "name": "Default",
            "description": "Default desc",
            "datasets": [{"name": "default"}],
            "models": [],
            "evaluation": {"parallel_jobs": 2, "batch_size": 16, "timeout_minutes": 30},
        }
        mock_service.get_default_config.return_value = default_config

        # Mock interactive inputs
        mock_prompt.ask.side_effect = [
            "Custom Experiment",  # name
            "Custom description",  # description
            "custom-model",  # openai model name
            "gpt-4",  # model path
            "2048",  # max tokens
            "0.2",  # temperature
            "./data/custom.jsonl",  # dataset path
            "500",  # max samples
            "4",  # parallel jobs
            "32",  # batch size
            "60",  # timeout
        ]
        mock_confirm.ask.side_effect = [True, False]  # use_openai=True, use_anthropic=False

        # Run command
        with runner.isolated_filesystem():
            result = runner.invoke(config, ["generate", "--interactive"])

            assert result.exit_code == 0
            assert "Interactive Configuration Generator" in result.output
            assert Path("config.yaml").exists()


class TestConfigShowCommand:
    """Test the config show command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def sample_config_file(self):
        """Create a sample config file."""
        config_data = {
            "name": "Display Test",
            "description": "Test config for display",
            "output_dir": "./results",
            "datasets": [{"name": "test_dataset", "source": "local", "path": "./data/test.jsonl"}],
            "models": [{"name": "test_model", "type": "openai_api", "path": "gpt-3.5-turbo"}],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        os.unlink(f.name)

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_show_command_default(self, mock_service_class, runner, sample_config_file):
        """Test default show command."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock loaded config
        mock_config = ExperimentConfig(
            name="Display Test",
            description="Test config for display",
            output_dir="./results",
            datasets=[DatasetConfig(name="test_dataset", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(name="test_model", type="openai_api", path="gpt-3.5-turbo", config={})
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config

        # Run command
        result = runner.invoke(config, ["show", sample_config_file])

        # Assertions
        assert result.exit_code == 0
        assert "Configuration:" in result.output
        assert "Display Test" in result.output
        assert "Configuration Content" in result.output

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_show_command_json_format(self, mock_service_class, runner, sample_config_file):
        """Test show command with JSON format."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_config = ExperimentConfig(
            name="Display Test",
            description="Test",
            output_dir="./results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[ModelConfig(name="test", type="openai_api", path="gpt-3.5-turbo", config={})],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config

        # Run command
        result = runner.invoke(config, ["show", sample_config_file, "--format", "json"])

        assert result.exit_code == 0
        assert "JSON" in result.output

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_show_command_no_pretty(self, mock_service_class, runner, sample_config_file):
        """Test show command without pretty printing."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_config = ExperimentConfig(
            name="Display Test",
            description="Test",
            output_dir="./results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[ModelConfig(name="test", type="openai_api", path="gpt-3.5-turbo", config={})],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config

        # Run command
        result = runner.invoke(config, ["show", sample_config_file, "--no-pretty"])

        assert result.exit_code == 0
        # Should have plain YAML output without rich formatting
        assert "Configuration:" not in result.output  # No rich headers


class TestConfigCheckEnvCommand:
    """Test the config check-env command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def config_with_env_vars(self):
        """Create config file with environment variables."""
        config_data = {
            "name": "Env Test",
            "description": "Test config with env vars",
            "datasets": [{"name": "test", "source": "local", "path": "./data/test.jsonl"}],
            "models": [
                {
                    "name": "openai_model",
                    "type": "openai_api",
                    "path": "gpt-3.5-turbo",
                    "config": {"api_key": "${OPENAI_API_KEY}"},
                }
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        os.unlink(f.name)

    @patch("benchmark.cli.config_commands.ConfigurationService")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"})
    def test_check_env_command_all_set(self, mock_service_class, runner, config_with_env_vars):
        """Test check-env when all variables are set."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock config loading
        mock_config = ExperimentConfig(
            name="Env Test",
            description="Test",
            output_dir="./results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(
                    name="openai_model",
                    type="openai_api",
                    path="gpt-3.5-turbo",
                    config={"api_key": "sk-test123456789012345678901234567890"},
                )
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config
        mock_service.get_required_env_vars.return_value = {"OPENAI_API_KEY"}

        # Run command
        result = runner.invoke(config, ["check-env", config_with_env_vars])

        assert result.exit_code == 0
        assert "All required environment variables are set!" in result.output
        assert "Set Variables" in result.output
        assert "OPENAI_API_KEY" in result.output

    @patch("benchmark.cli.config_commands.ConfigurationService")
    @patch.dict(os.environ, {}, clear=True)
    def test_check_env_command_missing_vars(self, mock_service_class, runner, config_with_env_vars):
        """Test check-env when variables are missing."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_config = ExperimentConfig(
            name="Env Test",
            description="Test",
            output_dir="./results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(name="openai_model", type="openai_api", path="gpt-3.5-turbo", config={})
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config
        mock_service.get_required_env_vars.return_value = {"OPENAI_API_KEY"}

        # Run command
        result = runner.invoke(config, ["check-env", config_with_env_vars])

        assert result.exit_code == 1
        assert "Missing Variables" in result.output
        assert "OPENAI_API_KEY" in result.output
        assert "export OPENAI_API_KEY=your_value_here" in result.output

    @patch("benchmark.cli.config_commands.ConfigurationService")
    def test_check_env_command_no_env_vars(self, mock_service_class, runner, config_with_env_vars):
        """Test check-env when no environment variables are required."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_config = ExperimentConfig(
            name="Env Test",
            description="Test",
            output_dir="./results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(name="test_model", type="openai_api", path="gpt-3.5-turbo", config={})
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config
        mock_service.get_required_env_vars.return_value = set()

        # Run command
        result = runner.invoke(config, ["check-env", config_with_env_vars])

        assert result.exit_code == 0
        assert "No environment variables required" in result.output

    @patch("benchmark.cli.config_commands.ConfigurationService")
    @patch("benchmark.cli.config_commands.Confirm")
    @patch("benchmark.cli.config_commands.Prompt")
    @patch.dict(os.environ, {}, clear=True)
    def test_check_env_command_interactive_set(
        self, mock_prompt, mock_confirm, mock_service_class, runner, config_with_env_vars
    ):
        """Test interactive environment variable setting."""
        # Mock service
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_config = ExperimentConfig(
            name="Env Test",
            description="Test",
            output_dir="./results",
            datasets=[DatasetConfig(name="test", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(name="openai_model", type="openai_api", path="gpt-3.5-turbo", config={})
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )
        mock_service.load_experiment_config.return_value = mock_config
        mock_service.get_required_env_vars.return_value = {"OPENAI_API_KEY"}

        # Mock interactive inputs
        mock_confirm.ask.return_value = True  # Set OPENAI_API_KEY
        mock_prompt.ask.return_value = "sk-test123456789012345678901234567890"

        # Run command
        result = runner.invoke(config, ["check-env", config_with_env_vars, "--set-missing"])

        assert result.exit_code == 0
        assert "Interactive Setup" in result.output
        assert "OPENAI_API_KEY set for this session" in result.output


class TestConfigCommandsIntegration:
    """Integration tests for all config commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_config_command_group(self, runner):
        """Test that the config command group works."""
        result = runner.invoke(config, ["--help"])

        assert result.exit_code == 0
        assert "Configuration management commands" in result.output
        assert "validate" in result.output
        assert "generate" in result.output
        assert "show" in result.output
        assert "check-env" in result.output

    def test_config_validate_help(self, runner):
        """Test validate command help."""
        result = runner.invoke(config, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate configuration file" in result.output
        assert "--quiet" in result.output
        assert "--json-output" in result.output

    def test_config_generate_help(self, runner):
        """Test generate command help."""
        result = runner.invoke(config, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate sample configuration" in result.output
        assert "--interactive" in result.output
        assert "--format" in result.output

    def test_config_show_help(self, runner):
        """Test show command help."""
        result = runner.invoke(config, ["show", "--help"])

        assert result.exit_code == 0
        assert "Display parsed configuration" in result.output
        assert "--format" in result.output
        assert "--pretty" in result.output

    def test_config_check_env_help(self, runner):
        """Test check-env command help."""
        result = runner.invoke(config, ["check-env", "--help"])

        assert result.exit_code == 0
        assert "Check environment variable requirements" in result.output
        assert "--set-missing" in result.output

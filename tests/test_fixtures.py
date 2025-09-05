"""
Tests for pytest fixtures to ensure they work correctly.

This module validates that all fixtures provide the expected functionality
and maintain proper state isolation between tests.
"""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from benchmark.core.config import DatasetConfig, EvaluationConfig, ExperimentConfig, ModelConfig
from benchmark.core.database_manager import DatabaseManager


class TestDatabaseFixtures:
    """Test database-related fixtures."""

    @pytest.mark.asyncio
    async def test_db_session_fixture_creates_clean_state(self, db_session: AsyncSession):
        """Test that db_session fixture provides clean database state."""
        # Verify session is valid
        assert db_session is not None
        assert hasattr(db_session, "execute")
        assert hasattr(db_session, "commit")
        assert hasattr(db_session, "rollback")

        # Test that we can execute queries
        result = await db_session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_db_manager_fixture_provides_initialized_manager(
        self, db_manager: DatabaseManager
    ):
        """Test that db_manager fixture provides initialized database manager."""
        # Verify manager is initialized
        assert db_manager is not None
        assert db_manager._initialized is True
        assert db_manager._engine is not None

        # Test that we can create a session
        async with db_manager.session_scope() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_database_fixtures_provide_isolated_state(self, db_session: AsyncSession):
        """Test that database fixtures provide isolated state between tests."""
        from benchmark.core.database import Experiment

        # Add test data
        experiment = Experiment(
            name="test-isolation", config={"test": True}, output_dir="/tmp/test"
        )
        db_session.add(experiment)
        await db_session.commit()

        # Verify data exists
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM experiments WHERE name = 'test-isolation'")
        )
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_database_isolation_between_tests(self, db_session: AsyncSession):
        """Test that previous test data doesn't leak to this test."""
        # This test should not see data from the previous test
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM experiments WHERE name = 'test-isolation'")
        )
        assert result.scalar() == 0


class TestConfigurationFixtures:
    """Test configuration-related fixtures."""

    def test_sample_dataset_config_fixture(self, sample_dataset_config: DatasetConfig):
        """Test sample_dataset_config fixture provides valid configuration."""
        assert isinstance(sample_dataset_config, DatasetConfig)
        assert sample_dataset_config.name == "test_dataset"
        assert sample_dataset_config.source == "local"
        assert sample_dataset_config.path == "/tmp/test_dataset"
        assert sample_dataset_config.max_samples == 100
        assert sample_dataset_config.test_split == 0.2
        assert sample_dataset_config.validation_split == 0.1
        assert "tokenize" in sample_dataset_config.preprocessing

    def test_sample_model_config_fixture(self, sample_model_config: ModelConfig):
        """Test sample_model_config fixture provides valid configuration."""
        assert isinstance(sample_model_config, ModelConfig)
        assert sample_model_config.name == "test_model"
        assert sample_model_config.type == "openai_api"
        assert sample_model_config.path == "gpt-3.5-turbo"
        assert sample_model_config.max_tokens == 512
        assert sample_model_config.temperature == 0.1
        assert "api_key" in sample_model_config.config

    def test_sample_evaluation_config_fixture(self, sample_evaluation_config: EvaluationConfig):
        """Test sample_evaluation_config fixture provides valid configuration."""
        assert isinstance(sample_evaluation_config, EvaluationConfig)
        assert "accuracy" in sample_evaluation_config.metrics
        assert "precision" in sample_evaluation_config.metrics
        assert "recall" in sample_evaluation_config.metrics
        assert "f1_score" in sample_evaluation_config.metrics
        assert sample_evaluation_config.parallel_jobs == 2
        assert sample_evaluation_config.timeout_minutes == 30
        assert sample_evaluation_config.batch_size == 16

    def test_sample_experiment_config_fixture(self, sample_experiment_config: ExperimentConfig):
        """Test sample_experiment_config fixture provides valid configuration."""
        assert isinstance(sample_experiment_config, ExperimentConfig)
        assert sample_experiment_config.name == "Test Experiment"
        assert len(sample_experiment_config.datasets) == 1
        assert len(sample_experiment_config.models) == 1
        assert isinstance(sample_experiment_config.datasets[0], DatasetConfig)
        assert isinstance(sample_experiment_config.models[0], ModelConfig)
        assert isinstance(sample_experiment_config.evaluation, EvaluationConfig)


class TestMockFixtures:
    """Test mock service fixtures."""

    def test_mock_openai_client_fixture(self, mock_openai_client: MagicMock):
        """Test mock_openai_client fixture provides proper mock."""
        assert isinstance(mock_openai_client, MagicMock)
        assert hasattr(mock_openai_client, "chat")
        assert hasattr(mock_openai_client.chat, "completions")

        # Test mock response
        response = mock_openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test"}]
        )

        assert response.choices[0].message.content == "ATTACK"
        assert response.choices[0].message.role == "assistant"
        assert response.usage.prompt_tokens == 50
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 55

    def test_mock_anthropic_client_fixture(self, mock_anthropic_client: MagicMock):
        """Test mock_anthropic_client fixture provides proper mock."""
        assert isinstance(mock_anthropic_client, MagicMock)
        assert hasattr(mock_anthropic_client, "messages")

        # Test mock response
        response = mock_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229", messages=[{"role": "user", "content": "Test"}]
        )

        assert response.content[0].text == "BENIGN"
        assert response.usage.input_tokens == 45
        assert response.usage.output_tokens == 3

    def test_mock_async_openai_client_fixture(self, mock_async_openai_client: AsyncMock):
        """Test mock_async_openai_client fixture provides proper async mock."""
        assert isinstance(mock_async_openai_client, AsyncMock)

        # Test that the mock is properly configured
        assert hasattr(mock_async_openai_client, "chat")
        assert hasattr(mock_async_openai_client.chat, "completions")
        assert hasattr(mock_async_openai_client.chat.completions, "create")

    def test_mock_async_anthropic_client_fixture(self, mock_async_anthropic_client: AsyncMock):
        """Test mock_async_anthropic_client fixture provides proper async mock."""
        assert isinstance(mock_async_anthropic_client, AsyncMock)

        # Test that the mock is properly configured
        assert hasattr(mock_async_anthropic_client, "messages")
        assert hasattr(mock_async_anthropic_client.messages, "create")


class TestFileFixtures:
    """Test file and directory fixtures."""

    def test_temp_dir_fixture_creates_directory(self, temp_dir: Path):
        """Test that temp_dir fixture creates a valid directory."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Test we can create files in it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_temp_dir_cleanup_isolation(self, temp_dir: Path):
        """Test that temp_dir fixtures are isolated between tests."""
        # This should be a different directory than the previous test
        test_file = temp_dir / "test.txt"
        assert not test_file.exists()  # Should not exist from previous test

    def test_sample_file_dataset_fixture(self, sample_file_dataset: Path):
        """Test sample_file_dataset fixture creates valid dataset file."""
        assert isinstance(sample_file_dataset, Path)
        assert sample_file_dataset.exists()
        assert sample_file_dataset.suffix == ".jsonl"

        # Verify content
        lines = sample_file_dataset.read_text().strip().split("\n")
        assert len(lines) == 4  # Should have 4 sample entries

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "text" in data
            assert "label" in data
            assert data["label"] in ["ATTACK", "BENIGN"]

    def test_sample_config_file_fixture(self, sample_config_file: Path):
        """Test sample_config_file fixture creates valid config file."""
        assert isinstance(sample_config_file, Path)
        assert sample_config_file.exists()
        assert sample_config_file.suffix == ".yaml"

        # Verify we can load as YAML
        import yaml

        with open(sample_config_file) as f:
            config = yaml.safe_load(f)

        assert "experiment" in config
        assert "datasets" in config
        assert "models" in config
        assert "evaluation" in config

        # Verify structure
        assert config["experiment"]["name"] == "Test Experiment"
        assert len(config["datasets"]) == 1
        assert len(config["models"]) == 1


class TestDataFixtures:
    """Test data fixtures."""

    def test_sample_config_fixture(self, sample_config: dict[str, Any]):
        """Test sample_config fixture provides valid configuration dictionary."""
        assert isinstance(sample_config, dict)
        assert "experiment" in sample_config
        assert "models" in sample_config
        assert "datasets" in sample_config
        assert "evaluation" in sample_config

        assert sample_config["experiment"]["name"] == "Test Experiment"
        assert len(sample_config["models"]) == 1
        assert len(sample_config["datasets"]) == 1

    def test_sample_cybersec_data_fixture(self, sample_cybersec_data: list[dict[str, Any]]):
        """Test sample_cybersec_data fixture provides valid cybersecurity data."""
        assert isinstance(sample_cybersec_data, list)
        assert len(sample_cybersec_data) == 3

        for item in sample_cybersec_data:
            assert "text" in item
            assert "label" in item
            assert item["label"] in ["ATTACK", "BENIGN"]

            if item["label"] == "ATTACK":
                assert "attack_type" in item
                assert item["attack_type"] is not None
            else:
                assert item.get("attack_type") is None

    def test_sample_predictions_fixture(self, sample_predictions: list[dict[str, Any]]):
        """Test sample_predictions fixture provides valid prediction data."""
        assert isinstance(sample_predictions, list)
        assert len(sample_predictions) == 2

        for prediction in sample_predictions:
            assert "sample_id" in prediction
            assert "input_text" in prediction
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert "explanation" in prediction
            assert "inference_time_ms" in prediction

            assert prediction["prediction"] in ["ATTACK", "BENIGN"]
            assert 0.0 <= prediction["confidence"] <= 1.0
            assert prediction["inference_time_ms"] > 0

    def test_sample_ground_truth_fixture(self, sample_ground_truth: list[dict[str, Any]]):
        """Test sample_ground_truth fixture provides valid ground truth data."""
        assert isinstance(sample_ground_truth, list)
        assert len(sample_ground_truth) == 2

        for truth in sample_ground_truth:
            assert "sample_id" in truth
            assert "label" in truth
            assert truth["label"] in ["ATTACK", "BENIGN"]


class TestEnvironmentFixtures:
    """Test environment fixtures."""

    def test_clean_env_fixture_sets_defaults(self):
        """Test clean_env fixture sets default environment variables."""
        # The clean_env fixture should set these automatically
        assert os.environ.get("OPENAI_API_KEY") == "test_key"
        assert os.environ.get("ANTHROPIC_API_KEY") == "test_key"

    def test_environment_isolation_between_tests(self):
        """Test that environment changes are isolated between tests."""
        # Modify environment in this test
        os.environ["TEST_VAR"] = "test_value"
        assert os.environ.get("TEST_VAR") == "test_value"

    def test_environment_cleanup_works(self):
        """Test that environment is cleaned up between tests."""
        # The TEST_VAR from previous test should not be present
        assert "TEST_VAR" not in os.environ

"""
Pytest configuration and shared fixtures for LLM Cybersecurity Benchmark tests.
"""

import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from benchmark.core.config import DatasetConfig, EvaluationConfig, ExperimentConfig, ModelConfig
from benchmark.core.database_manager import DatabaseManager

# Test configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Clean database session for each test using in-memory SQLite."""
    db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db_manager.initialize()
    await db_manager.create_tables()

    async with db_manager.session_scope() as session:
        yield session

    await db_manager.close()


@pytest.fixture
async def db_manager() -> AsyncGenerator[DatabaseManager, None]:
    """Database manager fixture with in-memory SQLite."""
    manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await manager.initialize()
    await manager.create_tables()

    yield manager

    await manager.close()


@pytest.fixture
def sample_dataset_config() -> DatasetConfig:
    """Valid dataset configuration for testing."""
    return DatasetConfig(
        name="test_dataset",
        source="local",
        path="/tmp/test_dataset",
        max_samples=100,
        test_split=0.2,
        validation_split=0.1,
        preprocessing=["tokenize", "normalize"],
    )


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Valid model configuration for testing."""
    return ModelConfig(
        name="test_model",
        type="openai_api",
        path="gpt-3.5-turbo",
        config={"api_key": "test_key"},
        max_tokens=512,
        temperature=0.1,
    )


@pytest.fixture
def sample_evaluation_config() -> EvaluationConfig:
    """Valid evaluation configuration for testing."""
    return EvaluationConfig(
        metrics=["accuracy", "precision", "recall", "f1_score"],
        parallel_jobs=2,
        timeout_minutes=30,
        batch_size=16,
    )


@pytest.fixture
def sample_experiment_config() -> ExperimentConfig:
    """Valid experiment configuration for testing."""
    return ExperimentConfig(
        name="Test Experiment",
        description="Test experiment for unit testing",
        output_dir="/tmp/test_output",
        datasets=[DatasetConfig(name="test_dataset", source="local", path="/tmp/test_dataset")],
        models=[ModelConfig(name="test_model", type="openai_api", path="gpt-3.5-turbo")],
        evaluation=EvaluationConfig(metrics=["accuracy", "f1_score"], parallel_jobs=1),
    )


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "experiment": {
            "name": "Test Experiment",
            "description": "Test experiment for unit testing",
        },
        "models": [{"name": "test_model", "type": "mock", "path": "test://mock-model"}],
        "datasets": [{"name": "test_dataset", "source": "mock", "path": "test://mock-dataset"}],
        "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 1},
    }


@pytest.fixture
def sample_cybersec_data() -> list[dict[str, Any]]:
    """Sample cybersecurity data for testing."""
    return [
        {
            "text": "192.168.1.100 -> 10.0.0.5 PORT_SCAN detected on ports 22,23,80,443",
            "label": "ATTACK",
            "attack_type": "reconnaissance",
        },
        {
            "text": "Normal HTTP GET request to /api/users/profile",
            "label": "BENIGN",
            "attack_type": None,
        },
        {
            "text": "Email with malicious attachment: invoice.pdf.exe",
            "label": "ATTACK",
            "attack_type": "malware",
        },
    ]


@pytest.fixture
def sample_predictions() -> list[dict[str, Any]]:
    """Sample model predictions for testing."""
    return [
        {
            "sample_id": "1",
            "input_text": "Port scan detected",
            "prediction": "ATTACK",
            "confidence": 0.95,
            "explanation": "Multiple port connection attempts detected",
            "inference_time_ms": 150.0,
        },
        {
            "sample_id": "2",
            "input_text": "Normal web request",
            "prediction": "BENIGN",
            "confidence": 0.85,
            "explanation": "Standard HTTP request pattern",
            "inference_time_ms": 120.0,
        },
    ]


@pytest.fixture
def sample_ground_truth() -> list[dict[str, Any]]:
    """Sample ground truth data for testing."""
    return [
        {"sample_id": "1", "label": "ATTACK", "attack_type": "reconnaissance"},
        {"sample_id": "2", "label": "BENIGN", "attack_type": None},
    ]


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for API testing."""
    mock_client = MagicMock()

    # Mock completion response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="ATTACK", role="assistant"), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=5, total_tokens=55)

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mock Anthropic client for API testing."""
    mock_client = MagicMock()

    # Mock message response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="BENIGN")]
    mock_response.usage = MagicMock(input_tokens=45, output_tokens=3)

    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_async_openai_client() -> AsyncMock:
    """Mock async OpenAI client for API testing."""
    mock_client = AsyncMock()

    # Mock async completion response
    mock_response = AsyncMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="ATTACK", role="assistant"), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=5, total_tokens=55)

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_async_anthropic_client() -> AsyncMock:
    """Mock async Anthropic client for API testing."""
    mock_client = AsyncMock()

    # Mock async message response
    mock_response = AsyncMock()
    mock_response.content = [MagicMock(text="BENIGN")]
    mock_response.usage = MagicMock(input_tokens=45, output_tokens=3)

    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_file_dataset(temp_dir: Path) -> Path:
    """Create a sample dataset file for testing."""
    dataset_file = temp_dir / "test_dataset.jsonl"

    # Create sample JSONL data
    sample_data = [
        '{"text": "Port scan detected on 192.168.1.1", "label": "ATTACK", "attack_type": "reconnaissance"}',
        '{"text": "Normal HTTP GET request", "label": "BENIGN", "attack_type": null}',
        '{"text": "SQL injection attempt detected", "label": "ATTACK", "attack_type": "injection"}',
        '{"text": "User login successful", "label": "BENIGN", "attack_type": null}',
    ]

    with open(dataset_file, "w") as f:
        f.write("\n".join(sample_data))

    return dataset_file


@pytest.fixture
def sample_config_file(temp_dir: Path) -> Path:
    """Create a sample configuration file for testing."""
    config_file = temp_dir / "test_config.yaml"

    config_content = """
experiment:
  name: "Test Experiment"
  description: "Sample test experiment"
  output_dir: "/tmp/test_output"

datasets:
  - name: "test_dataset"
    source: "local"
    path: "/tmp/test_dataset.jsonl"
    max_samples: 100
    test_split: 0.2
    validation_split: 0.1

models:
  - name: "test_model"
    type: "openai_api"
    path: "gpt-3.5-turbo"
    config:
      api_key: "${OPENAI_API_KEY:test_key}"
    max_tokens: 512
    temperature: 0.1

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
  parallel_jobs: 1
  timeout_minutes: 30
  batch_size: 16
"""

    with open(config_file, "w") as f:
        f.write(config_content.strip())

    return config_file


# Environment fixtures
@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables for each test."""
    # Store original environment
    original_env = os.environ.copy()

    # Set test defaults
    os.environ.setdefault("OPENAI_API_KEY", "test_key")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Test markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests that run quickly in isolation")
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interaction"
    )
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "slow: Slow running tests that may take several minutes")

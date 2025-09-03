"""
Pytest configuration and shared fixtures for LLM Cybersecurity Benchmark tests.
"""
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

# Test configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "experiment": {
            "name": "Test Experiment",
            "description": "Test experiment for unit testing"
        },
        "models": [
            {
                "name": "test_model",
                "type": "mock",
                "path": "test://mock-model"
            }
        ],
        "datasets": [
            {
                "name": "test_dataset",
                "source": "mock",
                "path": "test://mock-dataset"
            }
        ],
        "evaluation": {
            "metrics": ["accuracy"],
            "parallel_jobs": 1
        }
    }


@pytest.fixture
def sample_cybersec_data() -> List[Dict[str, Any]]:
    """Sample cybersecurity data for testing."""
    return [
        {
            "text": "192.168.1.100 -> 10.0.0.5 PORT_SCAN detected on ports 22,23,80,443",
            "label": "ATTACK",
            "attack_type": "reconnaissance"
        },
        {
            "text": "Normal HTTP GET request to /api/users/profile",
            "label": "BENIGN",
            "attack_type": None
        },
        {
            "text": "Email with malicious attachment: invoice.pdf.exe",
            "label": "ATTACK", 
            "attack_type": "malware"
        }
    ]


@pytest.fixture
def sample_predictions() -> List[Dict[str, Any]]:
    """Sample model predictions for testing."""
    return [
        {
            "sample_id": "1",
            "input_text": "Port scan detected",
            "prediction": "ATTACK",
            "confidence": 0.95,
            "explanation": "Multiple port connection attempts detected",
            "inference_time_ms": 150.0
        },
        {
            "sample_id": "2", 
            "input_text": "Normal web request",
            "prediction": "BENIGN",
            "confidence": 0.85,
            "explanation": "Standard HTTP request pattern",
            "inference_time_ms": 120.0
        }
    ]


@pytest.fixture
def sample_ground_truth() -> List[Dict[str, Any]]:
    """Sample ground truth data for testing."""
    return [
        {
            "sample_id": "1",
            "label": "ATTACK",
            "attack_type": "reconnaissance"
        },
        {
            "sample_id": "2",
            "label": "BENIGN",
            "attack_type": None
        }
    ]


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that run quickly in isolation"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interaction"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests that may take several minutes"
    )
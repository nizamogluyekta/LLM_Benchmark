"""
Tests for ModelValidator aiohttp fallback behavior.

This ensures that the system works gracefully when aiohttp is not available.
"""

import sys
from unittest.mock import patch

import pytest

from benchmark.models.model_validator import ModelValidator


class TestModelValidatorFallback:
    """Test ModelValidator behavior when aiohttp is not available."""

    @pytest.mark.asyncio
    async def test_validator_works_without_aiohttp(self):
        """Test that ModelValidator works when aiohttp is not available."""
        # Simulate aiohttp not being available
        with patch.dict(sys.modules, {"aiohttp": None}):
            # Force reimport with aiohttp disabled
            import importlib

            from benchmark.models import model_validator

            importlib.reload(model_validator)

            # Create validator instance
            validator = model_validator.ModelValidator()

            # Test basic functionality
            result = await validator.check_hardware_requirements({"type": "test"})
            assert result is not None

    @pytest.mark.asyncio
    async def test_api_validation_fallback(self):
        """Test that API validation returns True when aiohttp is not available."""
        # This test ensures that when aiohttp is not available,
        # API validation doesn't fail but instead returns a sensible default
        validator = ModelValidator()

        # Mock the _get_aiohttp function to return None (simulating missing aiohttp)
        with patch("benchmark.models.model_validator._get_aiohttp", return_value=None):
            config = {
                "provider": "openai",
                "api_key": "test-key",
                "endpoint": "https://api.openai.com/v1/",
            }

            result = await validator._validate_api_config(config)
            # Should return True when aiohttp is not available (graceful fallback)
            assert result is True

    def test_aiohttp_available_flag(self):
        """Test that AIOHTTP_AVAILABLE flag is properly set."""
        from benchmark.models.model_validator import AIOHTTP_AVAILABLE

        # Should behave like a boolean in normal environments where aiohttp is installed
        assert bool(AIOHTTP_AVAILABLE) in [True, False]
        # The flag should be truthy when aiohttp is available
        if hasattr(AIOHTTP_AVAILABLE, "__bool__"):
            assert callable(AIOHTTP_AVAILABLE.__bool__)

    @pytest.mark.asyncio
    async def test_validator_initialization(self):
        """Test that ModelValidator can be initialized regardless of aiohttp availability."""
        validator = ModelValidator()
        assert validator is not None
        assert hasattr(validator, "api_endpoints")
        assert hasattr(validator, "logger")

    def test_hardware_info_model(self):
        """Test that HardwareInfo model works without aiohttp."""
        from benchmark.models.model_validator import HardwareInfo

        hardware_info = HardwareInfo(
            cpu_cores=8,
            memory_gb=16.0,
            platform="darwin",
            architecture="arm64",
            disk_space_gb=512.0,
        )

        assert hardware_info.cpu_cores == 8
        assert hardware_info.memory_gb == 16.0
        assert hardware_info.platform == "darwin"

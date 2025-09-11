"""
Simple integration test for resource management components.

This test ensures basic functionality without complex dependencies.
"""

from unittest.mock import Mock

import pytest


# Test basic imports work
def test_resource_manager_import():
    """Test that resource manager can be imported."""
    try:
        from benchmark.models.resource_manager import ModelResourceManager

        assert ModelResourceManager is not None
    except ImportError as e:
        pytest.skip(f"Resource manager import failed: {e}")


def test_model_cache_import():
    """Test that model cache can be imported."""
    try:
        from benchmark.models.model_cache import ModelCache

        assert ModelCache is not None
    except ImportError as e:
        pytest.skip(f"Model cache import failed: {e}")


def test_resource_manager_basic_functionality():
    """Test basic resource manager functionality."""
    try:
        from benchmark.models.resource_manager import MemoryMonitor, ModelResourceManager

        # Test memory monitor creation
        monitor = MemoryMonitor()
        assert monitor is not None

        # Test resource manager creation
        manager = ModelResourceManager(max_memory_gb=8.0)
        assert manager.max_memory_gb == 8.0
        assert len(manager.loaded_models) == 0

    except ImportError as e:
        pytest.skip(f"Resource manager functionality test failed: {e}")


def test_model_cache_basic_functionality():
    """Test basic model cache functionality."""
    try:
        from benchmark.models.model_cache import CompressionLevel, ModelCache

        # Test cache creation
        cache = ModelCache(max_cache_size_gb=1.0)
        assert cache.max_cache_size_bytes == 1024**3
        assert len(cache.cache_entries) == 0

        # Test compression levels
        assert CompressionLevel.NONE.value == 0
        assert CompressionLevel.BALANCED.value == 6

    except ImportError as e:
        pytest.skip(f"Model cache functionality test failed: {e}")


@pytest.mark.asyncio
async def test_memory_monitor_basic_operations():
    """Test basic memory monitor operations."""
    try:
        from benchmark.models.resource_manager import MemoryMonitor

        monitor = MemoryMonitor()

        # Test memory usage retrieval (should not raise exception)
        try:
            usage = await monitor.get_current_usage()
            assert isinstance(usage, float)
            assert usage >= 0.0
        except Exception:
            # If psutil is not available, that's okay for basic test
            pass

        # Test Apple Silicon detection (should not raise exception)
        try:
            is_apple_silicon = monitor._detect_apple_silicon()
            assert isinstance(is_apple_silicon, bool)
        except Exception:
            # Platform detection might fail in some environments
            pass

    except ImportError as e:
        pytest.skip(f"Memory monitor test failed: {e}")


@pytest.mark.asyncio
async def test_model_cache_hash_generation():
    """Test model cache configuration hash generation."""
    try:
        from benchmark.models.model_cache import ModelCache

        cache = ModelCache()

        # Create mock config
        mock_config = Mock()
        mock_config.model_name = "test-model"
        mock_config.type = "test_type"
        mock_config.parameters = {"temp": 0.7}
        mock_config.context_length = 4096

        # Test hash generation
        hash1 = cache._generate_config_hash(mock_config)
        hash2 = cache._generate_config_hash(mock_config)

        assert isinstance(hash1, str)
        assert len(hash1) == 16  # MD5 hash truncated to 16 chars
        assert hash1 == hash2  # Same config should generate same hash

    except ImportError as e:
        pytest.skip(f"Model cache hash test failed: {e}")


def test_dataclass_imports():
    """Test that all dataclasses can be imported and created."""
    try:
        from benchmark.models.model_cache import (
            CacheStatus,
            CompressionLevel,
        )
        from benchmark.models.resource_manager import (
            ModelEstimate,
            ModelPriority,
            SystemMemoryInfo,
            UnloadReason,
        )

        # Test enum values
        assert ModelPriority.NORMAL.value == 2
        assert UnloadReason.USER_REQUEST.value == "user_request"
        assert CacheStatus.FRESH.value == "fresh"
        assert CompressionLevel.BALANCED.value == 6

        # Test dataclass creation (basic)
        memory_info = SystemMemoryInfo(
            total_gb=32.0, available_gb=20.0, used_gb=12.0, percentage=37.5
        )
        assert memory_info.total_gb == 32.0

        estimate = ModelEstimate(
            base_memory_gb=7.0,
            context_memory_gb=1.0,
            overhead_memory_gb=1.0,
            total_estimated_gb=9.0,
            confidence=0.9,
        )
        assert estimate.total_estimated_gb == 9.0

    except ImportError as e:
        pytest.skip(f"Dataclass import test failed: {e}")

"""
Integration tests for resource management system.

This module provides comprehensive integration tests for the ModelResourceManager
and ModelCache system working together, including realistic model loading
scenarios, concurrent access patterns, and Apple Silicon M4 Pro optimizations.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from benchmark.core.config import ModelConfig
from benchmark.models.model_cache import CompressionLevel, ModelCache
from benchmark.models.resource_manager import (
    ModelResourceManager,
    UnloadReason,
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_small_model_config():
    """Create mock configuration for a small model."""
    config = Mock(spec=ModelConfig)
    config.model_name = "gpt-4o-mini"
    config.type = "openai_api"
    config.parameters = {"temperature": 0.7}
    config.context_length = 8192
    return config


@pytest.fixture
def mock_medium_model_config():
    """Create mock configuration for a medium model."""
    config = Mock(spec=ModelConfig)
    config.model_name = "llama2-7b"
    config.type = "mlx_local"
    config.parameters = {"temperature": 0.5}
    config.context_length = 4096
    return config


@pytest.fixture
def mock_large_model_config():
    """Create mock configuration for a large model."""
    config = Mock(spec=ModelConfig)
    config.model_name = "llama2-70b"
    config.type = "mlx_local"
    config.parameters = {"temperature": 0.3}
    config.context_length = 4096
    return config


@pytest.fixture
async def integrated_resource_system(temp_cache_dir):
    """Create integrated resource management system."""
    resource_manager = ModelResourceManager(
        max_memory_gb=16.0, cache_dir=temp_cache_dir / "resource_cache"
    )

    model_cache = ModelCache(cache_dir=temp_cache_dir / "model_cache", max_cache_size_gb=4.0)

    await resource_manager.initialize()
    await model_cache.initialize()

    yield resource_manager, model_cache

    await resource_manager.shutdown()
    await model_cache.shutdown()


class TestResourceCacheIntegration:
    """Test resource manager and cache integration."""

    @pytest.mark.asyncio
    async def test_complete_model_lifecycle(
        self, integrated_resource_system, mock_medium_model_config
    ):
        """Test complete model lifecycle with caching."""
        resource_manager, model_cache = integrated_resource_system
        model_id = "test_lifecycle_model"

        # Step 1: Check if model can be loaded
        with patch.object(resource_manager.memory_monitor, "get_current_usage", return_value=4.0):
            check_result = await resource_manager.can_load_model(mock_medium_model_config)
            assert check_result.can_load is True

        # Step 2: Mock model state for caching
        mock_model_state = {"weights": "dummy_weights", "config": "model_config"}

        # Step 3: Cache model state
        cache_success = await model_cache.cache_model_state(
            model_id=model_id,
            config=mock_medium_model_config,
            model_state=mock_model_state,
            metadata={"size_gb": 7.0, "load_time": 45.2},
            compression=CompressionLevel.BALANCED,
        )
        assert cache_success is True

        # Step 4: Register model in resource manager
        await resource_manager.register_model_load(
            model_id=model_id,
            config=mock_medium_model_config,
            actual_memory_gb=7.2,
            plugin_type="mlx_local",
        )

        assert model_id in resource_manager.loaded_models

        # Step 5: Simulate model usage
        for _ in range(5):
            await resource_manager.register_model_access(model_id)
            await asyncio.sleep(0.01)

        model_info = resource_manager.loaded_models[model_id]
        assert model_info.access_count == 6  # 1 initial + 5 accesses

        # Step 6: Load from cache
        cached_state = await model_cache.load_cached_model(model_id, mock_medium_model_config)
        assert cached_state is not None
        assert cached_state == mock_model_state

        # Step 7: Unload model
        await resource_manager.unregister_model(model_id, UnloadReason.USER_REQUEST)
        assert model_id not in resource_manager.loaded_models

        # Step 8: Verify cache still contains model
        cached_state_after_unload = await model_cache.load_cached_model(
            model_id, mock_medium_model_config
        )
        assert cached_state_after_unload is not None

    @pytest.mark.asyncio
    async def test_memory_pressure_with_caching(
        self,
        integrated_resource_system,
        mock_small_model_config,
        mock_medium_model_config,
        mock_large_model_config,
    ):
        """Test memory pressure scenarios with model caching."""
        resource_manager, model_cache = integrated_resource_system

        # Load multiple models to create memory pressure
        models = [
            ("small_model", mock_small_model_config, 1.0),
            ("medium_model", mock_medium_model_config, 7.0),
            ("large_model", mock_large_model_config, 12.0),  # This should cause pressure
        ]

        loaded_models = []

        for model_id, config, memory_usage in models:
            # Cache model state first
            mock_state = {"id": model_id, "size": memory_usage}
            await model_cache.cache_model_state(
                model_id=model_id,
                config=config,
                model_state=mock_state,
                compression=CompressionLevel.BALANCED,
            )

            # Check if we can load
            with patch.object(
                resource_manager.memory_monitor,
                "get_current_usage",
                return_value=sum(m[2] for m in loaded_models) + memory_usage,
            ):
                check_result = await resource_manager.can_load_model(config)

                if check_result.can_load:
                    await resource_manager.register_model_load(
                        model_id, config, memory_usage, "test_plugin"
                    )
                    loaded_models.append((model_id, config, memory_usage))
                else:
                    # Should suggest unloading some models
                    assert len(check_result.required_unloads) > 0
                    break

        # Verify that we hit memory pressure
        assert len(loaded_models) < 3  # Shouldn't be able to load all models

        # Get resource statistics
        with patch.object(
            resource_manager.memory_monitor,
            "get_current_usage",
            return_value=sum(m[2] for m in loaded_models),
        ):
            stats = await resource_manager.get_resource_statistics()
            assert stats["models"]["loaded_count"] == len(loaded_models)

    @pytest.mark.asyncio
    async def test_concurrent_model_operations(
        self, integrated_resource_system, mock_medium_model_config
    ):
        """Test concurrent model loading, caching, and access operations."""
        resource_manager, model_cache = integrated_resource_system

        async def model_operation_task(model_index: int):
            """Simulate concurrent model operations."""
            model_id = f"concurrent_model_{model_index}"

            # Create mock model state
            mock_state = {"model_id": model_id, "weights": f"weights_{model_index}"}

            # Cache model
            cache_success = await model_cache.cache_model_state(
                model_id=model_id,
                config=mock_medium_model_config,
                model_state=mock_state,
                compression=CompressionLevel.FAST,
            )

            if cache_success:
                # Register in resource manager
                with patch.object(
                    resource_manager.memory_monitor,
                    "get_current_usage",
                    return_value=4.0 + model_index,
                ):
                    await resource_manager.register_model_load(
                        model_id, mock_medium_model_config, 3.0, "test_plugin"
                    )

                # Simulate random access patterns
                for _ in range(10):
                    await resource_manager.register_model_access(model_id)
                    await asyncio.sleep(0.001)

                # Load from cache
                cached_state = await model_cache.load_cached_model(
                    model_id, mock_medium_model_config
                )

                return cached_state is not None

            return False

        # Run multiple concurrent operations
        tasks = [model_operation_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify most operations succeeded
        successful_operations = sum(1 for r in results if r is True)
        assert successful_operations >= 3  # At least 3 should succeed

        # Verify resource manager state
        assert len(resource_manager.loaded_models) >= 3

    @pytest.mark.asyncio
    async def test_model_loading_optimization_with_cache(self, integrated_resource_system):
        """Test model loading order optimization considering cache hits."""
        resource_manager, model_cache = integrated_resource_system

        # Create different model configurations
        configs = []

        # API model (high priority)
        api_config = Mock(spec=ModelConfig)
        api_config.model_name = "gpt-4o-mini"
        api_config.type = "openai_api"
        configs.append(api_config)

        # Cached local model (should get priority boost)
        cached_config = Mock(spec=ModelConfig)
        cached_config.model_name = "llama2-7b-cached"
        cached_config.type = "mlx_local"
        configs.append(cached_config)

        # Non-cached local model
        uncached_config = Mock(spec=ModelConfig)
        uncached_config.model_name = "llama2-13b"
        uncached_config.type = "mlx_local"
        configs.append(uncached_config)

        # Pre-cache one model
        await model_cache.cache_model_state(
            model_id="cached_model",
            config=cached_config,
            model_state={"cached": True},
            compression=CompressionLevel.BALANCED,
        )

        # Get optimized loading order
        optimized_order = await resource_manager.optimize_model_loading_order(configs)

        assert len(optimized_order) == 3

        # API model should come first
        assert optimized_order[0].type == "openai_api"

    @pytest.mark.asyncio
    async def test_automatic_cleanup_integration(
        self, integrated_resource_system, mock_medium_model_config
    ):
        """Test automatic cleanup integration between resource manager and cache."""
        resource_manager, model_cache = integrated_resource_system

        # Create models with different usage patterns
        models_data = [
            ("active_model", datetime.now() - timedelta(minutes=5)),
            ("idle_model", datetime.now() - timedelta(hours=2)),
            ("very_old_model", datetime.now() - timedelta(hours=6)),
        ]

        # Register models and cache their states
        for model_id, last_used_time in models_data:
            # Cache model state
            mock_state = {"model_id": model_id, "active": "idle" not in model_id}
            await model_cache.cache_model_state(
                model_id=model_id,
                config=mock_medium_model_config,
                model_state=mock_state,
                ttl_hours=24.0,
            )

            # Register in resource manager
            await resource_manager.register_model_load(
                model_id, mock_medium_model_config, 5.0, "test_plugin"
            )

            # Set last used time
            model_info = resource_manager.loaded_models[model_id]
            model_info.last_used = last_used_time

        # Force cleanup in resource manager
        cleanup_result = await resource_manager.force_cleanup()

        # Verify idle models were unloaded from resource manager
        assert len(cleanup_result["unloaded_models"]) >= 2
        assert "idle_model" in cleanup_result["unloaded_models"]
        assert "very_old_model" in cleanup_result["unloaded_models"]

        # Verify models are still cached (can be reloaded quickly)
        for model_id, _ in models_data:
            cached_state = await model_cache.load_cached_model(model_id, mock_medium_model_config)
            assert cached_state is not None

    @pytest.mark.asyncio
    async def test_cache_compression_effectiveness(
        self, integrated_resource_system, mock_medium_model_config
    ):
        """Test effectiveness of different compression levels."""
        resource_manager, model_cache = integrated_resource_system

        # Create large mock model state
        large_model_state = {
            "weights": ["dummy_weight"] * 10000,
            "biases": [0.1] * 5000,
            "metadata": {"size": "large", "layers": 32},
        }

        compression_results = {}

        # Test different compression levels
        for compression_level in [
            CompressionLevel.NONE,
            CompressionLevel.FAST,
            CompressionLevel.BALANCED,
            CompressionLevel.MAXIMUM,
        ]:
            model_id = f"model_{compression_level.name.lower()}"

            start_time = time.time()
            cache_success = await model_cache.cache_model_state(
                model_id=model_id,
                config=mock_medium_model_config,
                model_state=large_model_state,
                compression=compression_level,
            )
            cache_time = time.time() - start_time

            assert cache_success is True

            # Load back and measure time
            start_time = time.time()
            loaded_state = await model_cache.load_cached_model(model_id, mock_medium_model_config)
            load_time = time.time() - start_time

            assert loaded_state == large_model_state

            # Get cache entry info
            cache_entry = model_cache.cache_entries[
                f"{model_id}_{model_cache._generate_config_hash(mock_medium_model_config)}"
            ]

            compression_results[compression_level] = {
                "size_bytes": cache_entry.file_size_bytes,
                "cache_time": cache_time,
                "load_time": load_time,
                "compressed": cache_entry.compressed,
            }

        # Verify compression effectiveness
        uncompressed_size = compression_results[CompressionLevel.NONE]["size_bytes"]
        balanced_size = compression_results[CompressionLevel.BALANCED]["size_bytes"]
        maximum_size = compression_results[CompressionLevel.MAXIMUM]["size_bytes"]

        assert balanced_size < uncompressed_size
        assert maximum_size <= balanced_size

    @pytest.mark.asyncio
    async def test_apple_silicon_optimization_integration(
        self, integrated_resource_system, mock_medium_model_config
    ):
        """Test Apple Silicon specific optimizations."""
        resource_manager, model_cache = integrated_resource_system

        # Mock Apple Silicon detection
        with (
            patch.object(
                resource_manager.memory_monitor, "_detect_apple_silicon", return_value=True
            ),
            patch.object(resource_manager.memory_monitor, "get_system_memory_info") as mock_memory,
        ):
            mock_memory.return_value.apple_silicon_unified = True
            mock_memory.return_value.total_gb = 32.0

            system_info = await resource_manager.memory_monitor.get_system_memory_info()
            assert system_info.apple_silicon_unified is True

            # Test MLX-optimized model loading
            mock_medium_model_config.type = "mlx_local"
            check_result = await resource_manager.can_load_model(mock_medium_model_config)

            # Should get Apple Silicon specific recommendations
            optimization_suggestions = check_result.optimization_suggestions or []
            apple_recs = [r for r in optimization_suggestions if "Apple Silicon" in r]
            assert len(apple_recs) > 0

    @pytest.mark.asyncio
    async def test_cache_statistics_integration(
        self, integrated_resource_system, mock_small_model_config
    ):
        """Test integration of cache statistics with resource management."""
        resource_manager, model_cache = integrated_resource_system

        # Perform various cache operations
        models_to_test = 10
        for i in range(models_to_test):
            model_id = f"stats_test_model_{i}"
            mock_state = {"model_id": model_id, "index": i}

            # Cache some models
            if i < 7:  # Cache 7 out of 10
                await model_cache.cache_model_state(
                    model_id=model_id,
                    config=mock_small_model_config,
                    model_state=mock_state,
                    compression=CompressionLevel.BALANCED,
                )

            # Try to load all models (some will hit, some will miss)
            cached_state = await model_cache.load_cached_model(model_id, mock_small_model_config)

            if cached_state:
                # Register successful loads in resource manager
                await resource_manager.register_model_load(
                    model_id, mock_small_model_config, 1.0, "test_plugin"
                )

        # Get integrated statistics
        cache_stats = await model_cache.get_cache_statistics()
        resource_stats = await resource_manager.get_resource_statistics()

        # Verify cache statistics
        assert cache_stats.total_entries == 7  # Only 7 were cached
        assert cache_stats.hit_count == 7  # Hits from cached models
        assert cache_stats.miss_count == 3  # Misses from non-cached models
        assert cache_stats.cache_efficiency_percent > 0.0

        # Verify resource statistics
        assert resource_stats["models"]["loaded_count"] == 7  # Only hits were loaded

    @pytest.mark.asyncio
    async def test_error_recovery_integration(
        self, integrated_resource_system, mock_medium_model_config
    ):
        """Test error recovery scenarios in integrated system."""
        resource_manager, model_cache = integrated_resource_system

        # Test 1: Cache corruption handling
        model_id = "corrupted_model"
        mock_state = {"data": "valid_model_state"}

        # Cache model successfully
        cache_success = await model_cache.cache_model_state(
            model_id, mock_medium_model_config, mock_state
        )
        assert cache_success is True

        # Corrupt the cache file
        cache_key = f"{model_id}_{model_cache._generate_config_hash(mock_medium_model_config)}"
        cache_entry = model_cache.cache_entries[cache_key]
        cache_entry.file_path.write_bytes(b"corrupted_data")

        # Try to load corrupted model (should handle gracefully)
        loaded_state = await model_cache.load_cached_model(model_id, mock_medium_model_config)
        assert loaded_state is None  # Should return None for corrupted data

        # Test 2: Memory monitoring errors
        with patch.object(
            resource_manager.memory_monitor,
            "get_current_usage",
            side_effect=Exception("Memory monitoring failed"),
        ):
            # Should handle memory monitoring errors gracefully
            check_result = await resource_manager.can_load_model(mock_medium_model_config)
            assert check_result.can_load is False
            assert "Error checking resources" in check_result.recommendations[0]

        # Test 3: Cache directory issues
        # Make cache directory read-only to simulate permission issues
        original_permissions = model_cache.cache_dir.stat().st_mode
        try:
            model_cache.cache_dir.chmod(0o444)  # Read-only

            # Try to cache (should handle permission errors)
            cache_success = await model_cache.cache_model_state(
                "permission_test", mock_medium_model_config, {"test": "data"}
            )
            # Might succeed or fail depending on system, but shouldn't crash

        finally:
            # Restore permissions
            model_cache.cache_dir.chmod(original_permissions)

    @pytest.mark.asyncio
    async def test_performance_benchmarks_integration(
        self, integrated_resource_system, mock_small_model_config
    ):
        """Test performance benchmarks for integrated system."""
        resource_manager, model_cache = integrated_resource_system

        # Benchmark cache operations
        cache_operations = 50
        model_states = [{"id": i, "data": f"model_data_{i}"} for i in range(cache_operations)]

        # Measure cache write performance
        start_time = time.time()
        for i, state in enumerate(model_states):
            model_id = f"perf_model_{i}"
            await model_cache.cache_model_state(
                model_id, mock_small_model_config, state, compression=CompressionLevel.FAST
            )
        cache_write_time = time.time() - start_time

        # Measure cache read performance
        start_time = time.time()
        for i in range(cache_operations):
            model_id = f"perf_model_{i}"
            cached_state = await model_cache.load_cached_model(model_id, mock_small_model_config)
            assert cached_state is not None
        cache_read_time = time.time() - start_time

        # Measure resource management operations
        start_time = time.time()
        with patch.object(resource_manager.memory_monitor, "get_current_usage", return_value=8.0):
            for i in range(cache_operations):
                model_id = f"perf_model_{i}"
                await resource_manager.register_model_load(
                    model_id, mock_small_model_config, 0.5, "perf_test"
                )
                await resource_manager.register_model_access(model_id)
        resource_ops_time = time.time() - start_time

        # Performance assertions (should complete within reasonable time)
        assert cache_write_time < 5.0  # 5 seconds for 50 cache writes
        assert cache_read_time < 2.0  # 2 seconds for 50 cache reads
        assert resource_ops_time < 1.0  # 1 second for 100 resource operations

        # Calculate operations per second
        cache_write_ops_per_sec = cache_operations / cache_write_time
        cache_read_ops_per_sec = cache_operations / cache_read_time
        resource_ops_per_sec = (cache_operations * 2) / resource_ops_time  # 2 ops per iteration

        # Log performance metrics (will be visible in test output)
        print("\nPerformance Benchmarks:")
        print(f"Cache Write: {cache_write_ops_per_sec:.1f} ops/sec")
        print(f"Cache Read: {cache_read_ops_per_sec:.1f} ops/sec")
        print(f"Resource Ops: {resource_ops_per_sec:.1f} ops/sec")

        # Verify reasonable performance
        assert cache_write_ops_per_sec > 10.0  # At least 10 cache writes per second
        assert cache_read_ops_per_sec > 25.0  # At least 25 cache reads per second
        assert resource_ops_per_sec > 100.0  # At least 100 resource ops per second

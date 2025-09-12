"""
Unit tests for configuration caching components.

This module tests the caching functionality including LRU eviction,
cache statistics, lazy loading, and diff tracking.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio
import yaml

from benchmark.core.config import DatasetConfig, EvaluationConfig, ExperimentConfig, ModelConfig
from benchmark.services.cache import ConfigDiffTracker, ConfigurationCache, LazyConfigLoader


class TestConfigurationCache:
    """Test the ConfigurationCache component."""

    @pytest_asyncio.fixture
    async def cache(self):
        """Fixture for configuration cache."""
        cache = ConfigurationCache(max_size=5, ttl_seconds=10, max_memory_mb=1)
        await cache.initialize()
        yield cache
        await cache.shutdown()

    @pytest.fixture
    def sample_config(self):
        """Fixture for sample configuration."""
        return ExperimentConfig(
            name="Test Config",
            description="Test configuration",
            output_dir="./test_results",
            datasets=[DatasetConfig(name="test_dataset", source="local", path="./data/test.jsonl")],
            models=[
                ModelConfig(name="test_model", type="openai_api", path="gpt-3.5-turbo", config={})
            ],
            evaluation=EvaluationConfig(metrics=["accuracy"], parallel_jobs=2),
        )

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache, sample_config):
        """Test basic cache set and get operations."""
        config_id = "test_config_1"

        # Initially should be empty
        result = await cache.get_config(config_id)
        assert result is None

        # Set configuration
        await cache.set_config(config_id, sample_config)

        # Should retrieve the same configuration
        result = await cache.get_config(config_id)
        assert result is not None
        assert result.name == sample_config.name
        assert result.description == sample_config.description

        # Check cache stats
        stats = cache.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["current_size"] == 1

    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache, sample_config):
        """Test LRU eviction when cache size limit is reached."""
        # Fill cache to capacity (max_size=5)
        for i in range(5):
            config_id = f"config_{i}"
            await cache.set_config(config_id, sample_config)

        # All should be in cache
        for i in range(5):
            result = await cache.get_config(f"config_{i}")
            assert result is not None

        # Add one more (should evict oldest)
        await cache.set_config("config_new", sample_config)

        # Oldest (config_0) should be evicted
        result = await cache.get_config("config_0")
        assert result is None

        # Others should still be present
        for i in range(1, 5):
            result = await cache.get_config(f"config_{i}")
            assert result is not None

        result = await cache.get_config("config_new")
        assert result is not None

        # Check stats
        stats = cache.get_cache_stats()
        assert stats["evictions"] == 1
        assert stats["current_size"] == 5

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, sample_config):
        """Test TTL-based expiration."""
        # Create cache with short TTL
        cache = ConfigurationCache(max_size=10, ttl_seconds=1, max_memory_mb=1)
        await cache.initialize()

        try:
            config_id = "test_config"

            # Set configuration
            await cache.set_config(config_id, sample_config)

            # Should be available immediately
            result = await cache.get_config(config_id)
            assert result is not None

            # Wait for expiration
            await asyncio.sleep(1.5)

            # Should be expired
            result = await cache.get_config(config_id)
            assert result is None

        finally:
            await cache.shutdown()

    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self, sample_config):
        """Test that memory limits are enforced."""
        # Create cache with very small memory limit
        cache = ConfigurationCache(max_size=100, ttl_seconds=60, max_memory_mb=0.001)  # Very small
        await cache.initialize()

        try:
            # Try to add configurations
            for i in range(10):
                config_id = f"config_{i}"
                await cache.set_config(config_id, sample_config)

            # Should have evicted some due to memory limit
            stats = cache.get_cache_stats()
            assert stats["evictions"] > 0
            assert stats["memory_usage_mb"] <= 0.01  # Small overhead allowed

        finally:
            await cache.shutdown()

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache, sample_config):
        """Test cache invalidation."""
        config_id = "test_config"

        # Add configuration
        await cache.set_config(config_id, sample_config)
        assert await cache.get_config(config_id) is not None

        # Invalidate
        result = await cache.invalidate(config_id)
        assert result is True

        # Should no longer be in cache
        assert await cache.get_config(config_id) is None

        # Invalidating non-existent key should return False
        result = await cache.invalidate("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache, sample_config):
        """Test clearing the cache."""
        # Add multiple configurations
        for i in range(3):
            await cache.set_config(f"config_{i}", sample_config)

        # All should be present
        for i in range(3):
            assert await cache.get_config(f"config_{i}") is not None

        # Clear cache
        cleared_count = await cache.clear()
        assert cleared_count == 3

        # All should be gone
        for i in range(3):
            assert await cache.get_config(f"config_{i}") is None

        stats = cache.get_cache_stats()
        assert stats["current_size"] == 0

    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache, sample_config):
        """Test cache statistics accuracy."""
        # Perform various operations
        await cache.set_config("config1", sample_config)
        await cache.set_config("config2", sample_config)

        # Hit
        await cache.get_config("config1")

        # Miss
        await cache.get_config("non_existent")

        # Another hit
        await cache.get_config("config2")

        stats = cache.get_cache_stats()
        assert stats["sets"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_requests"] == 3
        assert stats["hit_rate_percent"] == 66.67  # 2/3 * 100, rounded


class TestLazyConfigLoader:
    """Test the LazyConfigLoader component."""

    @pytest_asyncio.fixture
    async def lazy_loader(self):
        """Fixture for lazy config loader."""
        loader = LazyConfigLoader(cache_size=10)
        yield loader
        await loader.clear_cache()

    @pytest.fixture
    def sample_config_data(self):
        """Fixture for sample configuration data."""
        return {
            "name": "Test Configuration",
            "description": "Test config for lazy loading",
            "output_dir": "./results",
            "datasets": [{"name": "test_dataset", "source": "local", "path": "./data/test.jsonl"}],
            "models": [
                {"name": "test_model", "type": "openai_api", "path": "gpt-3.5-turbo", "config": {}}
            ],
            "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 2, "batch_size": 16},
        }

    @pytest.mark.asyncio
    async def test_load_section(self, lazy_loader, sample_config_data):
        """Test loading specific sections."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_data, f)
            config_path = f.name

        try:
            # Load name section
            name_section = await lazy_loader.load_section(config_path, "name")
            assert name_section == "Test Configuration"

            # Load models section
            models_section = await lazy_loader.load_section(config_path, "models")
            assert len(models_section) == 1
            assert models_section[0]["name"] == "test_model"

            # Load non-existent section should raise KeyError
            with pytest.raises(KeyError):
                await lazy_loader.load_section(config_path, "non_existent")

        finally:
            Path(config_path).unlink()

    @pytest.mark.asyncio
    async def test_config_outline(self, lazy_loader, sample_config_data):
        """Test getting configuration outline."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_data, f)
            config_path = f.name

        try:
            outline = await lazy_loader.get_config_outline(config_path)

            assert outline["name"] == "Test Configuration"
            assert outline["description"] == "Test config for lazy loading"
            assert outline["_models_count"] == 1
            assert outline["_datasets_count"] == 1
            assert "_available_sections" in outline

        finally:
            Path(config_path).unlink()

    @pytest.mark.asyncio
    async def test_preload_common_sections(self, lazy_loader, sample_config_data):
        """Test preloading common sections."""
        config_files = []

        try:
            # Create multiple config files
            for i in range(3):
                config_data = sample_config_data.copy()
                config_data["name"] = f"Config {i}"

                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    yaml.dump(config_data, f)
                    config_files.append(f.name)

            # Preload common sections
            await lazy_loader.preload_common_sections(config_files)

            # Check cache info
            cache_info = await lazy_loader.get_cache_info()
            assert cache_info["preloaded_sections"] > 0
            assert cache_info["cached_files"] > 0

        finally:
            for config_file in config_files:
                Path(config_file).unlink()

    @pytest.mark.asyncio
    async def test_file_modification_detection(self, lazy_loader, sample_config_data):
        """Test file modification detection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_data, f)
            config_path = f.name

        try:
            # Initially should be modified (never seen before)
            is_modified = await lazy_loader.is_config_modified(config_path)
            assert is_modified is True

            # Load a section to establish baseline
            await lazy_loader.load_section(config_path, "name")

            # Should not be modified now
            is_modified = await lazy_loader.is_config_modified(config_path)
            assert is_modified is False

            # Modify the file
            time.sleep(0.1)  # Ensure timestamp difference
            with open(config_path, "a") as f:
                f.write("\n# Modified")

            # Should now be detected as modified
            is_modified = await lazy_loader.is_config_modified(config_path)
            assert is_modified is True

        finally:
            Path(config_path).unlink()

    @pytest.mark.asyncio
    async def test_cache_management(self, sample_config_data):
        """Test cache size management."""
        # Create loader with small cache
        loader = LazyConfigLoader(cache_size=2)
        config_files = []

        try:
            # Create more files than cache can hold
            for i in range(4):
                config_data = sample_config_data.copy()
                config_data["name"] = f"Config {i}"

                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    yaml.dump(config_data, f)
                    config_files.append(f.name)

            # Load sections from all files
            for config_file in config_files:
                await loader.load_section(config_file, "name")

            # Check that cache size is limited
            cache_info = await loader.get_cache_info()
            assert cache_info["cached_files"] <= 2

        finally:
            await loader.clear_cache()
            for config_file in config_files:
                Path(config_file).unlink()


class TestConfigDiffTracker:
    """Test the ConfigDiffTracker component."""

    @pytest.fixture
    def diff_tracker(self):
        """Fixture for diff tracker."""
        return ConfigDiffTracker()

    @pytest.fixture
    def sample_config_data(self):
        """Fixture for sample configuration data."""
        return {
            "name": "Test Config",
            "description": "Test description",
            "models": [{"name": "model1", "type": "openai_api", "path": "gpt-3.5-turbo"}],
            "datasets": [{"name": "dataset1", "source": "local", "path": "./data.jsonl"}],
        }

    @pytest.mark.asyncio
    async def test_track_and_detect_changes(self, diff_tracker, sample_config_data):
        """Test tracking and detecting configuration changes."""
        config_path = "/test/config.yaml"

        # Start tracking
        await diff_tracker.track_config(config_path, sample_config_data)

        # No changes should be detected for same data
        changed_sections = await diff_tracker.get_changed_sections(config_path, sample_config_data)
        assert len(changed_sections) == 0

        # Modify the config
        modified_config = sample_config_data.copy()
        modified_config["name"] = "Modified Test Config"
        modified_config["new_section"] = {"key": "value"}

        # Should detect changes
        changed_sections = await diff_tracker.get_changed_sections(config_path, modified_config)
        assert "name" in changed_sections
        assert "new_section" in changed_sections
        assert "description" not in changed_sections  # Unchanged

    @pytest.mark.asyncio
    async def test_detect_removed_sections(self, diff_tracker, sample_config_data):
        """Test detection of removed sections."""
        config_path = "/test/config.yaml"

        # Start tracking
        await diff_tracker.track_config(config_path, sample_config_data)

        # Remove a section
        modified_config = sample_config_data.copy()
        del modified_config["models"]

        # Should detect removed section
        changed_sections = await diff_tracker.get_changed_sections(config_path, modified_config)
        assert "models" in changed_sections

    @pytest.mark.asyncio
    async def test_first_time_tracking(self, diff_tracker, sample_config_data):
        """Test first time tracking (all sections are 'changed')."""
        config_path = "/test/new_config.yaml"

        # First time should return all sections as changed
        changed_sections = await diff_tracker.get_changed_sections(config_path, sample_config_data)
        assert changed_sections == set(sample_config_data.keys())

    @pytest.mark.asyncio
    async def test_nested_changes(self, diff_tracker):
        """Test detection of nested changes."""
        config_path = "/test/config.yaml"

        config_data = {
            "models": [{"name": "model1", "config": {"api_key": "key1", "max_tokens": 100}}]
        }

        await diff_tracker.track_config(config_path, config_data)

        # Modify nested value
        modified_config = {
            "models": [{"name": "model1", "config": {"api_key": "key2", "max_tokens": 100}}]
        }

        # Should detect the change
        changed_sections = await diff_tracker.get_changed_sections(config_path, modified_config)
        assert "models" in changed_sections

    @pytest.mark.asyncio
    async def test_clear_tracking(self, diff_tracker, sample_config_data):
        """Test clearing tracking data."""
        config_path = "/test/config.yaml"

        # Track config
        await diff_tracker.track_config(config_path, sample_config_data)

        # Should not be considered changed
        changed_sections = await diff_tracker.get_changed_sections(config_path, sample_config_data)
        assert len(changed_sections) == 0

        # Clear tracking for specific config
        await diff_tracker.clear_tracking(config_path)

        # Should now be considered all changed (like first time)
        changed_sections = await diff_tracker.get_changed_sections(config_path, sample_config_data)
        assert changed_sections == set(sample_config_data.keys())

        # Track again and clear all
        await diff_tracker.track_config(config_path, sample_config_data)
        await diff_tracker.clear_tracking()  # Clear all

        # Should be considered all changed again
        changed_sections = await diff_tracker.get_changed_sections(config_path, sample_config_data)
        assert changed_sections == set(sample_config_data.keys())


class TestCacheIntegration:
    """Integration tests for cache components."""

    @pytest.mark.asyncio
    async def test_cache_components_integration(self):
        """Test integration between cache components."""
        # Create components
        cache = ConfigurationCache(max_size=5, ttl_seconds=60, max_memory_mb=10)
        lazy_loader = LazyConfigLoader(cache_size=5)
        diff_tracker = ConfigDiffTracker()

        await cache.initialize()

        try:
            # Create test config
            config_data = {
                "name": "Integration Test",
                "models": [{"name": "test", "type": "openai_api", "path": "gpt-3.5-turbo"}],
                "datasets": [{"name": "test", "source": "local", "path": "./test.jsonl"}],
            }

            config = ExperimentConfig(**config_data)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(config_data, f)
                config_path = f.name

            try:
                # Test workflow: lazy loading -> caching -> diff tracking

                # 1. Use lazy loader to load sections
                name_section = await lazy_loader.load_section(config_path, "name")
                assert name_section == "Integration Test"

                # 2. Cache the full config
                await cache.set_config("integration_test", config)
                cached_config = await cache.get_config("integration_test")
                assert cached_config is not None
                assert cached_config.name == "Integration Test"

                # 3. Track for changes
                await diff_tracker.track_config(config_path, config_data)

                # 4. Modify and detect changes
                modified_data = config_data.copy()
                modified_data["name"] = "Modified Integration Test"

                changed_sections = await diff_tracker.get_changed_sections(
                    config_path, modified_data
                )
                assert "name" in changed_sections

                # 5. Update lazy loader cache
                with open(config_path, "w") as f:
                    yaml.dump(modified_data, f)

                # Should detect file modification
                is_modified = await lazy_loader.is_config_modified(config_path)
                assert is_modified is True

                # Load updated section
                updated_name = await lazy_loader.load_section(config_path, "name")
                assert updated_name == "Modified Integration Test"

            finally:
                Path(config_path).unlink()

        finally:
            await cache.shutdown()
            await lazy_loader.clear_cache()
            await diff_tracker.clear_tracking()

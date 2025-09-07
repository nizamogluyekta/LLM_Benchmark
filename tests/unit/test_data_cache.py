"""
Unit tests for data cache functionality.

This module tests the data caching system including storage, retrieval,
compression, cache invalidation, and statistics.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from benchmark.data.cache import CacheConfig, CacheStats, DataCache, PartialDataCache
from benchmark.data.models import Dataset, DatasetInfo, DatasetSample


class TestCacheStats:
    """Test cache statistics functionality."""

    def test_cache_stats_initialization(self):
        """Test cache stats initialization."""
        stats = CacheStats()

        assert stats.total_entries == 0
        assert stats.memory_entries == 0
        assert stats.file_entries == 0
        assert stats.memory_usage_mb == 0.0
        assert stats.disk_usage_mb == 0.0
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.eviction_count == 0
        assert stats.last_cleanup is None

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation."""
        stats = CacheStats()

        # No requests yet
        assert stats.hit_ratio == 0.0

        # Add some hits and misses
        stats.hit_count = 75
        stats.miss_count = 25

        assert stats.hit_ratio == 0.75

    def test_hit_ratio_with_zero_requests(self):
        """Test hit ratio when there are no requests."""
        stats = CacheStats()
        stats.hit_count = 0
        stats.miss_count = 0

        assert stats.hit_ratio == 0.0


class TestDataCache:
    """Test the main DataCache functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        samples = [
            DatasetSample(
                id=f"sample_{i}",
                input_text=f"Sample attack pattern {i}",
                label="ATTACK",
                attack_type="malware",
            )
            for i in range(5)
        ]

        info = DatasetInfo(
            name="Test Dataset",
            source="test",
            total_samples=5,
            attack_samples=5,
            benign_samples=0,
            attack_types=["malware"],
        )

        return Dataset(info=info, samples=samples)

    @pytest_asyncio.fixture
    async def cache(self, temp_cache_dir):
        """Create a DataCache instance for testing."""
        cache = DataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=100,  # Small for testing
            max_disk_mb=1000,
            compression_level=6,
            memory_ttl_hours=1,
            disk_ttl_days=1,
        )
        # Wait for metadata loading to complete
        await asyncio.sleep(0.01)
        return cache

    @pytest.mark.asyncio
    async def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = DataCache(cache_dir=temp_cache_dir, max_memory_mb=500, max_disk_mb=5000)

        assert cache.cache_dir == temp_cache_dir
        assert cache.max_memory_bytes == 500 * 1024 * 1024
        assert cache.max_disk_bytes == 5000 * 1024 * 1024
        assert cache.compression_level == 6
        assert len(cache.memory_cache) == 0
        assert len(cache.cache_entries) == 0

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        key1 = cache._generate_cache_key("dataset1", "config_hash_1")
        key2 = cache._generate_cache_key("dataset1", "config_hash_2")
        key3 = cache._generate_cache_key("dataset2", "config_hash_1")

        assert len(key1) == 64  # SHA256 hex string
        assert key1 != key2  # Different configs should generate different keys
        assert key1 != key3  # Different datasets should generate different keys

    @pytest.mark.asyncio
    async def test_dataset_compression_decompression(self, cache, sample_dataset):
        """Test dataset compression and decompression."""
        # Test compression
        compressed_data = await cache._compress_dataset(sample_dataset)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0

        # Test decompression
        decompressed_dataset = await cache._decompress_dataset(compressed_data)

        # Verify data integrity
        assert decompressed_dataset.info.name == sample_dataset.info.name
        assert decompressed_dataset.sample_count == sample_dataset.sample_count
        assert len(decompressed_dataset.samples) == len(sample_dataset.samples)
        assert decompressed_dataset.samples[0].input_text == sample_dataset.samples[0].input_text

    @pytest.mark.asyncio
    async def test_dataset_size_estimation(self, cache, sample_dataset):
        """Test dataset size estimation."""
        size = await cache._estimate_dataset_size(sample_dataset)

        assert isinstance(size, int)
        assert size > 0
        assert size < 10 * 1024 * 1024  # Should be reasonable for our test dataset

    @pytest.mark.asyncio
    async def test_cache_dataset_and_retrieval(self, cache, sample_dataset):
        """Test caching and retrieving a dataset."""
        dataset_id = "test_dataset_001"
        config_hash = "config_abc123"

        # Cache the dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        # Verify it was cached
        assert len(cache.cache_entries) == 1

        # Retrieve the dataset
        cached_dataset = await cache.get_cached_dataset(dataset_id, config_hash)

        assert cached_dataset is not None
        assert cached_dataset.info.name == sample_dataset.info.name
        assert cached_dataset.sample_count == sample_dataset.sample_count

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss scenario."""
        # Try to get a dataset that doesn't exist
        result = await cache.get_cached_dataset("nonexistent", "no_hash")

        assert result is None
        assert cache.stats.miss_count == 1
        assert cache.stats.hit_count == 0

    @pytest.mark.asyncio
    async def test_memory_cache_promotion(self, cache, sample_dataset):
        """Test promotion from disk to memory cache."""
        dataset_id = "test_dataset_002"
        config_hash = "config_def456"

        # Cache the dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        # Clear memory cache to simulate disk-only storage
        cache_key = cache._generate_cache_key(dataset_id, config_hash)
        if cache_key in cache.memory_cache:
            del cache.memory_cache[cache_key]
            cache.cache_entries[cache_key].is_in_memory = False
            cache.memory_usage = 0

        # Retrieve should promote to memory
        cached_dataset = await cache.get_cached_dataset(dataset_id, config_hash)

        assert cached_dataset is not None
        assert cache.cache_entries[cache_key].is_in_memory
        assert cache_key in cache.memory_cache

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache, sample_dataset):
        """Test cache invalidation."""
        dataset_id = "test_dataset_003"
        config_hash1 = "config_ghi789"
        config_hash2 = "config_jkl012"

        # Cache the same dataset with different configs
        await cache.cache_dataset(dataset_id, config_hash1, sample_dataset)
        await cache.cache_dataset(dataset_id, config_hash2, sample_dataset)

        assert len(cache.cache_entries) == 2

        # Invalidate all entries for the dataset
        await cache.invalidate_cache(dataset_id)

        assert len(cache.cache_entries) == 0
        assert len(cache.memory_cache) == 0

    @pytest.mark.asyncio
    async def test_memory_eviction(self, temp_cache_dir, sample_dataset):
        """Test memory cache eviction when limit is reached."""
        # Create cache with very small memory limit
        cache = DataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=1,  # Very small limit
            compression_level=1,  # Low compression for faster testing
        )
        await asyncio.sleep(0.01)  # Wait for initialization

        # Cache multiple datasets to trigger eviction
        datasets_cached = 0
        for i in range(5):
            dataset_id = f"dataset_{i}"
            config_hash = f"config_{i}"

            await cache.cache_dataset(dataset_id, config_hash, sample_dataset)
            datasets_cached += 1

            # Check if eviction occurred
            if cache.stats.eviction_count > 0:
                break

        # Should have triggered at least one eviction
        assert cache.stats.eviction_count > 0
        assert len(cache.memory_cache) < datasets_cached

    @pytest.mark.asyncio
    async def test_cleanup_old_cache(self, cache, sample_dataset):
        """Test cleanup of old cache entries."""
        dataset_id = "test_dataset_004"
        config_hash = "config_mno345"

        # Cache a dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        # Manually set the last accessed time to be old
        cache_key = cache._generate_cache_key(dataset_id, config_hash)
        cache.cache_entries[cache_key].last_accessed = datetime.now() - timedelta(days=10)

        assert len(cache.cache_entries) == 1

        # Run cleanup with max age of 5 days
        await cache.cleanup_old_cache(max_age_days=5)

        # Entry should be removed
        assert len(cache.cache_entries) == 0
        assert cache.stats.last_cleanup is not None

    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache, sample_dataset):
        """Test cache statistics tracking."""
        dataset_id = "test_dataset_005"
        config_hash = "config_pqr678"

        # Initial stats
        stats = await cache.get_cache_stats()
        assert stats.total_entries == 0
        assert stats.hit_count == 0
        assert stats.miss_count == 0

        # Cache a dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        # Get updated stats
        stats = await cache.get_cache_stats()
        assert stats.total_entries == 1
        assert stats.memory_entries >= 0
        assert stats.file_entries >= 0

        # Test cache hit
        await cache.get_cached_dataset(dataset_id, config_hash)
        stats = await cache.get_cache_stats()
        assert stats.hit_count == 1

        # Test cache miss
        await cache.get_cached_dataset("nonexistent", "no_hash")
        stats = await cache.get_cache_stats()
        assert stats.miss_count == 1

    @pytest.mark.asyncio
    async def test_disk_storage_and_retrieval(self, cache, sample_dataset):
        """Test disk storage and retrieval functionality."""
        dataset_id = "test_dataset_006"
        config_hash = "config_stu901"

        # Cache the dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        cache_key = cache._generate_cache_key(dataset_id, config_hash)
        entry = cache.cache_entries[cache_key]

        # Verify file was created
        assert entry.file_path is not None
        assert entry.file_path.exists()

        # Clear memory cache
        cache.memory_cache.clear()
        entry.is_in_memory = False
        cache.memory_usage = 0

        # Should still be able to retrieve from disk
        cached_dataset = await cache.get_cached_dataset(dataset_id, config_hash)
        assert cached_dataset is not None
        assert cached_dataset.info.name == sample_dataset.info.name

    @pytest.mark.asyncio
    async def test_corrupted_cache_handling(self, cache, sample_dataset):
        """Test handling of corrupted cache files."""
        dataset_id = "test_dataset_007"
        config_hash = "config_vwx234"

        # Cache the dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        cache_key = cache._generate_cache_key(dataset_id, config_hash)
        entry = cache.cache_entries[cache_key]

        # Corrupt the cache file
        with open(entry.file_path, "wb") as f:
            f.write(b"corrupted data")

        # Clear memory cache
        cache.memory_cache.clear()
        entry.is_in_memory = False
        cache.memory_usage = 0

        # Should handle corruption gracefully
        cached_dataset = await cache.get_cached_dataset(dataset_id, config_hash)
        assert cached_dataset is None

        # Corrupted entry should be removed
        assert cache_key not in cache.cache_entries

    @pytest.mark.asyncio
    async def test_metadata_persistence(self, temp_cache_dir, sample_dataset):
        """Test cache metadata persistence across restarts."""
        dataset_id = "test_dataset_008"
        config_hash = "config_yzab567"

        # Create first cache instance
        cache1 = DataCache(cache_dir=temp_cache_dir, max_memory_mb=100)
        await asyncio.sleep(0.01)

        await cache1.cache_dataset(dataset_id, config_hash, sample_dataset)
        await cache1._save_cache_metadata()

        # Create second cache instance (simulating restart)
        cache2 = DataCache(cache_dir=temp_cache_dir, max_memory_mb=100)
        await asyncio.sleep(0.01)  # Wait for metadata loading

        # Should have loaded the cache entry
        assert len(cache2.cache_entries) == 1

        # Should be able to retrieve the dataset
        cached_dataset = await cache2.get_cached_dataset(dataset_id, config_hash)
        assert cached_dataset is not None
        assert cached_dataset.info.name == sample_dataset.info.name

    @pytest.mark.asyncio
    async def test_orphaned_file_cleanup(self, cache, sample_dataset):
        """Test cleanup of orphaned cache files."""
        dataset_id = "test_dataset_009"
        config_hash = "config_cdef890"

        # Cache a dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        cache_key = cache._generate_cache_key(dataset_id, config_hash)

        # Create an orphaned file
        orphaned_file = cache.cache_dir / "orphaned_file.cache"
        orphaned_file.write_text("orphaned content")

        # Remove the cache entry but leave the original file
        del cache.cache_entries[cache_key]

        # Run cleanup
        await cache._cleanup_orphaned_files()

        # Orphaned file should be removed
        assert not orphaned_file.exists()
        # Original file should still exist (it's now orphaned too)


class TestPartialDataCache:
    """Test partial data cache functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_split_data(self):
        """Create sample split data for testing."""
        return [
            DatasetSample(
                id=f"train_sample_{i}",
                input_text=f"Training sample {i}",
                label="ATTACK",
                attack_type="malware",
            )
            for i in range(3)
        ]

    @pytest_asyncio.fixture
    async def partial_cache(self, temp_cache_dir):
        """Create a PartialDataCache instance for testing."""
        cache = PartialDataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=100,
            compression_level=1,  # Low compression for faster testing
        )
        await asyncio.sleep(0.01)
        return cache

    @pytest.mark.asyncio
    async def test_cache_dataset_split(self, partial_cache, sample_split_data):
        """Test caching a dataset split."""
        dataset_id = "test_dataset_split"
        split_name = "train"
        config_hash = "split_config_123"
        metadata = {"split_strategy": "random", "seed": 42}

        # Cache the split
        await partial_cache.cache_dataset_split(
            dataset_id, split_name, config_hash, sample_split_data, metadata
        )

        # Should have created a cache entry
        assert len(partial_cache.cache_entries) == 1

    @pytest.mark.asyncio
    async def test_retrieve_cached_split(self, partial_cache, sample_split_data):
        """Test retrieving a cached dataset split."""
        dataset_id = "test_dataset_split_2"
        split_name = "validation"
        config_hash = "split_config_456"
        metadata = {"split_strategy": "stratified"}

        # Cache the split
        await partial_cache.cache_dataset_split(
            dataset_id, split_name, config_hash, sample_split_data, metadata
        )

        # Retrieve the split
        cached_samples = await partial_cache.get_cached_dataset_split(
            dataset_id, split_name, config_hash
        )

        assert cached_samples is not None
        assert len(cached_samples) == len(sample_split_data)
        assert cached_samples[0].input_text == sample_split_data[0].input_text

    @pytest.mark.asyncio
    async def test_split_cache_miss(self, partial_cache):
        """Test split cache miss scenario."""
        result = await partial_cache.get_cached_dataset_split(
            "nonexistent_dataset", "train", "no_hash"
        )

        assert result is None


class TestCacheConfig:
    """Test cache configuration."""

    def test_cache_config_defaults(self):
        """Test cache config default values."""
        config = CacheConfig()

        assert config.cache_dir == "./cache"
        assert config.max_memory_mb == 1000
        assert config.max_disk_mb == 10000
        assert config.compression_level == 6
        assert config.memory_ttl_hours == 24
        assert config.disk_ttl_days == 7
        assert config.enable_compression is True
        assert config.auto_cleanup_enabled is True
        assert config.cleanup_interval_hours == 6

    def test_cache_config_validation(self):
        """Test cache config validation."""
        # Valid config
        config = CacheConfig(max_memory_mb=500, max_disk_mb=5000, compression_level=9)

        assert config.max_memory_mb == 500
        assert config.max_disk_mb == 5000
        assert config.compression_level == 9

    def test_cache_config_custom_values(self):
        """Test cache config with custom values."""
        config = CacheConfig(
            cache_dir="/custom/cache/dir",
            max_memory_mb=2000,
            max_disk_mb=20000,
            compression_level=1,
            memory_ttl_hours=48,
            disk_ttl_days=14,
            enable_compression=False,
            auto_cleanup_enabled=False,
            cleanup_interval_hours=12,
        )

        assert config.cache_dir == "/custom/cache/dir"
        assert config.max_memory_mb == 2000
        assert config.max_disk_mb == 20000
        assert config.compression_level == 1
        assert config.memory_ttl_hours == 48
        assert config.disk_ttl_days == 14
        assert config.enable_compression is False
        assert config.auto_cleanup_enabled is False
        assert config.cleanup_interval_hours == 12


class TestCacheErrorHandling:
    """Test cache error handling and edge cases."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest_asyncio.fixture
    async def cache(self, temp_cache_dir):
        """Create a DataCache instance for testing."""
        cache = DataCache(cache_dir=temp_cache_dir, max_memory_mb=100, compression_level=1)
        await asyncio.sleep(0.01)
        return cache

    @pytest.mark.asyncio
    async def test_invalid_cache_directory(self):
        """Test handling of invalid cache directory."""
        # Try to create cache in non-existent parent directory
        invalid_path = Path("/nonexistent/parent/cache")

        # Should create parent directories
        cache = DataCache(cache_dir=invalid_path)
        assert cache.cache_dir == invalid_path

    @pytest.mark.asyncio
    async def test_compression_error_handling(self, cache):
        """Test handling of compression errors."""
        # Mock compression to raise an error
        with patch.object(cache, "_compress_dataset", side_effect=Exception("Compression failed")):
            from benchmark.data.models import Dataset, DatasetInfo

            sample_dataset = Dataset(
                info=DatasetInfo(
                    name="Test", source="test", total_samples=0, attack_samples=0, benign_samples=0
                ),
                samples=[],
            )

            # Should raise the compression error
            with pytest.raises(Exception, match="Compression failed"):
                await cache.cache_dataset("test", "config", sample_dataset)

    @pytest.mark.asyncio
    async def test_decompression_error_handling(self, cache):
        """Test handling of decompression errors."""
        # Test with invalid compressed data
        invalid_data = b"not compressed data"

        with pytest.raises((Exception, ValueError, OSError)):
            await cache._decompress_dataset(invalid_data)

    @pytest.mark.asyncio
    async def test_disk_write_error_handling(self, cache, sample_dataset):
        """Test handling of disk write errors."""
        # Mock file writing to raise an error
        with (
            patch("builtins.open", side_effect=PermissionError("Permission denied")),
            pytest.raises(PermissionError),
        ):
            await cache.cache_dataset("test", "config", sample_dataset)

    @pytest.mark.asyncio
    async def test_metadata_save_error_handling(self, cache):
        """Test handling of metadata save errors."""
        # Mock json.dump to raise an error
        with patch("json.dump", side_effect=Exception("JSON error")):
            # Should not raise an error, just log a warning
            await cache._save_cache_metadata()

    @pytest.mark.asyncio
    async def test_metadata_load_error_handling(self, cache):
        """Test handling of metadata load errors."""
        # Create invalid metadata file
        metadata_file = cache.cache_dir / "cache_metadata.json"
        metadata_file.write_text("invalid json content")

        # Should not raise an error, just log a warning
        await cache._load_cache_metadata()


class TestCacheConcurrency:
    """Test cache behavior under concurrent access."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_datasets(self):
        """Create multiple sample datasets for concurrent testing."""
        datasets = []
        for i in range(5):
            samples = [
                DatasetSample(
                    id=f"sample_{i}_{j}",
                    input_text=f"Dataset {i} Sample {j}",
                    label="ATTACK",
                    attack_type="malware",
                )
                for j in range(3)
            ]

            info = DatasetInfo(
                name=f"Dataset {i}",
                source="test",
                total_samples=3,
                attack_samples=3,
                benign_samples=0,
                attack_types=["malware"],
            )

            datasets.append(Dataset(info=info, samples=samples))

        return datasets

    @pytest_asyncio.fixture
    async def cache(self, temp_cache_dir):
        """Create a DataCache instance for testing."""
        cache = DataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=100,
            compression_level=1,  # Fast compression for testing
        )
        await asyncio.sleep(0.01)
        return cache

    @pytest.mark.asyncio
    async def test_concurrent_caching(self, cache, sample_datasets):
        """Test concurrent caching operations."""

        async def cache_dataset(i, dataset):
            await cache.cache_dataset(f"dataset_{i}", f"config_{i}", dataset)

        # Run concurrent caching operations
        tasks = [cache_dataset(i, dataset) for i, dataset in enumerate(sample_datasets)]

        await asyncio.gather(*tasks)

        # All datasets should be cached
        assert len(cache.cache_entries) == len(sample_datasets)

    @pytest.mark.asyncio
    async def test_concurrent_retrieval(self, cache, sample_datasets):
        """Test concurrent retrieval operations."""

        # First cache all datasets
        for i, dataset in enumerate(sample_datasets):
            await cache.cache_dataset(f"dataset_{i}", f"config_{i}", dataset)

        async def retrieve_dataset(i):
            return await cache.get_cached_dataset(f"dataset_{i}", f"config_{i}")

        # Run concurrent retrieval operations
        tasks = [retrieve_dataset(i) for i in range(len(sample_datasets))]
        results = await asyncio.gather(*tasks)

        # All retrievals should succeed
        assert all(result is not None for result in results)
        assert len(results) == len(sample_datasets)

    @pytest.mark.asyncio
    async def test_concurrent_invalidation(self, cache, sample_datasets):
        """Test concurrent invalidation operations."""

        # First cache all datasets
        for i, dataset in enumerate(sample_datasets):
            await cache.cache_dataset(f"dataset_{i}", f"config_{i}", dataset)

        async def invalidate_dataset(i):
            await cache.invalidate_cache(f"dataset_{i}")

        # Run concurrent invalidation operations
        tasks = [invalidate_dataset(i) for i in range(len(sample_datasets))]
        await asyncio.gather(*tasks)

        # All cache entries should be removed
        assert len(cache.cache_entries) == 0

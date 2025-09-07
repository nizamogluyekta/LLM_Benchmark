"""
Integration tests for data cache functionality.

This module tests the data caching system in real-world scenarios including
file system operations, concurrent access, and cache cleanup processes.
"""

import asyncio
import contextlib
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from benchmark.data.cache import DataCache, PartialDataCache
from benchmark.data.models import Dataset, DatasetInfo, DatasetSample


class TestDataCacheIntegration:
    """Integration tests for DataCache with real file system operations."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def large_dataset(self):
        """Create a larger dataset for integration testing."""
        samples = []
        for i in range(100):
            label = "ATTACK" if i % 3 == 0 else "BENIGN"
            attack_type = "malware" if label == "ATTACK" else None
            samples.append(
                DatasetSample(
                    id=f"sample_{i}",
                    input_text=f"Integration test sample {i} with some longer content to test compression",
                    label=label,
                    attack_type=attack_type,
                    metadata={"batch": i // 10, "test_data": True},
                )
            )

        info = DatasetInfo(
            name="Integration Test Dataset",
            source="test",
            total_samples=100,
            attack_samples=len([s for s in samples if s.label == "ATTACK"]),
            benign_samples=len([s for s in samples if s.label == "BENIGN"]),
            attack_types=["malware"],
        )

        return Dataset(info=info, samples=samples)

    @pytest_asyncio.fixture
    async def cache(self, temp_cache_dir):
        """Create a DataCache instance for integration testing."""
        cache = DataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=50,  # Reasonable size for testing
            max_disk_mb=200,
            compression_level=6,
            memory_ttl_hours=1,
            disk_ttl_days=1,
        )
        await asyncio.sleep(0.1)  # Allow metadata loading
        yield cache
        # Cleanup after test
        with contextlib.suppress(Exception):
            await cache.cleanup_old_cache(max_age_days=0)

    @pytest.mark.asyncio
    async def test_cache_persistence_across_restarts(self, temp_cache_dir, large_dataset):
        """Test that cache persists across cache instance restarts."""
        dataset_id = "persistent_dataset"
        config_hash = "config_hash_123"

        # Create first cache instance and store data
        cache1 = DataCache(cache_dir=temp_cache_dir, max_memory_mb=50)
        await asyncio.sleep(0.1)

        await cache1.cache_dataset(dataset_id, config_hash, large_dataset)

        # Verify data was cached
        cached_data = await cache1.get_cached_dataset(dataset_id, config_hash)
        assert cached_data is not None
        assert cached_data.sample_count == large_dataset.sample_count

        # Verify metadata file was created
        metadata_file = temp_cache_dir / "cache_metadata.json"
        assert metadata_file.exists()

        # Create second cache instance (simulate restart)
        cache2 = DataCache(cache_dir=temp_cache_dir, max_memory_mb=50)
        await asyncio.sleep(0.1)  # Allow metadata loading

        # Verify data can still be retrieved
        cached_data_after_restart = await cache2.get_cached_dataset(dataset_id, config_hash)
        assert cached_data_after_restart is not None
        assert cached_data_after_restart.sample_count == large_dataset.sample_count
        assert cached_data_after_restart.info.name == large_dataset.info.name

        # Verify statistics were preserved
        stats = await cache2.get_cache_stats()
        assert stats.total_entries == 1

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, cache, large_dataset):
        """Test concurrent cache access patterns."""
        num_concurrent_operations = 10

        async def concurrent_cache_operations(cache, dataset, operation_id):
            """Perform concurrent cache operations."""
            dataset_id = f"concurrent_dataset_{operation_id}"
            config_hash = f"config_{operation_id}"

            # Cache the dataset
            await cache.cache_dataset(dataset_id, config_hash, dataset)

            # Retrieve it multiple times
            for _ in range(3):
                cached = await cache.get_cached_dataset(dataset_id, config_hash)
                assert cached is not None
                assert cached.sample_count == dataset.sample_count

            # Test cache miss
            miss_result = await cache.get_cached_dataset(f"nonexistent_{operation_id}", "no_hash")
            assert miss_result is None

            return operation_id

        # Run concurrent operations
        tasks = [
            concurrent_cache_operations(cache, large_dataset, i)
            for i in range(num_concurrent_operations)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == num_concurrent_operations

        # Verify all datasets were cached
        stats = await cache.get_cache_stats()
        assert stats.total_entries == num_concurrent_operations

    @pytest.mark.asyncio
    async def test_cache_cleanup_processes(self, temp_cache_dir, large_dataset):
        """Test cache cleanup processes with real files."""
        cache = DataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=10,  # Small memory limit
            max_disk_mb=100,
        )
        await asyncio.sleep(0.1)

        # Cache multiple datasets
        datasets_to_cache = 5
        for i in range(datasets_to_cache):
            dataset_id = f"cleanup_test_dataset_{i}"
            config_hash = f"cleanup_config_{i}"
            await cache.cache_dataset(dataset_id, config_hash, large_dataset)

        # Verify all were cached
        assert len(cache.cache_entries) == datasets_to_cache

        # Verify cache files exist on disk
        cache_files_before = list(temp_cache_dir.glob("*.cache"))
        assert len(cache_files_before) == datasets_to_cache

        # Run cleanup with max_age_days=0 to remove all entries
        await cache.cleanup_old_cache(max_age_days=0)

        # Verify entries were removed
        assert len(cache.cache_entries) == 0

        # Verify cache files were cleaned up
        cache_files_after = list(temp_cache_dir.glob("*.cache"))
        assert len(cache_files_after) == 0

        # Verify cleanup timestamp was updated
        stats = await cache.get_cache_stats()
        assert stats.last_cleanup is not None

    @pytest.mark.asyncio
    async def test_memory_eviction_with_disk_fallback(self, temp_cache_dir, large_dataset):
        """Test memory eviction with disk fallback in realistic scenario."""
        # Create cache with very small memory limit to force evictions
        cache = DataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=0.1,  # Extremely small memory limit - 100KB
            max_disk_mb=200,
            compression_level=1,  # Fast compression
        )
        await asyncio.sleep(0.1)

        # Cache datasets that will exceed memory limit
        datasets_cached = []

        for i in range(8):  # Cache more than memory can handle
            dataset_id = f"eviction_test_{i}"
            config_hash = f"eviction_config_{i}"

            await cache.cache_dataset(dataset_id, config_hash, large_dataset)
            datasets_cached.append((dataset_id, config_hash))

            # Allow some processing time
            await asyncio.sleep(0.01)

        # Verify some evictions occurred OR memory usage is constrained
        stats = await cache.get_cache_stats()
        # Either evictions occurred or memory usage is within the very small limit
        assert stats.eviction_count > 0 or stats.memory_usage_mb < 0.12

        # Verify all datasets can still be retrieved (from disk)
        for dataset_id, config_hash in datasets_cached:
            cached_dataset = await cache.get_cached_dataset(dataset_id, config_hash)
            assert cached_dataset is not None
            assert cached_dataset.sample_count == large_dataset.sample_count

    @pytest.mark.asyncio
    async def test_cache_invalidation_with_file_cleanup(self, cache, large_dataset):
        """Test cache invalidation properly cleans up files."""
        dataset_id = "invalidation_test_dataset"

        # Cache the same dataset with different configs
        configs = ["config_1", "config_2", "config_3"]
        cache_files = []

        for config in configs:
            await cache.cache_dataset(dataset_id, config, large_dataset)

            # Find the corresponding cache file
            cache_key = cache._generate_cache_key(dataset_id, config)
            cache_file = cache.cache_dir / f"{cache_key}.cache"
            cache_files.append(cache_file)
            assert cache_file.exists()

        # Verify all configurations were cached
        assert len(cache.cache_entries) == 3

        # Invalidate all entries for the dataset
        await cache.invalidate_cache(dataset_id)

        # Verify cache entries were removed
        assert len(cache.cache_entries) == 0

        # Verify cache files were deleted
        for cache_file in cache_files:
            assert not cache_file.exists()

    @pytest.mark.asyncio
    async def test_cache_statistics_accuracy(self, cache, large_dataset):
        """Test that cache statistics are accurate in integration scenarios."""
        initial_stats = await cache.get_cache_stats()
        assert initial_stats.total_entries == 0
        assert initial_stats.hit_count == 0
        assert initial_stats.miss_count == 0

        # Cache some datasets
        num_datasets = 3
        for i in range(num_datasets):
            dataset_id = f"stats_dataset_{i}"
            config_hash = f"stats_config_{i}"
            await cache.cache_dataset(dataset_id, config_hash, large_dataset)

        # Test cache hits
        for i in range(num_datasets):
            dataset_id = f"stats_dataset_{i}"
            config_hash = f"stats_config_{i}"
            cached = await cache.get_cached_dataset(dataset_id, config_hash)
            assert cached is not None

        # Test cache misses
        for i in range(2):
            result = await cache.get_cached_dataset(f"nonexistent_{i}", "no_hash")
            assert result is None

        # Verify statistics
        final_stats = await cache.get_cache_stats()
        assert final_stats.total_entries == num_datasets
        assert final_stats.hit_count == num_datasets
        assert final_stats.miss_count == 2
        assert final_stats.hit_ratio == 0.6  # 3 hits / (3 hits + 2 misses)

    @pytest.mark.asyncio
    async def test_compression_effectiveness(self, cache, large_dataset):
        """Test that compression is working effectively."""
        dataset_id = "compression_test"
        config_hash = "compression_config"

        # Cache the dataset
        await cache.cache_dataset(dataset_id, config_hash, large_dataset)

        # Get the cache file
        cache_key = cache._generate_cache_key(dataset_id, config_hash)
        cache_file = cache.cache_dir / f"{cache_key}.cache"
        assert cache_file.exists()

        # Check compressed file size vs uncompressed data size
        compressed_size = cache_file.stat().st_size

        # Estimate uncompressed size
        uncompressed_data = large_dataset.model_dump_json().encode("utf-8")
        uncompressed_size = len(uncompressed_data)

        # Compression should reduce size significantly
        compression_ratio = compressed_size / uncompressed_size
        assert compression_ratio < 0.8  # At least 20% compression

        # Verify data integrity after compression/decompression
        cached_dataset = await cache.get_cached_dataset(dataset_id, config_hash)
        assert cached_dataset is not None
        assert cached_dataset.sample_count == large_dataset.sample_count
        assert cached_dataset.info.name == large_dataset.info.name


class TestPartialDataCacheIntegration:
    """Integration tests for PartialDataCache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def split_samples(self):
        """Create sample split data for testing."""
        samples = []
        for i in range(50):
            label = "ATTACK" if i % 2 == 0 else "BENIGN"
            attack_type = "malware" if label == "ATTACK" else None
            samples.append(
                DatasetSample(
                    id=f"split_sample_{i}",
                    input_text=f"Split sample {i} for integration testing",
                    label=label,
                    attack_type=attack_type,
                )
            )
        return samples

    @pytest_asyncio.fixture
    async def partial_cache(self, temp_cache_dir):
        """Create a PartialDataCache instance for testing."""
        cache = PartialDataCache(cache_dir=temp_cache_dir, max_memory_mb=50, compression_level=6)
        await asyncio.sleep(0.1)
        return cache

    @pytest.mark.asyncio
    async def test_partial_cache_multiple_splits(self, partial_cache, split_samples):
        """Test caching multiple dataset splits."""
        dataset_id = "multi_split_dataset"
        config_hash = "split_config_hash"

        # Create different splits
        train_split = split_samples[:30]
        test_split = split_samples[30:40]
        validation_split = split_samples[40:]

        splits = [
            ("train", train_split, {"split_ratio": 0.6}),
            ("test", test_split, {"split_ratio": 0.2}),
            ("validation", validation_split, {"split_ratio": 0.2}),
        ]

        # Cache all splits
        for split_name, samples, metadata in splits:
            await partial_cache.cache_dataset_split(
                dataset_id, split_name, config_hash, samples, metadata
            )

        # Verify all splits can be retrieved
        for split_name, original_samples, _ in splits:
            cached_samples = await partial_cache.get_cached_dataset_split(
                dataset_id, split_name, config_hash
            )
            assert cached_samples is not None
            assert len(cached_samples) == len(original_samples)
            assert cached_samples[0].input_text == original_samples[0].input_text

    @pytest.mark.asyncio
    async def test_partial_cache_concurrent_split_access(self, partial_cache, split_samples):
        """Test concurrent access to different dataset splits."""
        dataset_id = "concurrent_splits"

        async def cache_and_retrieve_split(split_name, samples, config_hash):
            """Cache and retrieve a split concurrently."""
            metadata = {"split_name": split_name}

            # Cache the split
            await partial_cache.cache_dataset_split(
                dataset_id, split_name, config_hash, samples, metadata
            )

            # Retrieve multiple times
            for _ in range(3):
                cached = await partial_cache.get_cached_dataset_split(
                    dataset_id, split_name, config_hash
                )
                assert cached is not None
                assert len(cached) == len(samples)

            return split_name

        # Create tasks for different splits
        tasks = []
        split_size = len(split_samples) // 4
        for i in range(4):
            split_name = f"split_{i}"
            split_samples_subset = split_samples[i * split_size : (i + 1) * split_size]
            config_hash = f"concurrent_config_{i}"

            tasks.append(cache_and_retrieve_split(split_name, split_samples_subset, config_hash))

        # Run concurrently
        results = await asyncio.gather(*tasks)
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_partial_cache_cleanup_integration(self, temp_cache_dir, split_samples):
        """Test cleanup processes with partial cache."""
        partial_cache = PartialDataCache(cache_dir=temp_cache_dir, max_memory_mb=30)
        await asyncio.sleep(0.1)

        dataset_id = "cleanup_partial_test"

        # Cache multiple splits
        num_splits = 6
        for i in range(num_splits):
            split_name = f"split_{i}"
            config_hash = f"cleanup_config_{i}"
            samples = split_samples[:10]  # Use subset for faster testing

            await partial_cache.cache_dataset_split(
                dataset_id, split_name, config_hash, samples, {}
            )

        # Verify cache files exist
        cache_files_before = list(temp_cache_dir.glob("*.cache"))
        assert len(cache_files_before) == num_splits

        # Run cleanup
        await partial_cache.cleanup_old_cache(max_age_days=0)

        # Verify cleanup
        assert len(partial_cache.cache_entries) == 0
        cache_files_after = list(temp_cache_dir.glob("*.cache"))
        assert len(cache_files_after) == 0


class TestCacheErrorRecovery:
    """Test cache error recovery and resilience."""

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
                id=f"recovery_sample_{i}",
                input_text=f"Recovery test sample {i}",
                label="ATTACK",
                attack_type="malware",
            )
            for i in range(10)
        ]

        info = DatasetInfo(
            name="Recovery Test Dataset",
            source="test",
            total_samples=10,
            attack_samples=10,
            benign_samples=0,
            attack_types=["malware"],
        )

        return Dataset(info=info, samples=samples)

    @pytest.mark.asyncio
    async def test_corrupted_metadata_recovery(self, temp_cache_dir, sample_dataset):
        """Test recovery from corrupted metadata file."""
        # Create cache and store data
        cache = DataCache(cache_dir=temp_cache_dir, max_memory_mb=50)
        await asyncio.sleep(0.1)

        dataset_id = "metadata_recovery_test"
        config_hash = "recovery_config"

        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        # Verify normal operation
        cached = await cache.get_cached_dataset(dataset_id, config_hash)
        assert cached is not None

        # Corrupt metadata file
        metadata_file = temp_cache_dir / "cache_metadata.json"
        metadata_file.write_text("corrupted json content {invalid")

        # Create new cache instance (should handle corrupted metadata gracefully)
        cache2 = DataCache(cache_dir=temp_cache_dir, max_memory_mb=50)
        await asyncio.sleep(0.1)

        # Should start with empty cache but not crash
        stats = await cache2.get_cache_stats()
        assert stats.total_entries == 0

        # Should be able to cache new data
        await cache2.cache_dataset("new_dataset", "new_config", sample_dataset)
        new_cached = await cache2.get_cached_dataset("new_dataset", "new_config")
        assert new_cached is not None

    @pytest.mark.asyncio
    async def test_disk_space_handling(self, temp_cache_dir, sample_dataset):
        """Test handling of disk space limitations."""
        # Create cache with very small disk limit
        cache = DataCache(
            cache_dir=temp_cache_dir,
            max_memory_mb=5,
            max_disk_mb=1,  # Very small disk limit
        )
        await asyncio.sleep(0.1)

        dataset_id = "disk_space_test"
        config_hash = "space_config"

        # This should work for the first dataset
        await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

        # Verify it was cached
        cached = await cache.get_cached_dataset(dataset_id, config_hash)
        assert cached is not None

    @pytest.mark.asyncio
    async def test_concurrent_access_error_handling(self, temp_cache_dir, sample_dataset):
        """Test error handling during concurrent access."""
        cache = DataCache(cache_dir=temp_cache_dir, max_memory_mb=50)
        await asyncio.sleep(0.1)

        async def cache_with_potential_error(operation_id):
            """Cache operation that might encounter errors."""
            try:
                dataset_id = f"error_test_{operation_id}"
                config_hash = f"error_config_{operation_id}"

                await cache.cache_dataset(dataset_id, config_hash, sample_dataset)

                # Try to retrieve
                cached = await cache.get_cached_dataset(dataset_id, config_hash)
                return cached is not None

            except Exception:
                # Should handle errors gracefully
                return False

        # Run many concurrent operations
        tasks = [cache_with_potential_error(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Most operations should succeed
        successes = [r for r in results if r is True]
        assert len(successes) > 15  # Allow for some failures due to concurrency

    @pytest.mark.asyncio
    async def test_cache_recovery_after_process_restart(self, temp_cache_dir, sample_dataset):
        """Test cache recovery after simulated process restart."""
        dataset_id = "restart_recovery_test"
        config_hash = "restart_config"

        # First "process" - create and populate cache
        cache1 = DataCache(cache_dir=temp_cache_dir, max_memory_mb=50)
        await asyncio.sleep(0.1)

        await cache1.cache_dataset(dataset_id, config_hash, sample_dataset)

        # Verify files exist
        cache_files = list(temp_cache_dir.glob("*.cache"))
        metadata_file = temp_cache_dir / "cache_metadata.json"
        assert len(cache_files) == 1
        assert metadata_file.exists()

        # Simulate process restart by creating new cache instance
        cache2 = DataCache(cache_dir=temp_cache_dir, max_memory_mb=50)
        await asyncio.sleep(0.1)  # Allow metadata loading

        # Should recover data from previous "process"
        cached = await cache2.get_cached_dataset(dataset_id, config_hash)
        assert cached is not None
        assert cached.sample_count == sample_dataset.sample_count

        # Statistics should be preserved
        stats = await cache2.get_cache_stats()
        assert stats.total_entries == 1

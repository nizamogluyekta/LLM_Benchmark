"""
Performance tests for the optimized Data Service.

This module tests the performance improvements including streaming data loading,
memory usage monitoring, data compression, and hardware-specific optimizations.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio

from benchmark.core.config import DatasetConfig
from benchmark.services.data_service import DataService


class TestDataServicePerformance:
    """Performance tests for the optimized Data Service."""

    @pytest_asyncio.fixture
    async def optimized_data_service(self):
        """Create an optimized data service instance for testing."""
        service = DataService(
            cache_max_size=20,
            cache_max_memory_mb=128,
            cache_ttl=300,
            enable_compression=True,
            enable_hardware_optimization=True,
        )
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest_asyncio.fixture
    async def standard_data_service(self):
        """Create a standard data service instance for comparison."""
        service = DataService(
            cache_max_size=20,
            cache_max_memory_mb=128,
            cache_ttl=300,
            enable_compression=False,
            enable_hardware_optimization=False,
        )
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.fixture
    def large_dataset_file(self):
        """Create a large dataset file for performance testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create a dataset with 5000 samples
        large_data = []
        for i in range(5000):
            sample = {
                "text": f"Sample {i} - " + "This is some longer text content " * 10,
                "label": "ATTACK" if i % 3 == 0 else "BENIGN",
                "attack_type": "phishing" if i % 3 == 0 else None,
                "metadata": {"sample_id": i, "complexity": i % 5, "source": "performance_test"},
            }
            large_data.append(sample)

        file_path = temp_dir / "large_dataset.json"
        with open(file_path, "w") as f:
            json.dump(large_data, f)

        return {"file_path": file_path, "expected_samples": len(large_data), "temp_dir": temp_dir}

    @pytest.mark.asyncio
    async def test_hardware_optimization_initialization(self, optimized_data_service):
        """Test that hardware optimizations are properly initialized."""
        # Check that optimization components are initialized
        assert optimized_data_service.memory_manager is not None
        assert optimized_data_service.optimizer is not None
        assert optimized_data_service._optimization_settings

        # Check optimization settings
        settings = optimized_data_service._optimization_settings
        assert "optimal_batch_size" in settings
        assert "max_concurrent_batches" in settings
        assert settings["optimal_batch_size"] >= 32

        # Verify hardware detection
        health = await optimized_data_service.health_check()
        assert "memory_status" in health.checks
        memory_status = health.checks["memory_status"]
        assert "total_gb" in memory_status
        assert "available_gb" in memory_status

    @pytest.mark.asyncio
    async def test_compressed_cache_performance(self, optimized_data_service, large_dataset_file):
        """Test compressed cache performance and memory savings."""
        config = DatasetConfig(
            name="compression_test",
            path=str(large_dataset_file["file_path"]),
            source="local",
            format="json",
        )

        start_time = time.time()
        dataset = await optimized_data_service.load_dataset(config)
        load_time = time.time() - start_time

        # Verify dataset loaded correctly
        assert dataset.size == large_dataset_file["expected_samples"]

        # Check cache performance stats
        cache_stats = await optimized_data_service.get_cache_performance_stats()

        # Verify compression is enabled and working
        assert cache_stats["advanced_cache"] is True
        assert cache_stats["compression_enabled"] is True

        if "compression_ratio" in cache_stats:
            # Compression should provide some space savings
            assert cache_stats["compression_ratio"] > 1.0
            assert cache_stats["space_saved_bytes"] > 0

        print(f"Load time: {load_time:.3f}s")
        print(f"Cache stats: {cache_stats}")

    @pytest.mark.asyncio
    async def test_memory_monitoring_and_cleanup(self, optimized_data_service, large_dataset_file):
        """Test memory monitoring and cleanup functionality."""
        config = DatasetConfig(
            name="memory_test",
            path=str(large_dataset_file["file_path"]),
            source="local",
            format="json",
        )

        # Load dataset
        await optimized_data_service.load_dataset(config)

        # Check memory status
        memory_status = await optimized_data_service.get_memory_status()
        assert "memory_status" in memory_status
        assert "cache_info" in memory_status
        assert "memory_pressure" in memory_status

        initial_memory = memory_status["memory_status"]["process_memory_mb"]

        # Perform memory cleanup
        cleanup_stats = await optimized_data_service.cleanup_memory()
        assert "initial_memory_mb" in cleanup_stats
        assert "final_memory_mb" in cleanup_stats

        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Cleanup stats: {cleanup_stats}")

    @pytest.mark.asyncio
    async def test_streaming_batch_performance(self, optimized_data_service, large_dataset_file):
        """Test streaming batch performance for large datasets."""
        config = DatasetConfig(
            name="streaming_test",
            path=str(large_dataset_file["file_path"]),
            source="local",
            format="json",
        )

        # Track progress
        progress_reports = []

        async def progress_callback(progress):
            progress_reports.append(
                {
                    "current": progress.current,
                    "total": progress.total,
                    "percentage": progress.percentage,
                }
            )

        # Stream dataset in batches
        start_time = time.time()
        batch_count = 0
        sample_count = 0

        async for batch in optimized_data_service.stream_dataset_batches(
            config, batch_size=100, progress_callback=progress_callback
        ):
            batch_count += 1
            sample_count += len(batch.samples)

            # Verify batch properties
            assert hasattr(batch, "batch_id")
            assert hasattr(batch, "samples")
            assert len(batch.samples) <= 100

        streaming_time = time.time() - start_time

        # Verify all samples were processed
        assert sample_count == large_dataset_file["expected_samples"]

        # Verify progress reporting
        assert len(progress_reports) > 0
        final_progress = progress_reports[-1]
        assert final_progress["percentage"] == 100.0

        print(f"Streaming time: {streaming_time:.3f}s")
        print(f"Batches processed: {batch_count}")
        print(f"Samples processed: {sample_count}")

    @pytest.mark.asyncio
    async def test_optimized_vs_standard_performance(
        self, optimized_data_service, standard_data_service, large_dataset_file
    ):
        """Compare optimized vs standard data service performance."""
        config = DatasetConfig(
            name="benchmark_test",
            path=str(large_dataset_file["file_path"]),
            source="local",
            format="json",
        )

        # Test standard service
        start_time = time.time()
        standard_dataset = await standard_data_service.load_dataset(config)
        standard_load_time = time.time() - start_time
        standard_cache_stats = await standard_data_service.get_cache_stats()

        # Test optimized service
        start_time = time.time()
        optimized_dataset = await optimized_data_service.load_dataset(config)
        optimized_load_time = time.time() - start_time
        optimized_cache_stats = await optimized_data_service.get_cache_performance_stats()

        # Verify both loaded the same data
        assert standard_dataset.size == optimized_dataset.size
        assert standard_dataset.size == large_dataset_file["expected_samples"]

        # Compare memory usage
        standard_memory = standard_cache_stats["memory_usage_mb"]
        optimized_memory = optimized_cache_stats["memory_usage_mb"]

        print(f"Standard load time: {standard_load_time:.3f}s")
        print(f"Optimized load time: {optimized_load_time:.3f}s")
        print(f"Standard memory usage: {standard_memory:.2f}MB")
        print(f"Optimized memory usage: {optimized_memory:.2f}MB")

        # Log performance comparison
        if optimized_memory < standard_memory:
            memory_savings = ((standard_memory - optimized_memory) / standard_memory) * 100
            print(f"Memory savings: {memory_savings:.1f}%")

    @pytest.mark.asyncio
    async def test_concurrent_streaming_performance(
        self, optimized_data_service, large_dataset_file
    ):
        """Test concurrent streaming performance."""
        configs = []
        for i in range(3):  # Create 3 concurrent streams
            config = DatasetConfig(
                name=f"concurrent_stream_{i}",
                path=str(large_dataset_file["file_path"]),
                source="local",
                format="json",
            )
            configs.append(config)

        # Stream multiple datasets concurrently
        start_time = time.time()

        async def stream_dataset(config):
            batch_count = 0
            async for _batch in optimized_data_service.stream_dataset_batches(
                config, batch_size=50
            ):
                batch_count += 1
            return batch_count

        # Run concurrent streaming
        batch_counts = await asyncio.gather(*[stream_dataset(config) for config in configs])
        concurrent_time = time.time() - start_time

        # Verify all streams completed
        assert len(batch_counts) == 3
        assert all(count > 0 for count in batch_counts)

        print(f"Concurrent streaming time: {concurrent_time:.3f}s")
        print(f"Batch counts per stream: {batch_counts}")

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, optimized_data_service, large_dataset_file):
        """Test memory pressure detection and handling."""
        # Load multiple datasets to increase memory pressure
        configs = []
        for i in range(5):
            config = DatasetConfig(
                name=f"pressure_test_{i}",
                path=str(large_dataset_file["file_path"]),
                source="local",
                format="json",
            )
            configs.append(config)

        # Load datasets
        for config in configs:
            await optimized_data_service.load_dataset(config)

        # Check memory pressure
        memory_status = await optimized_data_service.get_memory_status()
        memory_pressure = memory_status["memory_pressure"]

        # Perform cleanup if under pressure
        if memory_pressure:
            cleanup_stats = await optimized_data_service.cleanup_memory()
            assert cleanup_stats["cleaned_datasets"] >= 0

        print(f"Memory pressure detected: {memory_pressure}")
        print(f"Memory status: {memory_status['memory_status']}")

    @pytest.mark.asyncio
    async def test_optimized_batch_sizing(self, optimized_data_service, large_dataset_file):
        """Test that optimized batch sizing is working correctly."""
        config = DatasetConfig(
            name="batch_size_test",
            path=str(large_dataset_file["file_path"]),
            source="local",
            format="json",
        )

        # Load dataset first
        await optimized_data_service.load_dataset(config)

        # Get optimized batch (should use hardware-optimized batch size)
        optimized_batch = await optimized_data_service.get_optimized_batch("batch_size_test")

        # Verify batch uses optimized settings
        optimal_size = optimized_data_service._optimization_settings.get("optimal_batch_size", 64)
        assert len(optimized_batch.samples) <= optimal_size

        print(f"Optimal batch size: {optimal_size}")
        print(f"Actual batch size: {len(optimized_batch.samples)}")

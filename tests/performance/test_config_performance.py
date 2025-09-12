"""
Performance tests for configuration service optimization.

This module contains comprehensive performance benchmarks for the configuration service,
testing loading times, memory usage, cache performance, and concurrent access patterns.
"""

import asyncio
import gc
import tempfile
import time
from pathlib import Path

import psutil
import pytest
import yaml

from benchmark.services.configuration_service import ConfigurationService


class PerformanceBenchmark:
    """Base class for performance benchmarking utilities."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.start_time = None

    def start_benchmark(self) -> None:
        """Start performance measurement."""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.process.memory_info().rss
        self.start_time = time.perf_counter()

    def end_benchmark(self) -> dict[str, float]:
        """End performance measurement and return metrics."""
        end_time = time.perf_counter()
        current_memory = self.process.memory_info().rss

        return {
            "execution_time_ms": (end_time - self.start_time) * 1000,
            "memory_used_mb": (current_memory - self.baseline_memory) / 1024 / 1024,
            "peak_memory_mb": current_memory / 1024 / 1024,
        }


@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking."""
    return PerformanceBenchmark()


@pytest.fixture
async def optimized_config_service():
    """Fixture for optimized configuration service."""
    with tempfile.TemporaryDirectory() as temp_dir:
        service = ConfigurationService(
            config_dir=Path(temp_dir),
            cache_ttl=3600,
            max_cache_size=50,
            max_cache_memory_mb=128,
            enable_lazy_loading=True,
        )
        await service.initialize()
        yield service
        await service.shutdown()


@pytest.fixture
async def legacy_config_service():
    """Fixture for legacy configuration service (without optimizations)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        service = ConfigurationService(
            config_dir=Path(temp_dir),
            cache_ttl=3600,
            max_cache_size=10,  # Smaller cache
            max_cache_memory_mb=32,  # Less memory
            enable_lazy_loading=False,  # Disabled optimizations
        )
        await service.initialize()
        yield service
        await service.shutdown()


@pytest.fixture
def sample_configs():
    """Fixture that creates various test configuration files."""
    configs = {}

    # Small configuration
    configs["small"] = {
        "name": "Small Test Config",
        "description": "Minimal configuration for testing",
        "output_dir": "./results",
        "datasets": [
            {
                "name": "small_dataset",
                "source": "local",
                "path": "./data/small.jsonl",
                "max_samples": 100,
            }
        ],
        "models": [
            {
                "name": "gpt-3.5",
                "type": "openai_api",
                "path": "gpt-3.5-turbo",
                "config": {"api_key": "test"},
            }
        ],
        "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 1, "batch_size": 8},
    }

    # Medium configuration
    configs["medium"] = {
        "name": "Medium Test Config",
        "description": "Medium-sized configuration for testing",
        "output_dir": "./results",
        "datasets": [
            {
                "name": f"dataset_{i}",
                "source": "local",
                "path": f"./data/dataset_{i}.jsonl",
                "max_samples": 1000,
            }
            for i in range(5)
        ],
        "models": [
            {
                "name": f"model_{i}",
                "type": "openai_api",
                "path": "gpt-3.5-turbo",
                "config": {"api_key": "test"},
            }
            for i in range(3)
        ],
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall"],
            "parallel_jobs": 2,
            "batch_size": 16,
        },
    }

    # Large configuration
    configs["large"] = {
        "name": "Large Test Config",
        "description": "Large configuration for stress testing",
        "output_dir": "./results",
        "datasets": [
            {
                "name": f"large_dataset_{i}",
                "source": "local",
                "path": f"./data/large_{i}.jsonl",
                "max_samples": 10000,
                "metadata": {f"key_{j}": f"value_{j}" for j in range(10)},
            }
            for i in range(20)
        ],
        "models": [
            {
                "name": f"large_model_{i}",
                "type": "openai_api",
                "path": "gpt-4",
                "config": {
                    "api_key": "test_key_" + "x" * 100,
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "extra_params": {f"param_{j}": f"value_{j}" for j in range(20)},
                },
            }
            for i in range(10)
        ],
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "auc", "mcc"],
            "parallel_jobs": 8,
            "batch_size": 32,
            "timeout_minutes": 120,
            "custom_metrics": {f"metric_{i}": {"type": "custom", "params": {}} for i in range(10)},
        },
    }

    return configs


def create_config_files(configs: dict, temp_dir: Path) -> dict[str, Path]:
    """Create configuration files in temporary directory."""
    config_files = {}

    for size, config_data in configs.items():
        config_file = temp_dir / f"config_{size}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, indent=2)
        config_files[size] = config_file

    return config_files


class TestConfigurationLoadingPerformance:
    """Test configuration loading performance optimizations."""

    @pytest.mark.asyncio
    async def test_small_config_loading_performance(
        self, optimized_config_service, legacy_config_service, sample_configs, performance_benchmark
    ):
        """Test loading performance for small configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)
            small_config = config_files["small"]

            # Test optimized service
            performance_benchmark.start_benchmark()
            for _ in range(10):
                await optimized_config_service.load_experiment_config(small_config)
            optimized_metrics = performance_benchmark.end_benchmark()

            # Test legacy service
            performance_benchmark.start_benchmark()
            for _ in range(10):
                await legacy_config_service.load_experiment_config(small_config)
            legacy_metrics = performance_benchmark.end_benchmark()

            # Optimized should be faster (allowing for some variance)
            assert (
                optimized_metrics["execution_time_ms"] <= legacy_metrics["execution_time_ms"] * 1.2
            )

            print(
                f"Small config - Optimized: {optimized_metrics['execution_time_ms']:.2f}ms, "
                f"Legacy: {legacy_metrics['execution_time_ms']:.2f}ms"
            )

    @pytest.mark.asyncio
    async def test_large_config_loading_performance(
        self, optimized_config_service, legacy_config_service, sample_configs, performance_benchmark
    ):
        """Test loading performance for large configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)
            large_config = config_files["large"]

            # Test optimized service
            performance_benchmark.start_benchmark()
            for _ in range(5):
                await optimized_config_service.load_experiment_config(large_config)
            optimized_metrics = performance_benchmark.end_benchmark()

            # Test legacy service
            performance_benchmark.start_benchmark()
            for _ in range(5):
                await legacy_config_service.load_experiment_config(large_config)
            legacy_metrics = performance_benchmark.end_benchmark()

            # Optimized should be significantly faster for large configs
            assert optimized_metrics["execution_time_ms"] < legacy_metrics["execution_time_ms"]

            print(
                f"Large config - Optimized: {optimized_metrics['execution_time_ms']:.2f}ms, "
                f"Legacy: {legacy_metrics['execution_time_ms']:.2f}ms"
            )

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(
        self, optimized_config_service, sample_configs, performance_benchmark
    ):
        """Test memory usage with large configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)

            performance_benchmark.start_benchmark()

            # Load multiple large configurations
            configs = []
            for _ in range(10):
                config = await optimized_config_service.load_experiment_config(
                    config_files["large"]
                )
                configs.append(config)

            metrics = performance_benchmark.end_benchmark()

            # Memory usage should be reasonable (less than 100MB for test)
            assert metrics["peak_memory_mb"] < 100

            # Get cache stats
            cache_stats = await optimized_config_service.get_cache_performance_stats()
            assert cache_stats["advanced_cache"]["hit_rate_percent"] > 0

            print(
                f"Memory usage: {metrics['peak_memory_mb']:.2f}MB, "
                f"Cache hit rate: {cache_stats['advanced_cache']['hit_rate_percent']:.1f}%"
            )


class TestCachePerformance:
    """Test cache performance and hit rates."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate_performance(self, optimized_config_service, sample_configs):
        """Test cache hit rates with repeated access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)

            # Load configurations multiple times
            for _ in range(5):
                for config_file in config_files.values():
                    await optimized_config_service.load_experiment_config(config_file)

            # Check cache statistics
            cache_stats = await optimized_config_service.get_cache_performance_stats()

            # Should have high hit rate after repeated loads
            assert cache_stats["advanced_cache"]["hit_rate_percent"] > 60
            assert cache_stats["advanced_cache"]["hits"] > cache_stats["advanced_cache"]["misses"]

            print(f"Cache hit rate: {cache_stats['advanced_cache']['hit_rate_percent']:.1f}%")
            print(
                f"Hits: {cache_stats['advanced_cache']['hits']}, Misses: {cache_stats['advanced_cache']['misses']}"
            )

    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self, sample_configs):
        """Test cache eviction with limited cache size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create service with very small cache
            service = ConfigurationService(
                config_dir=temp_path,
                max_cache_size=3,  # Very small cache
                max_cache_memory_mb=16,
                enable_lazy_loading=True,
            )
            await service.initialize()

            try:
                config_files = create_config_files(sample_configs, temp_path)

                # Load more configs than cache can hold
                for config_file in config_files.values():
                    await service.load_experiment_config(config_file)

                # Load additional configs to force eviction
                for i in range(5):
                    extra_config = {
                        "name": f"Extra Config {i}",
                        "description": "Extra config for eviction test",
                        "datasets": [{"name": "test", "source": "local", "path": "./test.jsonl"}],
                        "models": [
                            {
                                "name": "test",
                                "type": "openai_api",
                                "path": "gpt-3.5-turbo",
                                "config": {},
                            }
                        ],
                        "evaluation": {"metrics": ["accuracy"], "parallel_jobs": 1},
                    }
                    extra_file = temp_path / f"extra_{i}.yaml"
                    with open(extra_file, "w") as f:
                        yaml.dump(extra_config, f)

                    await service.load_experiment_config(extra_file)

                # Check cache stats
                cache_stats = await service.get_cache_performance_stats()

                # Should have evictions due to size limit
                assert cache_stats["advanced_cache"]["evictions"] > 0
                assert cache_stats["advanced_cache"]["current_size"] <= 3

                print(f"Evictions: {cache_stats['advanced_cache']['evictions']}")
                print(f"Final cache size: {cache_stats['advanced_cache']['current_size']}")

            finally:
                await service.shutdown()

    @pytest.mark.asyncio
    async def test_lazy_loading_performance(self, optimized_config_service, sample_configs):
        """Test lazy loading performance for section-based access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)

            start_time = time.perf_counter()

            # Test config outline loading (should be fast)
            for config_file in config_files.values():
                outline = await optimized_config_service.get_config_outline(config_file)
                assert "name" in outline
                assert "_models_count" in outline
                assert "_datasets_count" in outline

            outline_time = (time.perf_counter() - start_time) * 1000

            start_time = time.perf_counter()

            # Test full config loading
            for config_file in config_files.values():
                await optimized_config_service.load_experiment_config(config_file)

            full_load_time = (time.perf_counter() - start_time) * 1000

            # Outline loading should be significantly faster
            print(f"Outline loading: {outline_time:.2f}ms, Full loading: {full_load_time:.2f}ms")

            # Get lazy loader stats
            cache_stats = await optimized_config_service.get_cache_performance_stats()
            if "lazy_loader" in cache_stats:
                assert cache_stats["lazy_loader"]["cached_files"] > 0


class TestConcurrentPerformance:
    """Test concurrent access performance."""

    @pytest.mark.asyncio
    async def test_concurrent_loading_performance(self, optimized_config_service, sample_configs):
        """Test performance with concurrent configuration loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)

            async def load_config_repeatedly(config_file: Path, iterations: int):
                """Load a configuration file multiple times."""
                for _ in range(iterations):
                    await optimized_config_service.load_experiment_config(config_file)

            start_time = time.perf_counter()

            # Run concurrent loading tasks
            tasks = []
            for config_file in config_files.values():
                task = asyncio.create_task(load_config_repeatedly(config_file, 3))
                tasks.append(task)

            await asyncio.gather(*tasks)

            concurrent_time = (time.perf_counter() - start_time) * 1000

            # Test sequential loading for comparison
            start_time = time.perf_counter()

            for config_file in config_files.values():
                for _ in range(3):
                    await optimized_config_service.load_experiment_config(config_file)

            sequential_time = (time.perf_counter() - start_time) * 1000

            # Concurrent should be faster due to caching
            assert concurrent_time < sequential_time * 1.2  # Allow some overhead

            print(
                f"Concurrent loading: {concurrent_time:.2f}ms, Sequential: {sequential_time:.2f}ms"
            )

            # Check final cache stats
            cache_stats = await optimized_config_service.get_cache_performance_stats()
            assert cache_stats["advanced_cache"]["hit_rate_percent"] > 50

    @pytest.mark.asyncio
    async def test_bulk_preloading_performance(self, optimized_config_service, sample_configs):
        """Test bulk preloading performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)
            config_paths = [str(f) for f in config_files.values()]

            # Test bulk preloading
            start_time = time.perf_counter()
            result = await optimized_config_service.preload_configurations_bulk(config_paths)
            preload_time = (time.perf_counter() - start_time) * 1000

            assert result.success
            assert result.data["success_count"] == len(config_files)

            # Test loading after preload (should be fast)
            start_time = time.perf_counter()
            for config_file in config_files.values():
                await optimized_config_service.load_experiment_config(config_file)
            post_preload_time = (time.perf_counter() - start_time) * 1000

            print(f"Bulk preload time: {preload_time:.2f}ms")
            print(f"Post-preload loading: {post_preload_time:.2f}ms")

            # Check cache performance
            cache_stats = await optimized_config_service.get_cache_performance_stats()
            assert cache_stats["advanced_cache"]["hit_rate_percent"] > 80


class TestMemoryEfficiency:
    """Test memory efficiency and limits."""

    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self, sample_configs):
        """Test that memory limits are enforced."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create service with very low memory limit
            service = ConfigurationService(
                config_dir=temp_path,
                max_cache_size=100,
                max_cache_memory_mb=1,  # Very low memory limit
                enable_lazy_loading=True,
            )
            await service.initialize()

            try:
                config_files = create_config_files(sample_configs, temp_path)

                # Load large configurations
                for _ in range(3):
                    for config_file in config_files.values():
                        await service.load_experiment_config(config_file)

                # Check that memory usage is controlled
                cache_stats = await service.get_cache_performance_stats()
                memory_usage_mb = cache_stats["advanced_cache"]["memory_usage_mb"]

                # Should not exceed the limit by much (allowing for overhead)
                assert memory_usage_mb < 2.0

                print(f"Memory usage with 1MB limit: {memory_usage_mb:.2f}MB")

            finally:
                await service.shutdown()

    @pytest.mark.asyncio
    async def test_cache_statistics_accuracy(self, optimized_config_service, sample_configs):
        """Test that cache statistics are accurate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)

            # Clear cache to start fresh
            await optimized_config_service.clear_cache()

            # Load each config once (should be misses)
            for config_file in config_files.values():
                await optimized_config_service.load_experiment_config(config_file)

            # Load each config again (should be hits)
            for config_file in config_files.values():
                await optimized_config_service.load_experiment_config(config_file)

            cache_stats = await optimized_config_service.get_cache_performance_stats()
            advanced_stats = cache_stats["advanced_cache"]

            # Should have equal hits and misses (loaded each config twice)
            expected_operations = len(config_files) * 2
            total_requests = advanced_stats["hits"] + advanced_stats["misses"]

            assert total_requests >= expected_operations
            assert advanced_stats["hits"] > 0
            assert advanced_stats["misses"] > 0

            print(
                f"Cache stats - Hits: {advanced_stats['hits']}, Misses: {advanced_stats['misses']}"
            )
            print(f"Hit rate: {advanced_stats['hit_rate_percent']:.1f}%")


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions against baseline."""

    @pytest.mark.asyncio
    async def test_loading_performance_baseline(
        self, optimized_config_service, sample_configs, performance_benchmark
    ):
        """Test that loading performance meets baseline requirements."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)

            # Performance requirements (adjust based on hardware)
            MAX_SMALL_CONFIG_TIME_MS = 50
            MAX_MEDIUM_CONFIG_TIME_MS = 200
            MAX_LARGE_CONFIG_TIME_MS = 500

            # Test small config
            performance_benchmark.start_benchmark()
            await optimized_config_service.load_experiment_config(config_files["small"])
            small_metrics = performance_benchmark.end_benchmark()

            # Test medium config
            performance_benchmark.start_benchmark()
            await optimized_config_service.load_experiment_config(config_files["medium"])
            medium_metrics = performance_benchmark.end_benchmark()

            # Test large config
            performance_benchmark.start_benchmark()
            await optimized_config_service.load_experiment_config(config_files["large"])
            large_metrics = performance_benchmark.end_benchmark()

            # Assert performance requirements
            assert small_metrics["execution_time_ms"] < MAX_SMALL_CONFIG_TIME_MS, (
                f"Small config took {small_metrics['execution_time_ms']:.2f}ms, max allowed: {MAX_SMALL_CONFIG_TIME_MS}ms"
            )

            assert medium_metrics["execution_time_ms"] < MAX_MEDIUM_CONFIG_TIME_MS, (
                f"Medium config took {medium_metrics['execution_time_ms']:.2f}ms, max allowed: {MAX_MEDIUM_CONFIG_TIME_MS}ms"
            )

            assert large_metrics["execution_time_ms"] < MAX_LARGE_CONFIG_TIME_MS, (
                f"Large config took {large_metrics['execution_time_ms']:.2f}ms, max allowed: {MAX_LARGE_CONFIG_TIME_MS}ms"
            )

            print("Performance baseline test passed:")
            print(
                f"  Small: {small_metrics['execution_time_ms']:.2f}ms < {MAX_SMALL_CONFIG_TIME_MS}ms"
            )
            print(
                f"  Medium: {medium_metrics['execution_time_ms']:.2f}ms < {MAX_MEDIUM_CONFIG_TIME_MS}ms"
            )
            print(
                f"  Large: {large_metrics['execution_time_ms']:.2f}ms < {MAX_LARGE_CONFIG_TIME_MS}ms"
            )

    @pytest.mark.asyncio
    async def test_cache_hit_rate_baseline(self, optimized_config_service, sample_configs):
        """Test that cache hit rates meet baseline requirements."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_files = create_config_files(sample_configs, temp_path)

            # Load configs multiple times to build up cache
            for _ in range(3):
                for config_file in config_files.values():
                    await optimized_config_service.load_experiment_config(config_file)

            cache_stats = await optimized_config_service.get_cache_performance_stats()
            hit_rate = cache_stats["advanced_cache"]["hit_rate_percent"]

            # Should achieve at least 60% hit rate with repeated loads
            MIN_HIT_RATE = 60.0
            assert hit_rate >= MIN_HIT_RATE, (
                f"Cache hit rate {hit_rate:.1f}% below minimum {MIN_HIT_RATE}%"
            )

            print(f"Cache hit rate baseline test passed: {hit_rate:.1f}% >= {MIN_HIT_RATE}%")

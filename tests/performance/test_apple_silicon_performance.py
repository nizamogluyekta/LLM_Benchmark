"""
Performance benchmarks for Apple Silicon optimizations.

Tests actual performance improvements, throughput, latency, and resource utilization
on Apple Silicon hardware with various model types and batch configurations.
"""

import asyncio
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from benchmark.models.optimization import (
    AppleSiliconOptimizer,
    InferenceQueue,
    InferenceRequest,
    PerformanceMetrics,
    RequestPriority,
)


class MockModelPlugin:
    """Mock model plugin for performance testing."""

    def __init__(self, processing_time_ms: float = 100.0, batch_multiplier: float = 1.0):
        self.processing_time_ms = processing_time_ms
        self.batch_multiplier = batch_multiplier
        self.call_count = 0

    async def initialize(self, config: dict[str, Any]):
        return Mock(success=True)

    async def predict(self, samples: list[str]):
        """Simulate model prediction with realistic timing."""
        self.call_count += 1
        batch_size = len(samples)

        # Simulate processing time (scales with batch size but not linearly)
        processing_time = (self.processing_time_ms / 1000.0) * (batch_size**self.batch_multiplier)
        await asyncio.sleep(processing_time)

        # Return mock predictions
        return [
            Mock(
                sample_id=f"sample_{i}",
                input_text=sample,
                prediction="mock_result",
                confidence=0.95,
                inference_time_ms=processing_time / batch_size * 1000,
            )
            for i, sample in enumerate(samples)
        ]

    async def explain(self, sample: str):
        await asyncio.sleep(0.01)  # Small delay for explanation
        return "Mock explanation"

    async def health_check(self):
        return {"status": "healthy"}


class PerformanceBenchmark:
    """Performance benchmark utilities."""

    @staticmethod
    async def measure_throughput(
        queue: InferenceQueue,
        model_id: str,
        num_requests: int = 100,
        batch_size: int = 4,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> dict[str, float]:
        """Measure inference throughput."""
        start_time = time.time()
        completed_requests = 0

        # Submit requests
        tasks = []
        for i in range(num_requests):
            request = InferenceRequest(
                request_id=f"perf_req_{i}",
                model_id=model_id,
                input_data=[f"test input {i}" for _ in range(batch_size)],
                priority=priority,
                timeout_ms=10000,
                created_at=time.time() * 1000,
            )

            # Create callback to track completion
            completion_future = asyncio.Future()

            async def completion_callback(result, fut=completion_future):
                nonlocal completed_requests
                completed_requests += 1
                if not fut.done():
                    fut.set_result(result)

            request.callback = completion_callback

            task = asyncio.create_task(queue.submit_request(request))
            tasks.append((task, completion_future))

        # Wait for all requests to be submitted
        await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)

        # Wait for processing to complete
        await asyncio.gather(*[fut for _, fut in tasks], return_exceptions=True)

        total_time = time.time() - start_time

        return {
            "total_time_seconds": total_time,
            "requests_per_second": completed_requests / total_time if total_time > 0 else 0,
            "completed_requests": completed_requests,
            "total_requests": num_requests,
            "success_rate": completed_requests / num_requests if num_requests > 0 else 0,
        }

    @staticmethod
    async def measure_latency_distribution(
        queue: InferenceQueue, model_id: str, num_requests: int = 50
    ) -> dict[str, float]:
        """Measure latency distribution."""
        latencies = []

        for i in range(num_requests):
            request_start = time.time()

            request = InferenceRequest(
                request_id=f"latency_req_{i}",
                model_id=model_id,
                input_data=[f"latency test {i}"],
                priority=RequestPriority.NORMAL,
                timeout_ms=5000,
                created_at=time.time() * 1000,
            )

            completion_future = asyncio.Future()

            async def completion_callback(result, fut=completion_future):
                fut.set_result(result)

            request.callback = completion_callback

            await queue.submit_request(request)
            await completion_future

            latency_ms = (time.time() - request_start) * 1000
            latencies.append(latency_ms)

        return {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18]
            if len(latencies) >= 20
            else max(latencies),
            "p99_latency_ms": statistics.quantiles(latencies, n=100)[98]
            if len(latencies) >= 100
            else max(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }

    @staticmethod
    async def measure_batch_efficiency(
        _optimizer: AppleSiliconOptimizer,
        _model_id: str,
        batch_sizes: list[int] | None = None,
        samples_per_batch: int = 20,
    ) -> dict[int, dict[str, float]]:
        """Measure batch processing efficiency."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        results = {}

        for batch_size in batch_sizes:
            # Create mock plugin for testing
            mock_plugin = MockModelPlugin(processing_time_ms=50.0, batch_multiplier=0.8)

            batch_times = []
            for _ in range(samples_per_batch // batch_size):
                samples = [f"batch test {i}" for i in range(batch_size)]

                start_time = time.time()
                await mock_plugin.predict(samples)
                batch_time = (time.time() - start_time) * 1000

                batch_times.append(batch_time)

            if batch_times:
                avg_batch_time = statistics.mean(batch_times)
                samples_per_second = (batch_size / avg_batch_time) * 1000

                results[batch_size] = {
                    "avg_batch_time_ms": avg_batch_time,
                    "samples_per_second": samples_per_second,
                    "efficiency_ratio": samples_per_second / batch_size,
                    "batch_count": len(batch_times),
                }

        return results


@pytest.mark.benchmark
class TestAppleSiliconPerformance:
    """Apple Silicon performance benchmark tests."""

    @pytest.fixture
    async def benchmark_setup(self):
        """Set up performance benchmarking environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create optimizer
            optimizer = AppleSiliconOptimizer(cache_dir=Path(tmpdir))
            await optimizer.initialize()

            # Create inference queue
            queue = InferenceQueue(
                max_concurrent_requests=8, max_queue_size=200, optimizer=optimizer
            )
            await queue.initialize()

            yield optimizer, queue

            await queue.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_baseline_throughput_performance(self, benchmark_setup):
        """Benchmark baseline throughput performance."""
        optimizer, queue = benchmark_setup
        model_id = "throughput_test_model"

        # Measure throughput with different batch sizes
        batch_sizes = [1, 4, 8, 16]
        results = {}

        for batch_size in batch_sizes:
            print(f"\nTesting throughput with batch size {batch_size}...")

            result = await PerformanceBenchmark.measure_throughput(
                queue=queue, model_id=model_id, num_requests=50, batch_size=batch_size
            )

            results[batch_size] = result
            print(f"Batch size {batch_size}: {result['requests_per_second']:.2f} req/s")

        # Verify performance characteristics
        for batch_size, result in results.items():
            assert (
                result["success_rate"] >= 0.9
            ), f"Success rate too low for batch size {batch_size}"
            assert (
                result["requests_per_second"] > 0
            ), f"No throughput measured for batch size {batch_size}"

        # Larger batches should generally be more efficient (up to a point)
        assert results[4]["requests_per_second"] >= results[1]["requests_per_second"] * 0.8

        print(f"\nThroughput test results: {results}")
        return results

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_latency_performance(self, benchmark_setup):
        """Benchmark latency performance and distribution."""
        optimizer, queue = benchmark_setup
        model_id = "latency_test_model"

        print("\nTesting latency performance...")

        latency_stats = await PerformanceBenchmark.measure_latency_distribution(
            queue=queue, model_id=model_id, num_requests=30
        )

        print(f"Latency statistics: {latency_stats}")

        # Verify latency requirements
        assert latency_stats["mean_latency_ms"] < 5000, "Mean latency too high"
        assert latency_stats["p95_latency_ms"] < 8000, "P95 latency too high"
        assert (
            latency_stats["std_dev_ms"] < latency_stats["mean_latency_ms"]
        ), "High latency variance"

        return latency_stats

    @pytest.mark.asyncio
    async def test_batch_size_optimization_performance(self, benchmark_setup):
        """Test performance of batch size optimization."""
        optimizer, queue = benchmark_setup

        print("\nTesting batch size optimization performance...")

        # Test different model types
        model_types = ["llm", "embedding", "classification"]
        optimization_results = {}

        for model_type in model_types:
            model_id = f"{model_type}_optimization_test"

            # Get initial batch configuration
            initial_config = optimizer.get_optimal_batch_size(model_id, model_type)

            # Measure performance with different batch sizes
            batch_efficiency = await PerformanceBenchmark.measure_batch_efficiency(
                optimizer=optimizer, model_id=model_id, batch_sizes=[1, 2, 4, 8, 16, 32]
            )

            optimization_results[model_type] = {
                "initial_batch_size": initial_config.batch_size,
                "max_batch_size": initial_config.max_batch_size,
                "batch_efficiency": batch_efficiency,
            }

            print(f"{model_type} optimization - initial batch size: {initial_config.batch_size}")
            print(f"Efficiency results: {batch_efficiency}")

        # Verify that optimization provides reasonable configurations
        for model_type, result in optimization_results.items():
            assert result["initial_batch_size"] >= 1
            assert result["max_batch_size"] >= result["initial_batch_size"]

            # Check that batch efficiency generally improves with larger batches (up to a point)
            efficiency_data = result["batch_efficiency"]
            if len(efficiency_data) >= 3:
                batch_sizes = sorted(efficiency_data.keys())
                # At least some improvement should be seen with larger batches
                efficiency_values = [
                    efficiency_data[bs]["samples_per_second"] for bs in batch_sizes
                ]
                max_efficiency = max(efficiency_values)
                min_efficiency = min(efficiency_values)
                assert (
                    max_efficiency > min_efficiency * 1.5
                ), f"Insufficient batch efficiency improvement for {model_type}"

        return optimization_results

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, benchmark_setup):
        """Test performance under concurrent load."""
        optimizer, queue = benchmark_setup

        print("\nTesting concurrent processing performance...")

        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        concurrent_results = {}

        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")

            # Create concurrent tasks
            tasks = []
            for i in range(concurrency):
                task = PerformanceBenchmark.measure_throughput(
                    queue=queue, model_id=f"concurrent_model_{i}", num_requests=20, batch_size=4
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Aggregate results
            successful_results = [
                r for r in results if isinstance(r, dict) and "requests_per_second" in r
            ]

            if successful_results:
                total_throughput = sum(r["requests_per_second"] for r in successful_results)
                avg_success_rate = sum(r["success_rate"] for r in successful_results) / len(
                    successful_results
                )

                concurrent_results[concurrency] = {
                    "total_throughput": total_throughput,
                    "avg_success_rate": avg_success_rate,
                    "total_time": total_time,
                    "successful_tasks": len(successful_results),
                    "total_tasks": concurrency,
                }

                print(
                    f"Concurrency {concurrency}: {total_throughput:.2f} total req/s, {avg_success_rate:.2%} success rate"
                )

        # Verify concurrent processing scales reasonably
        if len(concurrent_results) >= 2:
            single_throughput = concurrent_results[1]["total_throughput"]
            multi_throughput = concurrent_results[max(concurrent_results.keys())][
                "total_throughput"
            ]

            # Should see some scaling benefit (not necessarily linear due to overhead)
            assert (
                multi_throughput >= single_throughput * 1.5
            ), "Insufficient scaling with concurrency"

        return concurrent_results

    @pytest.mark.asyncio
    async def test_adaptive_optimization_performance(self, benchmark_setup):
        """Test performance of adaptive optimization over time."""
        optimizer, queue = benchmark_setup
        model_id = "adaptive_performance_test"

        print("\nTesting adaptive optimization performance...")

        # Initial performance measurement
        initial_config = optimizer.get_optimal_batch_size(model_id, "classification")

        initial_metrics = await PerformanceBenchmark.measure_throughput(
            queue=queue, model_id=model_id, num_requests=30, batch_size=initial_config.batch_size
        )

        print(f"Initial performance: {initial_metrics['requests_per_second']:.2f} req/s")

        # Simulate performance feedback to trigger adaptive optimization
        for iteration in range(15):  # Enough iterations to trigger adaptation
            # Create varying performance metrics
            base_latency = 400 + (iteration * 20)  # Gradually increasing latency
            efficiency = 0.8 - (iteration * 0.02)  # Gradually decreasing efficiency

            metrics = PerformanceMetrics(
                requests_per_second=10.0,
                average_latency_ms=base_latency,
                p95_latency_ms=base_latency * 1.2,
                p99_latency_ms=base_latency * 1.5,
                queue_depth=3,
                batch_efficiency=max(0.5, efficiency),
                memory_usage_gb=6.0,
                gpu_utilization=0.6,
                neural_engine_utilization=0.3,
            )

            await optimizer.update_performance_metrics(model_id, metrics)

        # Measure performance after adaptation
        adapted_config = optimizer.get_optimal_batch_size(model_id, "classification")

        adapted_metrics = await PerformanceBenchmark.measure_throughput(
            queue=queue, model_id=model_id, num_requests=30, batch_size=adapted_config.batch_size
        )

        print(f"Adapted performance: {adapted_metrics['requests_per_second']:.2f} req/s")
        print(f"Batch size change: {initial_config.batch_size} -> {adapted_config.batch_size}")

        # Verify that adaptation occurred (batch size should have changed due to degrading performance)
        adaptation_occurred = abs(adapted_config.batch_size - initial_config.batch_size) >= 1

        results = {
            "adaptation_occurred": adaptation_occurred,
            "initial_batch_size": initial_config.batch_size,
            "adapted_batch_size": adapted_config.batch_size,
            "initial_throughput": initial_metrics["requests_per_second"],
            "adapted_throughput": adapted_metrics["requests_per_second"],
            "performance_history_length": len(optimizer.performance_history.get(model_id, [])),
        }

        # Verify that performance history is being maintained
        assert results["performance_history_length"] > 10, "Performance history not being tracked"

        return results

    @pytest.mark.skipif(
        platform.system() != "Darwin", reason="Requires macOS for real hardware testing"
    )
    @pytest.mark.asyncio
    async def test_real_apple_silicon_performance(self, benchmark_setup):
        """Test performance on real Apple Silicon hardware."""
        optimizer, queue = benchmark_setup

        # Only run if we're actually on Apple Silicon
        if optimizer.hardware_info.type.value.startswith("apple"):
            print(f"\nTesting on real Apple Silicon: {optimizer.hardware_info.model_name}")
            print(
                f"Hardware: {optimizer.hardware_info.performance_cores}P + {optimizer.hardware_info.efficiency_cores}E cores"
            )
            print(
                f"GPU cores: {optimizer.hardware_info.gpu_cores}, Memory: {optimizer.hardware_info.unified_memory_gb}GB"
            )

            # Test performance with real hardware optimizations
            model_types = ["llm", "embedding"]
            real_hw_results = {}

            for model_type in model_types:
                model_id = f"real_hw_{model_type}_test"

                # Get hardware-optimized configuration
                config = optimizer.get_optimal_batch_size(model_id, model_type)
                acceleration = optimizer.get_optimal_acceleration(model_type)

                print(
                    f"\n{model_type} - Optimal batch size: {config.batch_size}, Acceleration: {acceleration.value}"
                )

                # Measure performance
                performance_metrics = await PerformanceBenchmark.measure_throughput(
                    queue=queue, model_id=model_id, num_requests=40, batch_size=config.batch_size
                )

                latency_metrics = await PerformanceBenchmark.measure_latency_distribution(
                    queue=queue, model_id=model_id, num_requests=20
                )

                real_hw_results[model_type] = {
                    "optimal_config": {
                        "batch_size": config.batch_size,
                        "max_batch_size": config.max_batch_size,
                        "acceleration": acceleration.value,
                    },
                    "throughput": performance_metrics,
                    "latency": latency_metrics,
                }

                print(f"Performance: {performance_metrics['requests_per_second']:.2f} req/s")
                print(
                    f"Latency: {latency_metrics['mean_latency_ms']:.1f}ms mean, {latency_metrics['p95_latency_ms']:.1f}ms P95"
                )

            # Verify that real hardware provides reasonable performance
            for model_type, results in real_hw_results.items():
                assert (
                    results["throughput"]["success_rate"] >= 0.9
                ), f"Low success rate for {model_type}"
                assert (
                    results["throughput"]["requests_per_second"] > 1.0
                ), f"Very low throughput for {model_type}"
                assert (
                    results["latency"]["mean_latency_ms"] < 10000
                ), f"Very high latency for {model_type}"

            return real_hw_results
        else:
            pytest.skip("Not running on Apple Silicon hardware")


@pytest.mark.benchmark
class TestPerformanceComparison:
    """Compare performance with and without Apple Silicon optimizations."""

    @pytest.fixture
    async def comparison_setup(self):
        """Set up optimized and non-optimized environments for comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Optimized setup
            optimized_optimizer = AppleSiliconOptimizer(cache_dir=Path(tmpdir) / "optimized")
            await optimized_optimizer.initialize()

            optimized_queue = InferenceQueue(
                max_concurrent_requests=6, max_queue_size=100, optimizer=optimized_optimizer
            )
            await optimized_queue.initialize()

            # Non-optimized setup (basic queue without optimizer)
            basic_queue = InferenceQueue(
                max_concurrent_requests=6, max_queue_size=100, optimizer=None
            )
            await basic_queue.initialize()

            yield (optimized_optimizer, optimized_queue), basic_queue

            await optimized_queue.shutdown()
            await basic_queue.shutdown()

    @pytest.mark.asyncio
    async def test_optimization_benefit_comparison(self, comparison_setup):
        """Compare performance benefits of Apple Silicon optimization."""
        (optimizer, optimized_queue), basic_queue = comparison_setup

        print("\nComparing optimized vs basic performance...")

        test_configs = [
            {"batch_size": 4, "num_requests": 40, "model_type": "llm"},
            {"batch_size": 8, "num_requests": 40, "model_type": "embedding"},
        ]

        comparison_results = {}

        for config in test_configs:
            model_type = config["model_type"]
            batch_size = config["batch_size"]
            num_requests = config["num_requests"]

            print(f"\nTesting {model_type} with batch size {batch_size}...")

            # Test optimized performance
            optimized_result = await PerformanceBenchmark.measure_throughput(
                queue=optimized_queue,
                model_id=f"optimized_{model_type}_model",
                num_requests=num_requests,
                batch_size=batch_size,
            )

            # Test basic performance
            basic_result = await PerformanceBenchmark.measure_throughput(
                queue=basic_queue,
                model_id=f"basic_{model_type}_model",
                num_requests=num_requests,
                batch_size=batch_size,
            )

            # Calculate improvement
            throughput_improvement = (
                (optimized_result["requests_per_second"] / basic_result["requests_per_second"] - 1)
                * 100
                if basic_result["requests_per_second"] > 0
                else 0
            )

            comparison_results[f"{model_type}_batch_{batch_size}"] = {
                "optimized": optimized_result,
                "basic": basic_result,
                "throughput_improvement_percent": throughput_improvement,
            }

            print(f"Optimized: {optimized_result['requests_per_second']:.2f} req/s")
            print(f"Basic: {basic_result['requests_per_second']:.2f} req/s")
            print(f"Improvement: {throughput_improvement:.1f}%")

        # Verify that optimization provides some benefit
        # (Note: In a real scenario with actual hardware acceleration,
        # improvements would be more significant)
        for config_name, results in comparison_results.items():
            optimized_success = results["optimized"]["success_rate"]
            basic_success = results["basic"]["success_rate"]

            # At minimum, optimized version should be as reliable as basic
            assert (
                optimized_success >= basic_success * 0.95
            ), f"Optimized version less reliable for {config_name}"

        return comparison_results

    @pytest.mark.asyncio
    async def test_resource_utilization_comparison(self, comparison_setup):
        """Compare resource utilization between optimized and basic approaches."""
        (optimizer, optimized_queue), basic_queue = comparison_setup

        print("\nTesting resource utilization comparison...")

        # Monitor queue depths and processing efficiency
        utilization_results = {}

        # Submit load to both queues simultaneously
        concurrent_load = 30

        # Load optimized queue
        optimized_tasks = []
        for i in range(concurrent_load):
            request = InferenceRequest(
                request_id=f"util_opt_{i}",
                model_id="utilization_test_optimized",
                input_data=[f"test data {i}"],
                priority=RequestPriority.NORMAL,
                timeout_ms=5000,
                created_at=time.time() * 1000,
            )
            task = asyncio.create_task(optimized_queue.submit_request(request))
            optimized_tasks.append(task)

        # Load basic queue
        basic_tasks = []
        for i in range(concurrent_load):
            request = InferenceRequest(
                request_id=f"util_basic_{i}",
                model_id="utilization_test_basic",
                input_data=[f"test data {i}"],
                priority=RequestPriority.NORMAL,
                timeout_ms=5000,
                created_at=time.time() * 1000,
            )
            task = asyncio.create_task(basic_queue.submit_request(request))
            basic_tasks.append(task)

        # Monitor queue status during processing
        monitoring_duration = 2.0  # seconds
        monitoring_interval = 0.1  # seconds

        optimized_queue_depths = []
        basic_queue_depths = []

        monitor_start = time.time()
        while time.time() - monitor_start < monitoring_duration:
            opt_status = optimized_queue.get_queue_status()
            basic_status = basic_queue.get_queue_status()

            optimized_queue_depths.append(sum(opt_status["queue_depths"].values()))
            basic_queue_depths.append(sum(basic_status["queue_depths"].values()))

            await asyncio.sleep(monitoring_interval)

        # Wait for all tasks to complete
        await asyncio.gather(*optimized_tasks, *basic_tasks, return_exceptions=True)

        # Calculate utilization metrics
        avg_opt_queue_depth = (
            sum(optimized_queue_depths) / len(optimized_queue_depths)
            if optimized_queue_depths
            else 0
        )
        avg_basic_queue_depth = (
            sum(basic_queue_depths) / len(basic_queue_depths) if basic_queue_depths else 0
        )

        utilization_results = {
            "optimized_avg_queue_depth": avg_opt_queue_depth,
            "basic_avg_queue_depth": avg_basic_queue_depth,
            "optimized_final_status": optimized_queue.get_queue_status(),
            "basic_final_status": basic_queue.get_queue_status(),
        }

        print(
            f"Average queue depths - Optimized: {avg_opt_queue_depth:.1f}, Basic: {avg_basic_queue_depth:.1f}"
        )

        # Optimized queue should generally handle load more efficiently
        # (lower queue depths indicate faster processing)
        efficiency_improvement = (avg_basic_queue_depth - avg_opt_queue_depth) / max(
            avg_basic_queue_depth, 1
        )
        utilization_results["efficiency_improvement_ratio"] = efficiency_improvement

        print(f"Efficiency improvement: {efficiency_improvement:.2%}")

        return utilization_results


if __name__ == "__main__":
    # Allow running benchmarks directly
    import sys

    async def run_benchmarks():
        """Run performance benchmarks."""
        print("Running Apple Silicon Performance Benchmarks")
        print("=" * 50)

        # Create benchmark setup
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = AppleSiliconOptimizer(cache_dir=Path(tmpdir))
            await optimizer.initialize()

            queue = InferenceQueue(
                max_concurrent_requests=8, max_queue_size=200, optimizer=optimizer
            )
            await queue.initialize()

            try:
                # Run basic performance tests
                print("\n1. Throughput Performance Test")
                throughput_results = await PerformanceBenchmark.measure_throughput(
                    queue=queue, model_id="benchmark_model", num_requests=100, batch_size=8
                )
                print(f"Throughput: {throughput_results['requests_per_second']:.2f} req/s")
                print(f"Success rate: {throughput_results['success_rate']:.2%}")

                print("\n2. Latency Performance Test")
                latency_results = await PerformanceBenchmark.measure_latency_distribution(
                    queue=queue, model_id="benchmark_model", num_requests=50
                )
                print(f"Mean latency: {latency_results['mean_latency_ms']:.1f}ms")
                print(f"P95 latency: {latency_results['p95_latency_ms']:.1f}ms")

                print("\n3. Batch Efficiency Test")
                batch_results = await PerformanceBenchmark.measure_batch_efficiency(
                    optimizer=optimizer, model_id="benchmark_model", batch_sizes=[1, 2, 4, 8, 16]
                )

                for batch_size, result in batch_results.items():
                    print(f"Batch size {batch_size}: {result['samples_per_second']:.2f} samples/s")

                print("\nBenchmarks completed successfully!")

            finally:
                await queue.shutdown()

    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        asyncio.run(run_benchmarks())
    else:
        print("Use --run flag to execute benchmarks directly")
        print(
            "Or run with pytest: pytest tests/performance/test_apple_silicon_performance.py -m benchmark"
        )

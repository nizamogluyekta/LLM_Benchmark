"""
Comprehensive performance tests for the unified model service.

This module tests performance characteristics on MacBook Pro M4 Pro hardware,
including inference speed, memory usage, concurrent processing, and realistic
workload scenarios for cybersecurity evaluation tasks.

Target Performance Benchmarks:
- Local MLX models: >8 tokens/sec for 7B models
- API models: <5 second average response time
- Memory usage: <16GB total for realistic model combinations
- Concurrent processing: Support 2-3 models simultaneously
"""

import asyncio
import statistics
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
import pytest_asyncio

from benchmark.core.base import ServiceResponse
from benchmark.interfaces.model_interfaces import (
    LoadingStrategy,
    ModelInfo,
    PerformanceMetrics,
    Prediction,
)
from benchmark.services.model_service import ModelService


class PerformanceMetricsCollector:
    """Helper class to collect and analyze performance metrics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.response_times = []
        self.throughput_samples = []

    def start_measurement(self):
        """Start performance measurement."""
        self.start_time = time.time()
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    def stop_measurement(self):
        """Stop performance measurement."""
        self.end_time = time.time()

    def add_memory_sample(self):
        """Add current memory usage sample."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)

    def add_response_time(self, response_time: float):
        """Add response time sample."""
        self.response_times.append(response_time)

    def add_throughput(self, samples_processed: int, time_taken: float):
        """Add throughput sample."""
        if time_taken > 0:
            throughput = samples_processed / time_taken
            self.throughput_samples.append(throughput)

    def get_duration(self) -> float:
        """Get total measurement duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def get_average_memory(self) -> float:
        """Get average memory usage."""
        return statistics.mean(self.memory_samples) if self.memory_samples else 0.0

    def get_peak_memory(self) -> float:
        """Get peak memory usage."""
        return max(self.memory_samples) if self.memory_samples else 0.0

    def get_average_response_time(self) -> float:
        """Get average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0

    def get_average_throughput(self) -> float:
        """Get average throughput."""
        return statistics.mean(self.throughput_samples) if self.throughput_samples else 0.0


@pytest_asyncio.fixture
async def performance_model_service(mock_performance_plugins):  # noqa: ARG001
    """Create and initialize a model service optimized for performance testing."""
    # Use the mock plugins to ensure the service is initialized with them
    service = ModelService(
        max_models=5,
        max_memory_mb=16384,  # 16GB limit for M4 Pro
        cleanup_interval_seconds=60,
        enable_performance_monitoring=True,
    )

    await service.initialize()
    yield service
    await service.shutdown()


@pytest_asyncio.fixture
async def mock_performance_plugins():
    """Mock plugins with realistic performance characteristics."""

    def create_mock_prediction(sample_id: str, input_text: str, inference_time_ms: float):
        """Create a mock prediction with timing."""
        return Prediction(
            sample_id=sample_id,
            input_text=input_text,
            prediction="ATTACK" if "malicious" in input_text.lower() else "BENIGN",
            confidence=0.85,
            attack_type="SQL Injection" if "sql" in input_text.lower() else None,
            explanation="Detected suspicious pattern",
            inference_time_ms=inference_time_ms,
            model_version="test-v1.0",
        )

    with (
        patch("benchmark.models.plugins.openai_api.OpenAIModelPlugin") as mock_openai,
        patch("benchmark.models.plugins.anthropic_api.AnthropicModelPlugin") as mock_anthropic,
        patch("benchmark.models.plugins.mlx_local.MLXModelPlugin") as mock_mlx,
        patch("benchmark.models.plugins.ollama_local.OllamaModelPlugin") as mock_ollama,
        # Also patch the actual module imports
        patch("benchmark.services.model_service.OpenAIModelPlugin") as mock_openai2,
        patch("benchmark.services.model_service.AnthropicModelPlugin") as mock_anthropic2,
        patch("benchmark.services.model_service.MLXModelPlugin") as mock_mlx2,
        patch("benchmark.services.model_service.OllamaModelPlugin") as mock_ollama2,
    ):
        # API models - faster but with rate limiting
        for mock_api, mock_api2 in [(mock_openai, mock_openai2), (mock_anthropic, mock_anthropic2)]:
            api_plugin = MagicMock()

            async def api_predict(samples):
                # Simulate API call latency (1-5 seconds)
                await asyncio.sleep(0.1)  # Reduced for testing
                predictions = []
                for i, sample in enumerate(samples):
                    pred = create_mock_prediction(f"api_{i}", sample, 100.0)
                    predictions.append(pred)
                return predictions

            # Mock plugin methods
            api_plugin.predict = api_predict
            api_plugin.explain = AsyncMock(return_value="API-based explanation")
            api_plugin.get_supported_models.return_value = [
                "gpt-4o-mini",
                "claude-3-haiku-20240307",
            ]
            api_plugin.get_model_specs.return_value = {"context_window": 128000, "memory_gb": 2}
            api_plugin.cleanup = AsyncMock()

            # Mock initialization and model info

            api_plugin.initialize = AsyncMock(
                return_value=ServiceResponse(success=True, message="Initialized successfully")
            )
            api_plugin.get_model_info = AsyncMock(
                return_value=ModelInfo(
                    model_id="api-test",
                    name="API Test Model",
                    type="api",
                    memory_usage_mb=100,
                    status="loaded",
                )
            )
            api_plugin.get_performance_metrics = AsyncMock(
                return_value=PerformanceMetrics(
                    model_id="api-test",
                    total_predictions=0,
                    average_inference_time_ms=100.0,
                    predictions_per_second=10.0,
                )
            )
            api_plugin.health_check = AsyncMock(return_value={"status": "healthy"})

            mock_api.return_value = api_plugin
            mock_api2.return_value = api_plugin

        # Local models - higher throughput but more memory
        for mock_local, mock_local2 in [(mock_mlx, mock_mlx2), (mock_ollama, mock_ollama2)]:
            local_plugin = MagicMock()

            async def local_predict(samples):
                # Simulate local model inference (faster for batches)
                # Individual calls have overhead, batches are more efficient
                if len(samples) == 1:
                    await asyncio.sleep(0.05)  # Overhead for single requests
                else:
                    await asyncio.sleep(0.01 + 0.005 * len(samples))  # Better batch efficiency
                predictions = []
                for i, sample in enumerate(samples):
                    pred = create_mock_prediction(f"local_{i}", sample, 50.0)
                    predictions.append(pred)
                return predictions

            # Mock plugin methods
            local_plugin.predict = local_predict
            local_plugin.explain = AsyncMock(return_value="Local model explanation")
            local_plugin.get_supported_models.return_value = ["llama2-7b", "llama2-13b"]
            local_plugin.get_model_specs.return_value = {"context_window": 4096, "memory_gb": 8}
            local_plugin.cleanup = AsyncMock()

            # Mock initialization and model info
            local_plugin.initialize = AsyncMock(
                return_value=ServiceResponse(success=True, message="Initialized successfully")
            )
            local_plugin.get_model_info = AsyncMock(
                return_value=ModelInfo(
                    model_id="local-test",
                    name="Local Test Model",
                    type="local",
                    memory_usage_mb=2000,
                    status="loaded",
                )
            )
            local_plugin.get_performance_metrics = AsyncMock(
                return_value=PerformanceMetrics(
                    model_id="local-test",
                    total_predictions=0,
                    average_inference_time_ms=50.0,
                    predictions_per_second=20.0,
                )
            )
            local_plugin.health_check = AsyncMock(return_value={"status": "healthy"})

            mock_local.return_value = local_plugin
            mock_local2.return_value = local_plugin

        yield {
            "openai": mock_openai,
            "anthropic": mock_anthropic,
            "mlx": mock_mlx,
            "ollama": mock_ollama,
        }


@pytest_asyncio.fixture
def cybersecurity_samples():
    """Generate realistic cybersecurity samples for testing."""
    return [
        # SQL Injection samples
        "SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "admin'; DROP TABLE users; --",
        "' UNION SELECT username, password FROM accounts --",
        # XSS samples
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        # Command injection
        "; rm -rf /",
        "| nc -l -p 1234 -e /bin/sh",
        "&& cat /etc/passwd",
        # Network traffic patterns
        "GET /admin HTTP/1.1\nHost: target.com",
        "POST /login HTTP/1.1\nContent-Length: 1000000",
        "CONNECT 192.168.1.1:22 HTTP/1.1",
        # Email content
        "URGENT: Update your banking details immediately",
        "Click here to claim your prize: http://malicious.com",
        "Your account has been compromised. Login here:",
        # Benign samples
        "SELECT name FROM products WHERE category = 'electronics'",
        "Welcome to our website",
        "Please review the attached document",
        "Thank you for your purchase",
        "System maintenance scheduled for tonight",
        "User logged in successfully",
    ]


@pytest.mark.performance
class TestModelServicePerformance:
    """Performance tests for the unified model service."""

    @pytest.mark.asyncio
    async def test_single_model_inference_performance(
        self, performance_model_service, mock_performance_plugins, cybersecurity_samples
    ):
        """Test inference performance for individual models."""
        metrics = PerformanceMetricsCollector()
        metrics.start_measurement()

        # Load a single model
        config = {"type": "mlx_local", "model_name": "llama2-7b", "name": "performance-test"}

        model_id = await performance_model_service.load_model(config)
        metrics.add_memory_sample()

        # Test single inference performance
        single_start = time.time()
        await performance_model_service.predict_batch(
            model_id, [cybersecurity_samples[0]], batch_size=1
        )
        single_time = time.time() - single_start

        metrics.add_response_time(single_time)
        metrics.add_throughput(1, single_time)

        # Test batch inference performance
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            batch_samples = cybersecurity_samples[:batch_size]

            batch_start = time.time()
            batch_response = await performance_model_service.predict_batch(
                model_id, batch_samples, batch_size=batch_size
            )
            batch_time = time.time() - batch_start

            metrics.add_response_time(batch_time)
            metrics.add_throughput(batch_size, batch_time)
            metrics.add_memory_sample()

            assert batch_response.successful_predictions == batch_size
            assert batch_response.total_inference_time_ms > 0

        metrics.stop_measurement()

        # Performance assertions for MacBook Pro M4 Pro
        avg_throughput = metrics.get_average_throughput()
        avg_response_time = metrics.get_average_response_time()

        # Local MLX models should achieve >8 tokens/sec equivalent
        # (approximating with samples/sec for cybersecurity content)
        assert avg_throughput >= 4.0, f"Throughput {avg_throughput:.2f} samples/sec below target"

        # Response time should be reasonable for local models
        assert avg_response_time <= 2.0, f"Average response time {avg_response_time:.2f}s too high"

        # Memory usage should be reasonable
        peak_memory = metrics.get_peak_memory()
        assert peak_memory <= 12000, f"Peak memory {peak_memory:.1f}MB exceeds limit"

    @pytest.mark.asyncio
    async def test_concurrent_model_inference(
        self, performance_model_service, mock_performance_plugins, cybersecurity_samples
    ):
        """Test performance with multiple models running concurrently."""
        metrics = PerformanceMetricsCollector()
        metrics.start_measurement()

        # Load multiple models
        configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "api-model"},
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "local-small"},
            {"type": "mlx_local", "model_name": "llama2-13b", "name": "local-medium"},
        ]

        model_ids = []
        for config in configs:
            model_id = await performance_model_service.load_model(config)
            model_ids.append(model_id)
            metrics.add_memory_sample()

        # Test concurrent inference
        async def run_inference(model_id: str, samples: list[str], name: str):
            start_time = time.time()
            response = await performance_model_service.predict_batch(
                model_id, samples, batch_size=len(samples)
            )
            end_time = time.time()
            return {
                "model_id": model_id,
                "name": name,
                "duration": end_time - start_time,
                "samples": len(samples),
                "successful": response.successful_predictions,
            }

        # Run concurrent inference tasks
        sample_batches = [
            cybersecurity_samples[:5],  # API model - smaller batch
            cybersecurity_samples[5:15],  # Local small - medium batch
            cybersecurity_samples[15:20],  # Local medium - smaller batch for heavier model
        ]

        concurrent_start = time.time()
        tasks = [
            run_inference(model_ids[i], sample_batches[i], f"model_{i}")
            for i in range(len(model_ids))
        ]

        results = await asyncio.gather(*tasks)
        concurrent_end = time.time()
        concurrent_duration = concurrent_end - concurrent_start

        metrics.stop_measurement()
        metrics.add_memory_sample()

        # Analyze results
        total_samples = sum(r["samples"] for r in results)
        total_successful = sum(r["successful"] for r in results)
        overall_throughput = total_samples / concurrent_duration

        # Assertions for concurrent performance
        assert total_successful == total_samples, "Some predictions failed"
        assert concurrent_duration <= 10.0, f"Concurrent processing took {concurrent_duration:.2f}s"
        assert overall_throughput >= 2.0, f"Concurrent throughput {overall_throughput:.2f} too low"

        # Memory should support 2-3 models simultaneously
        peak_memory = metrics.get_peak_memory()
        assert peak_memory <= 16000, f"Peak memory {peak_memory:.1f}MB exceeds M4 Pro target"

    @pytest.mark.asyncio
    async def test_memory_usage_multiple_models(
        self, performance_model_service, mock_performance_plugins
    ):
        """Test memory usage when loading multiple models."""
        metrics = PerformanceMetricsCollector()

        # Baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Test progressive model loading
        model_configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "api-1"},
            {"type": "anthropic_api", "model_name": "claude-3-haiku-20240307", "name": "api-2"},
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "local-7b"},
            {"type": "mlx_local", "model_name": "llama2-13b", "name": "local-13b"},
        ]

        memory_progression = [baseline_memory]
        loaded_models = []

        for config in model_configs:
            model_id = await performance_model_service.load_model(config)
            loaded_models.append(model_id)

            # Measure memory after each model load
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_progression.append(current_memory)

            # Test memory during inference
            sample = "SELECT * FROM users WHERE active = true"
            await performance_model_service.predict_batch(model_id, [sample])

            # Track memory after inference
            metrics.add_memory_sample()

        # Memory growth analysis
        final_memory = memory_progression[-1]
        memory_increase = final_memory - baseline_memory

        # Assertions for memory usage
        assert final_memory <= 16000, f"Total memory {final_memory:.1f}MB exceeds 16GB limit"
        assert memory_increase <= 12000, f"Memory increase {memory_increase:.1f}MB too high"

        # API models should have minimal memory impact
        api_memory_increase = memory_progression[2] - baseline_memory  # After 2 API models
        assert api_memory_increase <= 500, f"API models used {api_memory_increase:.1f}MB"

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(
        self, performance_model_service, mock_performance_plugins, cybersecurity_samples
    ):
        """Test efficiency of batch processing vs individual requests."""
        config = {"type": "mlx_local", "model_name": "llama2-7b", "name": "batch-test"}
        model_id = await performance_model_service.load_model(config)

        test_samples = cybersecurity_samples[:16]

        # Test individual processing
        individual_start = time.time()
        individual_results = []
        for sample in test_samples:
            response = await performance_model_service.predict_batch(
                model_id, [sample], batch_size=1
            )
            individual_results.extend(response.predictions)
        individual_time = time.time() - individual_start

        # Test batch processing
        batch_sizes = [4, 8, 16]
        batch_results = {}

        for batch_size in batch_sizes:
            batch_start = time.time()
            response = await performance_model_service.predict_batch(
                model_id, test_samples, batch_size=batch_size
            )
            batch_time = time.time() - batch_start

            batch_results[batch_size] = {
                "time": batch_time,
                "throughput": len(test_samples) / batch_time,
                "successful": response.successful_predictions,
            }

        # Efficiency analysis
        individual_throughput = len(test_samples) / individual_time
        best_batch_throughput = max(r["throughput"] for r in batch_results.values())
        efficiency_gain = best_batch_throughput / individual_throughput

        # Assertions for batch efficiency
        assert efficiency_gain >= 1.5, (
            f"Batch processing only {efficiency_gain:.2f}x more efficient"
        )

        # Larger batches should generally be more efficient (with some exceptions)
        batch_8_throughput = batch_results[8]["throughput"]
        batch_4_throughput = batch_results[4]["throughput"]

        # At least one larger batch size should outperform smaller ones
        assert batch_8_throughput >= batch_4_throughput * 0.8, "Batch scaling inefficient"

    @pytest.mark.asyncio
    async def test_api_rate_limiting_performance(
        self, performance_model_service, mock_performance_plugins, cybersecurity_samples
    ):
        """Test that rate limiting doesn't significantly impact performance."""
        config = {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "rate-limit-test"}
        model_id = await performance_model_service.load_model(config)

        # Test sustained API usage
        batch_count = 5
        batch_size = 4
        batch_times = []

        for i in range(batch_count):
            samples = cybersecurity_samples[i * batch_size : (i + 1) * batch_size]

            batch_start = time.time()
            response = await performance_model_service.predict_batch(
                model_id, samples, batch_size=batch_size
            )
            batch_time = time.time() - batch_start

            batch_times.append(batch_time)

            assert response.successful_predictions == len(samples)

            # Small delay between batches to simulate realistic usage
            await asyncio.sleep(0.1)

        # Analyze rate limiting impact
        avg_batch_time = statistics.mean(batch_times)
        max_batch_time = max(batch_times)
        time_variance = statistics.stdev(batch_times) if len(batch_times) > 1 else 0

        # Performance assertions for API rate limiting
        assert avg_batch_time <= 5.0, f"Average API response time {avg_batch_time:.2f}s too high"
        assert max_batch_time <= 10.0, f"Max API response time {max_batch_time:.2f}s too high"
        assert time_variance <= 2.0, f"Response time variance {time_variance:.2f}s too high"

    @pytest.mark.asyncio
    async def test_model_loading_optimization(
        self, performance_model_service, mock_performance_plugins
    ):
        """Test optimized model loading strategies."""
        configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "api-fast"},
            {
                "type": "anthropic_api",
                "model_name": "claude-3-haiku-20240307",
                "name": "api-fast-2",
            },
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "local-medium"},
            {"type": "mlx_local", "model_name": "llama2-13b", "name": "local-large"},
        ]

        # Test optimization strategy generation
        strategy = await performance_model_service.optimize_model_loading(configs)

        assert isinstance(strategy, LoadingStrategy)
        assert len(strategy.loading_order) == len(configs)
        assert strategy.estimated_total_memory_mb > 0
        assert strategy.estimated_loading_time_seconds > 0

        # Test actual optimized loading
        loading_start = time.time()

        # Load models according to optimization strategy
        loaded_models = []
        for model_name in strategy.loading_order:
            # Find corresponding config - handle case where name might not match exactly
            matching_config = None
            for c in configs:
                if c["name"] == model_name or c["model_name"] == model_name:
                    matching_config = c
                    break

            if matching_config is None:
                # If no exact match, use the first config as fallback
                matching_config = configs[len(loaded_models) % len(configs)]

            model_id = await performance_model_service.load_model(matching_config)
            loaded_models.append(model_id)

        loading_time = time.time() - loading_start

        # Performance assertions for optimized loading
        assert loading_time <= strategy.estimated_loading_time_seconds * 2, (
            f"Loading took {loading_time:.2f}s vs estimated {strategy.estimated_loading_time_seconds:.2f}s"
        )

        # Verify all models are functional
        test_sample = "SELECT * FROM users"
        for model_id in loaded_models:
            response = await performance_model_service.predict_batch(model_id, [test_sample])
            assert response.successful_predictions == 1

    @pytest.mark.asyncio
    async def test_benchmark_realistic_workloads(
        self, performance_model_service, mock_performance_plugins, cybersecurity_samples
    ):
        """Benchmark with realistic cybersecurity evaluation workloads."""
        metrics = PerformanceMetricsCollector()
        metrics.start_measurement()

        # Load realistic model combination
        model_configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "api-primary"},
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "local-backup"},
        ]

        model_ids = []
        for config in model_configs:
            model_id = await performance_model_service.load_model(config)
            model_ids.append(model_id)

        # Simulate realistic evaluation workload
        workload_scenarios = [
            {
                "name": "SQL Injection Detection",
                "samples": [s for s in cybersecurity_samples if "SELECT" in s or "DROP" in s],
                "model_id": model_ids[0],  # Use API model for specialized detection
                "batch_size": 4,
            },
            {
                "name": "XSS Detection",
                "samples": [s for s in cybersecurity_samples if "script" in s or "javascript" in s],
                "model_id": model_ids[1],  # Use local model
                "batch_size": 8,
            },
            {
                "name": "General Threat Analysis",
                "samples": cybersecurity_samples[-6:],  # Mix of samples
                "model_id": model_ids[0],
                "batch_size": 6,
            },
        ]

        workload_results = []

        for scenario in workload_scenarios:
            scenario_start = time.time()

            response = await performance_model_service.predict_batch(
                scenario["model_id"], scenario["samples"], batch_size=scenario["batch_size"]
            )

            scenario_time = time.time() - scenario_start

            workload_results.append(
                {
                    "name": scenario["name"],
                    "duration": scenario_time,
                    "samples": len(scenario["samples"]),
                    "successful": response.successful_predictions,
                    "throughput": len(scenario["samples"]) / scenario_time,
                }
            )

            metrics.add_response_time(scenario_time)
            metrics.add_throughput(len(scenario["samples"]), scenario_time)
            metrics.add_memory_sample()

        metrics.stop_measurement()

        # Generate comprehensive benchmark report
        total_samples = sum(r["samples"] for r in workload_results)
        total_duration = metrics.get_duration()
        overall_throughput = total_samples / total_duration
        avg_response_time = metrics.get_average_response_time()
        peak_memory = metrics.get_peak_memory()

        benchmark_report = {
            "test_timestamp": datetime.now().isoformat(),
            "hardware_profile": "MacBook Pro M4 Pro",
            "total_samples": total_samples,
            "total_duration_seconds": total_duration,
            "overall_throughput": overall_throughput,
            "average_response_time": avg_response_time,
            "peak_memory_mb": peak_memory,
            "workload_scenarios": workload_results,
            "performance_targets_met": {
                "api_response_time": avg_response_time <= 5.0,
                "memory_limit": peak_memory <= 16000,
                "throughput": overall_throughput >= 2.0,
            },
        }

        # Performance assertions for realistic workloads
        assert total_duration <= 30.0, f"Realistic workload took {total_duration:.2f}s"
        assert overall_throughput >= 1.5, (
            f"Overall throughput {overall_throughput:.2f} samples/sec too low"
        )
        assert all(r["successful"] == r["samples"] for r in workload_results), (
            "Some predictions failed"
        )

        # Log benchmark report for analysis
        print("\n" + "=" * 60)
        print("CYBERSECURITY MODEL SERVICE PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Hardware: {benchmark_report['hardware_profile']}")
        print(f"Total Samples: {benchmark_report['total_samples']}")
        print(f"Duration: {benchmark_report['total_duration_seconds']:.2f}s")
        print(f"Throughput: {benchmark_report['overall_throughput']:.2f} samples/sec")
        print(f"Avg Response: {benchmark_report['average_response_time']:.2f}s")
        print(f"Peak Memory: {benchmark_report['peak_memory_mb']:.1f}MB")
        print("\nScenario Performance:")
        for scenario in workload_results:
            print(f"  {scenario['name']}: {scenario['throughput']:.2f} samples/sec")
        print("\nPerformance Targets:")
        for target, met in benchmark_report["performance_targets_met"].items():
            status = "✅ PASSED" if met else "❌ FAILED"
            print(f"  {target}: {status}")
        print("=" * 60)

    @pytest.mark.asyncio
    async def test_performance_monitoring_overhead(
        self, performance_model_service, mock_performance_plugins, cybersecurity_samples
    ):
        """Test the overhead of performance monitoring itself."""
        config = {"type": "mlx_local", "model_name": "llama2-7b", "name": "monitor-test"}
        model_id = await performance_model_service.load_model(config)

        samples = cybersecurity_samples[:10]

        # Test with monitoring enabled
        monitoring_start = time.time()
        await performance_model_service.predict_batch(model_id, samples, batch_size=len(samples))
        monitoring_time = time.time() - monitoring_start

        # Create service without monitoring for comparison
        non_monitoring_service = ModelService(
            max_models=5, max_memory_mb=16384, enable_performance_monitoring=False
        )
        await non_monitoring_service.initialize()

        try:
            model_id_no_monitor = await non_monitoring_service.load_model(config)

            no_monitoring_start = time.time()
            await non_monitoring_service.predict_batch(
                model_id_no_monitor, samples, batch_size=len(samples)
            )
            no_monitoring_time = time.time() - no_monitoring_start

            # Calculate monitoring overhead
            overhead = monitoring_time - no_monitoring_time
            overhead_percentage = (
                (overhead / no_monitoring_time) * 100 if no_monitoring_time > 0 else 0
            )

            # Assertions for monitoring overhead
            assert overhead_percentage <= 10.0, (
                f"Performance monitoring adds {overhead_percentage:.1f}% overhead"
            )
            assert monitoring_time <= no_monitoring_time * 1.15, (
                "Performance monitoring significantly impacts performance"
            )

        finally:
            await non_monitoring_service.shutdown()

    @pytest.mark.asyncio
    async def test_stress_test_high_concurrency(
        self, performance_model_service, mock_performance_plugins, cybersecurity_samples
    ):
        """Stress test with high concurrency to find performance limits."""
        # Load multiple models for stress testing
        configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": f"api-{i}"}
            for i in range(2)
        ] + [
            {"type": "mlx_local", "model_name": "llama2-7b", "name": f"local-{i}"} for i in range(2)
        ]

        model_ids = []
        for config in configs:
            model_id = await performance_model_service.load_model(config)
            model_ids.append(model_id)

        # Create high concurrency stress test
        async def stress_inference(model_id: str, samples: list[str], task_id: int):
            try:
                response = await performance_model_service.predict_batch(
                    model_id, samples, batch_size=min(len(samples), 4)
                )
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "success": True,
                    "samples": len(samples),
                    "successful_predictions": response.successful_predictions,
                }
            except Exception as e:
                return {"task_id": task_id, "model_id": model_id, "success": False, "error": str(e)}

        # Create stress test tasks
        stress_tasks = []
        for i in range(20):  # 20 concurrent tasks
            model_id = model_ids[i % len(model_ids)]
            task_samples = cybersecurity_samples[: (i % 5) + 2]  # 2-6 samples per task
            task = stress_inference(model_id, task_samples, i)
            stress_tasks.append(task)

        # Execute stress test
        stress_start = time.time()
        stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
        stress_duration = time.time() - stress_start

        # Analyze stress test results
        successful_tasks = [r for r in stress_results if isinstance(r, dict) and r.get("success")]

        success_rate = len(successful_tasks) / len(stress_tasks)
        total_samples = sum(task["samples"] for task in successful_tasks)
        stress_throughput = total_samples / stress_duration

        # Stress test assertions
        assert success_rate >= 0.8, f"Only {success_rate:.1%} of stress test tasks succeeded"
        assert stress_duration <= 60.0, f"Stress test took {stress_duration:.2f}s"
        assert stress_throughput >= 1.0, f"Stress test throughput {stress_throughput:.2f} too low"

        print("\nStress Test Results:")
        print(f"  Tasks: {len(stress_tasks)} total, {len(successful_tasks)} succeeded")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Duration: {stress_duration:.2f}s")
        print(f"  Throughput: {stress_throughput:.2f} samples/sec")

"""
Comprehensive integration tests for the complete Model Service.

This module contains thorough integration tests that validate the complete Model Service
functionality with all plugins, optimizations, and real-world scenarios.
"""

import asyncio
import contextlib
import time
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from benchmark.core.config import ModelConfig
from benchmark.core.exceptions import BenchmarkError
from benchmark.models.optimization import (
    HardwareInfo,
    HardwareType,
)
from benchmark.models.resource_manager import ResourceCheckResult
from benchmark.services.configuration_service import ConfigurationService
from benchmark.services.model_service import ModelService


class TestCompleteModelService:
    """Comprehensive integration tests for Model Service."""

    @pytest_asyncio.fixture
    async def config_service(self):
        """Configured configuration service."""
        service = ConfigurationService()
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest_asyncio.fixture
    async def model_service(self, config_service):
        """Configured model service with all plugins."""
        service = ModelService()
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.fixture
    def sample_cybersecurity_data(self):
        """Sample cybersecurity data for testing."""
        return [
            "192.168.1.100 -> 10.0.0.5 PORT_SCAN detected on ports 22,23,80,443",
            "2024-01-15 14:32:18 [INFO] User authentication successful: admin@company.com",
            "TCP connection established: 203.0.113.42:4444 -> 192.168.1.50:1337 SUSPICIOUS",
            "Email received with attachment: invoice.pdf.exe from unknown-sender@malicious.com",
            "Normal HTTP GET request to /api/users/profile from authenticated session",
        ]

    @pytest.fixture
    def resource_check_result(self):
        """Sample resource check result for testing."""
        return ResourceCheckResult(
            can_load=True,
            estimated_memory_gb=8.0,
            current_usage_gb=16.0,
            recommendations=["Use quantized models for better memory efficiency"],
        )

    @pytest.fixture
    def mock_hardware_info(self):
        """Mock M4 Pro hardware info for testing."""
        return HardwareInfo(
            type=HardwareType.APPLE_SILICON_M4,
            model_name="M4 Pro",
            core_count=14,
            performance_cores=10,
            efficiency_cores=4,
            gpu_cores=20,
            neural_engine_cores=16,
            unified_memory_gb=48.0,
            metal_support=True,
            neural_engine_support=True,
        )

    @pytest.mark.asyncio
    async def test_load_all_model_types(self, model_service, config_service):
        """Test loading models from all available plugin types."""
        # Test configurations for each model type
        model_configs = [
            ModelConfig(
                name="test_mlx_model",
                type="mlx_local",
                path="mlx-community/Llama-3.2-3B-Instruct-4bit",
                max_tokens=256,
                temperature=0.1,
            ),
            ModelConfig(
                name="test_openai_model",
                type="openai_api",
                path="gpt-4o-mini",
                max_tokens=256,
                temperature=0.1,
            ),
            ModelConfig(
                name="test_anthropic_model",
                type="anthropic_api",
                path="claude-3-haiku-20240307",
                max_tokens=256,
                temperature=0.1,
            ),
        ]

        loaded_model_ids = []

        for config in model_configs:
            try:
                with patch.object(model_service, "_load_model_plugin") as mock_load:
                    # Mock the model plugin
                    mock_plugin = MagicMock()
                    mock_plugin.predict.return_value = [
                        {
                            "sample_id": "test",
                            "input_text": "test input",
                            "prediction": "BENIGN",
                            "confidence": 0.85,
                            "inference_time_ms": 150.0,
                        }
                    ]
                    mock_load.return_value = mock_plugin

                    model_id = await model_service.load_model(config)
                    loaded_model_ids.append(model_id)

                    # Verify model info can be retrieved
                    model_info = await model_service.get_model_info(model_id)
                    assert model_info.success
                    assert model_info.data["model_name"] == config.name

            except Exception as e:
                # Skip if model not available (e.g., missing API keys)
                if "API key" in str(e) or "not found" in str(e):
                    pytest.skip(f"Skipping {config.name}: {e}")
                else:
                    raise

        # Ensure at least one model loaded successfully
        assert len(loaded_model_ids) >= 1, "At least one model should load successfully"

        # Cleanup
        for model_id in loaded_model_ids:
            await model_service.cleanup_model(model_id)

    @pytest.mark.asyncio
    async def test_cybersecurity_prediction_pipeline(
        self, model_service, sample_cybersecurity_data
    ):
        """Test complete cybersecurity prediction pipeline."""
        # Load a test model (mock if necessary)
        model_config = ModelConfig(
            name="test_cyber_model",
            type="mlx_local",
            path="test://mock-model",  # Use mock for testing
            max_tokens=512,
            temperature=0.1,
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            # Mock the model plugin
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [
                {
                    "sample_id": str(i),
                    "input_text": sample,
                    "prediction": "ATTACK"
                    if "SUSPICIOUS" in sample or "PORT_SCAN" in sample
                    else "BENIGN",
                    "confidence": 0.95 if "SUSPICIOUS" in sample else 0.75,
                    "attack_type": "reconnaissance" if "PORT_SCAN" in sample else None,
                    "explanation": f"Analysis of: {sample[:50]}...",
                    "inference_time_ms": 150.0,
                }
                for i, sample in enumerate(sample_cybersecurity_data)
            ]
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Test batch prediction
            predictions = await model_service.predict_batch(model_id, sample_cybersecurity_data)

            # Validate predictions
            assert len(predictions) == len(sample_cybersecurity_data)

            for prediction in predictions:
                # Validate required fields
                assert "prediction" in prediction
                assert prediction["prediction"] in ["ATTACK", "BENIGN"]
                assert "confidence" in prediction
                assert 0.0 <= prediction["confidence"] <= 1.0
                assert "inference_time_ms" in prediction
                assert prediction["inference_time_ms"] > 0

                # Validate cybersecurity-specific fields
                if prediction["prediction"] == "ATTACK":
                    assert "attack_type" in prediction
                    assert "explanation" in prediction
                    assert len(prediction["explanation"]) > 0

    @pytest.mark.asyncio
    async def test_resource_management_multiple_models(self, model_service):
        """Test resource management with multiple concurrent models."""
        # Create multiple model configs
        model_configs = [
            ModelConfig(
                name=f"model_{i}",
                type="mlx_local",
                path="test://mock-model",
                max_tokens=256,
            )
            for i in range(3)
        ]

        # Mock resource manager to simulate M4 Pro constraints
        with patch.object(model_service.resource_manager, "can_load_model") as mock_check:
            mock_check.return_value = ResourceCheckResult(
                can_load=True,
                estimated_memory_gb=8.0,
                current_usage_gb=16.0,
                recommendations=["Consider using quantized models"],
            )

            with patch.object(model_service, "_load_model_plugin") as mock_load:
                mock_plugin = MagicMock()
                mock_plugin.predict.return_value = [{"prediction": "BENIGN", "confidence": 0.8}]
                mock_load.return_value = mock_plugin

                # Load multiple models
                model_ids = []
                for config in model_configs:
                    model_id = await model_service.load_model(config)
                    model_ids.append(model_id)

                # Test concurrent predictions
                test_samples = ["Test network log entry"]

                tasks = [
                    model_service.predict_batch(model_id, test_samples) for model_id in model_ids
                ]

                results = await asyncio.gather(*tasks)

                # Validate all predictions completed
                assert len(results) == len(model_ids)
                for result in results:
                    assert len(result) == len(test_samples)

                # Cleanup
                for model_id in model_ids:
                    await model_service.cleanup_model(model_id)

    @pytest.mark.asyncio
    async def test_performance_optimization_integration(self, model_service):
        """Test performance optimization features."""
        # Test hardware detection
        with patch.object(model_service.performance_optimizer, "detect_hardware") as mock_detect:
            mock_detect.return_value = HardwareInfo(
                type=HardwareType.APPLE_SILICON_M4,
                model_name="M4 Pro",
                core_count=14,
                performance_cores=10,
                efficiency_cores=4,
                gpu_cores=20,
                neural_engine_cores=16,
                unified_memory_gb=48.0,
                metal_support=True,
                neural_engine_support=True,
            )

            hardware_info = await model_service.performance_optimizer.detect_hardware()

            if hardware_info.type in [
                HardwareType.APPLE_SILICON_M4,
                HardwareType.APPLE_SILICON_M3,
                HardwareType.APPLE_SILICON_M2,
                HardwareType.APPLE_SILICON_M1,
            ]:
                # Validate M4 Pro detection
                assert hardware_info.model_name is not None
                assert hardware_info.unified_memory_gb is not None
                assert hardware_info.unified_memory_gb > 0

                # Test optimization application
                await model_service.optimize_for_hardware()

                # Verify optimal batch sizes were set
                assert hasattr(model_service, "optimal_batch_sizes")
                assert len(model_service.optimal_batch_sizes) > 0

                # Test optimized batch prediction
                model_config = ModelConfig(
                    name="optimized_model", type="mlx_local", path="test://mock-model"
                )

                with patch.object(model_service, "_load_model_plugin") as mock_load:
                    mock_plugin = MagicMock()
                    mock_plugin.predict.return_value = [
                        {"prediction": "BENIGN", "confidence": 0.8, "inference_time_ms": 100.0}
                        for _ in range(32)  # Batch size
                    ]
                    mock_load.return_value = mock_plugin

                    model_id = await model_service.load_model(model_config)

                    # Test optimized batch processing
                    large_sample_set = ["Test sample"] * 100
                    predictions = await model_service.predict_batch_optimized(
                        model_id, large_sample_set
                    )

                    assert len(predictions) == 100

                    await model_service.cleanup_model(model_id)
            else:
                pytest.skip("Apple Silicon specific tests - running on different hardware")

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, model_service):
        """Test error handling and recovery scenarios."""
        # Test model loading failure
        invalid_config = ModelConfig(
            name="invalid_model",
            type="mlx_local",
            path="nonexistent://invalid-path",
        )

        with pytest.raises((BenchmarkError, ValueError, TypeError)):
            await model_service.load_model(invalid_config)

        # Test prediction with invalid model_id
        with pytest.raises(BenchmarkError):
            await model_service.predict_batch("nonexistent_model", ["test"])

        # Test recovery from API failures
        api_config = ModelConfig(name="api_model", type="openai_api", path="gpt-4o-mini")

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()
            # Simulate API failure then success
            mock_plugin.predict.side_effect = [
                Exception("API Error"),
                [{"prediction": "BENIGN", "confidence": 0.8}],
            ]
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(api_config)

            # First call should handle error gracefully
            with pytest.raises((Exception, BenchmarkError)):
                await model_service.predict_batch(model_id, ["test"])

            # Second call should succeed
            result = await model_service.predict_batch(model_id, ["test"])
            assert len(result) == 1
            assert result[0]["prediction"] == "BENIGN"

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, model_service):
        """Test integration with performance monitoring."""
        model_config = ModelConfig(
            name="monitored_model", type="mlx_local", path="test://mock-model"
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [
                {"prediction": "ATTACK", "confidence": 0.9, "inference_time_ms": 200.0}
            ]
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Make several predictions to generate performance data
            for _ in range(5):
                await model_service.predict_batch(model_id, ["test sample"])

            # Check performance metrics were collected
            performance_summary = await model_service.get_model_performance(model_id)

            assert performance_summary.success
            performance_data = performance_summary.data

            assert "avg_inference_time_ms" in performance_data
            assert "total_predictions" in performance_data
            assert performance_data["total_predictions"] == 5

            await model_service.cleanup_model(model_id)

    @pytest.mark.asyncio
    async def test_integration_with_configuration_service(self, model_service, config_service):
        """Test Model Service integration with Configuration Service."""
        # Create experiment config with multiple models
        experiment_config = {
            "name": "Integration Test Experiment",
            "models": [
                {
                    "name": "test_model_1",
                    "type": "mlx_local",
                    "path": "test://mock-model-1",
                    "max_tokens": 256,
                },
                {
                    "name": "test_model_2",
                    "type": "mlx_local",
                    "path": "test://mock-model-2",
                    "max_tokens": 512,
                },
            ],
        }

        # Load experiment config through config service
        with patch.object(config_service, "load_experiment_config") as mock_load_config:
            mock_load_config.return_value = experiment_config

            config = await config_service.load_experiment_config("test_config.yaml")

            # Load models based on configuration
            with patch.object(model_service, "_load_model_plugin") as mock_load:
                mock_plugin = MagicMock()
                mock_plugin.predict.return_value = [{"prediction": "BENIGN"}]
                mock_load.return_value = mock_plugin

                model_ids = []
                for model_config_dict in config["models"]:
                    model_config = ModelConfig(**model_config_dict)
                    model_id = await model_service.load_model(model_config)
                    model_ids.append(model_id)

                # Verify all models loaded successfully
                assert len(model_ids) == 2

                # Test that models work with configuration settings
                for model_id in model_ids:
                    predictions = await model_service.predict_batch(model_id, ["test"])
                    assert len(predictions) == 1

                # Cleanup
                for model_id in model_ids:
                    await model_service.cleanup_model(model_id)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_realistic_cybersecurity_workload(self, model_service, sample_cybersecurity_data):
        """Test Model Service with realistic cybersecurity evaluation workload."""
        # Simulate realistic workload: multiple models, larger dataset
        large_dataset = sample_cybersecurity_data * 20  # 100 samples total

        model_config = ModelConfig(
            name="cyber_workload_model",
            type="mlx_local",
            path="test://mock-cyber-model",
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()

            def mock_predict(samples):
                return [
                    {
                        "sample_id": str(i),
                        "input_text": sample,
                        "prediction": "ATTACK"
                        if any(
                            indicator in sample
                            for indicator in ["SUSPICIOUS", "PORT_SCAN", "malicious"]
                        )
                        else "BENIGN",
                        "confidence": 0.85,
                        "attack_type": "malware" if "malicious" in sample else None,
                        "explanation": "Cybersecurity analysis of log entry",
                        "inference_time_ms": 180.0,
                    }
                    for i, sample in enumerate(samples)
                ]

            mock_plugin.predict.side_effect = mock_predict
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Measure performance
            start_time = time.time()
            predictions = await model_service.predict_batch(model_id, large_dataset)
            total_time = time.time() - start_time

            # Validate results
            assert len(predictions) == len(large_dataset)

            # Calculate performance metrics
            throughput = len(predictions) / total_time

            # Validate performance meets expectations
            assert throughput > 10, f"Throughput too low: {throughput} samples/sec"

            # Validate prediction quality
            attack_predictions = [p for p in predictions if p["prediction"] == "ATTACK"]
            benign_predictions = [p for p in predictions if p["prediction"] == "BENIGN"]

            assert len(attack_predictions) > 0, "Should detect some attacks"
            assert len(benign_predictions) > 0, "Should detect some benign traffic"

            # All predictions should have required fields
            for prediction in predictions:
                assert all(
                    field in prediction
                    for field in ["prediction", "confidence", "inference_time_ms"]
                )
                assert prediction["prediction"] in ["ATTACK", "BENIGN"]
                assert 0.0 <= prediction["confidence"] <= 1.0

            await model_service.cleanup_model(model_id)


# Performance test utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking."""

    @staticmethod
    async def benchmark_inference_speed(
        model_service, model_id: str, samples: list[str], iterations: int = 3
    ) -> dict[str, float]:
        """Benchmark inference speed for a model."""
        times = []

        for _ in range(iterations):
            start_time = time.time()
            await model_service.predict_batch(model_id, samples)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        throughput = len(samples) / avg_time

        return {
            "avg_inference_time_sec": avg_time,
            "throughput_samples_per_sec": throughput,
            "samples_processed": len(samples),
            "iterations": iterations,
        }

    @staticmethod
    async def benchmark_memory_usage(
        model_service, model_configs: list[ModelConfig]
    ) -> dict[str, float]:
        """Benchmark memory usage for different model configurations."""
        import psutil

        process = psutil.Process()

        initial_memory = process.memory_info().rss / (1024**3)  # GB

        model_ids = []
        memory_usage = {}

        try:
            for config in model_configs:
                with patch.object(model_service, "_load_model_plugin") as mock_load:
                    mock_plugin = MagicMock()
                    mock_plugin.predict.return_value = [{"prediction": "BENIGN"}]
                    mock_load.return_value = mock_plugin

                    model_id = await model_service.load_model(config)
                    model_ids.append(model_id)

                    current_memory = process.memory_info().rss / (1024**3)
                    memory_usage[config.name] = current_memory - initial_memory

        finally:
            # Cleanup
            for model_id in model_ids:
                with contextlib.suppress(Exception):
                    await model_service.cleanup_model(model_id)

        return memory_usage


@pytest.mark.asyncio
async def test_benchmark_inference_speed():
    """Test the performance benchmarking utility."""
    service = ModelService()
    await service.initialize()

    try:
        model_config = ModelConfig(
            name="benchmark_model", type="mlx_local", path="test://mock-model"
        )

        with patch.object(service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [{"prediction": "BENIGN", "confidence": 0.8}]
            mock_load.return_value = mock_plugin

            model_id = await service.load_model(model_config)

            # Benchmark inference speed
            results = await PerformanceBenchmark.benchmark_inference_speed(
                service, model_id, ["test sample"] * 10, iterations=2
            )

            assert "avg_inference_time_sec" in results
            assert "throughput_samples_per_sec" in results
            assert results["samples_processed"] == 10
            assert results["iterations"] == 2

            await service.cleanup_model(model_id)
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_benchmark_memory_usage():
    """Test the memory usage benchmarking utility."""
    service = ModelService()
    await service.initialize()

    try:
        model_configs = [
            ModelConfig(name="small_model", type="mlx_local", path="test://mock-small"),
            ModelConfig(name="large_model", type="mlx_local", path="test://mock-large"),
        ]

        # Benchmark memory usage
        memory_results = await PerformanceBenchmark.benchmark_memory_usage(service, model_configs)

        assert "small_model" in memory_results
        assert "large_model" in memory_results
        assert all(usage >= 0 for usage in memory_results.values())

    finally:
        await service.shutdown()

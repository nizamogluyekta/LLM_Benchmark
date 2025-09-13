"""
Edge case integration tests for Model Service.

This module tests edge cases, error conditions, and resource exhaustion scenarios
for the Model Service with various model configurations.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from benchmark.core.config import ModelConfig
from benchmark.models.resource_manager import ResourceCheckResult
from benchmark.services.model_service import ModelService


class TestModelServiceEdgeCases:
    """Test edge cases and error conditions for Model Service."""

    @pytest_asyncio.fixture
    async def model_service(self):
        """Model service for edge case testing."""
        service = ModelService()
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_loading_extremely_large_model(self, model_service):
        """Test loading a model that exceeds available memory."""
        # Create config for extremely large model
        large_model_config = ModelConfig(
            name="huge_model",
            type="mlx_local",
            path="test://huge-model-200gb",
            max_tokens=4096,
        )

        # Mock resource manager to simulate memory exhaustion
        with patch.object(model_service.resource_manager, "can_load_model") as mock_check:
            mock_check.return_value = ResourceCheckResult(
                can_load=False,
                estimated_memory_gb=200.0,
                current_usage_gb=16.0,
                recommendations=[
                    "Not enough memory available",
                    "Consider using a smaller model or quantized version",
                ],
            )

            # Should raise ResourceExhaustedError or similar
            with pytest.raises(Exception) as exc_info:
                await model_service.load_model(large_model_config)

            assert any(
                keyword in str(exc_info.value).lower()
                for keyword in ["memory", "resource", "exhausted", "cannot load"]
            )

    @pytest.mark.asyncio
    async def test_concurrent_model_loading_limits(self, model_service):
        """Test loading models beyond the concurrent limit."""
        # Create multiple model configs
        model_configs = [
            ModelConfig(
                name=f"concurrent_model_{i}",
                type="mlx_local",
                path="test://mock-model",
                max_tokens=256,
            )
            for i in range(10)  # Try to load 10 models concurrently
        ]

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [{"prediction": "BENIGN"}]
            mock_load.return_value = mock_plugin

            # Mock resource manager to allow first few models but block later ones
            call_count = 0

            def mock_can_load(_config):
                nonlocal call_count
                call_count += 1
                return ResourceCheckResult(
                    can_load=call_count <= 5,  # Only allow first 5 models
                    estimated_memory_gb=8.0,
                    current_usage_gb=call_count * 8.0,
                    recommendations=["Memory usage increasing"],
                )

            with patch.object(
                model_service.resource_manager, "can_load_model", side_effect=mock_can_load
            ):
                # Try to load all models concurrently
                tasks = [model_service.load_model(config) for config in model_configs]

                # Some should succeed, others should fail
                results = await asyncio.gather(*tasks, return_exceptions=True)

                successful_loads = [
                    r for r in results if isinstance(r, str) and not isinstance(r, Exception)
                ]
                failed_loads = [r for r in results if isinstance(r, Exception)]

                # Should have some successes and some failures
                assert len(successful_loads) > 0, "At least some models should load"
                assert len(failed_loads) > 0, "Some models should fail due to resource limits"

                # Cleanup successful models
                for model_id in successful_loads:
                    if isinstance(model_id, str):
                        await model_service.cleanup_model(model_id)

    @pytest.mark.asyncio
    async def test_malformed_model_configurations(self, model_service):
        """Test handling of malformed model configurations."""
        malformed_configs = [
            # Missing required fields
            ModelConfig(name="", type="mlx_local", path=""),
            # Invalid type
            ModelConfig(name="invalid", type="invalid_type", path="test://model"),
            # Invalid path format
            ModelConfig(name="invalid_path", type="openai_api", path="not-a-valid-path"),
            # Negative values
            ModelConfig(
                name="negative_tokens",
                type="mlx_local",
                path="test://model",
                max_tokens=-100,
            ),
        ]

        for config in malformed_configs:
            with pytest.raises(Exception) as exc_info:
                await model_service.load_model(config)

            # Should get appropriate validation error
            assert any(
                keyword in str(exc_info.value).lower()
                for keyword in ["invalid", "validation", "configuration", "error"]
            )

    @pytest.mark.asyncio
    async def test_prediction_with_empty_and_oversized_inputs(self, model_service):
        """Test prediction handling with edge case inputs."""
        model_config = ModelConfig(
            name="edge_case_model",
            type="mlx_local",
            path="test://mock-model",
            max_tokens=100,
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()

            def mock_predict(samples):
                results = []
                for i, sample in enumerate(samples):
                    if not sample.strip():  # Empty sample
                        results.append(
                            {
                                "sample_id": str(i),
                                "input_text": sample,
                                "prediction": "INVALID",
                                "confidence": 0.0,
                                "error": "Empty input",
                                "inference_time_ms": 1.0,
                            }
                        )
                    elif len(sample) > 10000:  # Very large sample
                        results.append(
                            {
                                "sample_id": str(i),
                                "input_text": sample[:100] + "...",
                                "prediction": "TRUNCATED",
                                "confidence": 0.5,
                                "warning": "Input truncated due to size",
                                "inference_time_ms": 500.0,
                            }
                        )
                    else:
                        results.append(
                            {
                                "sample_id": str(i),
                                "input_text": sample,
                                "prediction": "BENIGN",
                                "confidence": 0.8,
                                "inference_time_ms": 150.0,
                            }
                        )
                return results

            mock_plugin.predict.side_effect = mock_predict
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Test edge case inputs
            edge_case_inputs = [
                "",  # Empty string
                "   ",  # Only whitespace
                "Normal input",  # Normal case
                "A" * 15000,  # Very large input
                "\n\n\n",  # Only newlines
                "Special chars: !@#$%^&*()_+{}[]|\\:\"<>?`,./;'",
                "Unicode: 你好世界 🌍 emoji test",
            ]

            predictions = await model_service.predict_batch(model_id, edge_case_inputs)

            # Should handle all inputs gracefully
            assert len(predictions) == len(edge_case_inputs)

            # Check that service handled edge cases appropriately
            for i, prediction in enumerate(predictions):
                assert "prediction" in prediction
                assert "confidence" in prediction
                assert isinstance(prediction["confidence"], float)
                assert 0.0 <= prediction["confidence"] <= 1.0

                # Empty inputs should be handled specially
                if not edge_case_inputs[i].strip():
                    assert prediction["prediction"] in ["INVALID", "ERROR", "UNKNOWN"]

            await model_service.cleanup_model(model_id)

    @pytest.mark.asyncio
    async def test_api_rate_limiting_and_retries(self, model_service):
        """Test handling of API rate limits and retry logic."""
        api_config = ModelConfig(
            name="rate_limited_model",
            type="openai_api",
            path="gpt-4o-mini",
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()

            # Simulate rate limiting then success
            call_count = 0

            def mock_predict_with_rate_limit(samples):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Rate limit exceeded - try again later")
                return [{"prediction": "BENIGN", "confidence": 0.8} for _ in samples]

            mock_plugin.predict.side_effect = mock_predict_with_rate_limit
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(api_config)

            # Should eventually succeed after retries
            with patch.object(model_service, "_retry_with_backoff") as mock_retry:
                # Mock retry logic to call the function multiple times
                async def mock_retry_func(func, *args, **kwargs):
                    for attempt in range(3):  # 3 attempts
                        try:
                            return await func(*args, **kwargs)
                        except Exception:
                            if attempt == 2:  # Last attempt
                                raise
                            await asyncio.sleep(0.1)  # Short delay for testing

                mock_retry.side_effect = mock_retry_func

                predictions = await model_service.predict_batch(model_id, ["test sample"])
                assert len(predictions) == 1
                assert predictions[0]["prediction"] == "BENIGN"

    @pytest.mark.asyncio
    async def test_model_plugin_crashes_and_recovery(self, model_service):
        """Test handling of plugin crashes and recovery."""
        crash_config = ModelConfig(
            name="crash_model",
            type="mlx_local",
            path="test://crash-model",
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()

            # Simulate plugin crash during prediction
            prediction_count = 0

            def mock_predict_with_crash(samples):
                nonlocal prediction_count
                prediction_count += 1
                if prediction_count == 2:  # Crash on second call
                    raise RuntimeError("Plugin crashed unexpectedly")
                return [{"prediction": "BENIGN", "confidence": 0.8} for _ in samples]

            mock_plugin.predict.side_effect = mock_predict_with_crash
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(crash_config)

            # First prediction should work
            result1 = await model_service.predict_batch(model_id, ["test1"])
            assert len(result1) == 1

            # Second prediction should crash
            with pytest.raises(RuntimeError):
                await model_service.predict_batch(model_id, ["test2"])

            # Model should be marked as failed or cleaned up
            # Test that service handles the crashed model appropriately
            health = await model_service.health_check()
            # The service should still be functional despite the model crash
            assert health.status in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_network_connectivity_issues(self, model_service):
        """Test handling of network connectivity issues for API models."""
        api_config = ModelConfig(
            name="network_test_model",
            type="anthropic_api",
            path="claude-3-haiku-20240307",
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()

            # Simulate various network issues
            network_errors = [
                ConnectionError("Network unreachable"),
                TimeoutError("Request timed out"),
                Exception("DNS resolution failed"),
            ]

            call_count = 0

            def mock_predict_with_network_issues(samples):
                nonlocal call_count
                call_count += 1
                if call_count <= len(network_errors):
                    raise network_errors[call_count - 1]
                return [{"prediction": "BENIGN", "confidence": 0.8} for _ in samples]

            mock_plugin.predict.side_effect = mock_predict_with_network_issues
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(api_config)

            # Should handle network errors gracefully
            for _error in network_errors:
                with pytest.raises(Exception) as exc_info:
                    await model_service.predict_batch(model_id, ["test"])

                # Should get network-related error
                assert any(
                    keyword in str(exc_info.value).lower()
                    for keyword in ["network", "connection", "timeout", "dns"]
                )

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, model_service):
        """Test that the service prevents memory leaks during heavy usage."""
        import gc

        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB

        model_config = ModelConfig(
            name="memory_test_model",
            type="mlx_local",
            path="test://mock-model",
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            # Create a mock plugin that generates large response objects
            mock_plugin = MagicMock()

            def mock_predict_large_response(samples):
                # Generate large prediction objects
                return [
                    {
                        "sample_id": str(i),
                        "input_text": sample,
                        "prediction": "BENIGN",
                        "confidence": 0.8,
                        "large_data": "x" * 1000,  # 1KB of data per prediction
                        "detailed_analysis": ["detail"] * 100,
                        "inference_time_ms": 150.0,
                    }
                    for i, sample in enumerate(samples)
                ]

            mock_plugin.predict.side_effect = mock_predict_large_response
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Perform many predictions to test for memory leaks
            large_sample_set = ["test sample"] * 100
            for iteration in range(10):  # 10 iterations of 100 samples each
                predictions = await model_service.predict_batch(model_id, large_sample_set)
                assert len(predictions) == 100

                # Force garbage collection
                gc.collect()

                # Check memory usage every few iterations
                if iteration % 3 == 0:
                    current_memory = process.memory_info().rss / (1024**2)  # MB
                    memory_increase = current_memory - initial_memory

                    # Memory shouldn't grow excessively (allow for some reasonable growth)
                    assert memory_increase < 500, (
                        f"Memory usage grew by {memory_increase}MB, possible leak"
                    )

            await model_service.cleanup_model(model_id)

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_and_prediction(self, model_service):
        """Test race conditions between cleanup and prediction operations."""
        model_config = ModelConfig(
            name="race_condition_model",
            type="mlx_local",
            path="test://mock-model",
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [{"prediction": "BENIGN", "confidence": 0.8}]
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Start prediction task
            prediction_task = asyncio.create_task(
                model_service.predict_batch(model_id, ["test sample"])
            )

            # Start cleanup task with slight delay
            async def delayed_cleanup():
                await asyncio.sleep(0.05)  # Small delay
                await model_service.cleanup_model(model_id)

            cleanup_task = asyncio.create_task(delayed_cleanup())

            # Wait for both tasks
            results = await asyncio.gather(prediction_task, cleanup_task, return_exceptions=True)

            # At least one should succeed, or both should handle the race condition gracefully
            prediction_result, cleanup_result = results

            if isinstance(prediction_result, Exception):
                # If prediction failed, it should be due to model cleanup
                assert any(
                    keyword in str(prediction_result).lower()
                    for keyword in ["not found", "unavailable", "cleaned up"]
                )
            else:
                # If prediction succeeded, result should be valid
                assert len(prediction_result) == 1

            # Cleanup should always succeed (or be idempotent)
            if isinstance(cleanup_result, Exception):
                # Cleanup can fail if model was already cleaned up, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_resource_exhaustion_during_prediction(self, model_service):
        """Test handling of resource exhaustion during prediction."""
        model_config = ModelConfig(
            name="resource_exhaustion_model",
            type="mlx_local",
            path="test://mock-model",
        )

        with patch.object(model_service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()

            # Simulate resource exhaustion during prediction
            def mock_predict_with_exhaustion(samples):
                if len(samples) > 50:  # Simulate exhaustion for large batches
                    raise Exception("Out of GPU memory during inference")
                return [{"prediction": "BENIGN", "confidence": 0.8} for _ in samples]

            mock_plugin.predict.side_effect = mock_predict_with_exhaustion
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Small batch should work
            small_batch = ["sample"] * 10
            result = await model_service.predict_batch(model_id, small_batch)
            assert len(result) == 10

            # Large batch should fail with resource exhaustion
            large_batch = ["sample"] * 100
            with pytest.raises((Exception, RuntimeError)):
                await model_service.predict_batch(model_id, large_batch)

            # Should be able to recover and handle small batches again
            result = await model_service.predict_batch(model_id, small_batch)
            assert len(result) == 10

            await model_service.cleanup_model(model_id)


@pytest.mark.asyncio
async def test_service_recovery_after_total_failure():
    """Test service recovery after catastrophic failure."""
    service = ModelService()

    try:
        await service.initialize()

        # Simulate total service failure
        with patch.object(service, "plugins", {}):  # Clear all plugins
            health = await service.health_check()
            assert health.status in ["error", "degraded"]

        # Service should be able to recover by re-initializing
        await service.initialize()
        health = await service.health_check()
        assert health.status == "healthy"
        assert len(service.plugins) > 0

    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_extreme_concurrent_load():
    """Test service behavior under extreme concurrent load."""
    service = ModelService()

    try:
        await service.initialize()

        model_config = ModelConfig(
            name="stress_test_model",
            type="mlx_local",
            path="test://mock-model",
        )

        with patch.object(service, "_load_model_plugin") as mock_load:
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [{"prediction": "BENIGN"}]
            mock_load.return_value = mock_plugin

            model_id = await service.load_model(model_config)

            # Create extreme concurrent load (100 concurrent predictions)
            tasks = []
            for i in range(100):
                task = service.predict_batch(model_id, [f"sample_{i}"])
                tasks.append(task)

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes and failures
            successes = [r for r in results if not isinstance(r, Exception)]

            # Should handle most requests successfully (allow some failures under extreme load)
            success_rate = len(successes) / len(results)
            assert success_rate > 0.7, f"Success rate too low: {success_rate}"

            await service.cleanup_model(model_id)

    finally:
        await service.shutdown()

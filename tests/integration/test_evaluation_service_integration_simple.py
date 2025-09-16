"""
Simplified integration tests for the evaluation service to ensure basic functionality works.
"""

import asyncio
import time

import pytest
import pytest_asyncio

from benchmark.interfaces.evaluation_interfaces import (
    EvaluationRequest,
    MetricType,
)
from benchmark.services.evaluation_service import EvaluationService
from tests.fixtures.evaluation_test_scenarios import (
    MockAccuracyEvaluator,
    MockPerformanceEvaluator,
    MockPrecisionRecallEvaluator,
    generate_cybersecurity_test_data,
)


class TestEvaluationServiceIntegrationSimple:
    """Simplified integration tests for evaluation service."""

    @pytest_asyncio.fixture
    async def evaluation_service(self):
        """Create a clean evaluation service instance for each test."""
        service = EvaluationService()

        # Initialize with test configuration
        config = {
            "max_concurrent_evaluations": 5,
            "evaluation_timeout_seconds": 10.0,
            "max_history_size": 100,
        }

        response = await service.initialize(config)
        assert response.success, f"Failed to initialize service: {response.error}"

        yield service

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_service_initialization(self, evaluation_service):
        """Test that the service initializes correctly."""

        # Check health status
        health = await evaluation_service.health_check()
        assert health.status in ["healthy", "degraded", "unhealthy"], (
            f"Unexpected health status: {health.status}"
        )

        # Initially unhealthy due to no evaluators
        assert health.status == "unhealthy"
        assert "No evaluators registered" in str(health.checks.get("issue", ""))

    @pytest.mark.asyncio
    async def test_evaluator_registration(self, evaluation_service):
        """Test evaluator registration and management."""

        # Register accuracy evaluator
        accuracy_evaluator = MockAccuracyEvaluator()
        response = await evaluation_service.register_evaluator(
            MetricType.ACCURACY, accuracy_evaluator
        )

        assert response.success, f"Failed to register accuracy evaluator: {response.error}"
        assert response.data["metric_type"] == "accuracy"

        # Register precision evaluator
        precision_evaluator = MockPrecisionRecallEvaluator()
        response = await evaluation_service.register_evaluator(
            MetricType.PRECISION, precision_evaluator
        )

        assert response.success, f"Failed to register precision evaluator: {response.error}"

        # Check available metrics
        metrics_response = await evaluation_service.get_available_metrics()
        assert metrics_response.success, "Should retrieve available metrics"
        assert metrics_response.data["total_evaluators"] == 2

        # Health should improve with evaluators
        health = await evaluation_service.health_check()
        assert health.status in ["healthy", "degraded"], (
            f"Service should be healthy with evaluators: {health.status}"
        )

    @pytest.mark.asyncio
    async def test_basic_evaluation_workflow(self, evaluation_service):
        """Test a complete basic evaluation workflow."""

        # Register evaluators
        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())
        await evaluation_service.register_evaluator(
            MetricType.PRECISION, MockPrecisionRecallEvaluator()
        )

        # Generate test data
        test_data = generate_cybersecurity_test_data(sample_count=50)

        # Create evaluation request
        request = EvaluationRequest(
            experiment_id="test_basic_workflow",
            model_id="test_model_v1",
            dataset_id="test_network_data",
            predictions=test_data.network_predictions,
            ground_truth=test_data.network_ground_truth,
            metrics=[MetricType.ACCURACY, MetricType.PRECISION],
            metadata={
                "test_type": "basic_workflow",
                "model_version": "1.0",
            },
        )

        # Execute evaluation
        start_time = time.time()
        result = await evaluation_service.evaluate_predictions(request)
        execution_time = time.time() - start_time

        # Validate results
        assert result is not None, "Evaluation result should not be None"
        assert result.success, f"Evaluation should succeed: {result.error_message}"
        assert result.experiment_id == "test_basic_workflow"
        assert result.model_id == "test_model_v1"
        assert result.dataset_id == "test_network_data"

        # Check metrics
        assert "accuracy" in result.metrics, "Should have accuracy metric"
        assert "precision" in result.metrics, "Should have precision metric"
        assert "recall" in result.metrics, "Should have recall metric"
        assert "f1_score" in result.metrics, "Should have f1_score metric"

        # Validate metric ranges
        assert 0.0 <= result.metrics["accuracy"] <= 1.0, "Accuracy should be between 0 and 1"
        assert 0.0 <= result.metrics["precision"] <= 1.0, "Precision should be between 0 and 1"
        assert 0.0 <= result.metrics["recall"] <= 1.0, "Recall should be between 0 and 1"
        assert 0.0 <= result.metrics["f1_score"] <= 1.0, "F1 score should be between 0 and 1"

        # Check execution time
        assert execution_time < 2.0, f"Evaluation took too long: {execution_time:.2f}s"
        assert result.execution_time_seconds > 0, "Execution time should be recorded"

    @pytest.mark.asyncio
    async def test_multiple_evaluations(self, evaluation_service):
        """Test multiple sequential evaluations."""

        # Register evaluators
        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())
        await evaluation_service.register_evaluator(
            MetricType.PERFORMANCE, MockPerformanceEvaluator()
        )

        # Generate test data
        test_data = generate_cybersecurity_test_data(sample_count=30)

        results = []

        # Run multiple evaluations
        for i in range(3):
            request = EvaluationRequest(
                experiment_id=f"multi_eval_{i:02d}",
                model_id=f"model_{i}",
                dataset_id="test_data",
                predictions=test_data.network_predictions,
                ground_truth=test_data.network_ground_truth,
                metrics=[MetricType.ACCURACY, MetricType.PERFORMANCE],
                metadata={"batch": i, "test": "multiple_evaluations"},
            )

            result = await evaluation_service.evaluate_predictions(request)
            assert result.success, f"Evaluation {i} should succeed: {result.error_message}"
            results.append(result)

        # Validate all results
        assert len(results) == 3, "Should have 3 evaluation results"

        # Check unique experiment IDs
        experiment_ids = {r.experiment_id for r in results}
        assert len(experiment_ids) == 3, "All experiments should have unique IDs"

        # Check history
        history_response = await evaluation_service.get_evaluation_history(limit=10)
        assert history_response.success, "Should retrieve evaluation history"
        assert history_response.data["total_results"] >= 3, (
            "History should include recent evaluations"
        )

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, evaluation_service):
        """Test concurrent evaluations within service limits."""

        # Register evaluators
        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())

        # Generate test data
        test_data = generate_cybersecurity_test_data(sample_count=20)

        # Create concurrent evaluation tasks
        num_concurrent = min(3, evaluation_service.max_concurrent_evaluations)
        tasks = []

        for i in range(num_concurrent):
            request = EvaluationRequest(
                experiment_id=f"concurrent_{i:02d}",
                model_id=f"concurrent_model_{i}",
                dataset_id="concurrent_test_data",
                predictions=test_data.malware_predictions,
                ground_truth=test_data.malware_ground_truth,
                metrics=[MetricType.ACCURACY],
                metadata={"concurrent_batch": i},
            )

            tasks.append(evaluation_service.evaluate_predictions(request))

        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Check results
        successful_results = [r for r in results if hasattr(r, "success") and r.success]
        assert len(successful_results) >= num_concurrent - 1, (
            f"Most concurrent evaluations should succeed: {len(successful_results)}/{num_concurrent}"
        )

        # Should be faster than sequential execution
        assert total_time < num_concurrent * 0.5, (
            f"Concurrent execution should be efficient: {total_time:.2f}s"
        )

    @pytest.mark.asyncio
    async def test_evaluation_summary_generation(self, evaluation_service):
        """Test evaluation summary and statistics generation."""

        # Register evaluator
        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())

        # Generate test data
        test_data = generate_cybersecurity_test_data(sample_count=25)

        # Run several evaluations to build history
        for i in range(4):
            request = EvaluationRequest(
                experiment_id=f"summary_test_{i:02d}",
                model_id=f"summary_model_{i}",
                dataset_id="summary_dataset",
                predictions=test_data.phishing_predictions,
                ground_truth=test_data.phishing_ground_truth,
                metrics=[MetricType.ACCURACY],
                metadata={"summary_test": True, "batch": i},
            )

            result = await evaluation_service.evaluate_predictions(request)
            assert result.success, f"Summary test evaluation {i} should succeed"

        # Generate summary
        summary_response = await evaluation_service.get_evaluation_summary(days_back=1)
        assert summary_response.success, "Should generate summary successfully"

        summary_data = summary_response.data
        assert summary_data["total_evaluations"] >= 4, "Summary should include recent evaluations"
        assert summary_data["successful_evaluations"] >= 4, "All evaluations should be successful"
        assert summary_data["failed_evaluations"] == 0, "No evaluations should fail"

        # Check metric summaries
        assert "accuracy" in summary_data["metric_summaries"], (
            "Summary should include accuracy statistics"
        )
        accuracy_stats = summary_data["metric_summaries"]["accuracy"]
        assert "mean" in accuracy_stats, "Should have mean accuracy"
        assert "count" in accuracy_stats, "Should have count of accuracy measurements"
        assert accuracy_stats["count"] >= 4, "Should count all accuracy measurements"

    @pytest.mark.asyncio
    async def test_error_handling_basic(self, evaluation_service):
        """Test basic error handling scenarios."""

        # Register working evaluator
        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())

        # Test 1: Invalid data format
        invalid_request = EvaluationRequest(
            experiment_id="error_test_invalid",
            model_id="error_model",
            dataset_id="error_dataset",
            predictions=[{"invalid": "format"}] * 10,
            ground_truth=[{"also_invalid": "format"}] * 10,
            metrics=[MetricType.ACCURACY],
            metadata={"test": "invalid_data"},
        )

        # Should handle gracefully (may succeed or fail depending on evaluator implementation)
        try:
            result = await evaluation_service.evaluate_predictions(invalid_request)
            # If it succeeds, that's also valid (mock evaluator is lenient)
            assert result is not None, "Should return some result for invalid data"
        except (ValueError, TypeError, AttributeError) as e:
            # If it fails, that's expected for invalid data
            assert (
                "validation" in str(e).lower()
                or "data" in str(e).lower()
                or "format" in str(e).lower()
            ), f"Error should be related to data validation: {e}"

        # Test 2: Empty data - this will fail at request creation due to validation
        with pytest.raises(ValueError) as exc_info:
            EvaluationRequest(
                experiment_id="error_test_empty",
                model_id="error_model",
                dataset_id="error_dataset",
                predictions=[],
                ground_truth=[],
                metrics=[MetricType.ACCURACY],
                metadata={"test": "empty_data"},
            )

        assert "empty" in str(exc_info.value).lower(), (
            "Should mention empty data in validation error"
        )

        # Service should remain operational
        health = await evaluation_service.health_check()
        assert health.status != "error", (
            "Service should not be in error state after handling exceptions"
        )

    @pytest.mark.asyncio
    async def test_service_performance_basic(self, evaluation_service):
        """Test basic performance characteristics."""

        # Register fast evaluators
        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())
        await evaluation_service.register_evaluator(
            MetricType.PERFORMANCE, MockPerformanceEvaluator()
        )

        # Generate moderate-sized test data
        test_data = generate_cybersecurity_test_data(sample_count=100)

        # Create evaluation request
        request = EvaluationRequest(
            experiment_id="perf_test_basic",
            model_id="performance_model",
            dataset_id="performance_dataset",
            predictions=test_data.vulnerability_predictions,
            ground_truth=test_data.vulnerability_ground_truth,
            metrics=[MetricType.ACCURACY, MetricType.PERFORMANCE],
            metadata={"test": "performance_basic"},
        )

        # Measure execution time
        start_time = time.time()
        result = await evaluation_service.evaluate_predictions(request)
        execution_time = time.time() - start_time

        # Validate performance
        assert result.success, f"Performance test should succeed: {result.error_message}"
        assert execution_time < 1.0, f"Basic evaluation should be fast: {execution_time:.2f}s"

        # Check performance metrics were collected
        assert "inference_latency_ms" in result.metrics, "Should measure inference latency"
        assert "throughput_samples_per_sec" in result.metrics, "Should measure throughput"
        assert result.metrics["inference_latency_ms"] > 0, "Latency should be positive"
        assert result.metrics["throughput_samples_per_sec"] > 0, "Throughput should be positive"

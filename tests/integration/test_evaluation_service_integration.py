"""
Comprehensive integration tests for the evaluation service.

This module provides end-to-end testing of the complete evaluation service
functionality, including multi-model evaluations, cybersecurity dataset
integration, performance validation, and error recovery scenarios.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from benchmark.interfaces.evaluation_interfaces import (
    EvaluationRequest,
    EvaluationResult,
    MetricType,
)
from benchmark.services.evaluation_service import EvaluationService
from tests.fixtures.evaluation_test_scenarios import (
    CybersecurityTestData,
    PerformanceBenchmark,
    TestScenario,
    create_evaluation_request,
    create_mock_evaluation_service,
    create_test_scenarios,
    generate_cybersecurity_test_data,
    get_test_data_for_dataset,
    validate_scenario_results,
)


class TestEvaluationServiceIntegration:
    """Comprehensive integration tests for evaluation service."""

    @pytest.fixture(scope="class")
    def cybersecurity_test_data(self) -> CybersecurityTestData:
        """Generate realistic cybersecurity test data for all tests."""
        return generate_cybersecurity_test_data(sample_count=500)

    @pytest.fixture(scope="class")
    def test_scenarios(self) -> list[TestScenario]:
        """Get all test scenarios for comprehensive testing."""
        return create_test_scenarios()

    @pytest_asyncio.fixture
    async def evaluation_service(self) -> EvaluationService:
        """Create a clean evaluation service instance for each test."""
        service = EvaluationService()

        # Initialize with test configuration
        config = {
            "max_concurrent_evaluations": 8,
            "evaluation_timeout_seconds": 30.0,
            "max_history_size": 200,
        }

        response = await service.initialize(config)
        assert response.success, f"Failed to initialize service: {response.error}"

        yield service

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation(
        self, evaluation_service: EvaluationService, cybersecurity_test_data: CybersecurityTestData
    ) -> None:
        """Test complete evaluation workflow from request to results."""

        # Create and register mock evaluators
        from tests.fixtures.evaluation_test_scenarios import (
            MockAccuracyEvaluator,
            MockPrecisionRecallEvaluator,
        )

        accuracy_evaluator = MockAccuracyEvaluator(base_accuracy=0.88, variance=0.03)
        precision_evaluator = MockPrecisionRecallEvaluator(base_precision=0.85, base_recall=0.82)

        await evaluation_service.register_evaluator(MetricType.ACCURACY, accuracy_evaluator)
        await evaluation_service.register_evaluator(MetricType.PRECISION, precision_evaluator)

        # Create evaluation request
        predictions, ground_truth = get_test_data_for_dataset(
            "network_intrusion_detection", cybersecurity_test_data
        )

        request = EvaluationRequest(
            experiment_id="e2e_test_001",
            model_id="security_bert_v2",
            dataset_id="network_intrusion_detection",
            predictions=predictions[:200],  # Use subset for faster testing
            ground_truth=ground_truth[:200],
            metrics=[MetricType.ACCURACY, MetricType.PRECISION],
            metadata={
                "test_type": "end_to_end",
                "model_version": "2.1.0",
                "dataset_version": "1.0",
            },
        )

        # Execute evaluation
        start_time = time.time()
        result = await evaluation_service.evaluate_predictions(request)
        execution_time = time.time() - start_time

        # Validate results
        assert result is not None, "Evaluation result should not be None"
        assert result.success, f"Evaluation should succeed: {result.error_message}"
        assert result.experiment_id == "e2e_test_001"
        assert result.model_id == "security_bert_v2"
        assert result.dataset_id == "network_intrusion_detection"

        # Check metrics were computed
        assert "accuracy" in result.metrics, "Accuracy metric should be computed"
        assert "precision" in result.metrics, "Precision metric should be computed"
        assert "recall" in result.metrics, "Recall metric should be computed"
        assert "f1_score" in result.metrics, "F1 score metric should be computed"

        # Validate metric ranges
        assert 0.80 <= result.metrics["accuracy"] <= 0.95, (
            f"Accuracy {result.metrics['accuracy']} out of expected range"
        )
        assert 0.75 <= result.metrics["precision"] <= 0.95, (
            f"Precision {result.metrics['precision']} out of expected range"
        )
        assert 0.70 <= result.metrics["recall"] <= 0.95, (
            f"Recall {result.metrics['recall']} out of expected range"
        )

        # Check execution time is reasonable
        assert execution_time < 5.0, f"Evaluation took too long: {execution_time:.2f}s"
        assert result.execution_time_seconds > 0, "Execution time should be recorded"

        # Validate detailed results
        assert "accuracy" in result.detailed_results, (
            "Detailed accuracy results should be available"
        )
        assert "precision" in result.detailed_results, (
            "Detailed precision results should be available"
        )

        # Check metadata preservation
        assert result.metadata == request.metadata, "Metadata should be preserved"

        # Validate timestamp
        result_time = datetime.fromisoformat(result.timestamp)
        assert (datetime.now() - result_time).total_seconds() < 60, "Timestamp should be recent"

    @pytest.mark.asyncio
    async def test_multi_model_evaluation(
        self, evaluation_service: EvaluationService, cybersecurity_test_data: CybersecurityTestData
    ) -> None:
        """Test evaluation across multiple models simultaneously."""

        # Register evaluators
        from tests.fixtures.evaluation_test_scenarios import (
            MockAccuracyEvaluator,
            MockPerformanceEvaluator,
        )

        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())
        await evaluation_service.register_evaluator(
            MetricType.PERFORMANCE, MockPerformanceEvaluator()
        )

        # Define multiple models with different characteristics
        models = [
            {"id": "security_bert", "accuracy": 0.92, "latency": 0.15},
            {"id": "cyber_lstm", "accuracy": 0.88, "latency": 0.08},
            {"id": "threat_cnn", "accuracy": 0.85, "latency": 0.05},
            {"id": "baseline_svm", "accuracy": 0.78, "latency": 0.02},
        ]

        predictions, ground_truth = get_test_data_for_dataset(
            "malware_classification", cybersecurity_test_data
        )

        # Create evaluation requests for all models
        evaluation_tasks = []
        for i, model in enumerate(models):
            request = EvaluationRequest(
                experiment_id=f"multi_model_{i:02d}",
                model_id=model["id"],
                dataset_id="malware_classification",
                predictions=predictions[:150],  # Use subset for each model
                ground_truth=ground_truth[:150],
                metrics=[MetricType.ACCURACY, MetricType.PERFORMANCE],
                metadata={
                    "model_type": model["id"].split("_")[-1],
                    "expected_accuracy": model["accuracy"],
                    "expected_latency": model["latency"],
                },
            )
            evaluation_tasks.append(evaluation_service.evaluate_predictions(request))

        # Execute all evaluations concurrently
        start_time = time.time()
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Validate all evaluations completed
        successful_results = [r for r in results if isinstance(r, EvaluationResult) and r.success]
        assert len(successful_results) == len(models), (
            f"Expected {len(models)} successful results, got {len(successful_results)}"
        )

        # Check that concurrent execution was efficient
        assert total_time < 10.0, f"Multi-model evaluation took too long: {total_time:.2f}s"

        # Validate model diversity in results
        model_ids = {r.model_id for r in successful_results}
        assert len(model_ids) == len(models), "All models should be represented in results"

        # Check accuracy variation across models
        accuracies = [r.metrics["accuracy"] for r in successful_results]
        accuracy_range = max(accuracies) - min(accuracies)
        assert accuracy_range > 0.05, (
            f"Expected accuracy variation across models, got range: {accuracy_range:.3f}"
        )

        # Validate performance metrics
        for result in successful_results:
            assert "inference_latency_ms" in result.metrics, (
                f"Performance metrics missing for {result.model_id}"
            )
            assert result.metrics["inference_latency_ms"] > 0, "Latency should be positive"
            assert "throughput_samples_per_sec" in result.metrics, (
                f"Throughput missing for {result.model_id}"
            )

        # Check evaluation history was updated
        history_response = await evaluation_service.get_evaluation_history(limit=10)
        assert history_response.success, "Should retrieve evaluation history successfully"
        assert history_response.data["total_results"] >= len(models), (
            "History should include all evaluations"
        )

    @pytest.mark.asyncio
    async def test_evaluation_with_real_cybersecurity_data(
        self, evaluation_service: EvaluationService, cybersecurity_test_data: CybersecurityTestData
    ) -> None:
        """Test evaluation using realistic cybersecurity datasets."""

        # Register comprehensive set of evaluators
        from tests.fixtures.evaluation_test_scenarios import (
            MockAccuracyEvaluator,
            MockPerformanceEvaluator,
            MockPrecisionRecallEvaluator,
        )

        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())
        await evaluation_service.register_evaluator(
            MetricType.PRECISION, MockPrecisionRecallEvaluator()
        )
        await evaluation_service.register_evaluator(
            MetricType.PERFORMANCE, MockPerformanceEvaluator()
        )

        # Test different cybersecurity domains
        cybersec_domains = [
            {
                "name": "network_intrusion_detection",
                "data": (
                    cybersecurity_test_data.network_predictions,
                    cybersecurity_test_data.network_ground_truth,
                ),
                "expected_accuracy": 0.85,
                "description": "Network intrusion detection with multi-class classification",
            },
            {
                "name": "malware_classification",
                "data": (
                    cybersecurity_test_data.malware_predictions,
                    cybersecurity_test_data.malware_ground_truth,
                ),
                "expected_accuracy": 0.82,
                "description": "Malware family classification",
            },
            {
                "name": "phishing_detection",
                "data": (
                    cybersecurity_test_data.phishing_predictions,
                    cybersecurity_test_data.phishing_ground_truth,
                ),
                "expected_accuracy": 0.90,
                "description": "Email phishing detection",
            },
            {
                "name": "vulnerability_assessment",
                "data": (
                    cybersecurity_test_data.vulnerability_predictions,
                    cybersecurity_test_data.vulnerability_ground_truth,
                ),
                "expected_accuracy": 0.78,
                "description": "Vulnerability severity assessment",
            },
        ]

        domain_results = []

        for domain in cybersec_domains:
            predictions, ground_truth = domain["data"]

            request = EvaluationRequest(
                experiment_id=f"cybersec_{domain['name']}",
                model_id="cybersec_specialist_v1",
                dataset_id=domain["name"],
                predictions=predictions[:100],  # Use subset for faster testing
                ground_truth=ground_truth[:100],
                metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.PERFORMANCE],
                metadata={
                    "domain": domain["name"],
                    "description": domain["description"],
                    "expected_accuracy": domain["expected_accuracy"],
                    "data_size": len(predictions),
                },
            )

            result = await evaluation_service.evaluate_predictions(request)

            # Validate domain-specific results
            assert result.success, f"Evaluation failed for {domain['name']}: {result.error_message}"
            assert result.dataset_id == domain["name"], "Dataset ID should match domain name"

            # Check accuracy is within reasonable range for domain
            accuracy = result.metrics.get("accuracy", 0.0)
            expected_acc = domain["expected_accuracy"]
            assert abs(accuracy - expected_acc) < 0.15, (
                f"Accuracy {accuracy:.3f} too far from expected {expected_acc:.3f} for {domain['name']}"
            )

            # Validate comprehensive metrics
            assert "precision" in result.metrics, f"Precision missing for {domain['name']}"
            assert "recall" in result.metrics, f"Recall missing for {domain['name']}"
            assert "f1_score" in result.metrics, f"F1 score missing for {domain['name']}"
            assert "inference_latency_ms" in result.metrics, f"Latency missing for {domain['name']}"

            domain_results.append(result)

        # Cross-domain analysis
        assert len(domain_results) == len(cybersec_domains), "Should have results for all domains"

        # Check performance variation across domains
        latencies = [r.metrics["inference_latency_ms"] for r in domain_results]
        assert all(lat > 0 for lat in latencies), "All latencies should be positive"

        # Validate accuracy distribution makes sense for cybersecurity
        accuracies = [r.metrics["accuracy"] for r in domain_results]
        assert all(0.70 <= acc <= 0.98 for acc in accuracies), (
            "Accuracies should be in realistic range for cybersecurity"
        )

        # Check that different domains show different performance characteristics
        accuracy_variance = max(accuracies) - min(accuracies)
        assert accuracy_variance > 0.05, (
            f"Expected variation in accuracy across domains: {accuracy_variance:.3f}"
        )

    @pytest.mark.asyncio
    async def test_results_storage_and_retrieval(
        self, evaluation_service: EvaluationService, cybersecurity_test_data: CybersecurityTestData
    ) -> None:
        """Test complete results lifecycle including storage and retrieval."""

        # Register evaluator
        from tests.fixtures.evaluation_test_scenarios import MockAccuracyEvaluator

        await evaluation_service.register_evaluator(MetricType.ACCURACY, MockAccuracyEvaluator())

        # Generate multiple evaluation results with different characteristics
        test_evaluations = [
            {
                "experiment_id": "storage_test_001",
                "model_id": "model_alpha",
                "dataset_id": "dataset_1",
                "expected_accuracy": 0.90,
            },
            {
                "experiment_id": "storage_test_002",
                "model_id": "model_beta",
                "dataset_id": "dataset_1",
                "expected_accuracy": 0.85,
            },
            {
                "experiment_id": "storage_test_003",
                "model_id": "model_alpha",
                "dataset_id": "dataset_2",
                "expected_accuracy": 0.88,
            },
            {
                "experiment_id": "storage_test_004",
                "model_id": "model_gamma",
                "dataset_id": "dataset_2",
                "expected_accuracy": 0.82,
            },
        ]

        stored_results = []
        predictions, ground_truth = get_test_data_for_dataset(
            "network_intrusion_detection", cybersecurity_test_data
        )

        # Execute and store multiple evaluations
        for eval_config in test_evaluations:
            request = EvaluationRequest(
                experiment_id=eval_config["experiment_id"],
                model_id=eval_config["model_id"],
                dataset_id=eval_config["dataset_id"],
                predictions=predictions[:50],
                ground_truth=ground_truth[:50],
                metrics=[MetricType.ACCURACY],
                metadata={
                    "test_batch": "storage_retrieval_test",
                    "expected_accuracy": eval_config["expected_accuracy"],
                },
            )

            result = await evaluation_service.evaluate_predictions(request)
            assert result.success, f"Evaluation should succeed for {eval_config['experiment_id']}"
            stored_results.append(result)

            # Small delay to ensure different timestamps
            await asyncio.sleep(0.01)

        # Test comprehensive history retrieval
        all_history = await evaluation_service.get_evaluation_history(limit=100)
        assert all_history.success, "Should retrieve evaluation history successfully"
        assert all_history.data["total_results"] >= len(test_evaluations), (
            "History should include all evaluations"
        )

        retrieved_results = all_history.data["results"]
        stored_experiment_ids = {r.experiment_id for r in stored_results}
        retrieved_experiment_ids = {r.experiment_id for r in retrieved_results}

        assert stored_experiment_ids.issubset(retrieved_experiment_ids), (
            "All stored results should be retrievable"
        )

        # Test filtered retrieval by experiment ID
        specific_experiment = test_evaluations[0]["experiment_id"]
        experiment_history = await evaluation_service.get_evaluation_history(
            experiment_id=specific_experiment
        )
        assert experiment_history.success, "Should retrieve specific experiment history"

        experiment_results = experiment_history.data["results"]
        assert len(experiment_results) == 1, (
            "Should retrieve exactly one result for specific experiment"
        )
        assert experiment_results[0].experiment_id == specific_experiment, (
            "Retrieved result should match requested experiment"
        )

        # Test filtered retrieval by model ID
        model_alpha_history = await evaluation_service.get_evaluation_history(
            model_id="model_alpha", limit=10
        )
        assert model_alpha_history.success, "Should retrieve model-specific history"

        model_alpha_results = model_alpha_history.data["results"]
        alpha_experiment_ids = {r.experiment_id for r in model_alpha_results}
        expected_alpha_ids = {"storage_test_001", "storage_test_003"}

        assert expected_alpha_ids.issubset(alpha_experiment_ids), (
            "Should retrieve all results for model_alpha"
        )

        # Test filtered retrieval by dataset ID
        dataset_1_history = await evaluation_service.get_evaluation_history(dataset_id="dataset_1")
        assert dataset_1_history.success, "Should retrieve dataset-specific history"

        dataset_1_results = dataset_1_history.data["results"]
        dataset_1_experiment_ids = {r.experiment_id for r in dataset_1_results}
        expected_dataset_1_ids = {"storage_test_001", "storage_test_002"}

        assert expected_dataset_1_ids.issubset(dataset_1_experiment_ids), (
            "Should retrieve all results for dataset_1"
        )

        # Test evaluation summary generation
        summary = await evaluation_service.get_evaluation_summary(days_back=1)
        assert summary.success, "Should generate evaluation summary successfully"

        summary_data = summary.data
        assert summary_data["total_evaluations"] >= len(test_evaluations), (
            "Summary should include all recent evaluations"
        )
        assert summary_data["successful_evaluations"] >= len(test_evaluations), (
            "All test evaluations should be successful"
        )
        assert summary_data["failed_evaluations"] == 0, "No evaluations should fail in this test"

        # Validate metric summaries
        assert "accuracy" in summary_data["metric_summaries"], (
            "Summary should include accuracy statistics"
        )
        accuracy_stats = summary_data["metric_summaries"]["accuracy"]
        assert "mean" in accuracy_stats, "Accuracy summary should include mean"
        assert "min" in accuracy_stats, "Accuracy summary should include min"
        assert "max" in accuracy_stats, "Accuracy summary should include max"
        assert "count" in accuracy_stats, "Accuracy summary should include count"

        assert accuracy_stats["count"] >= len(test_evaluations), (
            "Accuracy count should include all evaluations"
        )
        assert 0.7 <= accuracy_stats["mean"] <= 1.0, "Mean accuracy should be realistic"

    @pytest.mark.asyncio
    async def test_performance_under_load(
        self, evaluation_service: EvaluationService, cybersecurity_test_data: CybersecurityTestData
    ) -> None:
        """Test evaluation service performance under load."""

        # Register fast mock evaluators
        from tests.fixtures.evaluation_test_scenarios import (
            MockAccuracyEvaluator,
            MockPerformanceEvaluator,
        )

        fast_accuracy = MockAccuracyEvaluator(base_accuracy=0.85, variance=0.02)
        fast_performance = MockPerformanceEvaluator(base_latency=0.05)  # Fast evaluator

        await evaluation_service.register_evaluator(MetricType.ACCURACY, fast_accuracy)
        await evaluation_service.register_evaluator(MetricType.PERFORMANCE, fast_performance)

        # Create performance benchmark
        benchmark = PerformanceBenchmark()

        # Generate high-throughput evaluation workload
        num_concurrent_evaluations = 12
        predictions, ground_truth = get_test_data_for_dataset(
            "large_network_dataset", cybersecurity_test_data
        )

        evaluation_tasks = []

        benchmark.start()

        for i in range(num_concurrent_evaluations):
            # Create staggered subset of data for each evaluation
            start_idx = (i * 100) % len(predictions)
            end_idx = start_idx + 200

            eval_predictions = predictions[start_idx:end_idx]
            eval_ground_truth = ground_truth[start_idx:end_idx]

            request = EvaluationRequest(
                experiment_id=f"perf_test_{i:03d}",
                model_id=f"fast_detector_{i % 3 + 1}",  # Cycle through 3 models
                dataset_id="large_network_dataset",
                predictions=eval_predictions,
                ground_truth=eval_ground_truth,
                metrics=[MetricType.ACCURACY, MetricType.PERFORMANCE],
                metadata={
                    "test_type": "performance_load_test",
                    "batch_id": i,
                    "data_size": len(eval_predictions),
                },
            )

            evaluation_tasks.append(evaluation_service.evaluate_predictions(request))

        # Execute all evaluations concurrently
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        benchmark.stop()

        # Analyze results
        successful_results = [r for r in results if isinstance(r, EvaluationResult) and r.success]
        failed_results = [
            r
            for r in results
            if isinstance(r, Exception) or (isinstance(r, EvaluationResult) and not r.success)
        ]

        # Performance validations
        total_duration = benchmark.duration
        assert total_duration < 15.0, f"Load test took too long: {total_duration:.2f}s"

        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.8, f"Success rate too low under load: {success_rate:.2%}"

        # Throughput validation
        throughput = len(successful_results) / total_duration
        assert throughput >= 1.0, f"Throughput too low: {throughput:.1f} evaluations/sec"

        # Check that concurrent evaluation limit was respected
        concurrent_limit = evaluation_service.max_concurrent_evaluations
        assert len(results) > concurrent_limit, (
            "Should attempt more evaluations than concurrent limit"
        )

        # Validate individual result quality under load
        for result in successful_results[:5]:  # Check first 5 results
            assert "accuracy" in result.metrics, "Accuracy should be computed under load"
            assert "inference_latency_ms" in result.metrics, (
                "Performance metrics should be available under load"
            )
            assert result.execution_time_seconds > 0, "Execution time should be recorded"
            assert 0.70 <= result.metrics["accuracy"] <= 1.0, (
                "Accuracy should be reasonable under load"
            )

        # Check evaluation history performance
        history_response = await evaluation_service.get_evaluation_history(limit=20)
        assert history_response.success, "Should retrieve history efficiently after load test"
        assert len(history_response.data["results"]) >= 15, "History should contain recent results"

        # Generate performance summary
        benchmark.get_performance_summary()

        # Log performance metrics for analysis
        print("\n--- Performance Load Test Results ---")
        print(f"Total evaluations: {len(results)}")
        print(f"Successful evaluations: {len(successful_results)}")
        print(f"Failed evaluations: {len(failed_results)}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Throughput: {throughput:.1f} evaluations/sec")
        print(f"Success rate: {success_rate:.1%}")
        print(
            f"Average execution time: {sum(r.execution_time_seconds for r in successful_results) / len(successful_results):.3f}s"
        )

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, evaluation_service: EvaluationService, cybersecurity_test_data: CybersecurityTestData
    ) -> None:
        """Test service behavior under various error conditions."""

        # Register mix of working and failing evaluators
        from tests.fixtures.evaluation_test_scenarios import MockAccuracyEvaluator

        # Working evaluator
        working_evaluator = MockAccuracyEvaluator(base_accuracy=0.85)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, working_evaluator)

        # Create failing evaluator
        failing_evaluator = MagicMock()
        failing_evaluator.evaluate = AsyncMock(side_effect=Exception("Simulated evaluator failure"))
        failing_evaluator.get_metric_names.return_value = ["failed_metric"]
        failing_evaluator.get_metric_type.return_value = MetricType.PRECISION
        failing_evaluator.validate_data_compatibility.return_value = True

        await evaluation_service.register_evaluator(MetricType.PRECISION, failing_evaluator)

        # Test 1: Evaluation with partial metric failures
        predictions, ground_truth = get_test_data_for_dataset(
            "error_test_dataset", cybersecurity_test_data
        )

        request_partial_failure = EvaluationRequest(
            experiment_id="error_test_partial_failure",
            model_id="test_model",
            dataset_id="error_test_dataset",
            predictions=predictions,
            ground_truth=ground_truth,
            metrics=[MetricType.ACCURACY, MetricType.PRECISION],  # One works, one fails
            metadata={"test_type": "partial_failure"},
        )

        # Should complete with partial results
        result = await evaluation_service.evaluate_predictions(request_partial_failure)

        assert result is not None, "Should return result even with partial failure"
        assert not result.success, "Overall evaluation should be marked as failed"
        assert result.error_message is not None, "Should have error message"

        # Should have results from working evaluator
        assert "accuracy" in result.metrics, "Should have accuracy from working evaluator"

        # Should have error information in detailed results
        assert "precision" in result.detailed_results, (
            "Should have error details for failing evaluator"
        )
        error_details = result.detailed_results["precision"]
        assert "error" in error_details, "Should record error information"

        # Test 2: Timeout handling
        timeout_evaluator = MagicMock()
        timeout_evaluator.evaluate = AsyncMock(side_effect=asyncio.sleep(100))  # Never completes
        timeout_evaluator.get_metric_names.return_value = ["timeout_metric"]
        timeout_evaluator.get_metric_type.return_value = MetricType.F1_SCORE
        timeout_evaluator.validate_data_compatibility.return_value = True

        await evaluation_service.register_evaluator(MetricType.F1_SCORE, timeout_evaluator)

        request_timeout = EvaluationRequest(
            experiment_id="error_test_timeout",
            model_id="timeout_model",
            dataset_id="error_test_dataset",
            predictions=predictions[:10],
            ground_truth=ground_truth[:10],
            metrics=[MetricType.F1_SCORE],
            metadata={"test_type": "timeout"},
        )

        # Should handle timeout gracefully
        with pytest.raises((asyncio.TimeoutError, ValueError)):
            await evaluation_service.evaluate_predictions(request_timeout)

        # Service should still be functional after timeout
        health_check = await evaluation_service.health_check()
        assert health_check.status.value in ["healthy", "degraded"], (
            "Service should remain operational after timeout"
        )

        # Test 3: Invalid data handling
        invalid_predictions = [{"invalid": "data"}] * 10
        invalid_ground_truth = [{"also_invalid": "data"}] * 10

        request_invalid_data = EvaluationRequest(
            experiment_id="error_test_invalid_data",
            model_id="test_model",
            dataset_id="error_test_dataset",
            predictions=invalid_predictions,
            ground_truth=invalid_ground_truth,
            metrics=[MetricType.ACCURACY],
            metadata={"test_type": "invalid_data"},
        )

        # Should handle data validation failure
        with pytest.raises((ValueError, TypeError)):
            await evaluation_service.evaluate_predictions(request_invalid_data)

        # Test 4: Concurrent evaluation limit handling
        predictions_small, ground_truth_small = predictions[:20], ground_truth[:20]

        # Fill up concurrent evaluation capacity
        concurrent_tasks = []
        max_concurrent = evaluation_service.max_concurrent_evaluations

        # Create blocking evaluator to fill capacity
        blocking_evaluator = MagicMock()
        blocking_evaluator.evaluate = AsyncMock(side_effect=lambda *_: asyncio.sleep(2.0))
        blocking_evaluator.get_metric_names.return_value = ["blocking_metric"]
        blocking_evaluator.get_metric_type.return_value = MetricType.ROC_AUC
        blocking_evaluator.validate_data_compatibility.return_value = True

        await evaluation_service.register_evaluator(MetricType.ROC_AUC, blocking_evaluator)

        # Start maximum concurrent evaluations
        for i in range(max_concurrent):
            request = EvaluationRequest(
                experiment_id=f"concurrent_test_{i}",
                model_id="blocking_model",
                dataset_id="error_test_dataset",
                predictions=predictions_small,
                ground_truth=ground_truth_small,
                metrics=[MetricType.ROC_AUC],
                metadata={"test_type": "concurrent_limit", "batch": i},
            )
            concurrent_tasks.append(evaluation_service.evaluate_predictions(request))

        # Try to start one more (should fail due to limit)
        excess_request = EvaluationRequest(
            experiment_id="excess_evaluation",
            model_id="excess_model",
            dataset_id="error_test_dataset",
            predictions=predictions_small,
            ground_truth=ground_truth_small,
            metrics=[MetricType.ACCURACY],  # Use working evaluator
            metadata={"test_type": "excess_evaluation"},
        )

        # Should be rejected due to concurrent limit
        with pytest.raises(Exception) as exc_info:
            await evaluation_service.evaluate_predictions(excess_request)

        assert (
            "concurrent" in str(exc_info.value).lower() or "limit" in str(exc_info.value).lower()
        ), "Should mention concurrent limit in error"

        # Clean up concurrent tasks
        for task in concurrent_tasks:
            task.cancel()

        # Wait a bit for cleanup
        await asyncio.sleep(0.1)

        # Test 5: Service recovery after errors
        # Service should be able to handle new valid requests after various errors
        recovery_request = EvaluationRequest(
            experiment_id="error_test_recovery",
            model_id="recovery_model",
            dataset_id="error_test_dataset",
            predictions=predictions[:30],
            ground_truth=ground_truth[:30],
            metrics=[MetricType.ACCURACY],  # Use working evaluator
            metadata={"test_type": "recovery_validation"},
        )

        recovery_result = await evaluation_service.evaluate_predictions(recovery_request)

        assert recovery_result.success, (
            f"Service should recover and handle valid requests: {recovery_result.error_message}"
        )
        assert "accuracy" in recovery_result.metrics, (
            "Should compute metrics correctly after recovery"
        )
        assert recovery_result.experiment_id == "error_test_recovery", (
            "Should process request correctly"
        )

        # Validate service health after error scenarios
        final_health = await evaluation_service.health_check()
        assert final_health.status.value != "unhealthy", (
            "Service should not be unhealthy after error recovery tests"
        )

    @pytest.mark.asyncio
    async def test_comprehensive_scenario_validation(
        self,
        evaluation_service: EvaluationService,
        test_scenarios: list[TestScenario],
        cybersecurity_test_data: CybersecurityTestData,
    ) -> None:
        """Test all predefined scenarios with comprehensive validation."""

        scenario_results = {}

        for scenario in test_scenarios[:4]:  # Test first 4 scenarios to keep test time reasonable
            print(f"\nExecuting scenario: {scenario.name}")
            print(f"Description: {scenario.description}")

            # Create mock service for this scenario
            mock_service = await create_mock_evaluation_service(
                scenario, cybersecurity_test_data, introduce_failures=("error" in scenario.name)
            )

            scenario_start_time = time.time()
            evaluation_results = []

            try:
                # Execute evaluations for all model-dataset combinations
                for model_id in scenario.models:
                    for dataset_name in scenario.datasets:
                        request = create_evaluation_request(
                            model_id=model_id,
                            dataset_name=dataset_name,
                            test_data=cybersecurity_test_data,
                            metrics=scenario.metrics,
                        )

                        result = await mock_service.evaluate_predictions(request)
                        evaluation_results.append(result)

                scenario_duration = time.time() - scenario_start_time

                # Validate scenario results
                validation_results = await validate_scenario_results(scenario, evaluation_results)

                scenario_results[scenario.name] = {
                    "success": True,
                    "duration": scenario_duration,
                    "results_count": len(evaluation_results),
                    "validations": validation_results,
                    "within_time_limit": scenario_duration
                    <= scenario.expected_duration * 2,  # Allow 2x expected time
                }

                # Scenario-specific assertions
                if scenario.name == "single_model_basic":
                    assert len(evaluation_results) == 1, (
                        "Single model scenario should have 1 result"
                    )
                    assert evaluation_results[0].success, "Single model evaluation should succeed"

                elif scenario.name == "multi_model_comparison":
                    successful_results = [r for r in evaluation_results if r.success]
                    assert len(successful_results) >= 3, "Most model comparisons should succeed"

                    # Check model diversity
                    model_ids = {r.model_id for r in successful_results}
                    assert len(model_ids) >= 3, "Should have results from multiple models"

                elif scenario.name == "cross_domain_evaluation":
                    assert len(evaluation_results) == 8, (
                        "Cross-domain should have 2 models × 4 datasets = 8 results"
                    )

                    # Check domain coverage
                    dataset_ids = {r.dataset_id for r in evaluation_results}
                    assert len(dataset_ids) == 4, "Should cover all 4 domains"

                elif scenario.name == "performance_stress_test":
                    # Check performance metrics are available
                    perf_results = [
                        r for r in evaluation_results if "inference_latency_ms" in r.metrics
                    ]
                    assert len(perf_results) > 0, "Performance stress test should measure latency"

                print(f"✅ Scenario {scenario.name} completed successfully")
                print(
                    f"   Duration: {scenario_duration:.2f}s (expected: {scenario.expected_duration:.2f}s)"
                )
                print(f"   Results: {len(evaluation_results)}")
                print(
                    f"   Validations passed: {sum(validation_results.values())}/{len(validation_results)}"
                )

            except Exception as e:
                print(f"❌ Scenario {scenario.name} failed: {e}")
                scenario_results[scenario.name] = {
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - scenario_start_time,
                }

            finally:
                # Cleanup mock service
                await mock_service.shutdown()

        # Overall scenario validation
        successful_scenarios = [
            name for name, result in scenario_results.items() if result["success"]
        ]
        total_scenarios = len(scenario_results)

        assert len(successful_scenarios) >= total_scenarios * 0.75, (
            f"At least 75% of scenarios should succeed. Passed: {len(successful_scenarios)}/{total_scenarios}"
        )

        print("\n🎯 Scenario Summary:")
        print(f"   Total scenarios: {total_scenarios}")
        print(f"   Successful: {len(successful_scenarios)}")
        print(f"   Success rate: {len(successful_scenarios) / total_scenarios:.1%}")

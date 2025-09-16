"""
Comprehensive evaluation scenarios demonstrating complete evaluation service integration.

This test suite validates complex, realistic evaluation workflows that showcase
the full capabilities of the evaluation service with cybersecurity-focused test cases.
"""

import asyncio
from datetime import datetime

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
    PerformanceBenchmark,
    generate_cybersecurity_test_data,
    get_test_data_for_dataset,
)


class TestComprehensiveEvaluationScenarios:
    """Test comprehensive, realistic evaluation scenarios."""

    @pytest_asyncio.fixture
    async def full_evaluation_service(self):
        """Create a fully configured evaluation service with all evaluators."""
        service = EvaluationService()

        # Initialize with production-like configuration
        config = {
            "max_concurrent_evaluations": 8,
            "evaluation_timeout_seconds": 30.0,
            "max_history_size": 200,
        }

        response = await service.initialize(config)
        assert response.success, f"Failed to initialize service: {response.error}"

        # Register all evaluators
        await service.register_evaluator(
            MetricType.ACCURACY, MockAccuracyEvaluator(base_accuracy=0.88, variance=0.05)
        )
        await service.register_evaluator(
            MetricType.PRECISION,
            MockPrecisionRecallEvaluator(base_precision=0.85, base_recall=0.82),
        )
        await service.register_evaluator(
            MetricType.PERFORMANCE, MockPerformanceEvaluator(base_latency=0.12)
        )

        yield service

        await service.shutdown()

    @pytest.fixture(scope="class")
    def comprehensive_test_data(self):
        """Generate comprehensive test data for all scenarios."""
        return generate_cybersecurity_test_data(sample_count=500)

    @pytest.mark.asyncio
    async def test_cybersecurity_model_comparison_scenario(
        self, full_evaluation_service, comprehensive_test_data
    ):
        """Test comprehensive cybersecurity model comparison across multiple domains."""

        # Define realistic cybersecurity models with different characteristics
        cybersec_models = [
            {
                "id": "SecurityBERT_v2",
                "description": "Advanced transformer model fine-tuned for cybersecurity",
                "expected_domains": ["network", "malware", "phishing"],
                "expected_accuracy": 0.90,
            },
            {
                "id": "CyberLSTM_Pro",
                "description": "LSTM-based model specialized for sequential threat analysis",
                "expected_domains": ["network", "vulnerability"],
                "expected_accuracy": 0.85,
            },
            {
                "id": "ThreatCNN_Fast",
                "description": "Lightweight CNN optimized for real-time threat detection",
                "expected_domains": ["malware", "network"],
                "expected_accuracy": 0.82,
            },
            {
                "id": "EnsembleDefender",
                "description": "Ensemble model combining multiple detection approaches",
                "expected_domains": ["network", "malware", "phishing", "vulnerability"],
                "expected_accuracy": 0.92,
            },
        ]

        # Define cybersecurity datasets
        cybersec_datasets = [
            {"name": "network_intrusion_detection", "type": "network", "samples": 200},
            {"name": "malware_classification", "type": "malware", "samples": 200},
            {"name": "phishing_detection", "type": "phishing", "samples": 200},
            {"name": "vulnerability_assessment", "type": "vulnerability", "samples": 200},
        ]

        benchmark = PerformanceBenchmark()
        benchmark.start()

        # Execute comprehensive model-dataset evaluation matrix
        evaluation_tasks = []

        for model in cybersec_models:
            for dataset in cybersec_datasets:
                # Get appropriate test data for dataset
                predictions, ground_truth = get_test_data_for_dataset(
                    dataset["name"], comprehensive_test_data
                )

                # Use subset of data for reasonable test execution time
                sample_size = min(dataset["samples"], len(predictions))
                eval_predictions = predictions[:sample_size]
                eval_ground_truth = ground_truth[:sample_size]

                request = EvaluationRequest(
                    experiment_id=f"cybersec_{model['id']}_{dataset['name']}",
                    model_id=model["id"],
                    dataset_id=dataset["name"],
                    predictions=eval_predictions,
                    ground_truth=eval_ground_truth,
                    metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.PERFORMANCE],
                    metadata={
                        "scenario": "cybersecurity_model_comparison",
                        "model_description": model["description"],
                        "dataset_type": dataset["type"],
                        "expected_accuracy": model["expected_accuracy"],
                        "sample_count": sample_size,
                    },
                )

                evaluation_tasks.append(full_evaluation_service.evaluate_predictions(request))

        # Execute all evaluations with controlled concurrency
        results = []
        batch_size = 6  # Process in batches to avoid overwhelming the service

        for i in range(0, len(evaluation_tasks), batch_size):
            batch = evaluation_tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)

            # Brief pause between batches
            if i + batch_size < len(evaluation_tasks):
                await asyncio.sleep(0.1)

        benchmark.stop()

        # Analyze results comprehensively
        successful_results = [r for r in results if hasattr(r, "success") and r.success]
        failed_results = [
            r
            for r in results
            if isinstance(r, Exception) or (hasattr(r, "success") and not r.success)
        ]

        print("\n🔍 Cybersecurity Model Comparison Results:")
        print(f"   Total evaluations: {len(results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(failed_results)}")
        print(f"   Success rate: {len(successful_results) / len(results):.1%}")
        print(f"   Duration: {benchmark.duration:.2f}s")
        print(f"   Throughput: {len(successful_results) / benchmark.duration:.1f} evaluations/sec")

        # Validate comprehensive results
        assert len(successful_results) >= len(results) * 0.85, (
            f"At least 85% of evaluations should succeed: {len(successful_results)}/{len(results)}"
        )
        assert benchmark.duration < 60.0, (
            f"Comprehensive evaluation should complete in reasonable time: {benchmark.duration:.2f}s"
        )

        # Analyze performance by model
        model_performance = {}
        for result in successful_results:
            model_id = result.model_id
            if model_id not in model_performance:
                model_performance[model_id] = {
                    "accuracies": [],
                    "precisions": [],
                    "latencies": [],
                    "datasets": [],
                }

            model_performance[model_id]["accuracies"].append(result.metrics.get("accuracy", 0.0))
            model_performance[model_id]["precisions"].append(result.metrics.get("precision", 0.0))
            model_performance[model_id]["latencies"].append(
                result.metrics.get("inference_latency_ms", 0.0)
            )
            model_performance[model_id]["datasets"].append(result.dataset_id)

        print("\n📊 Model Performance Analysis:")
        for model_id, perf in model_performance.items():
            avg_accuracy = (
                sum(perf["accuracies"]) / len(perf["accuracies"]) if perf["accuracies"] else 0.0
            )
            avg_precision = (
                sum(perf["precisions"]) / len(perf["precisions"]) if perf["precisions"] else 0.0
            )
            avg_latency = (
                sum(perf["latencies"]) / len(perf["latencies"]) if perf["latencies"] else 0.0
            )

            print(
                f"   {model_id:20}: Accuracy={avg_accuracy:.3f}, Precision={avg_precision:.3f}, Latency={avg_latency:.1f}ms, Datasets={len(perf['datasets'])}"
            )

            # Validate model performed reasonably across domains
            assert len(perf["datasets"]) >= 2, (
                f"Model {model_id} should be evaluated on multiple datasets"
            )
            assert avg_accuracy >= 0.70, f"Model {model_id} accuracy too low: {avg_accuracy:.3f}"

        # Analyze performance by dataset type
        dataset_performance = {}
        for result in successful_results:
            dataset_type = result.metadata.get("dataset_type", "unknown")
            if dataset_type not in dataset_performance:
                dataset_performance[dataset_type] = {
                    "accuracies": [],
                    "models": set(),
                }

            dataset_performance[dataset_type]["accuracies"].append(
                result.metrics.get("accuracy", 0.0)
            )
            dataset_performance[dataset_type]["models"].add(result.model_id)

        print("\n🎯 Dataset Type Analysis:")
        for dataset_type, perf in dataset_performance.items():
            avg_accuracy = sum(perf["accuracies"]) / len(perf["accuracies"])
            print(
                f"   {dataset_type:15}: Average Accuracy={avg_accuracy:.3f}, Models Tested={len(perf['models'])}"
            )

            # Validate dataset had multiple models tested
            assert len(perf["models"]) >= 2, (
                f"Dataset type {dataset_type} should have multiple models tested"
            )

        # Validate expected model rankings
        model_avg_accuracies = {
            model_id: sum(perf["accuracies"]) / len(perf["accuracies"])
            for model_id, perf in model_performance.items()
            if perf["accuracies"]
        }

        # EnsembleDefender should perform best (if evaluated successfully)
        if "EnsembleDefender" in model_avg_accuracies and len(model_avg_accuracies) > 1:
            ensemble_accuracy = model_avg_accuracies["EnsembleDefender"]
            other_accuracies = [
                acc for model, acc in model_avg_accuracies.items() if model != "EnsembleDefender"
            ]
            max_other_accuracy = max(other_accuracies) if other_accuracies else 0.0

            # Allow for some variance in mock results
            assert ensemble_accuracy >= max_other_accuracy - 0.05, (
                f"EnsembleDefender should perform competitively: {ensemble_accuracy:.3f} vs {max_other_accuracy:.3f}"
            )

    @pytest.mark.asyncio
    async def test_high_throughput_evaluation_workflow(
        self, full_evaluation_service, comprehensive_test_data
    ):
        """Test high-throughput evaluation workflow with performance monitoring."""

        # Create high-throughput scenario
        models = [f"HighPerf_Model_{i}" for i in range(1, 6)]  # 5 models
        datasets = ["network_intrusion_detection", "malware_classification"]  # 2 datasets

        benchmark = PerformanceBenchmark()
        benchmark.start()

        # Create evaluation requests for high throughput
        evaluation_requests = []
        request_count = 0

        for model_id in models:
            for dataset_name in datasets:
                predictions, ground_truth = get_test_data_for_dataset(
                    dataset_name, comprehensive_test_data
                )

                # Use smaller samples for higher throughput
                sample_size = 50
                eval_predictions = predictions[:sample_size]
                eval_ground_truth = ground_truth[:sample_size]

                request = EvaluationRequest(
                    experiment_id=f"throughput_test_{request_count:03d}",
                    model_id=model_id,
                    dataset_id=dataset_name,
                    predictions=eval_predictions,
                    ground_truth=eval_ground_truth,
                    metrics=[MetricType.ACCURACY, MetricType.PERFORMANCE],
                    metadata={
                        "scenario": "high_throughput_test",
                        "batch_number": request_count // 5,
                        "sample_size": sample_size,
                    },
                )
                evaluation_requests.append(request)
                request_count += 1

        # Execute with high concurrency
        total_requests = len(evaluation_requests)
        concurrent_limit = min(8, full_evaluation_service.max_concurrent_evaluations)

        results = []
        # Process in batches with maximum concurrency
        for i in range(0, total_requests, concurrent_limit):
            batch_requests = evaluation_requests[i : i + concurrent_limit]
            batch_tasks = [
                full_evaluation_service.evaluate_predictions(req) for req in batch_requests
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)

        benchmark.stop()

        # Analyze throughput performance
        successful_results = [r for r in results if hasattr(r, "success") and r.success]
        failed_results = [
            r
            for r in results
            if isinstance(r, Exception) or (hasattr(r, "success") and not r.success)
        ]

        total_samples_processed = sum(
            result.metadata.get("sample_size", 0)
            for result in successful_results
            if isinstance(result.metadata, dict)
        )

        throughput_evaluations_per_sec = len(successful_results) / benchmark.duration
        throughput_samples_per_sec = total_samples_processed / benchmark.duration

        print("\n⚡ High Throughput Evaluation Results:")
        print(f"   Total requests: {total_requests}")
        print(f"   Successful evaluations: {len(successful_results)}")
        print(f"   Failed evaluations: {len(failed_results)}")
        print(f"   Success rate: {len(successful_results) / total_requests:.1%}")
        print(f"   Duration: {benchmark.duration:.2f}s")
        print(f"   Throughput: {throughput_evaluations_per_sec:.1f} evaluations/sec")
        print(f"   Sample throughput: {throughput_samples_per_sec:.0f} samples/sec")
        print(f"   Total samples processed: {total_samples_processed}")

        # Validate throughput performance
        assert len(successful_results) >= total_requests * 0.90, (
            f"High success rate required for throughput test: {len(successful_results)}/{total_requests}"
        )
        assert benchmark.duration < 30.0, (
            f"High throughput test should complete quickly: {benchmark.duration:.2f}s"
        )
        assert throughput_evaluations_per_sec >= 1.5, (
            f"Should achieve reasonable evaluation throughput: {throughput_evaluations_per_sec:.1f}/sec"
        )

        # Validate individual evaluation quality under load
        accuracy_values = [
            r.metrics.get("accuracy", 0.0) for r in successful_results if "accuracy" in r.metrics
        ]
        if accuracy_values:
            avg_accuracy = sum(accuracy_values) / len(accuracy_values)
            min_accuracy = min(accuracy_values)
            max_accuracy = max(accuracy_values)

            print(
                f"   Accuracy under load: avg={avg_accuracy:.3f}, min={min_accuracy:.3f}, max={max_accuracy:.3f}"
            )

            assert avg_accuracy >= 0.75, (
                f"Average accuracy should remain reasonable under load: {avg_accuracy:.3f}"
            )
            assert min_accuracy >= 0.60, (
                f"Minimum accuracy should not degrade too much under load: {min_accuracy:.3f}"
            )

        # Check performance metrics are available
        latency_values = [
            r.metrics.get("inference_latency_ms", 0.0)
            for r in successful_results
            if "inference_latency_ms" in r.metrics
        ]
        if latency_values:
            avg_latency = sum(latency_values) / len(latency_values)
            print(f"   Average inference latency: {avg_latency:.1f}ms")

            assert avg_latency > 0, "Should measure inference latency"
            assert avg_latency < 500, f"Average latency should be reasonable: {avg_latency:.1f}ms"

    @pytest.mark.asyncio
    async def test_evaluation_service_resilience_scenario(
        self, full_evaluation_service, comprehensive_test_data
    ):
        """Test evaluation service resilience under stress and error conditions."""

        # Create mixed workload with some problematic requests
        valid_requests = []
        problematic_requests = []

        # Create valid requests
        for i in range(5):
            predictions, ground_truth = get_test_data_for_dataset(
                "network_intrusion_detection", comprehensive_test_data
            )

            request = EvaluationRequest(
                experiment_id=f"resilience_valid_{i}",
                model_id=f"reliable_model_{i}",
                dataset_id="network_intrusion_detection",
                predictions=predictions[:30],
                ground_truth=ground_truth[:30],
                metrics=[MetricType.ACCURACY, MetricType.PRECISION],
                metadata={"request_type": "valid", "batch": i},
            )
            valid_requests.append(request)

        # Create requests with challenging data (but still valid)
        for i in range(3):
            # Create requests with minimal data (but still valid)
            predictions, ground_truth = get_test_data_for_dataset(
                "malware_classification", comprehensive_test_data
            )

            request = EvaluationRequest(
                experiment_id=f"resilience_minimal_{i}",
                model_id=f"stress_model_{i}",
                dataset_id="malware_classification",
                predictions=predictions[:5],  # Very small dataset
                ground_truth=ground_truth[:5],
                metrics=[MetricType.ACCURACY],
                metadata={"request_type": "minimal", "batch": i},
            )
            problematic_requests.append(request)

        # Mix all requests
        all_requests = valid_requests + problematic_requests

        # Execute with controlled stress
        benchmark = PerformanceBenchmark()
        benchmark.start()

        # Submit all requests concurrently to stress the service
        tasks = [full_evaluation_service.evaluate_predictions(req) for req in all_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        benchmark.stop()

        # Analyze resilience
        successful_results = [r for r in results if hasattr(r, "success") and r.success]
        failed_results = [
            r
            for r in results
            if isinstance(r, Exception) or (hasattr(r, "success") and not r.success)
        ]
        exception_results = [r for r in results if isinstance(r, Exception)]

        print("\n🛡️ Service Resilience Test Results:")
        print(f"   Total requests: {len(all_requests)}")
        print(f"   Successful evaluations: {len(successful_results)}")
        print(f"   Failed evaluations: {len(failed_results)}")
        print(f"   Exceptions: {len(exception_results)}")
        print(f"   Success rate: {len(successful_results) / len(all_requests):.1%}")
        print(f"   Duration: {benchmark.duration:.2f}s")

        # Service should handle most requests successfully
        assert len(successful_results) >= len(all_requests) * 0.75, (
            f"Service should handle most requests under stress: {len(successful_results)}/{len(all_requests)}"
        )

        # Should complete in reasonable time even under stress
        assert benchmark.duration < 15.0, (
            f"Resilience test should complete reasonably quickly: {benchmark.duration:.2f}s"
        )

        # Analyze successful results by request type
        valid_successes = [
            r for r in successful_results if r.metadata.get("request_type") == "valid"
        ]
        minimal_successes = [
            r for r in successful_results if r.metadata.get("request_type") == "minimal"
        ]

        print(f"   Valid request successes: {len(valid_successes)}/{len(valid_requests)}")
        print(f"   Minimal request successes: {len(minimal_successes)}/{len(problematic_requests)}")

        # Most valid requests should succeed
        assert len(valid_successes) >= len(valid_requests) * 0.8, (
            f"Most valid requests should succeed: {len(valid_successes)}/{len(valid_requests)}"
        )

        # Service should remain healthy after stress test
        health_check = await full_evaluation_service.health_check()
        assert health_check.status in ["healthy", "degraded"], (
            f"Service should remain operational after stress: {health_check.status}"
        )

        # Should be able to process new requests after stress
        recovery_request = EvaluationRequest(
            experiment_id="resilience_recovery_test",
            model_id="recovery_model",
            dataset_id="network_intrusion_detection",
            predictions=comprehensive_test_data.network_predictions[:20],
            ground_truth=comprehensive_test_data.network_ground_truth[:20],
            metrics=[MetricType.ACCURACY],
            metadata={"test_type": "recovery"},
        )

        recovery_result = await full_evaluation_service.evaluate_predictions(recovery_request)
        assert recovery_result.success, (
            f"Service should recover and handle new requests: {recovery_result.error_message if hasattr(recovery_result, 'error_message') else 'Unknown error'}"
        )

    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_history_analysis(
        self, full_evaluation_service, comprehensive_test_data
    ):
        """Test comprehensive evaluation history tracking and analysis capabilities."""

        # Generate diverse evaluation history
        evaluation_scenarios = [
            {"model": "HistoryModel_A", "dataset": "network_intrusion_detection", "samples": 40},
            {"model": "HistoryModel_A", "dataset": "malware_classification", "samples": 35},
            {"model": "HistoryModel_B", "dataset": "network_intrusion_detection", "samples": 45},
            {"model": "HistoryModel_B", "dataset": "phishing_detection", "samples": 30},
            {"model": "HistoryModel_C", "dataset": "vulnerability_assessment", "samples": 25},
        ]

        # Execute evaluations to build history
        history_results = []

        for i, scenario in enumerate(evaluation_scenarios):
            predictions, ground_truth = get_test_data_for_dataset(
                scenario["dataset"], comprehensive_test_data
            )

            request = EvaluationRequest(
                experiment_id=f"history_test_{i:02d}",
                model_id=scenario["model"],
                dataset_id=scenario["dataset"],
                predictions=predictions[: scenario["samples"]],
                ground_truth=ground_truth[: scenario["samples"]],
                metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.PERFORMANCE],
                metadata={
                    "scenario": "history_analysis",
                    "execution_order": i,
                    "expected_samples": scenario["samples"],
                },
            )

            result = await full_evaluation_service.evaluate_predictions(request)
            assert result.success, (
                f"History building evaluation {i} should succeed: {result.error_message}"
            )
            history_results.append(result)

            # Small delay to ensure different timestamps
            await asyncio.sleep(0.01)

        # Test comprehensive history retrieval
        all_history = await full_evaluation_service.get_evaluation_history(limit=20)
        assert all_history.success, "Should retrieve complete evaluation history"

        retrieved_results = all_history.data["results"]
        assert len(retrieved_results) >= len(evaluation_scenarios), (
            f"History should include all recent evaluations: {len(retrieved_results)} >= {len(evaluation_scenarios)}"
        )

        # Test filtered history by model
        for model_id in ["HistoryModel_A", "HistoryModel_B", "HistoryModel_C"]:
            model_history = await full_evaluation_service.get_evaluation_history(
                model_id=model_id, limit=10
            )
            assert model_history.success, f"Should retrieve history for {model_id}"

            model_results = model_history.data["results"]
            expected_count = len([s for s in evaluation_scenarios if s["model"] == model_id])

            assert len(model_results) >= expected_count, (
                f"Should retrieve all evaluations for {model_id}: {len(model_results)} >= {expected_count}"
            )

            # Validate all results are for the correct model
            for result in model_results:
                assert result.model_id == model_id, f"All results should be for model {model_id}"

        # Test filtered history by dataset
        dataset_types = [
            "network_intrusion_detection",
            "malware_classification",
            "phishing_detection",
            "vulnerability_assessment",
        ]

        for dataset_id in dataset_types:
            dataset_history = await full_evaluation_service.get_evaluation_history(
                dataset_id=dataset_id, limit=10
            )
            assert dataset_history.success, f"Should retrieve history for {dataset_id}"

            dataset_results = dataset_history.data["results"]
            expected_count = len([s for s in evaluation_scenarios if s["dataset"] == dataset_id])

            if expected_count > 0:
                assert len(dataset_results) >= expected_count, (
                    f"Should retrieve evaluations for {dataset_id}: {len(dataset_results)} >= {expected_count}"
                )

                # Validate all results are for the correct dataset
                for result in dataset_results:
                    assert result.dataset_id == dataset_id, (
                        f"All results should be for dataset {dataset_id}"
                    )

        # Test comprehensive evaluation summary
        summary_response = await full_evaluation_service.get_evaluation_summary(days_back=1)
        assert summary_response.success, "Should generate comprehensive evaluation summary"

        summary_data = summary_response.data

        print("\n📈 Evaluation History Analysis:")
        print(f"   Total evaluations in history: {summary_data['total_evaluations']}")
        print(f"   Successful evaluations: {summary_data['successful_evaluations']}")
        print(f"   Failed evaluations: {summary_data['failed_evaluations']}")
        print(
            f"   Success rate: {summary_data['successful_evaluations'] / summary_data['total_evaluations']:.1%}"
        )
        print(f"   Average execution time: {summary_data['average_execution_time']:.3f}s")
        print(f"   Models evaluated: {len(summary_data['models_evaluated'])}")
        print(f"   Datasets evaluated: {len(summary_data['datasets_evaluated'])}")

        # Validate summary completeness
        assert summary_data["total_evaluations"] >= len(evaluation_scenarios), (
            "Summary should include all test evaluations"
        )
        assert summary_data["successful_evaluations"] >= len(evaluation_scenarios), (
            "All test evaluations should be successful"
        )
        assert summary_data["failed_evaluations"] == 0, "No test evaluations should fail"

        # Validate metric summaries
        metric_summaries = summary_data["metric_summaries"]
        assert "accuracy" in metric_summaries, "Summary should include accuracy statistics"
        assert "precision" in metric_summaries, "Summary should include precision statistics"

        accuracy_stats = metric_summaries["accuracy"]
        assert accuracy_stats["count"] >= len(evaluation_scenarios), (
            "Accuracy count should include all evaluations"
        )
        assert 0.70 <= accuracy_stats["mean"] <= 1.0, (
            f"Mean accuracy should be reasonable: {accuracy_stats['mean']:.3f}"
        )
        assert 0.0 <= accuracy_stats["min"] <= accuracy_stats["max"] <= 1.0, (
            "Accuracy range should be valid"
        )

        # Validate model and dataset coverage
        assert len(summary_data["models_evaluated"]) >= 3, "Should have evaluated multiple models"
        assert len(summary_data["datasets_evaluated"]) >= 4, (
            "Should have evaluated multiple datasets"
        )

        print(f"   Models: {', '.join(summary_data['models_evaluated'])}")
        print(f"   Datasets: {', '.join(summary_data['datasets_evaluated'])}")

        # Validate time range
        time_range = summary_data["time_range"]
        assert "start" in time_range and "end" in time_range, "Summary should include time range"

        start_time = datetime.fromisoformat(time_range["start"])
        end_time = datetime.fromisoformat(time_range["end"])
        assert start_time <= end_time, "Time range should be valid"
        assert (end_time - start_time).total_seconds() < 3600, (
            "Time range should be within test duration"
        )

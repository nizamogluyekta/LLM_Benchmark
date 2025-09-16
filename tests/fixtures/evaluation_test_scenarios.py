"""
Test fixtures and scenarios for comprehensive evaluation service testing.

This module provides realistic test scenarios, mock data, and helper functions
for testing the complete evaluation service functionality across various
cybersecurity domains and use cases.
"""

import asyncio
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from benchmark.interfaces.evaluation_interfaces import (
    EvaluationRequest,
    EvaluationResult,
    MetricEvaluator,
    MetricType,
)
from benchmark.services.evaluation_service import EvaluationService


@dataclass
class TestScenario:
    """Base class for evaluation test scenarios."""

    name: str
    description: str
    expected_duration: float  # Expected test duration in seconds
    models: list[str]
    datasets: list[str]
    metrics: list[MetricType]
    expected_results: dict[str, Any]


@dataclass
class CybersecurityTestData:
    """Realistic cybersecurity test data for various domains."""

    # Network intrusion detection data
    network_predictions: list[dict[str, Any]]
    network_ground_truth: list[dict[str, Any]]

    # Malware classification data
    malware_predictions: list[dict[str, Any]]
    malware_ground_truth: list[dict[str, Any]]

    # Phishing email detection data
    phishing_predictions: list[dict[str, Any]]
    phishing_ground_truth: list[dict[str, Any]]

    # Vulnerability assessment data
    vulnerability_predictions: list[dict[str, Any]]
    vulnerability_ground_truth: list[dict[str, Any]]


class MockAccuracyEvaluator(MetricEvaluator):
    """Mock accuracy evaluator for testing."""

    def __init__(self, base_accuracy: float = 0.85, variance: float = 0.05):
        self.base_accuracy = base_accuracy
        self.variance = variance

    async def evaluate(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Mock evaluation with realistic accuracy computation."""
        await asyncio.sleep(0.01)  # Simulate processing time

        if not predictions or not ground_truth:
            return {"accuracy": 0.0}

        # Simulate accuracy calculation with some variance
        accuracy = self.base_accuracy + random.uniform(-self.variance, self.variance)
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]

        return {
            "accuracy": accuracy,
            "samples_evaluated": len(predictions),
        }

    def get_metric_names(self) -> list[str]:
        return ["accuracy", "samples_evaluated"]

    def get_required_prediction_fields(self) -> list[str]:
        return ["predicted_class", "confidence"]

    def get_required_ground_truth_fields(self) -> list[str]:
        return ["true_class"]

    def get_metric_type(self) -> MetricType:
        return MetricType.ACCURACY


class MockPrecisionRecallEvaluator(MetricEvaluator):
    """Mock precision/recall evaluator for testing."""

    def __init__(self, base_precision: float = 0.80, base_recall: float = 0.78):
        self.base_precision = base_precision
        self.base_recall = base_recall

    async def evaluate(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Mock precision/recall computation."""
        await asyncio.sleep(0.02)  # Simulate processing time

        if not predictions or not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        # Add some realistic variance
        precision = self.base_precision + random.uniform(-0.03, 0.03)
        recall = self.base_recall + random.uniform(-0.04, 0.04)

        precision = max(0.0, min(1.0, precision))
        recall = max(0.0, min(1.0, recall))

        # Calculate F1 score
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def get_metric_names(self) -> list[str]:
        return ["precision", "recall", "f1_score"]

    def get_required_prediction_fields(self) -> list[str]:
        return ["predicted_class", "confidence"]

    def get_required_ground_truth_fields(self) -> list[str]:
        return ["true_class"]

    def get_metric_type(self) -> MetricType:
        return MetricType.PRECISION


class MockPerformanceEvaluator(MetricEvaluator):
    """Mock performance evaluator for testing."""

    def __init__(self, base_latency: float = 0.15):
        self.base_latency = base_latency

    async def evaluate(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Mock performance metrics computation."""
        await asyncio.sleep(0.005)  # Simulate processing time

        sample_count = len(predictions) if predictions else 0

        # Simulate realistic performance metrics
        latency = self.base_latency + random.uniform(-0.02, 0.05)
        throughput = sample_count / max(latency, 0.001) if sample_count > 0 else 0.0
        memory_usage = random.uniform(50, 200)  # MB
        cpu_usage = random.uniform(20, 80)  # Percentage

        return {
            "inference_latency_ms": latency * 1000,
            "throughput_samples_per_sec": throughput,
            "memory_usage_mb": memory_usage,
            "cpu_usage_percent": cpu_usage,
        }

    def get_metric_names(self) -> list[str]:
        return [
            "inference_latency_ms",
            "throughput_samples_per_sec",
            "memory_usage_mb",
            "cpu_usage_percent",
        ]

    def get_required_prediction_fields(self) -> list[str]:
        return ["predicted_class", "inference_time"]

    def get_required_ground_truth_fields(self) -> list[str]:
        return []

    def get_metric_type(self) -> MetricType:
        return MetricType.PERFORMANCE


def generate_cybersecurity_test_data(sample_count: int = 1000) -> CybersecurityTestData:
    """Generate realistic cybersecurity test data."""

    # Network intrusion detection data
    network_predictions = []
    network_ground_truth = []

    intrusion_types = ["normal", "dos", "probe", "r2l", "u2r"]

    for i in range(sample_count):
        true_class = random.choice(intrusion_types)

        # Simulate realistic model predictions with some errors
        if random.random() < 0.85:  # 85% accuracy baseline
            predicted_class = true_class
        else:
            predicted_class = random.choice([t for t in intrusion_types if t != true_class])

        confidence = (
            random.uniform(0.6, 0.95) if predicted_class == true_class else random.uniform(0.4, 0.8)
        )

        network_predictions.append(
            {
                "sample_id": f"net_{i}",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "inference_time": random.uniform(0.01, 0.05),
                "features": {
                    "packet_size": random.randint(64, 1500),
                    "connection_duration": random.uniform(0.1, 30.0),
                    "protocol": random.choice(["tcp", "udp", "icmp"]),
                },
            }
        )

        network_ground_truth.append(
            {
                "sample_id": f"net_{i}",
                "true_class": true_class,
                "severity": random.choice(["low", "medium", "high", "critical"]),
            }
        )

    # Malware classification data
    malware_predictions = []
    malware_ground_truth = []

    malware_families = ["benign", "trojan", "virus", "worm", "adware", "spyware", "ransomware"]

    for i in range(sample_count):
        true_family = random.choice(malware_families)

        # Simulate realistic malware detection with higher accuracy for benign samples
        if true_family == "benign":
            predicted_family = (
                true_family if random.random() < 0.95 else random.choice(malware_families[1:])
            )
        else:
            predicted_family = (
                true_family if random.random() < 0.80 else random.choice(malware_families)
            )

        confidence = (
            random.uniform(0.7, 0.99)
            if predicted_family == true_family
            else random.uniform(0.3, 0.7)
        )

        malware_predictions.append(
            {
                "sample_id": f"mal_{i}",
                "predicted_class": predicted_family,
                "confidence": confidence,
                "inference_time": random.uniform(0.05, 0.2),
                "file_hash": f"hash_{uuid.uuid4().hex[:16]}",
            }
        )

        malware_ground_truth.append(
            {
                "sample_id": f"mal_{i}",
                "true_class": true_family,
                "file_size": random.randint(1024, 10485760),  # 1KB to 10MB
            }
        )

    # Phishing email detection data
    phishing_predictions = []
    phishing_ground_truth = []

    email_classes = ["legitimate", "phishing", "spam"]

    for i in range(sample_count):
        true_class = random.choice(email_classes)

        # Simulate email classification with different accuracy for each class
        accuracy_rates = {"legitimate": 0.92, "phishing": 0.88, "spam": 0.90}
        if random.random() < accuracy_rates[true_class]:
            predicted_class = true_class
        else:
            predicted_class = random.choice([c for c in email_classes if c != true_class])

        confidence = (
            random.uniform(0.8, 0.98)
            if predicted_class == true_class
            else random.uniform(0.4, 0.75)
        )

        phishing_predictions.append(
            {
                "sample_id": f"email_{i}",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "inference_time": random.uniform(0.02, 0.1),
                "sender_domain": f"domain{random.randint(1, 100)}.com",
            }
        )

        phishing_ground_truth.append(
            {
                "sample_id": f"email_{i}",
                "true_class": true_class,
                "subject_length": random.randint(10, 100),
            }
        )

    # Vulnerability assessment data
    vuln_predictions = []
    vuln_ground_truth = []

    severity_levels = ["none", "low", "medium", "high", "critical"]

    for i in range(sample_count):
        true_severity = random.choice(severity_levels)

        # Vulnerability assessment tends to be more conservative (higher false positives)
        if true_severity == "none":
            predicted_severity = (
                true_severity if random.random() < 0.80 else random.choice(severity_levels[1:3])
            )
        else:
            predicted_severity = (
                true_severity if random.random() < 0.75 else random.choice(severity_levels)
            )

        confidence = random.uniform(0.6, 0.9)

        vuln_predictions.append(
            {
                "sample_id": f"vuln_{i}",
                "predicted_class": predicted_severity,
                "confidence": confidence,
                "inference_time": random.uniform(0.1, 0.5),
                "cve_score": random.uniform(0.0, 10.0) if predicted_severity != "none" else 0.0,
            }
        )

        vuln_ground_truth.append(
            {
                "sample_id": f"vuln_{i}",
                "true_class": true_severity,
                "asset_type": random.choice(["web_app", "database", "network_device", "server"]),
            }
        )

    return CybersecurityTestData(
        network_predictions=network_predictions,
        network_ground_truth=network_ground_truth,
        malware_predictions=malware_predictions,
        malware_ground_truth=malware_ground_truth,
        phishing_predictions=phishing_predictions,
        phishing_ground_truth=phishing_ground_truth,
        vulnerability_predictions=vuln_predictions,
        vulnerability_ground_truth=vuln_ground_truth,
    )


def create_test_scenarios() -> list[TestScenario]:
    """Create comprehensive test scenarios for evaluation service testing."""

    scenarios = [
        # Single model, single dataset scenario
        TestScenario(
            name="single_model_basic",
            description="Basic single model evaluation on network intrusion detection",
            expected_duration=2.0,
            models=["security_bert"],
            datasets=["network_intrusion_detection"],
            metrics=[MetricType.ACCURACY, MetricType.PRECISION],
            expected_results={
                "accuracy": {"min": 0.80, "max": 0.95},
                "precision": {"min": 0.75, "max": 0.90},
                "sample_count": 1000,
            },
        ),
        # Multi-model comparison scenario
        TestScenario(
            name="multi_model_comparison",
            description="Compare multiple cybersecurity models on the same dataset",
            expected_duration=8.0,
            models=["security_bert", "cyber_lstm", "threat_cnn", "baseline_svm"],
            datasets=["malware_classification"],
            metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.PERFORMANCE],
            expected_results={
                "model_count": 4,
                "accuracy_variance": {"min": 0.02, "max": 0.15},
                "performance_measured": True,
            },
        ),
        # Cross-domain evaluation scenario
        TestScenario(
            name="cross_domain_evaluation",
            description="Evaluate models across different cybersecurity domains",
            expected_duration=15.0,
            models=["multi_domain_transformer", "specialized_detector"],
            datasets=[
                "network_intrusion_detection",
                "malware_classification",
                "phishing_detection",
                "vulnerability_assessment",
            ],
            metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.F1_SCORE],
            expected_results={
                "domain_count": 4,
                "model_count": 2,
                "total_evaluations": 8,
            },
        ),
        # Performance stress test scenario
        TestScenario(
            name="performance_stress_test",
            description="High-throughput evaluation with performance monitoring",
            expected_duration=12.0,
            models=["fast_detector_1", "fast_detector_2", "fast_detector_3"],
            datasets=["large_network_dataset"],
            metrics=[MetricType.ACCURACY, MetricType.PERFORMANCE],
            expected_results={
                "high_throughput": True,
                "latency_measured": True,
                "memory_usage_tracked": True,
            },
        ),
        # Error recovery scenario
        TestScenario(
            name="error_recovery_test",
            description="Test evaluation service resilience under various error conditions",
            expected_duration=6.0,
            models=["failing_model", "timeout_model", "valid_model"],
            datasets=["error_test_dataset"],
            metrics=[MetricType.ACCURACY],
            expected_results={
                "some_failures_expected": True,
                "service_continues_operation": True,
                "error_handling_validated": True,
            },
        ),
        # Concurrent evaluation scenario
        TestScenario(
            name="concurrent_evaluation_test",
            description="Multiple concurrent evaluations across different models and datasets",
            expected_duration=10.0,
            models=[
                "concurrent_model_1",
                "concurrent_model_2",
                "concurrent_model_3",
                "concurrent_model_4",
            ],
            datasets=["dataset_a", "dataset_b", "dataset_c"],
            metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.PERFORMANCE],
            expected_results={
                "concurrent_evaluations": 12,  # 4 models × 3 datasets
                "no_resource_conflicts": True,
                "consistent_results": True,
            },
        ),
    ]

    return scenarios


async def create_mock_evaluation_service(
    scenario: TestScenario,
    introduce_failures: bool = False,
) -> EvaluationService:
    """Create a fully configured mock evaluation service for testing."""

    service = EvaluationService()

    # Initialize service with test configuration
    config = {
        "max_concurrent_evaluations": 10,
        "evaluation_timeout_seconds": 30.0,
        "max_history_size": 500,
    }
    await service.initialize(config)

    # Register mock evaluators based on scenario metrics
    for metric_type in scenario.metrics:
        if metric_type == MetricType.ACCURACY:
            evaluator = MockAccuracyEvaluator()
        elif metric_type == MetricType.PRECISION:
            evaluator = MockPrecisionRecallEvaluator()
        elif metric_type == MetricType.PERFORMANCE:
            evaluator = MockPerformanceEvaluator()
        else:
            # Create generic mock evaluator for other metrics
            evaluator = MockAccuracyEvaluator()  # Use as fallback

        await service.register_evaluator(metric_type, evaluator)

    # Configure failure injection for error testing scenarios
    if introduce_failures and "error" in scenario.name:
        # Replace one evaluator with a failing one
        failing_evaluator = MagicMock(spec=MetricEvaluator)
        failing_evaluator.evaluate = AsyncMock(
            side_effect=Exception("Simulated evaluation failure")
        )
        failing_evaluator.get_metric_names.return_value = ["failed_metric"]
        failing_evaluator.get_metric_type.return_value = MetricType.ACCURACY
        failing_evaluator.validate_data_compatibility.return_value = True

        await service.register_evaluator(MetricType.ACCURACY, failing_evaluator)

    return service


def get_test_data_for_dataset(
    dataset_name: str, test_data: CybersecurityTestData
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Get appropriate test data for a given dataset name."""

    dataset_mapping = {
        "network_intrusion_detection": (
            test_data.network_predictions,
            test_data.network_ground_truth,
        ),
        "large_network_dataset": (
            test_data.network_predictions * 5,
            test_data.network_ground_truth * 5,
        ),  # Larger dataset
        "malware_classification": (test_data.malware_predictions, test_data.malware_ground_truth),
        "phishing_detection": (test_data.phishing_predictions, test_data.phishing_ground_truth),
        "vulnerability_assessment": (
            test_data.vulnerability_predictions,
            test_data.vulnerability_ground_truth,
        ),
        "error_test_dataset": (
            test_data.network_predictions[:100],
            test_data.network_ground_truth[:100],
        ),  # Smaller dataset for error testing
        "dataset_a": (test_data.network_predictions[:300], test_data.network_ground_truth[:300]),
        "dataset_b": (test_data.malware_predictions[:300], test_data.malware_ground_truth[:300]),
        "dataset_c": (test_data.phishing_predictions[:300], test_data.phishing_ground_truth[:300]),
    }

    return dataset_mapping.get(
        dataset_name, (test_data.network_predictions, test_data.network_ground_truth)
    )


async def validate_scenario_results(
    scenario: TestScenario,
    actual_results: list[EvaluationResult],
) -> dict[str, bool]:
    """Validate that evaluation results match scenario expectations."""

    validation_results = {}
    expected = scenario.expected_results

    # Basic count validations
    if "model_count" in expected:
        validation_results["model_count_correct"] = (
            len({r.model_id for r in actual_results}) == expected["model_count"]
        )

    if "sample_count" in expected:
        validation_results["sample_count_correct"] = all(
            "samples_evaluated" in r.metrics
            and r.metrics["samples_evaluated"] >= expected["sample_count"] * 0.9
            for r in actual_results
        )

    if "total_evaluations" in expected:
        validation_results["total_evaluations_correct"] = (
            len(actual_results) == expected["total_evaluations"]
        )

    # Accuracy range validation
    if "accuracy" in expected:
        accuracy_range = expected["accuracy"]
        accuracy_values = [
            r.metrics.get("accuracy", 0.0) for r in actual_results if "accuracy" in r.metrics
        ]
        if accuracy_values:
            validation_results["accuracy_in_range"] = all(
                accuracy_range["min"] <= acc <= accuracy_range["max"] for acc in accuracy_values
            )

    # Performance validation
    if expected.get("performance_measured"):
        validation_results["performance_measured"] = any(
            "inference_latency_ms" in r.metrics for r in actual_results
        )

    if expected.get("latency_measured"):
        validation_results["latency_measured"] = any(
            "inference_latency_ms" in r.metrics and r.metrics["inference_latency_ms"] > 0
            for r in actual_results
        )

    # Error handling validation
    if expected.get("some_failures_expected"):
        validation_results["some_failures_occurred"] = any(not r.success for r in actual_results)
        validation_results["some_successes_occurred"] = any(r.success for r in actual_results)

    # Success rate validation (most results should succeed unless explicitly testing failures)
    if not expected.get("some_failures_expected"):
        success_rate = sum(1 for r in actual_results if r.success) / max(len(actual_results), 1)
        validation_results["high_success_rate"] = success_rate >= 0.8

    return validation_results


def create_evaluation_request(
    model_id: str,
    dataset_name: str,
    test_data: CybersecurityTestData,
    metrics: list[MetricType],
) -> EvaluationRequest:
    """Create an evaluation request for testing."""

    predictions, ground_truth = get_test_data_for_dataset(dataset_name, test_data)

    return EvaluationRequest(
        experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
        model_id=model_id,
        dataset_id=dataset_name,
        predictions=predictions,
        ground_truth=ground_truth,
        metrics=metrics,
        metadata={
            "test_scenario": True,
            "created_at": datetime.now().isoformat(),
            "dataset_size": len(predictions),
        },
    )


# Performance benchmarking utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking during tests."""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.memory_samples: list[float] = []
        self.cpu_samples: list[float] = []

    def start(self) -> None:
        """Start performance monitoring."""
        import time

        self.start_time = time.time()

    def stop(self) -> None:
        """Stop performance monitoring."""
        import time

        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get benchmark duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def add_memory_sample(self, memory_mb: float) -> None:
        """Add a memory usage sample."""
        self.memory_samples.append(memory_mb)

    def add_cpu_sample(self, cpu_percent: float) -> None:
        """Add a CPU usage sample."""
        self.cpu_samples.append(cpu_percent)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "duration_seconds": self.duration,
            "memory_peak_mb": max(self.memory_samples) if self.memory_samples else 0.0,
            "memory_avg_mb": sum(self.memory_samples) / len(self.memory_samples)
            if self.memory_samples
            else 0.0,
            "cpu_peak_percent": max(self.cpu_samples) if self.cpu_samples else 0.0,
            "cpu_avg_percent": sum(self.cpu_samples) / len(self.cpu_samples)
            if self.cpu_samples
            else 0.0,
        }
        return summary

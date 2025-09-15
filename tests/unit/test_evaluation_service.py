"""
Unit tests for the EvaluationService.

Tests service initialization, evaluator registration, evaluation request validation,
parallel metric evaluation, error handling, and health checks.
"""

import asyncio

import pytest
import pytest_asyncio

from benchmark.core.base import ServiceStatus
from benchmark.interfaces.evaluation_interfaces import (
    EvaluationProgressCallback,
    EvaluationRequest,
    MetricEvaluator,
    MetricType,
)
from benchmark.services.evaluation_service import EvaluationService


class MockEvaluator(MetricEvaluator):
    """Mock evaluator for testing."""

    def __init__(self, metric_type: MetricType, metric_names: list[str] | None = None):
        self.metric_type = metric_type
        self.metric_names = metric_names or [f"{metric_type.value}_score"]

    async def evaluate(self, predictions, ground_truth):
        """Mock evaluation that returns dummy metrics."""
        await asyncio.sleep(0.01)  # Simulate processing time
        return dict.fromkeys(self.metric_names, 0.85)

    def get_metric_names(self):
        return self.metric_names

    def get_required_prediction_fields(self):
        return ["prediction"]

    def get_required_ground_truth_fields(self):
        return ["label"]

    def get_metric_type(self):
        return self.metric_type


class MockFailingEvaluator(MetricEvaluator):
    """Mock evaluator that always fails."""

    def __init__(self, metric_type: MetricType):
        self.metric_type = metric_type

    async def evaluate(self, predictions, ground_truth):
        raise RuntimeError("Evaluation failed")

    def get_metric_names(self):
        return ["failing_metric"]

    def get_required_prediction_fields(self):
        return ["prediction"]

    def get_required_ground_truth_fields(self):
        return ["label"]

    def get_metric_type(self):
        return self.metric_type


class MockSlowEvaluator(MetricEvaluator):
    """Mock evaluator that takes longer to complete."""

    def __init__(self, metric_type: MetricType):
        self.metric_type = metric_type

    async def evaluate(self, predictions, ground_truth):
        """Mock evaluation that takes longer to simulate real work."""
        await asyncio.sleep(0.1)  # Longer sleep to ensure overlap
        return {"accuracy_score": 0.85}

    def get_metric_names(self):
        return ["accuracy_score"]

    def get_required_prediction_fields(self):
        return ["prediction"]

    def get_required_ground_truth_fields(self):
        return ["label"]

    def get_metric_type(self):
        return self.metric_type


class MockProgressCallback(EvaluationProgressCallback):
    """Mock progress callback for testing."""

    def __init__(self):
        self.events = []

    async def on_evaluation_started(self, request):
        self.events.append(("started", request.experiment_id))

    async def on_metric_completed(self, metric_type, result):
        self.events.append(("metric_completed", metric_type.value, result))

    async def on_evaluation_completed(self, result):
        self.events.append(("completed", result.experiment_id))

    async def on_evaluation_error(self, error):
        self.events.append(("error", str(error)))


class TestEvaluationService:
    """Test cases for EvaluationService."""

    @pytest_asyncio.fixture
    async def evaluation_service(self):
        """Create a fresh evaluation service for each test."""
        service = EvaluationService()
        await service.initialize()
        return service

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction data for testing."""
        return [
            {"prediction": "ATTACK", "confidence": 0.9},
            {"prediction": "BENIGN", "confidence": 0.8},
            {"prediction": "ATTACK", "confidence": 0.7},
        ]

    @pytest.fixture
    def sample_ground_truth(self):
        """Sample ground truth data for testing."""
        return [
            {"label": "ATTACK"},
            {"label": "BENIGN"},
            {"label": "ATTACK"},
        ]

    @pytest.fixture
    def sample_evaluation_request(self, sample_predictions, sample_ground_truth):
        """Sample evaluation request for testing."""
        return EvaluationRequest(
            experiment_id="test_experiment_001",
            model_id="test_model",
            dataset_id="test_dataset",
            predictions=sample_predictions,
            ground_truth=sample_ground_truth,
            metrics=[MetricType.ACCURACY],
            metadata={"test": "data"},
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test that the service initializes correctly."""
        service = EvaluationService()

        # Test initialization
        result = await service.initialize()

        assert result.success is True
        assert "evaluators_registered" in result.data
        assert "supported_metrics" in result.data
        assert "max_concurrent_evaluations" in result.data

    @pytest.mark.asyncio
    async def test_service_initialization_with_config(self):
        """Test service initialization with custom configuration."""
        service = EvaluationService()
        config = {
            "max_concurrent_evaluations": 10,
            "evaluation_timeout_seconds": 600.0,
            "max_history_size": 2000,
        }

        result = await service.initialize(config)

        assert result.success is True
        assert service.max_concurrent_evaluations == 10
        assert service.evaluation_timeout_seconds == 600.0
        assert service.max_history_size == 2000

    @pytest.mark.asyncio
    async def test_evaluator_registration(self, evaluation_service):
        """Test registering a new evaluator."""
        evaluator = MockEvaluator(MetricType.ACCURACY)

        result = await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        assert result.success is True
        assert MetricType.ACCURACY in evaluation_service.evaluators
        assert evaluation_service.evaluators[MetricType.ACCURACY] == evaluator

    @pytest.mark.asyncio
    async def test_evaluator_registration_type_mismatch(self, evaluation_service):
        """Test registering evaluator with mismatched type."""
        evaluator = MockEvaluator(MetricType.ACCURACY)

        # Try to register with different type
        result = await evaluation_service.register_evaluator(MetricType.PRECISION, evaluator)

        assert result.success is False
        assert "does not match registration type" in result.error

    @pytest.mark.asyncio
    async def test_evaluator_registration_invalid_type(self, evaluation_service):
        """Test registering invalid evaluator type."""
        invalid_evaluator = "not_an_evaluator"

        result = await evaluation_service.register_evaluator(MetricType.ACCURACY, invalid_evaluator)

        assert result.success is False
        assert "MetricEvaluator interface" in result.error

    @pytest.mark.asyncio
    async def test_evaluator_unregistration(self, evaluation_service):
        """Test unregistering an evaluator."""
        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        result = await evaluation_service.unregister_evaluator(MetricType.ACCURACY)

        assert result.success is True
        assert MetricType.ACCURACY not in evaluation_service.evaluators

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_evaluator(self, evaluation_service):
        """Test unregistering a non-existent evaluator."""
        result = await evaluation_service.unregister_evaluator(MetricType.ACCURACY)

        assert result.success is False
        assert "No evaluator registered" in result.error

    @pytest.mark.asyncio
    async def test_evaluation_data_validation_success(
        self, evaluation_service, sample_evaluation_request
    ):
        """Test successful evaluation data validation."""
        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        result = await evaluation_service._validate_evaluation_data(sample_evaluation_request)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_evaluation_data_validation_length_mismatch(self, evaluation_service):
        """Test validation with mismatched prediction/ground truth lengths."""
        request = EvaluationRequest(
            experiment_id="test",
            model_id="test",
            dataset_id="test",
            predictions=[{"prediction": "ATTACK"}],
            ground_truth=[{"label": "ATTACK"}, {"label": "BENIGN"}],
            metrics=[MetricType.ACCURACY],
            metadata={},
        )

        result = await evaluation_service._validate_evaluation_data(request)

        assert result.success is False
        assert "same length" in result.error

    @pytest.mark.asyncio
    async def test_evaluation_data_validation_empty_dataset(self, evaluation_service):
        """Test validation with empty dataset."""
        # Create request manually to bypass validation
        request = EvaluationRequest.__new__(EvaluationRequest)
        request.experiment_id = "test"
        request.model_id = "test"
        request.dataset_id = "test"
        request.predictions = []
        request.ground_truth = []
        request.metrics = [MetricType.ACCURACY]
        request.metadata = {}

        result = await evaluation_service._validate_evaluation_data(request)

        assert result.success is False
        assert "empty dataset" in result.error

    @pytest.mark.asyncio
    async def test_successful_evaluation(self, evaluation_service, sample_evaluation_request):
        """Test successful prediction evaluation."""
        evaluator = MockEvaluator(MetricType.ACCURACY, ["accuracy", "precision"])
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        result = await evaluation_service.evaluate_predictions(sample_evaluation_request)

        assert result.success is True
        assert result.experiment_id == "test_experiment_001"
        assert result.model_id == "test_model"
        assert result.dataset_id == "test_dataset"
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert result.execution_time_seconds > 0
        assert len(evaluation_service.evaluation_history) == 1

    @pytest.mark.asyncio
    async def test_evaluation_with_missing_evaluator(
        self, evaluation_service, sample_evaluation_request
    ):
        """Test evaluation when required evaluator is missing."""
        with pytest.raises(Exception) as exc_info:
            await evaluation_service.evaluate_predictions(sample_evaluation_request)

        assert "not available for metric" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluation_with_failing_evaluator(
        self, evaluation_service, sample_evaluation_request
    ):
        """Test evaluation with a failing evaluator."""
        failing_evaluator = MockFailingEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, failing_evaluator)

        result = await evaluation_service.evaluate_predictions(sample_evaluation_request)

        assert result.success is False
        assert result.error_message is not None
        assert "Evaluation failed" in result.detailed_results[MetricType.ACCURACY.value]["error"]

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self, evaluation_service, sample_evaluation_request):
        """Test parallel evaluation of multiple metrics."""
        accuracy_evaluator = MockEvaluator(MetricType.ACCURACY, ["accuracy"])
        precision_evaluator = MockEvaluator(MetricType.PRECISION, ["precision"])

        await evaluation_service.register_evaluator(MetricType.ACCURACY, accuracy_evaluator)
        await evaluation_service.register_evaluator(MetricType.PRECISION, precision_evaluator)

        # Update request to include both metrics
        sample_evaluation_request.metrics = [MetricType.ACCURACY, MetricType.PRECISION]

        result = await evaluation_service.evaluate_predictions(sample_evaluation_request)

        assert result.success is True
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert len(result.detailed_results) == 2

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_limit(self, evaluation_service):
        """Test that concurrent evaluation limit is enforced."""
        evaluation_service.max_concurrent_evaluations = 1

        # Use slow evaluator to ensure first evaluation is still running
        evaluator = MockSlowEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        # Create two evaluation requests
        request1 = EvaluationRequest(
            experiment_id="test_1",
            model_id="test",
            dataset_id="test",
            predictions=[{"prediction": "ATTACK"}],
            ground_truth=[{"label": "ATTACK"}],
            metrics=[MetricType.ACCURACY],
            metadata={},
        )

        request2 = EvaluationRequest(
            experiment_id="test_2",
            model_id="test",
            dataset_id="test",
            predictions=[{"prediction": "BENIGN"}],
            ground_truth=[{"label": "BENIGN"}],
            metrics=[MetricType.ACCURACY],
            metadata={},
        )

        # Start first evaluation (should succeed)
        task1 = asyncio.create_task(evaluation_service.evaluate_predictions(request1))

        # Give the first evaluation time to start
        await asyncio.sleep(0.01)

        # Start second evaluation immediately (should fail due to limit)
        with pytest.raises(Exception) as exc_info:
            await evaluation_service.evaluate_predictions(request2)

        assert "Maximum concurrent evaluations" in str(exc_info.value)

        # Wait for first evaluation to complete
        result1 = await task1
        assert result1.success is True

    @pytest.mark.asyncio
    async def test_progress_callbacks(self, evaluation_service, sample_evaluation_request):
        """Test progress callback functionality."""
        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        callback = MockProgressCallback()
        await evaluation_service.add_progress_callback(callback)

        result = await evaluation_service.evaluate_predictions(sample_evaluation_request)

        assert result.success is True

        # Check that callback events were triggered
        event_types = [event[0] for event in callback.events]
        assert "started" in event_types
        assert "metric_completed" in event_types
        assert "completed" in event_types

    @pytest.mark.asyncio
    async def test_progress_callback_removal(self, evaluation_service):
        """Test removing progress callbacks."""
        callback = MockProgressCallback()

        # Add callback
        result = await evaluation_service.add_progress_callback(callback)
        assert result.success is True
        assert callback in evaluation_service.progress_callbacks

        # Remove callback
        result = await evaluation_service.remove_progress_callback(callback)
        assert result.success is True
        assert callback not in evaluation_service.progress_callbacks

    @pytest.mark.asyncio
    async def test_remove_nonexistent_callback(self, evaluation_service):
        """Test removing a non-existent callback."""
        callback = MockProgressCallback()

        result = await evaluation_service.remove_progress_callback(callback)
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_get_available_metrics(self, evaluation_service):
        """Test getting available metrics information."""
        evaluator1 = MockEvaluator(MetricType.ACCURACY, ["accuracy"])
        evaluator2 = MockEvaluator(MetricType.PRECISION, ["precision", "recall"])

        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator1)
        await evaluation_service.register_evaluator(MetricType.PRECISION, evaluator2)

        result = await evaluation_service.get_available_metrics()

        assert result.success is True
        assert "metrics" in result.data
        assert MetricType.ACCURACY.value in result.data["metrics"]
        assert MetricType.PRECISION.value in result.data["metrics"]
        assert result.data["total_evaluators"] == 2

    @pytest.mark.asyncio
    async def test_get_evaluation_history(self, evaluation_service, sample_evaluation_request):
        """Test getting evaluation history."""
        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        # Perform evaluation to create history
        await evaluation_service.evaluate_predictions(sample_evaluation_request)

        result = await evaluation_service.get_evaluation_history()

        assert result.success is True
        assert "results" in result.data
        assert len(result.data["results"]) == 1
        assert result.data["total_results"] == 1

    @pytest.mark.asyncio
    async def test_get_evaluation_history_filtered(
        self, evaluation_service, sample_evaluation_request
    ):
        """Test getting filtered evaluation history."""
        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        # Perform evaluation
        await evaluation_service.evaluate_predictions(sample_evaluation_request)

        # Get history filtered by experiment ID
        result = await evaluation_service.get_evaluation_history(
            experiment_id="test_experiment_001"
        )

        assert result.success is True
        assert len(result.data["results"]) == 1

        # Get history filtered by non-existent experiment ID
        result = await evaluation_service.get_evaluation_history(experiment_id="nonexistent")

        assert result.success is True
        assert len(result.data["results"]) == 0

    @pytest.mark.asyncio
    async def test_get_evaluation_summary(self, evaluation_service, sample_evaluation_request):
        """Test getting evaluation summary statistics."""
        evaluator = MockEvaluator(MetricType.ACCURACY, ["accuracy"])
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        # Perform evaluation to create history
        await evaluation_service.evaluate_predictions(sample_evaluation_request)

        result = await evaluation_service.get_evaluation_summary()

        assert result.success is True
        summary = result.data
        assert summary["total_evaluations"] == 1
        assert summary["successful_evaluations"] == 1
        assert summary["failed_evaluations"] == 0
        # Calculate success rate manually since it's a property
        success_rate = (summary["successful_evaluations"] / summary["total_evaluations"]) * 100
        assert success_rate == 100.0

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, evaluation_service):
        """Test health check when service is healthy."""
        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        health = await evaluation_service.health_check()

        assert health.status == ServiceStatus.HEALTHY.value
        assert health.checks["evaluators_count"] == 1
        assert health.checks["active_evaluations"] == 0

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, evaluation_service):
        """Test health check when no evaluators are registered."""
        health = await evaluation_service.health_check()

        assert health.status == ServiceStatus.UNHEALTHY.value
        assert "No evaluators registered" in health.checks["issue"]

    @pytest.mark.asyncio
    async def test_service_shutdown(self, evaluation_service):
        """Test graceful service shutdown."""
        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        result = await evaluation_service.shutdown()

        assert result.success is True
        assert len(evaluation_service.evaluators) == 0
        assert len(evaluation_service.evaluation_history) == 0
        assert len(evaluation_service.active_evaluations) == 0

    @pytest.mark.asyncio
    async def test_evaluation_history_size_management(self, evaluation_service):
        """Test that evaluation history size is managed correctly."""
        evaluation_service.max_history_size = 2

        evaluator = MockEvaluator(MetricType.ACCURACY)
        await evaluation_service.register_evaluator(MetricType.ACCURACY, evaluator)

        # Create multiple evaluation requests
        for i in range(3):
            request = EvaluationRequest(
                experiment_id=f"test_{i}",
                model_id="test",
                dataset_id="test",
                predictions=[{"prediction": "ATTACK"}],
                ground_truth=[{"label": "ATTACK"}],
                metrics=[MetricType.ACCURACY],
                metadata={},
            )
            await evaluation_service.evaluate_predictions(request)

        # Should only keep the last 2 results
        assert len(evaluation_service.evaluation_history) == 2

        # Check that we kept the most recent ones
        experiment_ids = [result.experiment_id for result in evaluation_service.evaluation_history]
        assert "test_1" in experiment_ids
        assert "test_2" in experiment_ids
        assert "test_0" not in experiment_ids

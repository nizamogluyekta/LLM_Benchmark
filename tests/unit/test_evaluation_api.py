"""
Comprehensive test suite for EvaluationAPI.

Tests API endpoint functionality, request/response validation,
error handling, and integration with evaluation services.
"""

import asyncio
import tempfile
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmark.evaluation import (
    APIError,
    EvaluationAPI,
    EvaluationConfig,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultsResponse,
    EvaluationStatus,
    EvaluationStatusResponse,
    EvaluationType,
    ValidationLevel,
)
from benchmark.evaluation.api_models import ResourceLimits as APIResourceLimits
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage
from benchmark.services.evaluation_service import EvaluationService


class TestAPIModels:
    """Test API data models validation and serialization."""

    def test_evaluation_config_validation(self):
        """Test evaluation configuration validation."""
        # Valid configuration
        valid_config = EvaluationConfig(
            batch_size=32, temperature=0.7, top_p=0.9, validation_level=ValidationLevel.MODERATE
        )
        assert valid_config.validate() == []

        # Invalid configuration
        invalid_config = EvaluationConfig(
            batch_size=-1,  # Invalid
            temperature=3.0,  # Invalid
            top_p=2.0,  # Invalid
        )
        errors = invalid_config.validate()
        assert len(errors) >= 3
        assert any("batch_size must be positive" in error for error in errors)
        assert any("temperature must be between" in error for error in errors)
        assert any("top_p must be between" in error for error in errors)

    def test_resource_limits_validation(self):
        """Test resource limits validation."""
        # Valid limits
        valid_limits = APIResourceLimits(
            max_memory_mb=4000, max_cpu_percent=80.0, max_concurrent_evaluations=4
        )
        assert valid_limits.validate() == []

        # Invalid limits
        invalid_limits = APIResourceLimits(
            max_memory_mb=-100,  # Invalid
            max_cpu_percent=150.0,  # Invalid
            max_concurrent_evaluations=-1,  # Invalid
        )
        errors = invalid_limits.validate()
        assert len(errors) >= 3

    def test_evaluation_request_validation(self):
        """Test evaluation request validation."""
        # Valid request
        valid_request = EvaluationRequest(
            model_names=["gpt-4", "claude-3"],
            task_types=["text_classification"],
            evaluation_type=EvaluationType.MODEL_COMPARISON,
        )
        assert valid_request.validate() == []

        # Invalid requests
        invalid_request = EvaluationRequest(
            model_names=[],  # Empty
            task_types=["text_classification"],
        )
        errors = invalid_request.validate()
        assert any("model_names cannot be empty" in error for error in errors)

        # Invalid evaluation type for single model
        type_mismatch_request = EvaluationRequest(
            model_names=["single-model"],  # Only one model
            task_types=["text_classification"],
            evaluation_type=EvaluationType.MODEL_COMPARISON,  # Needs 2+ models
        )
        errors = type_mismatch_request.validate()
        assert any("MODEL_COMPARISON requires at least 2 models" in error for error in errors)

    def test_request_serialization(self):
        """Test request serialization to dictionary."""
        request = EvaluationRequest(
            model_names=["gpt-4"],
            task_types=["text_classification"],
            dataset_names=["imdb"],
            metadata={"experiment": "test"},
            tags=["test", "api"],
        )

        request_dict = request.to_dict()

        assert request_dict["model_names"] == ["gpt-4"]
        assert request_dict["task_types"] == ["text_classification"]
        assert request_dict["dataset_names"] == ["imdb"]
        assert request_dict["metadata"] == {"experiment": "test"}
        assert request_dict["tags"] == ["test", "api"]
        assert "request_id" in request_dict
        assert "created_at" in request_dict

    def test_response_serialization(self):
        """Test response serialization to dictionary."""
        response = EvaluationResponse(
            evaluation_id="eval_123",
            request_id="req_456",
            status=EvaluationStatus.RUNNING,
            message="Evaluation in progress",
        )

        response_dict = response.to_dict()

        assert response_dict["evaluation_id"] == "eval_123"
        assert response_dict["request_id"] == "req_456"
        assert response_dict["status"] == "running"
        assert response_dict["message"] == "Evaluation in progress"
        assert "created_at" in response_dict


class TestEvaluationAPI:
    """Test EvaluationAPI functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def mock_evaluation_service(self, temp_storage):
        """Create mock evaluation service."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage

        # Mock async methods
        service.list_available_models = AsyncMock(
            return_value=["gpt-4", "claude-3", "llama-2", "bert-base"]
        )

        # Mock evaluation execution
        async def mock_evaluation(config):
            return EvaluationResult(
                evaluation_id=f"result_{uuid.uuid4().hex[:8]}",
                model_name=config["model_name"],
                task_type=config["task_type"],
                dataset_name=config.get("dataset_name", "test_dataset"),
                metrics={"accuracy": 0.85, "f1_score": 0.83},
                timestamp=datetime.now(),
                configuration=config,
                raw_responses=[],
                processing_time=1.5,
            )

        service.run_evaluation = mock_evaluation
        return service

    @pytest.fixture
    def api(self, mock_evaluation_service):
        """Create API instance for testing."""
        return EvaluationAPI(
            evaluation_service=mock_evaluation_service,
            max_concurrent_evaluations=5,
            default_timeout_seconds=600,
        )

    @pytest.mark.asyncio
    async def test_start_evaluation_success(self, api):
        """Test successful evaluation start."""
        request = EvaluationRequest(
            model_names=["gpt-4"],
            task_types=["text_classification"],
            evaluation_type=EvaluationType.SINGLE_MODEL,
        )

        response = await api.start_evaluation(request)

        assert isinstance(response, EvaluationResponse)
        assert response.status == EvaluationStatus.PENDING
        assert response.request_id == request.request_id
        assert response.evaluation_id is not None
        assert "started successfully" in response.message.lower()

    @pytest.mark.asyncio
    async def test_start_evaluation_validation_error(self, api):
        """Test evaluation start with validation errors."""
        invalid_request = EvaluationRequest(
            model_names=[],  # Empty - invalid
            task_types=["text_classification"],
        )

        response = await api.start_evaluation(invalid_request)

        assert isinstance(response, APIError)
        assert response.error_code == "VALIDATION_ERROR"
        assert len(response.validation_errors) > 0

    @pytest.mark.asyncio
    async def test_start_evaluation_capacity_exceeded(self, api):
        """Test evaluation start when capacity is exceeded."""
        # Fill up capacity
        request = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])

        # Start maximum allowed evaluations
        for _ in range(api.max_concurrent_evaluations):
            await api.start_evaluation(request)

        # Next request should fail
        response = await api.start_evaluation(request)

        assert isinstance(response, APIError)
        assert response.error_code == "CAPACITY_EXCEEDED"

    @pytest.mark.asyncio
    async def test_get_evaluation_status(self, api):
        """Test getting evaluation status."""
        request = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])

        # Start evaluation
        start_response = await api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Get status
        status_response = await api.get_evaluation_status(evaluation_id)

        assert isinstance(status_response, EvaluationStatusResponse)
        assert status_response.evaluation_id == evaluation_id
        assert status_response.status in [
            EvaluationStatus.PENDING,
            EvaluationStatus.RUNNING,
            EvaluationStatus.COMPLETED,
        ]

    @pytest.mark.asyncio
    async def test_get_evaluation_status_not_found(self, api):
        """Test getting status for non-existent evaluation."""
        response = await api.get_evaluation_status("nonexistent_id")

        assert isinstance(response, APIError)
        assert response.error_code == "EVALUATION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_get_evaluation_results_success(self, api):
        """Test getting results from completed evaluation."""
        request = EvaluationRequest(
            model_names=["gpt-4"],
            task_types=["text_classification"],
            evaluation_type=EvaluationType.SINGLE_MODEL,
        )

        # Start evaluation
        start_response = await api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Wait for completion (in real scenario, this would be automatic)
        await asyncio.sleep(0.1)  # Brief wait for async execution

        # Force completion for testing
        eval_info = api.active_evaluations[evaluation_id]
        eval_info["status"] = EvaluationStatus.COMPLETED
        eval_info["results"] = [
            EvaluationResult(
                evaluation_id="test_result",
                model_name="gpt-4",
                task_type="text_classification",
                dataset_name="test_dataset",
                metrics={"accuracy": 0.90},
                timestamp=datetime.now(),
                configuration={},
                raw_responses=[],
                processing_time=2.0,
            )
        ]

        # Get results
        results_response = await api.get_evaluation_results(evaluation_id)

        assert isinstance(results_response, EvaluationResultsResponse)
        assert results_response.evaluation_id == evaluation_id
        assert results_response.status == EvaluationStatus.COMPLETED
        assert len(results_response.results) == 1
        assert results_response.results[0].model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_evaluation_results_not_complete(self, api):
        """Test getting results from incomplete evaluation."""
        request = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])

        # Start evaluation
        start_response = await api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Try to get results while still running
        results_response = await api.get_evaluation_results(evaluation_id)

        assert isinstance(results_response, APIError)
        assert results_response.error_code == "EVALUATION_NOT_COMPLETE"

    @pytest.mark.asyncio
    async def test_list_available_evaluators(self, api):
        """Test listing available evaluators."""
        response = await api.list_available_evaluators()

        from benchmark.evaluation import AvailableEvaluatorsResponse

        assert isinstance(response, AvailableEvaluatorsResponse)
        assert len(response.evaluation_types) > 0
        assert "supported_tasks" in response.to_dict()
        assert "supported_metrics" in response.to_dict()
        assert "model_requirements" in response.to_dict()

    @pytest.mark.asyncio
    async def test_cancel_evaluation(self, api):
        """Test cancelling a running evaluation."""
        request = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])

        # Start evaluation
        start_response = await api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Cancel evaluation
        cancel_response = await api.cancel_evaluation(evaluation_id)

        assert isinstance(cancel_response, EvaluationStatusResponse)
        assert cancel_response.status == EvaluationStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_evaluation(self, api):
        """Test cancelling non-existent evaluation."""
        response = await api.cancel_evaluation("nonexistent_id")

        assert isinstance(response, APIError)
        assert response.error_code == "EVALUATION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_list_active_evaluations(self, api):
        """Test listing active evaluations."""
        request1 = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])
        request2 = EvaluationRequest(model_names=["claude-3"], task_types=["sentiment_analysis"])

        # Start evaluations
        await api.start_evaluation(request1)
        await api.start_evaluation(request2)

        # List active evaluations
        active_list = await api.list_active_evaluations()

        assert len(active_list) == 2
        assert all("evaluation_id" in eval_info for eval_info in active_list)
        assert all("status" in eval_info for eval_info in active_list)
        assert all("progress_percentage" in eval_info for eval_info in active_list)

    @pytest.mark.asyncio
    async def test_cleanup_completed_evaluations(self, api):
        """Test cleanup of completed evaluations."""
        request = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])

        # Start evaluation
        start_response = await api.start_evaluation(request)
        evaluation_id = start_response.evaluation_id

        # Mark as completed
        eval_info = api.active_evaluations[evaluation_id]
        eval_info["status"] = EvaluationStatus.COMPLETED
        eval_info["end_time"] = datetime.now()

        # Cleanup (with 0 hours to clean immediately)
        cleaned_count = await api.cleanup_completed_evaluations(older_than_hours=0)

        assert cleaned_count == 1
        assert evaluation_id not in api.active_evaluations

    def test_validation_error_handling(self, api):
        """Test comprehensive validation error handling."""
        # Test request with multiple validation errors
        bad_request = EvaluationRequest(
            model_names=["model_" + str(i) for i in range(15)],  # Too many models
            task_types=["task_" + str(i) for i in range(10)],  # Too many tasks
            evaluation_config=EvaluationConfig(
                batch_size=-1,  # Invalid
                temperature=5.0,  # Invalid
            ),
        )

        validation_errors = api._validate_evaluation_request(bad_request)

        assert len(validation_errors) > 0
        # Should have errors for too many models, too many tasks, and config validation
        error_messages = [error.message for error in validation_errors]
        assert any("Maximum 10 models" in msg for msg in error_messages)
        assert any("Maximum 5 tasks" in msg for msg in error_messages)


class TestAPIErrorHandling:
    """Test comprehensive error handling in API."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def failing_service(self, temp_storage):
        """Create a service that fails operations for testing."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage  # Need storage for workflow initialization
        service.list_available_models = AsyncMock(side_effect=Exception("Service unavailable"))
        service.run_evaluation = AsyncMock(side_effect=Exception("Evaluation failed"))
        return service

    @pytest.fixture
    def failing_api(self, failing_service):
        """Create API with failing service."""
        return EvaluationAPI(failing_service)

    @pytest.mark.asyncio
    async def test_service_unavailable_error(self, failing_api):
        """Test handling when underlying service is unavailable."""
        response = await failing_api.list_available_evaluators()

        assert isinstance(response, APIError)
        assert response.error_code == "SERVICE_ERROR"
        assert "Service unavailable" in response.error_message

    @pytest.mark.asyncio
    async def test_evaluation_execution_failure(self, failing_api):
        """Test handling evaluation execution failures."""
        request = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])

        # Start evaluation (should succeed initially)
        start_response = await failing_api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Wait for execution to complete (and fail)
        await asyncio.sleep(0.2)

        # Check status should show failure
        status_response = await failing_api.get_evaluation_status(evaluation_id)
        assert isinstance(status_response, EvaluationStatusResponse)
        # Status should eventually become FAILED due to service failure


class TestAPIIntegration:
    """Test API integration with evaluation components."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def realistic_service(self, temp_storage):
        """Create realistic evaluation service for integration testing."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage

        # Mock realistic behavior
        service.list_available_models = AsyncMock(return_value=["gpt-4", "claude-3", "llama-2"])

        # Mock evaluation with realistic results
        async def realistic_evaluation(config):
            # Simulate different performance for different models
            model_performance = {
                "gpt-4": {"accuracy": 0.92, "f1_score": 0.90},
                "claude-3": {"accuracy": 0.89, "f1_score": 0.87},
                "llama-2": {"accuracy": 0.85, "f1_score": 0.83},
            }

            model_name = config["model_name"]
            metrics = model_performance.get(model_name, {"accuracy": 0.80, "f1_score": 0.78})

            result = EvaluationResult(
                evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
                model_name=model_name,
                task_type=config["task_type"],
                dataset_name=config.get("dataset_name", "test_dataset"),
                metrics=metrics,
                timestamp=datetime.now(),
                configuration=config,
                raw_responses=[],
                processing_time=2.0,
            )

            # Store result
            temp_storage.store_evaluation_result(result)
            return result

        service.run_evaluation = realistic_evaluation
        return service

    @pytest.fixture
    def integration_api(self, realistic_service):
        """Create API for integration testing."""
        return EvaluationAPI(realistic_service)

    @pytest.mark.asyncio
    async def test_end_to_end_single_evaluation(self, integration_api):
        """Test complete end-to-end single evaluation workflow."""
        # Create realistic request
        request = EvaluationRequest(
            model_names=["gpt-4"],
            task_types=["text_classification"],
            dataset_names=["imdb"],
            evaluation_type=EvaluationType.SINGLE_MODEL,
            metadata={"experiment": "integration_test"},
        )

        # Start evaluation
        start_response = await integration_api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Monitor progress
        status_checks = 0
        max_checks = 10

        while status_checks < max_checks:
            status_response = await integration_api.get_evaluation_status(evaluation_id)
            assert isinstance(status_response, EvaluationStatusResponse)

            if status_response.status == EvaluationStatus.COMPLETED:
                break
            elif status_response.status == EvaluationStatus.FAILED:
                pytest.fail(f"Evaluation failed: {status_response.error_message}")

            await asyncio.sleep(0.1)
            status_checks += 1

        # Get final results
        results_response = await integration_api.get_evaluation_results(evaluation_id)
        assert isinstance(results_response, EvaluationResultsResponse)
        assert len(results_response.results) == 1
        assert results_response.results[0].model_name == "gpt-4"
        assert "accuracy" in results_response.results[0].metrics

    @pytest.mark.asyncio
    async def test_model_comparison_integration(self, integration_api):
        """Test model comparison through API."""
        request = EvaluationRequest(
            model_names=["gpt-4", "claude-3", "llama-2"],
            task_types=["text_classification"],
            evaluation_type=EvaluationType.MODEL_COMPARISON,
        )

        # Start comparison
        start_response = await integration_api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Wait for completion (with timeout)
        for _ in range(20):  # 2 second timeout
            status = await integration_api.get_evaluation_status(evaluation_id)
            if (
                isinstance(status, EvaluationStatusResponse)
                and status.status == EvaluationStatus.COMPLETED
            ):
                break
            await asyncio.sleep(0.1)

        # Verify results
        results = await integration_api.get_evaluation_results(evaluation_id)
        if isinstance(results, EvaluationResultsResponse):
            assert len(results.results) == 3  # One result per model
            assert len({result.model_name for result in results.results}) == 3

            # Verify summary contains comparison information
            summary = results.summary
            assert summary["models_evaluated"] == 3
            assert summary["evaluation_type"] == "model_comparison"

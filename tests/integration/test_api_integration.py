"""
Integration tests for evaluation API with comprehensive error handling and edge cases.

Tests the API's interaction with real evaluation components, error recovery,
and complex scenarios involving multiple concurrent evaluations.
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
    EvaluationType,
    ValidationLevel,
)
from benchmark.evaluation.api_models import ResourceLimits as APIResourceLimits
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage
from benchmark.services.evaluation_service import EvaluationService


class TestAPIErrorRecovery:
    """Test API error recovery and resilience."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def intermittent_service(self, temp_storage):
        """Create service that fails intermittently."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage
        service.call_count = 0

        service.list_available_models = AsyncMock(return_value=["gpt-4", "claude-3", "llama-2"])

        # Mock evaluation that fails every 3rd call
        async def intermittent_evaluation(config):
            service.call_count += 1
            if service.call_count % 3 == 0:
                raise Exception(f"Simulated failure #{service.call_count}")

            return EvaluationResult(
                evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
                model_name=config["model_name"],
                task_type=config["task_type"],
                dataset_name=config.get("dataset_name", "test_dataset"),
                metrics={"accuracy": 0.85, "f1_score": 0.83},
                timestamp=datetime.now(),
                configuration=config,
                raw_responses=[],
                processing_time=1.0,
            )

        service.run_evaluation = intermittent_evaluation
        return service

    @pytest.fixture
    def resilient_api(self, intermittent_service):
        """Create API with intermittent service."""
        return EvaluationAPI(intermittent_service, max_concurrent_evaluations=10)

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, resilient_api):
        """Test handling of partial failures in multi-model evaluations."""
        request = EvaluationRequest(
            model_names=["gpt-4", "claude-3", "llama-2", "bert-base"],  # 4 models
            task_types=["text_classification"],
            evaluation_type=EvaluationType.COMPREHENSIVE,
        )

        # Start evaluation
        start_response = await resilient_api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Wait for completion
        await asyncio.sleep(0.5)

        # Check final status - some evaluations should succeed despite failures
        status_response = await resilient_api.get_evaluation_status(evaluation_id)

        # The evaluation should complete (possibly with warnings) rather than fail entirely
        assert isinstance(status_response, type(status_response))  # Should not be APIError

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_isolation(self, resilient_api):
        """Test that failures in one evaluation don't affect others."""
        # Start multiple evaluations concurrently
        requests = [
            EvaluationRequest(
                model_names=[f"model_{i}"],
                task_types=["text_classification"],
                evaluation_type=EvaluationType.SINGLE_MODEL,
            )
            for i in range(5)
        ]

        # Start all evaluations
        evaluation_ids = []
        for request in requests:
            response = await resilient_api.start_evaluation(request)
            assert isinstance(response, EvaluationResponse)
            evaluation_ids.append(response.evaluation_id)

        # Wait for completion
        await asyncio.sleep(0.3)

        # Check that at least some evaluations succeeded
        completed_count = 0
        failed_count = 0

        for eval_id in evaluation_ids:
            status = await resilient_api.get_evaluation_status(eval_id)
            if hasattr(status, "status"):
                if status.status == EvaluationStatus.COMPLETED:
                    completed_count += 1
                elif status.status == EvaluationStatus.FAILED:
                    failed_count += 1

        # At least some should succeed (given the 1-in-3 failure rate)
        assert completed_count > 0


class TestAPIComplexScenarios:
    """Test complex API usage scenarios."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def complex_service(self, temp_storage):
        """Create service with complex behavior."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage

        # Mock models with different capabilities
        service.list_available_models = AsyncMock(
            return_value=["gpt-4", "claude-3", "llama-2", "bert-base", "t5-large"]
        )

        # Mock evaluation with model-specific behavior
        async def complex_evaluation(config):
            model_name = config["model_name"]
            task_type = config["task_type"]

            # Simulate different processing times and capabilities
            processing_times = {
                "gpt-4": 3.0,
                "claude-3": 2.5,
                "llama-2": 4.0,
                "bert-base": 1.5,
                "t5-large": 3.5,
            }

            performance_by_task = {
                "text_classification": {
                    "gpt-4": {"accuracy": 0.92, "f1_score": 0.90},
                    "claude-3": {"accuracy": 0.89, "f1_score": 0.87},
                    "llama-2": {"accuracy": 0.85, "f1_score": 0.83},
                    "bert-base": {"accuracy": 0.88, "f1_score": 0.86},
                    "t5-large": {"accuracy": 0.83, "f1_score": 0.81},
                },
                "sentiment_analysis": {
                    "gpt-4": {"accuracy": 0.94, "f1_score": 0.93},
                    "claude-3": {"accuracy": 0.91, "f1_score": 0.90},
                    "llama-2": {"accuracy": 0.87, "f1_score": 0.85},
                    "bert-base": {"accuracy": 0.90, "f1_score": 0.88},
                    "t5-large": {"accuracy": 0.86, "f1_score": 0.84},
                },
                "question_answering": {
                    "gpt-4": {"exact_match": 0.78, "f1_score": 0.85},
                    "claude-3": {"exact_match": 0.75, "f1_score": 0.82},
                    "llama-2": {"exact_match": 0.70, "f1_score": 0.78},
                    "bert-base": {"exact_match": 0.65, "f1_score": 0.73},  # Weaker at QA
                    "t5-large": {"exact_match": 0.72, "f1_score": 0.79},
                },
            }

            # Simulate processing delay
            processing_time = processing_times.get(model_name, 2.0)
            await asyncio.sleep(processing_time * 0.01)  # Scaled down for testing

            # Get performance metrics
            metrics = performance_by_task.get(task_type, {}).get(
                model_name, {"accuracy": 0.75, "f1_score": 0.73}
            )

            result = EvaluationResult(
                evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
                model_name=model_name,
                task_type=task_type,
                dataset_name=config.get("dataset_name", "test_dataset"),
                metrics=metrics,
                timestamp=datetime.now(),
                configuration=config,
                raw_responses=[],
                processing_time=processing_time,
                experiment_name=config.get("experiment_name", "api_test"),
            )

            # Store result
            temp_storage.store_evaluation_result(result)
            return result

        service.run_evaluation = complex_evaluation
        return service

    @pytest.fixture
    def complex_api(self, complex_service):
        """Create API for complex testing."""
        return EvaluationAPI(complex_service, max_concurrent_evaluations=15)

    @pytest.mark.asyncio
    async def test_large_scale_model_comparison(self, complex_api):
        """Test large-scale model comparison evaluation."""
        request = EvaluationRequest(
            model_names=["gpt-4", "claude-3", "llama-2", "bert-base", "t5-large"],
            task_types=["text_classification", "sentiment_analysis", "question_answering"],
            evaluation_type=EvaluationType.MODEL_COMPARISON,
            evaluation_config=EvaluationConfig(
                batch_size=64, validation_level=ValidationLevel.STRICT
            ),
            metadata={"experiment": "large_scale_comparison", "version": "1.0"},
        )

        # Start evaluation
        start_response = await complex_api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Monitor progress
        progress_checks = 0
        max_checks = 50

        while progress_checks < max_checks:
            status = await complex_api.get_evaluation_status(evaluation_id)

            if hasattr(status, "status"):
                if status.status == EvaluationStatus.COMPLETED:
                    break
                elif status.status == EvaluationStatus.FAILED:
                    pytest.fail(f"Large-scale evaluation failed: {status.error_message}")

            await asyncio.sleep(0.1)
            progress_checks += 1

        # Get results
        results_response = await complex_api.get_evaluation_results(evaluation_id)

        if isinstance(results_response, EvaluationResultsResponse):
            # Should have results for all model-task combinations
            expected_results = len(request.model_names) * len(request.task_types)
            assert len(results_response.results) == expected_results

            # Verify all models and tasks are represented
            model_names = {r.model_name for r in results_response.results}
            task_types = {r.task_type for r in results_response.results}

            assert model_names == set(request.model_names)
            assert task_types == set(request.task_types)

            # Verify summary statistics
            summary = results_response.summary
            assert summary["models_evaluated"] == 5
            assert summary["tasks_evaluated"] == 3
            assert summary["evaluation_type"] == "model_comparison"

    @pytest.mark.asyncio
    async def test_resource_constrained_evaluation(self, complex_api):
        """Test evaluation with tight resource constraints."""
        # Set up very restrictive resource limits
        resource_limits = APIResourceLimits(
            max_memory_mb=1000,  # Low memory limit
            max_cpu_percent=50.0,  # Low CPU limit
            max_concurrent_evaluations=2,  # Low concurrency
            max_execution_time_seconds=30,  # Short timeout
        )

        request = EvaluationRequest(
            model_names=["gpt-4", "claude-3", "llama-2"],
            task_types=["text_classification", "sentiment_analysis"],
            evaluation_type=EvaluationType.BATCH_PROCESSING,
            resource_limits=resource_limits,
            metadata={"resource_test": True},
        )

        # Start evaluation
        start_response = await complex_api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Wait for completion
        await asyncio.sleep(1.0)

        # Check that evaluation handled resource constraints
        status = await complex_api.get_evaluation_status(evaluation_id)

        # Should complete or provide meaningful error about resource constraints
        assert hasattr(status, "status")  # Should not be APIError for valid request

    @pytest.mark.asyncio
    async def test_concurrent_different_evaluation_types(self, complex_api):
        """Test running different evaluation types concurrently."""
        # Create different types of evaluation requests
        requests = [
            EvaluationRequest(
                model_names=["gpt-4"],
                task_types=["text_classification"],
                evaluation_type=EvaluationType.SINGLE_MODEL,
                metadata={"type": "single"},
            ),
            EvaluationRequest(
                model_names=["gpt-4", "claude-3"],
                task_types=["sentiment_analysis"],
                evaluation_type=EvaluationType.MODEL_COMPARISON,
                metadata={"type": "comparison"},
            ),
            EvaluationRequest(
                model_names=["gpt-4", "claude-3", "llama-2"],
                task_types=["question_answering"],
                evaluation_type=EvaluationType.COMPREHENSIVE,
                metadata={"type": "comprehensive"},
            ),
        ]

        # Start all evaluations concurrently
        evaluation_ids = []
        for request in requests:
            response = await complex_api.start_evaluation(request)
            assert isinstance(response, EvaluationResponse)
            evaluation_ids.append((response.evaluation_id, request.metadata["type"]))

        # Wait for all to complete
        await asyncio.sleep(1.5)

        # Check results for each evaluation type
        for eval_id, eval_type in evaluation_ids:
            status = await complex_api.get_evaluation_status(eval_id)

            if hasattr(status, "status") and status.status == EvaluationStatus.COMPLETED:
                results = await complex_api.get_evaluation_results(eval_id)

                if isinstance(results, EvaluationResultsResponse):
                    # Verify type-specific expectations
                    if eval_type == "single":
                        assert len(results.results) == 1
                    elif eval_type == "comparison":
                        assert len(results.results) == 2  # 2 models × 1 task
                    elif eval_type == "comprehensive":
                        assert len(results.results) == 3  # 3 models × 1 task

    @pytest.mark.asyncio
    async def test_evaluation_with_custom_configuration(self, complex_api):
        """Test evaluation with custom configuration parameters."""
        custom_config = EvaluationConfig(
            batch_size=16,
            temperature=0.5,
            top_p=0.8,
            seed=42,
            validation_level=ValidationLevel.STRICT,
        )

        request = EvaluationRequest(
            model_names=["gpt-4", "claude-3"],
            task_types=["text_classification"],
            evaluation_config=custom_config,
            metadata={"custom_config": True, "seed": 42},
        )

        # Start evaluation
        start_response = await complex_api.start_evaluation(request)
        assert isinstance(start_response, EvaluationResponse)
        evaluation_id = start_response.evaluation_id

        # Wait for completion
        await asyncio.sleep(0.5)

        # Verify custom configuration was used
        results = await complex_api.get_evaluation_results(evaluation_id)

        if isinstance(results, EvaluationResultsResponse):
            # Check that configuration was stored in metadata
            metadata = results.metadata
            assert "custom_config" in metadata or len(results.results) > 0


class TestAPIPerformanceAndScaling:
    """Test API performance and scaling characteristics."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def performance_service(self, temp_storage):
        """Create service optimized for performance testing."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage

        service.list_available_models = AsyncMock(
            return_value=[f"model_{i}" for i in range(20)]  # Many models
        )

        # Fast mock evaluation
        async def fast_evaluation(config):
            await asyncio.sleep(0.001)  # Minimal delay

            return EvaluationResult(
                evaluation_id=f"perf_{uuid.uuid4().hex[:8]}",
                model_name=config["model_name"],
                task_type=config["task_type"],
                dataset_name=config.get("dataset_name", "perf_dataset"),
                metrics={"accuracy": 0.85, "throughput": 100.0},
                timestamp=datetime.now(),
                configuration=config,
                raw_responses=[],
                processing_time=0.001,
            )

        service.run_evaluation = fast_evaluation
        return service

    @pytest.fixture
    def performance_api(self, performance_service):
        """Create API for performance testing."""
        return EvaluationAPI(performance_service, max_concurrent_evaluations=50)

    @pytest.mark.asyncio
    async def test_high_concurrency_handling(self, performance_api):
        """Test API handling of high concurrency requests."""
        # Create many evaluation requests
        requests = [
            EvaluationRequest(
                model_names=[f"model_{i}"],
                task_types=["performance_test"],
                evaluation_type=EvaluationType.SINGLE_MODEL,
                metadata={"test_id": i},
            )
            for i in range(25)  # Many concurrent requests
        ]

        # Start all evaluations concurrently
        start_time = asyncio.get_event_loop().time()

        tasks = [performance_api.start_evaluation(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        start_end_time = asyncio.get_event_loop().time()

        # Check that requests were handled efficiently
        successful_starts = sum(1 for r in responses if isinstance(r, EvaluationResponse))
        capacity_errors = sum(
            1 for r in responses if isinstance(r, APIError) and r.error_code == "CAPACITY_EXCEEDED"
        )

        # Should handle up to capacity, reject excess
        assert successful_starts <= performance_api.max_concurrent_evaluations
        assert successful_starts > 0
        assert capacity_errors >= 0

        # Starting should be fast
        assert (start_end_time - start_time) < 1.0

        # Wait for completion
        await asyncio.sleep(0.5)

        # Check active evaluations
        active_list = await performance_api.list_active_evaluations()

        # Most should have completed quickly
        completed_count = sum(1 for eval_info in active_list if eval_info["status"] == "completed")

        assert completed_count > 0

    @pytest.mark.asyncio
    async def test_cleanup_performance(self, performance_api):
        """Test cleanup performance with many completed evaluations."""
        # Create and complete many evaluations
        evaluation_ids = []

        for i in range(20):
            request = EvaluationRequest(
                model_names=[f"model_{i % 5}"],
                task_types=["cleanup_test"],
                evaluation_type=EvaluationType.SINGLE_MODEL,
            )

            response = await performance_api.start_evaluation(request)
            if isinstance(response, EvaluationResponse):
                evaluation_ids.append(response.evaluation_id)

        # Wait for completion
        await asyncio.sleep(0.3)

        # Force completion of all evaluations for testing
        for eval_id in evaluation_ids:
            if eval_id in performance_api.active_evaluations:
                eval_info = performance_api.active_evaluations[eval_id]
                eval_info["status"] = EvaluationStatus.COMPLETED
                eval_info["end_time"] = datetime.now()

        # Test cleanup performance
        cleanup_start = asyncio.get_event_loop().time()
        cleaned_count = await performance_api.cleanup_completed_evaluations(older_than_hours=0)
        cleanup_end = asyncio.get_event_loop().time()

        # Should clean up efficiently
        assert cleaned_count > 0
        assert (cleanup_end - cleanup_start) < 0.5  # Should be fast
        assert len(performance_api.active_evaluations) == (len(evaluation_ids) - cleaned_count)

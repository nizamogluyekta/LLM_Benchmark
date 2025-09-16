"""
Comprehensive test suite for BatchEvaluator.

Tests batch processing, parallel evaluation, resource management,
and performance optimization capabilities.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmark.evaluation.batch_evaluator import (
    BatchEvaluator,
    BatchResult,
    BatchScheduler,
    BatchStatus,
    ResourceLimits,
    ResourceMonitor,
)
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage
from benchmark.services.evaluation_service import EvaluationService


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""

    @pytest.fixture
    def monitor(self):
        """Create resource monitor instance."""
        return ResourceMonitor()

    @pytest.mark.asyncio
    async def test_resource_collection(self, monitor):
        """Test resource usage collection."""
        usage = await monitor._collect_resource_usage()

        assert "timestamp" in usage
        assert "cpu_percent" in usage
        assert "memory_mb" in usage
        assert "gpu_memory_mb" in usage
        assert "disk_io_mbps" in usage

        # Values should be non-negative
        assert usage["cpu_percent"] >= 0
        assert usage["memory_mb"] >= 0

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring for a short period
        monitor_task = asyncio.create_task(monitor.start_monitoring(interval_seconds=0.1))

        # Let it collect some data
        await asyncio.sleep(0.3)

        # Stop monitoring
        summary = monitor.stop_monitoring()
        monitor_task.cancel()

        assert "monitoring_duration" in summary
        assert "cpu_usage" in summary
        assert "memory_usage" in summary
        assert "usage_history" in summary

        # Should have collected multiple samples
        assert summary["monitoring_duration"] >= 2
        assert len(summary["usage_history"]) >= 2

    def test_resource_limit_checking(self, monitor):
        """Test resource limit validation."""
        limits = ResourceLimits(
            max_memory_mb=1000, max_cpu_percent=80.0, max_concurrent_evaluations=4
        )

        # Usage within limits
        current_usage = {"memory_mb": 500.0, "cpu_percent": 60.0}

        within_limits, violations = monitor.check_resource_limits(limits, current_usage)
        assert within_limits
        assert len(violations) == 0

        # Usage exceeding limits
        excessive_usage = {"memory_mb": 1500.0, "cpu_percent": 95.0}

        within_limits, violations = monitor.check_resource_limits(limits, excessive_usage)
        assert not within_limits
        assert len(violations) == 2
        assert "Memory usage" in violations[0]
        assert "CPU usage" in violations[1]


class TestBatchScheduler:
    """Test BatchScheduler functionality."""

    @pytest.fixture
    def scheduler(self):
        """Create batch scheduler instance."""
        return BatchScheduler(max_concurrent_batches=2)

    def test_batch_scheduling(self, scheduler):
        """Test basic batch scheduling."""
        # First batch should be scheduled immediately
        scheduled = scheduler.schedule_batch("batch_1")
        assert scheduled
        assert "batch_1" in scheduler.running_batches

        # Second batch should also be scheduled
        scheduled = scheduler.schedule_batch("batch_2")
        assert scheduled
        assert "batch_2" in scheduler.running_batches

        # Third batch should be queued
        scheduled = scheduler.schedule_batch("batch_3", priority=5)
        assert not scheduled
        assert len(scheduler.pending_batches) == 1
        assert scheduler.pending_batches[0]["batch_id"] == "batch_3"

    def test_batch_completion_and_scheduling(self, scheduler):
        """Test batch completion and next batch scheduling."""
        # Fill up running slots
        scheduler.schedule_batch("batch_1")
        scheduler.schedule_batch("batch_2")

        # Queue additional batches with different priorities
        scheduler.schedule_batch("batch_3", priority=10)  # High priority
        scheduler.schedule_batch("batch_4", priority=5)  # Lower priority

        # Complete a batch
        next_batch = scheduler.complete_batch("batch_1")

        # High priority batch should be scheduled next
        assert next_batch == "batch_3"
        assert "batch_3" in scheduler.running_batches
        assert "batch_1" not in scheduler.running_batches

    def test_priority_ordering(self, scheduler):
        """Test priority-based batch ordering."""
        # Fill running slots
        scheduler.schedule_batch("batch_1")
        scheduler.schedule_batch("batch_2")

        # Add batches with different priorities
        scheduler.schedule_batch("batch_low", priority=1)
        scheduler.schedule_batch("batch_high", priority=20)
        scheduler.schedule_batch("batch_medium", priority=10)

        # Complete a running batch
        next_batch = scheduler.complete_batch("batch_1")

        # Highest priority should be scheduled
        assert next_batch == "batch_high"

        # Check remaining queue order
        remaining_priorities = [b["priority"] for b in scheduler.pending_batches]
        assert remaining_priorities == [10, 1]  # Sorted high to low

    def test_queue_status(self, scheduler):
        """Test queue status reporting."""
        # Add some batches
        scheduler.schedule_batch("batch_1")
        scheduler.schedule_batch("batch_2")
        scheduler.schedule_batch("batch_3", priority=5)

        status = scheduler.get_queue_status()

        assert status["running_batches"] == 2
        assert status["pending_batches"] == 1
        assert status["max_concurrent"] == 2
        assert len(status["queue_details"]) == 1


class TestBatchEvaluator:
    """Test BatchEvaluator functionality."""

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

        # Mock async evaluation method
        service.run_evaluation = AsyncMock(return_value=self._create_mock_result())

        return service

    @pytest.fixture
    def batch_evaluator(self, mock_evaluation_service):
        """Create batch evaluator instance."""
        resource_limits = ResourceLimits(
            max_concurrent_evaluations=2, max_memory_mb=1000, timeout_seconds=30
        )
        return BatchEvaluator(mock_evaluation_service, resource_limits)

    def _create_mock_result(self, model_name="test_model", task_type="test_task"):
        """Create mock evaluation result."""
        return EvaluationResult(
            evaluation_id=f"eval_{int(time.time() * 1000)}",
            model_name=model_name,
            task_type=task_type,
            dataset_name="test_dataset",
            metrics={"accuracy": 0.85, "f1_score": 0.82},
            timestamp=datetime.now(),
            configuration={"learning_rate": 0.001},
            raw_responses=[],
            processing_time=1.0,  # Fast for testing
        )

    def test_batch_evaluator_initialization(self, batch_evaluator):
        """Test batch evaluator initialization."""
        assert batch_evaluator.service is not None
        assert batch_evaluator.default_limits is not None
        assert isinstance(batch_evaluator.resource_monitor, ResourceMonitor)
        assert isinstance(batch_evaluator.scheduler, BatchScheduler)
        assert isinstance(batch_evaluator.active_batches, dict)

    @pytest.mark.asyncio
    async def test_simple_batch_evaluation(self, batch_evaluator):
        """Test basic batch evaluation functionality."""
        batch_config = {
            "models": ["model_a", "model_b"],
            "tasks": ["task_1"],
            "evaluation_params": {"batch_size": 16},
        }

        result = await batch_evaluator.evaluate_batch(batch_config)

        assert isinstance(result, BatchResult)
        assert result.status == BatchStatus.COMPLETED
        assert result.total_evaluations == 2  # 2 models × 1 task
        assert result.successful_evaluations == 2
        assert result.failed_evaluations == 0
        assert len(result.results) == 2
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0

        # Verify service calls
        assert batch_evaluator.service.run_evaluation.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_evaluation_with_failures(self, batch_evaluator):
        """Test batch evaluation with some failures."""

        # Make some evaluations fail
        def side_effect(config):
            if config["model_name"] == "failing_model":
                raise Exception("Model failed to load")
            return self._create_mock_result(config["model_name"], config["task_type"])

        batch_evaluator.service.run_evaluation = AsyncMock(side_effect=side_effect)

        batch_config = {
            "models": ["working_model", "failing_model", "another_working_model"],
            "tasks": ["task_1"],
        }

        result = await batch_evaluator.evaluate_batch(batch_config)

        assert result.status == BatchStatus.COMPLETED
        assert result.total_evaluations == 3
        assert result.successful_evaluations == 2
        assert result.failed_evaluations == 1
        assert len(result.errors) == 1
        assert "Model failed to load" in result.errors[0]

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self, batch_evaluator):
        """Test parallel evaluation with multiple workers."""
        # Create evaluation configs
        evaluation_configs = [
            {"model_name": f"model_{i}", "task_type": "test_task"} for i in range(6)
        ]

        start_time = time.time()
        results = await batch_evaluator.parallel_evaluation(evaluation_configs, max_workers=3)
        end_time = time.time()

        assert len(results) == 6

        # Count successful evaluations
        successful = [r for r in results if r.get("status") == "success"]
        assert len(successful) == 6

        # Parallel execution should be faster than sequential
        # (This is approximate since we're using mocked evaluations)
        duration = end_time - start_time
        assert duration < 10.0  # Should complete quickly with mocked evaluations

    @pytest.mark.asyncio
    async def test_parallel_evaluation_with_chunking(self, batch_evaluator):
        """Test parallel evaluation with chunking for large batches."""
        # Create large batch
        evaluation_configs = [
            {"model_name": f"model_{i}", "task_type": "test_task"} for i in range(10)
        ]

        results = await batch_evaluator.parallel_evaluation(
            evaluation_configs,
            max_workers=2,
            chunk_size=4,  # Process in chunks of 4
        )

        assert len(results) == 10
        successful = [r for r in results if r.get("status") == "success"]
        assert len(successful) == 10

    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self, batch_evaluator):
        """Test resource limit enforcement during batch processing."""
        # Create batch with timeout limit
        resource_limits = ResourceLimits(
            max_concurrent_evaluations=1,
            timeout_seconds=1,  # Very short timeout
        )

        # Make evaluation slow to trigger timeout
        async def slow_evaluation(_config):
            await asyncio.sleep(2.0)  # Longer than timeout
            return self._create_mock_result()

        batch_evaluator.service.run_evaluation = slow_evaluation

        batch_config = {"evaluations": [{"model_name": "slow_model", "task_type": "test_task"}]}

        result = await batch_evaluator.evaluate_batch(batch_config, resource_limits)

        assert result.status == BatchStatus.COMPLETED
        assert result.failed_evaluations == 1
        assert "timed out" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_adaptive_batch_processing(self, batch_evaluator):
        """Test adaptive batch processing with dynamic sizing."""
        # Create configurations for adaptive processing
        evaluation_configs = [
            {"model_name": f"model_{i}", "task_type": "test_task"} for i in range(20)
        ]

        # Mock fast evaluations to trigger batch size increases
        batch_evaluator.service.run_evaluation = AsyncMock(return_value=self._create_mock_result())

        batch_results = await batch_evaluator.adaptive_batch_processing(
            evaluation_configs,
            initial_batch_size=5,
            target_duration_minutes=0.1,  # Very short target for testing
            max_batch_size=15,
        )

        assert len(batch_results) >= 1

        # Total successful evaluations should equal input
        total_successful = sum(br.successful_evaluations for br in batch_results)
        assert total_successful == 20

        # Should have adapted batch sizes (though exact behavior depends on timing)
        assert all(br.status == BatchStatus.COMPLETED for br in batch_results)

    def test_batch_status_tracking(self, batch_evaluator):
        """Test batch status tracking functionality."""
        # Create a batch result manually
        batch_id = "test_batch_123"
        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.RUNNING,
            total_evaluations=10,
            successful_evaluations=7,
            failed_evaluations=1,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            results=[],
            errors=["Test error"],
            resource_usage={},
            performance_metrics={},
        )

        batch_evaluator.active_batches[batch_id] = batch_result

        status = batch_evaluator.get_batch_status(batch_id)

        assert status is not None
        assert status["batch_id"] == batch_id
        assert status["status"] == "running"
        assert status["progress"]["total"] == 10
        assert status["progress"]["completed"] == 8  # successful + failed
        assert status["progress"]["successful"] == 7
        assert status["progress"]["failed"] == 1
        assert len(status["errors"]) == 1

    def test_list_active_batches(self, batch_evaluator):
        """Test listing active batches."""
        # Create several active batches
        batch_data = [
            ("batch_1", BatchStatus.RUNNING, 10, 5),
            ("batch_2", BatchStatus.COMPLETED, 8, 8),
            ("batch_3", BatchStatus.FAILED, 12, 3),
        ]

        for batch_id, status, total, completed in batch_data:
            batch_result = BatchResult(
                batch_id=batch_id,
                status=status,
                total_evaluations=total,
                successful_evaluations=completed,
                failed_evaluations=0,
                start_time=datetime.now(),
                end_time=None,
                duration_seconds=None,
                results=[],
                errors=[],
                resource_usage={},
                performance_metrics={},
            )
            batch_evaluator.active_batches[batch_id] = batch_result

        active_batches = batch_evaluator.list_active_batches()

        assert len(active_batches) == 3

        # Find specific batch
        batch_1 = next(b for b in active_batches if b["batch_id"] == "batch_1")
        assert batch_1["status"] == "running"
        assert batch_1["progress_percent"] == 50.0  # 5/10 * 100

    @pytest.mark.asyncio
    async def test_batch_cancellation(self, batch_evaluator):
        """Test batch cancellation functionality."""
        # Create running batch
        batch_id = "cancellable_batch"
        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.RUNNING,
            total_evaluations=5,
            successful_evaluations=0,
            failed_evaluations=0,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            results=[],
            errors=[],
            resource_usage={},
            performance_metrics={},
        )

        batch_evaluator.active_batches[batch_id] = batch_result

        # Cancel the batch
        success = await batch_evaluator.cancel_batch(batch_id)

        assert success
        assert batch_result.status == BatchStatus.CANCELLED
        assert batch_result.end_time is not None

        # Cannot cancel already cancelled batch
        success = await batch_evaluator.cancel_batch(batch_id)
        assert not success

    def test_batch_cleanup(self, batch_evaluator):
        """Test cleanup of old completed batches."""
        # Create batches with different completion times
        old_time = datetime.now() - timedelta(hours=25)
        recent_time = datetime.now() - timedelta(hours=1)

        batch_data = [
            ("old_completed", BatchStatus.COMPLETED, old_time),
            ("recent_completed", BatchStatus.COMPLETED, recent_time),
            ("old_failed", BatchStatus.FAILED, old_time),
            ("running", BatchStatus.RUNNING, None),
        ]

        for batch_id, status, end_time in batch_data:
            batch_result = BatchResult(
                batch_id=batch_id,
                status=status,
                total_evaluations=5,
                successful_evaluations=5,
                failed_evaluations=0,
                start_time=datetime.now(),
                end_time=end_time,
                duration_seconds=None,
                results=[],
                errors=[],
                resource_usage={},
                performance_metrics={},
            )
            batch_evaluator.active_batches[batch_id] = batch_result

        # Cleanup old batches
        cleaned_count = batch_evaluator.cleanup_completed_batches(older_than_hours=24)

        assert cleaned_count == 2  # old_completed and old_failed
        assert "old_completed" not in batch_evaluator.active_batches
        assert "old_failed" not in batch_evaluator.active_batches
        assert "recent_completed" in batch_evaluator.active_batches
        assert "running" in batch_evaluator.active_batches

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, batch_evaluator):
        """Test performance metrics calculation."""
        batch_config = {"models": ["model_a", "model_b"], "tasks": ["task_1", "task_2"]}

        result = await batch_evaluator.evaluate_batch(batch_config)

        # Check performance metrics
        metrics = result.performance_metrics
        assert "evaluations_per_second" in metrics
        assert "avg_evaluation_time" in metrics
        assert "success_rate" in metrics

        # Verify calculations
        assert metrics["success_rate"] == 1.0  # All should succeed
        assert metrics["evaluations_per_second"] > 0
        assert metrics["avg_evaluation_time"] > 0

    @pytest.mark.asyncio
    async def test_config_extraction_methods(self, batch_evaluator):
        """Test evaluation config extraction methods."""
        # Test direct evaluation configs
        batch_config = {
            "evaluations": [
                {"model_name": "model_a", "task_type": "task_1"},
                {"model_name": "model_b", "task_type": "task_2"},
            ]
        }

        configs = await batch_evaluator._extract_evaluation_configs(batch_config)
        assert len(configs) == 2
        assert configs[0]["model_name"] == "model_a"

        # Test generated configs from parameters
        batch_config = {
            "models": ["model_a", "model_b"],
            "tasks": ["task_1", "task_2"],
            "datasets": ["dataset_1"],
            "evaluation_params": {"batch_size": 32},
        }

        configs = await batch_evaluator._extract_evaluation_configs(batch_config)
        assert len(configs) == 4  # 2 models × 2 tasks × 1 dataset

        # All configs should have the evaluation params
        for config in configs:
            assert config["batch_size"] == 32
            assert "model_name" in config
            assert "task_type" in config
            assert "dataset_name" in config

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, batch_evaluator):
        """Test processing multiple batches concurrently."""
        config1 = {"models": ["model_a"], "tasks": ["task_1"]}
        config2 = {"models": ["model_b"], "tasks": ["task_2"]}

        # Process batches concurrently
        task1 = asyncio.create_task(batch_evaluator.evaluate_batch(config1, batch_id="batch_1"))
        task2 = asyncio.create_task(batch_evaluator.evaluate_batch(config2, batch_id="batch_2"))

        result1, result2 = await asyncio.gather(task1, task2)

        assert result1.batch_id == "batch_1"
        assert result2.batch_id == "batch_2"
        assert result1.status == BatchStatus.COMPLETED
        assert result2.status == BatchStatus.COMPLETED

        # Both batches should be in active batches during execution
        assert len(batch_evaluator.active_batches) == 2

    @pytest.mark.asyncio
    async def test_resource_monitoring_integration(self, batch_evaluator):
        """Test integration with resource monitoring."""
        batch_config = {"models": ["model_a"], "tasks": ["task_1"]}

        result = await batch_evaluator.evaluate_batch(batch_config)

        # Resource usage should be recorded
        assert "usage_history" in result.resource_usage
        assert len(result.resource_usage["usage_history"]) > 0

        # Performance metrics should include resource efficiency
        if result.performance_metrics.get("avg_cpu_usage", 0) > 0:
            assert "cpu_efficiency" in result.performance_metrics

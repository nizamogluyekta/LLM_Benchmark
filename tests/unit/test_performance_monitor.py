"""
Unit tests for the ModelPerformanceMonitor.

This module tests performance metric collection, resource tracking, analysis,
and issue detection functionality with mocked system resources.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from benchmark.models.performance_monitor import (
    InferenceContext,
    InferenceMetric,
    ModelPerformanceMonitor,
    PerformanceIssue,
    PerformanceIssueType,
    PerformanceSummary,
    ResourceTracker,
    TimeRange,
)


class MockResourceTracker:
    """Mock resource tracker for consistent testing."""

    def __init__(self):
        self.mock_memory = 1024.0  # MB
        self.mock_cpu = 25.0  # %
        self.mock_gpu = 50.0  # %
        self.mock_neural_engine = None
        self.call_count = 0

    def get_memory_usage(self) -> float:
        """Return mock memory usage."""
        self.call_count += 1
        return self.mock_memory + (self.call_count * 10)  # Simulate slight increase

    def get_cpu_utilization(self) -> float:
        """Return mock CPU utilization."""
        return self.mock_cpu

    def get_gpu_utilization(self) -> float:
        """Return mock GPU utilization."""
        return self.mock_gpu

    def get_neural_engine_usage(self) -> float:
        """Return mock Neural Engine usage."""
        return self.mock_neural_engine

    def get_system_metrics(self) -> dict[str, Any]:
        """Return mock system metrics."""
        return {
            "timestamp": datetime.now(),
            "memory_total_gb": 16.0,
            "memory_available_gb": 8.0,
            "memory_percent": 50.0,
            "cpu_percent": self.mock_cpu,
            "disk_usage_percent": 60.0,
            "gpu_utilization": self.mock_gpu,
            "neural_engine_usage": self.mock_neural_engine,
        }


class TestResourceTracker:
    """Test cases for ResourceTracker."""

    def test_resource_tracker_initialization(self):
        """Test ResourceTracker initialization."""
        tracker = ResourceTracker()
        assert tracker is not None
        assert hasattr(tracker, "logger")

    @patch("psutil.Process")
    def test_get_memory_usage(self, mock_process):
        """Test memory usage retrieval."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info

        tracker = ResourceTracker()
        memory_mb = tracker.get_memory_usage()

        assert memory_mb == 1024.0  # Should be 1024 MB
        mock_process.return_value.memory_info.assert_called_once()

    @patch("psutil.cpu_percent")
    def test_get_cpu_utilization(self, mock_cpu_percent):
        """Test CPU utilization retrieval."""
        mock_cpu_percent.return_value = 75.5

        tracker = ResourceTracker()
        cpu_usage = tracker.get_cpu_utilization()

        assert cpu_usage == 75.5
        mock_cpu_percent.assert_called_once_with(interval=0.1)

    @patch("platform.system")
    @patch("platform.machine")
    def test_m4_pro_detection(self, mock_machine, mock_system):
        """Test M4 Pro detection."""
        # Test M4 Pro detection (positive case)
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"

        tracker = ResourceTracker()
        assert tracker._system_info.get("is_m4_pro", False) is True

        # Test non-M4 Pro (negative case)
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"

        tracker = ResourceTracker()
        assert tracker._system_info.get("is_m4_pro", False) is False

    def test_get_gpu_utilization_non_m4(self):
        """Test GPU utilization on non-M4 Pro systems."""
        with patch("platform.system", return_value="Linux"):
            tracker = ResourceTracker()
            gpu_util = tracker.get_gpu_utilization()
            assert gpu_util is None

    def test_get_neural_engine_usage_non_m4(self):
        """Test Neural Engine usage on non-M4 Pro systems."""
        with patch("platform.system", return_value="Linux"):
            tracker = ResourceTracker()
            ne_usage = tracker.get_neural_engine_usage()
            assert ne_usage is None


class TestInferenceMetric:
    """Test cases for InferenceMetric model."""

    def test_inference_metric_creation(self):
        """Test InferenceMetric model creation."""
        metric = InferenceMetric(
            model_id="test_model",
            request_id="req_001",
            inference_time_ms=150.5,
            memory_usage_mb=512.0,
            peak_memory_mb=600.0,
            batch_size=2,
            input_length=100,
            output_length=50,
            success=True,
            cpu_utilization=45.0,
        )

        assert metric.model_id == "test_model"
        assert metric.request_id == "req_001"
        assert metric.inference_time_ms == 150.5
        assert metric.batch_size == 2
        assert metric.success is True

    def test_inference_metric_validation(self):
        """Test InferenceMetric validation."""
        # Test negative inference time
        with pytest.raises(ValueError):
            InferenceMetric(
                model_id="test",
                request_id="test",
                inference_time_ms=-10.0,  # Invalid
                memory_usage_mb=100.0,
                peak_memory_mb=100.0,
                batch_size=1,
                input_length=10,
                output_length=10,
                success=True,
                cpu_utilization=50.0,
            )

        # Test invalid batch size
        with pytest.raises(ValueError):
            InferenceMetric(
                model_id="test",
                request_id="test",
                inference_time_ms=100.0,
                memory_usage_mb=100.0,
                peak_memory_mb=100.0,
                batch_size=0,  # Invalid
                input_length=10,
                output_length=10,
                success=True,
                cpu_utilization=50.0,
            )


class TestModelPerformanceMonitor:
    """Test cases for ModelPerformanceMonitor."""

    @pytest_asyncio.fixture
    async def performance_monitor(self):
        """Create a performance monitor with mocked resource tracker."""
        monitor = ModelPerformanceMonitor()
        monitor.resource_tracker = MockResourceTracker()
        return monitor

    @pytest.mark.asyncio
    async def test_start_inference_measurement(self, performance_monitor):
        """Test starting inference measurement."""
        context = await performance_monitor.start_inference_measurement(
            model_id="test_model",
            batch_size=4,
            input_length=256,
        )

        assert isinstance(context, InferenceContext)
        assert context.model_id == "test_model"
        assert context.batch_size == 4
        assert context.input_length == 256
        assert context.request_id is not None
        assert context.start_memory > 0

    @pytest.mark.asyncio
    async def test_end_inference_measurement(self, performance_monitor):
        """Test ending inference measurement."""
        # Start measurement
        context = await performance_monitor.start_inference_measurement(
            model_id="test_model",
            batch_size=2,
            input_length=128,
        )

        # Simulate some time passing
        await asyncio.sleep(0.01)

        # End measurement
        metric = await performance_monitor.end_inference_measurement(
            context=context,
            success=True,
            output_length=64,
            tokens_generated=64,
            time_to_first_token_ms=50.0,
        )

        assert isinstance(metric, InferenceMetric)
        assert metric.model_id == "test_model"
        assert metric.request_id == context.request_id
        assert metric.inference_time_ms > 0
        assert metric.output_length == 64
        assert metric.tokens_per_second is not None
        assert metric.tokens_per_second > 0
        assert metric.success is True

        # Check that metric was stored
        assert "test_model" in performance_monitor.metrics
        assert len(performance_monitor.metrics["test_model"]) == 1

    @pytest.mark.asyncio
    async def test_end_inference_measurement_failure(self, performance_monitor):
        """Test ending inference measurement with failure."""
        context = await performance_monitor.start_inference_measurement(
            model_id="test_model",
            batch_size=1,
        )

        metric = await performance_monitor.end_inference_measurement(
            context=context,
            success=False,
            error_message="Test error",
        )

        assert metric.success is False
        assert metric.error_message == "Test error"

    @pytest.mark.asyncio
    async def test_metric_storage_limits(self, performance_monitor):
        """Test that metric storage respects limits."""
        performance_monitor.max_metrics_per_model = 5

        # Generate more metrics than the limit
        for _ in range(10):
            context = await performance_monitor.start_inference_measurement(
                model_id="test_model",
                batch_size=1,
            )
            await performance_monitor.end_inference_measurement(
                context=context,
                success=True,
            )

        # Should only keep the most recent 5 metrics
        assert len(performance_monitor.metrics["test_model"]) == 5

    @pytest.mark.asyncio
    async def test_performance_summary_empty(self, performance_monitor):
        """Test performance summary with no metrics."""
        summary = await performance_monitor.get_performance_summary("nonexistent_model")

        assert isinstance(summary, PerformanceSummary)
        assert summary.model_id == "nonexistent_model"
        assert summary.total_inferences == 0
        assert summary.successful_inferences == 0
        assert summary.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_performance_summary_with_data(self, performance_monitor):
        """Test performance summary with actual metrics."""
        model_id = "test_model"

        # Generate test metrics
        for i in range(20):
            context = await performance_monitor.start_inference_measurement(
                model_id=model_id,
                batch_size=2 if i % 2 == 0 else 4,
                input_length=100,
            )

            success = i < 18  # 90% success rate
            await performance_monitor.end_inference_measurement(
                context=context,
                success=success,
                output_length=50,
                tokens_generated=50 if success else None,
                error_message=None if success else "Test error",
            )

        # Get performance summary
        summary = await performance_monitor.get_performance_summary(model_id)

        assert summary.model_id == model_id
        assert summary.total_inferences == 20
        assert summary.successful_inferences == 18
        assert abs(summary.error_rate - 0.1) < 0.01  # ~10% error rate
        assert summary.avg_inference_time_ms > 0
        assert summary.avg_memory_usage_mb > 0
        assert summary.performance_trend in ["improving", "stable", "degrading"]

        # Check batch size distribution
        assert 2 in summary.batch_size_distribution
        assert 4 in summary.batch_size_distribution
        assert summary.optimal_batch_size in [2, 4]

    @pytest.mark.asyncio
    async def test_performance_summary_with_time_range(self, performance_monitor):
        """Test performance summary with time range filtering."""
        model_id = "test_model"

        # Generate metrics with different timestamps
        now = datetime.now()
        old_time = now - timedelta(hours=2)

        # Create context manually with old timestamp
        context = InferenceContext(
            model_id=model_id,
            batch_size=1,
            input_length=100,
            start_memory=1000.0,
            initial_cpu=20.0,
        )
        context.start_time = old_time

        old_metric = InferenceMetric(
            model_id=model_id,
            request_id="old_request",
            timestamp=old_time,
            inference_time_ms=200.0,
            memory_usage_mb=1000.0,
            peak_memory_mb=1000.0,
            batch_size=1,
            input_length=100,
            output_length=50,
            success=True,
            cpu_utilization=20.0,
        )

        performance_monitor.metrics[model_id].append(old_metric)

        # Add recent metric
        recent_context = await performance_monitor.start_inference_measurement(
            model_id=model_id,
            batch_size=1,
        )
        await performance_monitor.end_inference_measurement(
            context=recent_context,
            success=True,
        )

        # Test filtering with time range
        time_range = TimeRange(
            start_time=now - timedelta(minutes=30),
            end_time=now + timedelta(minutes=30),
        )

        summary = await performance_monitor.get_performance_summary(model_id, time_range)

        # Should only include the recent metric
        assert summary.total_inferences == 1

    @pytest.mark.asyncio
    async def test_detect_performance_issues_high_memory(self, performance_monitor):
        """Test detection of high memory usage issues."""
        model_id = "memory_heavy_model"

        # Set low memory threshold for testing
        performance_monitor.thresholds["max_memory_usage_mb"] = 1000.0

        # Manually create high memory metrics
        for i in range(10):
            metric = InferenceMetric(
                model_id=model_id,
                request_id=f"req_{i}",
                timestamp=datetime.now(),
                inference_time_ms=100.0,
                memory_usage_mb=850.0,  # Above 80% of 1000MB threshold
                peak_memory_mb=850.0,
                batch_size=1,
                input_length=100,
                output_length=50,
                success=True,
                cpu_utilization=25.0,
            )
            performance_monitor.metrics[model_id].append(metric)

        issues = await performance_monitor.detect_performance_issues(model_id)

        # Should detect high memory usage issue
        memory_issues = [
            i for i in issues if i.issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE.value
        ]
        assert len(memory_issues) > 0
        assert memory_issues[0].severity == "high"

    @pytest.mark.asyncio
    async def test_detect_performance_issues_slow_inference(self, performance_monitor):
        """Test detection of slow inference issues."""
        model_id = "slow_model"

        # Set low time threshold for testing
        performance_monitor.thresholds["max_inference_time_ms"] = 1000.0

        # Generate metrics with slow inference (simulate by manipulating timestamps)
        for _ in range(10):
            context = await performance_monitor.start_inference_measurement(
                model_id=model_id,
                batch_size=1,
            )

            # Manually create slow metric
            slow_metric = InferenceMetric(
                model_id=model_id,
                request_id=context.request_id,
                timestamp=datetime.now(),
                inference_time_ms=800.0,  # Above 50% of threshold
                memory_usage_mb=500.0,
                peak_memory_mb=500.0,
                batch_size=1,
                input_length=100,
                output_length=50,
                success=True,
                cpu_utilization=50.0,
            )
            performance_monitor.metrics[model_id].append(slow_metric)

        issues = await performance_monitor.detect_performance_issues(model_id)

        # Should detect slow inference issue
        slow_issues = [
            i for i in issues if i.issue_type == PerformanceIssueType.SLOW_INFERENCE.value
        ]
        assert len(slow_issues) > 0

    @pytest.mark.asyncio
    async def test_detect_performance_issues_high_error_rate(self, performance_monitor):
        """Test detection of high error rate issues."""
        model_id = "error_prone_model"

        # Lower the error rate threshold for testing
        performance_monitor.thresholds["max_error_rate"] = 0.1  # 10%

        # Generate metrics with high error rate (20% failures)
        for i in range(20):
            success = i % 5 != 0  # 20% failure rate
            metric = InferenceMetric(
                model_id=model_id,
                request_id=f"req_{i}",
                timestamp=datetime.now(),
                inference_time_ms=100.0,
                memory_usage_mb=500.0,
                peak_memory_mb=500.0,
                batch_size=1,
                input_length=100,
                output_length=50,
                success=success,
                error_message=None if success else "Test error",
                cpu_utilization=25.0,
            )
            performance_monitor.metrics[model_id].append(metric)

        issues = await performance_monitor.detect_performance_issues(model_id)

        # Should detect high error rate issue
        error_issues = [
            i for i in issues if i.issue_type == PerformanceIssueType.HIGH_ERROR_RATE.value
        ]
        assert len(error_issues) > 0
        assert error_issues[0].severity in ["high", "critical"]

    @pytest.mark.asyncio
    async def test_detect_memory_leak(self, performance_monitor):
        """Test detection of memory leak issues."""
        model_id = "leaky_model"

        # Generate metrics with increasing memory usage
        base_memory = 500.0
        for i in range(50):
            # Create context manually to control memory values
            context = InferenceContext(
                model_id=model_id,
                batch_size=1,
                input_length=100,
                start_memory=base_memory + i * 10,  # Increasing memory
                initial_cpu=25.0,
            )

            metric = InferenceMetric(
                model_id=model_id,
                request_id=context.request_id,
                timestamp=datetime.now(),
                inference_time_ms=100.0,
                memory_usage_mb=base_memory + i * 10,  # Steadily increasing
                peak_memory_mb=base_memory + i * 10,
                batch_size=1,
                input_length=100,
                output_length=50,
                success=True,
                cpu_utilization=25.0,
            )
            performance_monitor.metrics[model_id].append(metric)

        issues = await performance_monitor.detect_performance_issues(model_id)

        # Should detect memory leak
        leak_issues = [i for i in issues if i.issue_type == PerformanceIssueType.MEMORY_LEAK.value]
        assert len(leak_issues) > 0
        assert leak_issues[0].severity == "high"

    @pytest.mark.asyncio
    async def test_performance_trend_analysis(self, performance_monitor):
        """Test performance trend analysis."""
        model_id = "trend_model"

        # Generate metrics with degrading performance
        base_time = 100.0
        for i in range(20):
            context = await performance_monitor.start_inference_measurement(
                model_id=model_id,
                batch_size=1,
            )

            # Create metric with increasing inference time (degrading performance)
            metric = InferenceMetric(
                model_id=model_id,
                request_id=context.request_id,
                timestamp=datetime.now(),
                inference_time_ms=base_time + i * 10,  # Getting slower
                memory_usage_mb=500.0,
                peak_memory_mb=500.0,
                batch_size=1,
                input_length=100,
                output_length=50,
                success=True,
                cpu_utilization=25.0,
            )
            performance_monitor.metrics[model_id].append(metric)

        summary = await performance_monitor.get_performance_summary(model_id)

        # Should detect degrading trend
        assert summary.performance_trend == "degrading"
        assert summary.trend_confidence > 0

    @pytest.mark.asyncio
    async def test_resource_summary(self, performance_monitor):
        """Test resource usage summary."""
        # Add some metrics first
        context = await performance_monitor.start_inference_measurement(
            model_id="test_model",
            batch_size=1,
        )
        await performance_monitor.end_inference_measurement(
            context=context,
            success=True,
        )

        resource_summary = await performance_monitor.get_resource_summary()

        assert "system" in resource_summary
        assert "monitoring" in resource_summary
        assert "thresholds" in resource_summary

        monitoring_info = resource_summary["monitoring"]
        assert monitoring_info["total_metrics_collected"] > 0
        assert monitoring_info["active_models"] > 0
        assert "test_model" in monitoring_info["models_tracked"]

    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, performance_monitor):
        """Test cleanup of old metrics."""
        model_id = "cleanup_test_model"

        # Generate metrics with different ages
        now = datetime.now()

        # Add old metrics
        for i in range(5):
            old_metric = InferenceMetric(
                model_id=model_id,
                request_id=f"old_{i}",
                timestamp=now - timedelta(days=10),  # 10 days old
                inference_time_ms=100.0,
                memory_usage_mb=500.0,
                peak_memory_mb=500.0,
                batch_size=1,
                input_length=100,
                output_length=50,
                success=True,
                cpu_utilization=25.0,
            )
            performance_monitor.metrics[model_id].append(old_metric)

        # Add recent metrics
        for i in range(3):
            recent_metric = InferenceMetric(
                model_id=model_id,
                request_id=f"recent_{i}",
                timestamp=now - timedelta(hours=1),  # 1 hour old
                inference_time_ms=100.0,
                memory_usage_mb=500.0,
                peak_memory_mb=500.0,
                batch_size=1,
                input_length=100,
                output_length=50,
                success=True,
                cpu_utilization=25.0,
            )
            performance_monitor.metrics[model_id].append(recent_metric)

        # Should have 8 total metrics
        assert len(performance_monitor.metrics[model_id]) == 8

        # Clean up metrics older than 7 days
        cleaned_count = await performance_monitor.cleanup_old_metrics(days=7)

        # Should have cleaned 5 old metrics
        assert cleaned_count == 5
        assert len(performance_monitor.metrics[model_id]) == 3

        # Remaining metrics should be recent ones
        for metric in performance_monitor.metrics[model_id]:
            assert "recent" in metric.request_id


class TestPerformanceIssue:
    """Test cases for PerformanceIssue model."""

    def test_performance_issue_creation(self):
        """Test PerformanceIssue creation."""
        issue = PerformanceIssue(
            issue_type=PerformanceIssueType.HIGH_MEMORY_USAGE,
            severity="high",
            description="High memory usage detected",
            recommendation="Reduce batch size",
            affected_models=["model1"],
            metrics={"memory_mb": 2048},
        )

        assert issue.issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE.value
        assert issue.severity == "high"
        assert issue.description == "High memory usage detected"
        assert issue.affected_models == ["model1"]
        assert issue.metrics["memory_mb"] == 2048


class TestTimeRange:
    """Test cases for TimeRange model."""

    def test_time_range_creation(self):
        """Test TimeRange creation."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        time_range = TimeRange(start_time=start_time, end_time=end_time)

        assert time_range.start_time == start_time
        assert time_range.end_time == end_time


if __name__ == "__main__":
    pytest.main([__file__])

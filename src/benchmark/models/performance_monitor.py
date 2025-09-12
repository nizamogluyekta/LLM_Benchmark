"""
Comprehensive performance monitoring system for model inference.

This module provides detailed tracking of inference metrics, resource usage, and model efficiency
with support for detecting performance degradation and generating optimization recommendations.
"""

import platform
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import psutil
from pydantic import BaseModel, Field

from benchmark.core.logging import get_logger


class PerformanceIssueType(Enum):
    """Types of performance issues that can be detected."""

    HIGH_MEMORY_USAGE = "high_memory_usage"
    SLOW_INFERENCE = "slow_inference"
    LOW_THROUGHPUT = "low_throughput"
    MEMORY_LEAK = "memory_leak"
    DEGRADING_PERFORMANCE = "degrading_performance"
    HIGH_ERROR_RATE = "high_error_rate"
    RESOURCE_CONTENTION = "resource_contention"


class TimeRange(BaseModel):
    """Time range for filtering performance data."""

    start_time: datetime
    end_time: datetime

    model_config = {"use_enum_values": True}


class InferenceMetric(BaseModel):
    """Detailed metrics for a single inference operation."""

    model_id: str = Field(..., description="Model identifier")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Inference timestamp")
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Total inference time in milliseconds"
    )
    time_to_first_token_ms: float | None = Field(
        None, ge=0.0, description="Time to first token (streaming)"
    )
    tokens_per_second: float | None = Field(None, ge=0.0, description="Token generation rate")
    memory_usage_mb: float = Field(..., ge=0.0, description="Memory usage during inference")
    peak_memory_mb: float = Field(..., ge=0.0, description="Peak memory usage")
    batch_size: int = Field(..., ge=1, description="Batch size for inference")
    input_length: int = Field(..., ge=0, description="Input sequence length")
    output_length: int = Field(..., ge=0, description="Output sequence length")
    success: bool = Field(..., description="Whether inference was successful")
    error_message: str | None = Field(None, description="Error message if inference failed")
    gpu_utilization: float | None = Field(
        None, ge=0.0, le=100.0, description="GPU utilization percentage"
    )
    neural_engine_usage: float | None = Field(
        None, ge=0.0, le=100.0, description="Neural Engine usage (M4 Pro)"
    )
    cpu_utilization: float = Field(..., ge=0.0, le=100.0, description="CPU utilization percentage")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"use_enum_values": True}


class PerformanceIssue(BaseModel):
    """Represents a detected performance issue."""

    issue_type: PerformanceIssueType = Field(..., description="Type of performance issue")
    severity: str = Field(..., description="Issue severity: low, medium, high, critical")
    description: str = Field(..., description="Human-readable description of the issue")
    recommendation: str = Field(..., description="Recommended action to resolve the issue")
    affected_models: list[str] = Field(
        default_factory=list, description="Models affected by this issue"
    )
    metrics: dict[str, Any] = Field(default_factory=dict, description="Supporting metrics")
    first_detected: datetime = Field(
        default_factory=datetime.now, description="When issue was first detected"
    )

    model_config = {"use_enum_values": True}


class PerformanceSummary(BaseModel):
    """Performance summary for a model or time period."""

    model_id: str = Field(..., description="Model identifier")
    time_range: TimeRange = Field(..., description="Time range for summary")
    total_inferences: int = Field(..., ge=0, description="Total number of inferences")
    successful_inferences: int = Field(..., ge=0, description="Number of successful inferences")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate percentage")

    # Timing metrics
    avg_inference_time_ms: float = Field(..., ge=0.0, description="Average inference time")
    p95_inference_time_ms: float = Field(..., ge=0.0, description="95th percentile inference time")
    p99_inference_time_ms: float = Field(..., ge=0.0, description="99th percentile inference time")
    avg_tokens_per_second: float | None = Field(
        None, ge=0.0, description="Average tokens per second"
    )

    # Memory metrics
    avg_memory_usage_mb: float = Field(..., ge=0.0, description="Average memory usage")
    peak_memory_usage_mb: float = Field(..., ge=0.0, description="Peak memory usage")
    memory_efficiency: float = Field(..., ge=0.0, le=1.0, description="Memory efficiency score")

    # Resource utilization
    avg_cpu_utilization: float = Field(..., ge=0.0, le=100.0, description="Average CPU utilization")
    avg_gpu_utilization: float | None = Field(
        None, ge=0.0, le=100.0, description="Average GPU utilization"
    )
    avg_neural_engine_usage: float | None = Field(
        None, ge=0.0, le=100.0, description="Average Neural Engine usage"
    )

    # Throughput metrics
    requests_per_second: float = Field(..., ge=0.0, description="Average requests per second")
    total_throughput_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall throughput score"
    )

    # Batch size analysis
    batch_size_distribution: dict[int, int] = Field(
        default_factory=dict, description="Distribution of batch sizes"
    )
    optimal_batch_size: int | None = Field(None, description="Recommended optimal batch size")

    # Performance trends
    performance_trend: str = Field(..., description="improving, stable, or degrading")
    trend_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in trend analysis")

    model_config = {"use_enum_values": True}


class InferenceContext(BaseModel):
    """Context for tracking an ongoing inference operation."""

    model_id: str = Field(..., description="Model identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique request identifier"
    )
    start_time: datetime = Field(default_factory=datetime.now, description="Start timestamp")
    start_memory: float = Field(..., description="Memory usage at start")
    batch_size: int = Field(..., ge=1, description="Batch size")
    input_length: int = Field(..., ge=0, description="Input length")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    # Resource tracking
    initial_cpu: float = Field(..., description="CPU usage at start")
    initial_gpu: float | None = Field(None, description="GPU usage at start")

    model_config = {"use_enum_values": True, "arbitrary_types_allowed": True}


class ResourceTracker:
    """Track system resource usage during inference."""

    def __init__(self) -> None:
        self.logger = get_logger("resource_tracker")
        self._system_info = self._get_system_info()

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for resource tracking."""
        try:
            return {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "is_m4_pro": self._detect_m4_pro(),
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {}

    def _detect_m4_pro(self) -> bool:
        """Detect if running on M4 Pro (Apple Silicon)."""
        try:
            return platform.system() == "Darwin" and "arm64" in platform.machine().lower()
        except Exception:
            return False

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return float(process.memory_info().rss / 1024 / 1024)
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def get_cpu_utilization(self) -> float:
        """Get current CPU utilization percentage."""
        try:
            return float(psutil.cpu_percent(interval=0.1))
        except Exception as e:
            self.logger.warning(f"Failed to get CPU utilization: {e}")
            return 0.0

    def get_gpu_utilization(self) -> float | None:
        """Get GPU utilization percentage (M4 Pro GPU)."""
        if not self._system_info.get("is_m4_pro", False):
            return None

        try:
            # On M4 Pro, we can use system_profiler or ioreg for GPU stats
            # For now, we'll simulate this as it requires special system access
            # In production, this would use Metal Performance Shaders or IOKit
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse JSON output to get GPU usage
                # This is a simplified version - real implementation would parse GPU metrics
                return 0.0  # Placeholder
        except Exception as e:
            self.logger.debug(f"GPU utilization not available: {e}")

        return None

    def get_neural_engine_usage(self) -> float | None:
        """Get Neural Engine usage percentage (M4 Pro Neural Engine)."""
        if not self._system_info.get("is_m4_pro", False):
            return None

        try:
            # Neural Engine usage would require CoreML metrics or private APIs
            # This is a placeholder for the actual implementation
            # In production, this would use CoreML performance counters
            return None  # Not yet implemented
        except Exception as e:
            self.logger.debug(f"Neural Engine usage not available: {e}")
            return None

    def get_system_metrics(self) -> dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "timestamp": datetime.now(),
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "cpu_percent": self.get_cpu_utilization(),
                "disk_usage_percent": disk.percent,
                "gpu_utilization": self.get_gpu_utilization(),
                "neural_engine_usage": self.get_neural_engine_usage(),
                "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system metrics: {e}")
            return {"timestamp": datetime.now(), "error": str(e)}


class ModelPerformanceMonitor:
    """Comprehensive performance monitoring for model inference."""

    def __init__(self, max_metrics_per_model: int = 10000):
        self.metrics: dict[str, list[InferenceMetric]] = defaultdict(list)
        self.resource_tracker = ResourceTracker()
        self.max_metrics_per_model = max_metrics_per_model
        self.logger = get_logger("performance_monitor")

        # Performance baselines and thresholds
        self.performance_baselines: dict[str, dict[str, float]] = {}
        self.thresholds = {
            "max_inference_time_ms": 30000,  # 30 seconds
            "max_memory_usage_mb": 8192,  # 8GB
            "min_success_rate": 0.95,  # 95%
            "max_error_rate": 0.05,  # 5%
            "memory_leak_threshold": 0.1,  # 10% memory growth
        }

    async def start_inference_measurement(
        self,
        model_id: str,
        batch_size: int = 1,
        input_length: int = 0,
        request_id: str | None = None,
    ) -> InferenceContext:
        """Start measuring inference performance."""
        try:
            context = InferenceContext(
                model_id=model_id,
                request_id=request_id or str(uuid4()),
                start_memory=self.resource_tracker.get_memory_usage(),
                batch_size=batch_size,
                input_length=input_length,
                initial_cpu=self.resource_tracker.get_cpu_utilization(),
                initial_gpu=self.resource_tracker.get_gpu_utilization(),
            )

            self.logger.debug(
                f"Started inference measurement for model {model_id}, request {context.request_id}"
            )
            return context

        except Exception as e:
            self.logger.error(f"Failed to start inference measurement: {e}")
            raise

    async def end_inference_measurement(
        self,
        context: InferenceContext,
        success: bool = True,
        output_length: int = 0,
        error_message: str | None = None,
        tokens_generated: int | None = None,
        time_to_first_token_ms: float | None = None,
    ) -> InferenceMetric:
        """End measurement and record metrics."""
        try:
            end_time = datetime.now()
            inference_time_ms = (end_time - context.start_time).total_seconds() * 1000

            current_memory = self.resource_tracker.get_memory_usage()
            peak_memory = max(context.start_memory, current_memory)

            # Calculate tokens per second if token info is available
            tokens_per_second = None
            if tokens_generated and inference_time_ms > 0:
                tokens_per_second = (tokens_generated / inference_time_ms) * 1000

            metric = InferenceMetric(
                model_id=context.model_id,
                request_id=context.request_id,
                timestamp=end_time,
                inference_time_ms=inference_time_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                tokens_per_second=tokens_per_second,
                memory_usage_mb=current_memory,
                peak_memory_mb=peak_memory,
                batch_size=context.batch_size,
                input_length=context.input_length,
                output_length=output_length,
                success=success,
                error_message=error_message,
                gpu_utilization=self.resource_tracker.get_gpu_utilization(),
                neural_engine_usage=self.resource_tracker.get_neural_engine_usage(),
                cpu_utilization=self.resource_tracker.get_cpu_utilization(),
                metadata=context.metadata.copy(),
            )

            # Store metric
            await self._store_metric(metric)

            # Update baselines
            await self._update_baselines(context.model_id, metric)

            self.logger.debug(
                f"Recorded inference metric for {context.model_id}: "
                f"{inference_time_ms:.2f}ms, {current_memory:.2f}MB"
            )

            return metric

        except Exception as e:
            self.logger.error(f"Failed to end inference measurement: {e}")
            raise

    async def _store_metric(self, metric: InferenceMetric) -> None:
        """Store metric with size limits."""
        model_metrics = self.metrics[metric.model_id]
        model_metrics.append(metric)

        # Trim old metrics if we exceed the limit
        if len(model_metrics) > self.max_metrics_per_model:
            # Keep the most recent metrics
            self.metrics[metric.model_id] = model_metrics[-self.max_metrics_per_model :]
            self.logger.debug(f"Trimmed metrics for model {metric.model_id}")

    async def _update_baselines(self, model_id: str, metric: InferenceMetric) -> None:
        """Update performance baselines for trend detection."""
        if model_id not in self.performance_baselines:
            self.performance_baselines[model_id] = {}

        baselines = self.performance_baselines[model_id]

        # Use exponential moving average for baselines
        alpha = 0.1  # Smoothing factor

        if "avg_inference_time" not in baselines:
            baselines["avg_inference_time"] = metric.inference_time_ms
        else:
            baselines["avg_inference_time"] = (
                alpha * metric.inference_time_ms + (1 - alpha) * baselines["avg_inference_time"]
            )

        if "avg_memory_usage" not in baselines:
            baselines["avg_memory_usage"] = metric.memory_usage_mb
        else:
            baselines["avg_memory_usage"] = (
                alpha * metric.memory_usage_mb + (1 - alpha) * baselines["avg_memory_usage"]
            )

    async def get_performance_summary(
        self, model_id: str, time_range: TimeRange | None = None
    ) -> PerformanceSummary:
        """Get performance summary for model."""
        try:
            if model_id not in self.metrics:
                # Return empty summary
                now = datetime.now()
                default_range = TimeRange(start_time=now, end_time=now)

                return PerformanceSummary(
                    model_id=model_id,
                    time_range=time_range or default_range,
                    total_inferences=0,
                    successful_inferences=0,
                    error_rate=0.0,
                    avg_inference_time_ms=0.0,
                    p95_inference_time_ms=0.0,
                    p99_inference_time_ms=0.0,
                    avg_tokens_per_second=None,
                    avg_memory_usage_mb=0.0,
                    peak_memory_usage_mb=0.0,
                    memory_efficiency=0.0,
                    avg_cpu_utilization=0.0,
                    avg_gpu_utilization=None,
                    avg_neural_engine_usage=None,
                    requests_per_second=0.0,
                    total_throughput_score=0.0,
                    optimal_batch_size=None,
                    performance_trend="stable",
                    trend_confidence=0.0,
                )

            # Filter metrics by time range
            all_metrics = self.metrics[model_id]
            if time_range:
                filtered_metrics = [
                    m
                    for m in all_metrics
                    if time_range.start_time <= m.timestamp <= time_range.end_time
                ]
            else:
                filtered_metrics = all_metrics
                # Default to last 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                time_range = TimeRange(start_time=cutoff_time, end_time=datetime.now())

            if not filtered_metrics:
                return PerformanceSummary(
                    model_id=model_id,
                    time_range=time_range,
                    total_inferences=0,
                    successful_inferences=0,
                    error_rate=0.0,
                    avg_inference_time_ms=0.0,
                    p95_inference_time_ms=0.0,
                    p99_inference_time_ms=0.0,
                    avg_tokens_per_second=None,
                    avg_memory_usage_mb=0.0,
                    peak_memory_usage_mb=0.0,
                    memory_efficiency=0.0,
                    avg_cpu_utilization=0.0,
                    avg_gpu_utilization=None,
                    avg_neural_engine_usage=None,
                    requests_per_second=0.0,
                    total_throughput_score=0.0,
                    optimal_batch_size=None,
                    performance_trend="stable",
                    trend_confidence=0.0,
                )

            # Calculate basic metrics
            total_inferences = len(filtered_metrics)
            successful_inferences = sum(1 for m in filtered_metrics if m.success)
            error_rate = 1.0 - (successful_inferences / total_inferences)

            # Timing metrics
            inference_times = [m.inference_time_ms for m in filtered_metrics if m.success]
            avg_inference_time = (
                sum(inference_times) / len(inference_times) if inference_times else 0.0
            )

            inference_times.sort()
            p95_time = inference_times[int(0.95 * len(inference_times))] if inference_times else 0.0
            p99_time = inference_times[int(0.99 * len(inference_times))] if inference_times else 0.0

            # Token metrics
            token_rates = [
                m.tokens_per_second for m in filtered_metrics if m.tokens_per_second is not None
            ]
            avg_tokens_per_second = sum(token_rates) / len(token_rates) if token_rates else None

            # Memory metrics
            memory_usages = [m.memory_usage_mb for m in filtered_metrics]
            avg_memory = sum(memory_usages) / len(memory_usages)
            peak_memory = max(m.peak_memory_mb for m in filtered_metrics)

            # Memory efficiency (simplified calculation)
            memory_efficiency = (
                1.0 / (1.0 + (peak_memory - avg_memory) / avg_memory) if avg_memory > 0 else 0.0
            )

            # Resource utilization
            cpu_utils = [m.cpu_utilization for m in filtered_metrics]
            avg_cpu = sum(cpu_utils) / len(cpu_utils)

            gpu_utils = [
                m.gpu_utilization for m in filtered_metrics if m.gpu_utilization is not None
            ]
            avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None

            ne_usages = [
                m.neural_engine_usage for m in filtered_metrics if m.neural_engine_usage is not None
            ]
            avg_ne = sum(ne_usages) / len(ne_usages) if ne_usages else None

            # Throughput metrics
            time_span_hours = (time_range.end_time - time_range.start_time).total_seconds() / 3600
            requests_per_second = (
                successful_inferences / (time_span_hours * 3600) if time_span_hours > 0 else 0.0
            )

            # Throughput score (simplified)
            if avg_inference_time > 0:
                throughput_score = min(1.0, 1000 / avg_inference_time)  # Based on inference speed
            else:
                throughput_score = 0.0

            # Batch size analysis
            batch_sizes = [m.batch_size for m in filtered_metrics]
            batch_distribution = {}
            for bs in set(batch_sizes):
                batch_distribution[bs] = batch_sizes.count(bs)

            # Find optimal batch size (simplified heuristic)
            batch_performance = {}
            for bs in set(batch_sizes):
                bs_metrics = [m for m in filtered_metrics if m.batch_size == bs]
                if bs_metrics:
                    avg_time_per_item = sum(
                        m.inference_time_ms / m.batch_size for m in bs_metrics
                    ) / len(bs_metrics)
                    batch_performance[bs] = avg_time_per_item

            if batch_performance:
                optimal_batch_size = min(
                    batch_performance.keys(), key=lambda k: batch_performance[k]
                )
            else:
                optimal_batch_size = None

            # Performance trend analysis
            trend, confidence = await self._analyze_performance_trend(filtered_metrics)

            return PerformanceSummary(
                model_id=model_id,
                time_range=time_range,
                total_inferences=total_inferences,
                successful_inferences=successful_inferences,
                error_rate=error_rate,
                avg_inference_time_ms=avg_inference_time,
                p95_inference_time_ms=p95_time,
                p99_inference_time_ms=p99_time,
                avg_tokens_per_second=avg_tokens_per_second,
                avg_memory_usage_mb=avg_memory,
                peak_memory_usage_mb=peak_memory,
                memory_efficiency=memory_efficiency,
                avg_cpu_utilization=avg_cpu,
                avg_gpu_utilization=avg_gpu,
                avg_neural_engine_usage=avg_ne,
                requests_per_second=requests_per_second,
                total_throughput_score=throughput_score,
                batch_size_distribution=batch_distribution,
                optimal_batch_size=optimal_batch_size,
                performance_trend=trend,
                trend_confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"Failed to generate performance summary: {e}")
            raise

    async def _analyze_performance_trend(self, metrics: list[InferenceMetric]) -> tuple[str, float]:
        """Analyze performance trend over time."""
        if len(metrics) < 10:
            return "stable", 0.0

        try:
            # Sort by timestamp
            sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

            # Split into first and second half
            mid_point = len(sorted_metrics) // 2
            first_half = sorted_metrics[:mid_point]
            second_half = sorted_metrics[mid_point:]

            # Calculate average performance for each half
            first_avg = sum(m.inference_time_ms for m in first_half) / len(first_half)
            second_avg = sum(m.inference_time_ms for m in second_half) / len(second_half)

            # Determine trend
            if first_avg == 0:
                return "stable", 0.0

            change_ratio = (second_avg - first_avg) / first_avg
            confidence = min(1.0, len(metrics) / 100.0)  # More data = higher confidence

            if change_ratio > 0.1:  # 10% slower
                return "degrading", confidence
            elif change_ratio < -0.1:  # 10% faster
                return "improving", confidence
            else:
                return "stable", confidence

        except Exception as e:
            self.logger.warning(f"Failed to analyze performance trend: {e}")
            return "stable", 0.0

    async def detect_performance_issues(self, model_id: str) -> list[PerformanceIssue]:
        """Detect potential performance problems."""
        issues: list[PerformanceIssue] = []

        try:
            if model_id not in self.metrics or not self.metrics[model_id]:
                return issues

            recent_metrics = self.metrics[model_id][-100:]  # Last 100 metrics

            # High memory usage
            avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            if avg_memory > self.thresholds["max_memory_usage_mb"] * 0.8:  # 80% of max
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.HIGH_MEMORY_USAGE,
                        severity="high",
                        description=f"Model {model_id} is using {avg_memory:.1f}MB of memory on average",
                        recommendation="Consider reducing batch size or model optimization",
                        affected_models=[model_id],
                        metrics={
                            "avg_memory_mb": avg_memory,
                            "threshold": self.thresholds["max_memory_usage_mb"],
                        },
                    )
                )

            # Slow inference
            avg_time = sum(m.inference_time_ms for m in recent_metrics if m.success) / sum(
                1 for m in recent_metrics if m.success
            )
            if avg_time > self.thresholds["max_inference_time_ms"] * 0.5:  # 50% of max
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.SLOW_INFERENCE,
                        severity="medium"
                        if avg_time < self.thresholds["max_inference_time_ms"]
                        else "high",
                        description=f"Model {model_id} has slow inference time: {avg_time:.1f}ms on average",
                        recommendation="Consider model optimization, caching, or hardware acceleration",
                        affected_models=[model_id],
                        metrics={"avg_inference_time_ms": avg_time},
                    )
                )

            # High error rate
            error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
            if error_rate > self.thresholds["max_error_rate"]:
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.HIGH_ERROR_RATE,
                        severity="critical" if error_rate > 0.2 else "high",
                        description=f"Model {model_id} has high error rate: {error_rate * 100:.1f}%",
                        recommendation="Investigate error causes and improve error handling",
                        affected_models=[model_id],
                        metrics={"error_rate": error_rate},
                    )
                )

            # Memory leak detection
            if len(recent_metrics) >= 50:
                first_half = recent_metrics[:25]
                second_half = recent_metrics[-25:]

                first_avg_memory = sum(m.memory_usage_mb for m in first_half) / len(first_half)
                second_avg_memory = sum(m.memory_usage_mb for m in second_half) / len(second_half)

                if first_avg_memory > 0:
                    memory_growth = (second_avg_memory - first_avg_memory) / first_avg_memory
                    if memory_growth > self.thresholds["memory_leak_threshold"]:
                        issues.append(
                            PerformanceIssue(
                                issue_type=PerformanceIssueType.MEMORY_LEAK,
                                severity="high",
                                description=f"Model {model_id} shows potential memory leak: {memory_growth * 100:.1f}% growth",
                                recommendation="Review memory management and cleanup procedures",
                                affected_models=[model_id],
                                metrics={"memory_growth_rate": memory_growth},
                            )
                        )

            # Performance degradation
            summary = await self.get_performance_summary(model_id)
            if summary.performance_trend == "degrading" and summary.trend_confidence > 0.5:
                issues.append(
                    PerformanceIssue(
                        issue_type=PerformanceIssueType.DEGRADING_PERFORMANCE,
                        severity="medium",
                        description=f"Model {model_id} shows degrading performance trend",
                        recommendation="Monitor resource usage and consider system maintenance",
                        affected_models=[model_id],
                        metrics={"trend_confidence": summary.trend_confidence},
                    )
                )

        except Exception as e:
            self.logger.error(f"Failed to detect performance issues for {model_id}: {e}")

        return issues

    async def get_resource_summary(self) -> dict[str, Any]:
        """Get current resource usage summary."""
        try:
            system_metrics = self.resource_tracker.get_system_metrics()

            # Add aggregated model metrics
            total_metrics = sum(len(metrics) for metrics in self.metrics.values())
            active_models = len([model_id for model_id, metrics in self.metrics.items() if metrics])

            return {
                "system": system_metrics,
                "monitoring": {
                    "total_metrics_collected": total_metrics,
                    "active_models": active_models,
                    "models_tracked": list(self.metrics.keys()),
                },
                "thresholds": self.thresholds.copy(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get resource summary: {e}")
            return {"error": str(e)}

    async def cleanup_old_metrics(self, days: int = 7) -> int:
        """Clean up metrics older than specified days."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            cleaned_count = 0

            for model_id in list(self.metrics.keys()):
                original_count = len(self.metrics[model_id])
                self.metrics[model_id] = [
                    m for m in self.metrics[model_id] if m.timestamp >= cutoff_time
                ]
                cleaned_count += original_count - len(self.metrics[model_id])

                # Remove models with no metrics
                if not self.metrics[model_id]:
                    del self.metrics[model_id]

            self.logger.info(f"Cleaned up {cleaned_count} old metrics (older than {days} days)")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old metrics: {e}")
            return 0

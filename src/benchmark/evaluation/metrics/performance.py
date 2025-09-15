"""
Performance evaluation for inference speed and resource usage.

This module provides a PerformanceEvaluator that analyzes inference speed,
throughput, resource efficiency metrics, and performance consistency for
machine learning model evaluation.
"""

import statistics
from dataclasses import dataclass
from typing import Any

from benchmark.evaluation.base_evaluator import BaseEvaluator
from benchmark.interfaces.evaluation_interfaces import MetricType


@dataclass
class PerformanceStats:
    """Container for performance statistics."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p95: float
    p99: float
    count: int


class PerformanceEvaluator(BaseEvaluator):
    """Evaluate model inference performance and efficiency."""

    def __init__(self) -> None:
        """Initialize the performance evaluator."""
        super().__init__(MetricType.PERFORMANCE)
        self.metric_names = [
            "avg_inference_time_ms",
            "median_inference_time_ms",
            "p95_inference_time_ms",
            "p99_inference_time_ms",
            "min_inference_time_ms",
            "max_inference_time_ms",
            "inference_time_std_ms",
            "throughput_samples_per_sec",
            "total_inference_time_sec",
            "avg_tokens_per_sec",
            "performance_consistency_score",
            "outlier_percentage",
        ]

    async def evaluate(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Calculate performance metrics from prediction timing data.

        Args:
            predictions: List of prediction dictionaries with timing information
            ground_truth: List of ground truth dictionaries (not used for performance)

        Returns:
            Dictionary mapping metric names to values

        Raises:
            ValueError: If no inference timing data found
        """
        # Validate input data
        self.validate_input_data(predictions, ground_truth)

        # Extract timing information
        inference_times = self._extract_inference_times(predictions)

        if not inference_times:
            raise ValueError("No inference timing data found in predictions")

        # Calculate core performance metrics
        metrics: dict[str, float] = {}
        metrics.update(self._calculate_latency_metrics(inference_times))
        metrics.update(self._calculate_throughput_metrics(inference_times))
        metrics.update(self._calculate_consistency_metrics(inference_times))

        # Calculate token-based metrics if available
        token_metrics = self._calculate_token_metrics(predictions, inference_times)
        if token_metrics:
            metrics.update(token_metrics)

        return metrics

    def _extract_inference_times(self, predictions: list[dict[str, Any]]) -> list[float]:
        """
        Extract inference times from predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            List of inference times in milliseconds
        """
        inference_times = []

        for pred in predictions:
            # Look for timing information in various possible fields
            time_ms = pred.get("inference_time_ms")
            if time_ms is not None and time_ms > 0:
                inference_times.append(float(time_ms))
                continue

            # Try alternative field names
            time_sec = pred.get("inference_time_sec")
            if time_sec is not None and time_sec > 0:
                inference_times.append(float(time_sec * 1000))  # Convert to ms
                continue

            # Try processing_time_ms
            proc_time = pred.get("processing_time_ms")
            if proc_time is not None and proc_time > 0:
                inference_times.append(float(proc_time))
                continue

            # Try latency_ms
            latency = pred.get("latency_ms")
            if latency is not None and latency > 0:
                inference_times.append(float(latency))

        return inference_times

    def _calculate_latency_metrics(self, inference_times: list[float]) -> dict[str, float]:
        """
        Calculate latency statistics.

        Args:
            inference_times: List of inference times in milliseconds

        Returns:
            Dictionary with latency metrics
        """
        if not inference_times:
            return {}

        stats = self._calculate_performance_stats(inference_times)

        return {
            "avg_inference_time_ms": stats.mean,
            "median_inference_time_ms": stats.median,
            "min_inference_time_ms": stats.min,
            "max_inference_time_ms": stats.max,
            "p95_inference_time_ms": stats.p95,
            "p99_inference_time_ms": stats.p99,
            "inference_time_std_ms": stats.std,
            "total_inference_time_sec": sum(inference_times) / 1000.0,
        }

    def _calculate_throughput_metrics(self, inference_times: list[float]) -> dict[str, float]:
        """
        Calculate throughput metrics.

        Args:
            inference_times: List of inference times in milliseconds

        Returns:
            Dictionary with throughput metrics
        """
        if not inference_times:
            return {}

        # Calculate samples per second
        total_time_sec = sum(inference_times) / 1000.0  # Convert ms to seconds
        total_samples = len(inference_times)

        throughput = total_samples / total_time_sec if total_time_sec > 0 else 0.0

        return {"throughput_samples_per_sec": throughput}

    def _calculate_consistency_metrics(self, inference_times: list[float]) -> dict[str, float]:
        """
        Calculate performance consistency metrics.

        Args:
            inference_times: List of inference times in milliseconds

        Returns:
            Dictionary with consistency metrics
        """
        if len(inference_times) < 2:
            return {"performance_consistency_score": 1.0, "outlier_percentage": 0.0}

        stats = self._calculate_performance_stats(inference_times)

        # Consistency score: higher is better (less variance)
        # Use coefficient of variation (std/mean) inverted
        cv = stats.std / stats.mean if stats.mean > 0 else float("inf")
        consistency_score = 1.0 / (1.0 + cv)  # Scale to 0-1, higher is more consistent

        # Calculate outlier percentage (values > 2 std deviations from mean)
        outlier_threshold = stats.mean + 2 * stats.std
        outliers = [t for t in inference_times if t > outlier_threshold]
        outlier_percentage = len(outliers) / len(inference_times) * 100

        return {
            "performance_consistency_score": consistency_score,
            "outlier_percentage": outlier_percentage,
        }

    def _calculate_token_metrics(
        self, predictions: list[dict[str, Any]], inference_times: list[float]
    ) -> dict[str, float] | None:
        """
        Calculate token-based performance metrics if token information available.

        Args:
            predictions: List of prediction dictionaries
            inference_times: List of inference times in milliseconds

        Returns:
            Dictionary with token metrics or None if no token data
        """
        valid_pairs = []

        for i, pred in enumerate(predictions):
            if i < len(inference_times):
                # Look for token count information
                tokens = None
                for field in ["output_tokens", "tokens_generated", "response_tokens", "tokens"]:
                    if field in pred:
                        tokens = pred[field]
                        break

                if tokens is not None and tokens > 0 and inference_times[i] > 0:
                    valid_pairs.append((tokens, inference_times[i]))

        if not valid_pairs:
            return None

        # Calculate tokens per second for each sample
        tokens_per_sec = []
        for tokens, time_ms in valid_pairs:
            time_sec = time_ms / 1000.0
            if time_sec > 0:
                tps = tokens / time_sec
                tokens_per_sec.append(tps)

        if tokens_per_sec:
            avg_tokens_per_sec = statistics.mean(tokens_per_sec)
            return {"avg_tokens_per_sec": avg_tokens_per_sec}

        return None

    def _calculate_performance_stats(self, values: list[float]) -> PerformanceStats:
        """
        Calculate comprehensive statistics for performance values.

        Args:
            values: List of performance values

        Returns:
            PerformanceStats object with calculated statistics
        """
        if not values:
            return PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0)

        sorted_values = sorted(values)
        count = len(values)

        # Calculate percentiles safely
        p95_idx = max(0, min(count - 1, int(0.95 * count)))
        p99_idx = max(0, min(count - 1, int(0.99 * count)))

        return PerformanceStats(
            mean=statistics.mean(values),
            median=statistics.median(values),
            std=statistics.stdev(values) if count > 1 else 0.0,
            min=min(values),
            max=max(values),
            p95=sorted_values[p95_idx],
            p99=sorted_values[p99_idx],
            count=count,
        )

    def analyze_performance_trends(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze performance trends over time.

        Args:
            predictions: List of prediction dictionaries with timestamps

        Returns:
            Dictionary with trend analysis results
        """
        # Extract timestamps and inference times
        time_series = []
        for pred in predictions:
            timestamp = pred.get("timestamp")
            inference_time = pred.get("inference_time_ms")

            if timestamp and inference_time:
                time_series.append((timestamp, inference_time))

        if len(time_series) < 10:  # Need minimum data for trend analysis
            return {"message": "Insufficient data for trend analysis"}

        # Sort by timestamp
        time_series.sort(key=lambda x: x[0])

        # Calculate moving average (window of 10)
        window_size = min(10, len(time_series) // 4)
        moving_averages = []

        for i in range(len(time_series) - window_size + 1):
            window_values = [t[1] for t in time_series[i : i + window_size]]
            moving_averages.append(statistics.mean(window_values))

        # Detect performance degradation
        if len(moving_averages) >= 2:
            first_half_avg = statistics.mean(moving_averages[: len(moving_averages) // 2])
            second_half_avg = statistics.mean(moving_averages[len(moving_averages) // 2 :])

            degradation_percentage = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        else:
            degradation_percentage = 0.0

        return {
            "performance_degradation_percentage": degradation_percentage,
            "trend_analysis_samples": len(time_series),
            "moving_average_window": window_size,
            "performance_trend": (
                "degrading"
                if degradation_percentage > 5
                else "improving"
                if degradation_percentage < -5
                else "stable"
            ),
        }

    def get_metric_names(self) -> list[str]:
        """Get list of metrics this evaluator produces."""
        return self.metric_names.copy()

    def get_required_prediction_fields(self) -> list[str]:
        """Get required fields in prediction data."""
        return []  # Performance evaluator checks for timing fields dynamically

    def get_required_ground_truth_fields(self) -> list[str]:
        """Get required fields in ground truth data."""
        return []  # Performance evaluation doesn't need ground truth

    def generate_performance_report(self, predictions: list[dict[str, Any]]) -> str:
        """
        Generate detailed performance report.

        Args:
            predictions: List of prediction dictionaries with timing data

        Returns:
            Formatted performance report as string
        """
        inference_times = self._extract_inference_times(predictions)

        if not inference_times:
            return "No performance data available"

        stats = self._calculate_performance_stats(inference_times)

        # Calculate additional metrics
        total_time_sec = sum(inference_times) / 1000.0
        throughput = len(inference_times) / total_time_sec if total_time_sec > 0 else 0

        report = f"""
Performance Analysis Report
==========================

Latency Metrics:
- Average: {stats.mean:.2f} ms
- Median: {stats.median:.2f} ms
- 95th Percentile: {stats.p95:.2f} ms
- 99th Percentile: {stats.p99:.2f} ms
- Min: {stats.min:.2f} ms
- Max: {stats.max:.2f} ms
- Standard Deviation: {stats.std:.2f} ms

Throughput Metrics:
- Samples per Second: {throughput:.2f}
- Total Samples: {stats.count}
- Total Time: {total_time_sec:.2f} seconds

Consistency Analysis:
- Coefficient of Variation: {(stats.std / stats.mean * 100):.1f}%
- Performance Range: {stats.max - stats.min:.2f} ms

Recommendations:
"""

        # Add recommendations based on performance
        if stats.mean > 0 and stats.std / stats.mean > 0.3:  # High variance
            report += "- High variance detected. Consider optimizing for consistency.\n"

        if throughput < 10:  # Low throughput
            report += "- Low throughput detected. Consider batch optimization.\n"

        if stats.p95 > 2 * stats.mean:  # Long tail latency
            report += "- Long tail latency detected. Check for outliers and optimize worst cases.\n"

        if stats.mean > 1000:  # Slow inference (> 1 second)
            report += "- Slow inference times detected. Consider model optimization or hardware acceleration.\n"

        if stats.std < 0.1 * stats.mean:  # Very consistent
            report += "- Excellent performance consistency detected.\n"

        return report

    def validate_data_compatibility(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> bool:
        """
        Validate that prediction data contains timing information.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries (not used)

        Returns:
            True if data is compatible, False otherwise
        """
        if not predictions:
            return False

        # Check if at least some predictions have timing data
        timing_fields = [
            "inference_time_ms",
            "inference_time_sec",
            "processing_time_ms",
            "latency_ms",
        ]

        valid_count = 0
        for pred in predictions:
            if any(
                field in pred and pred[field] is not None and pred[field] > 0
                for field in timing_fields
            ):
                valid_count += 1

        # Require at least 50% of predictions to have timing data
        return valid_count >= len(predictions) * 0.5

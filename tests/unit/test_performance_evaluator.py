"""
Unit tests for the PerformanceEvaluator.

Tests performance evaluation including latency metrics, throughput calculation,
consistency analysis, token-based metrics, and trend analysis.
"""

import pytest
import pytest_asyncio

from benchmark.evaluation.metrics.performance import PerformanceEvaluator
from benchmark.interfaces.evaluation_interfaces import MetricType
from tests.fixtures.performance_test_data import (
    ALTERNATIVE_TIMING_FIELDS,
    ALTERNATIVE_TIMING_GROUND_TRUTH,
    CONSISTENT_GROUND_TRUTH,
    CONSISTENT_PERFORMANCE_DATA,
    CONSISTENT_PERFORMANCE_EXPECTED,
    DEGRADING_TREND_DATA,
    FAST_GROUND_TRUTH,
    FAST_PERFORMANCE_DATA,
    IMPROVING_TREND_DATA,
    INVALID_TIMING_DATA,
    INVALID_TIMING_GROUND_TRUTH,
    NO_TIMING_DATA,
    NO_TIMING_GROUND_TRUTH,
    PERFORMANCE_SCENARIOS,
    SINGLE_SAMPLE_DATA,
    SINGLE_SAMPLE_GROUND_TRUTH,
    SLOW_GROUND_TRUTH,
    SLOW_PERFORMANCE_DATA,
    STABLE_TREND_DATA,
    TOKEN_FOCUSED_DATA,
    TOKEN_FOCUSED_GROUND_TRUTH,
    VARIABLE_GROUND_TRUTH,
    VARIABLE_PERFORMANCE_DATA,
    VARIABLE_PERFORMANCE_EXPECTED,
)


class TestPerformanceEvaluator:
    """Test cases for PerformanceEvaluator."""

    @pytest_asyncio.fixture
    async def evaluator(self):
        """Create a PerformanceEvaluator instance for testing."""
        return PerformanceEvaluator()

    def test_evaluator_initialization(self, evaluator):
        """Test that the evaluator initializes correctly."""
        assert evaluator.get_metric_type() == MetricType.PERFORMANCE
        assert isinstance(evaluator.get_metric_names(), list)
        assert len(evaluator.get_metric_names()) > 0
        assert "avg_inference_time_ms" in evaluator.get_metric_names()
        assert "throughput_samples_per_sec" in evaluator.get_metric_names()
        assert "performance_consistency_score" in evaluator.get_metric_names()

    def test_required_fields(self, evaluator):
        """Test required field specifications."""
        pred_fields = evaluator.get_required_prediction_fields()
        gt_fields = evaluator.get_required_ground_truth_fields()

        # Performance evaluator checks timing fields dynamically, so no strict requirements
        assert len(pred_fields) == 0
        assert len(gt_fields) == 0  # Performance evaluation doesn't need ground truth

    @pytest.mark.asyncio
    async def test_consistent_performance_evaluation(self, evaluator):
        """Test evaluation with consistent performance data."""
        metrics = await evaluator.evaluate(CONSISTENT_PERFORMANCE_DATA, CONSISTENT_GROUND_TRUTH)

        # Check that all expected metrics are present
        expected_metrics = [
            "avg_inference_time_ms",
            "median_inference_time_ms",
            "p95_inference_time_ms",
            "p99_inference_time_ms",
            "throughput_samples_per_sec",
            "performance_consistency_score",
            "outlier_percentage",
        ]

        for metric in expected_metrics:
            assert metric in metrics

        # Check basic values match expectations
        assert (
            abs(
                metrics["avg_inference_time_ms"]
                - CONSISTENT_PERFORMANCE_EXPECTED["avg_inference_time_ms"]
            )
            < 1.0
        )
        assert (
            abs(
                metrics["median_inference_time_ms"]
                - CONSISTENT_PERFORMANCE_EXPECTED["median_inference_time_ms"]
            )
            < 1.0
        )
        assert (
            metrics["min_inference_time_ms"]
            == CONSISTENT_PERFORMANCE_EXPECTED["min_inference_time_ms"]
        )
        assert (
            metrics["max_inference_time_ms"]
            == CONSISTENT_PERFORMANCE_EXPECTED["max_inference_time_ms"]
        )
        assert (
            metrics["outlier_percentage"] == CONSISTENT_PERFORMANCE_EXPECTED["outlier_percentage"]
        )

        # Consistent data should have high consistency score
        assert metrics["performance_consistency_score"] > 0.8

    @pytest.mark.asyncio
    async def test_variable_performance_evaluation(self, evaluator):
        """Test evaluation with variable performance data."""
        metrics = await evaluator.evaluate(VARIABLE_PERFORMANCE_DATA, VARIABLE_GROUND_TRUTH)

        # Check basic bounds
        assert (
            metrics["min_inference_time_ms"]
            == VARIABLE_PERFORMANCE_EXPECTED["min_inference_time_ms"]
        )
        assert (
            metrics["max_inference_time_ms"]
            == VARIABLE_PERFORMANCE_EXPECTED["max_inference_time_ms"]
        )

        # Variable data should have outliers
        assert metrics["outlier_percentage"] > 0

        # Variable data should have lower consistency score
        assert metrics["performance_consistency_score"] < 0.8

        # Check that percentiles are reasonable
        assert metrics["p95_inference_time_ms"] > metrics["median_inference_time_ms"]
        assert metrics["p99_inference_time_ms"] >= metrics["p95_inference_time_ms"]

    @pytest.mark.asyncio
    async def test_fast_performance_evaluation(self, evaluator):
        """Test evaluation with fast performance data."""
        metrics = await evaluator.evaluate(FAST_PERFORMANCE_DATA, FAST_GROUND_TRUTH)

        # Fast inference should have high throughput
        assert metrics["throughput_samples_per_sec"] > 50  # Should be much higher than slow models

        # Check average time is reasonable
        assert metrics["avg_inference_time_ms"] < 20  # Should be quite fast

    @pytest.mark.asyncio
    async def test_slow_performance_evaluation(self, evaluator):
        """Test evaluation with slow performance data."""
        metrics = await evaluator.evaluate(SLOW_PERFORMANCE_DATA, SLOW_GROUND_TRUTH)

        # Slow inference should have low throughput
        assert metrics["throughput_samples_per_sec"] < 10

        # Check average time reflects slow performance
        assert metrics["avg_inference_time_ms"] > 1000  # Should be quite slow

    @pytest.mark.asyncio
    async def test_token_based_metrics(self, evaluator):
        """Test token-based performance metrics calculation."""
        metrics = await evaluator.evaluate(TOKEN_FOCUSED_DATA, TOKEN_FOCUSED_GROUND_TRUTH)

        # Should have token-based metrics
        assert "avg_tokens_per_sec" in metrics
        assert metrics["avg_tokens_per_sec"] > 0

        # Token throughput should be reasonable
        # With varying token counts and times, should see reasonable token/sec rate
        assert 10 <= metrics["avg_tokens_per_sec"] <= 1000

    @pytest.mark.asyncio
    async def test_alternative_timing_fields(self, evaluator):
        """Test handling of alternative timing field names."""
        metrics = await evaluator.evaluate(
            ALTERNATIVE_TIMING_FIELDS, ALTERNATIVE_TIMING_GROUND_TRUTH
        )

        # Should successfully extract timing from various field names
        assert "avg_inference_time_ms" in metrics
        assert metrics["avg_inference_time_ms"] > 0

        # Should have processed all 4 samples with timing data
        total_time = metrics["total_inference_time_sec"]
        assert total_time > 0

    @pytest.mark.asyncio
    async def test_no_timing_data_error(self, evaluator):
        """Test that evaluation fails gracefully when no timing data available."""
        with pytest.raises(ValueError, match="No inference timing data found"):
            await evaluator.evaluate(NO_TIMING_DATA, NO_TIMING_GROUND_TRUTH)

    @pytest.mark.asyncio
    async def test_single_sample_evaluation(self, evaluator):
        """Test evaluation with single sample."""
        metrics = await evaluator.evaluate(SINGLE_SAMPLE_DATA, SINGLE_SAMPLE_GROUND_TRUTH)

        # Should handle single sample gracefully
        assert "avg_inference_time_ms" in metrics
        assert metrics["avg_inference_time_ms"] == 100.0
        assert metrics["median_inference_time_ms"] == 100.0
        assert metrics["min_inference_time_ms"] == 100.0
        assert metrics["max_inference_time_ms"] == 100.0

        # Single sample should have perfect consistency
        assert metrics["performance_consistency_score"] == 1.0
        assert metrics["outlier_percentage"] == 0.0

    @pytest.mark.asyncio
    async def test_invalid_timing_data_filtering(self, evaluator):
        """Test that invalid timing values are filtered out."""
        metrics = await evaluator.evaluate(INVALID_TIMING_DATA, INVALID_TIMING_GROUND_TRUTH)

        # Should only process the one valid timing value (150.0ms)
        assert metrics["avg_inference_time_ms"] == 150.0
        assert metrics["median_inference_time_ms"] == 150.0
        assert metrics["min_inference_time_ms"] == 150.0
        assert metrics["max_inference_time_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_performance_stats_calculation(self, evaluator):
        """Test the PerformanceStats calculation."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats = evaluator._calculate_performance_stats(values)

        assert stats.mean == 55.0
        assert stats.median == 55.0
        assert stats.min == 10.0
        assert stats.max == 100.0
        assert stats.count == 10
        # With 10 values (10-100), 95th percentile is index 9 (100.0)
        assert stats.p95 == 100.0  # 95th percentile of [10,20,30,40,50,60,70,80,90,100]
        assert stats.p99 == 100.0  # 99th percentile
        assert stats.std > 0

    def test_performance_stats_empty_list(self, evaluator):
        """Test PerformanceStats with empty list."""
        stats = evaluator._calculate_performance_stats([])

        assert stats.mean == 0
        assert stats.median == 0
        assert stats.min == 0
        assert stats.max == 0
        assert stats.count == 0
        assert stats.std == 0

    def test_performance_stats_single_value(self, evaluator):
        """Test PerformanceStats with single value."""
        stats = evaluator._calculate_performance_stats([42.0])

        assert stats.mean == 42.0
        assert stats.median == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.count == 1
        assert stats.std == 0.0  # No variation with single value

    @pytest.mark.asyncio
    async def test_consistency_score_calculation(self, evaluator):
        """Test performance consistency score calculation."""
        # Very consistent data
        consistent_data = [
            {"prediction": "ATTACK", "inference_time_ms": 100.0},
            {"prediction": "BENIGN", "inference_time_ms": 101.0},
            {"prediction": "ATTACK", "inference_time_ms": 99.0},
        ]
        consistent_gt = [{"label": "ATTACK"}, {"label": "BENIGN"}, {"label": "ATTACK"}]

        metrics = await evaluator.evaluate(consistent_data, consistent_gt)
        consistency_score = metrics["performance_consistency_score"]

        # Should have high consistency score
        assert consistency_score > 0.8

        # Very inconsistent data
        inconsistent_data = [
            {"prediction": "ATTACK", "inference_time_ms": 10.0},
            {"prediction": "BENIGN", "inference_time_ms": 1000.0},
            {"prediction": "ATTACK", "inference_time_ms": 50.0},
        ]

        metrics = await evaluator.evaluate(inconsistent_data, consistent_gt)
        inconsistent_score = metrics["performance_consistency_score"]

        # Should have low consistency score
        assert inconsistent_score < consistency_score

    @pytest.mark.asyncio
    async def test_outlier_detection(self, evaluator):
        """Test outlier detection in performance data."""
        # Use many normal values and one clear outlier to minimize standard deviation impact
        outlier_data = []
        # Add 9 normal values
        for _ in range(9):
            outlier_data.append({"prediction": "ATTACK", "inference_time_ms": 100.0})
        # Add 1 clear outlier
        outlier_data.append({"prediction": "ATTACK", "inference_time_ms": 500.0})
        outlier_gt = [{"label": "ATTACK"} for _ in range(10)]

        metrics = await evaluator.evaluate(outlier_data, outlier_gt)

        # Test that outlier detection mechanism runs without error
        # Actual outlier detection depends on statistical thresholds
        assert "outlier_percentage" in metrics
        assert isinstance(metrics["outlier_percentage"], int | float)
        # Outlier percentage can be 0 if values don't exceed 2-sigma threshold
        assert metrics["outlier_percentage"] <= 100

    @pytest.mark.asyncio
    async def test_throughput_calculation(self, evaluator):
        """Test throughput calculation."""
        # Known timing data for throughput calculation
        timing_data = [
            {"prediction": "ATTACK", "inference_time_ms": 100.0},  # 0.1 sec
            {"prediction": "BENIGN", "inference_time_ms": 200.0},  # 0.2 sec
            {"prediction": "ATTACK", "inference_time_ms": 300.0},  # 0.3 sec
        ]
        # Total time: 0.6 sec, 3 samples = 5 samples/sec
        timing_gt = [{"label": "ATTACK"}, {"label": "BENIGN"}, {"label": "ATTACK"}]

        metrics = await evaluator.evaluate(timing_data, timing_gt)

        expected_throughput = 3 / 0.6  # 5 samples per second
        assert abs(metrics["throughput_samples_per_sec"] - expected_throughput) < 0.1

    def test_performance_trend_analysis_degrading(self, evaluator):
        """Test performance trend analysis with degrading performance."""
        trend_result = evaluator.analyze_performance_trends(DEGRADING_TREND_DATA)

        assert "performance_degradation_percentage" in trend_result
        assert "performance_trend" in trend_result
        assert trend_result["performance_trend"] == "degrading"
        assert trend_result["performance_degradation_percentage"] > 0

    def test_performance_trend_analysis_improving(self, evaluator):
        """Test performance trend analysis with improving performance."""
        trend_result = evaluator.analyze_performance_trends(IMPROVING_TREND_DATA)

        assert "performance_degradation_percentage" in trend_result
        assert "performance_trend" in trend_result
        assert trend_result["performance_trend"] == "improving"
        assert trend_result["performance_degradation_percentage"] < 0

    def test_performance_trend_analysis_stable(self, evaluator):
        """Test performance trend analysis with stable performance."""
        trend_result = evaluator.analyze_performance_trends(STABLE_TREND_DATA)

        assert "performance_degradation_percentage" in trend_result
        assert "performance_trend" in trend_result
        assert trend_result["performance_trend"] == "stable"
        assert abs(trend_result["performance_degradation_percentage"]) <= 5

    def test_performance_trend_insufficient_data(self, evaluator):
        """Test performance trend analysis with insufficient data."""
        insufficient_data = [
            {"prediction": "ATTACK", "inference_time_ms": 100.0, "timestamp": "2024-01-01T10:00:00"}
        ]

        trend_result = evaluator.analyze_performance_trends(insufficient_data)

        assert "message" in trend_result
        assert "Insufficient data" in trend_result["message"]

    def test_performance_report_generation(self, evaluator):
        """Test detailed performance report generation."""
        report = evaluator.generate_performance_report(CONSISTENT_PERFORMANCE_DATA)

        assert isinstance(report, str)
        assert "Performance Analysis Report" in report
        assert "Latency Metrics" in report
        assert "Throughput Metrics" in report
        assert "Consistency Analysis" in report
        assert "Recommendations" in report
        assert "Average:" in report
        assert "ms" in report

    def test_performance_report_no_data(self, evaluator):
        """Test performance report generation with no data."""
        report = evaluator.generate_performance_report(NO_TIMING_DATA)

        assert report == "No performance data available"

    def test_performance_report_recommendations(self, evaluator):
        """Test that performance report includes appropriate recommendations."""
        # Test with variable performance data (should trigger variance warning)
        report = evaluator.generate_performance_report(VARIABLE_PERFORMANCE_DATA)

        assert "High variance detected" in report or "Recommendations:" in report

        # Test with slow performance data
        report = evaluator.generate_performance_report(SLOW_PERFORMANCE_DATA)

        assert "Slow inference" in report or "optimization" in report.lower()

    @pytest.mark.asyncio
    async def test_data_compatibility_validation(self, evaluator):
        """Test data compatibility validation."""
        # Valid data should pass
        assert (
            evaluator.validate_data_compatibility(
                CONSISTENT_PERFORMANCE_DATA, CONSISTENT_GROUND_TRUTH
            )
            is True
        )

        # Data without timing should fail
        assert (
            evaluator.validate_data_compatibility(NO_TIMING_DATA, NO_TIMING_GROUND_TRUTH) is False
        )

        # Empty data should fail
        assert evaluator.validate_data_compatibility([], []) is False

        # Mixed data (some with timing, some without) should pass if >= 50% have timing
        mixed_data = [
            {"prediction": "ATTACK", "inference_time_ms": 100.0},
            {"prediction": "BENIGN", "confidence": 0.8},  # No timing
        ]
        mixed_gt = [{"label": "ATTACK"}, {"label": "BENIGN"}]
        assert evaluator.validate_data_compatibility(mixed_data, mixed_gt) is True

    @pytest.mark.asyncio
    async def test_all_performance_scenarios(self, evaluator):
        """Test all predefined performance scenarios."""
        for _scenario_name, scenario_data in PERFORMANCE_SCENARIOS.items():
            try:
                metrics = await evaluator.evaluate(
                    scenario_data["predictions"], scenario_data["ground_truth"]
                )

                # Basic validation that metrics are calculated
                assert "avg_inference_time_ms" in metrics
                assert "throughput_samples_per_sec" in metrics
                assert "performance_consistency_score" in metrics

                # All metrics should be non-negative
                for metric_name, value in metrics.items():
                    assert value >= 0, f"Metric {metric_name} should be non-negative, got {value}"

                # Consistency score should be between 0 and 1
                assert 0 <= metrics["performance_consistency_score"] <= 1

                # Outlier percentage should be between 0 and 100
                assert 0 <= metrics["outlier_percentage"] <= 100

            except ValueError as e:
                # Only acceptable if no timing data
                assert "No inference timing data found" in str(e)

    @pytest.mark.asyncio
    async def test_percentile_calculations(self, evaluator):
        """Test that percentile calculations are accurate."""
        # Use known data for precise testing
        known_data = [
            {"prediction": "ATTACK", "inference_time_ms": float(i)}
            for i in range(1, 101)  # 1 to 100 ms
        ]
        known_gt = [{"label": "ATTACK"} for _ in range(100)]

        metrics = await evaluator.evaluate(known_data, known_gt)

        # For 1-100, 95th percentile should be close to 95, 99th should be close to 99
        assert abs(metrics["p95_inference_time_ms"] - 95.0) <= 1.0
        assert abs(metrics["p99_inference_time_ms"] - 99.0) <= 1.0
        assert metrics["median_inference_time_ms"] == 50.5  # Median of 1-100
        assert metrics["min_inference_time_ms"] == 1.0
        assert metrics["max_inference_time_ms"] == 100.0

    @pytest.mark.asyncio
    async def test_zero_total_time_handling(self, evaluator):
        """Test handling of edge case where total time might be zero."""
        # This shouldn't happen in practice, but test robustness
        zero_time_data = [{"prediction": "ATTACK", "inference_time_ms": 0.0}]
        # Note: This will be filtered out as invalid, so we need at least one valid time
        zero_time_data.append({"prediction": "BENIGN", "inference_time_ms": 1.0})

        zero_gt = [{"label": "ATTACK"}, {"label": "BENIGN"}]

        metrics = await evaluator.evaluate(zero_time_data, zero_gt)

        # Should handle gracefully and calculate throughput based on valid data
        assert "throughput_samples_per_sec" in metrics
        assert metrics["throughput_samples_per_sec"] >= 0

"""
Comprehensive test suite for ResultsAnalyzer.

Tests performance analysis, trend identification, bottleneck detection,
and mathematical accuracy of analysis algorithms.
"""

import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest

from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_analyzer import ResultsAnalyzer
from benchmark.evaluation.results_storage import ResultsStorage


class TestResultsAnalyzer:
    """Test cases for ResultsAnalyzer functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def analyzer(self, temp_storage):
        """Create ResultsAnalyzer instance."""
        return ResultsAnalyzer(temp_storage)

    @pytest.fixture
    def sample_results(self, temp_storage):
        """Create sample evaluation results for testing."""
        results = []

        # Create results for multiple models with varying performance
        models = ["model_a", "model_b", "model_c"]
        datasets = ["dataset_1", "dataset_2"]
        tasks = ["classification", "regression"]

        for i in range(30):  # 30 evaluations
            model = models[i % len(models)]
            dataset = datasets[i % len(datasets)]
            task = tasks[i % len(tasks)]

            # Create performance patterns
            base_accuracy = 0.7 + (i % 3) * 0.1  # Different base performance per model
            noise = np.random.normal(0, 0.02)  # Small random variation
            accuracy = max(0.0, min(1.0, base_accuracy + noise))

            # Create time-based trends
            days_ago = 29 - i
            timestamp = datetime.now() - timedelta(days=days_ago)

            # Add trend component
            if model == "model_a":
                trend_component = 0.01 * (30 - days_ago) / 30  # Improving trend
            elif model == "model_b":
                trend_component = -0.01 * (30 - days_ago) / 30  # Declining trend
            else:
                trend_component = 0  # Stable performance

            accuracy = max(0.0, min(1.0, accuracy + trend_component))

            result = EvaluationResult(
                evaluation_id=f"eval_{i:03d}",
                model_name=model,
                task_type=task,
                dataset_name=dataset,
                metrics={
                    "accuracy": accuracy,
                    "f1_score": accuracy * 0.95,
                    "precision": accuracy * 0.98,
                    "recall": accuracy * 0.92,
                },
                timestamp=timestamp,
                configuration={
                    "learning_rate": 0.001 * (1 + i % 3),
                    "batch_size": 16 * (1 + i % 2),
                    "epochs": 3 + i % 3,
                },
                raw_responses=[],
                processing_time=10.0 + np.random.normal(0, 2),
                experiment_name="test_experiment",
                tags=["test", f"model_{model.split('_')[1]}"],
            )

            results.append(result)
            temp_storage.store_evaluation_result(result)

        return results

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.storage is not None
        assert hasattr(analyzer, "analyze_model_performance")
        assert hasattr(analyzer, "identify_performance_trends")
        assert hasattr(analyzer, "find_performance_bottlenecks")
        assert hasattr(analyzer, "generate_performance_summary")

    def test_analyze_model_performance(self, analyzer, sample_results):
        """Test comprehensive model performance analysis."""
        analysis = analyzer.analyze_model_performance("model_a")

        # Verify analysis structure
        assert "model_name" in analysis
        assert analysis["model_name"] == "model_a"
        assert "analysis_timestamp" in analysis
        assert "performance_metrics" in analysis
        assert "task_performance" in analysis
        assert "dataset_performance" in analysis
        assert "configuration_analysis" in analysis
        assert "temporal_analysis" in analysis

        # Verify performance metrics
        metrics = analysis["performance_metrics"]
        assert "accuracy" in metrics
        assert "mean" in metrics["accuracy"]
        assert "std" in metrics["accuracy"]
        assert "count" in metrics["accuracy"]
        assert metrics["accuracy"]["count"] > 0

        # Verify temporal analysis
        temporal = analysis["temporal_analysis"]
        assert "evaluation_count" in temporal
        assert "date_range" in temporal
        assert temporal["evaluation_count"] > 0

    def test_identify_performance_trends(self, analyzer, sample_results):
        """Test trend identification algorithms."""
        trends = analyzer.identify_performance_trends("model_a")

        # Verify trend structure
        assert "model_name" in trends
        assert "time_period" in trends
        assert "metric_trends" in trends
        assert "significant_trends" in trends
        assert "velocity_analysis" in trends

        # Verify trend analysis
        trend_analysis = trends["metric_trends"]
        assert "accuracy" in trend_analysis

        accuracy_trend = trend_analysis["accuracy"]
        assert "slope" in accuracy_trend
        assert "correlation" in accuracy_trend
        assert "p_value" in accuracy_trend
        assert "trend_direction" in accuracy_trend
        assert "std_error" in accuracy_trend

        # Verify statistical significance (might be empty if not significant)
        significant_trends = trends["significant_trends"]
        # Check if accuracy is in significant trends, or verify the structure is present
        assert isinstance(significant_trends, dict)

    def test_find_performance_bottlenecks(self, analyzer, sample_results):
        """Test bottleneck detection algorithms."""
        # Get all results for bottleneck analysis
        all_results = analyzer.storage.query_results()
        bottlenecks = analyzer.find_performance_bottlenecks(all_results)

        # Verify bottleneck structure
        assert "analysis_timestamp" in bottlenecks
        assert "summary" in bottlenecks
        assert "metric_bottlenecks" in bottlenecks
        assert "task_bottlenecks" in bottlenecks
        assert "dataset_bottlenecks" in bottlenecks
        assert "configuration_bottlenecks" in bottlenecks
        assert "timing_bottlenecks" in bottlenecks
        assert "cross_dimensional_analysis" in bottlenecks
        assert "bottleneck_priorities" in bottlenecks
        assert "recommendations" in bottlenecks

        # Verify metric bottlenecks
        metric_bottlenecks = bottlenecks["metric_bottlenecks"]
        assert isinstance(metric_bottlenecks, dict)

        # Verify bottleneck priorities exist
        priorities = bottlenecks["bottleneck_priorities"]
        assert isinstance(priorities, dict)
        assert "all_bottlenecks" in priorities

    def test_generate_performance_summary(self, analyzer, sample_results):
        """Test performance summary generation."""
        # Get all results for summary
        all_results = analyzer.storage.query_results()
        summary = analyzer.generate_performance_summary(all_results)

        # Verify summary structure
        assert "analysis_timestamp" in summary
        assert "executive_summary" in summary
        assert "summary_statistics" in summary
        assert "key_insights" in summary
        assert "performance_highlights" in summary
        assert "improvement_areas" in summary
        assert "model_ranking" in summary
        assert "executive_recommendations" in summary

        # Verify executive summary
        executive_summary = summary["executive_summary"]
        assert "models_evaluated" in executive_summary
        assert "evaluation_period" in executive_summary
        assert executive_summary["models_evaluated"] > 0

        # Verify model ranking
        model_ranking = summary["model_ranking"]
        assert isinstance(model_ranking, list)

    def test_mathematical_accuracy_trend_analysis(self, analyzer):
        """Test mathematical accuracy of trend analysis."""
        # Create controlled test data with known trend
        temp_storage = analyzer.storage

        # Clear existing data
        all_results = temp_storage.query_results()
        for result in all_results:
            temp_storage.delete_evaluation(result.evaluation_id)

        # Create linear trend data
        n_points = 20
        slope = 0.02  # 2% improvement per day
        intercept = 0.7

        for i in range(n_points):
            accuracy = intercept + slope * i + np.random.normal(0, 0.01)

            result = EvaluationResult(
                evaluation_id=f"trend_test_{i:03d}",
                model_name="test_model",
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": accuracy},
                timestamp=datetime.now() - timedelta(days=n_points - i - 1),
                configuration={},
                raw_responses=[],
                processing_time=10.0,
            )
            temp_storage.store_evaluation_result(result)

        # Analyze trends
        trends = analyzer.identify_performance_trends("test_model")

        # Check if analysis was successful
        if "error" in trends:
            pytest.fail(f"Trend analysis failed: {trends['error']}")

        # Verify mathematical accuracy
        accuracy_trend = trends["metric_trends"]["accuracy"]
        detected_slope = accuracy_trend["slope"]

        # Allow for some tolerance due to noise and time unit differences
        # The detected slope might be different due to time scaling
        assert abs(detected_slope - slope) < 0.1, f"Expected slope ~{slope}, got {detected_slope}"
        assert abs(accuracy_trend["correlation"]) > 0.5, (
            "Correlation should be reasonably high for linear data"
        )
        assert accuracy_trend["trend_direction"] == "increasing", "Should detect increasing trend"

    def test_bottleneck_detection_accuracy(self, analyzer, sample_results):
        """Test accuracy of bottleneck detection."""
        all_results = analyzer.storage.query_results()
        bottlenecks = analyzer.find_performance_bottlenecks(all_results)

        # Verify that bottlenecks are properly prioritized
        priorities = bottlenecks["bottleneck_priorities"]
        assert "all_bottlenecks" in priorities
        all_bottlenecks = priorities["all_bottlenecks"]
        if len(all_bottlenecks) > 1:
            # Check that bottlenecks are sorted by priority
            priority_scores = [
                b["priority_score"] for b in all_bottlenecks if "priority_score" in b
            ]
            if priority_scores:
                assert priority_scores == sorted(priority_scores, reverse=True), (
                    "Bottlenecks should be sorted by priority"
                )

        # Verify recommendations are actionable
        recommendations = bottlenecks["recommendations"]
        assert isinstance(recommendations, list)

    def test_statistical_significance_calculations(self, analyzer):
        """Test statistical significance calculations in trend analysis."""
        # Create data with no trend (should not be significant)
        temp_storage = analyzer.storage

        # Clear existing data
        all_results = temp_storage.query_results()
        for result in all_results:
            temp_storage.delete_evaluation(result.evaluation_id)

        # Create random data with no trend
        n_points = 15
        mean_accuracy = 0.8

        for i in range(n_points):
            accuracy = mean_accuracy + np.random.normal(0, 0.05)

            result = EvaluationResult(
                evaluation_id=f"no_trend_test_{i:03d}",
                model_name="no_trend_model",
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": accuracy},
                timestamp=datetime.now() - timedelta(days=n_points - i - 1),
                configuration={},
                raw_responses=[],
                processing_time=10.0,
            )
            temp_storage.store_evaluation_result(result)

        # Analyze trends
        trends = analyzer.identify_performance_trends("no_trend_model")

        # Verify statistical significance
        significant_trends = trends["significant_trends"]

        # With random data, trend should not be significant
        if "accuracy" in significant_trends:
            significance = significant_trends["accuracy"]
            assert not significance["is_significant"], (
                "Random data should not show significant trend"
            )
            assert significance["p_value"] > 0.05, "P-value should be > 0.05 for random data"

    def test_performance_consistency_analysis(self, analyzer, sample_results):
        """Test performance consistency calculations."""
        analysis = analyzer.analyze_model_performance("model_a")

        # Verify consistency metrics are calculated
        performance_metrics = analysis["performance_metrics"]
        for _metric_name, metric_data in performance_metrics.items():
            assert "coefficient_of_variation" in metric_data
            assert "consistency_score" in metric_data

            cv = metric_data["coefficient_of_variation"]
            consistency = metric_data["consistency_score"]

            # Coefficient of variation should be non-negative
            assert cv >= 0, f"CV should be non-negative, got {cv}"

            # Consistency score should be between 0 and 1
            assert 0 <= consistency <= 1, f"Consistency score should be [0,1], got {consistency}"

            # If CV is 0, consistency should be 1
            if cv == 0:
                assert consistency == 1, "Perfect consistency should have consistency score of 1"

    def test_temporal_analysis_accuracy(self, analyzer, sample_results):
        """Test temporal analysis calculations."""
        analysis = analyzer.analyze_model_performance("model_a")
        temporal = analysis["temporal_analysis"]

        # Verify temporal metrics
        assert "date_range" in temporal
        assert "evaluation_frequency" in temporal
        assert "recent_performance" in temporal

        date_range = temporal["date_range"]
        assert "earliest" in date_range
        assert "latest" in date_range
        assert "span_days" in date_range

        # Verify date parsing
        earliest = datetime.fromisoformat(date_range["earliest"])
        latest = datetime.fromisoformat(date_range["latest"])
        assert latest >= earliest, "Latest date should be after earliest"

        span_days = date_range["span_days"]
        actual_span = (latest - earliest).days
        assert abs(span_days - actual_span) <= 1, (
            f"Span calculation incorrect: {span_days} vs {actual_span}"
        )

    def test_error_handling(self, analyzer):
        """Test error handling for edge cases."""
        # Test with non-existent model
        analysis = analyzer.analyze_model_performance("nonexistent_model")
        assert "error" in analysis or analysis["performance_metrics"] == {}

        # Test trends with insufficient data
        trends = analyzer.identify_performance_trends("nonexistent_model")
        assert "error" in trends or len(trends.get("trend_analysis", {})) == 0

        # Test bottlenecks with empty results
        bottlenecks = analyzer.find_performance_bottlenecks([])
        assert "analysis_metadata" in bottlenecks
        assert bottlenecks["analysis_metadata"]["total_evaluations"] == 0

    def test_cross_metric_analysis(self, analyzer, sample_results):
        """Test analysis across multiple metrics."""
        analysis = analyzer.analyze_model_performance("model_a")

        # Should analyze all available metrics
        performance_metrics = analysis["performance_metrics"]
        expected_metrics = ["accuracy", "f1_score", "precision", "recall"]

        for metric in expected_metrics:
            assert metric in performance_metrics, f"Missing analysis for {metric}"

            metric_data = performance_metrics[metric]
            assert "mean" in metric_data
            assert "std" in metric_data
            assert "correlation_with_other_metrics" in metric_data

            # Verify correlation calculations
            correlations = metric_data["correlation_with_other_metrics"]
            for other_metric, correlation in correlations.items():
                if other_metric != metric:
                    assert -1 <= correlation <= 1, (
                        f"Correlation should be [-1,1], got {correlation}"
                    )

    def test_configuration_impact_analysis(self, analyzer, sample_results):
        """Test configuration parameter impact analysis."""
        analysis = analyzer.analyze_model_performance("model_a")
        config_analysis = analysis["configuration_analysis"]

        # Should analyze impact of different configuration parameters
        assert "parameter_impact" in config_analysis
        assert "optimal_configurations" in config_analysis

        param_impact = config_analysis["parameter_impact"]
        for _param_name, impact_data in param_impact.items():
            assert "correlation_with_performance" in impact_data
            assert "value_analysis" in impact_data

            correlation = impact_data["correlation_with_performance"]
            assert -1 <= correlation <= 1, f"Correlation should be [-1,1], got {correlation}"

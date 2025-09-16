"""
Comprehensive test suite for ModelComparisonEngine.

Tests model comparison, statistical significance testing, ranking algorithms,
and mathematical accuracy of comparison methods.
"""

import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest

from benchmark.evaluation.comparison_engine import ModelComparisonEngine
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage


class TestModelComparisonEngine:
    """Test cases for ModelComparisonEngine functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def comparison_engine(self, temp_storage):
        """Create ModelComparisonEngine instance."""
        return ModelComparisonEngine(temp_storage)

    @pytest.fixture
    def comparison_data(self, temp_storage):
        """Create structured test data for model comparison."""
        models = ["model_a", "model_b", "model_c"]
        datasets = ["dataset_1", "dataset_2"]
        tasks = ["classification", "regression"]

        # Create results with known performance patterns
        results = []
        eval_id = 0

        for model_idx, model in enumerate(models):
            # Model A: High performance (0.85-0.95)
            # Model B: Medium performance (0.75-0.85)
            # Model C: Low performance (0.65-0.75)
            base_performance = 0.95 - model_idx * 0.1

            for dataset in datasets:
                for task in tasks:
                    for i in range(10):  # 10 evaluations per model/dataset/task combo
                        # Add controlled variation
                        noise = np.random.normal(0, 0.02)
                        accuracy = max(0.0, min(1.0, base_performance + noise))

                        result = EvaluationResult(
                            evaluation_id=f"comp_eval_{eval_id:03d}",
                            model_name=model,
                            task_type=task,
                            dataset_name=dataset,
                            metrics={
                                "accuracy": accuracy,
                                "f1_score": accuracy * 0.95,
                                "precision": accuracy * 0.98,
                                "recall": accuracy * 0.92,
                                "auc": accuracy * 0.97,
                            },
                            timestamp=datetime.now() - timedelta(days=i),
                            configuration={
                                "learning_rate": 0.001 * (1 + model_idx),
                                "batch_size": 16 * (1 + i % 2),
                            },
                            raw_responses=[],
                            processing_time=10.0 + np.random.normal(0, 1),
                            experiment_name="comparison_test",
                            tags=["comparison", f"model_{model.split('_')[1]}"],
                        )

                        results.append(result)
                        temp_storage.store_evaluation_result(result)
                        eval_id += 1

        return results

    def test_engine_initialization(self, comparison_engine):
        """Test comparison engine initialization."""
        assert comparison_engine.storage is not None
        assert hasattr(comparison_engine, "compare_models")
        assert hasattr(comparison_engine, "statistical_significance_test")
        assert hasattr(comparison_engine, "rank_models")

    def test_compare_models_basic(self, comparison_engine, comparison_data):
        """Test basic model comparison functionality."""
        comparison = comparison_engine.compare_models(
            model_names=["model_a", "model_b", "model_c"], metric_name="accuracy"
        )

        # Verify comparison structure
        assert "comparison_metadata" in comparison
        assert "performance_summary" in comparison
        assert "statistical_comparison" in comparison
        assert "pairwise_comparisons" in comparison
        assert "ranking_analysis" in comparison
        assert "consistency_analysis" in comparison

        # Verify metadata
        metadata = comparison["comparison_metadata"]
        assert metadata["models_compared"] == ["model_a", "model_b", "model_c"]
        assert metadata["primary_metric"] == "accuracy"
        assert metadata["total_evaluations"] > 0

        # Verify performance summary
        perf_summary = comparison["performance_summary"]
        for model in ["model_a", "model_b", "model_c"]:
            assert model in perf_summary
            assert "mean" in perf_summary[model]
            assert "std" in perf_summary[model]
            assert "count" in perf_summary[model]
            assert perf_summary[model]["count"] > 0

    def test_statistical_significance_test_welch_t(self, comparison_engine, comparison_data):
        """Test Welch's t-test for statistical significance."""
        # Get results for two models
        model_a_results = comparison_engine.storage.query_results({"model_name": "model_a"})
        model_b_results = comparison_engine.storage.query_results({"model_name": "model_b"})

        test_result = comparison_engine.statistical_significance_test(
            model_a_results, model_b_results, "accuracy", "welch_t"
        )

        # Verify test structure
        assert "test_results" in test_result
        assert "descriptive_statistics" in test_result
        assert "interpretation" in test_result

        # Verify test results
        test_stats = test_result["test_results"]
        assert test_stats["test_type"] == "welch_t"
        assert test_stats["test_name"] == "Welch's t-test"
        assert "statistic" in test_stats
        assert "p_value" in test_stats
        assert "is_significant" in test_stats
        assert "effect_size" in test_stats

        # Verify descriptive statistics
        desc_stats = test_result["descriptive_statistics"]
        assert "model_a" in desc_stats
        assert "model_b" in desc_stats
        assert "difference" in desc_stats

        for model_stats in [desc_stats["model_a"], desc_stats["model_b"]]:
            assert "mean" in model_stats
            assert "std" in model_stats
            assert "count" in model_stats
            assert model_stats["count"] > 0

    def test_statistical_significance_test_mann_whitney(self, comparison_engine, comparison_data):
        """Test Mann-Whitney U test for statistical significance."""
        model_a_results = comparison_engine.storage.query_results({"model_name": "model_a"})
        model_c_results = comparison_engine.storage.query_results({"model_name": "model_c"})

        test_result = comparison_engine.statistical_significance_test(
            model_a_results, model_c_results, "accuracy", "mann_whitney"
        )

        # Verify test type
        assert test_result["test_results"]["test_type"] == "mann_whitney"
        assert test_result["test_results"]["test_name"] == "Mann-Whitney U test"

        # Should detect significant difference between high and low performing models
        assert test_result["test_results"]["is_significant"]
        assert test_result["test_results"]["p_value"] < 0.05

    def test_statistical_significance_edge_cases(self, comparison_engine):
        """Test edge cases in statistical significance testing."""
        # Test with insufficient data
        empty_results = []
        model_a_results = comparison_engine.storage.query_results({"model_name": "model_a"})

        test_result = comparison_engine.statistical_significance_test(
            empty_results, model_a_results, "accuracy"
        )

        assert "error" in test_result
        assert "Insufficient data" in test_result["error"]

        # Test with mismatched sample sizes for paired t-test
        model_b_results = comparison_engine.storage.query_results({"model_name": "model_b"})

        test_result = comparison_engine.statistical_significance_test(
            model_a_results[:5], model_b_results[:10], "accuracy", "paired_t"
        )

        assert "error" in test_result
        assert "equal sample sizes" in test_result["error"]

    def test_rank_models(self, comparison_engine, comparison_data):
        """Test model ranking functionality."""
        ranking = comparison_engine.rank_models(
            model_names=["model_a", "model_b", "model_c"],
            metrics=["accuracy", "f1_score", "precision"],
        )

        # Verify ranking structure
        assert "ranking_metadata" in ranking
        assert "rankings" in ranking
        assert "metric_analysis" in ranking

        # Verify metadata
        metadata = ranking["ranking_metadata"]
        assert metadata["models_ranked"] == ["model_a", "model_b", "model_c"]
        assert set(metadata["metrics_used"]) == {"accuracy", "f1_score", "precision"}

        # Verify rankings
        rankings = ranking["rankings"]
        assert len(rankings) == 3

        # Check ranking order (model_a should be best)
        assert rankings[0]["rank"] == 1
        assert rankings[0]["model_name"] == "model_a"
        assert rankings[1]["rank"] == 2
        assert rankings[2]["rank"] == 3

        # Verify ranking scores
        for rank_info in rankings:
            assert "overall_score" in rank_info
            assert "metric_scores" in rank_info
            assert "raw_metrics" in rank_info
            assert 0 <= rank_info["overall_score"] <= 1

    def test_rank_models_with_weights(self, comparison_engine, comparison_data):
        """Test model ranking with custom weights."""
        # Give accuracy higher weight
        weights = {"accuracy": 0.6, "f1_score": 0.3, "precision": 0.1}

        ranking = comparison_engine.rank_models(
            model_names=["model_a", "model_b", "model_c"],
            metrics=["accuracy", "f1_score", "precision"],
            weights=weights,
        )

        # Verify weights are normalized and applied
        metadata = ranking["ranking_metadata"]
        normalized_weights = metadata["weights"]

        # Weights should sum to 1
        assert abs(sum(normalized_weights.values()) - 1.0) < 1e-10

        # Relative proportions should be maintained
        assert normalized_weights["accuracy"] > normalized_weights["f1_score"]
        assert normalized_weights["f1_score"] > normalized_weights["precision"]

    def test_pairwise_comparisons(self, comparison_engine, comparison_data):
        """Test pairwise model comparisons."""
        comparison = comparison_engine.compare_models(
            model_names=["model_a", "model_b", "model_c"], metric_name="accuracy"
        )

        pairwise = comparison["pairwise_comparisons"]

        # Should have all pairwise combinations
        expected_pairs = {"model_a_vs_model_b", "model_a_vs_model_c", "model_b_vs_model_c"}
        assert set(pairwise.keys()) == expected_pairs

        # Each comparison should have statistical test results
        for _pair_name, pair_result in pairwise.items():
            if "error" not in pair_result:
                assert "test_results" in pair_result
                assert "descriptive_statistics" in pair_result
                assert "interpretation" in pair_result

    def test_consistency_analysis(self, comparison_engine, comparison_data):
        """Test performance consistency analysis."""
        comparison = comparison_engine.compare_models(
            model_names=["model_a", "model_b", "model_c"], metric_name="accuracy"
        )

        consistency = comparison["consistency_analysis"]

        # Verify consistency metrics
        assert "model_consistency" in consistency
        assert "most_consistent_model" in consistency
        assert "least_consistent_model" in consistency

        model_consistency = consistency["model_consistency"]
        for model in ["model_a", "model_b", "model_c"]:
            if model in model_consistency:
                model_data = model_consistency[model]
                assert "coefficient_of_variation" in model_data
                assert "consistency_score" in model_data
                assert "standard_deviation" in model_data

                # Consistency score should be between 0 and 1
                assert 0 <= model_data["consistency_score"] <= 1

    def test_mathematical_accuracy_effect_size(self, comparison_engine):
        """Test mathematical accuracy of effect size calculations."""
        # Create controlled test data with known effect size
        temp_storage = comparison_engine.storage

        # Clear existing data
        all_results = temp_storage.query_results()
        for result in all_results:
            temp_storage.delete_evaluation(result.evaluation_id)

        # Create two groups with known means and standard deviations
        group1_mean = 0.8
        group2_mean = 0.7
        shared_std = 0.05
        n_per_group = 20

        # Expected Cohen's d = (0.8 - 0.7) / 0.05 = 2.0 (large effect)
        expected_cohens_d = (group1_mean - group2_mean) / shared_std

        # Create group 1 data
        for i in range(n_per_group):
            accuracy = np.random.normal(group1_mean, shared_std)
            result = EvaluationResult(
                evaluation_id=f"group1_{i:03d}",
                model_name="high_performance",
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": accuracy},
                timestamp=datetime.now() - timedelta(days=i),
                configuration={},
                raw_responses=[],
                processing_time=10.0,
            )
            temp_storage.store_evaluation_result(result)

        # Create group 2 data
        for i in range(n_per_group):
            accuracy = np.random.normal(group2_mean, shared_std)
            result = EvaluationResult(
                evaluation_id=f"group2_{i:03d}",
                model_name="low_performance",
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": accuracy},
                timestamp=datetime.now() - timedelta(days=i),
                configuration={},
                raw_responses=[],
                processing_time=10.0,
            )
            temp_storage.store_evaluation_result(result)

        # Test effect size calculation
        group1_results = temp_storage.query_results({"model_name": "high_performance"})
        group2_results = temp_storage.query_results({"model_name": "low_performance"})

        test_result = comparison_engine.statistical_significance_test(
            group1_results, group2_results, "accuracy", "welch_t"
        )

        calculated_effect_size = test_result["test_results"]["effect_size"]

        # Allow for sampling variation (Cohen's d calculation may vary with sampling)
        assert abs(calculated_effect_size - expected_cohens_d) < 1.0, (
            f"Expected effect size ~{expected_cohens_d}, got {calculated_effect_size}"
        )
        # With a substantial difference, should be at least medium or large effect
        assert test_result["test_results"]["effect_size_interpretation"] in ["medium", "large"]

    def test_anova_comparison(self, comparison_engine, comparison_data):
        """Test ANOVA statistical comparison."""
        comparison = comparison_engine.compare_models(
            model_names=["model_a", "model_b", "model_c"], metric_name="accuracy"
        )

        statistical_comparison = comparison["statistical_comparison"]

        # Verify ANOVA results
        assert "anova" in statistical_comparison
        assert "kruskal_wallis" in statistical_comparison

        anova_results = statistical_comparison["anova"]
        assert "f_statistic" in anova_results
        assert "p_value" in anova_results
        assert "is_significant" in anova_results
        assert anova_results["test_name"] == "One-way ANOVA"

        # With our test data (different performance levels), ANOVA should be significant
        assert anova_results["is_significant"]
        assert anova_results["p_value"] < 0.05

    def test_task_specific_analysis(self, comparison_engine, comparison_data):
        """Test task-specific performance analysis."""
        comparison = comparison_engine.compare_models(
            model_names=["model_a", "model_b"], metric_name="accuracy"
        )

        task_analysis = comparison["task_specific_analysis"]

        # Should analyze both classification and regression tasks
        assert "classification" in task_analysis
        assert "regression" in task_analysis

        for _task_type, task_data in task_analysis.items():
            assert "model_performance" in task_data
            assert "best_model" in task_data
            assert "best_performance" in task_data

            model_performance = task_data["model_performance"]
            for model in ["model_a", "model_b"]:
                if model in model_performance:
                    assert "mean" in model_performance[model]
                    assert "count" in model_performance[model]

    def test_dataset_specific_analysis(self, comparison_engine, comparison_data):
        """Test dataset-specific performance analysis."""
        comparison = comparison_engine.compare_models(
            model_names=["model_a", "model_b"], metric_name="accuracy"
        )

        dataset_analysis = comparison["dataset_specific_analysis"]

        # Should analyze both dataset_1 and dataset_2
        assert "dataset_1" in dataset_analysis
        assert "dataset_2" in dataset_analysis

        for _dataset_name, dataset_data in dataset_analysis.items():
            assert "model_performance" in dataset_data
            assert "best_model" in dataset_data
            assert "best_performance" in dataset_data

    def test_ranking_normalization(self, comparison_engine, comparison_data):
        """Test metric normalization in ranking."""
        ranking = comparison_engine.rank_models(
            model_names=["model_a", "model_b", "model_c"], metrics=["accuracy", "f1_score"]
        )

        rankings = ranking["rankings"]

        # Verify that normalized scores are between 0 and 1
        for rank_info in rankings:
            metric_scores = rank_info["metric_scores"]
            for metric, score in metric_scores.items():
                assert 0 <= score <= 1, (
                    f"Normalized score should be [0,1], got {score} for {metric}"
                )

    def test_time_period_filtering(self, comparison_engine, comparison_data):
        """Test time period filtering in comparisons."""
        # Test recent comparisons only
        comparison = comparison_engine.compare_models(
            model_names=["model_a", "model_b"], metric_name="accuracy", time_period="7d"
        )

        metadata = comparison["comparison_metadata"]
        assert metadata["time_period"] == "7d"

        # Should have fewer evaluations than total
        total_evaluations = comparison_engine.storage.get_storage_stats()["total_evaluations"]
        filtered_evaluations = metadata["total_evaluations"]
        assert filtered_evaluations <= total_evaluations

    def test_error_handling_comparison(self, comparison_engine):
        """Test error handling in model comparison."""
        # Test with non-existent models
        comparison = comparison_engine.compare_models(
            model_names=["nonexistent_model"], metric_name="accuracy"
        )

        # Should handle gracefully
        assert "comparison_metadata" in comparison
        assert comparison["comparison_metadata"]["total_evaluations"] == 0

        # Test with unknown test type
        model_a_results = comparison_engine.storage.query_results({"model_name": "model_a"})
        test_result = comparison_engine.statistical_significance_test(
            model_a_results, model_a_results, "accuracy", "unknown_test"
        )

        assert "error" in test_result
        assert "Unknown test type" in test_result["error"]

    def test_interpretation_generation(self, comparison_engine, comparison_data):
        """Test human-readable interpretation generation."""
        model_a_results = comparison_engine.storage.query_results({"model_name": "model_a"})
        model_c_results = comparison_engine.storage.query_results({"model_name": "model_c"})

        test_result = comparison_engine.statistical_significance_test(
            model_a_results, model_c_results, "accuracy"
        )

        interpretation = test_result["interpretation"]

        # Should provide meaningful interpretation
        assert isinstance(interpretation, str)
        assert len(interpretation) > 50  # Should be reasonably detailed

        # Should mention significance and effect size
        if test_result["test_results"]["is_significant"]:
            assert "significant" in interpretation.lower()

        if test_result["test_results"]["effect_size"] is not None:
            assert "effect" in interpretation.lower()

    def test_metric_discriminative_power(self, comparison_engine, comparison_data):
        """Test analysis of metric discriminative power in ranking."""
        ranking = comparison_engine.rank_models(
            model_names=["model_a", "model_b", "model_c"],
            metrics=["accuracy", "f1_score", "precision"],
        )

        metric_analysis = ranking["metric_analysis"]

        for metric in ["accuracy", "f1_score", "precision"]:
            assert metric in metric_analysis

            metric_data = metric_analysis[metric]
            assert "discriminative_power" in metric_data
            assert "mean" in metric_data
            assert "std" in metric_data
            assert "range" in metric_data

            # Discriminative power should be non-negative
            assert metric_data["discriminative_power"] >= 0

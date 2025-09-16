"""
Advanced model comparison engine for evaluation results.

This module provides comprehensive model comparison capabilities including
statistical significance testing, performance ranking, and comparative analysis.
"""

import itertools
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy import stats

from .result_models import EvaluationResult
from .results_storage import ResultsStorage


class ModelComparisonEngine:
    """
    Comprehensive model comparison and statistical analysis engine.

    Provides statistical comparison, significance testing, ranking,
    and comprehensive comparative performance assessment.
    """

    def __init__(self, storage: ResultsStorage):
        """
        Initialize comparison engine.

        Args:
            storage: ResultsStorage instance for data access
        """
        self.storage = storage

    def compare_models(
        self,
        model_names: list[str],
        task_types: list[str] | None = None,
        metric_name: str = "accuracy",
        time_period: str | None = None,
        dataset_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive comparison of multiple models.

        Args:
            model_names: List of model names to compare
            task_types: Optional list of task types to filter by
            metric_name: Primary metric for comparison
            time_period: Time period for analysis (e.g., "30d", "7d")
            dataset_names: Optional list of datasets to filter by

        Returns:
            Comprehensive comparison results including rankings and statistics
        """
        # Build query filters
        filters: dict[str, Any] = {}
        if time_period:
            days = int(time_period.rstrip("d"))
            filters["start_date"] = datetime.now() - timedelta(days=days)

        if task_types:
            filters["task_type"] = task_types[0]  # Storage API limitation

        if dataset_names:
            filters["dataset_name"] = dataset_names[0]  # Storage API limitation

        # Get results for all models
        model_results = {}
        for model_name in model_names:
            filters["model_name"] = model_name
            results = self.storage.query_results(filters)
            model_results[model_name] = [r for r in results if metric_name in r.metrics]

        # Perform comparison analysis
        comparison_results = {
            "comparison_metadata": {
                "models_compared": model_names,
                "primary_metric": metric_name,
                "comparison_timestamp": datetime.now().isoformat(),
                "time_period": time_period,
                "task_types": task_types,
                "dataset_names": dataset_names,
                "total_evaluations": sum(len(results) for results in model_results.values()),
            },
            "performance_summary": self._generate_performance_summary(model_results, metric_name),
            "statistical_comparison": self._perform_statistical_comparison(
                model_results, metric_name
            ),
            "pairwise_comparisons": self._perform_pairwise_comparisons(model_results, metric_name),
            "ranking_analysis": self._generate_ranking_analysis(model_results, metric_name),
            "consistency_analysis": self._analyze_performance_consistency(
                model_results, metric_name
            ),
            "task_specific_analysis": self._analyze_task_specific_performance(
                model_results, metric_name
            ),
            "dataset_specific_analysis": self._analyze_dataset_specific_performance(
                model_results, metric_name
            ),
        }

        return comparison_results

    def statistical_significance_test(
        self,
        model_a_results: list[EvaluationResult],
        model_b_results: list[EvaluationResult],
        metric_name: str = "accuracy",
        test_type: str = "welch_t",
    ) -> dict[str, Any]:
        """
        Perform statistical significance test between two models.

        Args:
            model_a_results: Results for first model
            model_b_results: Results for second model
            metric_name: Metric to compare
            test_type: Type of statistical test ('welch_t', 'mann_whitney', 'paired_t')

        Returns:
            Statistical test results with p-value and interpretation
        """
        # Extract metric values
        values_a = [r.metrics[metric_name] for r in model_a_results if metric_name in r.metrics]
        values_b = [r.metrics[metric_name] for r in model_b_results if metric_name in r.metrics]

        if not values_a or not values_b:
            return {
                "error": "Insufficient data for statistical comparison",
                "values_a_count": len(values_a),
                "values_b_count": len(values_b),
            }

        # Perform statistical test
        test_results = {}

        if test_type == "welch_t":
            statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
            test_results["test_name"] = "Welch's t-test"

        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            test_results["test_name"] = "Mann-Whitney U test"

        elif test_type == "paired_t":
            if len(values_a) != len(values_b):
                return {
                    "error": "Paired t-test requires equal sample sizes",
                    "values_a_count": len(values_a),
                    "values_b_count": len(values_b),
                }
            statistic, p_value = stats.ttest_rel(values_a, values_b)
            test_results["test_name"] = "Paired t-test"

        else:
            return {"error": f"Unknown test type: {test_type}"}

        # Calculate effect size (Cohen's d for t-tests)
        effect_size = None
        if test_type in ["welch_t", "paired_t"]:
            pooled_std = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2)
            if pooled_std > 0:
                effect_size = (np.mean(values_a) - np.mean(values_b)) / pooled_std

        # Interpret results
        significance_level = 0.05
        is_significant = p_value < significance_level

        # Effect size interpretation for Cohen's d
        effect_size_interpretation = None
        if effect_size is not None:
            abs_effect = abs(effect_size)
            if abs_effect < 0.2:
                effect_size_interpretation = "negligible"
            elif abs_effect < 0.5:
                effect_size_interpretation = "small"
            elif abs_effect < 0.8:
                effect_size_interpretation = "medium"
            else:
                effect_size_interpretation = "large"

        return {
            "test_results": {
                "test_type": test_type,
                "test_name": test_results["test_name"],
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_significant": is_significant,
                "significance_level": significance_level,
                "effect_size": float(effect_size) if effect_size is not None else None,
                "effect_size_interpretation": effect_size_interpretation,
            },
            "descriptive_statistics": {
                "model_a": {
                    "mean": float(np.mean(values_a)),
                    "std": float(np.std(values_a, ddof=1)),
                    "median": float(np.median(values_a)),
                    "count": len(values_a),
                    "min": float(np.min(values_a)),
                    "max": float(np.max(values_a)),
                },
                "model_b": {
                    "mean": float(np.mean(values_b)),
                    "std": float(np.std(values_b, ddof=1)),
                    "median": float(np.median(values_b)),
                    "count": len(values_b),
                    "min": float(np.min(values_b)),
                    "max": float(np.max(values_b)),
                },
                "difference": {
                    "mean_difference": float(np.mean(values_a) - np.mean(values_b)),
                    "median_difference": float(np.median(values_a) - np.median(values_b)),
                },
            },
            "interpretation": self._interpret_statistical_test(
                is_significant, effect_size, p_value
            ),
        }

    def rank_models(
        self,
        model_names: list[str],
        metrics: list[str],
        weights: dict[str, float] | None = None,
        task_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Rank models using multiple metrics with optional weighting.

        Args:
            model_names: List of model names to rank
            metrics: List of metrics to consider in ranking
            weights: Optional weights for each metric (defaults to equal weighting)
            task_types: Optional list of task types to filter by

        Returns:
            Model rankings with scores and detailed breakdown
        """
        if weights is None:
            weights = {metric: 1.0 / len(metrics) for metric in metrics}

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Get results for all models
        model_metrics = {}
        for model_name in model_names:
            filters = {"model_name": model_name}
            if task_types:
                filters["task_type"] = task_types[0]  # Storage API limitation

            results = self.storage.query_results(filters)

            # Calculate average metrics
            metric_values = defaultdict(list)
            for result in results:
                for metric in metrics:
                    if metric in result.metrics:
                        metric_values[metric].append(result.metrics[metric])

            # Calculate averages
            model_metrics[model_name] = {
                metric: np.mean(values) if values else 0.0
                for metric, values in metric_values.items()
            }

        # Normalize metrics to 0-1 scale for fair comparison
        normalized_metrics = self._normalize_metrics_for_ranking(model_metrics, metrics)

        # Calculate weighted scores
        model_scores = {}
        for model_name in model_names:
            score = sum(
                normalized_metrics[model_name].get(metric, 0.0) * normalized_weights[metric]
                for metric in metrics
            )
            model_scores[model_name] = score

        # Create rankings
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "ranking_metadata": {
                "models_ranked": model_names,
                "metrics_used": metrics,
                "weights": normalized_weights,
                "task_types": task_types,
                "ranking_timestamp": datetime.now().isoformat(),
            },
            "rankings": [
                {
                    "rank": i + 1,
                    "model_name": model_name,
                    "overall_score": float(score),
                    "metric_scores": {
                        metric: float(normalized_metrics[model_name].get(metric, 0.0))
                        for metric in metrics
                    },
                    "raw_metrics": {
                        metric: float(model_metrics[model_name].get(metric, 0.0))
                        for metric in metrics
                    },
                }
                for i, (model_name, score) in enumerate(ranked_models)
            ],
            "metric_analysis": self._analyze_ranking_metrics(model_metrics, metrics),
        }

    def _generate_performance_summary(
        self, model_results: dict[str, list[EvaluationResult]], metric_name: str
    ) -> dict[str, Any]:
        """Generate performance summary for all models."""
        summary = {}

        for model_name, results in model_results.items():
            values = [r.metrics[metric_name] for r in results if metric_name in r.metrics]

            if values:
                summary[model_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                    "coefficient_of_variation": float(np.std(values, ddof=1) / np.mean(values))
                    if len(values) > 1 and np.mean(values) > 0
                    else 0.0,
                }
            else:
                summary[model_name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                    "coefficient_of_variation": 0.0,
                }

        return summary

    def _perform_statistical_comparison(
        self, model_results: dict[str, list[EvaluationResult]], metric_name: str
    ) -> dict[str, Any]:
        """Perform comprehensive statistical comparison."""
        model_names = list(model_results.keys())

        if len(model_names) < 2:
            return {"error": "Need at least 2 models for comparison"}

        # ANOVA test for multiple groups
        groups = []
        group_labels = []

        for model_name, results in model_results.items():
            values = [r.metrics[metric_name] for r in results if metric_name in r.metrics]
            if values:
                groups.append(values)
                group_labels.append(model_name)

        if len(groups) < 2:
            return {"error": "Insufficient data for statistical comparison"}

        # Perform one-way ANOVA
        f_statistic, anova_p_value = stats.f_oneway(*groups)

        # Kruskal-Wallis test (non-parametric alternative)
        kw_statistic, kw_p_value = stats.kruskal(*groups)

        return {
            "anova": {
                "f_statistic": float(f_statistic),
                "p_value": float(anova_p_value),
                "is_significant": anova_p_value < 0.05,
                "test_name": "One-way ANOVA",
            },
            "kruskal_wallis": {
                "statistic": float(kw_statistic),
                "p_value": float(kw_p_value),
                "is_significant": kw_p_value < 0.05,
                "test_name": "Kruskal-Wallis H-test",
            },
            "interpretation": "Significant differences detected between models"
            if anova_p_value < 0.05
            else "No significant differences detected between models",
        }

    def _perform_pairwise_comparisons(
        self, model_results: dict[str, list[EvaluationResult]], metric_name: str
    ) -> dict[str, Any]:
        """Perform pairwise statistical comparisons between all model pairs."""
        model_names = list(model_results.keys())
        pairwise_results = {}

        for model_a, model_b in itertools.combinations(model_names, 2):
            pair_key = f"{model_a}_vs_{model_b}"

            results_a = model_results[model_a]
            results_b = model_results[model_b]

            comparison = self.statistical_significance_test(
                results_a, results_b, metric_name, "welch_t"
            )

            pairwise_results[pair_key] = comparison

        return pairwise_results

    def _generate_ranking_analysis(
        self, model_results: dict[str, list[EvaluationResult]], metric_name: str
    ) -> dict[str, Any]:
        """Generate ranking analysis based on performance metrics."""
        # Calculate mean performance for ranking
        model_means = {}
        for model_name, results in model_results.items():
            values = [r.metrics[metric_name] for r in results if metric_name in r.metrics]
            model_means[model_name] = np.mean(values) if values else 0.0

        # Sort by performance
        ranked_models = sorted(model_means.items(), key=lambda x: float(x[1]), reverse=True)

        return {
            "rankings": [
                {
                    "rank": i + 1,
                    "model_name": model_name,
                    "mean_performance": float(performance),
                    "evaluation_count": len(model_results[model_name]),
                }
                for i, (model_name, performance) in enumerate(ranked_models)
            ],
            "best_model": ranked_models[0][0] if ranked_models else None,
            "worst_model": ranked_models[-1][0] if ranked_models else None,
            "performance_gap": float(ranked_models[0][1] - ranked_models[-1][1])
            if len(ranked_models) >= 2
            else 0.0,
        }

    def _analyze_performance_consistency(
        self, model_results: dict[str, list[EvaluationResult]], metric_name: str
    ) -> dict[str, Any]:
        """Analyze performance consistency across evaluations."""
        consistency_analysis = {}

        for model_name, results in model_results.items():
            values = [r.metrics[metric_name] for r in results if metric_name in r.metrics]

            if len(values) > 1:
                cv = (
                    np.std(values, ddof=1) / np.mean(values)
                    if np.mean(values) > 0
                    else float("inf")
                )
                iqr = np.percentile(values, 75) - np.percentile(values, 25)

                consistency_analysis[model_name] = {
                    "coefficient_of_variation": float(cv),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "interquartile_range": float(iqr),
                    "range": float(np.max(values) - np.min(values)),
                    "consistency_score": float(1.0 / (1.0 + cv)) if cv != float("inf") else 0.0,
                }
            else:
                consistency_analysis[model_name] = {
                    "coefficient_of_variation": 0.0,
                    "standard_deviation": 0.0,
                    "interquartile_range": 0.0,
                    "range": 0.0,
                    "consistency_score": 1.0 if len(values) == 1 else 0.0,
                }

        # Rank by consistency
        if consistency_analysis:
            most_consistent = max(
                consistency_analysis.items(), key=lambda x: x[1]["consistency_score"]
            )
            least_consistent = min(
                consistency_analysis.items(), key=lambda x: x[1]["consistency_score"]
            )
        else:
            most_consistent = least_consistent = ("", {})

        return {
            "model_consistency": consistency_analysis,
            "most_consistent_model": most_consistent[0],
            "least_consistent_model": least_consistent[0],
        }

    def _analyze_task_specific_performance(
        self, model_results: dict[str, list[EvaluationResult]], metric_name: str
    ) -> dict[str, Any]:
        """Analyze performance by task type."""
        task_performance: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        # Group results by task type
        for model_name, results in model_results.items():
            for result in results:
                if metric_name in result.metrics:
                    task_performance[result.task_type][model_name].append(
                        result.metrics[metric_name]
                    )

        # Calculate statistics for each task type
        task_analysis = {}
        for task_type, model_data in task_performance.items():
            task_stats = {}
            for model_name, values in model_data.items():
                if values:
                    task_stats[model_name] = {
                        "mean": float(np.mean(values)),
                        "count": len(values),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    }

            if task_stats:
                best_model = max(task_stats.items(), key=lambda x: x[1]["mean"])
                task_analysis[task_type] = {
                    "model_performance": task_stats,
                    "best_model": best_model[0],
                    "best_performance": best_model[1]["mean"],
                }

        return task_analysis

    def _analyze_dataset_specific_performance(
        self, model_results: dict[str, list[EvaluationResult]], metric_name: str
    ) -> dict[str, Any]:
        """Analyze performance by dataset."""
        dataset_performance: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Group results by dataset
        for model_name, results in model_results.items():
            for result in results:
                if metric_name in result.metrics:
                    dataset_performance[result.dataset_name][model_name].append(
                        result.metrics[metric_name]
                    )

        # Calculate statistics for each dataset
        dataset_analysis = {}
        for dataset_name, model_data in dataset_performance.items():
            dataset_stats = {}
            for model_name, values in model_data.items():
                if values:
                    dataset_stats[model_name] = {
                        "mean": float(np.mean(values)),
                        "count": len(values),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    }

            if dataset_stats:
                best_model = max(dataset_stats.items(), key=lambda x: x[1]["mean"])
                dataset_analysis[dataset_name] = {
                    "model_performance": dataset_stats,
                    "best_model": best_model[0],
                    "best_performance": best_model[1]["mean"],
                }

        return dataset_analysis

    def _normalize_metrics_for_ranking(
        self, model_metrics: dict[str, dict[str, Any]], metrics: list[str]
    ) -> dict[str, dict[str, float]]:
        """Normalize metrics to 0-1 scale for fair ranking comparison."""
        normalized: dict[str, dict[str, float]] = {}

        # Find min/max for each metric across all models
        metric_ranges = {}
        for metric in metrics:
            values = [model_data.get(metric, 0.0) for model_data in model_metrics.values()]
            metric_ranges[metric] = {
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 1.0,
            }

        # Normalize each model's metrics
        for model_name, model_data in model_metrics.items():
            normalized[model_name] = {}
            for metric in metrics:
                value = model_data.get(metric, 0.0)
                min_val = metric_ranges[metric]["min"]
                max_val = metric_ranges[metric]["max"]

                # Normalize to 0-1 range
                if max_val > min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 1.0 if value == max_val else 0.0

                normalized[model_name][metric] = normalized_value

        return normalized

    def _analyze_ranking_metrics(
        self, model_metrics: dict[str, dict[str, Any]], metrics: list[str]
    ) -> dict[str, Any]:
        """Analyze the metrics used in ranking."""
        metric_analysis = {}

        for metric in metrics:
            values = [model_data.get(metric, 0.0) for model_data in model_metrics.values()]

            if values:
                metric_analysis[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "range": float(max(values) - min(values)),
                    "discriminative_power": float(np.std(values, ddof=1) / np.mean(values))
                    if np.mean(values) > 0 and len(values) > 1
                    else 0.0,
                }
            else:
                metric_analysis[metric] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "range": 0.0,
                    "discriminative_power": 0.0,
                }

        return metric_analysis

    def _interpret_statistical_test(
        self, is_significant: bool, effect_size: float | None, p_value: float
    ) -> str:
        """Provide human-readable interpretation of statistical test results."""
        if not is_significant:
            return f"No statistically significant difference detected (p={p_value:.4f}). The observed difference could be due to random variation."

        interpretation = f"Statistically significant difference detected (p={p_value:.4f}). "

        if effect_size is not None:
            if abs(effect_size) < 0.2:
                interpretation += "However, the effect size is negligible, suggesting the practical significance may be limited."
            elif abs(effect_size) < 0.5:
                interpretation += (
                    "The effect size is small, indicating a modest practical difference."
                )
            elif abs(effect_size) < 0.8:
                interpretation += (
                    "The effect size is medium, indicating a meaningful practical difference."
                )
            else:
                interpretation += (
                    "The effect size is large, indicating a substantial practical difference."
                )

        return interpretation

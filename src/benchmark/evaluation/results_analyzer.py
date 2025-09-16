"""
Advanced analysis tools for evaluation results.

This module provides comprehensive analysis capabilities for model performance
evaluation, including trend analysis, bottleneck detection, and statistical insights.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy import stats

from .result_models import EvaluationResult
from .results_storage import ResultsStorage


class PerformanceAnalysisError(Exception):
    """Exception raised when performance analysis fails."""

    pass


class ResultsAnalyzer:
    """
    Comprehensive analyzer for evaluation results.

    Provides advanced analysis capabilities including performance trends,
    bottleneck detection, and statistical insights across model evaluations.
    """

    def __init__(self, results_storage: ResultsStorage):
        """
        Initialize the results analyzer.

        Args:
            results_storage: Storage instance for accessing evaluation data
        """
        self.storage = results_storage

    def analyze_model_performance(self, model_name: str) -> dict[str, Any]:
        """
        Comprehensive analysis of model performance across tasks.

        Args:
            model_name: Name of the model to analyze

        Returns:
            Dictionary containing comprehensive performance analysis
        """
        try:
            # Get all results for the model
            results = self.storage.query_results({"model_name": model_name})

            if not results:
                return {
                    "model_name": model_name,
                    "error": "No evaluation results found for this model",
                    "total_evaluations": 0,
                }

            # Basic statistics
            total_evaluations = len(results)
            successful_evaluations = len([r for r in results if r.success_rate == 1.0])

            # Task-wise performance analysis
            task_performance = self._analyze_task_performance(results)

            # Dataset-wise performance analysis
            dataset_performance = self._analyze_dataset_performance(results)

            # Metric analysis
            metric_analysis = self._analyze_metrics(results)

            # Performance consistency
            consistency_analysis = self._analyze_consistency(results)

            # Recent performance vs overall
            recent_vs_overall = self._compare_recent_vs_overall_performance(results)

            # Processing time analysis
            timing_analysis = self._analyze_processing_times(results)

            return {
                "model_name": model_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_evaluations": total_evaluations,
                    "successful_evaluations": successful_evaluations,
                    "success_rate": successful_evaluations / total_evaluations
                    if total_evaluations > 0
                    else 0,
                    "evaluation_period": self._get_evaluation_period(results),
                    "unique_tasks": len({r.task_type for r in results}),
                    "unique_datasets": len({r.dataset_name for r in results}),
                },
                "task_performance": task_performance,
                "dataset_performance": dataset_performance,
                "metric_analysis": metric_analysis,
                "consistency_analysis": consistency_analysis,
                "recent_vs_overall": recent_vs_overall,
                "timing_analysis": timing_analysis,
                "recommendations": self._generate_recommendations(
                    results, metric_analysis, consistency_analysis
                ),
            }

        except Exception as e:
            raise PerformanceAnalysisError(f"Failed to analyze model performance: {e}") from e

    def identify_performance_trends(
        self, model_name: str, time_period: str = "30d"
    ) -> dict[str, Any]:
        """
        Identify performance trends over time.

        Args:
            model_name: Name of the model to analyze
            time_period: Time period for analysis ("7d", "30d", "90d", "1y")

        Returns:
            Dictionary containing trend analysis results
        """
        try:
            # Parse time period
            days = self._parse_time_period(time_period)
            start_date = datetime.now() - timedelta(days=days)

            # Get results within time period
            results = self.storage.query_results(
                {
                    "model_name": model_name,
                    "start_date": start_date,
                    "sort_by": "timestamp",
                    "sort_order": "asc",
                }
            )

            if len(results) < 2:
                return {
                    "model_name": model_name,
                    "time_period": time_period,
                    "error": "Insufficient data for trend analysis (need at least 2 evaluations)",
                    "total_evaluations": len(results),
                }

            # Group results by time windows
            time_windows = self._group_by_time_windows(results, days)

            # Analyze trends for each metric
            metric_trends = self._analyze_metric_trends(time_windows)

            # Identify significant trends
            significant_trends = self._identify_significant_trends(metric_trends)

            # Performance velocity (rate of change)
            velocity_analysis = self._calculate_performance_velocity(results)

            # Seasonality analysis (if enough data)
            seasonality_analysis = (
                self._analyze_seasonality(results) if len(results) >= 10 else None
            )

            return {
                "model_name": model_name,
                "time_period": time_period,
                "analysis_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_evaluations": len(results),
                    "time_span_days": days,
                    "evaluation_frequency": len(results) / days if days > 0 else 0,
                    "metrics_analyzed": len(metric_trends),
                },
                "metric_trends": metric_trends,
                "significant_trends": significant_trends,
                "velocity_analysis": velocity_analysis,
                "seasonality_analysis": seasonality_analysis,
                "trend_summary": self._summarize_trends(significant_trends),
            }

        except Exception as e:
            raise PerformanceAnalysisError(f"Failed to identify performance trends: {e}") from e

    def find_performance_bottlenecks(
        self, evaluation_results: list[EvaluationResult]
    ) -> dict[str, Any]:
        """
        Identify areas where model performance is weakest.

        Args:
            evaluation_results: List of evaluation results to analyze

        Returns:
            Dictionary containing bottleneck analysis
        """
        try:
            if not evaluation_results:
                return {"error": "No evaluation results provided"}

            # Metric-based bottlenecks
            metric_bottlenecks = self._find_metric_bottlenecks(evaluation_results)

            # Task-based bottlenecks
            task_bottlenecks = self._find_task_bottlenecks(evaluation_results)

            # Dataset-based bottlenecks
            dataset_bottlenecks = self._find_dataset_bottlenecks(evaluation_results)

            # Configuration-based bottlenecks
            config_bottlenecks = self._find_configuration_bottlenecks(evaluation_results)

            # Processing time bottlenecks
            timing_bottlenecks = self._find_timing_bottlenecks(evaluation_results)

            # Cross-dimensional analysis
            cross_analysis = self._perform_cross_dimensional_analysis(evaluation_results)

            # Priority scoring for bottlenecks
            bottleneck_priorities = self._score_bottleneck_priorities(
                metric_bottlenecks, task_bottlenecks, dataset_bottlenecks, timing_bottlenecks
            )

            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_evaluations": len(evaluation_results),
                    "metrics_analyzed": len(
                        set().union(*[r.metrics.keys() for r in evaluation_results])
                    ),
                    "tasks_analyzed": len({r.task_type for r in evaluation_results}),
                    "datasets_analyzed": len({r.dataset_name for r in evaluation_results}),
                },
                "metric_bottlenecks": metric_bottlenecks,
                "task_bottlenecks": task_bottlenecks,
                "dataset_bottlenecks": dataset_bottlenecks,
                "configuration_bottlenecks": config_bottlenecks,
                "timing_bottlenecks": timing_bottlenecks,
                "cross_dimensional_analysis": cross_analysis,
                "bottleneck_priorities": bottleneck_priorities,
                "recommendations": self._generate_bottleneck_recommendations(bottleneck_priorities),
            }

        except Exception as e:
            raise PerformanceAnalysisError(f"Failed to find performance bottlenecks: {e}") from e

    def generate_performance_summary(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """
        Generate executive summary of evaluation results.

        Args:
            results: List of evaluation results to summarize

        Returns:
            Dictionary containing performance summary
        """
        try:
            if not results:
                return {"error": "No evaluation results provided"}

            # High-level statistics
            summary_stats = self._calculate_summary_statistics(results)

            # Key insights
            key_insights = self._extract_key_insights(results)

            # Performance highlights
            highlights = self._identify_performance_highlights(results)

            # Areas for improvement
            improvement_areas = self._identify_improvement_areas(results)

            # Model ranking (if multiple models)
            model_ranking = self._rank_models(results)

            # Executive recommendations
            executive_recommendations = self._generate_executive_recommendations(
                summary_stats, key_insights, improvement_areas
            )

            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "executive_summary": {
                    "total_evaluations": len(results),
                    "models_evaluated": len({r.model_name for r in results}),
                    "tasks_covered": len({r.task_type for r in results}),
                    "datasets_used": len({r.dataset_name for r in results}),
                    "evaluation_period": self._get_evaluation_period(results),
                    "overall_success_rate": sum(r.success_rate for r in results) / len(results),
                },
                "summary_statistics": summary_stats,
                "key_insights": key_insights,
                "performance_highlights": highlights,
                "improvement_areas": improvement_areas,
                "model_ranking": model_ranking,
                "executive_recommendations": executive_recommendations,
            }

        except Exception as e:
            raise PerformanceAnalysisError(f"Failed to generate performance summary: {e}") from e

    # Helper methods for analysis

    def _analyze_task_performance(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze performance across different tasks."""
        task_groups = defaultdict(list)
        for result in results:
            task_groups[result.task_type].append(result)

        task_analysis = {}
        for task_type, task_results in task_groups.items():
            # Calculate average metrics for this task
            all_metrics: dict[str, list[float]] = {}
            for result in task_results:
                for metric, value in result.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

            avg_metrics = {
                metric: statistics.mean(values) for metric, values in all_metrics.items()
            }

            task_analysis[task_type] = {
                "evaluations_count": len(task_results),
                "success_rate": sum(r.success_rate for r in task_results) / len(task_results),
                "average_metrics": avg_metrics,
                "metric_stability": {
                    metric: statistics.stdev(values) if len(values) > 1 else 0.0
                    for metric, values in all_metrics.items()
                },
                "best_performance": max(
                    task_results, key=lambda r: r.get_primary_metric() or 0
                ).evaluation_id,
                "average_processing_time": statistics.mean(
                    [r.processing_time for r in task_results]
                ),
            }

        return task_analysis

    def _analyze_dataset_performance(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze performance across different datasets."""
        dataset_groups = defaultdict(list)
        for result in results:
            dataset_groups[result.dataset_name].append(result)

        dataset_analysis = {}
        for dataset_name, dataset_results in dataset_groups.items():
            all_metrics: dict[str, list[float]] = {}
            for result in dataset_results:
                for metric, value in result.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

            avg_metrics = {
                metric: statistics.mean(values) for metric, values in all_metrics.items()
            }

            dataset_analysis[dataset_name] = {
                "evaluations_count": len(dataset_results),
                "success_rate": sum(r.success_rate for r in dataset_results) / len(dataset_results),
                "average_metrics": avg_metrics,
                "performance_range": {
                    metric: {
                        "min": min(values),
                        "max": max(values),
                        "range": max(values) - min(values),
                    }
                    for metric, values in all_metrics.items()
                },
                "average_processing_time": statistics.mean(
                    [r.processing_time for r in dataset_results]
                ),
            }

        return dataset_analysis

    def _analyze_metrics(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze metric distributions and correlations."""
        # Collect all metrics
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        metric_analysis = {}
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                metric_analysis[metric_name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "quartiles": self._calculate_quartiles(values),
                    "distribution_shape": self._analyze_distribution_shape(values),
                }

        # Calculate correlations between metrics
        correlations = self._calculate_metric_correlations(all_metrics)

        return {
            "individual_metrics": metric_analysis,
            "metric_correlations": correlations,
            "dominant_metrics": self._identify_dominant_metrics(metric_analysis),
        }

    def _analyze_consistency(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze performance consistency across evaluations."""
        if len(results) < 2:
            return {"error": "Need at least 2 evaluations for consistency analysis"}

        # Calculate coefficient of variation for each metric
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        consistency_scores: dict[str, dict[str, Any]] = {}
        for metric_name, values in all_metrics.items():
            if len(values) > 1 and statistics.mean(values) != 0:
                cv = statistics.stdev(values) / statistics.mean(values)
                consistency_scores[metric_name] = {
                    "coefficient_of_variation": cv,
                    "consistency_level": self._classify_consistency_level(cv),
                    "stability_score": max(0, 1 - cv),  # Higher is more stable
                }

        # Overall consistency score
        overall_consistency = (
            statistics.mean(
                [float(score["stability_score"]) for score in consistency_scores.values()]
            )
            if consistency_scores
            else 0.0
        )

        return {
            "overall_consistency_score": overall_consistency,
            "consistency_level": self._classify_consistency_level(1 - overall_consistency),
            "metric_consistency": consistency_scores,
            "most_consistent_metric": max(
                consistency_scores.items(), key=lambda x: float(x[1]["stability_score"])
            )[0]
            if consistency_scores
            else None,
            "least_consistent_metric": min(
                consistency_scores.items(), key=lambda x: float(x[1]["stability_score"])
            )[0]
            if consistency_scores
            else None,
        }

    def _compare_recent_vs_overall_performance(
        self, results: list[EvaluationResult], recent_days: int = 30
    ) -> dict[str, Any]:
        """Compare recent performance vs overall historical performance."""
        if len(results) < 2:
            return {"error": "Need at least 2 evaluations for comparison"}

        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)

        # Split into recent and historical
        cutoff_date = datetime.now() - timedelta(days=recent_days)
        recent_results = [r for r in sorted_results if r.timestamp >= cutoff_date]
        historical_results = [r for r in sorted_results if r.timestamp < cutoff_date]

        if not recent_results:
            return {"error": f"No evaluations found in the last {recent_days} days"}

        # Calculate metrics for each period
        recent_metrics = self._calculate_average_metrics(recent_results)
        overall_metrics = self._calculate_average_metrics(results)
        historical_metrics = (
            self._calculate_average_metrics(historical_results) if historical_results else {}
        )

        # Calculate improvement/degradation
        improvements = {}
        for metric, recent_value in recent_metrics.items():
            if metric in overall_metrics:
                change = recent_value - overall_metrics[metric]
                improvements[metric] = {
                    "recent_value": recent_value,
                    "overall_value": overall_metrics[metric],
                    "absolute_change": change,
                    "percentage_change": (change / overall_metrics[metric] * 100)
                    if overall_metrics[metric] != 0
                    else 0,
                    "trend": "improving" if change > 0 else "declining" if change < 0 else "stable",
                }

        return {
            "recent_period_days": recent_days,
            "recent_evaluations": len(recent_results),
            "historical_evaluations": len(historical_results),
            "recent_metrics": recent_metrics,
            "overall_metrics": overall_metrics,
            "historical_metrics": historical_metrics,
            "performance_changes": improvements,
            "overall_trend": self._determine_overall_trend(improvements),
        }

    def _analyze_processing_times(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze processing time patterns."""
        processing_times = [r.processing_time for r in results]

        if not processing_times:
            return {"error": "No processing time data available"}

        return {
            "average_time": statistics.mean(processing_times),
            "median_time": statistics.median(processing_times),
            "min_time": min(processing_times),
            "max_time": max(processing_times),
            "std_dev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0,
            "time_efficiency_score": self._calculate_efficiency_score(processing_times),
            "time_by_task": self._analyze_time_by_task(results),
            "time_by_dataset": self._analyze_time_by_dataset(results),
            "time_trend": self._analyze_time_trend(results),
        }

    def _parse_time_period(self, time_period: str) -> int:
        """Parse time period string to days."""
        time_period = time_period.lower()
        if time_period.endswith("d"):
            return int(time_period[:-1])
        elif time_period.endswith("w"):
            return int(time_period[:-1]) * 7
        elif time_period.endswith("m"):
            return int(time_period[:-1]) * 30
        elif time_period.endswith("y"):
            return int(time_period[:-1]) * 365
        else:
            raise ValueError(f"Invalid time period format: {time_period}")

    def _group_by_time_windows(
        self, results: list[EvaluationResult], total_days: int
    ) -> list[list[EvaluationResult]]:
        """Group results into time windows for trend analysis."""
        if total_days <= 7:
            window_size = 1  # Daily windows
        elif total_days <= 30:
            window_size = 3  # 3-day windows
        elif total_days <= 90:
            window_size = 7  # Weekly windows
        else:
            window_size = 30  # Monthly windows

        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)

        # Group into windows
        windows = []
        current_window: list[EvaluationResult] = []
        window_start = sorted_results[0].timestamp if sorted_results else datetime.now()

        for result in sorted_results:
            days_diff = (result.timestamp - window_start).days
            if days_diff >= window_size and current_window:
                windows.append(current_window)
                current_window = [result]
                window_start = result.timestamp
            else:
                current_window.append(result)

        if current_window:
            windows.append(current_window)

        return windows

    def _analyze_metric_trends(self, time_windows: list[list[EvaluationResult]]) -> dict[str, Any]:
        """Analyze trends for each metric across time windows."""
        if len(time_windows) < 2:
            return {}

        # Calculate average metrics for each window
        window_metrics = []
        for window in time_windows:
            avg_metrics = self._calculate_average_metrics(window)
            window_metrics.append(avg_metrics)

        # Analyze trends for each metric
        metric_trends = {}
        all_metric_names: set[str] = set()
        for window_metric in window_metrics:
            all_metric_names.update(window_metric.keys())

        for metric_name in all_metric_names:
            values = []
            for window_metric in window_metrics:
                if metric_name in window_metric:
                    values.append(window_metric[metric_name])

            if len(values) >= 2:
                # Calculate trend using linear regression
                x = list(range(len(values)))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

                metric_trends[metric_name] = {
                    "values": values,
                    "slope": slope,
                    "intercept": intercept,
                    "correlation": r_value,
                    "p_value": p_value,
                    "std_error": std_err,
                    "trend_direction": "increasing"
                    if slope > 0
                    else "decreasing"
                    if slope < 0
                    else "stable",
                    "trend_strength": abs(r_value),
                    "is_significant": p_value < 0.05 if not np.isnan(p_value) else False,
                }

        return metric_trends

    def _identify_significant_trends(self, metric_trends: dict[str, Any]) -> dict[str, Any]:
        """Identify statistically significant trends."""
        significant_trends = {}

        for metric_name, trend_data in metric_trends.items():
            if (
                trend_data.get("is_significant", False)
                and trend_data.get("trend_strength", 0) > 0.5
            ):
                significant_trends[metric_name] = {
                    "trend_direction": trend_data["trend_direction"],
                    "trend_strength": trend_data["trend_strength"],
                    "slope": trend_data["slope"],
                    "p_value": trend_data["p_value"],
                    "significance_level": "strong"
                    if trend_data["trend_strength"] > 0.8
                    else "moderate",
                }

        return significant_trends

    def _calculate_performance_velocity(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Calculate rate of performance change (velocity)."""
        if len(results) < 2:
            return {"error": "Need at least 2 evaluations for velocity calculation"}

        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)

        # Calculate velocity for each metric
        velocities = {}
        all_metrics: set[str] = set()
        for result in sorted_results:
            all_metrics.update(result.metrics.keys())

        for metric_name in all_metrics:
            metric_values = []
            timestamps = []

            for result in sorted_results:
                if metric_name in result.metrics:
                    metric_values.append(result.metrics[metric_name])
                    timestamps.append(result.timestamp.timestamp())

            if len(metric_values) >= 2:
                # Calculate velocity (change per day)
                time_diffs = np.diff(timestamps) / (24 * 3600)  # Convert to days
                value_diffs = np.diff(metric_values)

                # Calculate average velocity
                daily_velocities = value_diffs / time_diffs
                avg_velocity = np.mean(daily_velocities)

                velocities[metric_name] = {
                    "daily_velocity": avg_velocity,
                    "velocity_stability": np.std(daily_velocities)
                    if len(daily_velocities) > 1
                    else 0,
                    "acceleration": np.mean(np.diff(daily_velocities))
                    if len(daily_velocities) > 1
                    else 0,
                    "velocity_trend": "accelerating"
                    if len(daily_velocities) > 1 and np.mean(np.diff(daily_velocities)) > 0
                    else "decelerating"
                    if len(daily_velocities) > 1 and np.mean(np.diff(daily_velocities)) < 0
                    else "constant",
                }

        return velocities

    def _analyze_seasonality(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze seasonal patterns in performance."""
        if len(results) < 10:
            return {"error": "Need at least 10 evaluations for seasonality analysis"}

        # Group by day of week and hour of day
        day_of_week_performance = defaultdict(list)
        hour_of_day_performance = defaultdict(list)

        for result in results:
            day_of_week = result.timestamp.weekday()
            hour_of_day = result.timestamp.hour
            primary_metric = result.get_primary_metric()

            if primary_metric is not None:
                day_of_week_performance[day_of_week].append(primary_metric)
                hour_of_day_performance[hour_of_day].append(primary_metric)

        # Calculate averages
        day_averages = {
            day: statistics.mean(values) for day, values in day_of_week_performance.items()
        }
        hour_averages = {
            hour: statistics.mean(values) for hour, values in hour_of_day_performance.items()
        }

        # Detect patterns
        day_variance = statistics.variance(day_averages.values()) if len(day_averages) > 1 else 0
        hour_variance = statistics.variance(hour_averages.values()) if len(hour_averages) > 1 else 0

        return {
            "day_of_week_patterns": day_averages,
            "hour_of_day_patterns": hour_averages,
            "day_variance": day_variance,
            "hour_variance": hour_variance,
            "has_day_seasonality": day_variance > 0.01,  # Threshold for meaningful variance
            "has_hour_seasonality": hour_variance > 0.01,
            "best_performing_day": max(day_averages.items(), key=lambda x: x[1])[0]
            if day_averages
            else None,
            "best_performing_hour": max(hour_averages.items(), key=lambda x: x[1])[0]
            if hour_averages
            else None,
        }

    def _summarize_trends(self, significant_trends: dict[str, Any]) -> dict[str, Any]:
        """Summarize trend analysis results."""
        if not significant_trends:
            return {"summary": "No significant trends detected"}

        improving_metrics = [
            metric
            for metric, data in significant_trends.items()
            if data["trend_direction"] == "increasing"
        ]
        declining_metrics = [
            metric
            for metric, data in significant_trends.items()
            if data["trend_direction"] == "decreasing"
        ]

        return {
            "total_significant_trends": len(significant_trends),
            "improving_metrics": improving_metrics,
            "declining_metrics": declining_metrics,
            "improvement_count": len(improving_metrics),
            "decline_count": len(declining_metrics),
            "overall_trend": "improving"
            if len(improving_metrics) > len(declining_metrics)
            else "declining"
            if len(declining_metrics) > len(improving_metrics)
            else "mixed",
            "strongest_trend": max(significant_trends.items(), key=lambda x: x[1]["trend_strength"])
            if significant_trends
            else None,
        }

    # Bottleneck detection methods

    def _find_metric_bottlenecks(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Find metrics that consistently underperform."""
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        metric_bottlenecks = {}
        for metric_name, values in all_metrics.items():
            avg_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            min_value = min(values)

            # Consider a metric a bottleneck if it's consistently low
            # This is a heuristic - adjust thresholds based on domain knowledge
            is_bottleneck = (
                avg_value < 0.7  # Low average performance
                or min_value < 0.5  # Very low minimum
                or (std_dev > 0.2 and avg_value < 0.8)  # High variance with moderate performance
            )

            if is_bottleneck:
                metric_bottlenecks[metric_name] = {
                    "average_value": avg_value,
                    "min_value": min_value,
                    "max_value": max(values),
                    "std_dev": std_dev,
                    "bottleneck_severity": self._calculate_bottleneck_severity(
                        avg_value, min_value, std_dev
                    ),
                    "affected_evaluations": len(values),
                }

        return metric_bottlenecks

    def _find_task_bottlenecks(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Find tasks where performance is consistently poor."""
        task_performance = {}
        task_groups = defaultdict(list)

        for result in results:
            task_groups[result.task_type].append(result)

        for task_type, task_results in task_groups.items():
            primary_metrics: list[float] = []
            for r in task_results:
                metric = r.get_primary_metric()
                if metric is not None:
                    primary_metrics.append(metric)

            if primary_metrics:
                avg_performance = statistics.mean(primary_metrics)
                min_performance = min(primary_metrics)

                # Consider a task a bottleneck if average performance is low
                if avg_performance < 0.7 or min_performance < 0.5:
                    task_performance[task_type] = {
                        "average_performance": avg_performance,
                        "min_performance": min_performance,
                        "max_performance": max(primary_metrics),
                        "evaluations_count": len(task_results),
                        "bottleneck_severity": self._calculate_bottleneck_severity(
                            avg_performance, min_performance, 0
                        ),
                        "success_rate": sum(r.success_rate for r in task_results)
                        / len(task_results),
                    }

        return task_performance

    def _find_dataset_bottlenecks(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Find datasets where performance is consistently poor."""
        dataset_performance = {}
        dataset_groups = defaultdict(list)

        for result in results:
            dataset_groups[result.dataset_name].append(result)

        for dataset_name, dataset_results in dataset_groups.items():
            primary_metrics: list[float] = []
            for r in dataset_results:
                metric = r.get_primary_metric()
                if metric is not None:
                    primary_metrics.append(metric)

            if primary_metrics:
                avg_performance = statistics.mean(primary_metrics)
                min_performance = min(primary_metrics)

                if avg_performance < 0.7 or min_performance < 0.5:
                    dataset_performance[dataset_name] = {
                        "average_performance": avg_performance,
                        "min_performance": min_performance,
                        "max_performance": max(primary_metrics),
                        "evaluations_count": len(dataset_results),
                        "bottleneck_severity": self._calculate_bottleneck_severity(
                            avg_performance, min_performance, 0
                        ),
                        "success_rate": sum(r.success_rate for r in dataset_results)
                        / len(dataset_results),
                    }

        return dataset_performance

    def _find_configuration_bottlenecks(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Find configuration patterns that lead to poor performance."""
        config_performance = defaultdict(list)

        # Group by configuration values
        for result in results:
            primary_metric = result.get_primary_metric()
            if primary_metric is not None:
                for config_key, config_value in result.configuration.items():
                    config_performance[f"{config_key}:{config_value}"].append(primary_metric)

        bottleneck_configs = {}
        for config_setting, performance_values in config_performance.items():
            if len(performance_values) >= 2:  # Need multiple data points
                avg_performance = statistics.mean(performance_values)

                if avg_performance < 0.7:
                    bottleneck_configs[config_setting] = {
                        "average_performance": avg_performance,
                        "min_performance": min(performance_values),
                        "max_performance": max(performance_values),
                        "evaluations_count": len(performance_values),
                        "bottleneck_severity": self._calculate_bottleneck_severity(
                            avg_performance, min(performance_values), 0
                        ),
                    }

        return bottleneck_configs

    def _find_timing_bottlenecks(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Find performance issues related to processing time."""
        processing_times = [r.processing_time for r in results]
        primary_metrics: list[float] = []
        for r in results:
            metric = r.get_primary_metric()
            if metric is not None:
                primary_metrics.append(metric)

        if len(processing_times) < 2 or len(primary_metrics) < 2:
            return {"error": "Insufficient data for timing analysis"}

        # Calculate correlation between processing time and performance
        time_performance_correlation = (
            stats.pearsonr(processing_times[: len(primary_metrics)], primary_metrics)[0]
            if len(processing_times) >= len(primary_metrics)
            else 0
        )

        # Find slow evaluations
        avg_time = statistics.mean(processing_times)
        slow_threshold = avg_time * 1.5  # 50% above average
        slow_evaluations = [r for r in results if r.processing_time > slow_threshold]

        # Analyze performance of slow evaluations
        slow_performance: list[float] = []
        for r in slow_evaluations:
            metric = r.get_primary_metric()
            if metric is not None:
                slow_performance.append(metric)

        timing_analysis = {
            "time_performance_correlation": time_performance_correlation,
            "slow_evaluations_count": len(slow_evaluations),
            "slow_threshold_seconds": slow_threshold,
            "average_processing_time": avg_time,
            "max_processing_time": max(processing_times),
            "timing_bottleneck_detected": len(slow_evaluations)
            > len(results) * 0.2,  # More than 20% are slow
        }

        if slow_performance:
            timing_analysis.update(
                {
                    "slow_evaluations_avg_performance": statistics.mean(slow_performance),
                    "performance_degradation_when_slow": statistics.mean(primary_metrics)
                    - statistics.mean(slow_performance)
                    if slow_performance
                    else 0,
                }
            )

        return timing_analysis

    def _perform_cross_dimensional_analysis(
        self, results: list[EvaluationResult]
    ) -> dict[str, Any]:
        """Analyze bottlenecks across multiple dimensions."""
        # Find combinations that consistently underperform
        combination_performance = defaultdict(list)

        for result in results:
            primary_metric = result.get_primary_metric()
            if primary_metric is not None:
                # Task + Dataset combination
                task_dataset = f"{result.task_type}:{result.dataset_name}"
                combination_performance[task_dataset].append(primary_metric)

                # Task + Model combination (if multiple models)
                task_model = f"{result.task_type}:{result.model_name}"
                combination_performance[task_model].append(primary_metric)

        poor_combinations = {}
        for combination, performance_values in combination_performance.items():
            if len(performance_values) >= 2:
                avg_performance = statistics.mean(performance_values)
                if avg_performance < 0.7:
                    poor_combinations[combination] = {
                        "average_performance": avg_performance,
                        "evaluations_count": len(performance_values),
                        "bottleneck_severity": self._calculate_bottleneck_severity(
                            avg_performance, min(performance_values), 0
                        ),
                    }

        return poor_combinations

    def _score_bottleneck_priorities(
        self,
        metric_bottlenecks: dict[str, Any],
        task_bottlenecks: dict[str, Any],
        dataset_bottlenecks: dict[str, Any],
        timing_bottlenecks: dict[str, Any],
    ) -> dict[str, Any]:
        """Score and prioritize bottlenecks for remediation."""
        priorities = []

        # Score metric bottlenecks
        for metric, data in metric_bottlenecks.items():
            priority_score = data["bottleneck_severity"] * data["affected_evaluations"]
            priorities.append(
                {
                    "type": "metric",
                    "name": metric,
                    "priority_score": priority_score,
                    "severity": data["bottleneck_severity"],
                    "impact": data["affected_evaluations"],
                    "details": data,
                }
            )

        # Score task bottlenecks
        for task, data in task_bottlenecks.items():
            priority_score = data["bottleneck_severity"] * data["evaluations_count"]
            priorities.append(
                {
                    "type": "task",
                    "name": task,
                    "priority_score": priority_score,
                    "severity": data["bottleneck_severity"],
                    "impact": data["evaluations_count"],
                    "details": data,
                }
            )

        # Score dataset bottlenecks
        for dataset, data in dataset_bottlenecks.items():
            priority_score = data["bottleneck_severity"] * data["evaluations_count"]
            priorities.append(
                {
                    "type": "dataset",
                    "name": dataset,
                    "priority_score": priority_score,
                    "severity": data["bottleneck_severity"],
                    "impact": data["evaluations_count"],
                    "details": data,
                }
            )

        # Sort by priority score
        priorities.sort(key=lambda x: x["priority_score"], reverse=True)

        return {
            "top_priorities": priorities[:5],  # Top 5 priorities
            "all_bottlenecks": priorities,
            "total_bottlenecks": len(priorities),
            "high_priority_count": len([p for p in priorities if p["priority_score"] > 10]),
            "medium_priority_count": len([p for p in priorities if 5 <= p["priority_score"] <= 10]),
            "low_priority_count": len([p for p in priorities if p["priority_score"] < 5]),
        }

    # Helper utility methods

    def _get_evaluation_period(self, results: list[EvaluationResult]) -> dict[str, str]:
        """Get the time period covered by evaluations."""
        if not results:
            return {}

        timestamps = [r.timestamp for r in results]
        start_date = min(timestamps)
        end_date = max(timestamps)

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "duration_days": str((end_date - start_date).days),
        }

    def _calculate_average_metrics(self, results: list[EvaluationResult]) -> dict[str, float]:
        """Calculate average metrics across results."""
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        return {metric: statistics.mean(values) for metric, values in all_metrics.items()}

    def _calculate_quartiles(self, values: list[float]) -> dict[str, float]:
        """Calculate quartiles for a list of values."""
        if len(values) < 4:
            return {"q1": min(values), "q2": statistics.median(values), "q3": max(values)}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "q1": sorted_values[n // 4],
            "q2": statistics.median(sorted_values),
            "q3": sorted_values[3 * n // 4],
        }

    def _analyze_distribution_shape(self, values: list[float]) -> dict[str, Any]:
        """Analyze the shape of a distribution."""
        if len(values) < 3:
            return {"shape": "insufficient_data"}

        # Calculate skewness and kurtosis
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        if std_val == 0:
            return {"shape": "constant", "skewness": 0, "kurtosis": 0}

        # Simple skewness calculation
        skewness = sum(((x - mean_val) / std_val) ** 3 for x in values) / len(values)
        kurtosis = sum(((x - mean_val) / std_val) ** 4 for x in values) / len(values) - 3

        # Classify distribution shape
        if abs(skewness) < 0.5:
            shape = "symmetric"
        elif skewness > 0.5:
            shape = "right_skewed"
        else:
            shape = "left_skewed"

        return {"shape": shape, "skewness": skewness, "kurtosis": kurtosis}

    def _calculate_metric_correlations(
        self, all_metrics: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate correlations between metrics."""
        correlations: dict[str, dict[str, float]] = {}
        metric_names = list(all_metrics.keys())

        for i, metric1 in enumerate(metric_names):
            correlations[metric1] = {}
            for j, metric2 in enumerate(metric_names):
                if (
                    i != j
                    and len(all_metrics[metric1]) == len(all_metrics[metric2])
                    and len(all_metrics[metric1]) > 1
                ):
                    try:
                        corr, _ = stats.pearsonr(all_metrics[metric1], all_metrics[metric2])
                        correlations[metric1][metric2] = corr if not np.isnan(corr) else 0.0
                    except (ValueError, TypeError):
                        correlations[metric1][metric2] = 0.0

        return correlations

    def _identify_dominant_metrics(self, metric_analysis: dict[str, Any]) -> list[str]:
        """Identify metrics that have the most impact."""
        # Sort metrics by range (variability) and mean value
        metrics_with_scores = []

        for metric_name, data in metric_analysis.items():
            # Higher range and higher mean suggest more important metric
            importance_score = data["range"] * data["mean"]
            metrics_with_scores.append((metric_name, importance_score))

        # Sort by importance score
        metrics_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [metric for metric, _ in metrics_with_scores[:3]]  # Top 3 dominant metrics

    def _classify_consistency_level(self, cv_or_stability: float) -> str:
        """Classify consistency level based on coefficient of variation or stability score."""
        if cv_or_stability < 0.1:
            return "very_consistent"
        elif cv_or_stability < 0.2:
            return "consistent"
        elif cv_or_stability < 0.3:
            return "moderately_consistent"
        else:
            return "inconsistent"

    def _determine_overall_trend(self, improvements: dict[str, Any]) -> str:
        """Determine overall performance trend."""
        if not improvements:
            return "no_data"

        improving_count = sum(1 for data in improvements.values() if data["trend"] == "improving")
        declining_count = sum(1 for data in improvements.values() if data["trend"] == "declining")

        if improving_count > declining_count:
            return "improving"
        elif declining_count > improving_count:
            return "declining"
        else:
            return "stable"

    def _calculate_efficiency_score(self, processing_times: list[float]) -> float:
        """Calculate efficiency score based on processing times."""
        if not processing_times:
            return 0.0

        avg_time = statistics.mean(processing_times)
        min_time = min(processing_times)

        # Efficiency score: higher is better, based on how close times are to minimum
        if avg_time == 0:
            return 1.0

        efficiency = min_time / avg_time
        return efficiency

    def _analyze_time_by_task(self, results: list[EvaluationResult]) -> dict[str, float]:
        """Analyze processing time by task type."""
        task_times = defaultdict(list)
        for result in results:
            task_times[result.task_type].append(result.processing_time)

        return {task: statistics.mean(times) for task, times in task_times.items()}

    def _analyze_time_by_dataset(self, results: list[EvaluationResult]) -> dict[str, float]:
        """Analyze processing time by dataset."""
        dataset_times = defaultdict(list)
        for result in results:
            dataset_times[result.dataset_name].append(result.processing_time)

        return {dataset: statistics.mean(times) for dataset, times in dataset_times.items()}

    def _analyze_time_trend(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze trend in processing times over time."""
        if len(results) < 2:
            return {"trend": "insufficient_data"}

        # Sort by timestamp and analyze time trend
        sorted_results = sorted(results, key=lambda r: r.timestamp)
        times = [r.processing_time for r in sorted_results]

        if len(times) >= 2:
            x = list(range(len(times)))
            slope, _, r_value, p_value, _ = stats.linregress(x, times)

            return {
                "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "slope": slope,
                "correlation": r_value,
                "is_significant": p_value < 0.05 if not np.isnan(p_value) else False,
            }

        return {"trend": "insufficient_data"}

    def _calculate_bottleneck_severity(
        self, avg_value: float, min_value: float, std_dev: float
    ) -> float:
        """Calculate severity score for bottlenecks."""
        # Severity increases with lower performance and higher variance
        performance_penalty = max(0, 1 - avg_value)
        consistency_penalty = min(std_dev, 0.5)  # Cap at 0.5
        min_penalty = max(0, 0.7 - min_value)

        severity = (performance_penalty * 0.5) + (consistency_penalty * 0.3) + (min_penalty * 0.2)
        return min(severity, 1.0)  # Cap at 1.0

    def _calculate_summary_statistics(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Calculate high-level summary statistics."""
        if not results:
            return {}

        # Overall success rate
        success_rate = sum(r.success_rate for r in results) / len(results)

        # Average processing time
        avg_processing_time = statistics.mean([r.processing_time for r in results])

        # Metric statistics
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        metric_averages = {
            metric: statistics.mean(values) for metric, values in all_metrics.items()
        }

        return {
            "total_evaluations": len(results),
            "overall_success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "metric_averages": metric_averages,
            "evaluation_period": self._get_evaluation_period(results),
            "models_count": len({r.model_name for r in results}),
            "tasks_count": len({r.task_type for r in results}),
            "datasets_count": len({r.dataset_name for r in results}),
        }

    def _extract_key_insights(self, results: list[EvaluationResult]) -> list[str]:
        """Extract key insights from evaluation results."""
        insights = []

        if not results:
            return ["No evaluation data available"]

        # Success rate insights
        success_rate = sum(r.success_rate for r in results) / len(results)
        if success_rate >= 0.95:
            insights.append("Excellent success rate across all evaluations")
        elif success_rate < 0.8:
            insights.append("Success rate below 80% - investigate failure causes")

        # Processing time insights
        processing_times = [r.processing_time for r in results]
        avg_time = statistics.mean(processing_times)
        if avg_time > 60:
            insights.append("Processing times are high - consider optimization")

        # Performance variance insights
        primary_metrics: list[float] = []
        for r in results:
            metric = r.get_primary_metric()
            if metric is not None:
                primary_metrics.append(metric)
        if primary_metrics and len(primary_metrics) > 1:
            cv = statistics.stdev(primary_metrics) / statistics.mean(primary_metrics)
            if cv > 0.3:
                insights.append("High performance variance detected - inconsistent results")

        # Task diversity insights
        task_count = len({r.task_type for r in results})
        if task_count > 3:
            insights.append(f"Good task diversity with {task_count} different task types")

        return insights

    def _identify_performance_highlights(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Identify performance highlights and achievements."""
        if not results:
            return {}

        # Best performing evaluation
        best_result = max(results, key=lambda r: r.get_primary_metric() or 0)

        # Fastest evaluation
        fastest_result = min(results, key=lambda r: r.processing_time)

        # Most consistent metrics
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        most_consistent_metric = None
        lowest_cv = float("inf")
        for metric, values in all_metrics.items():
            if len(values) > 1:
                cv = (
                    statistics.stdev(values) / statistics.mean(values)
                    if statistics.mean(values) != 0
                    else float("inf")
                )
                if cv < lowest_cv:
                    lowest_cv = cv
                    most_consistent_metric = metric

        return {
            "best_performance": {
                "evaluation_id": best_result.evaluation_id,
                "primary_metric_value": best_result.get_primary_metric(),
                "task_type": best_result.task_type,
                "dataset_name": best_result.dataset_name,
            },
            "fastest_evaluation": {
                "evaluation_id": fastest_result.evaluation_id,
                "processing_time": fastest_result.processing_time,
                "task_type": fastest_result.task_type,
            },
            "most_consistent_metric": {
                "metric_name": most_consistent_metric,
                "coefficient_of_variation": lowest_cv,
            }
            if most_consistent_metric
            else None,
        }

    def _identify_improvement_areas(self, results: list[EvaluationResult]) -> list[dict[str, Any]]:
        """Identify areas for improvement."""
        improvement_areas: list[dict[str, Any]] = []

        if not results:
            return improvement_areas

        # Find bottlenecks
        bottlenecks = self.find_performance_bottlenecks(results)

        # Convert bottlenecks to improvement areas
        for bottleneck_type, bottleneck_data in bottlenecks.items():
            if isinstance(bottleneck_data, dict) and "error" not in bottleneck_data:
                if bottleneck_type == "metric_bottlenecks":
                    for metric, data in bottleneck_data.items():
                        improvement_areas.append(
                            {
                                "area": "metric_performance",
                                "focus": metric,
                                "current_performance": data["average_value"],
                                "priority": "high"
                                if data["bottleneck_severity"] > 0.7
                                else "medium",
                                "recommendation": f"Improve {metric} performance - currently averaging {data['average_value']:.3f}",
                            }
                        )

                elif bottleneck_type == "task_bottlenecks":
                    for task, data in bottleneck_data.items():
                        improvement_areas.append(
                            {
                                "area": "task_performance",
                                "focus": task,
                                "current_performance": data["average_performance"],
                                "priority": "high"
                                if data["bottleneck_severity"] > 0.7
                                else "medium",
                                "recommendation": f"Focus on improving performance for {task} tasks",
                            }
                        )

        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        improvement_areas.sort(
            key=lambda x: priority_order.get(str(x["priority"]), 0), reverse=True
        )

        return improvement_areas[:5]  # Top 5 improvement areas

    def _rank_models(self, results: list[EvaluationResult]) -> list[dict[str, Any]]:
        """Rank models by performance."""
        model_performance = defaultdict(list)

        for result in results:
            primary_metric = result.get_primary_metric()
            if primary_metric is not None:
                model_performance[result.model_name].append(primary_metric)

        model_rankings: list[dict[str, Any]] = []
        for model_name, performance_values in model_performance.items():
            avg_performance = statistics.mean(performance_values)
            model_rankings.append(
                {
                    "model_name": model_name,
                    "average_performance": avg_performance,
                    "evaluations_count": len(performance_values),
                    "best_performance": max(performance_values),
                    "consistency": 1 - (statistics.stdev(performance_values) / avg_performance)
                    if len(performance_values) > 1 and avg_performance != 0
                    else 1,
                }
            )

        # Sort by average performance
        model_rankings.sort(key=lambda x: float(x["average_performance"]), reverse=True)

        # Add ranks
        for i, model_data in enumerate(model_rankings):
            model_data["rank"] = i + 1

        return model_rankings

    def _generate_recommendations(
        self,
        results: list[EvaluationResult],
        metric_analysis: dict[str, Any],
        consistency_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Success rate recommendations
        success_rate = sum(r.success_rate for r in results) / len(results) if results else 0
        if success_rate < 0.9:
            recommendations.append(
                "Investigate and address evaluation failures to improve success rate"
            )

        # Consistency recommendations
        overall_consistency = consistency_analysis.get("overall_consistency_score", 0)
        if overall_consistency < 0.7:
            recommendations.append(
                "Work on improving model consistency - results show high variance"
            )

        # Metric-specific recommendations
        dominant_metrics = metric_analysis.get("dominant_metrics", [])
        if dominant_metrics:
            worst_metric = dominant_metrics[-1] if len(dominant_metrics) > 0 else None
            if worst_metric:
                recommendations.append(f"Focus on improving {worst_metric} performance")

        # Processing time recommendations
        avg_time = statistics.mean([r.processing_time for r in results]) if results else 0
        if avg_time > 30:
            recommendations.append(
                "Consider optimizing processing pipeline to reduce evaluation time"
            )

        return recommendations

    def _generate_executive_recommendations(
        self,
        summary_stats: dict[str, Any],
        key_insights: list[str],
        improvement_areas: list[dict[str, Any]],
    ) -> list[str]:
        """Generate executive-level recommendations."""
        recommendations = []

        # Success rate
        success_rate = summary_stats.get("overall_success_rate", 0)
        if success_rate < 0.95:
            recommendations.append(
                "Prioritize improving evaluation success rate to ensure reliable performance measurement"
            )

        # Top improvement areas
        high_priority_areas = [area for area in improvement_areas if area.get("priority") == "high"]
        if high_priority_areas:
            recommendations.append(
                f"Address high-priority bottlenecks in: {', '.join([area['focus'] for area in high_priority_areas[:3]])}"
            )

        # Model diversity
        models_count = summary_stats.get("models_count", 0)
        if models_count == 1:
            recommendations.append(
                "Consider evaluating multiple models for comparison and robustness assessment"
            )

        # Task coverage
        tasks_count = summary_stats.get("tasks_count", 0)
        if tasks_count < 3:
            recommendations.append("Expand evaluation coverage to include more diverse task types")

        return recommendations

    def _generate_bottleneck_recommendations(
        self, bottleneck_priorities: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for addressing bottlenecks."""
        recommendations = []

        top_priorities = bottleneck_priorities.get("top_priorities", [])

        for priority in top_priorities[:3]:  # Top 3 priorities
            bottleneck_type = priority["type"]
            bottleneck_name = priority["name"]

            if bottleneck_type == "metric":
                recommendations.append(f"Urgent: Address {bottleneck_name} metric underperformance")
            elif bottleneck_type == "task":
                recommendations.append(
                    f"High priority: Improve model performance on {bottleneck_name} tasks"
                )
            elif bottleneck_type == "dataset":
                recommendations.append(
                    f"Focus: Optimize model for {bottleneck_name} dataset characteristics"
                )

        # General recommendations based on bottleneck count
        total_bottlenecks = bottleneck_priorities.get("total_bottlenecks", 0)
        if total_bottlenecks > 5:
            recommendations.append(
                "Consider systematic model architecture review due to multiple performance bottlenecks"
            )

        return recommendations

"""Results aggregation workflow step for organizing and analyzing evaluation results."""

from datetime import datetime
from typing import Any

from benchmark.core.logging import get_logger
from benchmark.interfaces.orchestration_interfaces import WorkflowContext, WorkflowStep


class ResultsAggregationStep(WorkflowStep):
    """Workflow step for aggregating and organizing evaluation results with comprehensive analysis."""

    def __init__(self) -> None:
        self.logger = get_logger("results_aggregation_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute results aggregation step with comprehensive analysis."""
        self.logger.info(
            "Starting results aggregation step for experiment %s", context.experiment_id
        )

        # Get evaluation results from previous step
        evaluation_step_result = context.step_results.get("evaluation_execution", {})
        evaluation_results = evaluation_step_result.get("results", {})

        if not evaluation_results:
            self.logger.warning("No evaluation results found for aggregation")
            return {"message": "No results to aggregate", "summary": {}}

        try:
            # Generate comprehensive analysis
            comprehensive_analysis = await self._generate_comprehensive_analysis(
                evaluation_results, context
            )

            # Create model comparison analysis
            model_comparisons = self._create_model_comparisons(evaluation_results)

            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(evaluation_results)

            # Generate insights and recommendations
            insights = self._generate_experiment_insights(
                evaluation_results, context.config, summary_stats
            )

            # Create experiment summary
            experiment_summary = self._create_experiment_summary(
                context, evaluation_results, summary_stats
            )

            # Prepare final aggregated results
            aggregated_results = {
                "aggregation_timestamp": datetime.now().isoformat(),
                "experiment_summary": experiment_summary,
                "comprehensive_analysis": comprehensive_analysis,
                "model_comparisons": model_comparisons,
                "summary_statistics": summary_stats,
                "insights_and_recommendations": insights,
                "raw_evaluation_results": evaluation_results,
                "metadata": {
                    "total_evaluations": len(evaluation_results),
                    "aggregation_method": "comprehensive_analysis",
                    "analysis_version": "1.0",
                },
            }

            self.logger.info("Results aggregation step completed successfully")
            return aggregated_results

        except Exception as e:
            self.logger.error("Results aggregation step failed: %s", e)
            raise

    async def _generate_comprehensive_analysis(
        self, evaluation_results: dict[str, Any], context: WorkflowContext
    ) -> dict[str, Any]:
        """Generate comprehensive analysis of evaluation results."""
        try:
            evaluation_service = context.services.get("evaluation")

            if evaluation_service and hasattr(evaluation_service, "generate_comprehensive_report"):
                try:
                    comprehensive_report = await evaluation_service.generate_comprehensive_report(
                        context.experiment_id
                    )
                    return comprehensive_report  # type: ignore
                except Exception as e:
                    self.logger.warning("Failed to generate service report: %s", e)

            # Fallback to local analysis
            return self._create_local_comprehensive_analysis(evaluation_results)

        except Exception as e:
            self.logger.warning("Comprehensive analysis failed, using fallback: %s", e)
            return self._create_local_comprehensive_analysis(evaluation_results)

    def _create_local_comprehensive_analysis(
        self, evaluation_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Create comprehensive analysis using local methods."""
        analysis: dict[str, Any] = {
            "performance_overview": {},
            "detailed_metrics": {},
            "error_analysis": {},
            "performance_trends": {},
        }

        # Performance overview
        successful_results = [
            result for result in evaluation_results.values() if result.get("status") == "success"
        ]

        failed_results = [
            result for result in evaluation_results.values() if result.get("status") == "failed"
        ]

        analysis["performance_overview"] = {
            "total_evaluations": len(evaluation_results),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(failed_results),
            "success_rate": len(successful_results) / len(evaluation_results)
            if evaluation_results
            else 0,
        }

        # Detailed metrics analysis
        metrics_data: dict[str, list[float]] = {}
        for result in successful_results:
            evaluation_data = result.get("evaluation_data", {})
            for metric_name, metric_value in evaluation_data.items():
                if isinstance(metric_value, int | float):
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(metric_value)

        for metric_name, values in metrics_data.items():
            if values:
                analysis["detailed_metrics"][metric_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": self._calculate_std(values),
                }

        # Error analysis
        error_types: dict[str, int] = {}
        for result in failed_results:
            error = result.get("error", "Unknown error")
            error_type = error.split(":")[0] if ":" in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1

        analysis["error_analysis"] = {
            "error_distribution": error_types,
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0]
            if error_types
            else None,
        }

        return analysis

    def _create_model_comparisons(self, evaluation_results: dict[str, Any]) -> dict[str, Any]:
        """Create model comparison analysis."""
        model_performance: dict[str, dict[str, Any]] = {}

        # Group results by model
        for result in evaluation_results.values():
            if result.get("status") != "success":
                continue

            model_id = result.get("model_id")
            if not model_id:
                continue

            if model_id not in model_performance:
                model_performance[model_id] = {
                    "evaluations": [],
                    "datasets": set(),
                    "metrics": {},
                }

            model_performance[model_id]["evaluations"].append(result)
            model_performance[model_id]["datasets"].add(result.get("dataset_id"))

            # Aggregate metrics
            evaluation_data = result.get("evaluation_data", {})
            for metric_name, metric_value in evaluation_data.items():
                if isinstance(metric_value, int | float):
                    if metric_name not in model_performance[model_id]["metrics"]:
                        model_performance[model_id]["metrics"][metric_name] = []
                    model_performance[model_id]["metrics"][metric_name].append(metric_value)

        # Calculate model summaries
        model_summaries = {}
        for model_id, performance in model_performance.items():
            metrics_summary = {}
            for metric_name, values in performance["metrics"].items():
                if values:
                    metrics_summary[metric_name] = {
                        "average": sum(values) / len(values),
                        "best": max(values),
                        "worst": min(values),
                        "consistency": 1.0
                        - (self._calculate_std(values) / (sum(values) / len(values)))
                        if values
                        else 0.0,
                    }

            model_summaries[model_id] = {
                "total_evaluations": len(performance["evaluations"]),
                "datasets_tested": len(performance["datasets"]),
                "metrics_summary": metrics_summary,
                "overall_score": self._calculate_overall_model_score(metrics_summary),
            }

        # Find best performing model
        best_model: str | None = None
        best_score = 0.0
        for model_id, summary in model_summaries.items():
            score = float(summary["overall_score"])  # type: ignore
            if score > best_score:
                best_score = score
                best_model = model_id

        return {
            "model_summaries": model_summaries,
            "best_performing_model": {
                "model_id": best_model,
                "overall_score": best_score,
            }
            if best_model
            else None,
            "model_rankings": sorted(
                [
                    (model_id, summary["overall_score"])
                    for model_id, summary in model_summaries.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            ),
        }

    def _calculate_overall_model_score(self, metrics_summary: dict[str, Any]) -> float:
        """Calculate overall model score from metrics."""
        if not metrics_summary:
            return 0.0

        # Prioritize accuracy and f1_score if available
        priority_metrics = ["accuracy", "f1_score", "precision", "recall"]

        scores = []
        for metric_name in priority_metrics:
            if metric_name in metrics_summary:
                scores.append(float(metrics_summary[metric_name]["average"]))

        # If no priority metrics, use all available metrics
        if not scores:
            for metric_data in metrics_summary.values():
                if isinstance(metric_data, dict) and "average" in metric_data:
                    scores.append(float(metric_data["average"]))

        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_summary_statistics(self, evaluation_results: dict[str, Any]) -> dict[str, Any]:
        """Calculate summary statistics across all evaluations."""
        stats: dict[str, Any] = {
            "metrics_summary": {},
            "model_performance_summary": {},
            "dataset_difficulty_analysis": {},
        }

        # Group all metric values
        all_metrics: dict[str, list[float]] = {}
        model_metrics: dict[str, dict[str, list[float]]] = {}
        dataset_metrics: dict[str, dict[str, list[float]]] = {}

        for result in evaluation_results.values():
            if result.get("status") != "success":
                continue

            model_id = result.get("model_id")
            dataset_id = result.get("dataset_id")
            evaluation_data = result.get("evaluation_data", {})

            for metric_name, metric_value in evaluation_data.items():
                if isinstance(metric_value, int | float):
                    # All metrics
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

                    # Model-specific metrics
                    if model_id:
                        if model_id not in model_metrics:
                            model_metrics[model_id] = {}
                        if metric_name not in model_metrics[model_id]:
                            model_metrics[model_id][metric_name] = []
                        model_metrics[model_id][metric_name].append(metric_value)

                    # Dataset-specific metrics
                    if dataset_id:
                        if dataset_id not in dataset_metrics:
                            dataset_metrics[dataset_id] = {}
                        if metric_name not in dataset_metrics[dataset_id]:
                            dataset_metrics[dataset_id][metric_name] = []
                        dataset_metrics[dataset_id][metric_name].append(metric_value)

        # Calculate statistics for all metrics
        for metric_name, values in all_metrics.items():
            if values:
                stats["metrics_summary"][metric_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": self._calculate_std(values),
                }

        # Model performance summary
        for model_id, metrics in model_metrics.items():
            model_stats = {}
            for metric_name, values in metrics.items():
                if values:
                    model_stats[metric_name] = sum(values) / len(values)
            stats["model_performance_summary"][model_id] = model_stats

        # Dataset difficulty analysis
        for dataset_id, metrics in dataset_metrics.items():
            dataset_stats = {}
            for metric_name, values in metrics.items():
                if values:
                    dataset_stats[metric_name] = sum(values) / len(values)
            stats["dataset_difficulty_analysis"][dataset_id] = dataset_stats

        return stats

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return float(variance**0.5)

    def _generate_experiment_insights(
        self,
        evaluation_results: dict[str, Any],
        config: dict[str, Any],
        summary_stats: dict[str, Any],
    ) -> list[str]:
        """Generate insights from experiment results."""
        insights = []

        # Performance insights
        accuracy_stats = summary_stats.get("metrics_summary", {}).get("accuracy", {})
        if accuracy_stats:
            avg_accuracy = accuracy_stats["mean"]
            if avg_accuracy > 0.9:
                insights.append(f"Excellent overall accuracy achieved ({avg_accuracy:.3f})")
            elif avg_accuracy > 0.8:
                insights.append(f"Good overall accuracy achieved ({avg_accuracy:.3f})")
            elif avg_accuracy > 0.6:
                insights.append(
                    f"Moderate accuracy achieved ({avg_accuracy:.3f}) - consider model tuning"
                )
            else:
                insights.append(
                    f"Accuracy below expectations ({avg_accuracy:.3f}) - review model selection"
                )

        # Model consistency insights
        model_performance = summary_stats.get("model_performance_summary", {})
        if len(model_performance) > 1:
            accuracy_scores = [perf.get("accuracy", 0) for perf in model_performance.values()]
            if accuracy_scores:
                score_range = max(accuracy_scores) - min(accuracy_scores)
                if score_range > 0.2:
                    insights.append(f"High variability between models (range: {score_range:.3f})")
                else:
                    insights.append(
                        f"Consistent performance across models (range: {score_range:.3f})"
                    )

        # Configuration insights
        models_count = len(config.get("models", []))
        datasets_count = len(config.get("datasets", []))

        if models_count == 1 and accuracy_stats and accuracy_stats["mean"] < 0.7:
            insights.append(
                "Single model evaluation with suboptimal performance - consider testing additional models"
            )

        if datasets_count == 1:
            insights.append(
                "Single dataset evaluation - consider testing on multiple datasets for robustness validation"
            )

        # Success rate insights
        successful_results = sum(
            1 for r in evaluation_results.values() if r.get("status") == "success"
        )
        total_results = len(evaluation_results)
        success_rate = successful_results / total_results if total_results > 0 else 0

        if success_rate < 0.8:
            insights.append(
                f"Low evaluation success rate ({success_rate:.2%}) - investigate infrastructure issues"
            )
        elif success_rate == 1.0:
            insights.append(
                "Perfect evaluation success rate - all model-dataset combinations completed successfully"
            )

        return insights

    def _create_experiment_summary(
        self,
        context: WorkflowContext,
        evaluation_results: dict[str, Any],
        summary_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Create experiment summary."""
        successful_results = [
            r for r in evaluation_results.values() if r.get("status") == "success"
        ]

        models_evaluated = {r.get("model_id") for r in successful_results if r.get("model_id")}
        datasets_used = {r.get("dataset_id") for r in successful_results if r.get("dataset_id")}

        return {
            "experiment_id": context.experiment_id,
            "experiment_name": context.config.get("name", "Unnamed Experiment"),
            "total_evaluations": len(evaluation_results),
            "successful_evaluations": len(successful_results),
            "models_evaluated": len(models_evaluated),
            "datasets_used": len(datasets_used),
            "completion_rate": len(successful_results) / len(evaluation_results)
            if evaluation_results
            else 0,
            "completed_at": datetime.now().isoformat(),
            "primary_metrics": list(summary_stats.get("metrics_summary", {}).keys()),
        }

    def get_step_name(self) -> str:
        """Get step name."""
        return "results_aggregation"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return []  # No direct service dependencies

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 60.0  # 1 minute for results aggregation

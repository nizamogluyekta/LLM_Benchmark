"""Concrete workflow step implementations for experiment execution."""

from datetime import datetime
from typing import Any

from benchmark.core.logging import get_logger
from benchmark.interfaces.orchestration_interfaces import WorkflowContext, WorkflowStep


class DataLoadingStep(WorkflowStep):
    """Workflow step for loading datasets."""

    def __init__(self) -> None:
        self.logger = get_logger("data_loading_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute data loading step."""
        self.logger.info("Starting data loading step for experiment %s", context.experiment_id)

        data_service = context.services.get("data")
        if not data_service:
            raise Exception("Data service not available")

        datasets_config = context.config.get("datasets", [])
        if not datasets_config:
            self.logger.info("No datasets specified, skipping data loading")
            return {"datasets_loaded": 0, "datasets": []}

        loaded_datasets = {}
        dataset_info = []

        for dataset_config in datasets_config:
            dataset_id = dataset_config.get("id")
            if not dataset_id:
                raise Exception("Dataset configuration missing 'id' field")

            try:
                self.logger.info("Loading dataset: %s", dataset_id)

                # Load dataset using data service
                load_response = await data_service.load_dataset(
                    dataset_id=dataset_id,
                    loader_type=dataset_config.get("loader_type", "local"),
                    config=dataset_config.get("config", {}),
                )

                if not load_response.success:
                    raise Exception(f"Failed to load dataset {dataset_id}: {load_response.error}")

                loaded_datasets[dataset_id] = load_response.data
                dataset_info.append(
                    {
                        "id": dataset_id,
                        "size": load_response.data.get("size", 0),
                        "features": load_response.data.get("features", []),
                        "loader_type": dataset_config.get("loader_type"),
                    }
                )

                self.logger.info("Successfully loaded dataset: %s", dataset_id)

            except Exception as e:
                self.logger.error("Failed to load dataset %s: %s", dataset_id, str(e))
                raise Exception(f"Data loading failed for dataset {dataset_id}: {e}") from e

        # Store loaded datasets in context resources
        context.resources["loaded_datasets"] = loaded_datasets

        result = {
            "datasets_loaded": len(loaded_datasets),
            "datasets": dataset_info,
            "total_samples": sum(info.get("size", 0) for info in dataset_info),
        }

        self.logger.info("Data loading step completed: %d datasets loaded", len(loaded_datasets))
        return result

    def get_step_name(self) -> str:
        """Get step name."""
        return "data_loading"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return ["data"]

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 120.0  # 2 minutes for data loading


class ModelLoadingStep(WorkflowStep):
    """Workflow step for loading models."""

    def __init__(self) -> None:
        self.logger = get_logger("model_loading_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute model loading step."""
        self.logger.info("Starting model loading step for experiment %s", context.experiment_id)

        model_service = context.services.get("model")
        if not model_service:
            raise Exception("Model service not available")

        models_config = context.config.get("models", [])
        if not models_config:
            raise Exception("No models specified in configuration")

        loaded_models = {}
        model_info = []

        for model_config in models_config:
            model_id = model_config.get("id")
            if not model_id:
                raise Exception("Model configuration missing 'id' field")

            try:
                self.logger.info("Loading model: %s", model_id)

                # Load model using model service
                load_response = await model_service.load_model(
                    model_id=model_id,
                    plugin_type=model_config.get("plugin_type", "local"),
                    config=model_config.get("config", {}),
                )

                if not load_response.success:
                    raise Exception(f"Failed to load model {model_id}: {load_response.error}")

                loaded_models[model_id] = load_response.data
                model_info.append(
                    {
                        "id": model_id,
                        "plugin_type": model_config.get("plugin_type"),
                        "status": "loaded",
                        "memory_usage": load_response.data.get("memory_usage_mb", 0),
                    }
                )

                self.logger.info("Successfully loaded model: %s", model_id)

            except Exception as e:
                self.logger.error("Failed to load model %s: %s", model_id, str(e))
                raise Exception(f"Model loading failed for model {model_id}: {e}") from e

        # Store loaded models in context resources
        context.resources["loaded_models"] = loaded_models

        result = {
            "models_loaded": len(loaded_models),
            "models": model_info,
            "total_memory_usage_mb": sum(info.get("memory_usage", 0) for info in model_info),
        }

        self.logger.info("Model loading step completed: %d models loaded", len(loaded_models))
        return result

    def get_step_name(self) -> str:
        """Get step name."""
        return "model_loading"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return ["model"]

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 300.0  # 5 minutes for model loading


class EvaluationExecutionStep(WorkflowStep):
    """Workflow step for executing evaluations."""

    def __init__(self) -> None:
        self.logger = get_logger("evaluation_execution_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute evaluation step."""
        self.logger.info(
            "Starting evaluation execution step for experiment %s", context.experiment_id
        )

        evaluation_service = context.services.get("evaluation")
        if not evaluation_service:
            raise Exception("Evaluation service not available")

        # Get loaded datasets and models from previous steps
        loaded_datasets = context.resources.get("loaded_datasets", {})
        loaded_models = context.resources.get("loaded_models", {})

        if not loaded_datasets:
            raise Exception("No datasets available for evaluation")

        if not loaded_models:
            raise Exception("No models available for evaluation")

        evaluation_config = context.config.get("evaluation", {})
        metrics = evaluation_config.get("metrics", ["accuracy"])

        evaluation_results = {}
        total_evaluations = len(loaded_models) * len(loaded_datasets)
        completed_evaluations = 0

        self.logger.info(
            "Running %d evaluations (%d models × %d datasets)",
            total_evaluations,
            len(loaded_models),
            len(loaded_datasets),
        )

        for model_id in loaded_models:
            model_results = {}

            for dataset_id in loaded_datasets:
                try:
                    self.logger.info("Evaluating model %s on dataset %s", model_id, dataset_id)

                    # Run evaluation
                    eval_response = await evaluation_service.evaluate_model(
                        model_id=model_id,
                        dataset_id=dataset_id,
                        metrics=metrics,
                        config=evaluation_config,
                    )

                    if not eval_response.success:
                        raise Exception(f"Evaluation failed: {eval_response.error}")

                    model_results[dataset_id] = eval_response.data
                    completed_evaluations += 1

                    self.logger.info(
                        "Completed evaluation %d/%d: %s on %s",
                        completed_evaluations,
                        total_evaluations,
                        model_id,
                        dataset_id,
                    )

                except Exception as e:
                    self.logger.error(
                        "Failed evaluation of %s on %s: %s", model_id, dataset_id, str(e)
                    )
                    # Continue with other evaluations
                    model_results[dataset_id] = {"error": str(e), "status": "failed"}

            evaluation_results[model_id] = model_results

        # Calculate summary statistics
        successful_evaluations = 0
        failed_evaluations = 0

        for model_results in evaluation_results.values():
            for dataset_result in model_results.values():
                if "error" in dataset_result:
                    failed_evaluations += 1
                else:
                    successful_evaluations += 1

        result = {
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "success_rate": successful_evaluations / total_evaluations
            if total_evaluations > 0
            else 0,
            "results": evaluation_results,
            "metrics_used": metrics,
        }

        self.logger.info(
            "Evaluation execution step completed: %d/%d successful",
            successful_evaluations,
            total_evaluations,
        )

        return result

    def get_step_name(self) -> str:
        """Get step name."""
        return "evaluation_execution"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return ["evaluation", "model", "data"]

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 600.0  # 10 minutes for evaluation


class ResultsAggregationStep(WorkflowStep):
    """Workflow step for aggregating and analyzing results."""

    def __init__(self) -> None:
        self.logger = get_logger("results_aggregation_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute results aggregation step."""
        self.logger.info(
            "Starting results aggregation step for experiment %s", context.experiment_id
        )

        # Get evaluation results from previous step
        evaluation_step_result = context.step_results.get("evaluation_execution", {})
        evaluation_results = evaluation_step_result.get("results", {})

        if not evaluation_results:
            self.logger.warning("No evaluation results found for aggregation")
            return {"message": "No results to aggregate", "summary": {}}

        # Aggregate results by model
        model_summaries = {}
        dataset_summaries = {}
        overall_summary = {
            "best_model": None,
            "best_score": 0.0,
            "worst_model": None,
            "worst_score": float("inf"),
            "average_score": 0.0,
            "total_models": len(evaluation_results),
            "total_datasets": 0,
        }

        all_scores = []

        for model_id, model_results in evaluation_results.items():
            model_scores = []
            model_datasets = []

            for dataset_id, dataset_result in model_results.items():
                if "error" not in dataset_result:
                    # Extract primary score (assuming first metric)
                    score = self._extract_primary_score(dataset_result)
                    if score is not None:
                        model_scores.append(score)
                        all_scores.append(score)

                    model_datasets.append(dataset_id)

                    # Update dataset summary
                    if dataset_id not in dataset_summaries:
                        dataset_summaries[dataset_id] = {
                            "scores": [],
                            "models_evaluated": 0,
                            "best_model": None,
                            "best_score": 0.0,
                        }

                    if score is not None:
                        dataset_summaries[dataset_id]["scores"].append(score)  # type: ignore
                        dataset_summaries[dataset_id]["models_evaluated"] += 1  # type: ignore

                        if score > dataset_summaries[dataset_id]["best_score"]:  # type: ignore
                            dataset_summaries[dataset_id]["best_score"] = score
                            dataset_summaries[dataset_id]["best_model"] = model_id

            # Calculate model summary
            if model_scores:
                avg_score = sum(model_scores) / len(model_scores)
                model_summaries[model_id] = {
                    "average_score": avg_score,
                    "max_score": max(model_scores),
                    "min_score": min(model_scores),
                    "datasets_evaluated": len(model_datasets),
                    "successful_evaluations": len(model_scores),
                }

                # Update overall best/worst
                if avg_score > overall_summary["best_score"]:  # type: ignore
                    overall_summary["best_score"] = avg_score
                    overall_summary["best_model"] = model_id

                if avg_score < overall_summary["worst_score"]:  # type: ignore
                    overall_summary["worst_score"] = avg_score
                    overall_summary["worst_model"] = model_id

        # Calculate overall average
        if all_scores:
            overall_summary["average_score"] = sum(all_scores) / len(all_scores)

        overall_summary["total_datasets"] = len(dataset_summaries)

        # Generate recommendations
        recommendations = self._generate_recommendations(model_summaries, dataset_summaries)

        result = {
            "aggregation_timestamp": datetime.now().isoformat(),
            "overall_summary": overall_summary,
            "model_summaries": model_summaries,
            "dataset_summaries": dataset_summaries,
            "recommendations": recommendations,
            "metadata": {
                "total_scores_analyzed": len(all_scores),
                "aggregation_method": "average",
            },
        }

        self.logger.info("Results aggregation step completed")
        return result

    def _extract_primary_score(self, result: dict[str, Any]) -> float | None:
        """Extract primary score from evaluation result."""
        # Try to find accuracy score first
        if "accuracy" in result:
            return float(result["accuracy"])

        # Try other common metrics
        for metric in ["precision", "recall", "f1_score", "score"]:
            if metric in result:
                return float(result[metric])

        # Try to find any numeric value
        for value in result.values():
            if isinstance(value, int | float):
                return float(value)

        return None

    def _generate_recommendations(
        self, model_summaries: dict[str, Any], dataset_summaries: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on results."""
        recommendations = []

        if not model_summaries:
            recommendations.append("No successful evaluations completed")
            return recommendations

        # Find best performing model
        best_model = max(model_summaries.items(), key=lambda x: x[1]["average_score"])
        recommendations.append(
            f"Best performing model: {best_model[0]} "
            f"(average score: {best_model[1]['average_score']:.3f})"
        )

        # Check for consistent performance
        model_scores = [summary["average_score"] for summary in model_summaries.values()]
        mean_score = sum(model_scores) / len(model_scores)
        score_std = (
            sum((score - mean_score) ** 2 for score in model_scores) / len(model_scores)
        ) ** 0.5

        if score_std < 0.05:
            recommendations.append("Models show very similar performance - consider other factors")
        elif score_std > 0.2:
            recommendations.append("Significant performance differences between models")

        # Dataset-specific recommendations
        if len(dataset_summaries) > 1:
            dataset_scores = [
                sum(summary["scores"]) / len(summary["scores"]) if summary["scores"] else 0
                for summary in dataset_summaries.values()
            ]
            if dataset_scores and max(dataset_scores) - min(dataset_scores) > 0.1:
                recommendations.append("Performance varies significantly across datasets")

        return recommendations

    def get_step_name(self) -> str:
        """Get step name."""
        return "results_aggregation"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return []  # No direct service dependencies

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 30.0  # 30 seconds for aggregation

"""Evaluation execution workflow step for running model-dataset evaluations."""

import asyncio
from typing import Any

from benchmark.core.logging import get_logger
from benchmark.interfaces.orchestration_interfaces import WorkflowContext, WorkflowStep


class EvaluationExecutionStep(WorkflowStep):
    """Workflow step for executing evaluations across all model-dataset combinations."""

    def __init__(self) -> None:
        self.logger = get_logger("evaluation_execution_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute evaluation step across all model-dataset combinations."""
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
        parallel_jobs = evaluation_config.get("parallel_jobs", 2)

        # Create all model-dataset combinations
        evaluation_tasks = []
        for model_id in loaded_models:
            for dataset_id in loaded_datasets:
                task_id = f"{model_id}_{dataset_id}"
                evaluation_tasks.append(
                    {
                        "task_id": task_id,
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "metrics": metrics,
                    }
                )

        self.logger.info(
            "Running %d evaluations (%d models × %d datasets) with %d parallel jobs",
            len(evaluation_tasks),
            len(loaded_models),
            len(loaded_datasets),
            parallel_jobs,
        )

        evaluation_results = {}
        total_evaluations = len(evaluation_tasks)
        completed_evaluations = 0

        # Execute evaluations with concurrency control
        semaphore = asyncio.Semaphore(parallel_jobs)

        async def execute_single_evaluation(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            """Execute a single model-dataset evaluation."""
            async with semaphore:
                task_id = task["task_id"]
                model_id = task["model_id"]
                dataset_id = task["dataset_id"]
                metrics_list = task["metrics"]

                try:
                    self.logger.info("Evaluating model %s on dataset %s", model_id, dataset_id)

                    # Generate predictions for this model-dataset combination
                    predictions_result = await self._generate_predictions(
                        context, model_id, dataset_id
                    )

                    if not predictions_result["success"]:
                        raise Exception(predictions_result["error"])

                    predictions = predictions_result["predictions"]
                    ground_truth = predictions_result["ground_truth"]

                    # Run evaluation using evaluation service
                    eval_response = await evaluation_service.evaluate_model(
                        model_id=model_id,
                        dataset_id=dataset_id,
                        metrics=metrics_list,
                        config=evaluation_config,
                    )

                    if not eval_response.success:
                        raise Exception(f"Evaluation failed: {eval_response.error}")

                    # Store evaluation result
                    evaluation_data = eval_response.data
                    result = {
                        "status": "success",
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "predictions_count": len(predictions),
                        "ground_truth_count": len(ground_truth),
                        "metrics_evaluated": metrics_list,
                        "evaluation_data": evaluation_data,
                        "task_id": task_id,
                    }

                    nonlocal completed_evaluations
                    completed_evaluations += 1

                    self.logger.info(
                        "Completed evaluation %d/%d: %s on %s",
                        completed_evaluations,
                        total_evaluations,
                        model_id,
                        dataset_id,
                    )

                    return task_id, result

                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(
                        "Failed evaluation of %s on %s: %s", model_id, dataset_id, error_msg
                    )

                    result = {
                        "status": "failed",
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "error": error_msg,
                        "task_id": task_id,
                    }

                    return task_id, result

        # Execute all evaluation tasks
        try:
            evaluation_coroutines = [execute_single_evaluation(task) for task in evaluation_tasks]
            completed_tasks = await asyncio.gather(*evaluation_coroutines, return_exceptions=True)

            # Process results
            successful_evaluations = 0
            failed_evaluations = 0

            for result in completed_tasks:
                if isinstance(result, Exception):
                    failed_evaluations += 1
                    self.logger.error("Evaluation task failed with exception: %s", result)
                else:
                    task_id, task_result = result  # type: ignore
                    evaluation_results[task_id] = task_result

                    if task_result["status"] == "success":
                        successful_evaluations += 1
                    else:
                        failed_evaluations += 1

            if successful_evaluations == 0:
                raise Exception("All evaluations failed")

            # Organize results by model and dataset
            results_by_model: dict[str, dict[str, Any]] = {}
            results_by_dataset: dict[str, dict[str, Any]] = {}

            for _task_id, result in evaluation_results.items():  # type: ignore
                model_id = str(result["model_id"])  # type: ignore
                dataset_id = str(result["dataset_id"])  # type: ignore

                if model_id not in results_by_model:
                    results_by_model[model_id] = {}
                results_by_model[model_id][dataset_id] = result

                if dataset_id not in results_by_dataset:
                    results_by_dataset[dataset_id] = {}
                results_by_dataset[dataset_id][model_id] = result

            # Calculate success rate
            success_rate = (
                successful_evaluations / total_evaluations if total_evaluations > 0 else 0
            )

            # Generate evaluation summary
            evaluation_summary = {
                "total_evaluations": total_evaluations,
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "success_rate": success_rate,
                "results": evaluation_results,
                "results_by_model": results_by_model,
                "results_by_dataset": results_by_dataset,
                "metrics_used": metrics,
                "evaluation_config": evaluation_config,
            }

            self.logger.info(
                "Evaluation execution step completed: %d/%d successful evaluations (%.1f%% success rate)",
                successful_evaluations,
                total_evaluations,
                success_rate * 100,
            )

            return evaluation_summary

        except Exception as e:
            self.logger.error("Evaluation execution step failed: %s", e)
            raise

    async def _generate_predictions(
        self, context: WorkflowContext, model_id: str, dataset_id: str
    ) -> dict[str, Any]:
        """Generate predictions for a model on a dataset."""
        try:
            model_service = context.services.get("model")
            if not model_service:
                return {"success": False, "error": "Model service not available"}

            # Get dataset from loaded datasets
            loaded_datasets = context.resources.get("loaded_datasets", {})
            dataset = loaded_datasets.get(dataset_id)
            if not dataset:
                return {"success": False, "error": f"Dataset {dataset_id} not found"}

            # Extract test samples from dataset
            samples = dataset.get("samples", [])
            if not samples:
                return {"success": False, "error": "No samples in dataset"}

            # Extract input texts and ground truth labels
            input_texts = []
            ground_truth = []

            for sample in samples:
                if isinstance(sample, dict):
                    input_text = sample.get("input_text", sample.get("input", str(sample)))
                    label = sample.get("label", sample.get("ground_truth", "UNKNOWN"))
                else:
                    input_text = str(sample)
                    label = "UNKNOWN"

                input_texts.append(input_text)
                ground_truth.append({"label": label})

            # Generate predictions using model service
            try:
                predictions = await model_service.predict_batch(model_id, input_texts)
            except AttributeError:
                # Fallback for services without batch prediction
                predictions = []
                for input_text in input_texts:
                    pred_response = await model_service.predict(model_id, input_text)
                    if pred_response.success:
                        predictions.append(pred_response.data)
                    else:
                        predictions.append({"prediction": "UNKNOWN", "confidence": 0.0})

            # Ensure predictions have required fields
            processed_predictions = []
            for _i, pred in enumerate(predictions):
                if isinstance(pred, dict):
                    processed_pred = {
                        "prediction": pred.get("prediction", "UNKNOWN"),
                        "confidence": pred.get("confidence", 0.5),
                        "explanation": pred.get("explanation", ""),
                        "inference_time_ms": pred.get("inference_time_ms", 0.0),
                    }
                else:
                    processed_pred = {
                        "prediction": str(pred),
                        "confidence": 0.5,
                        "explanation": "",
                        "inference_time_ms": 0.0,
                    }

                processed_predictions.append(processed_pred)

            return {
                "success": True,
                "predictions": processed_predictions,
                "ground_truth": ground_truth,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_step_name(self) -> str:
        """Get step name."""
        return "evaluation_execution"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return ["evaluation", "model", "data"]

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 900.0  # 15 minutes for evaluation execution

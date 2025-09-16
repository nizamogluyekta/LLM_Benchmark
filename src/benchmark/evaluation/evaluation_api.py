"""
Clean API interface for evaluation service integration.

This module provides a comprehensive REST-like API interface that allows
external services and tools to interact with the evaluation service in a
standardized, well-documented manner.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from ..services.evaluation_service import EvaluationService
from .api_models import (
    APIError,
    APIResponse,
    AvailableEvaluatorsResponse,
    EvaluationID,
    EvaluationProgress,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultsResponse,
    EvaluationStatus,
    EvaluationStatusResponse,
    EvaluationType,
    ValidationError,
)
from .batch_evaluator import BatchEvaluator
from .evaluation_workflow import EvaluationWorkflow
from .result_models import EvaluationResult


class EvaluationAPI:
    """
    Clean API interface for evaluation service integration.

    Provides standardized endpoints for starting evaluations, monitoring progress,
    retrieving results, and managing evaluation workflows. Designed for both
    internal service integration and external tool interaction.

    Features:
    - Comprehensive request validation
    - Real-time progress tracking
    - Robust error handling
    - Resource management integration
    - Batch processing capabilities
    - Workflow orchestration
    """

    def __init__(
        self,
        evaluation_service: EvaluationService,
        max_concurrent_evaluations: int = 10,
        default_timeout_seconds: int = 3600,
    ):
        """
        Initialize the evaluation API.

        Args:
            evaluation_service: Core evaluation service instance
            max_concurrent_evaluations: Maximum concurrent evaluations allowed
            default_timeout_seconds: Default timeout for evaluations
        """
        self.service = evaluation_service
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.default_timeout_seconds = default_timeout_seconds

        # Initialize components
        self.workflow = EvaluationWorkflow(evaluation_service)
        self.batch_evaluator = BatchEvaluator(evaluation_service)

        # Tracking active evaluations
        self.active_evaluations: dict[EvaluationID, dict[str, Any]] = {}
        self.evaluation_lock = asyncio.Lock()

        # Logger
        self.logger = logging.getLogger(__name__)

        # Available evaluator types
        self._evaluator_types = {
            EvaluationType.SINGLE_MODEL: "Single model evaluation on specified tasks",
            EvaluationType.MODEL_COMPARISON: "Compare multiple models on same tasks",
            EvaluationType.BASELINE_EVALUATION: "Compare new model against established baselines",
            EvaluationType.COMPREHENSIVE: "Comprehensive evaluation with analysis and reporting",
            EvaluationType.BATCH_PROCESSING: "Efficient batch processing of multiple evaluations",
        }

    async def start_evaluation(self, request: EvaluationRequest) -> APIResponse:
        """
        Start a new evaluation based on the provided request.

        Args:
            request: Detailed evaluation request with configuration

        Returns:
            EvaluationResponse with evaluation ID and status, or APIError

        Example:
            ```python
            api = EvaluationAPI(evaluation_service)

            request = EvaluationRequest(
                model_names=["gpt-4", "claude-3"],
                task_types=["text_classification"],
                evaluation_type=EvaluationType.MODEL_COMPARISON
            )

            response = await api.start_evaluation(request)
            if isinstance(response, EvaluationResponse):
                evaluation_id = response.evaluation_id
            ```
        """
        # Validate request
        validation_errors = self._validate_evaluation_request(request)
        if validation_errors:
            return APIError(
                error_code="VALIDATION_ERROR",
                error_message="Request validation failed",
                validation_errors=validation_errors,
            )

        # Check capacity
        async with self.evaluation_lock:
            if len(self.active_evaluations) >= self.max_concurrent_evaluations:
                return APIError(
                    error_code="CAPACITY_EXCEEDED",
                    error_message=f"Maximum concurrent evaluations ({self.max_concurrent_evaluations}) exceeded",
                )

            # Create evaluation ID
            evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"

            # Initialize evaluation tracking
            self.active_evaluations[evaluation_id] = {
                "request": request,
                "status": EvaluationStatus.PENDING,
                "created_at": datetime.now(),
                "start_time": None,
                "end_time": None,
                "progress": None,
                "results": [],
                "error_message": None,
                "warnings": [],
            }

        try:
            # Start evaluation asynchronously
            asyncio.create_task(self._execute_evaluation(evaluation_id, request))

            # Estimate completion time
            estimated_completion = self._estimate_completion_time(request)

            self.logger.info(f"Started evaluation {evaluation_id} for request {request.request_id}")

            return EvaluationResponse(
                evaluation_id=evaluation_id,
                request_id=request.request_id,
                status=EvaluationStatus.PENDING,
                message="Evaluation started successfully",
                estimated_completion_time=estimated_completion,
            )

        except Exception as e:
            # Clean up on failure
            async with self.evaluation_lock:
                if evaluation_id in self.active_evaluations:
                    del self.active_evaluations[evaluation_id]

            self.logger.error(f"Failed to start evaluation: {e}")
            return APIError(
                error_code="EVALUATION_START_FAILED",
                error_message=f"Failed to start evaluation: {str(e)}",
            )

    async def get_evaluation_status(self, evaluation_id: EvaluationID) -> APIResponse:
        """
        Get the current status of a running or completed evaluation.

        Args:
            evaluation_id: Unique identifier for the evaluation

        Returns:
            EvaluationStatusResponse with current status and progress, or APIError

        Example:
            ```python
            status_response = await api.get_evaluation_status(evaluation_id)
            if isinstance(status_response, EvaluationStatusResponse):
                progress = status_response.progress.progress_percentage
                print(f"Evaluation is {progress:.1f}% complete")
            ```
        """
        if evaluation_id not in self.active_evaluations:
            return APIError(
                error_code="EVALUATION_NOT_FOUND",
                error_message=f"Evaluation {evaluation_id} not found",
            )

        eval_info = self.active_evaluations[evaluation_id]

        # Calculate duration if evaluation has started
        duration_seconds = None
        if eval_info["start_time"]:
            end_time = eval_info["end_time"] or datetime.now()
            duration_seconds = (end_time - eval_info["start_time"]).total_seconds()

        return EvaluationStatusResponse(
            evaluation_id=evaluation_id,
            status=eval_info["status"],
            progress=eval_info["progress"],
            start_time=eval_info["start_time"],
            end_time=eval_info["end_time"],
            duration_seconds=duration_seconds,
            error_message=eval_info["error_message"],
            warnings=eval_info["warnings"],
        )

    async def get_evaluation_results(self, evaluation_id: EvaluationID) -> APIResponse:
        """
        Get results from a completed evaluation.

        Args:
            evaluation_id: Unique identifier for the evaluation

        Returns:
            EvaluationResultsResponse with complete results, or APIError

        Example:
            ```python
            results_response = await api.get_evaluation_results(evaluation_id)
            if isinstance(results_response, EvaluationResultsResponse):
                for result in results_response.results:
                    print(f"{result.model_name}: {result.metrics['accuracy']:.3f}")
            ```
        """
        if evaluation_id not in self.active_evaluations:
            return APIError(
                error_code="EVALUATION_NOT_FOUND",
                error_message=f"Evaluation {evaluation_id} not found",
            )

        eval_info = self.active_evaluations[evaluation_id]

        if eval_info["status"] not in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]:
            return APIError(
                error_code="EVALUATION_NOT_COMPLETE",
                error_message=f"Evaluation {evaluation_id} is not yet complete (status: {eval_info['status'].value})",
            )

        # Retrieve results from workflow/batch evaluator
        results = eval_info.get("results", [])
        analysis = eval_info.get("analysis", {})
        comparison_results = eval_info.get("comparison_results", {})

        # Generate summary
        summary = self._generate_results_summary(results, eval_info["request"])

        return EvaluationResultsResponse(
            evaluation_id=evaluation_id,
            status=eval_info["status"],
            results=results,
            summary=summary,
            analysis=analysis,
            comparison_results=comparison_results,
            metadata={
                "request_id": eval_info["request"].request_id,
                "evaluation_type": eval_info["request"].evaluation_type.value,
                "duration_seconds": eval_info.get("duration_seconds"),
                "created_at": eval_info["created_at"].isoformat(),
            },
        )

    async def list_available_evaluators(self) -> APIResponse:
        """
        List all available evaluation types and their capabilities.

        Returns:
            AvailableEvaluatorsResponse with evaluation types and requirements

        Example:
            ```python
            evaluators_response = await api.list_available_evaluators()
            if isinstance(evaluators_response, AvailableEvaluatorsResponse):
                for eval_type in evaluators_response.evaluation_types:
                    print(f"Available: {eval_type}")
            ```
        """
        try:
            # Get available models from service (for future use)
            await self.service.list_available_models()  # type: ignore[attr-defined]

            # Define supported tasks (this would typically come from service)
            supported_tasks = {
                "text_classification": ["accuracy", "f1_score", "precision", "recall"],
                "sentiment_analysis": ["accuracy", "f1_score", "confusion_matrix"],
                "question_answering": ["exact_match", "f1_score", "rouge_l"],
                "text_generation": ["bleu", "rouge", "meteor", "perplexity"],
                "named_entity_recognition": ["f1_score", "precision", "recall"],
            }

            # Define model requirements for each evaluation type
            model_requirements = {
                EvaluationType.SINGLE_MODEL.value: {
                    "min_models": 1,
                    "max_models": 1,
                    "description": "Single model evaluation on specified tasks",
                },
                EvaluationType.MODEL_COMPARISON.value: {
                    "min_models": 2,
                    "max_models": 10,
                    "description": "Statistical comparison between multiple models",
                },
                EvaluationType.BASELINE_EVALUATION.value: {
                    "min_models": 2,
                    "max_models": 10,
                    "description": "Compare new model against established baselines",
                },
                EvaluationType.COMPREHENSIVE.value: {
                    "min_models": 1,
                    "max_models": 5,
                    "description": "Complete evaluation with analysis and reporting",
                },
                EvaluationType.BATCH_PROCESSING.value: {
                    "min_models": 1,
                    "max_models": 50,
                    "description": "Efficient parallel processing of multiple evaluations",
                },
            }

            return AvailableEvaluatorsResponse(
                evaluation_types=[eval_type.value for eval_type in self._evaluator_types],
                supported_tasks=supported_tasks,
                supported_metrics=list(set().union(*supported_tasks.values())),
                model_requirements=model_requirements,
            )

        except Exception as e:
            self.logger.error(f"Failed to get available evaluators: {e}")
            return APIError(
                error_code="SERVICE_ERROR",
                error_message=f"Failed to retrieve available evaluators: {str(e)}",
            )

    async def cancel_evaluation(self, evaluation_id: EvaluationID) -> APIResponse:
        """
        Cancel a running evaluation.

        Args:
            evaluation_id: Unique identifier for the evaluation

        Returns:
            EvaluationStatusResponse with updated status, or APIError
        """
        if evaluation_id not in self.active_evaluations:
            return APIError(
                error_code="EVALUATION_NOT_FOUND",
                error_message=f"Evaluation {evaluation_id} not found",
            )

        eval_info = self.active_evaluations[evaluation_id]

        if eval_info["status"] not in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
            return APIError(
                error_code="EVALUATION_NOT_CANCELLABLE",
                error_message=f"Evaluation {evaluation_id} cannot be cancelled (status: {eval_info['status'].value})",
            )

        # Update status
        eval_info["status"] = EvaluationStatus.CANCELLED
        eval_info["end_time"] = datetime.now()

        self.logger.info(f"Cancelled evaluation {evaluation_id}")

        return await self.get_evaluation_status(evaluation_id)

    async def list_active_evaluations(self) -> list[dict[str, Any]]:
        """
        List all currently active evaluations.

        Returns:
            List of evaluation summaries
        """
        active_list = []

        for eval_id, eval_info in self.active_evaluations.items():
            duration_seconds = None
            if eval_info["start_time"]:
                end_time = eval_info["end_time"] or datetime.now()
                duration_seconds = (end_time - eval_info["start_time"]).total_seconds()

            progress_percentage = 0.0
            if eval_info["progress"]:
                progress_percentage = eval_info["progress"].progress_percentage

            active_list.append(
                {
                    "evaluation_id": eval_id,
                    "request_id": eval_info["request"].request_id,
                    "status": eval_info["status"].value,
                    "evaluation_type": eval_info["request"].evaluation_type.value,
                    "model_count": len(eval_info["request"].model_names),
                    "task_count": len(eval_info["request"].task_types),
                    "created_at": eval_info["created_at"].isoformat(),
                    "duration_seconds": duration_seconds,
                    "progress_percentage": progress_percentage,
                }
            )

        return active_list

    async def cleanup_completed_evaluations(self, older_than_hours: int = 24) -> int:
        """
        Clean up completed evaluations older than specified time.

        Args:
            older_than_hours: Remove evaluations completed more than this many hours ago

        Returns:
            Number of evaluations cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        evaluations_to_remove = []

        for eval_id, eval_info in self.active_evaluations.items():
            if (
                eval_info["status"]
                in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED]
                and eval_info["end_time"]
                and eval_info["end_time"] < cutoff_time
            ):
                evaluations_to_remove.append(eval_id)

        async with self.evaluation_lock:
            for eval_id in evaluations_to_remove:
                del self.active_evaluations[eval_id]

        self.logger.info(f"Cleaned up {len(evaluations_to_remove)} completed evaluations")
        return len(evaluations_to_remove)

    # Private helper methods

    def _validate_evaluation_request(self, request: EvaluationRequest) -> list[ValidationError]:
        """Validate evaluation request and return any errors."""
        errors = []

        # Basic validation using request's validate method
        validation_messages = request.validate()
        for message in validation_messages:
            errors.append(ValidationError(field="request", message=message))

        # Additional API-specific validation
        if len(request.model_names) > 10:
            errors.append(
                ValidationError(
                    field="model_names",
                    message="Maximum 10 models allowed per evaluation",
                    value=len(request.model_names),
                )
            )

        if len(request.task_types) > 5:
            errors.append(
                ValidationError(
                    field="task_types",
                    message="Maximum 5 tasks allowed per evaluation",
                    value=len(request.task_types),
                )
            )

        return errors

    def _estimate_completion_time(self, request: EvaluationRequest) -> datetime | None:
        """Estimate when evaluation will complete based on request complexity."""
        base_time_per_eval = 120  # seconds

        total_evaluations = len(request.model_names) * len(request.task_types)
        if request.dataset_names:
            total_evaluations *= len(request.dataset_names)

        # Adjust based on evaluation type
        type_multipliers = {
            EvaluationType.SINGLE_MODEL: 1.0,
            EvaluationType.MODEL_COMPARISON: 1.5,
            EvaluationType.BASELINE_EVALUATION: 1.3,
            EvaluationType.COMPREHENSIVE: 2.0,
            EvaluationType.BATCH_PROCESSING: 0.8,
        }

        multiplier = type_multipliers.get(request.evaluation_type, 1.0)
        estimated_seconds = total_evaluations * base_time_per_eval * multiplier

        return datetime.now() + timedelta(seconds=estimated_seconds)

    def _generate_results_summary(
        self, results: list[EvaluationResult], request: EvaluationRequest
    ) -> dict[str, Any]:
        """Generate summary statistics for evaluation results."""
        if not results:
            return {"total_evaluations": 0, "successful_evaluations": 0}

        # Basic counts
        total_evaluations = len(results)
        model_names = list({result.model_name for result in results})
        task_types = list({result.task_type for result in results})

        # Performance summary
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, int | float):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

        metric_summaries = {}
        for metric_name, values in all_metrics.items():
            if values:
                metric_summaries[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return {
            "total_evaluations": total_evaluations,
            "successful_evaluations": total_evaluations,
            "models_evaluated": len(model_names),
            "tasks_evaluated": len(task_types),
            "evaluation_type": request.evaluation_type.value,
            "metric_summaries": metric_summaries,
            "model_names": model_names,
            "task_types": task_types,
        }

    async def _execute_evaluation(
        self, evaluation_id: EvaluationID, request: EvaluationRequest
    ) -> None:
        """Execute the evaluation asynchronously."""
        eval_info = self.active_evaluations[evaluation_id]

        try:
            eval_info["status"] = EvaluationStatus.RUNNING
            eval_info["start_time"] = datetime.now()

            # Create progress tracker
            total_steps = len(request.model_names) * len(request.task_types)
            if request.dataset_names:
                total_steps *= len(request.dataset_names)

            progress = EvaluationProgress(
                total_steps=total_steps,
                completed_steps=0,
                current_step_description="Initializing evaluation",
            )
            eval_info["progress"] = progress

            # Execute based on evaluation type
            if request.evaluation_type == EvaluationType.COMPREHENSIVE:
                workflow_config = self._convert_request_to_workflow_config(request)
                workflow_id = await self.workflow.run_comprehensive_evaluation(workflow_config)

                # Monitor workflow progress
                workflow_progress = self.workflow.track_workflow_progress(workflow_id)
                results = workflow_progress.get("results", {}).get("evaluation_results", [])
                analysis = workflow_progress.get("results", {}).get("analysis_results", {})
                comparison = workflow_progress.get("results", {}).get("comparison_results", {})

                eval_info["results"] = results
                eval_info["analysis"] = analysis
                eval_info["comparison_results"] = comparison

            elif request.evaluation_type == EvaluationType.BATCH_PROCESSING:
                batch_config = self._convert_request_to_batch_config(request)
                batch_result = await self.batch_evaluator.evaluate_batch(batch_config)

                eval_info["results"] = batch_result.results
                eval_info["performance_metrics"] = batch_result.performance_metrics

            else:
                # Handle other evaluation types with basic evaluation
                results = []
                completed = 0

                for model_name in request.model_names:
                    for task_type in request.task_types:
                        datasets = (
                            request.dataset_names if request.dataset_names else ["default_dataset"]
                        )
                        for dataset_name in datasets:
                            eval_config = {
                                "model_name": model_name,
                                "task_type": task_type,
                                "dataset_name": dataset_name,
                                **request.evaluation_config.__dict__,
                            }

                            result = await self.service.run_evaluation(eval_config)  # type: ignore[attr-defined]
                            results.append(result)

                            completed += 1
                            progress.completed_steps = completed
                            progress.current_step_description = (
                                f"Evaluating {model_name} on {task_type}"
                            )

                eval_info["results"] = results

            # Mark as completed
            eval_info["status"] = EvaluationStatus.COMPLETED
            eval_info["end_time"] = datetime.now()
            progress.completed_steps = progress.total_steps
            progress.current_step_description = "Evaluation completed"

            self.logger.info(f"Evaluation {evaluation_id} completed successfully")

        except Exception as e:
            eval_info["status"] = EvaluationStatus.FAILED
            eval_info["end_time"] = datetime.now()
            eval_info["error_message"] = str(e)

            self.logger.error(f"Evaluation {evaluation_id} failed: {e}")

    def _convert_request_to_workflow_config(self, request: EvaluationRequest) -> dict[str, Any]:
        """Convert API request to workflow configuration."""
        return {
            "models": request.model_names,
            "tasks": request.task_types,
            "datasets": request.dataset_names,
            "evaluation_params": request.evaluation_config.__dict__,
            "metadata": request.metadata,
        }

    def _convert_request_to_batch_config(self, request: EvaluationRequest) -> dict[str, Any]:
        """Convert API request to batch configuration."""
        return {
            "models": request.model_names,
            "tasks": request.task_types,
            "datasets": request.dataset_names,
            "evaluation_params": request.evaluation_config.__dict__,
        }

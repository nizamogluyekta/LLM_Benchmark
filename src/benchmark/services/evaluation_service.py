"""
Evaluation service for managing and executing model evaluation pipelines.

This service provides a plugin architecture for different evaluation metrics,
supports parallel evaluation, progress tracking, and result aggregation.
"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from benchmark.core.base import BaseService, HealthCheck, ServiceResponse, ServiceStatus
from benchmark.core.exceptions import BenchmarkError, ErrorCode
from benchmark.core.logging import get_logger
from benchmark.interfaces.evaluation_interfaces import (
    EvaluationFilter,
    EvaluationProgressCallback,
    EvaluationRequest,
    EvaluationResult,
    EvaluationSummary,
    MetricEvaluator,
    MetricType,
)


class EvaluationService(BaseService):
    """Service for evaluating model predictions using various metrics."""

    def __init__(self) -> None:
        """Initialize the evaluation service."""
        super().__init__("evaluation_service")
        self.logger = get_logger("evaluation_service")

        # Plugin registry for metric evaluators
        self.evaluators: dict[MetricType, MetricEvaluator] = {}

        # Evaluation tracking
        self.evaluation_history: list[EvaluationResult] = []
        self.active_evaluations: dict[str, EvaluationRequest] = {}

        # Progress tracking
        self.progress_callbacks: list[EvaluationProgressCallback] = []

        # Result filtering
        self.result_filters: list[EvaluationFilter] = []

        # Configuration
        self.max_concurrent_evaluations = 5
        self.evaluation_timeout_seconds = 300.0
        self.max_history_size = 1000

    async def initialize(self, config: dict[str, Any] | None = None) -> ServiceResponse:
        """
        Initialize evaluation service and register default evaluators.

        Args:
            config: Optional configuration dictionary

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            self.logger.info("Initializing evaluation service")

            # Apply configuration if provided
            if config:
                self.max_concurrent_evaluations = config.get(
                    "max_concurrent_evaluations", self.max_concurrent_evaluations
                )
                self.evaluation_timeout_seconds = config.get(
                    "evaluation_timeout_seconds", self.evaluation_timeout_seconds
                )
                self.max_history_size = config.get("max_history_size", self.max_history_size)

            # Register default evaluators
            await self._register_default_evaluators()

            self.logger.info(
                f"Evaluation service initialized with {len(self.evaluators)} evaluators"
            )

            return ServiceResponse(
                success=True,
                message="Evaluation service initialized successfully",
                data={
                    "evaluators_registered": len(self.evaluators),
                    "supported_metrics": [mt.value for mt in self.evaluators],
                    "max_concurrent_evaluations": self.max_concurrent_evaluations,
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize evaluation service: {e}")
            return ServiceResponse(
                success=False,
                message="Evaluation service initialization failed",
                error=f"Evaluation service initialization failed: {str(e)}",
            )

    async def _register_default_evaluators(self) -> None:
        """Register all available metric evaluators."""
        try:
            # Import evaluators here to avoid circular imports
            # For now, we'll create a placeholder that can be expanded later
            self.logger.info("Default evaluators registration placeholder")
            # TODO: Import and register actual evaluators when they are implemented
            # from benchmark.evaluation.metrics.accuracy import AccuracyEvaluator
            # self.evaluators[MetricType.ACCURACY] = AccuracyEvaluator()

        except Exception as e:
            self.logger.warning(f"Failed to register some default evaluators: {e}")

    async def register_evaluator(self, metric_type: MetricType, evaluator: Any) -> ServiceResponse:
        """
        Register a new metric evaluator.

        Args:
            metric_type: Type of metric this evaluator handles
            evaluator: The metric evaluator instance

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            if not isinstance(evaluator, MetricEvaluator):
                return ServiceResponse(
                    success=False,
                    message="Evaluator registration failed",
                    error="Evaluator must implement MetricEvaluator interface",
                )

            # Validate evaluator compatibility
            if evaluator.get_metric_type() != metric_type:
                return ServiceResponse(
                    success=False,
                    message="Evaluator registration failed",
                    error=f"Evaluator metric type {evaluator.get_metric_type()} does not match registration type {metric_type}",
                )

            self.evaluators[metric_type] = evaluator
            self.logger.info(f"Registered evaluator for metric type: {metric_type.value}")

            return ServiceResponse(
                success=True,
                message=f"Evaluator registered for {metric_type.value}",
                data={
                    "metric_type": metric_type.value,
                    "metric_names": evaluator.get_metric_names(),
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to register evaluator: {e}")
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def unregister_evaluator(self, metric_type: MetricType) -> ServiceResponse:
        """
        Unregister a metric evaluator.

        Args:
            metric_type: Type of metric evaluator to remove

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            if metric_type in self.evaluators:
                del self.evaluators[metric_type]
                self.logger.info(f"Unregistered evaluator for metric type: {metric_type.value}")
                return ServiceResponse(
                    success=True, message=f"Evaluator unregistered for {metric_type.value}"
                )
            else:
                return ServiceResponse(
                    success=False,
                    message="Evaluator unregistration failed",
                    error=f"No evaluator registered for {metric_type.value}",
                )

        except Exception as e:
            self.logger.error(f"Failed to unregister evaluator: {e}")
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def evaluate_predictions(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate predictions using requested metrics.

        Args:
            request: Evaluation request containing predictions and ground truth

        Returns:
            EvaluationResult with computed metrics

        Raises:
            BenchmarkError: If evaluation fails
        """
        start_time = time.time()
        experiment_id = request.experiment_id

        # Check concurrent evaluation limit
        if len(self.active_evaluations) >= self.max_concurrent_evaluations:
            raise BenchmarkError(
                f"Maximum concurrent evaluations ({self.max_concurrent_evaluations}) reached",
                ErrorCode.SERVICE_UNAVAILABLE,
            )

        # Add to active evaluations for tracking
        self.active_evaluations[experiment_id] = request

        try:
            self.logger.info(f"Starting evaluation for experiment {experiment_id}")

            # Notify progress callbacks
            for callback in self.progress_callbacks:
                await callback.on_evaluation_started(request)

            # Validate input data
            validation_result = await self._validate_evaluation_data(request)
            if not validation_result.success:
                raise BenchmarkError(
                    f"Data validation failed: {validation_result.error}",
                    ErrorCode.PREDICTION_FORMAT_ERROR,
                )

            # Prepare evaluation tasks
            evaluation_tasks = []
            for metric_type in request.metrics:
                if metric_type in self.evaluators:
                    task = self._evaluate_metric_with_timeout(
                        metric_type, request.predictions, request.ground_truth
                    )
                    evaluation_tasks.append((metric_type, task))
                else:
                    raise BenchmarkError(
                        f"Evaluator not available for metric: {metric_type.value}",
                        ErrorCode.SERVICE_UNAVAILABLE,
                    )

            # Execute all evaluations in parallel
            metric_results: dict[str, float] = {}
            detailed_results: dict[str, Any] = {}

            for metric_type, task in evaluation_tasks:
                try:
                    result = await task
                    metric_results.update(result)
                    detailed_results[metric_type.value] = result

                    # Notify progress callbacks
                    for callback in self.progress_callbacks:
                        await callback.on_metric_completed(metric_type, result)

                    self.logger.debug(
                        f"Completed {metric_type.value} evaluation: {list(result.keys())}"
                    )

                except Exception as e:
                    error_msg = f"Failed to evaluate {metric_type.value}: {str(e)}"
                    self.logger.error(error_msg)
                    detailed_results[metric_type.value] = {"error": error_msg}

            execution_time = time.time() - start_time

            # Create evaluation result
            evaluation_result = EvaluationResult(
                experiment_id=experiment_id,
                model_id=request.model_id,
                dataset_id=request.dataset_id,
                metrics=metric_results,
                detailed_results=detailed_results,
                execution_time_seconds=execution_time,
                timestamp=datetime.now().isoformat(),
                metadata=request.metadata,
                success=len(metric_results) > 0,
                error_message=None
                if len(metric_results) > 0
                else "No metrics computed successfully",
            )

            # Apply result filters
            for filter_instance in self.result_filters:
                if filter_instance.applies_to(evaluation_result):
                    evaluation_result = filter_instance.filter_result(evaluation_result)

            # Store result in history
            await self._store_evaluation_result(evaluation_result)

            # Notify progress callbacks
            for callback in self.progress_callbacks:
                await callback.on_evaluation_completed(evaluation_result)

            self.logger.info(
                f"Completed evaluation for experiment {experiment_id} in {execution_time:.2f}s"
            )

            return evaluation_result

        except Exception as e:
            self.logger.error(f"Evaluation failed for experiment {experiment_id}: {e}")

            # Notify progress callbacks of error
            for callback in self.progress_callbacks:
                await callback.on_evaluation_error(e)

            # Create error result
            error_result = EvaluationResult(
                experiment_id=experiment_id,
                model_id=request.model_id,
                dataset_id=request.dataset_id,
                metrics={},
                detailed_results={},
                execution_time_seconds=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                metadata=request.metadata,
                success=False,
                error_message=str(e),
            )

            await self._store_evaluation_result(error_result)
            raise

        finally:
            # Remove from active evaluations
            self.active_evaluations.pop(experiment_id, None)

    async def _evaluate_metric_with_timeout(
        self,
        metric_type: MetricType,
        predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Evaluate a specific metric with timeout protection."""
        try:
            evaluator = self.evaluators[metric_type]
            return await asyncio.wait_for(
                evaluator.evaluate(predictions, ground_truth),
                timeout=self.evaluation_timeout_seconds,
            )
        except TimeoutError as e:
            raise BenchmarkError(
                f"Evaluation timeout for {metric_type.value}", ErrorCode.EVALUATION_TIMEOUT
            ) from e
        except Exception as e:
            raise BenchmarkError(
                f"Evaluation error for {metric_type.value}: {str(e)}",
                ErrorCode.METRIC_CALCULATION_FAILED,
            ) from e

    async def _validate_evaluation_data(self, request: EvaluationRequest) -> ServiceResponse:
        """
        Validate evaluation request data.

        Args:
            request: Evaluation request to validate

        Returns:
            ServiceResponse indicating validation success or failure
        """
        try:
            # Check that predictions and ground truth have same length
            if len(request.predictions) != len(request.ground_truth):
                return ServiceResponse(
                    success=False,
                    message="Data validation failed",
                    error="Predictions and ground truth must have same length",
                )

            if len(request.predictions) == 0:
                return ServiceResponse(
                    success=False,
                    message="Data validation failed",
                    error="Cannot evaluate empty dataset",
                )

            # Validate that all required fields are present for requested metrics
            for metric_type in request.metrics:
                evaluator = self.evaluators.get(metric_type)
                if evaluator and not evaluator.validate_data_compatibility(
                    request.predictions, request.ground_truth
                ):
                    return ServiceResponse(
                        success=False,
                        message="Data validation failed",
                        error=f"Data incompatible with {metric_type.value} evaluator",
                    )

            return ServiceResponse(success=True, message="Data validation successful")

        except Exception as e:
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def _store_evaluation_result(self, result: EvaluationResult) -> None:
        """Store evaluation result in history with size management."""
        self.evaluation_history.append(result)

        # Manage history size
        if len(self.evaluation_history) > self.max_history_size:
            # Remove oldest results
            excess_count = len(self.evaluation_history) - self.max_history_size
            self.evaluation_history = self.evaluation_history[excess_count:]
            self.logger.debug(f"Trimmed evaluation history, removed {excess_count} old results")

    async def get_available_metrics(self) -> ServiceResponse:
        """
        Get list of available evaluation metrics.

        Returns:
            ServiceResponse with metrics information
        """
        try:
            metrics_info = {}
            for metric_type, evaluator in self.evaluators.items():
                metrics_info[metric_type.value] = evaluator.get_evaluator_info()

            return ServiceResponse(
                success=True,
                message="Available metrics retrieved",
                data={
                    "metrics": metrics_info,
                    "total_evaluators": len(self.evaluators),
                    "supported_metric_types": [mt.value for mt in self.evaluators],
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to get available metrics: {e}")
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def get_evaluation_history(
        self,
        experiment_id: str | None = None,
        model_id: str | None = None,
        dataset_id: str | None = None,
        limit: int = 100,
    ) -> ServiceResponse:
        """
        Get evaluation history with optional filtering.

        Args:
            experiment_id: Filter by experiment ID
            model_id: Filter by model ID
            dataset_id: Filter by dataset ID
            limit: Maximum number of results to return

        Returns:
            ServiceResponse with filtered evaluation history
        """
        try:
            filtered_history = self.evaluation_history

            # Apply filters
            if experiment_id:
                filtered_history = [
                    result for result in filtered_history if result.experiment_id == experiment_id
                ]

            if model_id:
                filtered_history = [
                    result for result in filtered_history if result.model_id == model_id
                ]

            if dataset_id:
                filtered_history = [
                    result for result in filtered_history if result.dataset_id == dataset_id
                ]

            # Limit results (most recent first)
            limited_history = list(reversed(filtered_history))[:limit]

            return ServiceResponse(
                success=True,
                message="Evaluation history retrieved",
                data={
                    "results": limited_history,
                    "total_results": len(filtered_history),
                    "returned_results": len(limited_history),
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to get evaluation history: {e}")
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def get_evaluation_summary(self, days_back: int = 7) -> ServiceResponse:
        """
        Get evaluation summary statistics.

        Args:
            days_back: Number of days to include in summary

        Returns:
            ServiceResponse with evaluation summary
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_back * 24 * 3600)

            # Filter results by time
            recent_results = [
                result
                for result in self.evaluation_history
                if datetime.fromisoformat(result.timestamp).timestamp() > cutoff_time
            ]

            if not recent_results:
                summary = EvaluationSummary(
                    total_evaluations=0,
                    successful_evaluations=0,
                    failed_evaluations=0,
                    average_execution_time=0.0,
                    metric_summaries={},
                    time_range={},
                    models_evaluated=[],
                    datasets_evaluated=[],
                )
                return ServiceResponse(
                    success=True,
                    message="Empty evaluation summary generated",
                    data=summary.__dict__,
                )

            # Calculate summary statistics
            successful_results = [r for r in recent_results if r.success]
            failed_results = [r for r in recent_results if not r.success]

            # Aggregate metrics
            metric_summaries = defaultdict(list)
            for result in successful_results:
                for metric_name, value in result.metrics.items():
                    metric_summaries[metric_name].append(value)

            # Calculate metric statistics
            metric_stats = {}
            for metric_name, values in metric_summaries.items():
                if values:
                    metric_stats[metric_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }

            # Get unique models and datasets
            models_evaluated = list({result.model_id for result in recent_results})
            datasets_evaluated = list({result.dataset_id for result in recent_results})

            # Time range
            timestamps = [datetime.fromisoformat(r.timestamp) for r in recent_results]
            time_range = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
            }

            summary = EvaluationSummary(
                total_evaluations=len(recent_results),
                successful_evaluations=len(successful_results),
                failed_evaluations=len(failed_results),
                average_execution_time=sum(r.execution_time_seconds for r in recent_results)
                / len(recent_results),
                metric_summaries=metric_stats,
                time_range=time_range,
                models_evaluated=models_evaluated,
                datasets_evaluated=datasets_evaluated,
            )

            return ServiceResponse(
                success=True, message="Evaluation summary generated", data=summary.__dict__
            )

        except Exception as e:
            self.logger.error(f"Failed to get evaluation summary: {e}")
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def add_progress_callback(self, callback: EvaluationProgressCallback) -> ServiceResponse:
        """Add a progress callback for evaluation tracking."""
        try:
            self.progress_callbacks.append(callback)
            return ServiceResponse(success=True, message="Progress callback added")
        except Exception as e:
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def remove_progress_callback(
        self, callback: EvaluationProgressCallback
    ) -> ServiceResponse:
        """Remove a progress callback."""
        try:
            if callback in self.progress_callbacks:
                self.progress_callbacks.remove(callback)
                return ServiceResponse(success=True, message="Progress callback removed")
            else:
                return ServiceResponse(
                    success=False, message="Callback not found", error="Callback not found"
                )
        except Exception as e:
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

    async def health_check(self) -> HealthCheck:
        """
        Check evaluation service health.

        Returns:
            HealthCheck with service status and details
        """
        try:
            status = ServiceStatus.HEALTHY
            details: dict[str, Any] = {
                "evaluators_count": len(self.evaluators),
                "active_evaluations": len(self.active_evaluations),
                "total_evaluations_completed": len(self.evaluation_history),
                "progress_callbacks": len(self.progress_callbacks),
                "result_filters": len(self.result_filters),
            }

            # Check if any evaluators are failing
            if len(self.evaluators) == 0:
                status = ServiceStatus.UNHEALTHY
                details["issue"] = "No evaluators registered"
            elif len(self.active_evaluations) >= self.max_concurrent_evaluations:
                status = ServiceStatus.DEGRADED
                details["issue"] = "Maximum concurrent evaluations reached"

            return HealthCheck(
                status=status,
                message="Evaluation service health check",
                checks=details,
            )

        except Exception as e:
            return HealthCheck(
                status=ServiceStatus.UNHEALTHY,
                message=f"Evaluation service error: {str(e)}",
                checks={"error": str(e)},
            )

    async def shutdown(self) -> ServiceResponse:
        """
        Graceful shutdown of evaluation service.

        Returns:
            ServiceResponse indicating shutdown status
        """
        try:
            self.logger.info("Shutting down evaluation service")

            # Wait for active evaluations to complete (with timeout)
            if self.active_evaluations:
                self.logger.info(f"Waiting for {len(self.active_evaluations)} active evaluations")
                await asyncio.sleep(1.0)  # Give active evaluations time to complete

            # Clear all data
            self.evaluators.clear()
            self.evaluation_history.clear()
            self.active_evaluations.clear()
            self.progress_callbacks.clear()
            self.result_filters.clear()

            self.logger.info("Evaluation service shut down successfully")

            return ServiceResponse(
                success=True, message="Evaluation service shut down successfully"
            )

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return ServiceResponse(success=False, message="Operation failed", error=str(e))

"""
Comprehensive evaluation workflows that coordinate all evaluation components.

This module provides high-level workflow orchestration for complex evaluation scenarios,
including multi-model comparisons, baseline evaluations, and comprehensive assessments.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..services.evaluation_service import EvaluationService
from .comparison_engine import ModelComparisonEngine
from .result_models import EvaluationResult
from .results_analyzer import ResultsAnalyzer


class WorkflowState(Enum):
    """Workflow execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStep(Enum):
    """Individual workflow step types."""

    INITIALIZATION = "initialization"
    MODEL_LOADING = "model_loading"
    DATA_PREPARATION = "data_preparation"
    EVALUATION_EXECUTION = "evaluation_execution"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    REPORTING = "reporting"
    CLEANUP = "cleanup"


class WorkflowProgress:
    """Track workflow execution progress."""

    def __init__(self, workflow_id: str, total_steps: int):
        self.workflow_id = workflow_id
        self.total_steps = total_steps
        self.completed_steps = 0
        self.current_step: WorkflowStep | None = None
        self.start_time = datetime.now()
        self.end_time: datetime | None = None
        self.state = WorkflowState.PENDING
        self.step_progress: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.results: dict[str, Any] = {}

    def start_step(self, step: WorkflowStep, substeps: int = 1) -> None:
        """Start a workflow step."""
        self.current_step = step
        self.step_progress[step.value] = {
            "status": "running",
            "start_time": datetime.now(),
            "substeps_total": substeps,
            "substeps_completed": 0,
            "errors": [],
        }

    def update_step_progress(self, substeps_completed: int = 1) -> None:
        """Update progress within current step."""
        if self.current_step:
            progress = self.step_progress[self.current_step.value]
            progress["substeps_completed"] += substeps_completed

    def complete_step(self, step: WorkflowStep, result: Any = None) -> None:
        """Complete a workflow step."""
        if step.value in self.step_progress:
            self.step_progress[step.value]["status"] = "completed"
            self.step_progress[step.value]["end_time"] = datetime.now()
            if result is not None:
                self.step_progress[step.value]["result"] = result
        self.completed_steps += 1

    def fail_step(self, step: WorkflowStep, error: str) -> None:
        """Mark a step as failed."""
        if step.value in self.step_progress:
            self.step_progress[step.value]["status"] = "failed"
            self.step_progress[step.value]["end_time"] = datetime.now()
            self.step_progress[step.value]["errors"].append(error)
        self.errors.append(f"{step.value}: {error}")
        self.state = WorkflowState.FAILED

    def get_progress_percentage(self) -> float:
        """Get overall progress percentage."""
        if self.total_steps == 0:
            return 100.0
        return (self.completed_steps / self.total_steps) * 100.0

    def get_estimated_completion(self) -> datetime | None:
        """Estimate completion time based on current progress."""
        if self.completed_steps == 0:
            return None

        elapsed = datetime.now() - self.start_time
        estimated_total = elapsed * (self.total_steps / self.completed_steps)
        return self.start_time + estimated_total


class EvaluationWorkflow:
    """
    Comprehensive evaluation workflow orchestration.

    Coordinates all evaluation components to run complex evaluation scenarios
    including multi-model comparisons, baseline evaluations, and comprehensive assessments.
    """

    def __init__(self, evaluation_service: EvaluationService):
        """
        Initialize evaluation workflow.

        Args:
            evaluation_service: Service for running individual evaluations
        """
        self.service = evaluation_service
        self.workflow_states: dict[str, WorkflowProgress] = {}
        self.logger = logging.getLogger(__name__)
        self.analyzer = ResultsAnalyzer(evaluation_service.storage)  # type: ignore[attr-defined]
        self.comparison_engine = ModelComparisonEngine(evaluation_service.storage)  # type: ignore[attr-defined]

    async def run_comprehensive_evaluation(
        self, config: dict[str, Any], workflow_id: str | None = None
    ) -> str:
        """
        Run complete evaluation across multiple models and tasks.

        Args:
            config: Evaluation configuration including models, tasks, datasets
            workflow_id: Optional workflow ID, generated if not provided

        Returns:
            Workflow ID for tracking progress
        """
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        # Initialize workflow progress
        total_steps = self._calculate_total_steps(config, "comprehensive")
        progress = WorkflowProgress(workflow_id, total_steps)
        self.workflow_states[workflow_id] = progress

        try:
            progress.state = WorkflowState.RUNNING
            self.logger.info(f"Starting comprehensive evaluation workflow {workflow_id}")

            # Step 1: Initialization and validation
            progress.start_step(WorkflowStep.INITIALIZATION)
            await self._validate_workflow_config(config)
            progress.complete_step(WorkflowStep.INITIALIZATION)

            # Step 2: Model loading and preparation
            progress.start_step(WorkflowStep.MODEL_LOADING, len(config.get("models", [])))
            models_info = await self._prepare_models(config.get("models", []), progress)
            progress.complete_step(WorkflowStep.MODEL_LOADING, models_info)

            # Step 3: Data preparation
            progress.start_step(WorkflowStep.DATA_PREPARATION, len(config.get("tasks", [])))
            data_info = await self._prepare_datasets(config.get("tasks", []), progress)
            progress.complete_step(WorkflowStep.DATA_PREPARATION, data_info)

            # Step 4: Evaluation execution
            total_evaluations = len(config.get("models", [])) * len(config.get("tasks", []))
            progress.start_step(WorkflowStep.EVALUATION_EXECUTION, total_evaluations)
            evaluation_results = await self._run_evaluations(config, progress)
            progress.complete_step(WorkflowStep.EVALUATION_EXECUTION, len(evaluation_results))

            # Step 5: Analysis
            progress.start_step(WorkflowStep.ANALYSIS)
            analysis_results = await self._perform_analysis(evaluation_results, config, progress)
            progress.complete_step(WorkflowStep.ANALYSIS, analysis_results)

            # Step 6: Model comparison
            progress.start_step(WorkflowStep.COMPARISON)
            comparison_results = await self._perform_comparisons(config, progress)
            progress.complete_step(WorkflowStep.COMPARISON, comparison_results)

            # Step 7: Report generation
            progress.start_step(WorkflowStep.REPORTING)
            report = await self._generate_comprehensive_report(
                evaluation_results, analysis_results, comparison_results, config
            )
            progress.complete_step(WorkflowStep.REPORTING, report)

            # Step 8: Cleanup
            progress.start_step(WorkflowStep.CLEANUP)
            await self._cleanup_workflow_resources(workflow_id)
            progress.complete_step(WorkflowStep.CLEANUP)

            progress.state = WorkflowState.COMPLETED
            progress.end_time = datetime.now()
            progress.results = {
                "evaluation_results": len(evaluation_results),
                "analysis_results": analysis_results,
                "comparison_results": comparison_results,
                "report": report,
            }

            self.logger.info(
                f"Comprehensive evaluation workflow {workflow_id} completed successfully"
            )

        except Exception as e:
            error_msg = f"Workflow failed: {str(e)}"
            self.logger.error(f"Workflow {workflow_id} failed: {error_msg}")
            if progress.current_step:
                progress.fail_step(progress.current_step, error_msg)
            progress.state = WorkflowState.FAILED
            progress.end_time = datetime.now()
            raise

        return workflow_id

    async def run_model_comparison(
        self,
        models: list[str],
        tasks: list[str],
        config: dict[str, Any] | None = None,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run comparative evaluation across specified models and tasks.

        Args:
            models: List of model names to compare
            tasks: List of task types to evaluate on
            config: Optional additional configuration
            workflow_id: Optional workflow ID

        Returns:
            Comparison results with detailed analysis
        """
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        if config is None:
            config = {}

        # Prepare comparison configuration
        comparison_config = {"models": models, "tasks": tasks, "comparison_mode": True, **config}

        total_steps = 6  # Simplified workflow for comparison
        progress = WorkflowProgress(workflow_id, total_steps)
        self.workflow_states[workflow_id] = progress

        try:
            progress.state = WorkflowState.RUNNING
            self.logger.info(f"Starting model comparison workflow {workflow_id}")

            # Step 1: Validation
            progress.start_step(WorkflowStep.INITIALIZATION)
            await self._validate_comparison_config(models, tasks, config)
            progress.complete_step(WorkflowStep.INITIALIZATION)

            # Step 2: Model preparation
            progress.start_step(WorkflowStep.MODEL_LOADING, len(models))
            models_info = await self._prepare_models(models, progress)
            progress.complete_step(WorkflowStep.MODEL_LOADING, models_info)

            # Step 3: Evaluation execution
            total_evaluations = len(models) * len(tasks)
            progress.start_step(WorkflowStep.EVALUATION_EXECUTION, total_evaluations)
            evaluation_results = await self._run_evaluations(comparison_config, progress)
            progress.complete_step(WorkflowStep.EVALUATION_EXECUTION, len(evaluation_results))

            # Step 4: Statistical comparison
            progress.start_step(WorkflowStep.COMPARISON)
            comparison_results = await self._perform_statistical_comparison(
                models, tasks, config.get("metrics", ["accuracy"]), progress
            )
            progress.complete_step(WorkflowStep.COMPARISON, comparison_results)

            # Step 5: Analysis
            progress.start_step(WorkflowStep.ANALYSIS)
            analysis_summary = await self._analyze_comparison_results(
                comparison_results, evaluation_results, progress
            )
            progress.complete_step(WorkflowStep.ANALYSIS, analysis_summary)

            # Step 6: Cleanup
            progress.start_step(WorkflowStep.CLEANUP)
            await self._cleanup_workflow_resources(workflow_id)
            progress.complete_step(WorkflowStep.CLEANUP)

            progress.state = WorkflowState.COMPLETED
            progress.end_time = datetime.now()

            final_results = {
                "workflow_id": workflow_id,
                "models_compared": models,
                "tasks_evaluated": tasks,
                "comparison_results": comparison_results,
                "analysis_summary": analysis_summary,
                "evaluation_count": len(evaluation_results),
                "completion_time": progress.end_time.isoformat(),
            }

            progress.results = final_results

            self.logger.info(f"Model comparison workflow {workflow_id} completed successfully")
            return final_results

        except Exception as e:
            error_msg = f"Comparison workflow failed: {str(e)}"
            self.logger.error(f"Workflow {workflow_id} failed: {error_msg}")
            if progress.current_step:
                progress.fail_step(progress.current_step, error_msg)
            progress.state = WorkflowState.FAILED
            progress.end_time = datetime.now()
            raise

    async def run_baseline_evaluation(
        self,
        new_model: str,
        baseline_models: list[str] | None = None,
        config: dict[str, Any] | None = None,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run evaluation against established baselines.

        Args:
            new_model: Name of the new model to evaluate
            baseline_models: List of baseline model names, auto-detected if None
            config: Optional configuration for evaluation
            workflow_id: Optional workflow ID

        Returns:
            Baseline comparison results
        """
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        if config is None:
            config = {}

        total_steps = 7
        progress = WorkflowProgress(workflow_id, total_steps)
        self.workflow_states[workflow_id] = progress

        try:
            progress.state = WorkflowState.RUNNING
            self.logger.info(f"Starting baseline evaluation workflow {workflow_id}")

            # Step 1: Initialize and detect baselines
            progress.start_step(WorkflowStep.INITIALIZATION)
            if baseline_models is None:
                baseline_models = await self._detect_baseline_models(config)
            baseline_config = await self._prepare_baseline_config(
                new_model, baseline_models, config
            )
            progress.complete_step(WorkflowStep.INITIALIZATION, baseline_config)

            # Step 2: Model preparation
            all_models = [new_model] + baseline_models
            progress.start_step(WorkflowStep.MODEL_LOADING, len(all_models))
            models_info = await self._prepare_models(all_models, progress)
            progress.complete_step(WorkflowStep.MODEL_LOADING, models_info)

            # Step 3: Data preparation
            tasks = baseline_config.get("tasks", [])
            progress.start_step(WorkflowStep.DATA_PREPARATION, len(tasks))
            data_info = await self._prepare_datasets(tasks, progress)
            progress.complete_step(WorkflowStep.DATA_PREPARATION, data_info)

            # Step 4: Evaluation execution
            total_evaluations = len(all_models) * len(tasks)
            progress.start_step(WorkflowStep.EVALUATION_EXECUTION, total_evaluations)
            evaluation_results = await self._run_evaluations(baseline_config, progress)
            progress.complete_step(WorkflowStep.EVALUATION_EXECUTION, len(evaluation_results))

            # Step 5: Baseline comparison
            progress.start_step(WorkflowStep.COMPARISON)
            baseline_comparison = await self._perform_baseline_comparison(
                new_model, baseline_models, evaluation_results, progress
            )
            progress.complete_step(WorkflowStep.COMPARISON, baseline_comparison)

            # Step 6: Analysis and recommendations
            progress.start_step(WorkflowStep.ANALYSIS)
            analysis_results = await self._analyze_baseline_performance(
                new_model, baseline_comparison, evaluation_results, progress
            )
            progress.complete_step(WorkflowStep.ANALYSIS, analysis_results)

            # Step 7: Cleanup
            progress.start_step(WorkflowStep.CLEANUP)
            await self._cleanup_workflow_resources(workflow_id)
            progress.complete_step(WorkflowStep.CLEANUP)

            progress.state = WorkflowState.COMPLETED
            progress.end_time = datetime.now()

            final_results = {
                "workflow_id": workflow_id,
                "new_model": new_model,
                "baseline_models": baseline_models,
                "baseline_comparison": baseline_comparison,
                "analysis_results": analysis_results,
                "evaluation_count": len(evaluation_results),
                "completion_time": progress.end_time.isoformat(),
                "performance_summary": {
                    "beats_baselines": baseline_comparison.get("outperforms_baselines", 0),
                    "total_baselines": len(baseline_models),
                    "improvement_percentage": baseline_comparison.get("avg_improvement", 0.0),
                },
            }

            progress.results = final_results

            self.logger.info(f"Baseline evaluation workflow {workflow_id} completed successfully")
            return final_results

        except Exception as e:
            error_msg = f"Baseline evaluation workflow failed: {str(e)}"
            self.logger.error(f"Workflow {workflow_id} failed: {error_msg}")
            if progress.current_step:
                progress.fail_step(progress.current_step, error_msg)
            progress.state = WorkflowState.FAILED
            progress.end_time = datetime.now()
            raise

    def track_workflow_progress(self, workflow_id: str) -> dict[str, Any]:
        """
        Track progress of running evaluation workflow.

        Args:
            workflow_id: ID of the workflow to track

        Returns:
            Detailed progress information
        """
        if workflow_id not in self.workflow_states:
            return {"error": f"Workflow {workflow_id} not found"}

        progress = self.workflow_states[workflow_id]

        return {
            "workflow_id": workflow_id,
            "state": progress.state.value,
            "progress_percentage": progress.get_progress_percentage(),
            "completed_steps": progress.completed_steps,
            "total_steps": progress.total_steps,
            "current_step": progress.current_step.value if progress.current_step else None,
            "start_time": progress.start_time.isoformat(),
            "end_time": progress.end_time.isoformat() if progress.end_time else None,
            "estimated_completion": (
                estimated_completion.isoformat()
                if (estimated_completion := progress.get_estimated_completion())
                else None
            ),
            "step_progress": progress.step_progress,
            "errors": progress.errors,
            "results": progress.results if progress.state == WorkflowState.COMPLETED else {},
        }

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel

        Returns:
            True if successfully cancelled, False otherwise
        """
        if workflow_id not in self.workflow_states:
            return False

        progress = self.workflow_states[workflow_id]
        if progress.state not in [WorkflowState.RUNNING, WorkflowState.PAUSED]:
            return False

        progress.state = WorkflowState.CANCELLED
        progress.end_time = datetime.now()

        self.logger.info(f"Workflow {workflow_id} cancelled")
        return True

    def pause_workflow(self, workflow_id: str) -> bool:
        """
        Pause a running workflow.

        Args:
            workflow_id: ID of the workflow to pause

        Returns:
            True if successfully paused, False otherwise
        """
        if workflow_id not in self.workflow_states:
            return False

        progress = self.workflow_states[workflow_id]
        if progress.state != WorkflowState.RUNNING:
            return False

        progress.state = WorkflowState.PAUSED
        self.logger.info(f"Workflow {workflow_id} paused")
        return True

    def resume_workflow(self, workflow_id: str) -> bool:
        """
        Resume a paused workflow.

        Args:
            workflow_id: ID of the workflow to resume

        Returns:
            True if successfully resumed, False otherwise
        """
        if workflow_id not in self.workflow_states:
            return False

        progress = self.workflow_states[workflow_id]
        if progress.state != WorkflowState.PAUSED:
            return False

        progress.state = WorkflowState.RUNNING
        self.logger.info(f"Workflow {workflow_id} resumed")
        return True

    def list_workflows(
        self, state_filter: WorkflowState | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        List workflows with optional filtering.

        Args:
            state_filter: Optional state to filter by
            limit: Maximum number of workflows to return

        Returns:
            List of workflow summaries
        """
        workflows = []

        for workflow_id, progress in self.workflow_states.items():
            if state_filter and progress.state != state_filter:
                continue

            workflows.append(
                {
                    "workflow_id": workflow_id,
                    "state": progress.state.value,
                    "progress_percentage": progress.get_progress_percentage(),
                    "start_time": progress.start_time.isoformat(),
                    "end_time": progress.end_time.isoformat() if progress.end_time else None,
                    "current_step": progress.current_step.value if progress.current_step else None,
                    "error_count": len(progress.errors),
                }
            )

        # Sort by start time, most recent first
        workflows.sort(key=lambda x: str(x["start_time"]), reverse=True)

        return workflows[:limit]

    def cleanup_completed_workflows(self, older_than_hours: int = 24) -> int:
        """
        Clean up completed workflow states older than specified time.

        Args:
            older_than_hours: Remove workflows completed more than this many hours ago

        Returns:
            Number of workflows cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        workflows_to_remove = []

        for workflow_id, progress in self.workflow_states.items():
            if (
                progress.state
                in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED]
                and progress.end_time
                and progress.end_time < cutoff_time
            ):
                workflows_to_remove.append(workflow_id)

        for workflow_id in workflows_to_remove:
            del self.workflow_states[workflow_id]

        self.logger.info(f"Cleaned up {len(workflows_to_remove)} completed workflows")
        return len(workflows_to_remove)

    # Private helper methods

    def _calculate_total_steps(self, config: dict[str, Any], workflow_type: str) -> int:
        """Calculate total number of steps for workflow."""
        step_counts = {
            "comprehensive": 8,  # All workflow steps
            "comparison": 6,  # Simplified comparison workflow
            "baseline": 7,  # Baseline evaluation steps
        }
        return step_counts.get(workflow_type, 5)  # Default fallback

    async def _validate_workflow_config(self, config: dict[str, Any]) -> None:
        """Validate workflow configuration."""
        required_fields = ["models", "tasks"]
        for field in required_fields:
            if field not in config or not config[field]:
                raise ValueError(f"Missing required field: {field}")

        # Validate models exist
        available_models = await self.service.list_available_models()  # type: ignore[attr-defined]
        for model in config["models"]:
            if model not in available_models:
                raise ValueError(f"Model {model} not available")

    async def _validate_comparison_config(
        self, models: list[str], tasks: list[str], config: dict[str, Any]
    ) -> None:
        """Validate model comparison configuration."""
        if len(models) < 2:
            raise ValueError("Need at least 2 models for comparison")
        if not tasks:
            raise ValueError("Need at least 1 task for comparison")

        # Validate models exist
        available_models = await self.service.list_available_models()  # type: ignore[attr-defined]
        for model in models:
            if model not in available_models:
                raise ValueError(f"Model {model} not available")

    async def _prepare_models(
        self, models: list[str], progress: WorkflowProgress
    ) -> dict[str, Any]:
        """Prepare and validate models for evaluation."""
        models_info = {}

        for model in models:
            try:
                # Load and validate model
                model_info = await self.service.prepare_model(model)  # type: ignore[attr-defined]
                models_info[model] = model_info
                progress.update_step_progress()

            except Exception as e:
                error_msg = f"Failed to prepare model {model}: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg) from e

        return models_info

    async def _prepare_datasets(
        self, tasks: list[str], progress: WorkflowProgress
    ) -> dict[str, Any]:
        """Prepare datasets for evaluation tasks."""
        data_info = {}

        for task in tasks:
            try:
                # Prepare dataset for task
                task_data = await self.service.prepare_task_data(task)  # type: ignore[attr-defined]
                data_info[task] = task_data
                progress.update_step_progress()

            except Exception as e:
                error_msg = f"Failed to prepare data for task {task}: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg) from e

        return data_info

    async def _run_evaluations(
        self, config: dict[str, Any], progress: WorkflowProgress
    ) -> list[EvaluationResult]:
        """Run all evaluations according to configuration."""
        evaluation_results = []

        models = config.get("models", [])
        tasks = config.get("tasks", [])

        for model in models:
            for task in tasks:
                try:
                    # Run evaluation for this model-task combination
                    eval_config = {
                        "model_name": model,
                        "task_type": task,
                        **config.get("evaluation_params", {}),
                    }

                    result = await self.service.run_evaluation(eval_config)  # type: ignore[attr-defined]
                    evaluation_results.append(result)
                    progress.update_step_progress()

                except Exception as e:
                    error_msg = f"Evaluation failed for {model} on {task}: {str(e)}"
                    self.logger.error(error_msg)
                    # Continue with other evaluations
                    progress.update_step_progress()

        return evaluation_results

    async def _perform_analysis(
        self,
        evaluation_results: list[EvaluationResult],
        config: dict[str, Any],
        progress: WorkflowProgress,
    ) -> dict[str, Any]:
        """Perform comprehensive analysis of evaluation results."""
        analysis_results = {}

        # Group results by model
        model_results: dict[str, list[EvaluationResult]] = {}
        for result in evaluation_results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)

        # Analyze each model
        for model_name, _results in model_results.items():
            model_analysis = self.analyzer.analyze_model_performance(model_name)
            trends = self.analyzer.identify_performance_trends(model_name)
            analysis_results[model_name] = {
                "performance_analysis": model_analysis,
                "trends": trends,
            }

        # Overall analysis
        overall_summary = self.analyzer.generate_performance_summary(evaluation_results)
        bottlenecks = self.analyzer.find_performance_bottlenecks(evaluation_results)

        analysis_results["overall"] = {"summary": overall_summary, "bottlenecks": bottlenecks}

        return analysis_results

    async def _perform_comparisons(
        self, config: dict[str, Any], progress: WorkflowProgress
    ) -> dict[str, Any]:
        """Perform model comparisons."""
        models = config.get("models", [])
        metrics = config.get("metrics", ["accuracy"])

        comparison_results = {}

        if len(models) >= 2:
            # Multi-model comparison
            comparison = self.comparison_engine.compare_models(
                model_names=models, metric_name=metrics[0] if metrics else "accuracy"
            )
            comparison_results["multi_model"] = comparison

            # Pairwise comparisons for statistical significance
            pairwise_results = {}
            for i, model_a in enumerate(models):
                for model_b in models[i + 1 :]:
                    results_a = self.service.storage.query_results({"model_name": model_a})  # type: ignore[attr-defined]
                    results_b = self.service.storage.query_results({"model_name": model_b})  # type: ignore[attr-defined]

                    if results_a and results_b:
                        significance = self.comparison_engine.statistical_significance_test(
                            results_a, results_b, metrics[0] if metrics else "accuracy"
                        )
                        pairwise_results[f"{model_a}_vs_{model_b}"] = significance

            comparison_results["pairwise"] = pairwise_results

            # Model ranking
            if len(metrics) > 1:
                ranking = self.comparison_engine.rank_models(model_names=models, metrics=metrics)
                comparison_results["ranking"] = ranking

        return comparison_results

    async def _perform_statistical_comparison(
        self, models: list[str], tasks: list[str], metrics: list[str], progress: WorkflowProgress
    ) -> dict[str, Any]:
        """Perform statistical comparison between models."""
        comparison_results = {}

        # Overall comparison
        overall_comparison = self.comparison_engine.compare_models(
            model_names=models, metric_name=metrics[0] if metrics else "accuracy"
        )
        comparison_results["overall"] = overall_comparison

        # Task-specific comparisons
        task_comparisons = {}
        for task in tasks:
            task_comparison = self.comparison_engine.compare_models(
                model_names=models,
                task_types=[task],
                metric_name=metrics[0] if metrics else "accuracy",
            )
            task_comparisons[task] = task_comparison

        comparison_results["by_task"] = task_comparisons

        # Multi-metric ranking if multiple metrics provided
        if len(metrics) > 1:
            ranking = self.comparison_engine.rank_models(model_names=models, metrics=metrics)
            comparison_results["multi_metric_ranking"] = ranking

        return comparison_results

    async def _analyze_comparison_results(
        self,
        comparison_results: dict[str, Any],
        evaluation_results: list[EvaluationResult],
        progress: WorkflowProgress,
    ) -> dict[str, Any]:
        """Analyze comparison results to generate insights."""
        comparison_insights: list[str] = []

        analysis_summary = {
            "total_evaluations": len(evaluation_results),
            "models_compared": len({r.model_name for r in evaluation_results}),
            "tasks_evaluated": len({r.task_type for r in evaluation_results}),
            "comparison_insights": comparison_insights,
        }

        # Extract insights from overall comparison
        overall = comparison_results.get("overall", {})
        if "ranking_analysis" in overall:
            rankings = overall["ranking_analysis"].get("rankings", [])
            if rankings:
                best_model = rankings[0]
                worst_model = rankings[-1]
                performance_gap = overall["ranking_analysis"].get("performance_gap", 0)

                comparison_insights.extend(
                    [
                        f"Best performing model: {best_model.get('model_name', 'Unknown')} "
                        f"(avg: {best_model.get('mean_performance', 0):.3f})",
                        f"Lowest performing model: {worst_model.get('model_name', 'Unknown')} "
                        f"(avg: {worst_model.get('mean_performance', 0):.3f})",
                        f"Performance gap: {performance_gap:.3f}",
                    ]
                )

        # Statistical significance insights
        if "statistical_comparison" in overall:
            stats = overall["statistical_comparison"]
            if stats.get("anova", {}).get("is_significant", False):
                comparison_insights.append(
                    "Statistically significant differences detected between models"
                )

        return analysis_summary

    async def _detect_baseline_models(self, config: dict[str, Any]) -> list[str]:
        """Detect suitable baseline models from historical evaluations."""
        # Get recent evaluation results to find established models
        recent_results = self.service.storage.query_results(  # type: ignore[attr-defined]
            {"start_date": datetime.now() - timedelta(days=30)}
        )

        # Count evaluations per model
        model_counts: dict[str, int] = {}
        for result in recent_results:
            model_counts[result.model_name] = model_counts.get(result.model_name, 0) + 1

        # Select models with sufficient evaluation history
        baseline_candidates = [
            model
            for model, count in model_counts.items()
            if count >= config.get("min_baseline_evaluations", 5)
        ]

        # Sort by evaluation count and take top models
        baseline_candidates.sort(key=lambda m: model_counts[m], reverse=True)

        return baseline_candidates[: config.get("max_baselines", 3)]

    async def _prepare_baseline_config(
        self, new_model: str, baseline_models: list[str], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare configuration for baseline evaluation."""
        all_models = [new_model] + baseline_models

        # Use standard benchmark tasks if not specified
        tasks = config.get(
            "tasks", ["text_classification", "sentiment_analysis", "question_answering"]
        )

        baseline_config = {
            "models": all_models,
            "tasks": tasks,
            "baseline_mode": True,
            "new_model": new_model,
            "baseline_models": baseline_models,
            **config,
        }

        return baseline_config

    async def _perform_baseline_comparison(
        self,
        new_model: str,
        baseline_models: list[str],
        evaluation_results: list[EvaluationResult],
        progress: WorkflowProgress,
    ) -> dict[str, Any]:
        """Compare new model against baselines."""
        baseline_comparison: dict[str, Any] = {
            "new_model": new_model,
            "baseline_models": baseline_models,
            "comparisons": {},
            "summary": {},
        }

        # Compare against each baseline
        new_model_results = [r for r in evaluation_results if r.model_name == new_model]

        improvements = []
        significance_tests = []

        for baseline_model in baseline_models:
            baseline_results = [r for r in evaluation_results if r.model_name == baseline_model]

            if new_model_results and baseline_results:
                # Statistical comparison
                comparison = self.comparison_engine.statistical_significance_test(
                    new_model_results, baseline_results, "accuracy"
                )

                baseline_comparison["comparisons"][baseline_model] = comparison

                # Calculate improvement
                if "descriptive_statistics" in comparison:
                    new_mean = comparison["descriptive_statistics"]["model_a"]["mean"]
                    baseline_mean = comparison["descriptive_statistics"]["model_b"]["mean"]
                    improvement = ((new_mean - baseline_mean) / baseline_mean) * 100
                    improvements.append(improvement)

                    if comparison["test_results"]["is_significant"]:
                        significance_tests.append(baseline_model)

        # Summary statistics
        baseline_comparison["summary"] = {
            "outperforms_baselines": len([imp for imp in improvements if imp > 0]),
            "total_baselines": len(baseline_models),
            "avg_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
            "significant_improvements": len(significance_tests),
            "significantly_improved_baselines": significance_tests,
        }

        return baseline_comparison

    async def _analyze_baseline_performance(
        self,
        new_model: str,
        baseline_comparison: dict[str, Any],
        evaluation_results: list[EvaluationResult],
        progress: WorkflowProgress,
    ) -> dict[str, Any]:
        """Analyze baseline performance and generate recommendations."""
        summary = baseline_comparison["summary"]

        recommendations: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        analysis_results = {
            "performance_verdict": "unknown",
            "recommendations": recommendations,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "detailed_analysis": {},
        }

        # Determine overall performance verdict
        if summary["avg_improvement"] > 5.0:
            analysis_results["performance_verdict"] = "excellent"
            recommendations.append(
                f"{new_model} shows excellent performance, significantly outperforming baselines"
            )
        elif summary["avg_improvement"] > 0:
            analysis_results["performance_verdict"] = "good"
            recommendations.append(
                f"{new_model} shows improvements over baselines but with room for optimization"
            )
        elif summary["avg_improvement"] > -5.0:
            analysis_results["performance_verdict"] = "comparable"
            recommendations.append(
                f"{new_model} performs comparably to baselines - consider targeted improvements"
            )
        else:
            analysis_results["performance_verdict"] = "below_baseline"
            recommendations.append(
                f"{new_model} underperforms compared to baselines - significant improvements needed"
            )

        # Task-specific analysis
        new_model_results = [r for r in evaluation_results if r.model_name == new_model]
        task_performance: dict[str, list[float]] = {}

        for result in new_model_results:
            task = result.task_type
            if task not in task_performance:
                task_performance[task] = []
            metric = result.get_primary_metric()
            if metric is not None:
                task_performance[task].append(metric)

        # Identify strengths and weaknesses
        for task, metrics in task_performance.items():
            if metrics:
                avg_metric = sum(m for m in metrics if m is not None) / len(
                    [m for m in metrics if m is not None]
                )
                if avg_metric > 0.8:
                    strengths.append(f"Strong performance on {task} tasks")
                elif avg_metric < 0.6:
                    weaknesses.append(f"Weak performance on {task} tasks")

        analysis_results["detailed_analysis"] = {
            "task_performance": task_performance,
            "improvement_breakdown": summary,
            "evaluation_count": len(new_model_results),
        }

        return analysis_results

    async def _generate_comprehensive_report(
        self,
        evaluation_results: list[EvaluationResult],
        analysis_results: dict[str, Any],
        comparison_results: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report_recommendations: list[str] = []
        next_steps: list[str] = []

        report = {
            "evaluation_summary": {
                "total_evaluations": len(evaluation_results),
                "models_evaluated": len({r.model_name for r in evaluation_results}),
                "tasks_covered": len({r.task_type for r in evaluation_results}),
                "evaluation_period": {
                    "start": min(r.timestamp for r in evaluation_results).isoformat()
                    if evaluation_results
                    else None,
                    "end": max(r.timestamp for r in evaluation_results).isoformat()
                    if evaluation_results
                    else None,
                },
            },
            "performance_analysis": analysis_results,
            "model_comparisons": comparison_results,
            "configuration": config,
            "recommendations": report_recommendations,
            "next_steps": next_steps,
        }

        # Generate recommendations based on analysis
        if "overall" in analysis_results:
            overall = analysis_results["overall"]
            if "bottlenecks" in overall:
                bottlenecks = overall["bottlenecks"]
                if bottlenecks.get("recommendations"):
                    report_recommendations.extend(bottlenecks["recommendations"][:3])

        # Model-specific recommendations
        for model_name, model_analysis in analysis_results.items():
            if model_name != "overall" and "performance_analysis" in model_analysis:
                perf = model_analysis["performance_analysis"]
                if "error" not in perf:
                    # Add model-specific insights
                    next_steps.append(f"Detailed analysis available for {model_name}")

        # Comparison insights
        if "multi_model" in comparison_results:
            multi_model = comparison_results["multi_model"]
            if "ranking_analysis" in multi_model:
                rankings = multi_model["ranking_analysis"].get("rankings", [])
                if rankings:
                    best_model = rankings[0]
                    next_steps.append(
                        f"Consider {best_model.get('model_name')} as the top performer "
                        f"with {best_model.get('mean_performance', 0):.3f} average performance"
                    )

        return report

    async def _cleanup_workflow_resources(self, workflow_id: str) -> None:
        """Clean up resources allocated for workflow."""
        # This is a placeholder for resource cleanup
        # In a real implementation, this might:
        # - Release model memory
        # - Clean up temporary files
        # - Close database connections
        # - Free up compute resources

        self.logger.debug(f"Cleaning up resources for workflow {workflow_id}")

        # Simulate cleanup time
        await asyncio.sleep(0.1)

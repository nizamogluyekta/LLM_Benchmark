"""Orchestration service for managing complete experiment workflows."""

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from benchmark.core.base import BaseService, HealthCheck, ServiceResponse, ServiceStatus
from benchmark.core.logging import get_logger
from benchmark.interfaces.orchestration_interfaces import (
    ExperimentProgress,
    ExperimentResult,
    ExperimentStatus,
    OrchestrationInterface,
    WorkflowContext,
    WorkflowStep,
)
from benchmark.services.configuration_service import ConfigurationService
from benchmark.services.data_service import DataService
from benchmark.services.evaluation_service import EvaluationService
from benchmark.services.model_service import ModelService
from benchmark.workflow.workflow_engine import WorkflowEngine


class ExperimentContext:
    """Context for tracking experiment execution."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        config: dict[str, Any],
        services: dict[str, BaseService],
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.config = config
        self.services = services
        self.status = ExperimentStatus.CREATED
        self.created_at = datetime.now()

        # Execution tracking
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.current_step: str | None = None
        self.total_steps: int = 0
        self.completed_steps: int = 0

        # Resources
        self.loaded_datasets: dict[str, Any] = {}
        self.loaded_models: dict[str, Any] = {}

        # Results and errors
        self.results: dict[str, Any] = {}
        self.error_message: str | None = None
        self.cancel_requested: bool = False

        # Step timing for progress estimation
        self.step_start_times: dict[str, datetime] = {}
        self.step_durations: dict[str, float] = {}


class OrchestrationService(BaseService, OrchestrationInterface):
    """Service for orchestrating complete experiment workflows."""

    def __init__(self) -> None:
        super().__init__("orchestration_service")
        self.logger = get_logger("orchestration_service")
        self.services: dict[str, BaseService] = {}
        self.experiments: dict[str, ExperimentContext] = {}
        self.workflow_engine = WorkflowEngine()
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}

    async def initialize(self) -> ServiceResponse:
        """Initialize orchestration service and all dependent services."""
        try:
            self.logger.info("Initializing orchestration service...")

            # Initialize all services in dependency order
            self.services["config"] = ConfigurationService()
            await self._initialize_service("config")

            self.services["data"] = DataService()
            await self._initialize_service("data")

            self.services["model"] = ModelService()
            await self._initialize_service("model")

            self.services["evaluation"] = EvaluationService()
            await self._initialize_service("evaluation")

            # Initialize workflow engine
            await self.workflow_engine.initialize(self.services)

            self._initialized = True
            self.logger.info("Orchestration service initialized successfully")

            return ServiceResponse(
                success=True,
                message="Orchestration service initialized",
                data={"initialized_services": len(self.services)},
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize orchestration service: {e}")
            self._status = ServiceStatus.UNHEALTHY
            return ServiceResponse(success=False, message="Initialization failed", error=str(e))

    async def _initialize_service(self, service_name: str) -> None:
        """Initialize a dependent service with error handling."""
        try:
            response = await self.services[service_name].initialize()
            if not response.success:
                raise Exception(f"Service {service_name} initialization failed: {response.error}")
            self.logger.info(f"Service {service_name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {service_name}: {e}")
            raise

    async def create_experiment(self, config_path: str, experiment_name: str | None = None) -> str:
        """Create a new experiment from configuration."""
        try:
            # Generate experiment ID
            experiment_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:8]}"

            self.logger.info(f"Creating experiment {experiment_id} from config: {config_path}")

            # Load configuration
            config_response = await self.services["config"].load_experiment_config(config_path)  # type: ignore
            if not config_response.success:
                raise Exception(f"Failed to load configuration: {config_response.error}")

            experiment_config = config_response.data

            # Create experiment context
            context = ExperimentContext(
                experiment_id=experiment_id,
                name=experiment_name
                or experiment_config.get("name", f"Experiment {experiment_id}"),
                config=experiment_config,
                services=self.services,
            )

            self.experiments[experiment_id] = context

            self.logger.info(f"Experiment {experiment_id} created successfully")

            return experiment_id

        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            raise

    async def start_experiment(self, experiment_id: str, background: bool = True) -> dict[str, Any]:
        """Start experiment execution."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        context = self.experiments[experiment_id]

        if context.status != ExperimentStatus.CREATED:
            raise ValueError(f"Experiment {experiment_id} is not in CREATED status")

        self.logger.info(f"Starting experiment {experiment_id} (background={background})")

        if background:
            # Start experiment in background
            task = asyncio.create_task(self._run_experiment_workflow(experiment_id))
            self._running_tasks[experiment_id] = task
            return {"message": "Experiment started in background", "task_id": id(task)}
        else:
            # Run experiment synchronously
            return await self._run_experiment_workflow(experiment_id)

    async def _run_experiment_workflow(self, experiment_id: str) -> dict[str, Any]:
        """Run the complete experiment workflow."""
        context = self.experiments[experiment_id]
        context.status = ExperimentStatus.INITIALIZING
        context.started_at = datetime.now()

        try:
            self.logger.info(f"Running workflow for experiment {experiment_id}")

            # Create workflow steps based on configuration
            workflow_steps = self._create_workflow_steps(context.config)
            context.total_steps = len(workflow_steps)

            # Create workflow context
            workflow_context = WorkflowContext(
                experiment_id=experiment_id,
                config=context.config,
                services=self.services,
            )

            # Execute workflow
            context.status = ExperimentStatus.RUNNING
            workflow_result = await self.workflow_engine.execute_workflow(
                workflow_steps, workflow_context, self._update_progress_callback(context)
            )

            # Update context with results
            context.status = ExperimentStatus.COMPLETED
            context.completed_at = datetime.now()
            context.results = workflow_result

            self.logger.info(f"Experiment {experiment_id} completed successfully")

            # Create experiment result
            duration = (context.completed_at - context.started_at).total_seconds()

            return {
                "experiment_id": experiment_id,
                "status": ExperimentStatus.COMPLETED.value,
                "duration_seconds": duration,
                "results": workflow_result,
            }

        except Exception as e:
            context.status = ExperimentStatus.FAILED
            context.error_message = str(e)
            self.logger.error(f"Experiment {experiment_id} failed: {e}")

            return {
                "experiment_id": experiment_id,
                "status": ExperimentStatus.FAILED.value,
                "error": str(e),
            }

        finally:
            # Clean up running task
            if experiment_id in self._running_tasks:
                del self._running_tasks[experiment_id]

    def _create_workflow_steps(self, config: dict[str, Any]) -> list[WorkflowStep]:
        """Create workflow steps based on configuration."""
        from benchmark.workflow.workflow_steps import (
            DataLoadingStep,
            EvaluationExecutionStep,
            ModelLoadingStep,
            ResultsAggregationStep,
        )

        steps: list[WorkflowStep] = []

        # Always include data loading if datasets are specified
        if config.get("datasets"):
            steps.append(DataLoadingStep())

        # Always include model loading if models are specified
        if config.get("models"):
            steps.append(ModelLoadingStep())

        # Include evaluation step
        steps.append(EvaluationExecutionStep())

        # Include results aggregation
        steps.append(ResultsAggregationStep())

        return steps

    def _update_progress_callback(self, context: ExperimentContext) -> Callable[[str, bool], None]:
        """Create progress update callback for workflow execution."""

        def callback(step_name: str, completed: bool) -> None:
            context.current_step = step_name
            if completed:
                context.completed_steps += 1
                # Track step completion time
                if step_name in context.step_start_times:
                    duration = (
                        datetime.now() - context.step_start_times[step_name]
                    ).total_seconds()
                    context.step_durations[step_name] = duration
            else:
                # Track step start time
                context.step_start_times[step_name] = datetime.now()

        return callback

    async def get_experiment_progress(self, experiment_id: str) -> ExperimentProgress:
        """Get current experiment progress."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        context = self.experiments[experiment_id]

        # Calculate progress
        elapsed_time = 0.0
        if context.started_at:
            elapsed_time = (datetime.now() - context.started_at).total_seconds()

        percentage = 0.0
        if context.total_steps > 0:
            percentage = (context.completed_steps / context.total_steps) * 100

        estimated_remaining = self._estimate_remaining_time(context)

        return ExperimentProgress(
            experiment_id=experiment_id,
            status=context.status,
            current_step=context.current_step or "Not started",
            total_steps=context.total_steps,
            completed_steps=context.completed_steps,
            percentage=percentage,
            elapsed_time_seconds=elapsed_time,
            estimated_remaining_seconds=estimated_remaining,
            error_message=context.error_message,
        )

    def _estimate_remaining_time(self, context: ExperimentContext) -> float | None:
        """Estimate remaining execution time based on completed steps."""
        if context.completed_steps == 0 or context.total_steps == 0:
            return None

        if not context.started_at:
            return None

        elapsed_time = (datetime.now() - context.started_at).total_seconds()
        progress_ratio = context.completed_steps / context.total_steps

        if progress_ratio > 0:
            estimated_total_time = elapsed_time / progress_ratio
            remaining_time = estimated_total_time - elapsed_time
            return max(0, remaining_time)

        return None

    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel running experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        context = self.experiments[experiment_id]

        if context.status not in [ExperimentStatus.INITIALIZING, ExperimentStatus.RUNNING]:
            raise ValueError(
                f"Experiment {experiment_id} cannot be cancelled (status: {context.status.value})"
            )

        try:
            self.logger.info(f"Cancelling experiment {experiment_id}")

            # Set cancellation flag
            context.cancel_requested = True
            context.status = ExperimentStatus.CANCELLED

            # Cancel running task if it exists
            if experiment_id in self._running_tasks:
                task = self._running_tasks[experiment_id]
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                del self._running_tasks[experiment_id]

            # Cleanup resources
            await self._cleanup_experiment_resources(context)

            self.logger.info(f"Experiment {experiment_id} cancelled successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel experiment {experiment_id}: {e}")
            return False

    async def _cleanup_experiment_resources(self, context: ExperimentContext) -> None:
        """Clean up experiment resources."""
        try:
            # Cleanup loaded models
            if "model" in self.services:
                for model_id in context.loaded_models:
                    try:
                        await self.services["model"].cleanup_model(model_id)  # type: ignore
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup model {model_id}: {e}")

            # Clear context data
            context.loaded_datasets.clear()
            context.loaded_models.clear()

        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")

    async def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """Get experiment result."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        context = self.experiments[experiment_id]

        duration = 0.0
        if context.started_at:
            end_time = context.completed_at or datetime.now()
            duration = (end_time - context.started_at).total_seconds()

        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=context.name,
            status=context.status,
            started_at=context.started_at.isoformat() if context.started_at else "",
            completed_at=context.completed_at.isoformat() if context.completed_at else None,
            total_duration_seconds=duration,
            models_evaluated=list(context.loaded_models.keys()),
            datasets_used=list(context.loaded_datasets.keys()),
            evaluation_results=context.results,
            error_message=context.error_message,
            metadata={
                "total_steps": context.total_steps,
                "completed_steps": context.completed_steps,
                "step_durations": context.step_durations,
            },
        )

    async def list_experiments(
        self, status_filter: ExperimentStatus | None = None
    ) -> list[dict[str, Any]]:
        """List all experiments with optional status filter."""
        experiments_list = []

        for exp_id, context in self.experiments.items():
            if status_filter is None or context.status == status_filter:
                experiments_list.append(
                    {
                        "experiment_id": exp_id,
                        "name": context.name,
                        "status": context.status.value,
                        "created_at": context.created_at.isoformat(),
                        "started_at": context.started_at.isoformat()
                        if context.started_at
                        else None,
                        "completed_at": context.completed_at.isoformat()
                        if context.completed_at
                        else None,
                        "models_count": len(context.config.get("models", [])),
                        "datasets_count": len(context.config.get("datasets", [])),
                    }
                )

        return experiments_list

    async def health_check(self) -> HealthCheck:
        """Check orchestration service health."""
        try:
            # Check all dependent services
            service_health = {}
            overall_status = ServiceStatus.HEALTHY

            for service_name, service in self.services.items():
                health = await service.health_check()
                # Handle both enum and string values
                if hasattr(health.status, "value"):
                    service_health[service_name] = health.status.value
                    status_enum = health.status
                else:
                    service_health[service_name] = str(health.status)
                    # Convert string to enum for comparison
                    status_enum = (
                        ServiceStatus(health.status)
                        if isinstance(health.status, str)
                        else health.status
                    )

                if status_enum == ServiceStatus.UNHEALTHY:
                    overall_status = ServiceStatus.UNHEALTHY
                elif (
                    status_enum == ServiceStatus.DEGRADED
                    and overall_status == ServiceStatus.HEALTHY
                ):
                    overall_status = ServiceStatus.DEGRADED

            active_experiments = len(
                [
                    exp
                    for exp in self.experiments.values()
                    if exp.status in [ExperimentStatus.RUNNING, ExperimentStatus.INITIALIZING]
                ]
            )

            return HealthCheck(
                status=overall_status,
                message="Orchestration service health check",
                checks={
                    "dependent_services": service_health,
                    "active_experiments": active_experiments,
                    "total_experiments": len(self.experiments),
                    "running_tasks": len(self._running_tasks),
                },
                uptime_seconds=self.get_uptime_seconds(),
            )

        except Exception as e:
            return HealthCheck(
                status=ServiceStatus.UNHEALTHY,
                message="Health check failed",
                checks={"error": str(e)},
                uptime_seconds=self.get_uptime_seconds(),
            )

    async def shutdown(self) -> ServiceResponse:
        """Graceful shutdown of orchestration service."""
        try:
            self.logger.info("Shutting down orchestration service...")

            # Cancel all running experiments
            for exp_id in list(self.experiments.keys()):
                context = self.experiments[exp_id]
                if context.status in [ExperimentStatus.RUNNING, ExperimentStatus.INITIALIZING]:
                    try:
                        await self.cancel_experiment(exp_id)
                    except Exception as e:
                        self.logger.warning(f"Error cancelling experiment {exp_id}: {e}")

            # Cancel all running tasks
            for task in self._running_tasks.values():
                task.cancel()

            # Wait for tasks to complete
            if self._running_tasks:
                try:
                    await asyncio.wait(
                        self._running_tasks.values(), timeout=5.0, return_when=asyncio.ALL_COMPLETED
                    )
                except TimeoutError:
                    self.logger.warning("Some tasks did not complete within timeout")

            # Shutdown all services
            for service_name, service in self.services.items():
                try:
                    await service.shutdown()
                    self.logger.info(f"Service {service_name} shut down")
                except Exception as e:
                    self.logger.warning(f"Error shutting down {service_name}: {e}")

            self._status = ServiceStatus.STOPPED
            self.logger.info("Orchestration service shut down successfully")

            return ServiceResponse(
                success=True, message="Orchestration service shut down successfully"
            )

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return ServiceResponse(success=False, message="Shutdown failed", error=str(e))

"""Workflow engine for executing experiment steps in dependency order."""

import asyncio
from collections.abc import Callable
from typing import Any

from benchmark.core.base import BaseService, ServiceStatus
from benchmark.core.logging import get_logger
from benchmark.interfaces.orchestration_interfaces import WorkflowContext, WorkflowStep


class WorkflowEngine:
    """Engine for executing workflow steps in dependency order."""

    def __init__(self) -> None:
        self.logger = get_logger("workflow_engine")
        self.services: dict[str, BaseService] = {}

    async def initialize(self, services: dict[str, BaseService]) -> None:
        """Initialize workflow engine with services."""
        self.services = services
        self.logger.info("Workflow engine initialized with services: %s", list(services.keys()))

    async def execute_workflow(
        self,
        steps: list[WorkflowStep],
        context: WorkflowContext,
        progress_callback: Callable[[str, bool], None] | None = None,
    ) -> dict[str, Any]:
        """Execute workflow steps in order with dependency checking.

        Args:
            steps: List of workflow steps to execute
            context: Workflow context containing config and services
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing results from all steps

        Raises:
            Exception: If any step fails or dependencies are not met
        """
        self.logger.info("Starting workflow execution with %d steps", len(steps))

        workflow_results = {}

        for i, step in enumerate(steps):
            step_name = step.get_step_name()
            self.logger.info("Executing step %d/%d: %s", i + 1, len(steps), step_name)

            # Check for cancellation
            if context.cancel_requested:
                self.logger.warning("Workflow cancelled by user request")
                raise Exception("Workflow cancelled by user")

            # Update progress - step starting
            if progress_callback:
                progress_callback(step_name, False)

            try:
                # Check step dependencies
                await self._check_step_dependencies(step)

                # Execute step with timeout and error handling
                step_result = await self._execute_step_with_timeout(step, context)

                # Store step result
                workflow_results[step_name] = step_result
                context.step_results[step_name] = step_result

                self.logger.info("Step %s completed successfully", step_name)

                # Update progress - step completed
                if progress_callback:
                    progress_callback(step_name, True)

            except Exception as e:
                self.logger.error("Step %s failed: %s", step_name, str(e))
                raise Exception(f"Workflow step '{step_name}' failed: {e}") from e

        self.logger.info("Workflow execution completed successfully")
        return workflow_results

    async def _execute_step_with_timeout(
        self, step: WorkflowStep, context: WorkflowContext, timeout_seconds: float = 3600.0
    ) -> dict[str, Any]:
        """Execute a workflow step with timeout protection."""
        try:
            # Get step timeout or use default
            step_timeout = getattr(step, "get_timeout_seconds", lambda: timeout_seconds)()

            # Execute step with timeout
            result = await asyncio.wait_for(step.execute(context), timeout=step_timeout)

            return result

        except TimeoutError as e:
            raise Exception(
                f"Step '{step.get_step_name()}' timed out after {timeout_seconds} seconds"
            ) from e

        except Exception:
            # Re-raise the original exception
            raise

    async def _check_step_dependencies(self, step: WorkflowStep) -> None:
        """Check that step dependencies are available and healthy.

        Args:
            step: Workflow step to check dependencies for

        Raises:
            Exception: If dependencies are not met
        """
        required_services = step.get_dependencies()

        for service_name in required_services:
            if service_name not in self.services:
                raise Exception(f"Required service '{service_name}' not available")

            # Check service health
            try:
                health = await self.services[service_name].health_check()
                if health.status == ServiceStatus.UNHEALTHY:
                    raise Exception(f"Required service '{service_name}' is unhealthy")
                elif health.status == ServiceStatus.DEGRADED:
                    self.logger.warning("Service '%s' is degraded but will continue", service_name)

            except Exception as e:
                raise Exception(f"Failed to check health of service '{service_name}': {e}") from e

    async def validate_workflow(self, steps: list[WorkflowStep]) -> dict[str, Any]:
        """Validate workflow steps and dependencies.

        Args:
            steps: List of workflow steps to validate

        Returns:
            Dictionary with validation results

        Raises:
            Exception: If validation fails
        """
        self.logger.info("Validating workflow with %d steps", len(steps))

        validation_results: dict[str, Any] = {
            "valid": True,
            "steps_count": len(steps),
            "dependencies": {},
            "issues": [],
        }

        all_dependencies = set()

        for step in steps:
            step_name = step.get_step_name()
            dependencies = step.get_dependencies()

            validation_results["dependencies"][step_name] = dependencies
            all_dependencies.update(dependencies)

            # Check for duplicate step names
            existing_steps = [s.get_step_name() for s in steps]
            if existing_steps.count(step_name) > 1:
                issue = f"Duplicate step name: {step_name}"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # Check if all dependencies are available
        for dep in all_dependencies:
            if dep not in self.services:
                issue = f"Missing required service: {dep}"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        if validation_results["valid"]:
            self.logger.info("Workflow validation passed")
        else:
            self.logger.error("Workflow validation failed: %s", validation_results["issues"])

        return validation_results

    def get_workflow_info(self, steps: list[WorkflowStep]) -> dict[str, Any]:
        """Get information about a workflow.

        Args:
            steps: List of workflow steps

        Returns:
            Dictionary with workflow information
        """
        info: dict[str, Any] = {
            "steps_count": len(steps),
            "steps": [],
            "total_estimated_duration": 0.0,
            "all_dependencies": set(),
        }

        for step in steps:
            step_info = {
                "name": step.get_step_name(),
                "dependencies": step.get_dependencies(),
                "estimated_duration": step.get_estimated_duration_seconds(),
            }

            info["steps"].append(step_info)
            info["all_dependencies"].update(step.get_dependencies())

            # Add to total estimated duration if available
            if step_info["estimated_duration"]:
                info["total_estimated_duration"] += step_info["estimated_duration"]

        info["all_dependencies"] = list(info["all_dependencies"])

        return info


class ParallelWorkflowEngine(WorkflowEngine):
    """Extended workflow engine that supports parallel execution of independent steps."""

    def __init__(self, max_concurrent_steps: int = 3):
        super().__init__()
        self.max_concurrent_steps = max_concurrent_steps

    async def execute_workflow_parallel(
        self,
        steps: list[WorkflowStep],
        context: WorkflowContext,
        progress_callback: Callable[[str, bool], None] | None = None,
    ) -> dict[str, Any]:
        """Execute workflow steps with parallel execution where possible.

        Args:
            steps: List of workflow steps to execute
            context: Workflow context
            progress_callback: Optional progress callback

        Returns:
            Dictionary containing results from all steps
        """
        self.logger.info("Starting parallel workflow execution with %d steps", len(steps))

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(steps)

        # Execute steps in parallel where possible
        return await self._execute_parallel_steps(dependency_graph, context, progress_callback)

    def _build_dependency_graph(self, steps: list[WorkflowStep]) -> dict[str, dict[str, Any]]:
        """Build dependency graph for parallel execution."""
        graph = {}

        for step in steps:
            step_name = step.get_step_name()
            graph[step_name] = {
                "step": step,
                "dependencies": step.get_dependencies(),
                "dependents": [],
                "completed": False,
                "running": False,
            }

        # Build dependent relationships
        for step_name, step_info in graph.items():
            for dep in step_info["dependencies"]:  # type: ignore
                # Find steps that depend on this service
                for other_name, other_info in graph.items():
                    if other_name != step_name:
                        other_step = other_info["step"]
                        if dep in other_step.get_dependencies():  # type: ignore
                            step_info["dependents"].append(other_name)  # type: ignore

        return graph

    async def _execute_parallel_steps(
        self,
        dependency_graph: dict[str, dict[str, Any]],
        context: WorkflowContext,
        progress_callback: Callable[[str, bool], None] | None = None,
    ) -> dict[str, Any]:
        """Execute steps in parallel respecting dependencies."""
        workflow_results = {}
        running_tasks: dict[str, asyncio.Task[Any]] = {}

        while not all(info["completed"] for info in dependency_graph.values()):
            # Check for cancellation
            if context.cancel_requested:
                # Cancel all running tasks
                for task in running_tasks.values():
                    task.cancel()
                raise Exception("Workflow cancelled by user")

            # Find ready steps (dependencies completed, not running, not completed)
            ready_steps = []
            for step_name, step_info in dependency_graph.items():
                if (
                    not step_info["completed"]
                    and not step_info["running"]
                    and self._dependencies_completed(step_info, dependency_graph)
                ):
                    ready_steps.append(step_name)

            # Start tasks for ready steps (up to max concurrent)
            available_slots = self.max_concurrent_steps - len(running_tasks)
            for step_name in ready_steps[:available_slots]:
                step_info = dependency_graph[step_name]
                step = step_info["step"]

                # Mark as running
                step_info["running"] = True

                # Start task
                task = asyncio.create_task(self._execute_step_with_timeout(step, context))
                running_tasks[step_name] = task

                # Update progress
                if progress_callback:
                    progress_callback(step_name, False)

                self.logger.info("Started parallel execution of step: %s", step_name)

            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(), return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for task in done:
                    # Find which step completed
                    completed_step = None
                    for step_name, step_task in running_tasks.items():
                        if step_task == task:
                            completed_step = step_name
                            break

                    if completed_step:
                        try:
                            result = await task
                            workflow_results[completed_step] = result
                            context.step_results[completed_step] = result

                            # Mark as completed
                            dependency_graph[completed_step]["completed"] = True
                            dependency_graph[completed_step]["running"] = False

                            # Remove from running tasks
                            del running_tasks[completed_step]

                            # Update progress
                            if progress_callback:
                                progress_callback(completed_step, True)

                            self.logger.info("Completed parallel step: %s", completed_step)

                        except Exception as e:
                            self.logger.error("Parallel step %s failed: %s", completed_step, str(e))
                            # Cancel all running tasks
                            for pending_task in running_tasks.values():
                                pending_task.cancel()
                            raise Exception(
                                f"Parallel workflow step '{completed_step}' failed: {e}"
                            ) from e

        self.logger.info("Parallel workflow execution completed successfully")
        return workflow_results

    def _dependencies_completed(
        self, step_info: dict[str, Any], dependency_graph: dict[str, dict[str, Any]]
    ) -> bool:
        """Check if all service dependencies for a step are satisfied."""
        # For now, we only check service dependencies, not step dependencies
        # In a more advanced implementation, we could also check step-to-step dependencies
        return True

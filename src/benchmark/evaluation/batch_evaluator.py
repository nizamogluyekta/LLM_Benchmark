"""
Batch evaluation system with parallel processing and resource management.

This module provides efficient batch processing capabilities for running
multiple evaluations in parallel while managing system resources optimally.
"""

import asyncio
import logging
import resource
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ..services.evaluation_service import EvaluationService
from .result_models import EvaluationResult


class ResourceType(Enum):
    """Types of system resources to monitor."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


class BatchStatus(Enum):
    """Batch execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ResourceLimits:
    """Resource usage limits for batch processing."""

    max_memory_mb: int | None = None
    max_cpu_percent: float | None = None
    max_gpu_memory_mb: int | None = None
    max_disk_io_mbps: float | None = None
    max_concurrent_evaluations: int = 4
    timeout_seconds: int | None = None


@dataclass
class BatchResult:
    """Result of a batch evaluation."""

    batch_id: str
    status: BatchStatus
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    start_time: datetime
    end_time: datetime | None
    duration_seconds: float | None
    results: list[EvaluationResult]
    errors: list[str]
    resource_usage: dict[str, Any]
    performance_metrics: dict[str, float]


class ResourceMonitor:
    """Monitor system resource usage during batch processing."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._usage_history: list[dict[str, Any]] = []

    async def start_monitoring(self, interval_seconds: float = 1.0) -> None:
        """Start resource monitoring."""
        self._monitoring = True
        self._usage_history = []

        while self._monitoring:
            usage = await self._collect_resource_usage()
            self._usage_history.append(usage)
            await asyncio.sleep(interval_seconds)

    def stop_monitoring(self) -> dict[str, Any]:
        """Stop monitoring and return usage summary."""
        self._monitoring = False

        if not self._usage_history:
            return {}

        # Calculate summary statistics
        cpu_usage = [u["cpu_percent"] for u in self._usage_history]
        memory_usage = [u["memory_mb"] for u in self._usage_history]

        return {
            "monitoring_duration": len(self._usage_history),
            "cpu_usage": {
                "min": min(cpu_usage) if cpu_usage else 0,
                "max": max(cpu_usage) if cpu_usage else 0,
                "avg": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            },
            "memory_usage": {
                "min": min(memory_usage) if memory_usage else 0,
                "max": max(memory_usage) if memory_usage else 0,
                "avg": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            },
            "peak_usage": max(self._usage_history, key=lambda x: x["cpu_percent"])
            if self._usage_history
            else {},
            "usage_history": self._usage_history,
        }

    async def _collect_resource_usage(self) -> dict[str, Any]:
        """Collect current resource usage metrics."""
        usage = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "gpu_memory_mb": 0.0,
            "disk_io_mbps": 0.0,
        }

        try:
            # Get memory usage
            memory_info = resource.getrusage(resource.RUSAGE_SELF)
            usage["memory_mb"] = memory_info.ru_maxrss / 1024  # Convert KB to MB on Linux

            # Note: In a real implementation, you would use libraries like psutil
            # for more accurate and cross-platform resource monitoring
            # For this example, we'll use basic resource information

        except Exception as e:
            self.logger.warning(f"Failed to collect resource usage: {e}")

        return usage

    def check_resource_limits(
        self, limits: ResourceLimits, current_usage: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Check if current usage exceeds resource limits."""
        violations = []

        if limits.max_memory_mb and current_usage.get("memory_mb", 0) > limits.max_memory_mb:
            violations.append(
                f"Memory usage {current_usage['memory_mb']:.1f}MB exceeds limit {limits.max_memory_mb}MB"
            )

        if limits.max_cpu_percent and current_usage.get("cpu_percent", 0) > limits.max_cpu_percent:
            violations.append(
                f"CPU usage {current_usage['cpu_percent']:.1f}% exceeds limit {limits.max_cpu_percent}%"
            )

        if (
            limits.max_gpu_memory_mb
            and current_usage.get("gpu_memory_mb", 0) > limits.max_gpu_memory_mb
        ):
            violations.append(
                f"GPU memory {current_usage['gpu_memory_mb']:.1f}MB exceeds limit {limits.max_gpu_memory_mb}MB"
            )

        return len(violations) == 0, violations


class BatchScheduler:
    """Schedule and prioritize batch evaluations."""

    def __init__(self, max_concurrent_batches: int = 2):
        self.max_concurrent_batches = max_concurrent_batches
        self.running_batches: set[str] = set()
        self.pending_batches: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def schedule_batch(
        self, batch_id: str, priority: int = 0, dependencies: list[str] | None = None
    ) -> bool:
        """Schedule a batch for execution."""
        if len(self.running_batches) < self.max_concurrent_batches:
            self.running_batches.add(batch_id)
            return True
        else:
            self.pending_batches.append(
                {
                    "batch_id": batch_id,
                    "priority": priority,
                    "dependencies": dependencies or [],
                    "scheduled_time": datetime.now(),
                }
            )
            # Sort by priority (higher priority first)
            self.pending_batches.sort(key=lambda x: x["priority"], reverse=True)
            return False

    def complete_batch(self, batch_id: str) -> str | None:
        """Mark batch as completed and schedule next if available."""
        if batch_id in self.running_batches:
            self.running_batches.remove(batch_id)

        # Schedule next pending batch
        if self.pending_batches and len(self.running_batches) < self.max_concurrent_batches:
            next_batch = self.pending_batches.pop(0)
            next_batch_id = next_batch["batch_id"]
            self.running_batches.add(next_batch_id)
            return next_batch_id  # type: ignore[no-any-return]

        return None

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        return {
            "running_batches": len(self.running_batches),
            "pending_batches": len(self.pending_batches),
            "max_concurrent": self.max_concurrent_batches,
            "queue_details": self.pending_batches,
        }


class BatchEvaluator:
    """
    Efficient batch evaluation system with parallel processing and resource management.

    Provides capabilities for running multiple evaluations concurrently while
    monitoring and managing system resources to optimize performance.
    """

    def __init__(
        self,
        evaluation_service: EvaluationService,
        default_resource_limits: ResourceLimits | None = None,
    ):
        """
        Initialize batch evaluator.

        Args:
            evaluation_service: Service for running individual evaluations
            default_resource_limits: Default resource limits for batch processing
        """
        self.service = evaluation_service
        self.default_limits = default_resource_limits or ResourceLimits()
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = ResourceMonitor()
        self.scheduler = BatchScheduler()
        self.active_batches: dict[str, BatchResult] = {}

    async def evaluate_batch(
        self,
        batch_config: dict[str, Any],
        resource_limits: ResourceLimits | None = None,
        batch_id: str | None = None,
    ) -> BatchResult:
        """
        Process batch of evaluations efficiently.

        Args:
            batch_config: Configuration for batch evaluation
            resource_limits: Optional resource limits for this batch
            batch_id: Optional batch ID, generated if not provided

        Returns:
            BatchResult with evaluation outcomes and performance metrics
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}_{len(self.active_batches)}"

        limits = resource_limits or self.default_limits

        # Initialize batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PENDING,
            total_evaluations=0,
            successful_evaluations=0,
            failed_evaluations=0,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            results=[],
            errors=[],
            resource_usage={},
            performance_metrics={},
        )

        self.active_batches[batch_id] = batch_result

        try:
            self.logger.info(f"Starting batch evaluation {batch_id}")
            batch_result.status = BatchStatus.RUNNING

            # Start resource monitoring
            monitor_task = asyncio.create_task(
                self.resource_monitor.start_monitoring(interval_seconds=1.0)
            )

            # Extract evaluation configurations from batch
            eval_configs = await self._extract_evaluation_configs(batch_config)
            batch_result.total_evaluations = len(eval_configs)

            # Process evaluations with concurrency control
            results, errors = await self._process_evaluations_with_limits(
                eval_configs, limits, batch_id
            )

            # Stop monitoring and collect metrics
            resource_usage = self.resource_monitor.stop_monitoring()
            monitor_task.cancel()

            # Update batch result
            batch_result.results = results
            batch_result.errors = errors
            batch_result.successful_evaluations = len(results)
            batch_result.failed_evaluations = len(errors)
            batch_result.resource_usage = resource_usage
            batch_result.end_time = datetime.now()
            batch_result.duration_seconds = (
                batch_result.end_time - batch_result.start_time
            ).total_seconds()

            # Calculate performance metrics
            batch_result.performance_metrics = self._calculate_performance_metrics(
                batch_result, resource_usage
            )

            batch_result.status = BatchStatus.COMPLETED

            self.logger.info(
                f"Batch {batch_id} completed: {batch_result.successful_evaluations}/"
                f"{batch_result.total_evaluations} successful"
            )

        except Exception as e:
            error_msg = f"Batch evaluation failed: {str(e)}"
            self.logger.error(f"Batch {batch_id} failed: {error_msg}")

            batch_result.status = BatchStatus.FAILED
            batch_result.errors.append(error_msg)
            batch_result.end_time = datetime.now()

            # Stop monitoring if still running
            self.resource_monitor.stop_monitoring()

            raise

        return batch_result

    async def parallel_evaluation(
        self,
        evaluation_configs: list[dict[str, Any]],
        max_workers: int = 4,
        resource_limits: ResourceLimits | None = None,
        chunk_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run multiple evaluations in parallel with resource management.

        Args:
            evaluation_configs: List of evaluation configurations
            max_workers: Maximum number of parallel workers
            resource_limits: Optional resource limits
            chunk_size: Optional chunk size for batching

        Returns:
            List of evaluation results
        """
        limits = resource_limits or self.default_limits
        limits.max_concurrent_evaluations = min(max_workers, limits.max_concurrent_evaluations)

        if chunk_size is None:
            chunk_size = max_workers * 2  # Process in chunks of 2x workers

        self.logger.info(
            f"Starting parallel evaluation of {len(evaluation_configs)} configs "
            f"with {max_workers} workers"
        )

        all_results = []
        # Create semaphore to limit concurrency across all chunks
        semaphore = asyncio.Semaphore(max_workers)

        async def run_single_evaluation(config: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                try:
                    # Monitor resource usage before evaluation
                    current_usage = await self.resource_monitor._collect_resource_usage()
                    within_limits, violations = self.resource_monitor.check_resource_limits(
                        limits, current_usage
                    )

                    if not within_limits:
                        self.logger.warning(f"Resource limits exceeded: {violations}")
                        # Optionally wait or skip evaluation
                        await asyncio.sleep(1.0)

                    # Run evaluation
                    result = await self.service.run_evaluation(config)  # type: ignore[attr-defined]

                    return {
                        "status": "success",
                        "result": result,
                        "config": config,
                        "resource_usage": current_usage,
                    }

                except Exception as e:
                    self.logger.error(f"Evaluation failed: {str(e)}")
                    return {"status": "error", "error": str(e), "config": config}

        # Process in chunks to manage memory
        for i in range(0, len(evaluation_configs), chunk_size):
            chunk = evaluation_configs[i : i + chunk_size]

            self.logger.info(
                f"Processing chunk {i // chunk_size + 1} with {len(chunk)} evaluations"
            )

            # Run chunk in parallel
            tasks = [run_single_evaluation(config) for config in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in chunk_results:
                if isinstance(result, Exception):
                    all_results.append({"status": "error", "error": str(result)})
                else:
                    # result is guaranteed to be dict[str, Any] here
                    all_results.append(result)  # type: ignore[arg-type]

            # Small delay between chunks to allow resource recovery
            if i + chunk_size < len(evaluation_configs):
                await asyncio.sleep(0.5)

        success_count = len([r for r in all_results if r.get("status") == "success"])
        self.logger.info(
            f"Parallel evaluation completed: {success_count}/{len(evaluation_configs)} successful"
        )

        return all_results

    async def adaptive_batch_processing(
        self,
        evaluation_configs: list[dict[str, Any]],
        initial_batch_size: int = 10,
        target_duration_minutes: float = 5.0,
        max_batch_size: int = 50,
    ) -> list[BatchResult]:
        """
        Adaptively process evaluations with dynamic batch sizing based on performance.

        Args:
            evaluation_configs: List of evaluation configurations
            initial_batch_size: Starting batch size
            target_duration_minutes: Target duration per batch
            max_batch_size: Maximum allowed batch size

        Returns:
            List of batch results
        """
        batch_results = []
        current_batch_size = initial_batch_size
        processed = 0

        self.logger.info(
            f"Starting adaptive batch processing of {len(evaluation_configs)} evaluations"
        )

        while processed < len(evaluation_configs):
            # Get next batch
            batch_end = min(processed + current_batch_size, len(evaluation_configs))
            batch_configs = evaluation_configs[processed:batch_end]

            # Create batch configuration
            batch_config = {"evaluations": batch_configs, "adaptive_mode": True}

            # Process batch
            batch_result = await self.evaluate_batch(batch_config)
            batch_results.append(batch_result)

            # Adapt batch size based on performance
            if batch_result.duration_seconds:
                duration_minutes = batch_result.duration_seconds / 60.0

                # Adjust batch size based on duration
                if duration_minutes < target_duration_minutes * 0.5:
                    # Too fast, increase batch size
                    current_batch_size = min(int(current_batch_size * 1.5), max_batch_size)
                elif duration_minutes > target_duration_minutes * 2.0:
                    # Too slow, decrease batch size
                    current_batch_size = max(int(current_batch_size * 0.7), 1)

                self.logger.info(
                    f"Batch completed in {duration_minutes:.1f} minutes, "
                    f"adjusting batch size to {current_batch_size}"
                )

            processed = batch_end

        self.logger.info(f"Adaptive batch processing completed with {len(batch_results)} batches")
        return batch_results

    def get_batch_status(self, batch_id: str) -> dict[str, Any] | None:
        """Get status of a specific batch."""
        if batch_id not in self.active_batches:
            return None

        batch_result = self.active_batches[batch_id]

        return {
            "batch_id": batch_id,
            "status": batch_result.status.value,
            "progress": {
                "total": batch_result.total_evaluations,
                "completed": batch_result.successful_evaluations + batch_result.failed_evaluations,
                "successful": batch_result.successful_evaluations,
                "failed": batch_result.failed_evaluations,
            },
            "timing": {
                "start_time": batch_result.start_time.isoformat(),
                "end_time": batch_result.end_time.isoformat() if batch_result.end_time else None,
                "duration_seconds": batch_result.duration_seconds,
            },
            "resource_usage": batch_result.resource_usage,
            "performance_metrics": batch_result.performance_metrics,
            "errors": batch_result.errors,
        }

    def list_active_batches(self) -> list[dict[str, Any]]:
        """List all active batch evaluations."""
        return [
            {
                "batch_id": batch_id,
                "status": batch_result.status.value,
                "progress_percent": (
                    (batch_result.successful_evaluations + batch_result.failed_evaluations)
                    / max(batch_result.total_evaluations, 1)
                )
                * 100,
                "start_time": batch_result.start_time.isoformat(),
                "duration_seconds": batch_result.duration_seconds,
            }
            for batch_id, batch_result in self.active_batches.items()
        ]

    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch evaluation."""
        if batch_id not in self.active_batches:
            return False

        batch_result = self.active_batches[batch_id]
        if batch_result.status != BatchStatus.RUNNING:
            return False

        batch_result.status = BatchStatus.CANCELLED
        batch_result.end_time = datetime.now()

        self.logger.info(f"Batch {batch_id} cancelled")
        return True

    def cleanup_completed_batches(self, older_than_hours: int = 24) -> int:
        """Clean up completed batch results older than specified time."""
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        batches_to_remove = []

        for batch_id, batch_result in self.active_batches.items():
            if (
                batch_result.status
                in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]
                and batch_result.end_time
                and batch_result.end_time < cutoff_time
            ):
                batches_to_remove.append(batch_id)

        for batch_id in batches_to_remove:
            del self.active_batches[batch_id]

        self.logger.info(f"Cleaned up {len(batches_to_remove)} completed batches")
        return len(batches_to_remove)

    # Private helper methods

    async def _extract_evaluation_configs(
        self, batch_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract individual evaluation configurations from batch config."""
        if "evaluations" in batch_config:
            return batch_config["evaluations"]  # type: ignore[no-any-return]

        # Generate configurations from batch parameters
        configs = []
        models = batch_config.get("models", [])
        tasks = batch_config.get("tasks", [])
        datasets = batch_config.get("datasets", [])

        for model in models:
            for task in tasks:
                for dataset in datasets or [None]:
                    config = {
                        "model_name": model,
                        "task_type": task,
                        **batch_config.get("evaluation_params", {}),
                    }
                    if dataset:
                        config["dataset_name"] = dataset
                    configs.append(config)

        return configs

    async def _process_evaluations_with_limits(
        self, eval_configs: list[dict[str, Any]], limits: ResourceLimits, batch_id: str
    ) -> tuple[list[EvaluationResult], list[str]]:
        """Process evaluations while respecting resource limits."""
        results = []
        errors = []

        # Use semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(limits.max_concurrent_evaluations)

        async def run_with_monitoring(config: dict[str, Any]) -> None:
            async with semaphore:
                try:
                    # Check resource limits before starting
                    current_usage = await self.resource_monitor._collect_resource_usage()
                    within_limits, violations = self.resource_monitor.check_resource_limits(
                        limits, current_usage
                    )

                    if not within_limits:
                        self.logger.warning(
                            f"Resource limits exceeded for {batch_id}: {violations}"
                        )
                        # Wait a bit for resources to free up
                        await asyncio.sleep(2.0)

                    # Apply timeout if specified
                    if limits.timeout_seconds:
                        result = await asyncio.wait_for(
                            self.service.run_evaluation(config),  # type: ignore[attr-defined]
                            timeout=limits.timeout_seconds,
                        )
                    else:
                        result = await self.service.run_evaluation(config)  # type: ignore[attr-defined]

                    results.append(result)

                except TimeoutError:
                    error_msg = f"Evaluation timed out after {limits.timeout_seconds}s"
                    errors.append(error_msg)
                    self.logger.error(f"Timeout in batch {batch_id}: {error_msg}")

                except Exception as e:
                    error_msg = f"Evaluation failed: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(f"Error in batch {batch_id}: {error_msg}")

        # Run all evaluations
        tasks = [run_with_monitoring(config) for config in eval_configs]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results, errors

    def _calculate_performance_metrics(
        self, batch_result: BatchResult, resource_usage: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate performance metrics for batch evaluation."""
        metrics = {}

        if batch_result.duration_seconds and batch_result.total_evaluations > 0:
            metrics["evaluations_per_second"] = (
                batch_result.successful_evaluations / batch_result.duration_seconds
            )
            metrics["avg_evaluation_time"] = (
                batch_result.duration_seconds / batch_result.total_evaluations
            )

        metrics["success_rate"] = batch_result.successful_evaluations / max(
            batch_result.total_evaluations, 1
        )

        # Resource efficiency metrics
        if resource_usage:
            cpu_usage = resource_usage.get("cpu_usage", {})
            memory_usage = resource_usage.get("memory_usage", {})

            metrics["avg_cpu_usage"] = cpu_usage.get("avg", 0)
            metrics["peak_cpu_usage"] = cpu_usage.get("max", 0)
            metrics["avg_memory_usage_mb"] = memory_usage.get("avg", 0)
            metrics["peak_memory_usage_mb"] = memory_usage.get("max", 0)

            # Calculate efficiency score (success rate weighted by resource usage)
            if metrics["avg_cpu_usage"] > 0:
                metrics["cpu_efficiency"] = metrics["success_rate"] / (
                    metrics["avg_cpu_usage"] / 100
                )

        return metrics

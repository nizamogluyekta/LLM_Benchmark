"""
Apple Silicon M4 Pro performance optimization module.

This module provides hardware-specific optimizations for model inference on Apple Silicon,
including batch processing optimization, async inference management, and hardware acceleration.
"""

import asyncio
import json
import logging
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Hardware type enumeration."""

    APPLE_SILICON_M1 = "apple_m1"
    APPLE_SILICON_M2 = "apple_m2"
    APPLE_SILICON_M3 = "apple_m3"
    APPLE_SILICON_M4 = "apple_m4"
    INTEL_MAC = "intel_mac"
    UNKNOWN = "unknown"


class AccelerationType(Enum):
    """Hardware acceleration type."""

    CPU_ONLY = "cpu"
    METAL_GPU = "metal"
    NEURAL_ENGINE = "neural_engine"
    UNIFIED_MEMORY = "unified_memory"


class RequestPriority(Enum):
    """Request priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class HardwareInfo:
    """Hardware information for optimization."""

    type: HardwareType
    model_name: str
    core_count: int
    performance_cores: int
    efficiency_cores: int
    gpu_cores: int
    neural_engine_cores: int
    unified_memory_gb: float
    metal_support: bool
    neural_engine_support: bool

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility with Pydantic models."""
        return {
            "type": self.type.value,
            "model_name": self.model_name,
            "core_count": self.core_count,
            "performance_cores": self.performance_cores,
            "efficiency_cores": self.efficiency_cores,
            "gpu_cores": self.gpu_cores,
            "neural_engine_cores": self.neural_engine_cores,
            "unified_memory_gb": self.unified_memory_gb,
            "metal_support": self.metal_support,
            "neural_engine_support": self.neural_engine_support,
        }


@dataclass
class BatchConfig:
    """Batch processing configuration."""

    batch_size: int
    max_batch_size: int
    min_batch_size: int
    dynamic_sizing: bool
    timeout_ms: int
    memory_limit_gb: float


@dataclass
class InferenceRequest:
    """Individual inference request."""

    request_id: str
    model_id: str
    input_data: Any
    priority: RequestPriority
    timeout_ms: int
    created_at: float
    callback: Callable[[Any], Any] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class InferenceResult:
    """Inference result with metadata."""

    request_id: str
    model_id: str
    result: Any
    processing_time_ms: float
    queue_time_ms: float
    batch_size: int
    hardware_used: AccelerationType
    success: bool
    error: str | None = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""

    requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queue_depth: int
    batch_efficiency: float
    memory_usage_gb: float
    gpu_utilization: float
    neural_engine_utilization: float


class AppleSiliconOptimizer:
    """
    Apple Silicon M4 Pro specific optimizer for model inference.

    Provides hardware detection, batch size optimization, and acceleration
    management for maximum performance on Apple Silicon architecture.
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the Apple Silicon optimizer."""
        self.cache_dir = cache_dir or Path.home() / ".benchmark" / "optimization"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hardware_info: HardwareInfo | None = None
        self.batch_configs: dict[str, BatchConfig] = {}
        self.performance_history: dict[str, list[PerformanceMetrics]] = {}
        self._optimization_cache: dict[str, dict[str, Any]] = {}

        # Initialize hardware detection
        asyncio.create_task(
            self._detect_hardware()
        ) if asyncio.get_event_loop().is_running() else None

    async def initialize(self) -> None:
        """Initialize the optimizer with hardware detection."""
        await self._detect_hardware()
        await self._load_optimization_cache()
        logger.info(
            f"AppleSiliconOptimizer initialized for {self.hardware_info.type.value if self.hardware_info else 'unknown'}"
        )

    async def _detect_hardware(self) -> HardwareInfo:
        """Detect Apple Silicon hardware capabilities."""
        try:
            # Get system information
            machine = platform.machine()
            processor = platform.processor()

            # Detect Apple Silicon type
            hardware_type = HardwareType.UNKNOWN
            if machine == "arm64":
                # Try to detect specific Apple Silicon model
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    brand_string = result.stdout.strip().lower()

                    if "m4" in brand_string:
                        hardware_type = HardwareType.APPLE_SILICON_M4
                    elif "m3" in brand_string:
                        hardware_type = HardwareType.APPLE_SILICON_M3
                    elif "m2" in brand_string:
                        hardware_type = HardwareType.APPLE_SILICON_M2
                    elif "m1" in brand_string:
                        hardware_type = HardwareType.APPLE_SILICON_M1
                    else:
                        hardware_type = HardwareType.APPLE_SILICON_M4  # Default to M4
                except subprocess.TimeoutExpired:
                    hardware_type = HardwareType.APPLE_SILICON_M4
            elif "intel" in processor.lower():
                hardware_type = HardwareType.INTEL_MAC

            # Get core counts
            core_count = self._get_core_count()
            perf_cores, eff_cores = self._get_core_distribution(hardware_type)

            # Get GPU and Neural Engine info
            gpu_cores = self._get_gpu_cores(hardware_type)
            neural_engine_cores = self._get_neural_engine_cores(hardware_type)

            # Get unified memory
            unified_memory_gb = self._get_unified_memory()

            # Check acceleration support
            metal_support = self._check_metal_support()
            neural_engine_support = self._check_neural_engine_support()

            self.hardware_info = HardwareInfo(
                type=hardware_type,
                model_name=brand_string if "brand_string" in locals() else "Apple Silicon",
                core_count=core_count,
                performance_cores=perf_cores,
                efficiency_cores=eff_cores,
                gpu_cores=gpu_cores,
                neural_engine_cores=neural_engine_cores,
                unified_memory_gb=unified_memory_gb,
                metal_support=metal_support,
                neural_engine_support=neural_engine_support,
            )

            return self.hardware_info

        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
            # Fallback to default M4 Pro configuration
            self.hardware_info = HardwareInfo(
                type=HardwareType.APPLE_SILICON_M4,
                model_name="Apple M4 Pro (detected)",
                core_count=12,
                performance_cores=8,
                efficiency_cores=4,
                gpu_cores=16,
                neural_engine_cores=16,
                unified_memory_gb=32.0,
                metal_support=True,
                neural_engine_support=True,
            )
            return self.hardware_info

    def _get_core_count(self) -> int:
        """Get total CPU core count."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError):
            return 12  # Default M4 Pro

    def _get_core_distribution(self, hardware_type: HardwareType) -> tuple[int, int]:
        """Get performance and efficiency core distribution."""
        if hardware_type == HardwareType.APPLE_SILICON_M4:
            return (8, 4)  # M4 Pro typical
        elif hardware_type == HardwareType.APPLE_SILICON_M3:
            return (6, 6)  # M3 Pro typical
        elif hardware_type == HardwareType.APPLE_SILICON_M2:
            return (8, 4)  # M2 Pro typical
        elif hardware_type == HardwareType.APPLE_SILICON_M1:
            return (4, 4)  # M1 Pro typical
        else:
            return (4, 4)  # Conservative default

    def _get_gpu_cores(self, hardware_type: HardwareType) -> int:
        """Get GPU core count estimate."""
        if hardware_type == HardwareType.APPLE_SILICON_M4:
            return 16  # M4 Pro typical
        elif hardware_type == HardwareType.APPLE_SILICON_M3:
            return 14  # M3 Pro typical
        elif hardware_type == HardwareType.APPLE_SILICON_M2:
            return 16  # M2 Pro typical
        elif hardware_type == HardwareType.APPLE_SILICON_M1:
            return 14  # M1 Pro typical
        else:
            return 0  # Intel/Unknown

    def _get_neural_engine_cores(self, hardware_type: HardwareType) -> int:
        """Get Neural Engine core count."""
        if hardware_type in [
            HardwareType.APPLE_SILICON_M1,
            HardwareType.APPLE_SILICON_M2,
            HardwareType.APPLE_SILICON_M3,
            HardwareType.APPLE_SILICON_M4,
        ]:
            return 16  # Standard for Apple Silicon
        else:
            return 0  # Intel/Unknown

    def _get_unified_memory(self) -> float:
        """Get unified memory size in GB."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            bytes_memory = int(result.stdout.strip())
            return bytes_memory / (1024**3)  # Convert to GB
        except (subprocess.TimeoutExpired, ValueError):
            return 32.0  # Default M4 Pro

    def _check_metal_support(self) -> bool:
        """Check if Metal GPU acceleration is available."""
        try:
            # Try importing Metal-related frameworks
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "Metal" in result.stdout
        except Exception:
            # Assume Metal support on Apple Silicon
            return platform.machine() == "arm64"

    def _check_neural_engine_support(self) -> bool:
        """Check if Neural Engine acceleration is available."""
        # Neural Engine is available on all Apple Silicon
        return platform.machine() == "arm64"

    async def _load_optimization_cache(self) -> None:
        """Load optimization cache from disk."""
        cache_file = self.cache_dir / "optimization_cache.json"
        try:
            if cache_file.exists():
                with open(cache_file) as f:
                    self._optimization_cache = json.load(f)
                logger.info(
                    f"Loaded optimization cache with {len(self._optimization_cache)} entries"
                )
        except Exception as e:
            logger.warning(f"Failed to load optimization cache: {e}")
            self._optimization_cache = {}

    async def _save_optimization_cache(self) -> None:
        """Save optimization cache to disk."""
        cache_file = self.cache_dir / "optimization_cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self._optimization_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save optimization cache: {e}")

    def get_optimal_batch_size(self, model_id: str, model_type: str = "default") -> BatchConfig:
        """
        Get optimal batch configuration for a specific model.

        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., "llm", "embedding", "classification")

        Returns:
            BatchConfig with optimized settings
        """
        cache_key = f"{model_id}_{model_type}"

        # Check cache first
        if cache_key in self.batch_configs:
            return self.batch_configs[cache_key]

        # Calculate optimal batch size based on hardware
        if not self.hardware_info:
            # Default configuration if hardware not detected
            config = BatchConfig(
                batch_size=4,
                max_batch_size=16,
                min_batch_size=1,
                dynamic_sizing=True,
                timeout_ms=5000,
                memory_limit_gb=8.0,
            )
        else:
            # Hardware-optimized configuration
            base_batch_size = self._calculate_base_batch_size(model_type)
            max_batch_size = self._calculate_max_batch_size(model_type)

            config = BatchConfig(
                batch_size=base_batch_size,
                max_batch_size=max_batch_size,
                min_batch_size=1,
                dynamic_sizing=True,
                timeout_ms=3000,  # Faster on Apple Silicon
                memory_limit_gb=min(self.hardware_info.unified_memory_gb * 0.6, 16.0),
            )

        # Cache and return
        self.batch_configs[cache_key] = config
        return config

    def _calculate_base_batch_size(self, model_type: str) -> int:
        """Calculate base batch size based on model type and hardware."""
        if not self.hardware_info:
            return 4

        # Base calculation on performance cores and memory
        base_size = max(2, self.hardware_info.performance_cores // 2)

        # Adjust based on model type
        type_multipliers = {
            "llm": 0.5,  # Large language models need more memory per item
            "embedding": 2.0,  # Embedding models can handle larger batches
            "classification": 1.5,  # Classification models are efficient
            "default": 1.0,
        }

        multiplier = type_multipliers.get(model_type, 1.0)
        return max(1, int(base_size * multiplier))

    def _calculate_max_batch_size(self, model_type: str) -> int:
        """Calculate maximum batch size based on model type and hardware."""
        if not self.hardware_info:
            return 16

        # Base on unified memory and cores
        memory_factor = max(4, int(self.hardware_info.unified_memory_gb / 4))
        core_factor = max(4, self.hardware_info.performance_cores)

        base_max = min(memory_factor, core_factor * 2)

        # Adjust based on model type
        type_limits = {
            "llm": 0.5,  # LLMs are memory intensive
            "embedding": 2.0,  # Embeddings can batch well
            "classification": 1.5,  # Classification is efficient
            "default": 1.0,
        }

        multiplier = type_limits.get(model_type, 1.0)
        return max(4, int(base_max * multiplier))

    def get_optimal_acceleration(self, model_type: str) -> AccelerationType:
        """Get optimal hardware acceleration for model type."""
        if not self.hardware_info:
            return AccelerationType.CPU_ONLY

        # Prioritize based on model type and hardware capabilities
        if model_type in ["llm", "large_model"] and self.hardware_info.metal_support:
            return AccelerationType.METAL_GPU
        elif (
            model_type in ["embedding", "classification"]
            and self.hardware_info.neural_engine_support
        ):
            return AccelerationType.NEURAL_ENGINE
        elif self.hardware_info.unified_memory_gb > 16:
            return AccelerationType.UNIFIED_MEMORY
        else:
            return AccelerationType.CPU_ONLY

    async def update_performance_metrics(self, model_id: str, metrics: PerformanceMetrics) -> None:
        """Update performance metrics for adaptive optimization."""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []

        self.performance_history[model_id].append(metrics)

        # Keep only recent metrics (last 100 measurements)
        if len(self.performance_history[model_id]) > 100:
            self.performance_history[model_id] = self.performance_history[model_id][-100:]

        # Trigger adaptive optimization if we have enough data
        if len(self.performance_history[model_id]) >= 10:
            await self._adaptive_optimization(model_id)

    async def _adaptive_optimization(self, model_id: str) -> None:
        """Perform adaptive optimization based on performance history."""
        if model_id not in self.performance_history:
            return

        recent_metrics = self.performance_history[model_id][-10:]
        avg_latency = sum(m.average_latency_ms for m in recent_metrics) / len(recent_metrics)
        avg_batch_efficiency = sum(m.batch_efficiency for m in recent_metrics) / len(recent_metrics)

        # Adjust batch configuration if performance is suboptimal
        if model_id in self.batch_configs:
            config = self.batch_configs[model_id]

            # If latency is high and batch efficiency is low, reduce batch size
            if avg_latency > 1000 and avg_batch_efficiency < 0.7:
                config.batch_size = max(config.min_batch_size, config.batch_size - 1)
                logger.info(f"Reduced batch size for {model_id} to {config.batch_size}")

            # If latency is good and batch efficiency is high, try larger batches
            elif avg_latency < 500 and avg_batch_efficiency > 0.9:
                config.batch_size = min(config.max_batch_size, config.batch_size + 1)
                logger.info(f"Increased batch size for {model_id} to {config.batch_size}")

        # Save optimizations
        await self._save_optimization_cache()


class InferenceQueue:
    """
    High-performance inference queue with priority management and concurrency control.

    Manages request queuing, batching, and execution with Apple Silicon optimizations.
    """

    def __init__(
        self,
        max_concurrent_requests: int = 10,
        max_queue_size: int = 1000,
        optimizer: AppleSiliconOptimizer | None = None,
    ):
        """Initialize the inference queue."""
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.optimizer = optimizer or AppleSiliconOptimizer()

        # Queue management
        self._request_queues: dict[RequestPriority, asyncio.Queue[InferenceRequest]] = {
            priority: asyncio.Queue(maxsize=max_queue_size // 4) for priority in RequestPriority
        }
        self._active_requests: dict[str, InferenceRequest] = {}
        self._processing_tasks: set[asyncio.Task[None]] = set()

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._metrics: dict[str, Any] = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "queue_depths": dict.fromkeys(RequestPriority, 0),
            "average_processing_time": 0.0,
        }

        # Background task for processing
        self._processor_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        """Initialize the inference queue."""
        await self.optimizer.initialize()
        self._processor_task = asyncio.create_task(self._process_requests())
        logger.info("InferenceQueue initialized")

    async def shutdown(self) -> None:
        """Shutdown the inference queue gracefully."""
        self._shutdown_event.set()

        if self._processor_task:
            await self._processor_task

        # Wait for active requests to complete (with timeout)
        try:
            await asyncio.wait_for(self._wait_for_completion(), timeout=30.0)
        except TimeoutError:
            logger.warning("Shutdown timeout - some requests may be incomplete")

        logger.info("InferenceQueue shutdown complete")

    async def _wait_for_completion(self) -> None:
        """Wait for all active requests to complete."""
        while self._processing_tasks:
            await asyncio.sleep(0.1)

    async def submit_request(self, request: InferenceRequest) -> str:
        """
        Submit a request for inference.

        Args:
            request: The inference request to process

        Returns:
            Request ID for tracking

        Raises:
            QueueFullError: If the queue is at capacity
        """
        # Check queue capacity
        current_queue_size = sum(q.qsize() for q in self._request_queues.values())
        if current_queue_size >= self.max_queue_size:
            raise asyncio.QueueFull("Inference queue is at capacity")

        # Add to appropriate priority queue
        priority_queue = self._request_queues[request.priority]
        await priority_queue.put(request)

        self._active_requests[request.request_id] = request
        self._metrics["total_requests"] += 1
        self._metrics["queue_depths"][request.priority] += 1

        logger.debug(f"Request {request.request_id} queued with priority {request.priority}")
        return request.request_id

    async def get_result(self, request_id: str, timeout: float | None = None) -> InferenceResult:
        """
        Get the result for a specific request.

        Args:
            request_id: ID of the request
            timeout: Maximum time to wait for result

        Returns:
            InferenceResult when complete

        Raises:
            TimeoutError: If timeout is exceeded
            KeyError: If request_id is not found
        """
        if request_id not in self._active_requests:
            raise KeyError(f"Request {request_id} not found")

        # Wait for request completion
        start_time = time.time()
        while request_id in self._active_requests:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Request {request_id} timed out")
            await asyncio.sleep(0.01)

        # Result should be available through callback or storage mechanism
        # This is a simplified implementation - in practice, you'd have a result store
        logger.debug(f"Request {request_id} completed")
        return InferenceResult(
            request_id=request_id,
            model_id="placeholder",
            result=None,
            processing_time_ms=0.0,
            queue_time_ms=0.0,
            batch_size=1,
            hardware_used=AccelerationType.CPU_ONLY,
            success=True,
        )

    async def _process_requests(self) -> None:
        """Main request processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get next batch of requests to process
                batch = await self._get_next_batch()

                if batch:
                    # Process batch asynchronously
                    task = asyncio.create_task(self._process_batch(batch))
                    self._processing_tasks.add(task)

                    # Clean up completed tasks
                    self._processing_tasks = {t for t in self._processing_tasks if not t.done()}
                else:
                    # No requests available, short sleep
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in request processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _get_next_batch(self) -> list[InferenceRequest]:
        """Get the next batch of requests to process, prioritized appropriately."""
        batch = []

        # Process requests by priority (highest first)
        for priority in sorted(RequestPriority, key=lambda p: p.value, reverse=True):
            queue = self._request_queues[priority]

            try:
                # Try to get a request from this priority level
                request = queue.get_nowait()
                batch.append(request)
                self._metrics["queue_depths"][priority] -= 1

                # For high priority requests, process immediately
                if priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
                    break

                # For normal/low priority, try to batch more requests of same priority
                while len(batch) < 8 and not queue.empty():
                    try:
                        request = queue.get_nowait()
                        batch.append(request)
                        self._metrics["queue_depths"][priority] -= 1
                    except asyncio.QueueEmpty:
                        break

                # If we have requests, process them
                if batch:
                    break

            except asyncio.QueueEmpty:
                continue

        return batch

    async def _process_batch(self, batch: list[InferenceRequest]) -> None:
        """Process a batch of requests."""
        async with self._semaphore:
            start_time = time.time()

            try:
                # Group requests by model for efficient batching
                model_groups: dict[str, list[InferenceRequest]] = {}
                for request in batch:
                    if request.model_id not in model_groups:
                        model_groups[request.model_id] = []
                    model_groups[request.model_id].append(request)

                # Process each model group
                for model_id, model_requests in model_groups.items():
                    await self._process_model_batch(model_id, model_requests)

                processing_time = (time.time() - start_time) * 1000

                # Update metrics
                self._metrics["completed_requests"] += len(batch)

                # Update average processing time
                total_requests = self._metrics["completed_requests"]
                current_avg = self._metrics["average_processing_time"]
                self._metrics["average_processing_time"] = (
                    current_avg * (total_requests - len(batch)) + processing_time
                ) / total_requests

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                self._metrics["failed_requests"] += len(batch)

            finally:
                # Clean up active requests
                for request in batch:
                    self._active_requests.pop(request.request_id, None)

    async def _process_model_batch(self, model_id: str, requests: list[InferenceRequest]) -> None:
        """Process a batch of requests for a specific model."""
        # Get optimal batch configuration
        batch_config = self.optimizer.get_optimal_batch_size(model_id)

        # Split into optimally-sized batches if necessary
        optimal_batches = []
        for i in range(0, len(requests), batch_config.batch_size):
            batch_requests = requests[i : i + batch_config.batch_size]
            optimal_batches.append(batch_requests)

        # Process each optimal batch
        for batch_requests in optimal_batches:
            await self._execute_model_inference(model_id, batch_requests, batch_config)

    async def _execute_model_inference(
        self, model_id: str, requests: list[InferenceRequest], config: BatchConfig
    ) -> None:
        """Execute model inference for a batch of requests."""
        start_time = time.time()

        try:
            # This is where you'd integrate with your actual model inference
            # For now, we'll simulate processing
            await asyncio.sleep(0.1)  # Simulate inference time

            processing_time = (time.time() - start_time) * 1000

            # Create results for each request
            for request in requests:
                if request.callback:
                    result = InferenceResult(
                        request_id=request.request_id,
                        model_id=model_id,
                        result=f"processed_{request.request_id}",  # Placeholder
                        processing_time_ms=processing_time / len(requests),
                        queue_time_ms=start_time * 1000 - request.created_at,
                        batch_size=len(requests),
                        hardware_used=self.optimizer.get_optimal_acceleration("default"),
                        success=True,
                    )

                    # Execute callback
                    try:
                        if asyncio.iscoroutinefunction(request.callback):
                            await request.callback(result)
                        else:
                            request.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error for {request.request_id}: {e}")

            # Update optimizer with performance metrics
            metrics = PerformanceMetrics(
                requests_per_second=len(requests) / (processing_time / 1000),
                average_latency_ms=processing_time / len(requests),
                p95_latency_ms=processing_time * 1.2,  # Estimate
                p99_latency_ms=processing_time * 1.5,  # Estimate
                queue_depth=sum(q.qsize() for q in self._request_queues.values()),
                batch_efficiency=min(1.0, len(requests) / config.batch_size),
                memory_usage_gb=config.memory_limit_gb * 0.8,  # Estimate
                gpu_utilization=0.5,  # Placeholder
                neural_engine_utilization=0.3,  # Placeholder
            )

            await self.optimizer.update_performance_metrics(model_id, metrics)

        except Exception as e:
            logger.error(f"Inference execution failed for {model_id}: {e}")
            # Handle failed requests
            for request in requests:
                if request.callback:
                    result = InferenceResult(
                        request_id=request.request_id,
                        model_id=model_id,
                        result=None,
                        processing_time_ms=0.0,
                        queue_time_ms=0.0,
                        batch_size=len(requests),
                        hardware_used=AccelerationType.CPU_ONLY,
                        success=False,
                        error=str(e),
                    )

                    try:
                        if asyncio.iscoroutinefunction(request.callback):
                            await request.callback(result)
                        else:
                            request.callback(result)
                    except Exception as callback_error:
                        logger.error(
                            f"Callback error for failed request {request.request_id}: {callback_error}"
                        )

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status and metrics."""
        return {
            "queue_depths": {
                priority.name: queue.qsize() for priority, queue in self._request_queues.items()
            },
            "active_requests": len(self._active_requests),
            "processing_tasks": len(self._processing_tasks),
            "metrics": self._metrics.copy(),
            "semaphore_available": self._semaphore._value,
            "max_concurrent": self.max_concurrent_requests,
        }

"""
Performance optimization utilities for the LLM Cybersecurity Benchmark system.

This module provides comprehensive performance monitoring, optimization strategies,
and adaptive tuning for maximum efficiency across different hardware configurations.
"""

import asyncio
import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from benchmark.core.logging import get_logger


@dataclass
class PerformanceProfile:
    """Performance profile for different hardware configurations."""

    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    apple_silicon: bool
    architecture: str

    # Optimized settings
    max_concurrent_models: int = field(default=3)
    max_concurrent_evaluations: int = field(default=8)
    batch_size: int = field(default=16)
    memory_threshold_mb: int = field(default=8192)

    # Performance targets
    target_inference_latency_ms: float = field(default=500.0)
    target_throughput_per_sec: float = field(default=10.0)
    target_memory_efficiency: float = field(default=0.8)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    timestamp: str
    inference_latency_ms: float
    throughput_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    concurrent_operations: int
    error_rate: float
    queue_depth: int

    # Efficiency metrics
    memory_efficiency: float = field(default=0.0)
    cpu_efficiency: float = field(default=0.0)
    overall_score: float = field(default=0.0)


class HardwareProfiler:
    """Hardware detection and profiling for optimization."""

    def __init__(self) -> None:
        self.logger = get_logger("hardware_profiler")

    def detect_hardware_profile(self) -> PerformanceProfile:
        """Detect and create hardware performance profile."""

        # Basic system info
        cpu_cores = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Architecture detection
        architecture = platform.machine().lower()
        apple_silicon = architecture in ["arm64", "aarch64"] and platform.system() == "Darwin"

        # GPU detection
        gpu_available = self._detect_gpu()

        # Create optimized profile based on hardware
        profile = self._create_optimized_profile(
            cpu_cores, memory_gb, gpu_available, apple_silicon, architecture
        )

        self.logger.info(
            f"Detected hardware profile: {cpu_cores} cores, {memory_gb:.1f}GB RAM, "
            f"Apple Silicon: {apple_silicon}, GPU: {gpu_available}"
        )

        return profile

    def _detect_gpu(self) -> bool:
        """Detect GPU availability."""
        try:
            import subprocess

            # Try nvidia-smi for NVIDIA GPUs
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Check for Apple Silicon GPU (Metal)
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                return True

            # Check for AMD GPUs on Linux
            try:
                result = subprocess.run(["lspci"], capture_output=True, text=True)
                if "VGA" in result.stdout and ("AMD" in result.stdout or "Radeon" in result.stdout):
                    return True
            except FileNotFoundError:
                pass

            return False

        except Exception:
            return False

    def _create_optimized_profile(
        self,
        cpu_cores: int,
        memory_gb: float,
        gpu_available: bool,
        apple_silicon: bool,
        architecture: str,
    ) -> PerformanceProfile:
        """Create optimized performance profile based on hardware."""

        # Base profile
        profile = PerformanceProfile(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            apple_silicon=apple_silicon,
            architecture=architecture,
        )

        # Memory-based optimizations
        if memory_gb >= 32:
            # High-memory configuration
            profile.max_concurrent_models = 8
            profile.max_concurrent_evaluations = 32
            profile.batch_size = 32
            profile.memory_threshold_mb = 24576  # 24GB
            profile.target_throughput_per_sec = 50.0

        elif memory_gb >= 16:
            # Standard configuration
            profile.max_concurrent_models = 5
            profile.max_concurrent_evaluations = 16
            profile.batch_size = 16
            profile.memory_threshold_mb = 12288  # 12GB
            profile.target_throughput_per_sec = 25.0

        else:
            # Limited memory configuration
            profile.max_concurrent_models = 2
            profile.max_concurrent_evaluations = 8
            profile.batch_size = 8
            profile.memory_threshold_mb = 6144  # 6GB
            profile.target_throughput_per_sec = 10.0

        # CPU-based optimizations
        if cpu_cores >= 12:
            # High-performance CPU
            profile.max_concurrent_evaluations *= 2
            profile.target_inference_latency_ms = 200.0

        elif cpu_cores >= 8:
            # Standard CPU
            profile.target_inference_latency_ms = 350.0

        else:
            # Limited CPU
            profile.max_concurrent_evaluations //= 2
            profile.target_inference_latency_ms = 800.0

        # Apple Silicon optimizations
        if apple_silicon:
            profile.batch_size = min(profile.batch_size * 2, 64)  # Better batch processing
            profile.target_inference_latency_ms *= 0.7  # Faster inference expected
            profile.memory_threshold_mb = int(
                profile.memory_threshold_mb * 1.2
            )  # Better memory efficiency

        # GPU optimizations
        if gpu_available:
            profile.max_concurrent_models += 2
            profile.target_inference_latency_ms *= 0.5  # Much faster with GPU
            profile.target_throughput_per_sec *= 3.0

        return profile


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""

    def __init__(self, profile: PerformanceProfile):
        self.profile = profile
        self.logger = get_logger("performance_monitor")
        self.metrics_history: list[PerformanceMetrics] = []
        self.alert_thresholds = {
            "memory_usage_percent": 85.0,
            "cpu_usage_percent": 90.0,
            "error_rate_percent": 5.0,
            "latency_degradation_factor": 2.0,
        }

    def collect_metrics(
        self,
        inference_latency_ms: float = 0.0,
        throughput_per_sec: float = 0.0,
        concurrent_operations: int = 0,
        error_rate: float = 0.0,
        queue_depth: int = 0,
    ) -> PerformanceMetrics:
        """Collect current performance metrics."""

        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # GPU metrics (simplified)
        gpu_usage = self._get_gpu_usage()

        # Calculate efficiency metrics
        memory_efficiency = 1.0 - (memory.used / memory.total)
        cpu_efficiency = max(0.0, 1.0 - (cpu_percent / 100.0))

        # Overall performance score
        latency_score = max(
            0.0, 1.0 - (inference_latency_ms / (self.profile.target_inference_latency_ms * 2))
        )
        throughput_score = min(1.0, throughput_per_sec / self.profile.target_throughput_per_sec)
        error_score = max(0.0, 1.0 - (error_rate * 10))  # Penalize errors heavily

        overall_score = (
            latency_score + throughput_score + error_score + memory_efficiency + cpu_efficiency
        ) / 5.0

        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            inference_latency_ms=inference_latency_ms,
            throughput_per_sec=throughput_per_sec,
            memory_usage_mb=memory.used / (1024**2),
            cpu_usage_percent=cpu_percent,
            gpu_usage_percent=gpu_usage,
            concurrent_operations=concurrent_operations,
            error_rate=error_rate,
            queue_depth=queue_depth,
            memory_efficiency=memory_efficiency,
            cpu_efficiency=cpu_efficiency,
            overall_score=overall_score,
        )

        self.metrics_history.append(metrics)
        self._check_alerts(metrics)

        # Keep history manageable
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]

        return metrics

    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage (simplified implementation)."""
        try:
            if self.profile.apple_silicon:
                # Apple Silicon GPU usage is harder to measure directly
                # Use CPU usage as proxy for now
                return float(psutil.cpu_percent()) * 0.3
            else:
                # For other GPUs, would integrate with nvidia-ml-py or similar
                return 0.0
        except Exception:
            return 0.0

    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts."""

        alerts = []

        # Memory usage alert
        memory_percent = (metrics.memory_usage_mb / (self.profile.memory_gb * 1024)) * 100
        if memory_percent > self.alert_thresholds["memory_usage_percent"]:
            alerts.append(f"High memory usage: {memory_percent:.1f}%")

        # CPU usage alert
        if metrics.cpu_usage_percent > self.alert_thresholds["cpu_usage_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")

        # Error rate alert
        if metrics.error_rate > self.alert_thresholds["error_rate_percent"] / 100:
            alerts.append(f"High error rate: {metrics.error_rate:.1%}")

        # Latency degradation alert
        if len(self.metrics_history) > 10:
            recent_latencies = [m.inference_latency_ms for m in self.metrics_history[-10:]]
            baseline_latency = self.profile.target_inference_latency_ms
            avg_recent_latency = statistics.mean(recent_latencies)

            if (
                avg_recent_latency
                > baseline_latency * self.alert_thresholds["latency_degradation_factor"]
            ):
                alerts.append(
                    f"Latency degradation: {avg_recent_latency:.1f}ms (target: {baseline_latency:.1f}ms)"
                )

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Performance Alert: {alert}")

    def get_performance_summary(self, hours_back: int = 1) -> dict[str, Any]:
        """Get performance summary for the specified time period."""

        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_metrics = [
            m for m in self.metrics_history if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        if not recent_metrics:
            return {"error": "No metrics available for the specified period"}

        # Calculate statistics
        latencies = [m.inference_latency_ms for m in recent_metrics]
        throughputs = [m.throughput_per_sec for m in recent_metrics]
        memory_usages = [m.memory_usage_mb for m in recent_metrics]
        cpu_usages = [m.cpu_usage_percent for m in recent_metrics]
        scores = [m.overall_score for m in recent_metrics]

        return {
            "time_period_hours": hours_back,
            "sample_count": len(recent_metrics),
            "latency_stats": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
                "min": min(latencies),
                "max": max(latencies),
            },
            "throughput_stats": {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "max": max(throughputs),
            },
            "resource_usage": {
                "avg_memory_mb": statistics.mean(memory_usages),
                "peak_memory_mb": max(memory_usages),
                "avg_cpu_percent": statistics.mean(cpu_usages),
                "peak_cpu_percent": max(cpu_usages),
            },
            "performance_score": {
                "current": scores[-1] if scores else 0.0,
                "average": statistics.mean(scores),
                "trend": "improving" if len(scores) > 5 and scores[-5:] > scores[:5] else "stable",
            },
            "targets": {
                "latency_target_ms": self.profile.target_inference_latency_ms,
                "throughput_target": self.profile.target_throughput_per_sec,
                "meeting_latency_target": statistics.mean(latencies)
                <= self.profile.target_inference_latency_ms,
                "meeting_throughput_target": statistics.mean(throughputs)
                >= self.profile.target_throughput_per_sec,
            },
        }

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class AdaptiveOptimizer:
    """Adaptive performance optimization based on runtime metrics."""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.profile = monitor.profile
        self.logger = get_logger("adaptive_optimizer")
        self.optimization_history: list[dict[str, Any]] = []

    async def optimize_performance(self, service_configs: dict[str, Any]) -> dict[str, Any]:
        """Perform adaptive optimization based on current performance."""

        # Get recent performance data
        summary = self.monitor.get_performance_summary(hours_back=1)

        if "error" in summary:
            return service_configs  # No optimization possible

        optimized_configs = service_configs.copy()
        optimizations_applied = []

        # Memory optimization
        if summary["resource_usage"]["peak_memory_mb"] > self.profile.memory_threshold_mb:
            # Reduce concurrent operations
            if "max_concurrent_models" in optimized_configs:
                new_value = max(1, optimized_configs["max_concurrent_models"] - 1)
                optimized_configs["max_concurrent_models"] = new_value
                optimizations_applied.append(f"Reduced max_concurrent_models to {new_value}")

            if "batch_size" in optimized_configs:
                new_value = max(4, optimized_configs["batch_size"] // 2)
                optimized_configs["batch_size"] = new_value
                optimizations_applied.append(f"Reduced batch_size to {new_value}")

        # Latency optimization
        if not summary["targets"]["meeting_latency_target"]:
            current_latency = summary["latency_stats"]["mean"]
            target_latency = self.profile.target_inference_latency_ms

            if current_latency > target_latency * 1.5:
                # Significant latency issues
                if "max_concurrent_evaluations" in optimized_configs:
                    new_value = max(2, optimized_configs["max_concurrent_evaluations"] // 2)
                    optimized_configs["max_concurrent_evaluations"] = new_value
                    optimizations_applied.append(
                        f"Reduced max_concurrent_evaluations to {new_value}"
                    )

                # Enable more aggressive caching
                optimized_configs["enable_aggressive_caching"] = True
                optimizations_applied.append("Enabled aggressive caching")

        # Throughput optimization
        if (
            summary["targets"]["meeting_throughput_target"]
            and summary["resource_usage"]["avg_cpu_percent"] < 60
        ):
            # We're meeting targets with CPU headroom - can increase throughput
            if "max_concurrent_evaluations" in optimized_configs:
                new_value = min(
                    self.profile.max_concurrent_evaluations * 2,
                    optimized_configs["max_concurrent_evaluations"] + 2,
                )
                optimized_configs["max_concurrent_evaluations"] = new_value
                optimizations_applied.append(f"Increased max_concurrent_evaluations to {new_value}")

            if "batch_size" in optimized_configs and optimized_configs["batch_size"] < 64:
                new_value = min(64, optimized_configs["batch_size"] * 2)
                optimized_configs["batch_size"] = new_value
                optimizations_applied.append(f"Increased batch_size to {new_value}")

        # Error rate optimization
        error_rates = [m.error_rate for m in self.monitor.metrics_history[-10:]]
        if error_rates and statistics.mean(error_rates) > 0.02:  # >2% error rate
            # Reduce load to improve stability
            for param in ["max_concurrent_models", "max_concurrent_evaluations"]:
                if param in optimized_configs:
                    new_value = max(1, optimized_configs[param] - 1)
                    optimized_configs[param] = new_value
                    optimizations_applied.append(f"Reduced {param} to {new_value} for stability")

        # Log optimizations
        if optimizations_applied:
            self.logger.info(f"Applied optimizations: {', '.join(optimizations_applied)}")

            # Record optimization
            self.optimization_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "optimizations": optimizations_applied,
                    "performance_before": summary,
                    "configs_after": optimized_configs,
                }
            )

        return optimized_configs

    async def auto_tune_services(self, services: dict[str, Any]) -> dict[str, Any]:
        """Automatically tune service configurations for optimal performance."""

        optimized_services = {}

        for service_name, service in services.items():
            if hasattr(service, "get_current_config"):
                current_config = await service.get_current_config()
                optimized_config = await self.optimize_performance(current_config)

                # Apply optimizations if they differ significantly
                if self._configs_differ_significantly(current_config, optimized_config):
                    await service.update_configuration(optimized_config)
                    self.logger.info(f"Auto-tuned {service_name} configuration")

                optimized_services[service_name] = optimized_config

        return optimized_services

    def _configs_differ_significantly(
        self, config1: dict[str, Any], config2: dict[str, Any]
    ) -> bool:
        """Check if configurations differ significantly enough to warrant update."""

        significant_params = [
            "max_concurrent_models",
            "max_concurrent_evaluations",
            "batch_size",
            "memory_threshold_mb",
        ]

        for param in significant_params:
            if param in config1 and param in config2:
                val1, val2 = config1[param], config2[param]
                if abs(val1 - val2) / max(val1, val2) > 0.1:  # 10% difference
                    return True

        return False


class BenchmarkRunner:
    """Comprehensive benchmark runner for performance validation."""

    def __init__(self, profile: PerformanceProfile):
        self.profile = profile
        self.logger = get_logger("benchmark_runner")

    async def run_comprehensive_benchmark(
        self, model_service: Any, eval_service: Any, test_duration_minutes: int = 10
    ) -> dict[str, Any]:
        """Run comprehensive performance benchmark."""

        self.logger.info(f"Starting {test_duration_minutes}-minute comprehensive benchmark")

        benchmark_results: dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "duration_minutes": test_duration_minutes,
            "hardware_profile": self.profile.__dict__,
            "test_results": {},
        }

        # Test scenarios
        scenarios = [
            ("single_model_inference", self._test_single_model_inference),
            ("concurrent_model_inference", self._test_concurrent_model_inference),
            ("batch_processing", self._test_batch_processing),
            ("memory_stress_test", self._test_memory_stress),
            ("sustained_load_test", self._test_sustained_load),
        ]

        for scenario_name, test_func in scenarios:
            self.logger.info(f"Running {scenario_name} benchmark")

            try:
                start_time = time.time()
                result = await test_func(model_service, eval_service)
                execution_time = time.time() - start_time

                benchmark_results["test_results"][scenario_name] = {
                    "status": "completed",
                    "execution_time_seconds": execution_time,
                    "results": result,
                }

                self.logger.info(f"Completed {scenario_name} in {execution_time:.2f}s")

            except Exception as e:
                self.logger.error(f"Failed {scenario_name}: {e}")
                benchmark_results["test_results"][scenario_name] = {
                    "status": "failed",
                    "error": str(e),
                }

        benchmark_results["end_time"] = datetime.now().isoformat()

        # Generate performance score
        benchmark_results["overall_score"] = self._calculate_overall_score(
            benchmark_results["test_results"]
        )

        return benchmark_results

    async def _test_single_model_inference(
        self, model_service: Any, eval_service: Any
    ) -> dict[str, Any]:
        """Test single model inference performance."""

        # Load a test model (mock)
        await model_service.load_model("test_model")

        # Prepare test inputs
        test_inputs = [f"Test cybersecurity analysis {i}" for i in range(100)]

        # Measure inference performance
        start_time = time.time()

        results = await model_service.predict("test_model", test_inputs)

        end_time = time.time()
        duration = end_time - start_time

        return {
            "total_inferences": len(test_inputs),
            "duration_seconds": duration,
            "throughput_per_second": len(test_inputs) / duration,
            "average_latency_ms": (duration / len(test_inputs)) * 1000,
            "success": results.success if hasattr(results, "success") else True,
        }

    async def _test_concurrent_model_inference(
        self, model_service: Any, eval_service: Any
    ) -> dict[str, Any]:
        """Test concurrent model inference performance."""

        # Load multiple models
        models = ["model_1", "model_2", "model_3"]
        for model_id in models:
            await model_service.load_model(model_id)

        # Create concurrent tasks
        test_inputs = [f"Concurrent test {i}" for i in range(50)]

        tasks = []
        for model_id in models:
            task = model_service.predict(model_id, test_inputs)
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        successful_results = [r for r in results if not isinstance(r, Exception)]

        return {
            "concurrent_models": len(models),
            "total_inferences": len(test_inputs) * len(models),
            "duration_seconds": end_time - start_time,
            "success_rate": len(successful_results) / len(results),
            "concurrent_throughput": (len(test_inputs) * len(models)) / (end_time - start_time),
        }

    async def _test_batch_processing(self, model_service: Any, eval_service: Any) -> dict[str, Any]:
        """Test batch processing performance."""

        await model_service.load_model("batch_test_model")

        # Test different batch sizes
        batch_sizes = [1, 8, 16, 32, 64]
        batch_results = {}

        for batch_size in batch_sizes:
            test_inputs = [f"Batch test {i}" for i in range(batch_size)]

            start_time = time.time()
            await model_service.predict("batch_test_model", test_inputs)
            end_time = time.time()

            duration = end_time - start_time

            batch_results[f"batch_size_{batch_size}"] = {
                "duration_seconds": duration,
                "throughput_per_second": batch_size / duration,
                "latency_per_item_ms": (duration / batch_size) * 1000,
            }

        return batch_results

    async def _test_memory_stress(self, model_service: Any, eval_service: Any) -> dict[str, Any]:
        """Test memory usage under stress."""

        initial_memory = psutil.virtual_memory().used / (1024**2)

        # Load maximum number of models
        max_models = self.profile.max_concurrent_models
        models = [f"memory_test_model_{i}" for i in range(max_models)]

        for model_id in models:
            await model_service.load_model(model_id)

        peak_memory = psutil.virtual_memory().used / (1024**2)

        # Test memory during inference
        large_inputs = [f"Large memory test input {i}" for i in range(200)]

        await model_service.predict(models[0], large_inputs)

        final_memory = psutil.virtual_memory().used / (1024**2)

        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": peak_memory - initial_memory,
            "models_loaded": len(models),
            "memory_per_model_mb": (peak_memory - initial_memory) / max(len(models), 1),
        }

    async def _test_sustained_load(self, model_service: Any, eval_service: Any) -> dict[str, Any]:
        """Test sustained load performance."""

        await model_service.load_model("sustained_test_model")

        duration_seconds = 60  # 1 minute sustained test
        total_requests = 0
        successful_requests = 0
        error_count = 0

        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            try:
                test_input = [f"Sustained test {total_requests}"]
                result = await model_service.predict("sustained_test_model", test_input)

                if hasattr(result, "success") and result.success:
                    successful_requests += 1
                else:
                    error_count += 1

                total_requests += 1

                # Small delay to simulate realistic usage
                await asyncio.sleep(0.1)

            except Exception:
                error_count += 1
                total_requests += 1

        return {
            "duration_seconds": duration_seconds,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_count": error_count,
            "success_rate": successful_requests / max(total_requests, 1),
            "average_requests_per_second": total_requests / duration_seconds,
            "successful_requests_per_second": successful_requests / duration_seconds,
        }

    def _calculate_overall_score(self, test_results: dict[str, Any]) -> float:
        """Calculate overall performance score from test results."""

        scores: list[float] = []

        # Single model inference score
        if (
            "single_model_inference" in test_results
            and test_results["single_model_inference"]["status"] == "completed"
        ):
            results = test_results["single_model_inference"]["results"]
            throughput = results.get("throughput_per_second", 0)
            target_throughput = self.profile.target_throughput_per_sec
            score = min(1.0, throughput / target_throughput)
            scores.append(score)

        # Concurrent inference score
        if (
            "concurrent_model_inference" in test_results
            and test_results["concurrent_model_inference"]["status"] == "completed"
        ):
            results = test_results["concurrent_model_inference"]["results"]
            success_rate = results.get("success_rate", 0)
            scores.append(success_rate)

        # Memory efficiency score
        if (
            "memory_stress_test" in test_results
            and test_results["memory_stress_test"]["status"] == "completed"
        ):
            results = test_results["memory_stress_test"]["results"]
            memory_per_model = results.get("memory_per_model_mb", float("inf"))
            # Score based on memory efficiency (lower is better)
            score = max(0.0, 1.0 - (memory_per_model / 2048))  # 2GB per model is baseline
            scores.append(score)

        # Sustained load score
        if (
            "sustained_load_test" in test_results
            and test_results["sustained_load_test"]["status"] == "completed"
        ):
            results = test_results["sustained_load_test"]["results"]
            success_rate = results.get("success_rate", 0)
            scores.append(success_rate)

        # Calculate weighted average
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0


def create_performance_optimizer() -> tuple[
    PerformanceProfile, PerformanceMonitor, AdaptiveOptimizer
]:
    """Factory function to create optimized performance monitoring stack."""

    # Detect hardware and create profile
    profiler: HardwareProfiler = HardwareProfiler()
    profile = profiler.detect_hardware_profile()

    # Create monitor and optimizer
    monitor = PerformanceMonitor(profile)
    optimizer = AdaptiveOptimizer(monitor)

    return profile, monitor, optimizer


async def export_performance_report(benchmark_results: dict[str, Any], output_path: Path) -> None:
    """Export comprehensive performance report."""

    report = {
        "generation_time": datetime.now().isoformat(),
        "benchmark_results": benchmark_results,
        "recommendations": _generate_performance_recommendations(benchmark_results),
        "optimization_suggestions": _generate_optimization_suggestions(benchmark_results),
    }

    # Export as JSON
    with open(output_path / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Export as human-readable report
    markdown_report = _generate_markdown_report(report)
    with open(output_path / "performance_report.md", "w") as f:
        f.write(markdown_report)


def _generate_performance_recommendations(benchmark_results: dict[str, Any]) -> list[str]:
    """Generate performance recommendations based on benchmark results."""

    recommendations = []

    if "test_results" not in benchmark_results:
        return ["Unable to generate recommendations - no test results available"]

    test_results = benchmark_results["test_results"]

    # Memory recommendations
    if "memory_stress_test" in test_results:
        memory_results = test_results["memory_stress_test"].get("results", {})
        memory_per_model = memory_results.get("memory_per_model_mb", 0)

        if memory_per_model > 2048:
            recommendations.append(
                f"High memory usage per model ({memory_per_model:.0f}MB). "
                "Consider using smaller models or enabling model quantization."
            )
        elif memory_per_model < 512:
            recommendations.append(
                "Efficient memory usage detected. You can potentially load more models concurrently."
            )

    # Throughput recommendations
    if "single_model_inference" in test_results:
        inference_results = test_results["single_model_inference"].get("results", {})
        throughput = inference_results.get("throughput_per_second", 0)

        if throughput < 5:
            recommendations.append(
                f"Low inference throughput ({throughput:.1f} req/sec). "
                "Consider using faster models, enabling GPU acceleration, or optimizing batch sizes."
            )
        elif throughput > 50:
            recommendations.append(
                f"Excellent throughput ({throughput:.1f} req/sec). "
                "Your system is well-optimized for high-performance inference."
            )

    # Concurrent processing recommendations
    if "concurrent_model_inference" in test_results:
        concurrent_results = test_results["concurrent_model_inference"].get("results", {})
        success_rate = concurrent_results.get("success_rate", 0)

        if success_rate < 0.9:
            recommendations.append(
                f"Low success rate in concurrent processing ({success_rate:.1%}). "
                "Consider reducing concurrent model limits or increasing memory allocation."
            )

    # Sustained load recommendations
    if "sustained_load_test" in test_results:
        sustained_results = test_results["sustained_load_test"].get("results", {})
        sustained_success = sustained_results.get("success_rate", 0)

        if sustained_success < 0.95:
            recommendations.append(
                f"Sustained load success rate is suboptimal ({sustained_success:.1%}). "
                "This may indicate memory leaks or resource exhaustion under load."
            )

    if not recommendations:
        recommendations.append(
            "Performance is within expected parameters. No specific recommendations."
        )

    return recommendations


def _generate_optimization_suggestions(benchmark_results: dict[str, Any]) -> list[str]:
    """Generate specific optimization suggestions."""

    suggestions = []

    # Hardware-specific suggestions
    hardware_profile = benchmark_results.get("hardware_profile", {})

    if hardware_profile.get("apple_silicon", False):
        suggestions.append(
            "Enable Apple Silicon optimizations for better memory efficiency and performance."
        )

    if hardware_profile.get("gpu_available", False):
        suggestions.append(
            "Consider GPU-accelerated models for significant performance improvements."
        )

    if hardware_profile.get("memory_gb", 0) >= 32:
        suggestions.append(
            "High memory system detected - increase concurrent model limits and batch sizes."
        )

    # Performance-specific suggestions
    if "test_results" in benchmark_results:
        overall_score = benchmark_results.get("overall_score", 0)

        if overall_score < 0.6:
            suggestions.extend(
                [
                    "Consider upgrading hardware for better performance",
                    "Enable performance monitoring and adaptive optimization",
                    "Review model selection - use smaller, faster models for real-time scenarios",
                ]
            )
        elif overall_score > 0.8:
            suggestions.extend(
                [
                    "Excellent performance detected - consider increasing workload capacity",
                    "Your system can handle more concurrent operations",
                    "Consider using larger, more accurate models",
                ]
            )

    return suggestions


def _generate_markdown_report(report: dict[str, Any]) -> str:
    """Generate human-readable markdown performance report."""

    markdown = f"""# Performance Benchmark Report

Generated: {report["generation_time"]}

## Hardware Profile

"""

    hardware = report["benchmark_results"].get("hardware_profile", {})
    markdown += f"""
- **CPU Cores**: {hardware.get("cpu_cores", "Unknown")}
- **Memory**: {hardware.get("memory_gb", "Unknown"):.1f} GB
- **Architecture**: {hardware.get("architecture", "Unknown")}
- **Apple Silicon**: {hardware.get("apple_silicon", False)}
- **GPU Available**: {hardware.get("gpu_available", False)}

## Benchmark Results

"""

    test_results = report["benchmark_results"].get("test_results", {})
    overall_score = report["benchmark_results"].get("overall_score", 0)

    markdown += f"**Overall Performance Score**: {overall_score:.2f}/1.00\n\n"

    for test_name, test_data in test_results.items():
        status = test_data.get("status", "unknown")
        markdown += f"### {test_name.replace('_', ' ').title()}\n\n"
        markdown += f"**Status**: {status}\n\n"

        if status == "completed" and "results" in test_data:
            results = test_data["results"]
            for key, value in results.items():
                if isinstance(value, int | float):
                    if "time" in key.lower() or "duration" in key.lower():
                        markdown += f"- **{key.replace('_', ' ').title()}**: {value:.3f}\n"
                    elif "rate" in key.lower():
                        markdown += f"- **{key.replace('_', ' ').title()}**: {value:.1%}\n"
                    else:
                        markdown += f"- **{key.replace('_', ' ').title()}**: {value:.2f}\n"
                else:
                    markdown += f"- **{key.replace('_', ' ').title()}**: {value}\n"

        markdown += "\n"

    markdown += "## Recommendations\n\n"
    for rec in report.get("recommendations", []):
        markdown += f"- {rec}\n"

    markdown += "\n## Optimization Suggestions\n\n"
    for suggestion in report.get("optimization_suggestions", []):
        markdown += f"- {suggestion}\n"

    return markdown

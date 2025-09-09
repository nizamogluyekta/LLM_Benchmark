"""
Data Service for the LLM Cybersecurity Benchmark system.

This service manages dataset loading, preprocessing, and caching with a plugin
architecture for different data sources.
"""

import asyncio
import gc
import gzip
import hashlib
import json
import pickle
import platform
import statistics
import threading
import time
import weakref
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from benchmark.core.base import BaseService, HealthCheck, ServiceResponse, ServiceStatus
from benchmark.core.config import DatasetConfig
from benchmark.core.exceptions import DataLoadingError, ErrorCode
from benchmark.core.logging import get_logger
from benchmark.data.loaders import HuggingFaceDataLoader, KaggleDataLoader
from benchmark.data.models import DataQualityReport, DatasetStatistics
from benchmark.interfaces.data_interfaces import (
    DataBatch,
    DataCache,
    DataFormat,
    DataLoader,
    DataPreprocessor,
    DataSample,
    Dataset,
    DatasetInfo,
    DataSource,
    DataSplits,
    DataSplitter,
    DataValidator,
)


@dataclass
class MemoryStatus:
    """Memory usage status information."""

    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float
    process_memory_mb: float
    cache_memory_mb: float


@dataclass
class ProgressReport:
    """Progress report for long operations."""

    operation: str
    current: int
    total: int
    percentage: float
    elapsed_seconds: float
    estimated_remaining_seconds: float | None = None

    @property
    def is_complete(self) -> bool:
        return self.current >= self.total


class MemoryManager:
    """Monitor and manage memory usage with hardware-specific optimizations."""

    def __init__(self, max_memory_gb: float = 8.0):
        """
        Initialize memory manager.

        Args:
            max_memory_gb: Maximum memory usage in GB (conservative for M4 Pro)
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.logger = get_logger("memory_manager")
        self._dataset_refs: weakref.WeakValueDictionary[str, Dataset] = (
            weakref.WeakValueDictionary()
        )

        # Auto-detect hardware if not specified
        if max_memory_gb == 8.0:  # Default value
            self._auto_configure_memory_limits()

    def _auto_configure_memory_limits(self) -> None:
        """Auto-configure memory limits based on available RAM."""
        try:
            total_memory = psutil.virtual_memory().total
            total_gb = total_memory / (1024**3)

            # Use 60% of available RAM for M-series Macs, 50% for others
            is_apple_silicon = platform.machine().lower() in ["arm64", "aarch64"]
            usage_ratio = 0.6 if is_apple_silicon else 0.5

            self.max_memory_gb = min(total_gb * usage_ratio, 16.0)  # Cap at 16GB
            self.max_memory_bytes = int(self.max_memory_gb * 1024 * 1024 * 1024)

            self.logger.info(
                f"Auto-configured memory limit: {self.max_memory_gb:.1f}GB "
                f"(Apple Silicon: {is_apple_silicon})"
            )
        except Exception as e:
            self.logger.warning(f"Failed to auto-configure memory limits: {e}")

    async def get_memory_status(self) -> MemoryStatus:
        """Get current memory usage status."""
        vm = psutil.virtual_memory()
        process = psutil.Process()

        total_gb = vm.total / (1024**3)
        used_gb = vm.used / (1024**3)
        available_gb = vm.available / (1024**3)
        usage_percent = vm.percent

        process_memory_mb = process.memory_info().rss / (1024**2)

        # Estimate cache memory from tracked datasets
        cache_memory_mb = sum(
            self._estimate_dataset_size(dataset) / (1024**2)
            for dataset in self._dataset_refs.values()
        )

        return MemoryStatus(
            total_gb=total_gb,
            used_gb=used_gb,
            available_gb=available_gb,
            usage_percent=usage_percent,
            process_memory_mb=process_memory_mb,
            cache_memory_mb=cache_memory_mb,
        )

    async def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high."""
        status = await self.get_memory_status()
        return (
            status.usage_percent > 80
            or status.process_memory_mb > self.max_memory_gb * 1024 * 0.8
            or status.available_gb < 1.0
        )

    async def cleanup_unused_datasets(self) -> int:
        """Clean up unused datasets and run garbage collection."""
        initial_count = len(self._dataset_refs)

        # Force garbage collection
        collected = gc.collect()

        # The weak references will automatically clean up unreferenced datasets
        final_count = len(self._dataset_refs)
        cleaned_count = initial_count - final_count

        self.logger.info(
            f"Cleaned up {cleaned_count} datasets, " f"collected {collected} objects via GC"
        )

        return cleaned_count

    def track_dataset(self, dataset_id: str, dataset: Dataset) -> None:
        """Track a dataset for memory management."""
        self._dataset_refs[dataset_id] = dataset

    def _estimate_dataset_size(self, dataset: Dataset) -> int:
        """Estimate dataset memory usage in bytes."""
        try:
            # Rough estimation: average sample size * number of samples + overhead
            if dataset.samples:
                sample_sizes = []
                for sample in dataset.samples[: min(10, len(dataset.samples))]:
                    size = len(str(sample.data).encode("utf-8"))
                    size += len(str(sample.label).encode("utf-8")) if sample.label else 0
                    size += sum(len(str(v).encode("utf-8")) for v in sample.metadata.values())
                    sample_sizes.append(size)

                avg_sample_size = sum(sample_sizes) / len(sample_sizes)
                return int(avg_sample_size * len(dataset.samples) * 1.5)  # 50% overhead
        except Exception:
            pass
        return 1024 * 1024  # Default 1MB estimate


class StreamingDataLoader:
    """Memory-efficient streaming loader for large datasets."""

    def __init__(self, chunk_size: int = 1000, max_memory_mb: int = 512):
        """
        Initialize streaming loader.

        Args:
            chunk_size: Number of samples to process at once
            max_memory_mb: Maximum memory usage for streaming operations
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.logger = get_logger("streaming_loader")

    async def stream_batches(
        self,
        loader: DataLoader,
        config: DatasetConfig,
        batch_size: int,
        progress_callback: Callable[..., Any] | None = None,
    ) -> AsyncIterator[DataBatch]:
        """
        Stream dataset batches for memory-efficient processing.

        Args:
            loader: Data loader to use
            config: Dataset configuration
            batch_size: Size of each batch
            progress_callback: Optional progress callback

        Yields:
            DataBatch: Batches of samples
        """
        # For now, implement basic streaming by loading full dataset and batching
        # In a production system, this would be implemented with true streaming
        dataset = await loader.load(config)

        total_samples = len(dataset.samples)
        total_batches = (total_samples + batch_size - 1) // batch_size

        for i in range(0, total_samples, batch_size):
            batch_samples = dataset.samples[i : i + batch_size]
            batch_id = f"{config.name}_stream_batch_{i // batch_size}"

            batch = DataBatch(
                batch_id=batch_id,
                samples=batch_samples,
                batch_info={
                    "dataset_id": config.name,
                    "batch_size": len(batch_samples),
                    "offset": i,
                    "total_samples": total_samples,
                    "batch_index": i // batch_size,
                    "total_batches": total_batches,
                    "streaming": True,
                },
            )

            if progress_callback:
                progress = ProgressReport(
                    operation=f"Streaming {config.name}",
                    current=i // batch_size + 1,
                    total=total_batches,
                    percentage=(i // batch_size + 1) / total_batches * 100,
                    elapsed_seconds=0.0,  # Would be calculated in real implementation
                )
                await progress_callback(progress)

            yield batch


class CompressedCache(DataCache):
    """Compressed cache implementation for better memory efficiency."""

    def __init__(self, max_size: int = 50, max_memory_mb: int = 256, ttl_default: int = 3600):
        """
        Initialize compressed cache.

        Args:
            max_size: Maximum number of cached items
            max_memory_mb: Maximum memory usage in MB
            ttl_default: Default TTL in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_default = ttl_default

        self._cache: dict[str, dict[str, Any]] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()
        self._memory_usage = 0

        self.logger = get_logger("compressed_cache")

    async def get(self, cache_key: str) -> Any | None:
        """Get and decompress data from cache."""
        with self._lock:
            if cache_key not in self._cache:
                return None

            entry = self._cache[cache_key]

            # Check TTL
            if time.time() > entry["expires_at"]:
                await self._remove_entry(cache_key)
                return None

            # Update access order for LRU
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)

            # Decompress data
            try:
                compressed_data = entry["data"]
                decompressed = gzip.decompress(compressed_data)
                data = pickle.loads(decompressed)
                return data
            except Exception as e:
                self.logger.error(f"Failed to decompress cache entry {cache_key}: {e}")
                await self._remove_entry(cache_key)
                return None

    async def set(self, cache_key: str, data: Any, ttl: int | None = None) -> None:
        """Compress and store data in cache."""
        if ttl is None:
            ttl = self.ttl_default

        try:
            # Compress data
            pickled_data = pickle.dumps(data)
            compressed_data = gzip.compress(pickled_data, compresslevel=6)
            compressed_size = len(compressed_data)

            with self._lock:
                # Remove existing entry if present
                if cache_key in self._cache:
                    await self._remove_entry(cache_key)

                # Check if we need to evict entries
                while (
                    len(self._cache) >= self.max_size
                    or self._memory_usage + compressed_size > self.max_memory_bytes
                ):
                    if not self._access_order:
                        break
                    oldest_key = self._access_order[0]
                    await self._remove_entry(oldest_key)

                # Add new entry
                entry = {
                    "data": compressed_data,
                    "expires_at": time.time() + ttl,
                    "size": compressed_size,
                    "created_at": time.time(),
                    "original_size": len(pickled_data),
                    "compression_ratio": len(pickled_data) / compressed_size
                    if compressed_size > 0
                    else 1.0,
                }

                self._cache[cache_key] = entry
                self._access_order.append(cache_key)
                self._memory_usage += compressed_size

                self.logger.debug(
                    f"Compressed cache entry {cache_key}: "
                    f"{entry['original_size']} -> {compressed_size} bytes "
                    f"(ratio: {entry['compression_ratio']:.2f})"
                )

        except Exception as e:
            self.logger.error(f"Failed to compress and cache data for {cache_key}: {e}")

    async def delete(self, cache_key: str) -> bool:
        """Delete data from cache."""
        with self._lock:
            if cache_key in self._cache:
                await self._remove_entry(cache_key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._memory_usage = 0
            self.logger.info("Compressed cache cleared")

    async def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics including compression info."""
        with self._lock:
            total_original_size = sum(entry["original_size"] for entry in self._cache.values())
            total_compressed_size = self._memory_usage
            overall_ratio = (
                total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0
            )

            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self._memory_usage,
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": (self._memory_usage / self.max_memory_bytes) * 100,
                "compression_enabled": True,
                "original_size_bytes": total_original_size,
                "compression_ratio": overall_ratio,
                "space_saved_bytes": total_original_size - total_compressed_size,
                "entries": list(self._cache.keys()),
            }

    async def _remove_entry(self, cache_key: str) -> None:
        """Remove entry from cache."""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            self._memory_usage -= entry["size"]
            del self._cache[cache_key]

            if cache_key in self._access_order:
                self._access_order.remove(cache_key)


class DataServiceOptimizer:
    """Optimize data service performance for specific hardware."""

    def __init__(self, data_service: "DataService"):
        """Initialize optimizer with reference to data service."""
        self.data_service = data_service
        self.logger = get_logger("data_optimizer")
        self._hardware_info: dict[str, Any] = {}

    async def optimize_for_hardware(self, hardware_info: dict[str, Any] | None = None) -> None:
        """
        Optimize data service settings for specific hardware.

        Args:
            hardware_info: Hardware information dict, auto-detected if None
        """
        if hardware_info is None:
            hardware_info = await self._detect_hardware()

        self._hardware_info = hardware_info

        # Optimize cache settings
        await self._optimize_cache_settings()

        # Optimize batch processing
        await self._optimize_batch_settings()

        # Set up memory management
        await self._setup_memory_management()

        self.logger.info(
            f"Optimized data service for: {hardware_info.get('cpu_name', 'Unknown CPU')}"
        )

    async def preload_common_datasets(self, dataset_ids: list[str]) -> None:
        """
        Preload commonly used datasets for faster access.

        Args:
            dataset_ids: List of dataset IDs to preload
        """
        preload_tasks: list[Any] = []

        for dataset_id in dataset_ids:
            # Check if dataset is already loaded
            if dataset_id not in self.data_service._datasets:
                # Would implement actual preloading logic here
                self.logger.info(f"Would preload dataset: {dataset_id}")

        if preload_tasks:
            await asyncio.gather(*preload_tasks, return_exceptions=True)

        self.logger.info(f"Preloaded {len(dataset_ids)} datasets")

    async def _detect_hardware(self) -> dict[str, Any]:
        """Detect hardware information."""
        try:
            cpu_info = {}

            # Basic CPU info
            cpu_info["cpu_count"] = psutil.cpu_count(logical=False)
            cpu_info["cpu_count_logical"] = psutil.cpu_count(logical=True)
            cpu_info["cpu_freq"] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}

            # Memory info
            memory = psutil.virtual_memory()
            cpu_info["memory_total_gb"] = memory.total / (1024**3)

            # Platform info
            cpu_info["platform"] = platform.platform()
            cpu_info["machine"] = platform.machine()
            cpu_info["processor"] = platform.processor()

            # Apple Silicon detection
            cpu_info["is_apple_silicon"] = platform.machine().lower() in ["arm64", "aarch64"]

            # Try to get CPU name on macOS
            if platform.system() == "Darwin":
                try:
                    import subprocess

                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        cpu_info["cpu_name"] = result.stdout.strip()
                except Exception:
                    pass

            return cpu_info

        except Exception as e:
            self.logger.error(f"Failed to detect hardware: {e}")
            return {"error": str(e)}

    async def _optimize_cache_settings(self) -> None:
        """Optimize cache settings based on hardware."""
        memory_gb = self._hardware_info.get("memory_total_gb", 8.0)
        is_apple_silicon = self._hardware_info.get("is_apple_silicon", False)

        if is_apple_silicon and memory_gb >= 16:
            # M-series Macs are efficient with memory, can use more cache
            cache_memory_mb = min(int(memory_gb * 1024 * 0.3), 4096)  # Up to 4GB cache
        else:
            # Conservative settings for other systems
            cache_memory_mb = min(int(memory_gb * 1024 * 0.2), 2048)  # Up to 2GB cache

        # Update cache if it's compressed cache
        if hasattr(self.data_service.cache, "max_memory_bytes"):
            self.data_service.cache.max_memory_bytes = cache_memory_mb * 1024 * 1024

        self.logger.info(f"Optimized cache memory limit: {cache_memory_mb}MB")

    async def _optimize_batch_settings(self) -> None:
        """Optimize batch processing settings."""
        cpu_count = self._hardware_info.get("cpu_count_logical", 4)

        # Optimal batch size based on CPU cores
        optimal_batch_size = max(32, cpu_count * 16)  # 16 samples per core minimum

        # Store optimization settings (would be used by batch processing)
        self.data_service._optimization_settings = {
            "optimal_batch_size": optimal_batch_size,
            "max_concurrent_batches": cpu_count,
            "chunk_size": optimal_batch_size * 2,
        }

        self.logger.info(
            f"Optimized batch settings: size={optimal_batch_size}, " f"concurrent={cpu_count}"
        )

    async def _setup_memory_management(self) -> None:
        """Set up memory management based on hardware."""
        memory_gb = self._hardware_info.get("memory_total_gb", 8.0)

        # Conservative memory limits
        max_memory_gb = min(memory_gb * 0.6, 16.0) if memory_gb >= 16 else memory_gb * 0.5

        # Initialize memory manager if not already present
        if not hasattr(self.data_service, "memory_manager"):
            self.data_service.memory_manager = MemoryManager(max_memory_gb)

        self.logger.info(f"Memory management configured: {max_memory_gb:.1f}GB limit")


class MemoryDataCache(DataCache):
    """In-memory implementation of DataCache with LRU eviction."""

    def __init__(self, max_size: int = 100, max_memory_mb: int = 512, ttl_default: int = 3600):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of cached items
            max_memory_mb: Maximum memory usage in MB
            ttl_default: Default TTL in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_default = ttl_default

        self._cache: dict[str, dict[str, Any]] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()
        self._memory_usage = 0

        self.logger = get_logger("data_cache")

    async def get(self, cache_key: str) -> Any | None:
        """Get data from cache."""
        with self._lock:
            if cache_key not in self._cache:
                return None

            entry = self._cache[cache_key]

            # Check TTL
            if time.time() > entry["expires_at"]:
                await self._remove_entry(cache_key)
                return None

            # Update access order for LRU
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)

            return entry["data"]

    async def set(self, cache_key: str, data: Any, ttl: int | None = None) -> None:
        """Store data in cache."""
        if ttl is None:
            ttl = self.ttl_default

        # Estimate data size
        data_size = self._estimate_size(data)

        with self._lock:
            # Remove existing entry if present
            if cache_key in self._cache:
                await self._remove_entry(cache_key)

            # Check if we need to evict entries
            while (
                len(self._cache) >= self.max_size
                or self._memory_usage + data_size > self.max_memory_bytes
            ):
                if not self._access_order:
                    break
                oldest_key = self._access_order[0]
                await self._remove_entry(oldest_key)

            # Add new entry
            entry = {
                "data": data,
                "expires_at": time.time() + ttl,
                "size": data_size,
                "created_at": time.time(),
            }

            self._cache[cache_key] = entry
            self._access_order.append(cache_key)
            self._memory_usage += data_size

            self.logger.debug(f"Cached data with key: {cache_key}, size: {data_size} bytes")

    async def delete(self, cache_key: str) -> bool:
        """Delete data from cache."""
        with self._lock:
            if cache_key in self._cache:
                await self._remove_entry(cache_key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._memory_usage = 0
            self.logger.info("Cache cleared")

    async def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self._memory_usage,
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": (self._memory_usage / self.max_memory_bytes) * 100,
                "entries": list(self._cache.keys()),
            }

    async def _remove_entry(self, cache_key: str) -> None:
        """Remove entry from cache."""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            self._memory_usage -= entry["size"]
            del self._cache[cache_key]

            if cache_key in self._access_order:
                self._access_order.remove(cache_key)

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            if isinstance(data, str | bytes):
                return len(data.encode("utf-8") if isinstance(data, str) else data)
            elif isinstance(data, int | float):
                return 8
            elif isinstance(data, list | tuple):
                return sum(self._estimate_size(item) for item in data) + 64
            elif isinstance(data, dict):
                return (
                    sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
                    + 64
                )
            elif hasattr(data, "__sizeof__"):
                return int(data.__sizeof__())
            else:
                # Fallback: estimate based on string representation
                return len(str(data).encode("utf-8"))
        except Exception:
            return 1024  # Default estimate


class LocalDataLoader(DataLoader):
    """Data loader for local files."""

    def __init__(self) -> None:
        self.logger = get_logger("local_data_loader")

    async def load(self, config: DatasetConfig) -> Dataset:
        """Load data from local file."""
        try:
            file_path = Path(config.path)

            if not file_path.exists():
                raise DataLoadingError(
                    f"Data file not found: {file_path}",
                    error_code=ErrorCode.DATASET_NOT_FOUND,
                    metadata={"path": str(file_path)},
                )

            # Determine format from file extension if not specified
            format_str = getattr(config, "format", None)
            if not format_str:
                format_str = file_path.suffix.lstrip(".")

            try:
                data_format = DataFormat(format_str.lower())
            except ValueError as e:
                raise DataLoadingError(
                    f"Unsupported data format: {format_str}",
                    error_code=ErrorCode.DATASET_FORMAT_ERROR,
                    metadata={"format": format_str},
                ) from e

            # Load data based on format
            samples = await self._load_by_format(file_path, data_format, config)

            # Create dataset info
            dataset_info = DatasetInfo(
                dataset_id=config.name,
                name=config.name,
                description=getattr(config, "description", None),
                source=DataSource.LOCAL,
                format=data_format,
                size_bytes=file_path.stat().st_size,
                num_samples=len(samples),
                created_at=str(file_path.stat().st_ctime),
                modified_at=str(file_path.stat().st_mtime),
            )

            dataset = Dataset(dataset_id=config.name, info=dataset_info, samples=samples)

            self.logger.info(f"Loaded dataset '{config.name}' with {len(samples)} samples")
            return dataset

        except DataLoadingError:
            raise
        except Exception as e:
            raise DataLoadingError(
                f"Failed to load dataset: {str(e)}",
                error_code=ErrorCode.DATA_PREPROCESSING_FAILED,
                metadata={"config": config.name, "path": config.path},
            ) from e

    async def validate_source(self, config: DatasetConfig) -> bool:
        """Validate that the local file exists and is readable."""
        try:
            file_path = Path(config.path)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False

    def get_supported_formats(self) -> list[DataFormat]:
        """Get supported formats for local loader."""
        return [DataFormat.JSON, DataFormat.JSONL, DataFormat.CSV, DataFormat.TSV]

    def get_source_type(self) -> DataSource:
        """Get source type."""
        return DataSource.LOCAL

    async def _load_by_format(
        self, file_path: Path, data_format: DataFormat, config: DatasetConfig
    ) -> list[DataSample]:
        """Load data based on file format."""
        samples = []

        if data_format == DataFormat.JSONL:
            samples = await self._load_jsonl(file_path, config)
        elif data_format == DataFormat.JSON:
            samples = await self._load_json(file_path, config)
        elif data_format == DataFormat.CSV:
            samples = await self._load_csv(file_path, config)
        elif data_format == DataFormat.TSV:
            samples = await self._load_tsv(file_path, config)
        else:
            raise DataLoadingError(
                f"Format {data_format} not implemented for local loader",
                error_code=ErrorCode.DATASET_FORMAT_ERROR,
            )

        # Apply max_samples limit if specified
        max_samples = getattr(config, "max_samples", None)
        if max_samples and max_samples > 0:
            samples = samples[:max_samples]

        return samples

    async def _load_jsonl(self, file_path: Path, config: DatasetConfig) -> list[DataSample]:
        """Load JSONL format."""
        samples = []

        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = DataSample(
                        sample_id=f"{config.name}_{line_num}",
                        data=data,
                        label=data.get("label"),
                        metadata={"line_number": line_num, "file_path": str(file_path)},
                    )
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue

        return samples

    async def _load_json(self, file_path: Path, config: DatasetConfig) -> list[DataSample]:
        """Load JSON format."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        if isinstance(data, list):
            for idx, item in enumerate(data):
                sample = DataSample(
                    sample_id=f"{config.name}_{idx}",
                    data=item if isinstance(item, dict) else {"data": item},
                    label=item.get("label") if isinstance(item, dict) else None,
                    metadata={"index": idx, "file_path": str(file_path)},
                )
                samples.append(sample)
        else:
            sample = DataSample(
                sample_id=f"{config.name}_0",
                data=data,
                label=data.get("label") if isinstance(data, dict) else None,
                metadata={"file_path": str(file_path)},
            )
            samples.append(sample)

        return samples

    async def _load_csv(self, file_path: Path, config: DatasetConfig) -> list[DataSample]:
        """Load CSV format."""
        import csv

        samples = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                sample = DataSample(
                    sample_id=f"{config.name}_{idx}",
                    data=dict(row),
                    label=row.get("label"),
                    metadata={"index": idx, "file_path": str(file_path)},
                )
                samples.append(sample)

        return samples

    async def _load_tsv(self, file_path: Path, config: DatasetConfig) -> list[DataSample]:
        """Load TSV format."""
        import csv

        samples = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                sample = DataSample(
                    sample_id=f"{config.name}_{idx}",
                    data=dict(row),
                    label=row.get("label"),
                    metadata={"index": idx, "file_path": str(file_path)},
                )
                samples.append(sample)

        return samples


class RandomDataSplitter(DataSplitter):
    """Random data splitter."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.logger = get_logger("random_data_splitter")

    async def create_splits(self, dataset: Dataset, config: DatasetConfig) -> DataSplits:
        """Create random train/validation/test splits."""
        import random

        random.seed(self.seed)

        # Get split ratios from config
        test_split = getattr(config, "test_split", 0.2)
        validation_split = getattr(config, "validation_split", 0.1)

        # Shuffle sample IDs
        sample_ids = [sample.sample_id for sample in dataset.samples]
        random.shuffle(sample_ids)

        total_samples = len(sample_ids)
        test_size = int(total_samples * test_split)
        validation_size = int(total_samples * validation_split)
        # train_size = total_samples - test_size - validation_size

        # Create splits
        test_samples = sample_ids[:test_size]
        validation_samples = sample_ids[test_size : test_size + validation_size]
        train_samples = sample_ids[test_size + validation_size :]

        splits = DataSplits(
            dataset_id=dataset.dataset_id,
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            split_info={
                "strategy": self.get_split_strategy(),
                "seed": self.seed,
                "test_ratio": test_split,
                "validation_ratio": validation_split,
                "train_ratio": 1.0 - test_split - validation_split,
            },
        )

        self.logger.info(
            f"Created splits for '{dataset.dataset_id}': "
            f"train={len(train_samples)}, val={len(validation_samples)}, test={len(test_samples)}"
        )

        return splits

    def get_split_strategy(self) -> str:
        """Get split strategy name."""
        return "random"


class DataService(BaseService):
    """
    Service for managing dataset loading, preprocessing, and caching.

    Features:
    - Plugin architecture for different data sources
    - Dataset caching system with memory management
    - Async data loading and processing
    - Data validation and schema checking
    - Memory management for large datasets
    """

    def __init__(
        self,
        cache_max_size: int = 50,
        cache_max_memory_mb: int = 1024,
        cache_ttl: int = 3600,
        enable_compression: bool = True,
        enable_hardware_optimization: bool = True,
    ):
        """
        Initialize the Data Service.

        Args:
            cache_max_size: Maximum number of datasets to cache
            cache_max_memory_mb: Maximum cache memory in MB
            cache_ttl: Cache time-to-live in seconds
            enable_compression: Enable compressed caching for better memory efficiency
            enable_hardware_optimization: Enable hardware-specific optimizations
        """
        super().__init__("data_service")

        # Plugin registries
        self.loaders: dict[DataSource, DataLoader] = {}
        self.preprocessors: dict[str, DataPreprocessor] = {}
        self.validators: dict[str, DataValidator] = {}
        self.splitters: dict[str, DataSplitter] = {}

        # Dataset management
        self._dataset_registry: dict[str, DatasetInfo] = {}
        self._quality_reports: dict[str, DataQualityReport] = {}
        self._dataset_statistics: dict[str, DatasetStatistics] = {}

        # Cache system - use compressed cache if enabled
        self.cache: DataCache
        if enable_compression:
            self.cache = CompressedCache(
                max_size=cache_max_size, max_memory_mb=cache_max_memory_mb, ttl_default=cache_ttl
            )
        else:
            self.cache = MemoryDataCache(
                max_size=cache_max_size, max_memory_mb=cache_max_memory_mb, ttl_default=cache_ttl
            )

        # Performance optimization components
        self.memory_manager: MemoryManager | None = None
        self.streaming_loader = StreamingDataLoader()
        self.optimizer: DataServiceOptimizer | None = None
        self._optimization_settings: dict[str, Any] = {}
        self._enable_hardware_optimization = enable_hardware_optimization

        # Internal state
        self._datasets: dict[str, Dataset] = {}
        self._dataset_splits: dict[str, DataSplits] = {}
        self._lock = asyncio.Lock()

        self.logger = get_logger("data_service")

    async def initialize(self) -> ServiceResponse:
        """Initialize the Data Service."""
        try:
            self.logger.info("Initializing Data Service")

            # Initialize performance optimization components
            if self._enable_hardware_optimization:
                self.memory_manager = MemoryManager()
                self.optimizer = DataServiceOptimizer(self)
                await self.optimizer.optimize_for_hardware()
                self.logger.info("Hardware optimizations enabled")

            # Register default plugins
            await self._register_default_plugins()

            self._set_status(ServiceStatus.HEALTHY)
            self.logger.info("Data Service initialized successfully")

            # Gather initialization info
            init_data = {
                "registered_loaders": list(self.loaders.keys()),
                "registered_splitters": list(self.splitters.keys()),
                "compression_enabled": isinstance(self.cache, CompressedCache),
                "hardware_optimization": self._enable_hardware_optimization,
            }

            # Add optimization settings if available
            if self._optimization_settings:
                init_data["optimization_settings"] = self._optimization_settings

            return ServiceResponse(
                success=True,
                message="Data Service initialized successfully",
                data=init_data,
            )

        except Exception as e:
            self._set_status(ServiceStatus.ERROR)
            error_msg = f"Failed to initialize Data Service: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ServiceResponse(success=False, message=error_msg, error=str(e))

    async def health_check(self) -> HealthCheck:
        """Perform a health check on the Data Service."""
        try:
            cache_info = await self.cache.get_cache_info()
            checks = {
                "cache_status": cache_info,
                "registered_loaders": len(self.loaders),
                "loaded_datasets": len(self._datasets),
                "cached_splits": len(self._dataset_splits),
            }

            # Add memory status if memory manager is available
            if self.memory_manager:
                memory_status = await self.memory_manager.get_memory_status()
                checks["memory_status"] = {
                    "total_gb": round(memory_status.total_gb, 2),
                    "used_gb": round(memory_status.used_gb, 2),
                    "available_gb": round(memory_status.available_gb, 2),
                    "usage_percent": round(memory_status.usage_percent, 1),
                    "process_memory_mb": round(memory_status.process_memory_mb, 1),
                    "cache_memory_mb": round(memory_status.cache_memory_mb, 1),
                }

            return HealthCheck(
                status=self.status,
                message="Data Service is healthy",
                checks=checks,
            )

        except Exception as e:
            return HealthCheck(
                status=ServiceStatus.ERROR,
                message=f"Health check failed: {str(e)}",
                checks={"error": str(e)},
            )

    async def shutdown(self) -> ServiceResponse:
        """Shutdown the Data Service."""
        try:
            self.logger.info("Shutting down Data Service")

            # Clear cache
            await self.cache.clear()

            # Clear internal state
            async with self._lock:
                self._datasets.clear()
                self._dataset_splits.clear()
                self._dataset_registry.clear()
                self._quality_reports.clear()
                self._dataset_statistics.clear()

            self._set_status(ServiceStatus.STOPPED)
            self.logger.info("Data Service shutdown completed")

            return ServiceResponse(success=True, message="Data Service shutdown successfully")

        except Exception as e:
            error_msg = f"Error during Data Service shutdown: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return ServiceResponse(success=False, message=error_msg, error=str(e))

    async def register_loader(self, source_type: DataSource, loader: DataLoader) -> None:
        """
        Register a data loader for a specific source type.

        Args:
            source_type: The data source type
            loader: The data loader instance
        """
        self.loaders[source_type] = loader
        self.logger.info(f"Registered data loader for source: {source_type}")

    async def register_splitter(self, strategy: str, splitter: DataSplitter) -> None:
        """
        Register a data splitter for a specific strategy.

        Args:
            strategy: The splitting strategy name
            splitter: The data splitter instance
        """
        self.splitters[strategy] = splitter
        self.logger.info(f"Registered data splitter for strategy: {strategy}")

    async def load_dataset(self, config: DatasetConfig) -> Dataset:
        """
        Load a dataset using the appropriate loader.

        Args:
            config: Dataset configuration

        Returns:
            Loaded dataset

        Raises:
            DataLoadingError: If loading fails
        """
        dataset_id = config.name
        cache_key = self._get_cache_key("dataset", dataset_id, config)

        # Check cache first
        cached_dataset = await self.cache.get(cache_key)
        if cached_dataset:
            self.logger.debug(f"Using cached dataset: {dataset_id}")
            return cached_dataset  # type: ignore[no-any-return]

        try:
            # Determine source type
            source_str = getattr(config, "source", "local")
            try:
                source_type = DataSource(source_str.lower())
            except ValueError as e:
                raise DataLoadingError(
                    f"Unsupported data source: {source_str}",
                    error_code=ErrorCode.DATASET_NOT_FOUND,
                    metadata={"source": source_str},
                ) from e

            # Get appropriate loader
            if source_type not in self.loaders:
                raise DataLoadingError(
                    f"No loader registered for source: {source_type}",
                    error_code=ErrorCode.DATASET_NOT_FOUND,
                    metadata={"source": source_type.value},
                )

            loader = self.loaders[source_type]

            # Validate source first
            if not await loader.validate_source(config):
                raise DataLoadingError(
                    f"Data source validation failed: {config.path}",
                    error_code=ErrorCode.DATASET_NOT_FOUND,
                    metadata={"path": config.path, "source": source_type.value},
                )

            # Load dataset
            self.logger.info(f"Loading dataset '{dataset_id}' from {source_type.value} source")
            dataset = await loader.load(config)

            # Cache the dataset
            await self.cache.set(cache_key, dataset)

            # Store in internal registry
            async with self._lock:
                self._datasets[dataset_id] = dataset

            # Track dataset in memory manager if available
            if self.memory_manager:
                self.memory_manager.track_dataset(dataset_id, dataset)

            self.logger.info(
                f"Successfully loaded dataset '{dataset_id}' with {dataset.size} samples"
            )
            return dataset

        except DataLoadingError:
            raise
        except Exception as e:
            raise DataLoadingError(
                f"Unexpected error loading dataset: {str(e)}",
                error_code=ErrorCode.DATA_PREPROCESSING_FAILED,
                metadata={"dataset_id": dataset_id},
            ) from e

    async def get_dataset_info(self, dataset_id: str) -> DatasetInfo:
        """
        Get dataset information.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset information

        Raises:
            DataLoadingError: If dataset not found
        """
        async with self._lock:
            if dataset_id in self._datasets:
                return self._datasets[dataset_id].info

        # Try to find in cache
        cache_keys = await self.cache.get_cache_info()
        for key in cache_keys.get("entries", []):
            if key.startswith(f"dataset_{dataset_id}_"):
                cached_dataset = await self.cache.get(key)
                if cached_dataset:
                    return cached_dataset.info  # type: ignore[no-any-return]

        raise DataLoadingError(
            f"Dataset not found: {dataset_id}",
            error_code=ErrorCode.DATASET_NOT_FOUND,
            metadata={"dataset_id": dataset_id},
        )

    async def create_data_splits(self, dataset_id: str, config: DatasetConfig) -> DataSplits:
        """
        Create train/validation/test splits for a dataset.

        Args:
            dataset_id: Dataset identifier
            config: Dataset configuration with split parameters

        Returns:
            Data splits

        Raises:
            DataLoadingError: If dataset not found or splitting fails
        """
        cache_key = self._get_cache_key("splits", dataset_id, config)

        # Check cache first
        cached_splits = await self.cache.get(cache_key)
        if cached_splits:
            self.logger.debug(f"Using cached splits for dataset: {dataset_id}")
            return cached_splits  # type: ignore[no-any-return]

        try:
            # Get dataset
            dataset = None
            async with self._lock:
                dataset = self._datasets.get(dataset_id)

            if not dataset:
                # Try to load dataset if not already loaded
                dataset = await self.load_dataset(config)

            # Get splitter strategy
            strategy = getattr(config, "split_strategy", "random")
            if strategy not in self.splitters:
                raise DataLoadingError(
                    f"No splitter registered for strategy: {strategy}",
                    error_code=ErrorCode.DATASET_NOT_FOUND,
                    metadata={"strategy": strategy},
                )

            splitter = self.splitters[strategy]

            # Create splits
            self.logger.info(
                f"Creating data splits for dataset '{dataset_id}' using {strategy} strategy"
            )
            splits = await splitter.create_splits(dataset, config)

            # Cache the splits
            await self.cache.set(cache_key, splits)

            # Store in internal registry
            async with self._lock:
                self._dataset_splits[dataset_id] = splits

            return splits

        except DataLoadingError:
            raise
        except Exception as e:
            raise DataLoadingError(
                f"Failed to create data splits: {str(e)}",
                error_code=ErrorCode.DATA_PREPROCESSING_FAILED,
                metadata={"dataset_id": dataset_id},
            ) from e

    async def get_batch(self, dataset_id: str, batch_size: int, offset: int = 0) -> DataBatch:
        """
        Get a batch of data samples.

        Args:
            dataset_id: Dataset identifier
            batch_size: Number of samples in batch
            offset: Starting offset

        Returns:
            Data batch

        Raises:
            DataLoadingError: If dataset not found
        """
        async with self._lock:
            if dataset_id not in self._datasets:
                raise DataLoadingError(
                    f"Dataset not loaded: {dataset_id}",
                    error_code=ErrorCode.DATASET_NOT_FOUND,
                    metadata={"dataset_id": dataset_id},
                )

            dataset = self._datasets[dataset_id]

            # Get batch samples
            start_idx = offset
            end_idx = min(offset + batch_size, len(dataset.samples))
            batch_samples = dataset.samples[start_idx:end_idx]

            batch = DataBatch(
                batch_id=f"{dataset_id}_batch_{offset}_{batch_size}",
                samples=batch_samples,
                batch_info={
                    "dataset_id": dataset_id,
                    "batch_size": batch_size,
                    "offset": offset,
                    "total_samples": len(dataset.samples),
                    "actual_size": len(batch_samples),
                },
            )

            return batch

    async def get_optimized_batch(
        self, dataset_id: str, batch_size: int | None = None, offset: int = 0
    ) -> DataBatch:
        """
        Get an optimized batch using hardware-specific settings.

        Args:
            dataset_id: Dataset identifier
            batch_size: Number of samples in batch (uses optimized size if None)
            offset: Starting offset

        Returns:
            Optimized data batch
        """
        # Use optimized batch size if not specified
        if batch_size is None:
            batch_size = self._optimization_settings.get("optimal_batch_size", 64)

        return await self.get_batch(dataset_id, batch_size, offset)

    async def stream_dataset_batches(
        self,
        config: DatasetConfig,
        batch_size: int | None = None,
        progress_callback: Callable[..., Any] | None = None,
    ) -> AsyncIterator[DataBatch]:
        """
        Stream dataset in batches for memory-efficient processing.

        Args:
            config: Dataset configuration
            batch_size: Size of each batch (uses optimized size if None)
            progress_callback: Optional progress callback function

        Yields:
            DataBatch: Batches of dataset samples
        """
        # Use optimized batch size if not specified
        if batch_size is None:
            batch_size = self._optimization_settings.get("optimal_batch_size", 64)

        # Get appropriate loader
        source_str = getattr(config, "source", "local")
        source_type = DataSource(source_str.lower())

        if source_type not in self.loaders:
            raise DataLoadingError(
                f"No loader registered for source: {source_type}",
                error_code=ErrorCode.DATASET_NOT_FOUND,
                metadata={"source": source_type.value},
            )

        loader = self.loaders[source_type]

        # Stream batches using the streaming loader
        async for batch in self.streaming_loader.stream_batches(
            loader, config, batch_size, progress_callback
        ):
            yield batch

    async def get_memory_status(self) -> dict[str, Any]:
        """
        Get current memory status and usage information.

        Returns:
            Memory status information
        """
        if not self.memory_manager:
            return {"error": "Memory manager not available"}

        memory_status = await self.memory_manager.get_memory_status()
        cache_info = await self.cache.get_cache_info()

        return {
            "memory_status": {
                "total_gb": round(memory_status.total_gb, 2),
                "used_gb": round(memory_status.used_gb, 2),
                "available_gb": round(memory_status.available_gb, 2),
                "usage_percent": round(memory_status.usage_percent, 1),
                "process_memory_mb": round(memory_status.process_memory_mb, 1),
                "cache_memory_mb": round(memory_status.cache_memory_mb, 1),
            },
            "cache_info": cache_info,
            "memory_pressure": await self.memory_manager.check_memory_pressure(),
        }

    async def cleanup_memory(self) -> dict[str, Any]:
        """
        Perform memory cleanup and return cleanup statistics.

        Returns:
            Cleanup statistics
        """
        if not self.memory_manager:
            return {"error": "Memory manager not available"}

        # Get initial memory status
        initial_status = await self.memory_manager.get_memory_status()

        # Perform cleanup
        cleaned_datasets = await self.memory_manager.cleanup_unused_datasets()

        # Get final memory status
        final_status = await self.memory_manager.get_memory_status()

        return {
            "cleaned_datasets": cleaned_datasets,
            "memory_freed_mb": round(
                initial_status.process_memory_mb - final_status.process_memory_mb, 2
            ),
            "initial_memory_mb": round(initial_status.process_memory_mb, 2),
            "final_memory_mb": round(final_status.process_memory_mb, 2),
        }

    async def get_cache_performance_stats(self) -> dict[str, Any]:
        """
        Get detailed cache performance statistics.

        Returns:
            Cache performance statistics including compression info if available
        """
        cache_info = await self.cache.get_cache_info()

        # Add optimization info
        stats = {
            **cache_info,
            "advanced_cache": isinstance(self.cache, CompressedCache),
            "optimization_settings": self._optimization_settings,
            "lazy_loading_enabled": hasattr(self, "streaming_loader"),
        }

        # Add hardware optimization info if available
        if self.optimizer:
            stats["hardware_optimized"] = True

        return stats

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_cache_info()

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        await self.cache.clear()
        self.logger.info("Data service cache cleared")

    async def _register_default_plugins(self) -> None:
        """Register default data loaders and processors."""
        # Register all available data loaders
        await self._auto_register_loaders()

        # Register random data splitter
        random_splitter = RandomDataSplitter()
        await self.register_splitter("random", random_splitter)

    async def _auto_register_loaders(self) -> None:
        """Auto-register all available data loaders."""
        # Register local data loader (using the existing one from the service for compatibility)
        local_loader: DataLoader = LocalDataLoader()
        await self.register_loader(DataSource.LOCAL, local_loader)
        self.logger.info("Registered LocalDataLoader for LOCAL source")

        # Register Kaggle data loader if available
        try:
            kaggle_loader: DataLoader = KaggleDataLoader()  # type: ignore[assignment]
            await self.register_loader(DataSource.KAGGLE, kaggle_loader)
            self.logger.info("Registered KaggleDataLoader for KAGGLE source")
        except Exception as e:
            self.logger.warning(f"Could not register KaggleDataLoader: {e}")

        # Register HuggingFace data loader if available
        try:
            hf_loader: DataLoader = HuggingFaceDataLoader()  # type: ignore[assignment]
            await self.register_loader(DataSource.HUGGINGFACE, hf_loader)
            self.logger.info("Registered HuggingFaceDataLoader for HUGGINGFACE source")
        except Exception as e:
            self.logger.warning(f"Could not register HuggingFaceDataLoader: {e}")

    def _get_cache_key(self, prefix: str, dataset_id: str, config: DatasetConfig) -> str:
        """Generate cache key for dataset or splits."""
        # Create a hash of the config to ensure uniqueness
        config_str = json.dumps(
            {
                "name": config.name,
                "path": config.path,
                "source": getattr(config, "source", "local"),
                "max_samples": getattr(config, "max_samples", None),
                "test_split": getattr(config, "test_split", 0.2),
                "validation_split": getattr(config, "validation_split", 0.1),
            },
            sort_keys=True,
        )

        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{prefix}_{dataset_id}_{config_hash}"

    async def list_available_datasets(self) -> list[DatasetInfo]:
        """List all available datasets in the registry."""
        async with self._lock:
            return list(self._dataset_registry.values())

    async def discover_datasets(self, source_paths: list[str]) -> list[DatasetInfo]:
        """Discover datasets in specified paths."""
        discovered = []

        for path_str in source_paths:
            try:
                path = Path(path_str)
                if not path.exists():
                    self.logger.warning(f"Path does not exist: {path}")
                    continue

                # Look for dataset files
                dataset_files: list[Path] = []
                for ext in ["*.json", "*.jsonl", "*.csv", "*.tsv", "*.parquet"]:
                    dataset_files.extend(path.glob(ext))
                    if path.is_dir():
                        dataset_files.extend(path.rglob(ext))

                for file_path in dataset_files:
                    try:
                        # Create minimal dataset info for discovery
                        try:
                            data_format = DataFormat(file_path.suffix.lstrip(".").lower())
                        except ValueError:
                            data_format = DataFormat.JSON  # Default format

                        dataset_info = DatasetInfo(
                            dataset_id=f"discovered_{file_path.stem}",
                            name=file_path.stem,
                            description=f"Auto-discovered dataset from {file_path}",
                            source=DataSource.LOCAL,
                            format=data_format,
                            size_bytes=file_path.stat().st_size,
                            created_at=str(file_path.stat().st_ctime),
                            modified_at=str(file_path.stat().st_mtime),
                            metadata={"file_path": str(file_path), "auto_discovered": True},
                        )

                        discovered.append(dataset_info)

                        # Add to registry
                        async with self._lock:
                            self._dataset_registry[dataset_info.dataset_id] = dataset_info

                    except Exception as e:
                        self.logger.warning(f"Could not analyze file {file_path}: {e}")

            except Exception as e:
                self.logger.error(f"Error discovering datasets in {path_str}: {e}")

        self.logger.info(f"Discovered {len(discovered)} datasets")
        return discovered

    async def validate_dataset_quality(self, dataset_id: str) -> DataQualityReport:
        """Validate dataset quality and generate a report."""
        # Check if we have a cached report
        async with self._lock:
            if dataset_id in self._quality_reports:
                return self._quality_reports[dataset_id]

        # Get dataset
        dataset = None
        async with self._lock:
            dataset = self._datasets.get(dataset_id)

        if not dataset:
            raise DataLoadingError(
                f"Dataset not loaded: {dataset_id}",
                error_code=ErrorCode.DATASET_NOT_FOUND,
                metadata={"dataset_id": dataset_id},
            )

        start_time = time.time()

        # Initialize counters
        empty_samples = 0
        malformed_samples = 0
        missing_labels = 0
        invalid_labels = 0
        duplicate_samples = 0

        content_lengths = []
        attack_count = 0
        benign_count = 0
        validation_errors = []
        warnings = []

        # Track duplicates
        content_hashes = set()

        # Analyze each sample
        for sample in dataset.samples:
            try:
                # Check for empty content - look for text field or general data content
                content_str = ""
                if hasattr(sample, "input_text") and sample.input_text:
                    content_str = str(sample.input_text)
                elif isinstance(sample.data, dict) and "text" in sample.data:
                    content_str = str(sample.data["text"])
                elif sample.data:
                    content_str = str(sample.data)

                if not content_str.strip():
                    empty_samples += 1
                    continue

                # Check for missing labels
                if not sample.label:
                    missing_labels += 1
                    warnings.append(f"Sample {sample.sample_id} missing label")
                    continue

                # Check for invalid labels
                if sample.label.upper() not in ["ATTACK", "BENIGN"]:
                    invalid_labels += 1
                    validation_errors.append(
                        f"Sample {sample.sample_id} has invalid label: {sample.label}"
                    )
                    continue

                # Count labels
                if sample.label.upper() == "ATTACK":
                    attack_count += 1
                else:
                    benign_count += 1

                # Check for duplicates (use the same content_str from above)
                content_hash = hashlib.md5(content_str.encode()).hexdigest()
                if content_hash in content_hashes:
                    duplicate_samples += 1
                else:
                    content_hashes.add(content_hash)

                # Collect content length
                content_lengths.append(len(content_str))

            except Exception as e:
                malformed_samples += 1
                validation_errors.append(f"Sample {sample.sample_id} is malformed: {str(e)}")

        # Calculate statistics
        total_samples = len(dataset.samples)
        avg_content_length = statistics.mean(content_lengths) if content_lengths else 0.0
        min_content_length = min(content_lengths) if content_lengths else 0
        max_content_length = max(content_lengths) if content_lengths else 0

        attack_ratio = attack_count / total_samples if total_samples > 0 else 0.0
        benign_ratio = benign_count / total_samples if total_samples > 0 else 0.0

        # Calculate label balance score
        label_balance_score = 1.0 - abs(attack_ratio - benign_ratio) if total_samples > 0 else 0.0

        # Calculate overall quality score
        issues_ratio = (
            (
                empty_samples
                + malformed_samples
                + missing_labels
                + invalid_labels
                + duplicate_samples
            )
            / total_samples
            if total_samples > 0
            else 0.0
        )

        quality_score = max(0.0, 1.0 - issues_ratio)

        # Create quality report
        report = DataQualityReport(
            dataset_id=dataset_id,
            total_samples=total_samples,
            quality_score=quality_score,
            empty_samples=empty_samples,
            duplicate_samples=duplicate_samples,
            malformed_samples=malformed_samples,
            missing_labels=missing_labels,
            invalid_labels=invalid_labels,
            avg_content_length=avg_content_length,
            min_content_length=min_content_length,
            max_content_length=max_content_length,
            attack_ratio=attack_ratio,
            benign_ratio=benign_ratio,
            label_balance_score=label_balance_score,
            validation_errors=validation_errors,
            warnings=warnings,
            analysis_duration_seconds=time.time() - start_time,
        )

        # Cache the report
        async with self._lock:
            self._quality_reports[dataset_id] = report

        self.logger.info(
            f"Quality analysis complete for '{dataset_id}': "
            f"score={quality_score:.3f}, issues={report.issues_count}"
        )

        return report

    async def compute_dataset_statistics(self, dataset_id: str) -> DatasetStatistics:
        """Compute comprehensive statistics for a dataset."""
        # Check if we have cached statistics
        async with self._lock:
            if dataset_id in self._dataset_statistics:
                return self._dataset_statistics[dataset_id]

        # Get dataset
        dataset = None
        async with self._lock:
            dataset = self._datasets.get(dataset_id)

        if not dataset:
            raise DataLoadingError(
                f"Dataset not loaded: {dataset_id}",
                error_code=ErrorCode.DATASET_NOT_FOUND,
                metadata={"dataset_id": dataset_id},
            )

        start_time = time.time()

        # Initialize collectors
        content_lengths: list[int] = []
        word_counts: list[int] = []
        character_counts: list[int] = []
        attack_types: Counter[str] = Counter()
        metadata_fields: Counter[str] = Counter()
        languages: Counter[str] = Counter()

        attack_count = 0
        benign_count = 0
        timestamps = []

        # Analyze each sample
        for sample in dataset.samples:
            try:
                # Get content string consistently
                content_str = ""
                if hasattr(sample, "input_text") and sample.input_text:
                    content_str = str(sample.input_text)
                elif isinstance(sample.data, dict) and "text" in sample.data:
                    content_str = str(sample.data["text"])
                elif sample.data:
                    content_str = str(sample.data)

                content_lengths.append(len(content_str))

                # Word and character counts
                words = content_str.split()
                word_counts.append(len(words))
                character_counts.append(len(content_str))

                # Label analysis
                if sample.label and sample.label.upper() == "ATTACK":
                    attack_count += 1
                    # Try to get attack type from different sources
                    attack_type = None
                    if hasattr(sample, "attack_type") and sample.attack_type:
                        attack_type = sample.attack_type
                    elif isinstance(sample.data, dict) and "attack_type" in sample.data:
                        attack_type = sample.data["attack_type"]

                    if attack_type and attack_type.strip():
                        attack_types[attack_type.strip()] += 1
                elif sample.label and sample.label.upper() == "BENIGN":
                    benign_count += 1

                # Metadata analysis
                for key in sample.metadata:
                    metadata_fields[key] += 1

                # Timestamp analysis
                if hasattr(sample, "timestamp") and sample.timestamp:
                    timestamps.append(sample.timestamp)

                # Simple language detection (basic heuristic)
                if any(ord(char) > 127 for char in content_str[:100]):
                    languages["non_ascii"] += 1
                else:
                    languages["ascii"] += 1

            except Exception as e:
                self.logger.warning(f"Error analyzing sample {sample.sample_id}: {e}")

        # Calculate statistics
        def calc_stats(values: list[int] | list[float]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
            return {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
                "median": statistics.median(values),
            }

        # Get quality report if available
        quality_report = None
        async with self._lock:
            quality_report = self._quality_reports.get(dataset_id)

        # Build timestamp range
        timestamp_range = {}
        if timestamps:
            timestamp_range["earliest"] = str(min(timestamps))
            timestamp_range["latest"] = str(max(timestamps))

        # Create statistics
        stats = DatasetStatistics(
            dataset_id=dataset_id,
            dataset_name=dataset.info.name,
            total_samples=len(dataset.samples),
            total_size_bytes=dataset.info.size_bytes,
            attack_samples=attack_count,
            benign_samples=benign_count,
            attack_types=dict(attack_types),
            content_length_stats=calc_stats(content_lengths),
            word_count_stats=calc_stats(word_counts),
            character_count_stats=calc_stats(character_counts),
            detected_languages=dict(languages),
            encoding_info={"analysis_method": "basic_heuristic"},
            metadata_fields=dict(metadata_fields),
            timestamp_range=timestamp_range,
            quality_report=quality_report,
            computation_duration_seconds=time.time() - start_time,
        )

        # Cache the statistics
        async with self._lock:
            self._dataset_statistics[dataset_id] = stats

        self.logger.info(
            f"Statistics computed for '{dataset_id}' in {stats.computation_duration_seconds:.2f}s"
        )

        return stats

    async def get_dataset_quality_report(self, dataset_id: str) -> DataQualityReport:
        """Get quality report for a dataset, computing if necessary."""
        async with self._lock:
            if dataset_id in self._quality_reports:
                return self._quality_reports[dataset_id]

        return await self.validate_dataset_quality(dataset_id)

    async def get_dataset_statistics(self, dataset_id: str) -> DatasetStatistics:
        """Get statistics for a dataset, computing if necessary."""
        async with self._lock:
            if dataset_id in self._dataset_statistics:
                return self._dataset_statistics[dataset_id]

        return await self.compute_dataset_statistics(dataset_id)

    async def get_registered_loaders(self) -> dict[str, dict[str, Any]]:
        """Get information about all registered loaders."""
        loader_info: dict[str, dict[str, Any]] = {}

        for source, loader in self.loaders.items():
            try:
                info: dict[str, Any] = {
                    "class_name": loader.__class__.__name__,
                    "source_type": source.value,
                }

                # Try to get supported formats if available
                if hasattr(loader, "get_supported_formats") and callable(
                    loader.get_supported_formats
                ):
                    try:
                        formats = loader.get_supported_formats()
                        if hasattr(formats[0], "value"):  # Enum format
                            info["supported_formats"] = [fmt.value for fmt in formats]
                        else:  # String format
                            info["supported_formats"] = formats
                    except Exception:
                        info["supported_formats"] = ["unknown"]
                else:
                    # Default supported formats based on loader type
                    if "LocalFile" in loader.__class__.__name__:
                        info["supported_formats"] = ["json", "jsonl", "csv", "tsv", "parquet"]
                    elif "Kaggle" in loader.__class__.__name__:
                        info["supported_formats"] = ["csv", "json", "zip"]
                    elif "HuggingFace" in loader.__class__.__name__:
                        info["supported_formats"] = ["datasets"]
                    else:
                        info["supported_formats"] = ["unknown"]

                loader_info[source.value] = info

            except Exception as e:
                loader_info[source.value] = {
                    "error": str(e),
                    "class_name": loader.__class__.__name__,
                }

        return loader_info

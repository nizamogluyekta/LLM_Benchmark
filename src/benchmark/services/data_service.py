"""
Data Service for the LLM Cybersecurity Benchmark system.

This service manages dataset loading, preprocessing, and caching with a plugin
architecture for different data sources.
"""

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from benchmark.core.base import BaseService, HealthCheck, ServiceResponse, ServiceStatus
from benchmark.core.config import DatasetConfig
from benchmark.core.exceptions import DataLoadingError, ErrorCode
from benchmark.core.logging import get_logger
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
        self, cache_max_size: int = 50, cache_max_memory_mb: int = 1024, cache_ttl: int = 3600
    ):
        """
        Initialize the Data Service.

        Args:
            cache_max_size: Maximum number of datasets to cache
            cache_max_memory_mb: Maximum cache memory in MB
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__("data_service")

        # Plugin registries
        self.loaders: dict[DataSource, DataLoader] = {}
        self.preprocessors: dict[str, DataPreprocessor] = {}
        self.validators: dict[str, DataValidator] = {}
        self.splitters: dict[str, DataSplitter] = {}

        # Cache system
        self.cache: DataCache = MemoryDataCache(
            max_size=cache_max_size, max_memory_mb=cache_max_memory_mb, ttl_default=cache_ttl
        )

        # Internal state
        self._datasets: dict[str, Dataset] = {}
        self._dataset_splits: dict[str, DataSplits] = {}
        self._lock = asyncio.Lock()

        self.logger = get_logger("data_service")

    async def initialize(self) -> ServiceResponse:
        """Initialize the Data Service."""
        try:
            self.logger.info("Initializing Data Service")

            # Register default plugins
            await self._register_default_plugins()

            self._set_status(ServiceStatus.HEALTHY)
            self.logger.info("Data Service initialized successfully")

            return ServiceResponse(
                success=True,
                message="Data Service initialized successfully",
                data={
                    "registered_loaders": list(self.loaders.keys()),
                    "registered_splitters": list(self.splitters.keys()),
                },
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

            return HealthCheck(
                status=self.status,
                message="Data Service is healthy",
                checks={
                    "cache_status": cache_info,
                    "registered_loaders": len(self.loaders),
                    "loaded_datasets": len(self._datasets),
                    "cached_splits": len(self._dataset_splits),
                },
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

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_cache_info()

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        await self.cache.clear()
        self.logger.info("Data service cache cleared")

    async def _register_default_plugins(self) -> None:
        """Register default data loaders and processors."""
        # Register local data loader
        local_loader: DataLoader = LocalDataLoader()
        await self.register_loader(DataSource.LOCAL, local_loader)

        # Register random data splitter
        random_splitter = RandomDataSplitter()
        await self.register_splitter("random", random_splitter)

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

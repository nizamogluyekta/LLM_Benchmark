"""
Data interfaces for the LLM Cybersecurity Benchmark system.

This module defines the abstract interfaces for data loading, processing,
and management components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmark.core.config import DatasetConfig


class DataFormat(Enum):
    """Supported data formats."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"
    PICKLE = "pickle"
    TSV = "tsv"


class DataSource(Enum):
    """Supported data sources."""

    LOCAL = "local"
    REMOTE = "remote"
    KAGGLE = "kaggle"
    HUGGINGFACE = "huggingface"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    dataset_id: str
    name: str
    description: str | None = None
    source: DataSource = DataSource.LOCAL
    format: DataFormat = DataFormat.JSONL
    size_bytes: int = 0
    num_samples: int = 0
    schema: dict[str, Any] | None = None
    created_at: str | None = None
    modified_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSample:
    """A single data sample."""

    sample_id: str
    data: dict[str, Any]
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBatch:
    """A batch of data samples."""

    batch_id: str
    samples: list[DataSample]
    batch_info: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of samples in the batch."""
        return len(self.samples)


@dataclass
class DataSplits:
    """Data splits for training, validation, and testing."""

    dataset_id: str
    train_samples: list[str]
    validation_samples: list[str]
    test_samples: list[str]
    split_info: dict[str, Any] = field(default_factory=dict)

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self.train_samples)

    @property
    def validation_size(self) -> int:
        """Number of validation samples."""
        return len(self.validation_samples)

    @property
    def test_size(self) -> int:
        """Number of test samples."""
        return len(self.test_samples)


@dataclass
class Dataset:
    """Complete dataset with metadata and samples."""

    dataset_id: str
    info: DatasetInfo
    samples: list[DataSample]
    splits: DataSplits | None = None

    @property
    def size(self) -> int:
        """Number of samples in the dataset."""
        return len(self.samples)


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    async def load(self, config: DatasetConfig) -> Dataset:
        """
        Load data from the configured source.

        Args:
            config: Dataset configuration

        Returns:
            Complete dataset with samples and metadata

        Raises:
            DataLoadingError: If loading fails
        """
        pass

    @abstractmethod
    async def validate_source(self, config: DatasetConfig) -> bool:
        """
        Validate that the data source is accessible and valid.

        Args:
            config: Dataset configuration

        Returns:
            True if source is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[DataFormat]:
        """
        Get list of supported data formats.

        Returns:
            List of supported data formats
        """
        pass

    @abstractmethod
    def get_source_type(self) -> DataSource:
        """
        Get the source type this loader handles.

        Returns:
            The data source type
        """
        pass

    async def get_dataset_info(self, config: DatasetConfig) -> DatasetInfo:
        """
        Get dataset information without loading all data.

        Args:
            config: Dataset configuration

        Returns:
            Dataset information
        """
        # Default implementation - can be overridden
        dataset = await self.load(config)
        return dataset.info


class DataPreprocessor(ABC):
    """Abstract base class for data preprocessors."""

    @abstractmethod
    async def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Preprocess a dataset.

        Args:
            dataset: Input dataset

        Returns:
            Preprocessed dataset
        """
        pass

    @abstractmethod
    def get_preprocessing_steps(self) -> list[str]:
        """
        Get list of preprocessing steps this processor performs.

        Returns:
            List of preprocessing step names
        """
        pass


class DataValidator(ABC):
    """Abstract base class for data validators."""

    @abstractmethod
    async def validate_schema(self, dataset: Dataset, expected_schema: dict[str, Any]) -> bool:
        """
        Validate dataset schema.

        Args:
            dataset: Dataset to validate
            expected_schema: Expected schema

        Returns:
            True if schema is valid, False otherwise
        """
        pass

    @abstractmethod
    async def validate_samples(self, dataset: Dataset) -> list[str]:
        """
        Validate individual samples in dataset.

        Args:
            dataset: Dataset to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        pass


class DataCache(ABC):
    """Abstract base class for data caching."""

    @abstractmethod
    async def get(self, cache_key: str) -> Any | None:
        """
        Get data from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not found
        """
        pass

    @abstractmethod
    async def set(self, cache_key: str, data: Any, ttl: int | None = None) -> None:
        """
        Store data in cache.

        Args:
            cache_key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
        """
        pass

    @abstractmethod
    async def delete(self, cache_key: str) -> bool:
        """
        Delete data from cache.

        Args:
            cache_key: Cache key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached data."""
        pass

    @abstractmethod
    async def get_cache_info(self) -> dict[str, Any]:
        """
        Get cache statistics and information.

        Returns:
            Dictionary with cache information
        """
        pass


class DataSplitter(ABC):
    """Abstract base class for data splitting."""

    @abstractmethod
    async def create_splits(self, dataset: Dataset, config: DatasetConfig) -> DataSplits:
        """
        Create train/validation/test splits.

        Args:
            dataset: Dataset to split
            config: Dataset configuration with split ratios

        Returns:
            Data splits
        """
        pass

    @abstractmethod
    def get_split_strategy(self) -> str:
        """
        Get the splitting strategy name.

        Returns:
            Strategy name (e.g., "random", "stratified")
        """
        pass

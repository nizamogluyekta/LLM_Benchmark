"""
Base data loader interface for the LLM Cybersecurity Benchmark system.

This module defines the abstract base class for all data loaders,
providing a consistent interface for loading datasets from various sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from benchmark.core.logging import get_logger
from benchmark.data.models import Dataset


class DataLoader(ABC):
    """
    Abstract base class for data loaders.

    All data loaders must implement this interface to provide
    consistent loading behavior across different data sources.
    """

    def __init__(self) -> None:
        """Initialize the data loader."""
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def load(self, config: dict[str, Any]) -> Dataset:
        """
        Load dataset from the configured source.

        Args:
            config: Configuration dictionary containing source parameters

        Returns:
            Loaded dataset

        Raises:
            ValueError: If configuration is invalid
            IOError: If data source cannot be accessed
            DataLoadError: If data cannot be loaded or parsed
        """
        pass

    @abstractmethod
    async def validate_source(self, config: dict[str, Any]) -> bool:
        """
        Validate that the data source exists and is accessible.

        Args:
            config: Configuration dictionary containing source parameters

        Returns:
            True if source is valid and accessible, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported data formats.

        Returns:
            List of supported format names (e.g., ['json', 'csv'])
        """
        pass

    def _validate_config(self, config: dict[str, Any], required_keys: list[str]) -> None:
        """
        Validate that required configuration keys are present.

        Args:
            config: Configuration dictionary
            required_keys: List of required configuration keys

        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")


class LoadProgress:
    """
    Progress reporting for data loading operations.

    Provides callbacks and progress tracking for long-running
    data loading operations.
    """

    def __init__(self, total_items: int = 0):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
        """
        self.total_items = total_items
        self.processed_items = 0
        self.start_time: float | None = None
        self.callbacks: list[Any] = []

    def add_callback(self, callback: Any) -> None:
        """Add a progress callback function."""
        self.callbacks.append(callback)

    def start(self) -> None:
        """Start progress tracking."""
        import time

        self.start_time = time.time()
        self._notify_callbacks()

    def update(self, items_processed: int = 1) -> None:
        """Update progress with number of items processed."""
        self.processed_items += items_processed
        self._notify_callbacks()

    def complete(self) -> None:
        """Mark progress as complete."""
        self.processed_items = self.total_items
        self._notify_callbacks()

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of progress update."""
        import contextlib

        for callback in self.callbacks:
            with contextlib.suppress(Exception):
                callback(self)

    @property
    def percentage(self) -> float:
        """Get completion percentage (0.0 to 1.0)."""
        if self.total_items == 0:
            return 0.0
        return min(1.0, self.processed_items / self.total_items)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        import time

        return time.time() - self.start_time


class DataLoadError(Exception):
    """Exception raised when data loading fails."""

    def __init__(
        self, message: str, source_path: Path | None = None, cause: Exception | None = None
    ):
        """
        Initialize DataLoadError.

        Args:
            message: Error message
            source_path: Path to the source that failed to load
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.source_path = source_path
        self.cause = cause

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [super().__str__()]
        if self.source_path:
            parts.append(f"Source: {self.source_path}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


class FieldMapping:
    """
    Configuration for mapping source fields to dataset fields.

    Handles mapping of source field names to standardized dataset fields
    and provides validation of mapped data.
    """

    def __init__(
        self,
        input_text_field: str = "input_text",
        label_field: str = "label",
        attack_type_field: str = "attack_type",
        metadata_fields: list[str] | None = None,
        required_fields: list[str] | None = None,
    ):
        """
        Initialize field mapping.

        Args:
            input_text_field: Source field name for input text
            label_field: Source field name for label
            attack_type_field: Source field name for attack type
            metadata_fields: List of fields to include in metadata
            required_fields: List of fields that must be present
        """
        self.input_text_field = input_text_field
        self.label_field = label_field
        self.attack_type_field = attack_type_field
        self.metadata_fields = metadata_fields or []
        self.required_fields = required_fields or [input_text_field, label_field]

    def validate_fields(self, sample: dict[str, Any]) -> None:
        """
        Validate that required fields are present in sample.

        Args:
            sample: Sample data dictionary

        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [
            field for field in self.required_fields if field not in sample or sample[field] is None
        ]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    def extract_metadata(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Extract metadata fields from sample.

        Args:
            sample: Sample data dictionary

        Returns:
            Dictionary of metadata fields
        """
        metadata = {}
        for field in self.metadata_fields:
            if field in sample and sample[field] is not None:
                metadata[field] = sample[field]
        return metadata

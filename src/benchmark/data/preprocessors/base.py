"""
Base interface for data preprocessors in the LLM Cybersecurity Benchmark system.

This module defines the abstract base class for all data preprocessors,
providing a consistent interface for preprocessing operations across different datasets.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from benchmark.core.logging import get_logger
from benchmark.data.models import DatasetSample


class PreprocessorError(Exception):
    """Exception raised when preprocessing fails."""

    def __init__(self, message: str, sample_id: str | None = None, cause: Exception | None = None):
        """
        Initialize PreprocessorError.

        Args:
            message: Error message
            sample_id: ID of the sample that failed processing
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.sample_id = sample_id
        self.cause = cause

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [super().__str__()]
        if self.sample_id:
            parts.append(f"Sample ID: {self.sample_id}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


class PreprocessingProgress:
    """
    Progress reporting for data preprocessing operations.

    Provides callbacks and progress tracking for long-running
    preprocessing operations.
    """

    def __init__(self, total_items: int = 0):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
        """
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
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

    def update(self, items_processed: int = 1, failed_items: int = 0) -> None:
        """
        Update progress with number of items processed.

        Args:
            items_processed: Number of items successfully processed
            failed_items: Number of items that failed processing
        """
        self.processed_items += items_processed
        self.failed_items += failed_items
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
    def success_rate(self) -> float:
        """Get success rate (0.0 to 1.0)."""
        if self.processed_items == 0:
            return 1.0
        return (self.processed_items - self.failed_items) / self.processed_items

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        import time

        return time.time() - self.start_time


class DataPreprocessor(ABC):
    """
    Abstract base class for data preprocessors.

    All data preprocessors must implement this interface to provide
    consistent preprocessing behavior across different data types.
    """

    def __init__(self, name: str | None = None):
        """
        Initialize the data preprocessor.

        Args:
            name: Optional name for the preprocessor (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)

    @abstractmethod
    async def process(
        self, samples: list[DatasetSample], config: dict[str, Any]
    ) -> list[DatasetSample]:
        """
        Process a list of dataset samples.

        Args:
            samples: List of samples to process
            config: Configuration dictionary for preprocessing options

        Returns:
            List of processed samples

        Raises:
            PreprocessorError: If preprocessing fails
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def get_required_fields(self) -> list[str]:
        """
        Get list of required fields for preprocessing.

        Returns:
            List of field names that must be present in samples
        """
        pass

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """
        Validate preprocessing configuration and return warnings.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of warning messages (empty if no warnings)

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def get_supported_config_keys(self) -> list[str]:
        """
        Get list of supported configuration keys.

        Returns:
            List of configuration keys supported by this preprocessor
        """
        return []

    def process_single(self, sample: DatasetSample, config: dict[str, Any]) -> DatasetSample:
        """
        Process a single dataset sample.

        Default implementation calls process() with a single-item list.
        Subclasses can override for more efficient single-sample processing.

        Args:
            sample: Sample to process
            config: Configuration dictionary

        Returns:
            Processed sample

        Raises:
            PreprocessorError: If preprocessing fails
        """
        # Run async process method in sync context

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            processed = loop.run_until_complete(self.process([sample], config))
            return processed[0] if processed else sample
        finally:
            if loop.is_running():
                pass  # Don't close running loop
            else:
                loop.close()

    async def process_batch(
        self,
        samples: list[DatasetSample],
        config: dict[str, Any],
        batch_size: int = 100,
        progress_callback: Callable[[PreprocessingProgress], None] | None = None,
    ) -> list[DatasetSample]:
        """
        Process samples in batches with progress reporting.

        Args:
            samples: List of samples to process
            config: Configuration dictionary
            batch_size: Number of samples to process in each batch
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed samples

        Raises:
            PreprocessorError: If preprocessing fails
        """
        if not samples:
            return []

        # Validate configuration
        warnings = self.validate_config(config)
        for warning in warnings:
            self.logger.warning(warning)

        # Setup progress tracking
        progress = PreprocessingProgress(len(samples))
        if progress_callback:
            progress.add_callback(progress_callback)
        progress.start()

        processed_samples = []
        failed_samples = 0

        try:
            # Process in batches
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]

                try:
                    # Process current batch
                    batch_processed = await self.process(batch, config)
                    processed_samples.extend(batch_processed)

                    # Update progress
                    progress.update(len(batch_processed))

                    self.logger.debug(
                        f"Processed batch {i // batch_size + 1}/{(len(samples) + batch_size - 1) // batch_size}"
                        f" ({len(batch_processed)}/{len(batch)} samples)"
                    )

                except Exception as e:
                    # Handle batch processing errors
                    failed_samples += len(batch)
                    progress.update(0, len(batch))

                    error_msg = f"Failed to process batch starting at index {i}: {e}"
                    self.logger.error(error_msg)

                    # Re-raise as PreprocessorError if not already
                    if not isinstance(e, PreprocessorError):
                        raise PreprocessorError(error_msg, cause=e) from e
                    raise

            progress.complete()

            self.logger.info(
                f"Preprocessing complete: {len(processed_samples)} samples processed, "
                f"{failed_samples} failed"
            )

            return processed_samples

        except Exception as e:
            # Ensure we have proper error reporting
            if not isinstance(e, PreprocessorError):
                raise PreprocessorError(f"Batch processing failed: {e}", cause=e) from e
            raise

    def _validate_samples(self, samples: list[DatasetSample]) -> None:
        """
        Validate that samples have required fields.

        Args:
            samples: List of samples to validate

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = self.get_required_fields()
        if not required_fields:
            return

        for i, sample in enumerate(samples):
            missing_fields = []

            for field in required_fields:
                if not hasattr(sample, field):
                    missing_fields.append(field)
                elif getattr(sample, field) is None and field in ["input_text", "label"]:
                    # Core required fields that can't be None
                    missing_fields.append(field)

            if missing_fields:
                raise ValueError(
                    f"Sample {i} (ID: {sample.id}) is missing required fields: {missing_fields}"
                )

    def _validate_config_keys(self, config: dict[str, Any]) -> list[str]:
        """
        Validate configuration keys and return warnings for unknown keys.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of warning messages for unknown keys
        """
        supported_keys = self.get_supported_config_keys()
        if not supported_keys:
            return []

        warnings = []
        for key in config:
            if key not in supported_keys:
                warnings.append(f"Unknown configuration key '{key}' for {self.name}")

        return warnings

    def __str__(self) -> str:
        """String representation of the preprocessor."""
        return f"{self.name}(required_fields={self.get_required_fields()})"

    def __repr__(self) -> str:
        """Detailed representation of the preprocessor."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"required_fields={self.get_required_fields()}, "
            f"supported_config={self.get_supported_config_keys()}"
            f")"
        )

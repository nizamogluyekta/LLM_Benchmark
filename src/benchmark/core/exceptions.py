"""
Custom exception hierarchy for the LLM Cybersecurity Benchmark system.
"""

from enum import Enum
from typing import Any


class ErrorCode(Enum):
    """Error codes for different exception types."""

    # General errors (1000-1999)
    UNKNOWN = 1000
    INTERNAL_ERROR = 1001
    INVALID_PARAMETER = 1002
    RESOURCE_NOT_FOUND = 1003

    # Configuration errors (2000-2999)
    CONFIG_VALIDATION_FAILED = 2000
    CONFIG_FILE_NOT_FOUND = 2001
    CONFIG_PARSE_ERROR = 2002
    INVALID_MODEL_CONFIG = 2003
    INVALID_DATASET_CONFIG = 2004
    INVALID_EXPERIMENT_CONFIG = 2005

    # Data loading errors (3000-3999)
    DATASET_NOT_FOUND = 3000
    DATASET_DOWNLOAD_FAILED = 3001
    DATASET_CORRUPTION = 3002
    DATASET_FORMAT_ERROR = 3003
    INSUFFICIENT_SAMPLES = 3004
    DATA_PREPROCESSING_FAILED = 3005

    # Model loading errors (4000-4999)
    MODEL_NOT_FOUND = 4000
    MODEL_DOWNLOAD_FAILED = 4001
    MODEL_INITIALIZATION_FAILED = 4002
    MODEL_INCOMPATIBLE = 4003
    INSUFFICIENT_MEMORY = 4004
    MLX_ERROR = 4005
    API_KEY_MISSING = 4006
    API_QUOTA_EXCEEDED = 4007

    # Evaluation errors (5000-5999)
    EVALUATION_SETUP_FAILED = 5000
    METRIC_CALCULATION_FAILED = 5001
    PREDICTION_FORMAT_ERROR = 5002
    GROUND_TRUTH_MISMATCH = 5003
    EVALUATION_TIMEOUT = 5004
    INSUFFICIENT_DATA = 5005

    # Service communication errors (6000-6999)
    SERVICE_UNAVAILABLE = 6000
    SERVICE_TIMEOUT = 6001
    SERVICE_AUTHENTICATION_FAILED = 6002
    SERVICE_RATE_LIMITED = 6003
    SERVICE_DEGRADED = 6004


class BenchmarkError(Exception):
    """Base exception for all benchmark-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        metadata: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """
        Initialize benchmark error.

        Args:
            message: Human-readable error message
            error_code: Specific error code for categorization
            metadata: Additional error context data
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.metadata = metadata or {}
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation of error."""
        return f"[{self.error_code.value}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation of error."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code={self.error_code}, "
            f"metadata={self.metadata}"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code.value,
            "error_code_name": self.error_code.name,
            "metadata": self.metadata,
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(BenchmarkError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CONFIG_VALIDATION_FAILED,
        metadata: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize configuration error."""
        super().__init__(message, error_code, metadata, cause)


class DataLoadingError(BenchmarkError):
    """Exception raised for data loading and processing errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATASET_NOT_FOUND,
        metadata: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize data loading error."""
        super().__init__(message, error_code, metadata, cause)


class ModelLoadingError(BenchmarkError):
    """Exception raised for model loading and initialization errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.MODEL_NOT_FOUND,
        metadata: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize model loading error."""
        super().__init__(message, error_code, metadata, cause)


class EvaluationError(BenchmarkError):
    """Exception raised for evaluation process errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.EVALUATION_SETUP_FAILED,
        metadata: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize evaluation error."""
        super().__init__(message, error_code, metadata, cause)


class ServiceUnavailableError(BenchmarkError):
    """Exception raised for service communication errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        metadata: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize service unavailable error."""
        super().__init__(message, error_code, metadata, cause)


# Convenience functions for creating common errors


def config_validation_error(field: str, value: Any, reason: str) -> ConfigurationError:
    """Create configuration validation error with detailed metadata."""
    return ConfigurationError(
        f"Configuration validation failed for field '{field}': {reason}",
        ErrorCode.CONFIG_VALIDATION_FAILED,
        metadata={"field": field, "value": str(value), "reason": reason},
    )


def dataset_not_found_error(dataset_name: str, source: str) -> DataLoadingError:
    """Create dataset not found error with source information."""
    return DataLoadingError(
        f"Dataset '{dataset_name}' not found from source '{source}'",
        ErrorCode.DATASET_NOT_FOUND,
        metadata={"dataset_name": dataset_name, "source": source},
    )


def model_memory_error(
    model_name: str, required_gb: float, available_gb: float
) -> ModelLoadingError:
    """Create model memory error with memory information."""
    return ModelLoadingError(
        f"Insufficient memory to load model '{model_name}': requires {required_gb}GB, "
        f"only {available_gb}GB available",
        ErrorCode.INSUFFICIENT_MEMORY,
        metadata={
            "model_name": model_name,
            "required_memory_gb": required_gb,
            "available_memory_gb": available_gb,
        },
    )


def api_key_missing_error(provider: str) -> ModelLoadingError:
    """Create API key missing error for specific provider."""
    return ModelLoadingError(
        f"API key missing for provider '{provider}'. "
        f"Please set the appropriate environment variable.",
        ErrorCode.API_KEY_MISSING,
        metadata={"provider": provider},
    )


def evaluation_timeout_error(experiment_id: str, timeout_seconds: float) -> EvaluationError:
    """Create evaluation timeout error."""
    return EvaluationError(
        f"Evaluation timed out for experiment '{experiment_id}' after {timeout_seconds} seconds",
        ErrorCode.EVALUATION_TIMEOUT,
        metadata={"experiment_id": experiment_id, "timeout_seconds": timeout_seconds},
    )


def service_timeout_error(service_name: str, timeout_seconds: float) -> ServiceUnavailableError:
    """Create service timeout error."""
    return ServiceUnavailableError(
        f"Service '{service_name}' timed out after {timeout_seconds} seconds",
        ErrorCode.SERVICE_TIMEOUT,
        metadata={"service_name": service_name, "timeout_seconds": timeout_seconds},
    )

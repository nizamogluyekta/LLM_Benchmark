"""Core module exports."""

from .base import (
    BaseService,
    HealthCheck,
    ServiceError,
    ServiceHealthError,
    ServiceInitializationError,
    ServiceResponse,
    ServiceShutdownError,
    ServiceStatus,
)
from .exceptions import (
    BenchmarkError,
    ConfigurationError,
    DataLoadingError,
    ErrorCode,
    EvaluationError,
    ModelLoadingError,
    ServiceUnavailableError,
    api_key_missing_error,
    config_validation_error,
    dataset_not_found_error,
    evaluation_timeout_error,
    model_memory_error,
    service_timeout_error,
)

__all__ = [
    # Base service classes
    "BaseService",
    "ServiceResponse",
    "HealthCheck",
    "ServiceStatus",
    "ServiceError",
    "ServiceInitializationError",
    "ServiceHealthError",
    "ServiceShutdownError",
    # Exception hierarchy
    "BenchmarkError",
    "ErrorCode",
    "ConfigurationError",
    "DataLoadingError",
    "ModelLoadingError",
    "EvaluationError",
    "ServiceUnavailableError",
    # Convenience functions
    "config_validation_error",
    "dataset_not_found_error",
    "model_memory_error",
    "api_key_missing_error",
    "evaluation_timeout_error",
    "service_timeout_error",
]

"""
Base service interface and common data structures for the benchmarking system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ServiceStatus(Enum):
    """Service health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceResponse(BaseModel):
    """Standard response format for service operations."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    model_config = {"use_enum_values": True}


class HealthCheck(BaseModel):
    """Health check response format."""

    service_name: str
    status: ServiceStatus
    timestamp: str
    details: dict[str, Any] | None = None
    uptime_seconds: float | None = None

    model_config = {"use_enum_values": True}


class BaseService(ABC):
    """Abstract base class for all services in the benchmarking system."""

    def __init__(self) -> None:
        """Initialize base service."""
        self._initialized = False
        self._start_time = datetime.now()

    @abstractmethod
    async def initialize(self) -> ServiceResponse:
        """Initialize the service with required resources."""
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheck:
        """Check the health status of the service."""
        pass

    @abstractmethod
    async def shutdown(self) -> ServiceResponse:
        """Gracefully shutdown the service and cleanup resources."""
        pass

    def is_initialized(self) -> bool:
        """Check if the service has been initialized."""
        return self._initialized

    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return (datetime.now() - self._start_time).total_seconds()

    def _mark_initialized(self) -> None:
        """Mark service as initialized (for subclass use)."""
        self._initialized = True

    def _mark_uninitialized(self) -> None:
        """Mark service as uninitialized (for subclass use)."""
        self._initialized = False


class ServiceError(Exception):
    """Base exception for service-related errors."""

    def __init__(self, message: str, service_name: str = "unknown") -> None:
        """Initialize service error."""
        super().__init__(message)
        self.service_name = service_name
        self.message = message


class ServiceInitializationError(ServiceError):
    """Exception raised when service initialization fails."""

    pass


class ServiceHealthError(ServiceError):
    """Exception raised when service health check fails."""

    pass


class ServiceShutdownError(ServiceError):
    """Exception raised when service shutdown fails."""

    pass

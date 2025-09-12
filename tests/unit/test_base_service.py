"""
Unit tests for base service interface and data structures.
"""

from datetime import datetime

import pytest

from benchmark.core import (
    BaseService,
    HealthCheck,
    ServiceError,
    ServiceHealthError,
    ServiceInitializationError,
    ServiceResponse,
    ServiceShutdownError,
    ServiceStatus,
)


class TestServiceResponse:
    """Test ServiceResponse data structure."""

    def test_success_response_creation(self):
        """Test creating successful response."""
        response = ServiceResponse(
            success=True, message="Operation successful", data={"key": "value"}
        )

        assert response.success is True
        assert response.message == "Operation successful"
        assert response.data == {"key": "value"}
        assert response.error is None
        assert response.metadata is None
        assert response.timestamp is not None

    def test_error_response_creation(self):
        """Test creating error response."""
        response = ServiceResponse(
            success=False,
            message="Operation failed",
            error="Something went wrong",
            metadata={"error_code": "E001"},
        )

        assert response.success is False
        assert response.data is None
        assert response.message == "Operation failed"
        assert response.error == "Something went wrong"
        assert response.metadata == {"error_code": "E001"}

    def test_response_serialization(self):
        """Test ServiceResponse can be serialized."""
        response = ServiceResponse(success=True, message="Success", data={"key": "value"})

        # Should be able to convert to dict
        response_dict = response.model_dump()
        assert isinstance(response_dict, dict)
        assert response_dict["success"] is True
        assert response_dict["data"] == {"key": "value"}

        # Should be able to convert to JSON
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "true" in json_str.lower()  # JSON boolean format


class TestHealthCheck:
    """Test HealthCheck data structure."""

    def test_healthy_status_creation(self):
        """Test creating healthy status."""
        health = HealthCheck(
            status=ServiceStatus.HEALTHY,
            message="Service is healthy",
            checks={"connections": 5, "memory_usage": "50MB"},
            timestamp=datetime.now().isoformat(),
        )

        assert health.status == ServiceStatus.HEALTHY.value
        assert health.message == "Service is healthy"
        assert health.checks == {"connections": 5, "memory_usage": "50MB"}
        assert health.uptime_seconds is None

    def test_unhealthy_status_creation(self):
        """Test creating unhealthy status."""
        health = HealthCheck(
            status=ServiceStatus.UNHEALTHY,
            message="Service is unhealthy",
            checks={"error": "Database connection failed"},
            timestamp=datetime.now().isoformat(),
            uptime_seconds=123.45,
        )

        assert health.status == ServiceStatus.UNHEALTHY.value
        assert health.message == "Service is unhealthy"
        assert health.checks == {"error": "Database connection failed"}
        assert health.uptime_seconds == 123.45

    def test_health_check_serialization(self):
        """Test HealthCheck can be serialized."""
        health = HealthCheck(
            status=ServiceStatus.DEGRADED,
            message="Service is degraded",
            timestamp=datetime.now().isoformat(),
        )

        # Should serialize to dict
        health_dict = health.model_dump()
        assert health_dict["status"] == "degraded"  # Enum value

        # Should serialize to JSON
        json_str = health.model_dump_json()
        assert "degraded" in json_str


class TestServiceStatus:
    """Test ServiceStatus enumeration."""

    def test_status_values(self):
        """Test service status enum values."""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"

    def test_status_comparison(self):
        """Test status comparison."""
        assert ServiceStatus.HEALTHY != ServiceStatus.UNHEALTHY
        assert ServiceStatus.DEGRADED != ServiceStatus.HEALTHY


class MockService(BaseService):
    """Mock service implementation for testing."""

    def __init__(self):
        super().__init__("mock_service")
        self.initialization_called = False
        self.health_check_called = False
        self.shutdown_called = False

    async def initialize(self) -> ServiceResponse:
        """Mock initialization."""
        self.initialization_called = True
        self._mark_initialized()
        return ServiceResponse(
            success=True, message="Service initialized", data={"initialized": True}
        )

    async def health_check(self) -> HealthCheck:
        """Mock health check."""
        self.health_check_called = True
        return HealthCheck(
            status=ServiceStatus.HEALTHY,
            message="Service is healthy",
            checks={"mock": True},
            timestamp=datetime.now().isoformat(),
            uptime_seconds=self.get_uptime_seconds(),
        )

    async def shutdown(self) -> ServiceResponse:
        """Mock shutdown."""
        self.shutdown_called = True
        self._mark_uninitialized()
        return ServiceResponse(success=True, message="Service shutdown", data={"shutdown": True})


class TestBaseService:
    """Test BaseService interface."""

    def test_base_service_creation(self):
        """Test base service can be instantiated."""
        service = MockService()

        assert not service.is_initialized()
        assert service.get_uptime_seconds() >= 0
        assert not service.initialization_called
        assert not service.health_check_called
        assert not service.shutdown_called

    @pytest.mark.asyncio
    async def test_service_lifecycle(self):
        """Test complete service lifecycle."""
        service = MockService()

        # Initial state
        assert not service.is_initialized()

        # Initialize
        init_response = await service.initialize()
        assert init_response.success is True
        assert service.is_initialized()
        assert service.initialization_called

        # Health check
        health = await service.health_check()
        assert health.message == "Service is healthy"
        assert health.status == ServiceStatus.HEALTHY.value
        assert health.uptime_seconds is not None
        assert health.uptime_seconds >= 0
        assert service.health_check_called

        # Shutdown
        shutdown_response = await service.shutdown()
        assert shutdown_response.success is True
        assert not service.is_initialized()
        assert service.shutdown_called

    def test_uptime_tracking(self):
        """Test uptime tracking works."""
        service = MockService()

        # Should have some uptime immediately
        uptime1 = service.get_uptime_seconds()
        assert uptime1 >= 0

        # Uptime should increase
        import time

        time.sleep(0.01)  # Sleep 10ms
        uptime2 = service.get_uptime_seconds()
        assert uptime2 > uptime1

    def test_initialization_state_tracking(self):
        """Test initialization state is tracked correctly."""
        service = MockService()

        assert not service.is_initialized()

        # Manually mark as initialized
        service._mark_initialized()
        assert service.is_initialized()

        # Manually mark as uninitialized
        service._mark_uninitialized()
        assert not service.is_initialized()


class TestServiceExceptions:
    """Test service exception classes."""

    def test_service_error_creation(self):
        """Test ServiceError creation."""
        error = ServiceError("Test error", "test_service")

        assert str(error) == "Test error"
        assert error.service_name == "test_service"
        assert error.message == "Test error"

    def test_service_error_default_service_name(self):
        """Test ServiceError with default service name."""
        error = ServiceError("Test error")

        assert error.service_name == "unknown"
        assert error.message == "Test error"

    def test_initialization_error(self):
        """Test ServiceInitializationError."""
        error = ServiceInitializationError("Init failed", "init_service")

        assert isinstance(error, ServiceError)
        assert str(error) == "Init failed"
        assert error.service_name == "init_service"

    def test_health_error(self):
        """Test ServiceHealthError."""
        error = ServiceHealthError("Health check failed", "health_service")

        assert isinstance(error, ServiceError)
        assert str(error) == "Health check failed"
        assert error.service_name == "health_service"

    def test_shutdown_error(self):
        """Test ServiceShutdownError."""
        error = ServiceShutdownError("Shutdown failed", "shutdown_service")

        assert isinstance(error, ServiceError)
        assert str(error) == "Shutdown failed"
        assert error.service_name == "shutdown_service"


class TestDataClassValidation:
    """Test Pydantic validation for data classes."""

    def test_service_response_validation(self):
        """Test ServiceResponse validates input."""
        # Valid response
        response = ServiceResponse(success=True, message="Success")
        assert response.success is True
        assert response.message == "Success"

        # Test with invalid data should still work (Pydantic is flexible with dict)
        response = ServiceResponse(success=False, message="Failed", data={"any": "value"})
        assert response.success is False
        assert response.message == "Failed"
        assert response.data == {"any": "value"}

    def test_health_check_validation(self):
        """Test HealthCheck validates input."""
        # Valid health check
        health = HealthCheck(
            status=ServiceStatus.HEALTHY,
            message="Service is healthy",
            timestamp="2024-01-01T10:00:00",
        )
        assert health.message == "Service is healthy"
        assert health.status == ServiceStatus.HEALTHY.value

        # Test enum serialization
        health_dict = health.model_dump()
        assert health_dict["status"] == "healthy"

    def test_required_fields(self):
        """Test required fields are enforced."""
        # ServiceResponse requires success
        with pytest.raises(ValueError):
            ServiceResponse()  # Missing required success field

        # HealthCheck requires service_name, status, timestamp
        with pytest.raises(ValueError):
            HealthCheck(service_name="test")  # Missing status and timestamp

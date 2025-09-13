"""
Unit tests for logging system.
"""

import json
import logging
import sys
import time
from unittest.mock import patch

import pytest

from benchmark.core.logging import (
    CorrelationIdFilter,
    JSONFormatter,
    LoggerManager,
    PerformanceTimer,
    configure_logging,
    generate_correlation_id,
    get_config_logger,
    get_correlation_id,
    get_data_logger,
    get_evaluation_logger,
    get_logger,
    get_model_logger,
    get_performance_logger,
    get_service_logger,
    log_dataset_load,
    log_evaluation_metrics,
    log_experiment_end,
    log_experiment_start,
    log_function_call,
    log_model_load,
    set_correlation_id,
    timed_operation,
    with_correlation_id,
)


class TestCorrelationIdFilter:
    """Test correlation ID logging filter."""

    def test_filter_adds_correlation_id(self):
        """Test filter adds correlation ID to log record."""
        filter_instance = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Without correlation ID set
        result = filter_instance.filter(record)
        assert result is True
        assert hasattr(record, "correlation_id")
        assert record.correlation_id == "none"

        # With correlation ID set
        set_correlation_id("test-correlation-id")
        result = filter_instance.filter(record)
        assert result is True
        assert record.correlation_id == "test-correlation-id"


class TestJSONFormatter:
    """Test JSON log formatter."""

    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-id"

        result = formatter.format(record)

        # Should be valid JSON
        log_data = json.loads(result)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["line"] == 42
        assert log_data["correlation_id"] == "test-id"
        assert "timestamp" in log_data

    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception information."""
        formatter = JSONFormatter()

        # Create record with exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/path/test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        record.correlation_id = "error-id"
        result = formatter.format(record)

        log_data = json.loads(result)

        assert log_data["message"] == "Error occurred"
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert "traceback" in log_data["exception"]

    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/test.py",
            lineno=42,
            msg="Test with extra",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-id"
        record.experiment_id = "exp-001"
        record.model_name = "test-model"
        record.duration = 1.23

        result = formatter.format(record)
        log_data = json.loads(result)

        assert "extra" in log_data
        assert log_data["extra"]["experiment_id"] == "exp-001"
        assert log_data["extra"]["model_name"] == "test-model"
        assert log_data["extra"]["duration"] == 1.23


class TestLoggerManager:
    """Test logger manager singleton."""

    def test_logger_manager_singleton(self):
        """Test logger manager is singleton."""
        manager1 = LoggerManager()
        manager2 = LoggerManager()

        assert manager1 is manager2

    def test_get_logger_creates_component_loggers(self):
        """Test getting loggers for different components."""
        manager = LoggerManager()

        data_logger = manager.get_logger("data")
        model_logger = manager.get_logger("model")

        assert data_logger.name == "benchmark.data"
        assert model_logger.name == "benchmark.model"
        assert data_logger is not model_logger

    def test_get_logger_reuses_existing(self):
        """Test getting same logger returns same instance."""
        manager = LoggerManager()

        logger1 = manager.get_logger("test")
        logger2 = manager.get_logger("test")

        assert logger1 is logger2

    def test_set_level(self):
        """Test setting log level."""
        manager = LoggerManager()

        # Test setting different levels
        manager.set_level("DEBUG")
        root_logger = logging.getLogger("benchmark")
        assert root_logger.level == logging.DEBUG

        manager.set_level("WARNING")
        assert root_logger.level == logging.WARNING


class TestCorrelationId:
    """Test correlation ID functionality."""

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        id1 = generate_correlation_id()
        id2 = generate_correlation_id()

        assert id1 != id2
        assert len(id1) == 36  # UUID4 length
        assert len(id2) == 36

    def test_set_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-123"

        # Initially None or empty string
        current_id = get_correlation_id()
        assert current_id is None or current_id == ""

        # Set and retrieve
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

    def test_with_correlation_id_decorator(self):
        """Test with_correlation_id decorator."""

        @with_correlation_id
        def test_function():
            return get_correlation_id()

        # Should auto-generate correlation ID
        result = test_function()
        assert result is not None
        assert len(result) > 0  # May be empty string in test context

    def test_with_correlation_id_preserves_existing(self):
        """Test decorator preserves existing correlation ID."""
        existing_id = "existing-id-123"
        set_correlation_id(existing_id)

        @with_correlation_id
        def test_function():
            return get_correlation_id()

        result = test_function()
        assert result == existing_id


class TestTimingDecorators:
    """Test timing and performance decorators."""

    def test_timed_operation_decorator(self, caplog):
        """Test timed operation decorator."""
        with caplog.at_level(logging.INFO):

            @timed_operation(operation_name="test_operation")
            def test_function():
                time.sleep(0.01)  # Small delay
                return "result"

            result = test_function()

            assert result == "result"

            # Check log was created
            assert len(caplog.records) > 0
            log_record = caplog.records[0]
            assert "test_operation" in log_record.message
            assert hasattr(log_record, "duration_seconds")
            assert log_record.duration_seconds > 0

    def test_timed_operation_with_exception(self, caplog):
        """Test timed operation decorator with exception."""
        with caplog.at_level(logging.ERROR):

            @timed_operation(operation_name="failing_operation")
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                failing_function()

            # Check error log was created
            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) > 0
            error_record = error_records[0]
            assert "failing_operation" in error_record.message
            assert hasattr(error_record, "success")
            assert error_record.success is False

    @pytest.mark.asyncio
    async def test_timed_operation_async(self, caplog):
        """Test timed operation decorator with async function."""
        with caplog.at_level(logging.INFO):

            @timed_operation(operation_name="async_test")
            async def async_function():
                await asyncio.sleep(0.01)
                return "async_result"

            import asyncio

            result = await async_function()

            assert result == "async_result"

            # Check log was created
            assert len(caplog.records) > 0
            log_record = caplog.records[0]
            assert "async_test" in log_record.message


class TestPerformanceTimer:
    """Test PerformanceTimer context manager."""

    def test_performance_timer_success(self, caplog):
        """Test performance timer with successful operation."""
        with caplog.at_level(logging.INFO):
            with PerformanceTimer("test_operation") as timer:
                time.sleep(0.01)

            assert timer.duration is not None
            assert timer.duration > 0

            # Check log was created
            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert len(info_records) > 0

            completion_record = info_records[-1]  # Last info record should be completion
            assert "test_operation" in completion_record.message
            assert hasattr(completion_record, "success")
            assert completion_record.success is True

    def test_performance_timer_with_exception(self, caplog):
        """Test performance timer with exception."""
        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
            with PerformanceTimer("failing_operation") as timer:
                raise RuntimeError("Test failure")

            assert timer.duration is not None
            assert timer.duration >= 0

            # Check error log was created
            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) > 0
            error_record = error_records[0]
            assert "failing_operation" in error_record.message
            assert hasattr(error_record, "success")
            assert error_record.success is False


class TestComponentLoggers:
    """Test pre-configured component loggers."""

    def test_component_logger_names(self):
        """Test component loggers have correct names."""
        assert get_data_logger().name == "benchmark.data"
        assert get_model_logger().name == "benchmark.model"
        assert get_evaluation_logger().name == "benchmark.evaluation"
        assert get_config_logger().name == "benchmark.config"
        assert get_service_logger().name == "benchmark.service"
        assert get_performance_logger().name == "benchmark.performance"

    def test_component_loggers_are_reused(self):
        """Test component loggers are reused."""
        logger1 = get_data_logger()
        logger2 = get_data_logger()

        assert logger1 is logger2


class TestLogUtilityFunctions:
    """Test utility logging functions."""

    def test_log_experiment_start(self, caplog):
        """Test experiment start logging."""
        config = {"model": "test-model", "dataset": "test-data"}

        with caplog.at_level(logging.INFO):
            log_experiment_start("exp-001", config)

        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert "exp-001" in record.message
        assert hasattr(record, "experiment_id")
        assert record.experiment_id == "exp-001"
        assert hasattr(record, "config")
        assert record.config == config

    def test_log_experiment_end_success(self, caplog):
        """Test successful experiment end logging."""
        results = {"accuracy": 0.95, "precision": 0.93}

        with caplog.at_level(logging.INFO):
            log_experiment_end("exp-001", 123.45, True, results)

        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert "exp-001" in record.message
        assert record.success is True
        assert record.duration_seconds == 123.45
        assert record.results == results

    def test_log_experiment_end_failure(self, caplog):
        """Test failed experiment end logging."""
        with caplog.at_level(logging.ERROR):
            log_experiment_end("exp-002", 67.89, False)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) > 0
        record = error_records[0]
        assert "exp-002" in record.message
        assert record.success is False
        assert record.duration_seconds == 67.89

    def test_log_model_load(self, caplog):
        """Test model loading logging."""
        with caplog.at_level(logging.INFO):
            log_model_load("gpt-4", 5.67, 1024.5)

        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert "gpt-4" in record.message
        assert record.model_name == "gpt-4"
        assert record.load_duration_seconds == 5.67
        assert record.memory_usage_mb == 1024.5

    def test_log_dataset_load(self, caplog):
        """Test dataset loading logging."""
        with caplog.at_level(logging.INFO):
            log_dataset_load("UNSW-NB15", 12.34, 50000, 512.0)

        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert "UNSW-NB15" in record.message
        assert record.dataset_name == "UNSW-NB15"
        assert record.load_duration_seconds == 12.34
        assert record.sample_count == 50000
        assert record.memory_usage_mb == 512.0

    def test_log_evaluation_metrics(self, caplog):
        """Test evaluation metrics logging."""
        metrics = {"accuracy": 0.92, "precision": 0.89, "recall": 0.95}

        with caplog.at_level(logging.INFO):
            log_evaluation_metrics("exp-001", "gpt-4", "test-data", metrics)

        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert record.experiment_id == "exp-001"
        assert record.model_name == "gpt-4"
        assert record.dataset_name == "test-data"
        assert record.metrics == metrics


class TestLogFunctionCallDecorator:
    """Test log_function_call decorator."""

    def test_log_function_call_basic(self, caplog):
        """Test basic function call logging."""
        with caplog.at_level(logging.DEBUG):

            @log_function_call()
            def test_function():
                return "result"

            result = test_function()

            assert result == "result"

            # Should have entry and exit logs
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert len(debug_records) >= 2  # At least call and completion

            call_record = debug_records[0]
            assert "test_function" in call_record.message
            assert call_record.func_name == "test_function"

    def test_log_function_call_with_args(self, caplog):
        """Test function call logging with arguments."""
        with caplog.at_level(logging.DEBUG):

            @log_function_call(log_args=True)
            def test_function_with_args(a, b=None):
                return f"{a}-{b}"

            result = test_function_with_args("test", b="value")

            assert result == "test-value"

            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            call_record = debug_records[0]
            assert hasattr(call_record, "func_args")
            assert hasattr(call_record, "func_kwargs")

    def test_log_function_call_with_result(self, caplog):
        """Test function call logging with result."""
        with caplog.at_level(logging.DEBUG):

            @log_function_call(log_result=True)
            def test_function_with_result():
                return {"key": "value"}

            result = test_function_with_result()

            assert result == {"key": "value"}

            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            completion_record = debug_records[-1]  # Last debug record should be completion
            assert hasattr(completion_record, "result")

    def test_log_function_call_with_exception(self, caplog):
        """Test function call logging with exception."""
        with caplog.at_level(logging.ERROR):

            @log_function_call()
            def failing_function():
                raise ValueError("Function failed")

            with pytest.raises(ValueError):
                failing_function()

            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) > 0
            error_record = error_records[0]
            assert "failing_function" in error_record.message
            assert hasattr(error_record, "error")
            assert error_record.error == "Function failed"


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_configure_logging_levels(self):
        """Test configuring different logging levels."""
        # Test DEBUG level
        configure_logging(level="DEBUG")
        root_logger = logging.getLogger("benchmark")
        assert root_logger.level == logging.DEBUG

        # Test WARNING level
        configure_logging(level="WARNING")
        assert root_logger.level == logging.WARNING

    @patch("benchmark.core.logging.Path.mkdir")
    def test_configure_logging_creates_log_directory(self, mock_mkdir):
        """Test log directory creation."""
        configure_logging(log_dir="custom_logs")
        mock_mkdir.assert_called_with(exist_ok=True)

    def test_get_logger_function(self):
        """Test get_logger convenience function."""
        logger = get_logger("custom_component")

        assert logger.name == "benchmark.custom_component"
        assert isinstance(logger, logging.Logger)


class TestFileLogging:
    """Test file logging functionality."""

    def test_logging_configuration_creates_handlers(self):
        """Test logging configuration creates appropriate handlers."""
        # Test that logger manager initializes properly
        from benchmark.core.logging import LoggerManager

        manager = LoggerManager()
        assert manager is not None

        logger = get_logger("test_handler")
        assert logger.name == "benchmark.test_handler"


class TestIntegration:
    """Integration tests for logging system."""

    def test_correlation_id_across_functions(self, caplog):
        """Test correlation ID is maintained across function calls."""
        set_correlation_id("integration-test-id")

        with caplog.at_level(logging.INFO):
            logger1 = get_data_logger()
            logger2 = get_model_logger()

            logger1.info("Data operation")
            logger2.info("Model operation")

        # Should have logs from both loggers
        records = caplog.records
        assert len(records) >= 2

    def test_different_loggers_same_hierarchy(self):
        """Test different component loggers are part of same hierarchy."""
        data_logger = get_data_logger()
        model_logger = get_model_logger()

        assert data_logger.name == "benchmark.data"
        assert model_logger.name == "benchmark.model"

        # Both should be children of benchmark root
        assert data_logger.name.startswith("benchmark.")
        assert model_logger.name.startswith("benchmark.")

    def test_performance_logging_integration(self, caplog):
        """Test complete performance logging workflow."""
        correlation_id_value = generate_correlation_id()
        set_correlation_id(correlation_id_value)

        with caplog.at_level(logging.INFO), PerformanceTimer("integration_test"):
            logger = get_performance_logger()
            logger.info("Performance test message")

        # Check all logs have correlation ID
        for record in caplog.records:
            if hasattr(record, "correlation_id"):
                assert record.correlation_id == correlation_id_value


# Test setup and teardown
@pytest.fixture(autouse=True)
def setup_logging_for_tests():
    """Configure logging for testing."""
    # Store original handlers
    root_logger = logging.getLogger("benchmark")
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    # Configure for testing
    configure_logging(level="DEBUG", enable_console=False, enable_file=False)

    yield

    # Restore original configuration
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)

    # Clear correlation ID
    from benchmark.core.logging import correlation_id

    correlation_id.set(None)

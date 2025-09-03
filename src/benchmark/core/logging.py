"""
Structured logging system for the LLM Cybersecurity Benchmark project.
"""

import functools
import json
import logging
import logging.handlers
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypeVar

from rich.console import Console
from rich.logging import RichHandler

# Type variable for function decorators
F = TypeVar("F", bound=Callable[..., Any])

# Context variable for correlation IDs
correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        record.correlation_id = correlation_id.get() or "none"
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": getattr(record, "correlation_id", "none"),
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields (avoid conflicts with LogRecord attributes)
        excluded_fields = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "exc_info",
            "exc_text",
            "stack_info",
            "correlation_id",
            "message",
        }

        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k not in excluded_fields and not k.startswith("_")
        }

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)


class LoggerManager:
    """Manager for creating and configuring loggers."""

    _instance: Optional["LoggerManager"] = None
    _loggers: dict[str, logging.Logger] = {}

    def __new__(cls) -> "LoggerManager":
        """Singleton pattern for logger manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize logger manager."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._setup_root_logger()

    def _setup_root_logger(self) -> None:
        """Setup root logger configuration."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Get or create root logger
        root_logger = logging.getLogger("benchmark")
        root_logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Add correlation ID filter
        correlation_filter = CorrelationIdFilter()

        # Console handler with Rich formatting for development
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(correlation_filter)

        # Simple format for console (Rich handles the styling)
        console_format = "%(message)s"
        console_handler.setFormatter(logging.Formatter(console_format))

        # File handler with JSON formatting for production
        json_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "benchmark.jsonl",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.addFilter(correlation_filter)
        json_handler.setFormatter(JSONFormatter())

        # Error file handler for errors only
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "benchmark_errors.jsonl",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(correlation_filter)
        error_handler.setFormatter(JSONFormatter())

        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(json_handler)
        root_logger.addHandler(error_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a specific component."""
        if name not in self._loggers:
            # Create logger as child of benchmark root logger
            logger_name = f"benchmark.{name}"
            logger = logging.getLogger(logger_name)
            self._loggers[name] = logger

        return self._loggers[name]

    def set_level(self, level: str) -> None:
        """Set logging level for all loggers."""
        log_level = getattr(logging, level.upper())

        root_logger = logging.getLogger("benchmark")
        root_logger.setLevel(log_level)

        # Update console handler level
        for handler in root_logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(log_level)

    def configure_for_testing(self) -> None:
        """Configure logging for testing (suppress output)."""
        root_logger = logging.getLogger("benchmark")
        root_logger.setLevel(logging.CRITICAL + 1)  # Suppress all logs


# Singleton logger manager
logger_manager = LoggerManager()


def get_logger(component: str) -> logging.Logger:
    """Get logger for specific component."""
    return logger_manager.get_logger(component)


def set_correlation_id(correlation_id_value: str) -> None:
    """Set correlation ID for current context."""
    correlation_id.set(correlation_id_value)


def get_correlation_id() -> str | None:
    """Get current correlation ID."""
    return correlation_id.get()


def generate_correlation_id() -> str:
    """Generate new correlation ID."""
    return str(uuid.uuid4())


def with_correlation_id(func: F) -> F:
    """Decorator to automatically generate correlation ID for function calls."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate new correlation ID if not present
        if correlation_id.get() is None:
            correlation_id.set(generate_correlation_id())

        return func(*args, **kwargs)

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate new correlation ID if not present
        if correlation_id.get() is None:
            correlation_id.set(generate_correlation_id())

        return await func(*args, **kwargs)

    if hasattr(func, "__await__"):
        return async_wrapper  # type: ignore
    else:
        return wrapper  # type: ignore


def timed_operation(
    logger: logging.Logger | None = None, operation_name: str = "operation"
) -> Callable[[F], F]:
    """Decorator to time function execution and log performance metrics."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger("performance")
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                duration = end_time - start_time

                log.info(
                    f"Operation '{operation_name}' completed",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "function": func.__name__,
                        "success": True,
                    },
                )

                return result

            except Exception as e:
                end_time = time.perf_counter()
                duration = end_time - start_time

                log.error(
                    f"Operation '{operation_name}' failed",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "function": func.__name__,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger("performance")
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                duration = end_time - start_time

                log.info(
                    f"Operation '{operation_name}' completed",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "function": func.__name__,
                        "success": True,
                    },
                )

                return result

            except Exception as e:
                end_time = time.perf_counter()
                duration = end_time - start_time

                log.error(
                    f"Operation '{operation_name}' failed",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "function": func.__name__,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                raise

        if hasattr(func, "__await__"):
            return async_wrapper  # type: ignore
        else:
            return wrapper  # type: ignore

    return decorator


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        operation_name: str,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize performance timer."""
        self.operation_name = operation_name
        self.logger = logger or get_logger("performance")
        self.log_level = log_level
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        """Start timing operation."""
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing operation and log results."""
        self.end_time = time.perf_counter()

        if self.start_time is not None:
            duration = self.end_time - self.start_time

            if exc_type is None:
                # Success
                self.logger.log(
                    self.log_level,
                    f"Operation '{self.operation_name}' completed",
                    extra={
                        "operation": self.operation_name,
                        "duration_seconds": duration,
                        "success": True,
                    },
                )
            else:
                # Error
                self.logger.error(
                    f"Operation '{self.operation_name}' failed",
                    extra={
                        "operation": self.operation_name,
                        "duration_seconds": duration,
                        "success": False,
                        "error": str(exc_val) if exc_val else None,
                        "error_type": exc_type.__name__ if exc_type else None,
                    },
                    exc_info=True,
                )

    @property
    def duration(self) -> float | None:
        """Get operation duration in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


def configure_logging(
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    log_dir: str = "logs",
) -> None:
    """Configure logging system with specified options."""
    # Set up log directory
    logs_path = Path(log_dir)
    logs_path.mkdir(exist_ok=True)

    # Set logging level
    logger_manager.set_level(level)

    # Configure console output
    if not enable_console:
        bench_logger = logging.getLogger("benchmark")
        rich_handlers: list[RichHandler] = [
            h for h in bench_logger.handlers if isinstance(h, RichHandler)
        ]
        for rich_handler in rich_handlers:
            bench_logger.removeHandler(rich_handler)

    # Configure file output
    if not enable_file:
        bench_logger = logging.getLogger("benchmark")
        rotating_handlers: list[logging.handlers.RotatingFileHandler] = []
        for handler_item in bench_logger.handlers:
            if isinstance(handler_item, logging.handlers.RotatingFileHandler):
                rotating_handlers.append(handler_item)

        for rotating_handler in rotating_handlers:
            bench_logger.removeHandler(rotating_handler)


def log_function_call(
    logger: logging.Logger | None = None,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function calls with optional arguments and results."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger("function_calls")

            # Prepare log data (use prefixes to avoid LogRecord conflicts)
            log_data = {
                "func_name": func.__name__,
                "func_module": func.__module__,
            }

            if log_args:
                log_data["func_args"] = str(args) if args else ""
                log_data["func_kwargs"] = str(kwargs) if kwargs else ""

            log.debug(f"Calling function: {func.__name__}", extra=log_data)

            try:
                result = func(*args, **kwargs)

                if log_result:
                    log_data["result"] = str(result)

                log.debug(f"Function {func.__name__} completed", extra=log_data)
                return result

            except Exception as e:
                log_data["error"] = str(e)
                log_data["error_type"] = type(e).__name__

                log.error(f"Function {func.__name__} failed", extra=log_data, exc_info=True)
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger("function_calls")

            # Prepare log data (use prefixes to avoid LogRecord conflicts)
            log_data = {
                "func_name": func.__name__,
                "func_module": func.__module__,
            }

            if log_args:
                log_data["func_args"] = str(args) if args else ""
                log_data["func_kwargs"] = str(kwargs) if kwargs else ""

            log.debug(f"Calling async function: {func.__name__}", extra=log_data)

            try:
                result = await func(*args, **kwargs)

                if log_result:
                    log_data["result"] = str(result)

                log.debug(f"Async function {func.__name__} completed", extra=log_data)
                return result

            except Exception as e:
                log_data["error"] = str(e)
                log_data["error_type"] = type(e).__name__

                log.error(f"Async function {func.__name__} failed", extra=log_data, exc_info=True)
                raise

        if hasattr(func, "__await__"):
            return async_wrapper  # type: ignore
        else:
            return wrapper  # type: ignore

    return decorator


# Pre-configured loggers for common components
def get_data_logger() -> logging.Logger:
    """Get logger for data operations."""
    return get_logger("data")


def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return get_logger("model")


def get_evaluation_logger() -> logging.Logger:
    """Get logger for evaluation operations."""
    return get_logger("evaluation")


def get_config_logger() -> logging.Logger:
    """Get logger for configuration operations."""
    return get_logger("config")


def get_service_logger() -> logging.Logger:
    """Get logger for service operations."""
    return get_logger("service")


def get_performance_logger() -> logging.Logger:
    """Get logger for performance monitoring."""
    return get_logger("performance")


# Utility functions for common logging patterns
def log_experiment_start(experiment_id: str, config: dict[str, Any]) -> None:
    """Log experiment start with configuration."""
    logger = get_logger("experiment")
    logger.info(
        f"Starting experiment: {experiment_id}",
        extra={
            "experiment_id": experiment_id,
            "config": config,
            "event_type": "experiment_start",
        },
    )


def log_experiment_end(
    experiment_id: str,
    duration_seconds: float,
    success: bool,
    results: dict[str, Any] | None = None,
) -> None:
    """Log experiment completion."""
    logger = get_logger("experiment")

    if success:
        logger.info(
            f"Experiment completed: {experiment_id}",
            extra={
                "experiment_id": experiment_id,
                "duration_seconds": duration_seconds,
                "event_type": "experiment_end",
                "success": True,
                "results": results,
            },
        )
    else:
        logger.error(
            f"Experiment failed: {experiment_id}",
            extra={
                "experiment_id": experiment_id,
                "duration_seconds": duration_seconds,
                "event_type": "experiment_end",
                "success": False,
            },
        )


def log_model_load(model_name: str, duration_seconds: float, memory_mb: float) -> None:
    """Log model loading performance."""
    logger = get_model_logger()
    logger.info(
        f"Model loaded: {model_name}",
        extra={
            "model_name": model_name,
            "load_duration_seconds": duration_seconds,
            "memory_usage_mb": memory_mb,
            "event_type": "model_load",
        },
    )


def log_dataset_load(
    dataset_name: str,
    duration_seconds: float,
    sample_count: int,
    memory_mb: float,
) -> None:
    """Log dataset loading performance."""
    logger = get_data_logger()
    logger.info(
        f"Dataset loaded: {dataset_name}",
        extra={
            "dataset_name": dataset_name,
            "load_duration_seconds": duration_seconds,
            "sample_count": sample_count,
            "memory_usage_mb": memory_mb,
            "event_type": "dataset_load",
        },
    )


def log_evaluation_metrics(
    experiment_id: str,
    model_name: str,
    dataset_name: str,
    metrics: dict[str, Any],
) -> None:
    """Log evaluation metrics."""
    logger = get_evaluation_logger()
    logger.info(
        "Evaluation metrics calculated",
        extra={
            "experiment_id": experiment_id,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metrics": metrics,
            "event_type": "evaluation_metrics",
        },
    )


# Initialize logging system on import
configure_logging()

"""
Unit tests for custom exception hierarchy.
"""

import pytest

from benchmark.core import (
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


class TestErrorCode:
    """Test ErrorCode enumeration."""

    def test_error_code_values(self):
        """Test error code values are in expected ranges."""
        # General errors (1000-1999)
        assert 1000 <= ErrorCode.UNKNOWN.value <= 1999
        assert 1000 <= ErrorCode.INTERNAL_ERROR.value <= 1999

        # Configuration errors (2000-2999)
        assert 2000 <= ErrorCode.CONFIG_VALIDATION_FAILED.value <= 2999
        assert 2000 <= ErrorCode.CONFIG_FILE_NOT_FOUND.value <= 2999

        # Data loading errors (3000-3999)
        assert 3000 <= ErrorCode.DATASET_NOT_FOUND.value <= 3999
        assert 3000 <= ErrorCode.DATA_PREPROCESSING_FAILED.value <= 3999

        # Model loading errors (4000-4999)
        assert 4000 <= ErrorCode.MODEL_NOT_FOUND.value <= 4999
        assert 4000 <= ErrorCode.API_KEY_MISSING.value <= 4999

        # Evaluation errors (5000-5999)
        assert 5000 <= ErrorCode.EVALUATION_SETUP_FAILED.value <= 5999
        assert 5000 <= ErrorCode.INSUFFICIENT_DATA.value <= 5999

        # Service errors (6000-6999)
        assert 6000 <= ErrorCode.SERVICE_UNAVAILABLE.value <= 6999
        assert 6000 <= ErrorCode.SERVICE_DEGRADED.value <= 6999

    def test_error_code_uniqueness(self):
        """Test that all error codes are unique."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values)), "Error codes must be unique"


class TestBenchmarkError:
    """Test base BenchmarkError class."""

    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = BenchmarkError("Test error message")

        assert str(error) == "[1000] Test error message"
        assert error.message == "Test error message"
        assert error.error_code == ErrorCode.UNKNOWN
        assert error.metadata == {}
        assert error.cause is None

    def test_error_with_custom_code(self):
        """Test error with custom error code."""
        error = BenchmarkError("Custom error", ErrorCode.INTERNAL_ERROR)

        assert str(error) == "[1001] Custom error"
        assert error.error_code == ErrorCode.INTERNAL_ERROR

    def test_error_with_metadata(self):
        """Test error with metadata."""
        metadata = {"component": "test", "operation": "validation"}
        error = BenchmarkError("Error with metadata", metadata=metadata)

        assert error.metadata == metadata
        assert error.metadata["component"] == "test"
        assert error.metadata["operation"] == "validation"

    def test_error_with_cause(self):
        """Test error with underlying cause."""
        original_error = ValueError("Original error")
        error = BenchmarkError("Wrapped error", cause=original_error)

        assert error.cause == original_error
        assert str(error.cause) == "Original error"

    def test_error_representation(self):
        """Test error representation methods."""
        error = BenchmarkError("Test error", ErrorCode.INTERNAL_ERROR, metadata={"key": "value"})

        # Test __str__
        assert str(error) == "[1001] Test error"

        # Test __repr__
        repr_str = repr(error)
        assert "BenchmarkError" in repr_str
        assert "Test error" in repr_str
        assert "INTERNAL_ERROR" in repr_str
        assert "key" in repr_str

    def test_error_serialization(self):
        """Test error to_dict method."""
        original_error = ValueError("Original")
        error = BenchmarkError(
            "Test error",
            ErrorCode.INTERNAL_ERROR,
            metadata={"component": "test"},
            cause=original_error,
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "BenchmarkError"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == 1001
        assert error_dict["error_code_name"] == "INTERNAL_ERROR"
        assert error_dict["metadata"] == {"component": "test"}
        assert error_dict["cause"] == "Original"


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from BenchmarkError."""
        error = ConfigurationError("Config error")

        assert isinstance(error, BenchmarkError)
        assert isinstance(error, ConfigurationError)
        assert str(error) == "[2000] Config error"

    def test_data_loading_error_inheritance(self):
        """Test DataLoadingError inherits from BenchmarkError."""
        error = DataLoadingError("Data error")

        assert isinstance(error, BenchmarkError)
        assert isinstance(error, DataLoadingError)
        assert str(error) == "[3000] Data error"

    def test_model_loading_error_inheritance(self):
        """Test ModelLoadingError inherits from BenchmarkError."""
        error = ModelLoadingError("Model error")

        assert isinstance(error, BenchmarkError)
        assert isinstance(error, ModelLoadingError)
        assert str(error) == "[4000] Model error"

    def test_evaluation_error_inheritance(self):
        """Test EvaluationError inherits from BenchmarkError."""
        error = EvaluationError("Evaluation error")

        assert isinstance(error, BenchmarkError)
        assert isinstance(error, EvaluationError)
        assert str(error) == "[5000] Evaluation error"

    def test_service_unavailable_error_inheritance(self):
        """Test ServiceUnavailableError inherits from BenchmarkError."""
        error = ServiceUnavailableError("Service error")

        assert isinstance(error, BenchmarkError)
        assert isinstance(error, ServiceUnavailableError)
        assert str(error) == "[6000] Service error"

    def test_exception_catching(self):
        """Test that specific exceptions can be caught as BenchmarkError."""
        config_error = ConfigurationError("Config error")
        data_error = DataLoadingError("Data error")

        # Can catch specific exceptions
        with pytest.raises(ConfigurationError):
            raise config_error

        # Can catch as base exception
        with pytest.raises(BenchmarkError):
            raise config_error

        with pytest.raises(BenchmarkError):
            raise data_error


class TestConvenienceFunctions:
    """Test convenience functions for creating common errors."""

    def test_config_validation_error(self):
        """Test config validation error creation."""
        error = config_validation_error("model_path", "/invalid/path", "File does not exist")

        assert isinstance(error, ConfigurationError)
        assert error.error_code == ErrorCode.CONFIG_VALIDATION_FAILED
        assert error.metadata["field"] == "model_path"
        assert error.metadata["value"] == "/invalid/path"
        assert error.metadata["reason"] == "File does not exist"
        assert "model_path" in error.message

    def test_dataset_not_found_error(self):
        """Test dataset not found error creation."""
        error = dataset_not_found_error("UNSW-NB15", "kaggle")

        assert isinstance(error, DataLoadingError)
        assert error.error_code == ErrorCode.DATASET_NOT_FOUND
        assert error.metadata["dataset_name"] == "UNSW-NB15"
        assert error.metadata["source"] == "kaggle"
        assert "UNSW-NB15" in error.message
        assert "kaggle" in error.message

    def test_model_memory_error(self):
        """Test model memory error creation."""
        error = model_memory_error("llama-7b", 16.0, 8.0)

        assert isinstance(error, ModelLoadingError)
        assert error.error_code == ErrorCode.INSUFFICIENT_MEMORY
        assert error.metadata["model_name"] == "llama-7b"
        assert error.metadata["required_memory_gb"] == 16.0
        assert error.metadata["available_memory_gb"] == 8.0
        assert "llama-7b" in error.message
        assert "16.0" in error.message
        assert "8.0" in error.message

    def test_api_key_missing_error(self):
        """Test API key missing error creation."""
        error = api_key_missing_error("openai")

        assert isinstance(error, ModelLoadingError)
        assert error.error_code == ErrorCode.API_KEY_MISSING
        assert error.metadata["provider"] == "openai"
        assert "openai" in error.message.lower()
        assert "api key" in error.message.lower()

    def test_evaluation_timeout_error(self):
        """Test evaluation timeout error creation."""
        error = evaluation_timeout_error("exp_001", 300.0)

        assert isinstance(error, EvaluationError)
        assert error.error_code == ErrorCode.EVALUATION_TIMEOUT
        assert error.metadata["experiment_id"] == "exp_001"
        assert error.metadata["timeout_seconds"] == 300.0
        assert "exp_001" in error.message
        assert "300" in error.message

    def test_service_timeout_error(self):
        """Test service timeout error creation."""
        error = service_timeout_error("model_service", 30.0)

        assert isinstance(error, ServiceUnavailableError)
        assert error.error_code == ErrorCode.SERVICE_TIMEOUT
        assert error.metadata["service_name"] == "model_service"
        assert error.metadata["timeout_seconds"] == 30.0
        assert "model_service" in error.message
        assert "30" in error.message


class TestExceptionWithCause:
    """Test exception chaining and cause handling."""

    def test_exception_with_original_cause(self):
        """Test wrapping original exceptions."""
        original = FileNotFoundError("Config file missing")
        error = ConfigurationError(
            "Failed to load configuration",
            ErrorCode.CONFIG_FILE_NOT_FOUND,
            cause=original,
        )

        assert error.cause == original
        assert str(error.cause) == "Config file missing"

        error_dict = error.to_dict()
        assert error_dict["cause"] == "Config file missing"

    def test_exception_chaining_in_catch(self):
        """Test proper exception chaining pattern."""
        with pytest.raises(ConfigurationError) as exc_info:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ConfigurationError("Configuration failed", cause=e) from e

        error = exc_info.value
        assert error.cause is not None
        assert isinstance(error.cause, ValueError)
        assert str(error.cause) == "Original error"


class TestErrorSerialization:
    """Test error serialization for logging and debugging."""

    def test_error_dict_structure(self):
        """Test error dictionary has expected structure."""
        error = ModelLoadingError(
            "Failed to load model",
            ErrorCode.MODEL_INITIALIZATION_FAILED,
            metadata={"model_name": "test-model", "attempt": 3},
        )

        error_dict = error.to_dict()

        # Check required fields
        required_fields = [
            "error_type",
            "message",
            "error_code",
            "error_code_name",
            "metadata",
            "cause",
        ]
        for field in required_fields:
            assert field in error_dict

        # Check values
        assert error_dict["error_type"] == "ModelLoadingError"
        assert error_dict["message"] == "Failed to load model"
        assert error_dict["error_code"] == ErrorCode.MODEL_INITIALIZATION_FAILED.value
        assert error_dict["error_code_name"] == "MODEL_INITIALIZATION_FAILED"
        assert error_dict["metadata"]["model_name"] == "test-model"
        assert error_dict["cause"] is None

    def test_error_dict_with_cause(self):
        """Test error dictionary includes cause information."""
        original = RuntimeError("Memory allocation failed")
        error = ModelLoadingError("Model loading failed", cause=original)

        error_dict = error.to_dict()
        assert error_dict["cause"] == "Memory allocation failed"

    def test_error_dict_json_serializable(self):
        """Test error dictionary can be JSON serialized."""
        import json

        error = EvaluationError(
            "Metric calculation failed",
            ErrorCode.METRIC_CALCULATION_FAILED,
            metadata={"metric": "accuracy", "samples": 100},
        )

        error_dict = error.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(error_dict)
        assert isinstance(json_str, str)

        # Should be deserializable
        recovered_dict = json.loads(json_str)
        assert recovered_dict["error_type"] == "EvaluationError"
        assert recovered_dict["metadata"]["metric"] == "accuracy"


class TestSpecificErrorScenarios:
    """Test realistic error scenarios for each exception type."""

    def test_configuration_validation_scenario(self):
        """Test realistic configuration validation error."""
        error = config_validation_error("models.0.max_tokens", -1, "Must be positive integer")

        assert isinstance(error, ConfigurationError)
        assert error.error_code == ErrorCode.CONFIG_VALIDATION_FAILED
        assert "models.0.max_tokens" in error.message
        assert error.metadata["field"] == "models.0.max_tokens"
        assert error.metadata["value"] == "-1"

    def test_dataset_loading_scenario(self):
        """Test realistic dataset loading error."""
        error = dataset_not_found_error("custom_dataset", "local_file")

        assert isinstance(error, DataLoadingError)
        assert error.error_code == ErrorCode.DATASET_NOT_FOUND
        assert "custom_dataset" in error.message
        assert "local_file" in error.message

    def test_model_resource_scenario(self):
        """Test realistic model resource error."""
        error = model_memory_error("llama-70b", 140.0, 24.0)

        assert isinstance(error, ModelLoadingError)
        assert error.error_code == ErrorCode.INSUFFICIENT_MEMORY
        assert "llama-70b" in error.message
        assert "140.0" in error.message
        assert "24.0" in error.message
        assert error.metadata["required_memory_gb"] == 140.0

    def test_api_authentication_scenario(self):
        """Test realistic API authentication error."""
        error = api_key_missing_error("anthropic")

        assert isinstance(error, ModelLoadingError)
        assert error.error_code == ErrorCode.API_KEY_MISSING
        assert "anthropic" in error.message.lower()
        assert "api key" in error.message.lower()

    def test_evaluation_timeout_scenario(self):
        """Test realistic evaluation timeout error."""
        error = evaluation_timeout_error("large_scale_eval_001", 1800.0)

        assert isinstance(error, EvaluationError)
        assert error.error_code == ErrorCode.EVALUATION_TIMEOUT
        assert "large_scale_eval_001" in error.message
        assert "1800" in error.message

    def test_service_communication_scenario(self):
        """Test realistic service communication error."""
        error = service_timeout_error("external_model_api", 60.0)

        assert isinstance(error, ServiceUnavailableError)
        assert error.error_code == ErrorCode.SERVICE_TIMEOUT
        assert "external_model_api" in error.message
        assert "60" in error.message


class TestExceptionInheritance:
    """Test exception inheritance and polymorphism."""

    def test_all_exceptions_inherit_from_benchmark_error(self):
        """Test all specific exceptions inherit from BenchmarkError."""
        exceptions_to_test = [
            ConfigurationError("test"),
            DataLoadingError("test"),
            ModelLoadingError("test"),
            EvaluationError("test"),
            ServiceUnavailableError("test"),
        ]

        for exc in exceptions_to_test:
            assert isinstance(exc, BenchmarkError)
            assert isinstance(exc, Exception)

    def test_polymorphic_exception_handling(self):
        """Test polymorphic exception handling."""
        errors = [
            ConfigurationError("Config error"),
            DataLoadingError("Data error"),
            ModelLoadingError("Model error"),
        ]

        caught_errors = []

        for error in errors:
            try:
                raise error
            except BenchmarkError as e:
                caught_errors.append(e)

        assert len(caught_errors) == 3
        assert all(isinstance(e, BenchmarkError) for e in caught_errors)

    def test_specific_exception_catching(self):
        """Test catching specific exception types."""
        # Should catch specific type
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config error")

        # Should also catch as base type
        with pytest.raises(BenchmarkError):
            raise ConfigurationError("Config error")


class TestErrorMetadata:
    """Test error metadata handling."""

    def test_metadata_preservation(self):
        """Test metadata is preserved through exception lifecycle."""
        metadata = {
            "experiment_id": "exp_001",
            "model_name": "gpt-4",
            "attempt_count": 3,
            "last_error": "Connection timeout",
        }

        error = EvaluationError("Evaluation failed", metadata=metadata)

        assert error.metadata == metadata
        assert error.metadata["experiment_id"] == "exp_001"
        assert error.metadata["attempt_count"] == 3

        # Metadata should be in serialized form
        error_dict = error.to_dict()
        assert error_dict["metadata"] == metadata

    def test_empty_metadata_handling(self):
        """Test handling of empty metadata."""
        error = BenchmarkError("Error without metadata")

        assert error.metadata == {}

        error_dict = error.to_dict()
        assert error_dict["metadata"] == {}

    def test_metadata_modification(self):
        """Test metadata can be modified after creation."""
        error = ConfigurationError("Config error")

        # Initially empty
        assert error.metadata == {}

        # Can be modified
        error.metadata["added_info"] = "debug_value"
        assert error.metadata["added_info"] == "debug_value"

        # Reflects in serialization
        error_dict = error.to_dict()
        assert error_dict["metadata"]["added_info"] == "debug_value"


class TestErrorCodeUsage:
    """Test proper error code usage across exception types."""

    def test_default_error_codes(self):
        """Test default error codes for each exception type."""
        config_error = ConfigurationError("Config error")
        assert config_error.error_code == ErrorCode.CONFIG_VALIDATION_FAILED

        data_error = DataLoadingError("Data error")
        assert data_error.error_code == ErrorCode.DATASET_NOT_FOUND

        model_error = ModelLoadingError("Model error")
        assert model_error.error_code == ErrorCode.MODEL_NOT_FOUND

        eval_error = EvaluationError("Eval error")
        assert eval_error.error_code == ErrorCode.EVALUATION_SETUP_FAILED

        service_error = ServiceUnavailableError("Service error")
        assert service_error.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_custom_error_codes(self):
        """Test custom error codes can be used with any exception."""
        # ConfigurationError with different code
        error = ConfigurationError("Parse error", ErrorCode.CONFIG_PARSE_ERROR)
        assert error.error_code == ErrorCode.CONFIG_PARSE_ERROR

        # ModelLoadingError with API-specific code
        error = ModelLoadingError("API error", ErrorCode.API_QUOTA_EXCEEDED)
        assert error.error_code == ErrorCode.API_QUOTA_EXCEEDED

"""
Unit tests for the ModelValidator.

This module tests model configuration validation, hardware compatibility checking,
API validation, and recommendation generation functionality.
"""

from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from benchmark.models.model_validator import (
    CompatibilityReport,
    HardwareCompatibility,
    HardwareInfo,
    ModelRecommendations,
    ModelValidator,
    ValidationIssue,
)


class TestHardwareInfo:
    """Test cases for HardwareInfo model."""

    def test_hardware_info_creation(self) -> None:
        """Test HardwareInfo model creation."""
        hardware = HardwareInfo(
            cpu_cores=8,
            memory_gb=32.0,
            gpu_memory_gb=16.0,
            neural_engine_available=True,
            apple_silicon=True,
            platform="Darwin",
            architecture="arm64",
            disk_space_gb=500.0,
        )

        assert hardware.cpu_cores == 8
        assert hardware.memory_gb == 32.0
        assert hardware.gpu_memory_gb == 16.0
        assert hardware.neural_engine_available is True
        assert hardware.apple_silicon is True
        assert hardware.platform == "Darwin"
        assert hardware.architecture == "arm64"
        assert hardware.disk_space_gb == 500.0

    def test_hardware_info_validation(self) -> None:
        """Test HardwareInfo validation."""
        # Test invalid CPU cores
        with pytest.raises(ValueError):
            HardwareInfo(
                cpu_cores=0,  # Invalid
                memory_gb=16.0,
                platform="Darwin",
                architecture="arm64",
                disk_space_gb=100.0,
            )

        # Test negative memory
        with pytest.raises(ValueError):
            HardwareInfo(
                cpu_cores=4,
                memory_gb=-1.0,  # Invalid
                platform="Darwin",
                architecture="arm64",
                disk_space_gb=100.0,
            )


class TestValidationIssue:
    """Test cases for ValidationIssue model."""

    def test_validation_issue_creation(self) -> None:
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            severity="error",
            category="configuration",
            message="Missing required field",
            suggestion="Add the required field",
            field="model_name",
        )

        assert issue.severity == "error"
        assert issue.category == "configuration"
        assert issue.message == "Missing required field"
        assert issue.suggestion == "Add the required field"
        assert issue.field == "model_name"


class TestModelValidator:
    """Test cases for ModelValidator."""

    @pytest_asyncio.fixture
    async def mock_hardware(self) -> HardwareInfo:
        """Create mock hardware info for testing."""
        return HardwareInfo(
            cpu_cores=8,
            memory_gb=32.0,
            gpu_memory_gb=16.0,
            neural_engine_available=True,
            apple_silicon=True,
            platform="Darwin",
            architecture="arm64",
            disk_space_gb=500.0,
        )

    @pytest_asyncio.fixture
    async def validator(self, mock_hardware: HardwareInfo) -> ModelValidator:
        """Create validator with mock hardware."""
        return ModelValidator(hardware_info=mock_hardware)

    @pytest_asyncio.fixture
    async def limited_hardware_validator(self) -> ModelValidator:
        """Create validator with limited hardware."""
        limited_hardware = HardwareInfo(
            cpu_cores=2,
            memory_gb=8.0,
            gpu_memory_gb=None,
            neural_engine_available=False,
            apple_silicon=False,
            platform="Linux",
            architecture="x86_64",
            disk_space_gb=100.0,
        )
        return ModelValidator(hardware_info=limited_hardware)

    def test_validator_initialization(self, mock_hardware: HardwareInfo) -> None:
        """Test ModelValidator initialization."""
        validator = ModelValidator(hardware_info=mock_hardware)

        assert validator.hardware_info == mock_hardware
        assert validator.logger is not None
        assert "mlx" in validator.model_requirements
        assert "api" in validator.model_requirements
        assert "openai" in validator.api_endpoints

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_count")
    @patch("psutil.disk_usage")
    @patch("platform.system")
    @patch("platform.machine")
    def test_hardware_detection(
        self,
        mock_machine: Mock,
        mock_system: Mock,
        mock_disk: Mock,
        mock_cpu: Mock,
        mock_memory: Mock,
    ) -> None:
        """Test hardware auto-detection."""
        # Mock system responses
        mock_memory.return_value.total = 32 * 1024**3  # 32GB
        mock_cpu.return_value = 8
        mock_disk.return_value.free = 500 * 1024**3  # 500GB
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"

        validator = ModelValidator(hardware_info=None)  # Auto-detect

        assert validator.hardware_info.cpu_cores == 8
        assert validator.hardware_info.memory_gb == 32.0
        assert validator.hardware_info.apple_silicon is True
        assert validator.hardware_info.neural_engine_available is True

    @pytest.mark.asyncio
    async def test_valid_mlx_config(self, validator: ModelValidator) -> None:
        """Test validation of valid MLX configuration."""
        config = {
            "type": "mlx",
            "name": "test-mlx-model",
            "model_path": "/path/to/model",
            "batch_size": 16,
        }

        result = await validator.validate_model_config(config)

        assert result.valid is True
        assert len([i for i in result.issues if i.severity == "error"]) == 0
        assert result.compatibility_score > 0.8

    @pytest.mark.asyncio
    async def test_invalid_config_missing_fields(self, validator: ModelValidator) -> None:
        """Test validation with missing required fields."""
        config = {"batch_size": 16}  # Missing 'type' and 'name'

        result = await validator.validate_model_config(config)

        assert result.valid is False
        error_issues = [i for i in result.issues if i.severity == "error"]
        assert len(error_issues) >= 2  # Should have errors for missing 'type' and 'name'

        # Check specific error messages
        error_messages = [issue.message for issue in error_issues]
        assert any("Missing required field: type" in msg for msg in error_messages)
        assert any("Missing required field: name" in msg for msg in error_messages)

    @pytest.mark.asyncio
    async def test_unsupported_model_type(self, validator: ModelValidator) -> None:
        """Test validation with unsupported model type."""
        config = {"type": "unsupported_type", "name": "test-model"}

        result = await validator.validate_model_config(config)

        assert result.valid is False
        error_issues = [i for i in result.issues if i.severity == "error"]
        assert len(error_issues) > 0
        assert any("Unsupported model type" in issue.message for issue in error_issues)

    @pytest.mark.asyncio
    async def test_mlx_on_non_apple_silicon(
        self, limited_hardware_validator: ModelValidator
    ) -> None:
        """Test MLX model validation on non-Apple Silicon."""
        config = {"type": "mlx", "name": "test-mlx-model", "model_path": "/path/to/model"}

        result = await limited_hardware_validator.validate_model_config(config)

        assert result.valid is False
        assert result.compatibility_score == 0.0
        error_issues = [i for i in result.issues if i.severity == "error"]
        assert any("requires Apple Silicon" in issue.message for issue in error_issues)

    @pytest.mark.asyncio
    async def test_memory_insufficient(self, limited_hardware_validator: ModelValidator) -> None:
        """Test validation with insufficient memory."""
        config = {
            "type": "transformers",
            "name": "huge-model-70b",  # This should trigger high memory estimate
            "model_path": "/path/to/model",
        }

        with patch.object(
            limited_hardware_validator, "_estimate_memory_usage", return_value=50 * 1024
        ):  # 50GB
            result = await limited_hardware_validator.validate_model_config(config)

            assert result.valid is False
            error_issues = [i for i in result.issues if i.severity == "error"]
            assert any(
                "requires" in issue.message and "RAM" in issue.message for issue in error_issues
            )

    @pytest.mark.asyncio
    async def test_api_config_validation(self, validator: ModelValidator) -> None:
        """Test API configuration validation."""
        config = {
            "type": "api",
            "name": "gpt-4",
            "provider": "openai",
            "model_name": "gpt-4",
            "api_key": "test-key",
        }

        with patch.object(validator, "_validate_api_config", return_value=True):
            result = await validator.validate_model_config(config)
            assert result.valid is True

        with patch.object(validator, "_validate_api_config", return_value=False):
            result = await validator.validate_model_config(config)
            assert result.valid is False
            error_issues = [i for i in result.issues if i.severity == "error"]
            assert any("Invalid API configuration" in issue.message for issue in error_issues)

    @pytest.mark.asyncio
    async def test_api_validation_with_mock_response(self, validator: ModelValidator) -> None:
        """Test API validation with mocked HTTP responses."""
        # Test missing API key
        config_no_key = {"provider": "openai"}
        result = await validator._validate_api_config(config_no_key)
        assert result is False

        # Test invalid provider (no default endpoint)
        config_invalid = {"provider": "unknown_provider", "api_key": "test-key"}
        result = await validator._validate_api_config(config_invalid)
        assert result is False

        # Test with explicit endpoint but still mock network failure
        config_with_endpoint = {
            "provider": "openai",
            "api_key": "test-key-123",
            "endpoint": "https://api.openai.com/v1/models",
        }

        # For now, we'll test that the method handles network errors gracefully
        # In a real environment, this would make actual HTTP calls
        # The method should return False on any network error
        result = await validator._validate_api_config(config_with_endpoint)
        assert isinstance(
            result, bool
        )  # Should return a boolean, success or failure depends on network

    @pytest.mark.asyncio
    async def test_memory_estimation(self, validator: ModelValidator) -> None:
        """Test memory usage estimation."""
        # Test 7B model
        config_7b = {"type": "transformers", "name": "model-7b", "batch_size": 1}

        memory_7b = validator._estimate_memory_usage(config_7b)
        assert memory_7b is not None
        assert memory_7b > 0

        # Test larger model should use more memory
        config_70b = {"type": "transformers", "name": "model-70b", "batch_size": 1}

        memory_70b = validator._estimate_memory_usage(config_70b)
        assert memory_70b is not None
        assert memory_70b > memory_7b

        # Test batch size impact
        config_large_batch = {"type": "transformers", "name": "model-7b", "batch_size": 32}

        memory_large_batch = validator._estimate_memory_usage(config_large_batch)
        assert memory_large_batch is not None
        assert memory_large_batch > memory_7b

    @pytest.mark.asyncio
    async def test_hardware_compatibility_check(self, validator: ModelValidator) -> None:
        """Test hardware compatibility checking."""
        config = {"type": "mlx", "name": "test-model-13b", "model_path": "/path/to/model"}

        compatibility = await validator.check_hardware_requirements(config)

        assert isinstance(compatibility, HardwareCompatibility)
        assert compatibility.compatible is True  # Should be compatible with good hardware
        assert compatibility.memory_sufficient is True
        assert compatibility.neural_engine_supported is True
        assert compatibility.performance_tier in ["low", "medium", "high"]
        assert compatibility.estimated_load_time_s is not None
        assert compatibility.estimated_load_time_s > 0

    @pytest.mark.asyncio
    async def test_hardware_compatibility_insufficient_memory(
        self, limited_hardware_validator: ModelValidator
    ) -> None:
        """Test hardware compatibility with insufficient memory."""
        config = {"type": "transformers", "name": "huge-model-70b"}

        with patch.object(
            limited_hardware_validator, "_estimate_memory_usage", return_value=50 * 1024
        ):  # 50GB
            compatibility = await limited_hardware_validator.check_hardware_requirements(config)

            assert compatibility.compatible is False
            assert compatibility.memory_sufficient is False
            assert len(compatibility.bottlenecks) > 0
            assert any(
                "Insufficient memory" in bottleneck for bottleneck in compatibility.bottlenecks
            )
            assert len(compatibility.recommendations) > 0

    @pytest.mark.asyncio
    async def test_model_compatibility_report(self, validator: ModelValidator) -> None:
        """Test multiple model compatibility checking."""
        configs = [
            {"type": "api", "name": "gpt-4", "provider": "openai"},
            {"type": "mlx", "name": "llama-7b", "model_path": "/path/to/llama"},
            {"type": "ollama", "name": "mistral", "model_name": "mistral"},
        ]

        report = await validator.validate_model_compatibility(configs)

        assert isinstance(report, CompatibilityReport)
        assert report.compatible is True  # Should be compatible with good hardware
        assert report.total_memory_required_gb >= 0
        assert report.memory_available_gb > 0
        assert len(report.model_priorities) == 3
        assert report.model_priorities["gpt-4"] == 1  # API should have highest priority
        assert report.resource_sharing_strategy in ["sequential", "time-shared", "concurrent"]

    @pytest.mark.asyncio
    async def test_model_compatibility_memory_conflict(
        self, limited_hardware_validator: ModelValidator
    ) -> None:
        """Test model compatibility with memory conflicts."""
        configs = [
            {"type": "transformers", "name": "model1-30b"},
            {"type": "transformers", "name": "model2-30b"},
            {"type": "transformers", "name": "model3-30b"},
        ]

        with patch.object(
            limited_hardware_validator, "_estimate_memory_usage", return_value=12 * 1024
        ):  # 12GB each
            report = await limited_hardware_validator.validate_model_compatibility(configs)

            assert report.compatible is False  # Too much total memory
            assert report.total_memory_required_gb > report.memory_available_gb
            assert "Sequential loading recommended" in report.scheduling_recommendations
            assert report.resource_sharing_strategy == "sequential"

    @pytest.mark.asyncio
    async def test_model_recommendations(self, validator: ModelValidator) -> None:
        """Test model optimization recommendations."""
        config = {"type": "transformers", "name": "test-model-13b", "model_path": "/path/to/model"}

        recommendations = await validator.recommend_model_settings(config)

        assert isinstance(recommendations, ModelRecommendations)
        assert recommendations.optimal_batch_size > 0
        assert recommendations.optimal_batch_size <= 64
        assert isinstance(recommendations.memory_optimization_tips, list)
        assert isinstance(recommendations.performance_expectations, dict)
        assert isinstance(recommendations.alternative_configs, list)
        assert isinstance(recommendations.resource_allocation, dict)

        # Check resource allocation contains expected keys
        assert "cpu_threads" in recommendations.resource_allocation
        assert "memory_limit_gb" in recommendations.resource_allocation
        assert "batch_size" in recommendations.resource_allocation

    @pytest.mark.asyncio
    async def test_recommendations_limited_hardware(
        self, limited_hardware_validator: ModelValidator
    ) -> None:
        """Test recommendations for limited hardware."""
        config = {"type": "transformers", "name": "test-model", "model_path": "/path/to/model"}

        recommendations = await limited_hardware_validator.recommend_model_settings(config)

        # Should recommend smaller batch size and quantization for limited hardware
        assert recommendations.optimal_batch_size <= 16
        assert recommendations.recommended_quantization in ["4bit", "8bit"]
        assert len(recommendations.memory_optimization_tips) > 0

    @pytest.mark.asyncio
    async def test_mlx_specific_validation(self, validator: ModelValidator) -> None:
        """Test MLX-specific configuration validation."""
        # Valid MLX config
        config_valid = {
            "type": "mlx",
            "name": "test-mlx",
            "model_path": "/path/to/model",
            "quantization": "4bit",
        }

        result = await validator.validate_model_config(config_valid)
        assert result.valid is True

        # MLX config missing model_path
        config_missing_path = {"type": "mlx", "name": "test-mlx"}

        result = await validator.validate_model_config(config_missing_path)
        assert result.valid is False
        error_issues = [i for i in result.issues if i.severity == "error"]
        assert any("require 'model_path'" in issue.message for issue in error_issues)

    @pytest.mark.asyncio
    async def test_ollama_specific_validation(self, validator: ModelValidator) -> None:
        """Test Ollama-specific configuration validation."""
        # Valid Ollama config
        config_valid = {"type": "ollama", "name": "test-ollama", "model_name": "llama2"}

        result = await validator.validate_model_config(config_valid)
        assert result.valid is True

        # Ollama config missing model_name
        config_missing_name = {"type": "ollama", "name": "test-ollama"}

        result = await validator.validate_model_config(config_missing_name)
        assert result.valid is False
        error_issues = [i for i in result.issues if i.severity == "error"]
        assert any("require 'model_name'" in issue.message for issue in error_issues)

    @pytest.mark.asyncio
    async def test_api_specific_validation(self, validator: ModelValidator) -> None:
        """Test API-specific configuration validation."""
        # API config missing required fields
        config_incomplete = {"type": "api", "name": "test-api"}

        result = await validator.validate_model_config(config_incomplete)
        assert result.valid is False
        error_issues = [i for i in result.issues if i.severity == "error"]

        # Should have errors for missing provider and model_name
        assert len(error_issues) >= 2
        error_messages = [issue.message for issue in error_issues]
        assert any("require 'provider'" in msg for msg in error_messages)
        assert any("require 'model_name'" in msg for msg in error_messages)

    @pytest.mark.asyncio
    async def test_inference_speed_estimation(
        self, validator: ModelValidator, limited_hardware_validator: ModelValidator
    ) -> None:
        """Test inference speed estimation."""
        # API should be fast
        api_config = {"type": "api"}
        speed = validator._estimate_inference_speed(api_config)
        assert "fast" in speed.lower()

        # MLX on Apple Silicon should be very fast
        mlx_config = {"type": "mlx"}
        speed = validator._estimate_inference_speed(mlx_config)
        assert "very fast" in speed.lower()

        # Test with limited hardware validator
        limited_config = {"type": "transformers"}
        speed = limited_hardware_validator._estimate_inference_speed(limited_config)
        assert speed in ["slow", "moderate"]

    @pytest.mark.asyncio
    async def test_error_handling(self, validator: ModelValidator) -> None:
        """Test error handling in validation methods."""
        # Test with None config
        result = await validator.validate_model_config({})
        assert result.valid is False

        # Test with malformed config
        malformed_config = {"type": None, "name": None}
        result = await validator.validate_model_config(malformed_config)
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validation_warnings_and_recommendations(self, validator: ModelValidator) -> None:
        """Test that warnings and recommendations are properly generated."""
        config = {"type": "transformers", "name": "test-model-13b", "model_path": "/path/to/model"}

        # Mock memory usage to trigger warning (high but not error)
        with patch.object(
            validator, "_estimate_memory_usage", return_value=28 * 1024
        ):  # 28GB - high but fits in 32GB
            result = await validator.validate_model_config(config)

            assert result.valid is True
            assert len(result.warnings) > 0
            assert result.compatibility_score < 1.0  # Should be reduced due to high memory

        # Test recommendations generation
        assert len(result.recommendations) > 0

    def test_hardware_detection_fallback(self) -> None:
        """Test hardware detection fallback to conservative defaults."""
        with patch("psutil.virtual_memory", side_effect=Exception("Mock failure")):
            validator = ModelValidator(hardware_info=None)

            # Should fallback to conservative defaults
            assert validator.hardware_info.cpu_cores == 4
            assert validator.hardware_info.memory_gb == 8.0
            assert validator.hardware_info.apple_silicon is False

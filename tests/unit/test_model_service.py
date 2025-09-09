"""
Unit tests for the ModelService.

This module tests the model service functionality including plugin registration,
model loading, batch prediction, resource management, and error handling.
"""

from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio

from benchmark.core.base import ServiceResponse, ServiceStatus
from benchmark.core.exceptions import BenchmarkError, ConfigurationError
from benchmark.interfaces.model_interfaces import (
    ModelInfo,
    ModelPlugin,
    PerformanceMetrics,
    Prediction,
)
from benchmark.services.model_service import ModelService


class MockModelPlugin(ModelPlugin):
    """Mock model plugin for testing."""

    def __init__(self):
        self.model_id = None
        self.config = {}
        self.initialized = False
        self.prediction_count = 0
        self.should_fail = False
        self.memory_usage = 100.0

    async def initialize(self, config: dict[str, Any]) -> ServiceResponse:
        """Initialize the mock plugin."""
        if self.should_fail:
            return ServiceResponse(success=False, message="Mock initialization failure")

        self.config = config
        self.model_id = config.get("model_id", "mock_model")
        self.initialized = True

        return ServiceResponse(
            success=True, message="Mock plugin initialized", data={"model_id": self.model_id}
        )

    async def predict(self, samples: list[str]) -> list[Prediction]:
        """Mock prediction method."""
        if self.should_fail:
            raise RuntimeError("Mock prediction failure")

        predictions = []
        for i, sample in enumerate(samples):
            # Simple mock logic: predict ATTACK if sample contains certain keywords
            is_attack = any(word in sample.lower() for word in ["attack", "malware", "virus"])

            prediction = Prediction(
                sample_id=f"sample_{self.prediction_count}_{i}",
                input_text=sample,
                prediction="ATTACK" if is_attack else "BENIGN",
                confidence=0.85 if is_attack else 0.92,
                attack_type="malware" if is_attack else None,
                inference_time_ms=50.0,
                metadata={"mock": True},
            )
            predictions.append(prediction)

        self.prediction_count += len(samples)
        return predictions

    async def explain(self, sample: str) -> str:
        """Mock explanation method."""
        if self.should_fail:
            raise RuntimeError("Mock explanation failure")

        return f"Mock explanation for: {sample[:50]}..."

    async def get_model_info(self) -> ModelInfo:
        """Mock model info method."""
        return ModelInfo(
            model_id=self.model_id or "mock_model",
            name="Mock Model",
            type="mock",
            version="1.0.0",
            description="Mock model for testing",
            capabilities=["prediction", "explanation"],
            memory_usage_mb=self.memory_usage,
            status="loaded",
        )

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Mock performance metrics method."""
        return PerformanceMetrics(
            model_id=self.model_id or "mock_model",
            total_predictions=self.prediction_count,
            average_inference_time_ms=50.0,
            predictions_per_second=20.0,
            memory_usage_mb=self.memory_usage,
        )

    async def health_check(self) -> dict[str, Any]:
        """Mock health check method."""
        return {
            "status": "healthy" if not self.should_fail else "unhealthy",
            "initialized": self.initialized,
            "prediction_count": self.prediction_count,
        }

    async def cleanup(self) -> None:
        """Mock cleanup method."""
        self.initialized = False
        self.prediction_count = 0


class TestModelService:
    """Test cases for the ModelService."""

    @pytest_asyncio.fixture
    async def model_service(self):
        """Create a model service for testing."""
        service = ModelService(
            max_models=3,
            max_memory_mb=1024,
            cleanup_interval_seconds=0,  # Disable background cleanup for tests
            enable_performance_monitoring=True,
        )

        await service.initialize()
        yield service
        await service.shutdown()

    @pytest_asyncio.fixture
    async def mock_plugin_class(self):
        """Mock plugin class fixture."""
        return MockModelPlugin

    @pytest.mark.asyncio
    async def test_service_initialization(self, model_service):
        """Test service initialization."""
        assert model_service.status == ServiceStatus.HEALTHY
        assert model_service.is_initialized()
        assert len(model_service.plugins) == 0
        assert len(model_service.loaded_models) == 0
        assert model_service.max_models == 3
        assert model_service.max_memory_mb == 1024

    @pytest.mark.asyncio
    async def test_plugin_registration(self, model_service, mock_plugin_class):
        """Test plugin registration."""
        response = await model_service.register_plugin("mock", mock_plugin_class)

        assert response.success
        assert "mock" in model_service.plugins
        assert model_service.plugins["mock"] == mock_plugin_class

    @pytest.mark.asyncio
    async def test_plugin_registration_invalid_class(self, model_service):
        """Test plugin registration with invalid class."""

        class InvalidPlugin:
            pass

        with pytest.raises(ConfigurationError):
            await model_service.register_plugin("invalid", InvalidPlugin)

    @pytest.mark.asyncio
    async def test_model_loading(self, model_service, mock_plugin_class):
        """Test model loading."""
        # Register plugin
        await model_service.register_plugin("mock", mock_plugin_class)

        # Load model
        config = {"type": "mock", "name": "test_model", "parameters": {"param1": "value1"}}

        model_id = await model_service.load_model(config)

        assert model_id is not None
        assert model_id in model_service.loaded_models
        assert len(model_service.loaded_models) == 1

        # Verify loaded model
        loaded_model = model_service.loaded_models[model_id]
        assert loaded_model.config["type"] == "mock"
        assert loaded_model.config["name"] == "test_model"
        assert loaded_model.plugin.initialized

    @pytest.mark.asyncio
    async def test_model_loading_no_plugin(self, model_service):
        """Test model loading with no registered plugin."""
        config = {"type": "nonexistent"}

        with pytest.raises(BenchmarkError):
            await model_service.load_model(config)

    @pytest.mark.asyncio
    async def test_model_loading_no_type(self, model_service):
        """Test model loading without type specification."""
        config = {"name": "test_model"}

        with pytest.raises(BenchmarkError):
            await model_service.load_model(config)

    @pytest.mark.asyncio
    async def test_model_loading_plugin_failure(self, model_service, mock_plugin_class):
        """Test model loading with plugin initialization failure."""
        # Register plugin
        await model_service.register_plugin("mock", mock_plugin_class)

        # Create a mock that will fail initialization
        with patch.object(MockModelPlugin, "initialize") as mock_init:
            mock_init.return_value = ServiceResponse(success=False, message="Test failure")

            config = {"type": "mock", "name": "failing_model"}

            with pytest.raises(BenchmarkError):
                await model_service.load_model(config)

    @pytest.mark.asyncio
    async def test_batch_prediction(self, model_service, mock_plugin_class):
        """Test batch prediction functionality."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        # Test prediction
        samples = [
            "This is a normal message",
            "This contains an attack pattern",
            "Another benign message",
            "Malware detected in this text",
        ]

        response = await model_service.predict_batch(model_id, samples)

        assert response.model_id == model_id
        assert len(response.predictions) == 4
        assert response.total_samples == 4
        assert response.successful_predictions == 4
        assert response.failed_predictions == 0
        assert response.total_inference_time_ms > 0

        # Check predictions
        predictions = response.predictions
        assert predictions[0].prediction == "BENIGN"
        assert predictions[1].prediction == "ATTACK"
        assert predictions[2].prediction == "BENIGN"
        assert predictions[3].prediction == "ATTACK"

    @pytest.mark.asyncio
    async def test_batch_prediction_with_explanations(self, model_service, mock_plugin_class):
        """Test batch prediction with explanations."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        # Test prediction with explanations
        samples = ["Test message"]

        response = await model_service.predict_batch(model_id, samples, include_explanations=True)

        assert len(response.predictions) == 1
        prediction = response.predictions[0]
        assert prediction.explanation is not None
        assert "Mock explanation" in prediction.explanation

    @pytest.mark.asyncio
    async def test_batch_prediction_nonexistent_model(self, model_service):
        """Test batch prediction with nonexistent model."""
        samples = ["test message"]

        response = await model_service.predict_batch("nonexistent", samples)

        assert response.successful_predictions == 0
        assert response.failed_predictions == len(samples)
        assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_batch_prediction_plugin_failure(self, model_service, mock_plugin_class):
        """Test batch prediction with plugin failure."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        # Make plugin fail
        model_service.loaded_models[model_id].plugin.should_fail = True

        samples = ["test message"]
        response = await model_service.predict_batch(model_id, samples)

        assert response.successful_predictions == 0
        assert response.failed_predictions == 1
        assert len(response.predictions) == 1
        assert response.predictions[0].prediction == "ERROR"

    @pytest.mark.asyncio
    async def test_explain_prediction(self, model_service, mock_plugin_class):
        """Test single prediction explanation."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        # Test explanation
        explanation = await model_service.explain_prediction(model_id, "test sample")

        assert "Mock explanation" in explanation
        assert "test sample" in explanation

    @pytest.mark.asyncio
    async def test_explain_prediction_nonexistent_model(self, model_service):
        """Test explanation with nonexistent model."""
        explanation = await model_service.explain_prediction("nonexistent", "test sample")
        assert (
            "explanation not available" in explanation.lower() or "not found" in explanation.lower()
        )

    @pytest.mark.asyncio
    async def test_get_model_info(self, model_service, mock_plugin_class):
        """Test getting model information."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        # Test model info
        info = await model_service.get_model_info(model_id)

        assert info.model_id == model_id
        assert info.name == "Mock Model"
        assert info.type == "mock"
        assert info.version == "1.0.0"
        assert "prediction" in info.capabilities

    @pytest.mark.asyncio
    async def test_get_model_info_nonexistent_model(self, model_service):
        """Test getting info for nonexistent model."""
        with pytest.raises(BenchmarkError):
            await model_service.get_model_info("nonexistent")

    @pytest.mark.asyncio
    async def test_get_model_performance(self, model_service, mock_plugin_class):
        """Test getting model performance metrics."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        # Make some predictions to generate metrics
        samples = ["test1", "test2"]
        await model_service.predict_batch(model_id, samples)

        # Test performance metrics
        performance = await model_service.get_model_performance(model_id)

        assert "basic_metrics" in performance
        assert performance["total_batches_processed"] > 0

    @pytest.mark.asyncio
    async def test_get_all_models(self, model_service, mock_plugin_class):
        """Test getting information about all loaded models."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)

        # Load multiple models
        config1 = {"type": "mock", "name": "model1"}
        config2 = {"type": "mock", "name": "model2"}

        model_id1 = await model_service.load_model(config1)
        model_id2 = await model_service.load_model(config2)

        # Test get all models
        all_models = await model_service.get_all_models()

        assert len(all_models) == 2
        assert model_id1 in all_models
        assert model_id2 in all_models
        assert all_models[model_id1].name == "Mock Model"
        assert all_models[model_id2].name == "Mock Model"

    @pytest.mark.asyncio
    async def test_cleanup_model(self, model_service, mock_plugin_class):
        """Test model cleanup."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        assert len(model_service.loaded_models) == 1

        # Test cleanup
        response = await model_service.cleanup_model(model_id)

        assert response.success
        assert len(model_service.loaded_models) == 0

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_model(self, model_service):
        """Test cleanup of nonexistent model."""
        response = await model_service.cleanup_model("nonexistent")

        assert response.success  # Should succeed with warning message

    @pytest.mark.asyncio
    async def test_max_models_limit(self, model_service, mock_plugin_class):
        """Test maximum models limit enforcement."""
        # Setup
        await model_service.register_plugin("mock", mock_plugin_class)

        # Load models up to the limit
        model_ids = []
        for i in range(model_service.max_models):
            config = {"type": "mock", "name": f"model_{i}"}
            model_id = await model_service.load_model(config)
            model_ids.append(model_id)

        assert len(model_service.loaded_models) == model_service.max_models

        # Load one more model (should trigger cleanup)
        config = {"type": "mock", "name": "extra_model"}
        extra_model_id = await model_service.load_model(config)

        # Should still be at the limit (one old model cleaned up)
        assert len(model_service.loaded_models) == model_service.max_models
        assert extra_model_id in model_service.loaded_models

    @pytest.mark.asyncio
    async def test_health_check(self, model_service, mock_plugin_class):
        """Test service health check."""
        # Test initial health
        health = await model_service.health_check()

        assert health.status in [ServiceStatus.HEALTHY.value, ServiceStatus.DEGRADED.value]
        assert "service_status" in health.checks
        assert health.checks["loaded_models"] == 0

        # Load a model and test health again
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        health = await model_service.health_check()
        assert health.checks["loaded_models"] == 1
        assert model_id in health.checks["model_health"]

    @pytest.mark.asyncio
    async def test_service_stats(self, model_service, mock_plugin_class):
        """Test service statistics."""
        # Test initial stats
        stats = await model_service.get_service_stats()

        assert stats["service_status"] == ServiceStatus.HEALTHY.value
        assert stats["loaded_models"] == 0
        assert "registered_plugins" in stats
        assert "current_memory_mb" in stats

        # Load a model and test stats again
        await model_service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        model_id = await model_service.load_model(config)

        stats = await model_service.get_service_stats()
        assert stats["loaded_models"] == 1
        assert "mock" in stats["registered_plugins"]
        assert model_id in stats["model_statistics"]

    @pytest.mark.asyncio
    async def test_service_shutdown(self, mock_plugin_class):
        """Test service shutdown."""
        # Create service
        service = ModelService(cleanup_interval_seconds=0)
        await service.initialize()

        # Load a model
        await service.register_plugin("mock", mock_plugin_class)
        config = {"type": "mock", "name": "test_model"}
        await service.load_model(config)

        assert len(service.loaded_models) == 1

        # Test shutdown
        response = await service.shutdown()

        assert response.success
        assert service.status == ServiceStatus.STOPPED
        assert len(service.loaded_models) == 0


class TestModelInterfaces:
    """Test cases for model interfaces and data models."""

    def test_prediction_model(self):
        """Test Prediction model validation."""
        prediction = Prediction(
            sample_id="test_001",
            input_text="Test sample",
            prediction="ATTACK",
            confidence=0.95,
            attack_type="malware",
            explanation="Test explanation",
            inference_time_ms=45.0,
        )

        assert prediction.sample_id == "test_001"
        assert prediction.prediction == "ATTACK"
        assert prediction.confidence == 0.95
        assert prediction.attack_type == "malware"
        assert prediction.inference_time_ms == 45.0

    def test_prediction_model_validation(self):
        """Test Prediction model validation constraints."""
        # Test confidence bounds
        with pytest.raises(ValueError):
            Prediction(
                sample_id="test",
                input_text="test",
                prediction="ATTACK",
                confidence=1.5,  # Invalid confidence > 1.0
                inference_time_ms=50.0,
            )

        with pytest.raises(ValueError):
            Prediction(
                sample_id="test",
                input_text="test",
                prediction="ATTACK",
                confidence=-0.1,  # Invalid confidence < 0.0
                inference_time_ms=50.0,
            )

    def test_model_info_model(self):
        """Test ModelInfo model."""
        info = ModelInfo(
            model_id="test_model_001",
            name="Test Model",
            type="mock",
            version="1.0.0",
            description="Test model",
            capabilities=["prediction", "explanation"],
            memory_usage_mb=256.0,
        )

        assert info.model_id == "test_model_001"
        assert info.name == "Test Model"
        assert info.type == "mock"
        assert "prediction" in info.capabilities
        assert info.memory_usage_mb == 256.0

    def test_performance_metrics_model(self):
        """Test PerformanceMetrics model."""
        metrics = PerformanceMetrics(
            model_id="test_model",
            total_predictions=1000,
            total_inference_time_ms=50000.0,
            average_inference_time_ms=50.0,
            predictions_per_second=20.0,
            memory_usage_mb=512.0,
            success_rate=0.99,
        )

        assert metrics.model_id == "test_model"
        assert metrics.total_predictions == 1000
        assert metrics.average_inference_time_ms == 50.0
        assert metrics.success_rate == 0.99


if __name__ == "__main__":
    pytest.main([__file__])

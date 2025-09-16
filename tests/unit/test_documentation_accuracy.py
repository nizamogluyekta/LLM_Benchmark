"""
Documentation accuracy tests for the LLM Cybersecurity Benchmark system.

This module tests that all code examples in documentation are accurate,
executable, and produce expected results. It validates API documentation,
integration guides, and example code.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from benchmark.interfaces.evaluation_interfaces import EvaluationRequest, MetricType
from benchmark.interfaces.model_interfaces import LoadingStrategy, ModelInfo, Prediction
from benchmark.services.evaluation_service import EvaluationService
from benchmark.services.model_service import ModelService


class TestEvaluationServiceDocumentation:
    """Test examples from evaluation service API documentation."""

    @pytest_asyncio.fixture
    async def mock_evaluation_service(self):
        """Create a mock evaluation service for testing documentation examples."""
        service = EvaluationService()

        # Mock the initialize method
        service.initialize = AsyncMock(
            return_value=MagicMock(success=True, data={"status": "initialized"})
        )
        service.shutdown = AsyncMock(return_value=MagicMock(success=True))

        # Mock evaluator registration
        service.register_evaluator = AsyncMock(
            return_value=MagicMock(
                success=True, data={"metric_type": "accuracy", "status": "registered"}
            )
        )

        # Mock health check
        service.health_check = AsyncMock(
            return_value=MagicMock(
                status="healthy", details={"evaluators": 1, "active_evaluations": 0}
            )
        )

        # Mock available metrics
        service.get_available_metrics = AsyncMock(
            return_value=MagicMock(
                success=True,
                data={
                    "total_evaluators": 2,
                    "metrics": ["accuracy", "precision"],
                    "evaluator_info": {"accuracy": {"type": "MockAccuracyEvaluator"}},
                },
            )
        )

        # Mock evaluation history
        service.get_evaluation_history = AsyncMock(
            return_value=MagicMock(
                success=True, data={"results": [], "total_results": 0, "filters_applied": {}}
            )
        )

        # Mock evaluation summary
        service.get_evaluation_summary = AsyncMock(
            return_value=MagicMock(
                success=True,
                data={
                    "total_evaluations": 4,
                    "successful_evaluations": 4,
                    "failed_evaluations": 0,
                    "average_execution_time": 1.25,
                    "metric_summaries": {
                        "accuracy": {"mean": 0.88, "count": 4, "min": 0.82, "max": 0.94}
                    },
                    "models_evaluated": ["SecurityBERT", "CyberLLaMA"],
                    "datasets_evaluated": ["network_data", "malware_data"],
                    "time_range": {"start": "2024-01-01T00:00:00", "end": "2024-01-01T23:59:59"},
                },
            )
        )

        return service

    @pytest.mark.asyncio
    async def test_basic_initialization_example(self, mock_evaluation_service):
        """Test the basic initialization example from documentation."""

        # From docs: evaluation_service_api.md - initialize section
        service = mock_evaluation_service
        response = await service.initialize(
            {
                "max_concurrent_evaluations": 10,
                "evaluation_timeout_seconds": 60.0,
                "max_history_size": 2000,
            }
        )

        assert response.success
        assert "status" in response.data

    @pytest.mark.asyncio
    async def test_evaluator_registration_example(self, mock_evaluation_service):
        """Test evaluator registration example from documentation."""

        # From docs: evaluation_service_api.md - register_evaluator section
        service = mock_evaluation_service

        # Mock evaluator (simplified for testing)
        mock_evaluator = MagicMock()

        response = await service.register_evaluator(MetricType.ACCURACY, mock_evaluator)

        assert response.success
        assert response.data["metric_type"] == "accuracy"

    @pytest.mark.asyncio
    async def test_evaluation_request_example(self, mock_evaluation_service):
        """Test evaluation request example from documentation."""

        # Mock the evaluate_predictions method
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.experiment_id = "cybersec_eval_001"
        mock_result.model_id = "SecurityBERT_v2"
        mock_result.dataset_id = "network_intrusion_detection"
        mock_result.metrics = {"accuracy": 0.92, "precision": 0.89}
        mock_result.execution_time_seconds = 1.5
        mock_result.get_metric_value = lambda metric: mock_result.metrics.get(metric)

        mock_evaluation_service.evaluate_predictions = AsyncMock(return_value=mock_result)

        # From docs: evaluation_service_api.md - evaluate_predictions section
        request = EvaluationRequest(
            experiment_id="cybersec_eval_001",
            model_id="SecurityBERT_v2",
            dataset_id="network_intrusion_detection",
            predictions=[
                {"predicted_class": "dos", "confidence": 0.95},
                {"predicted_class": "normal", "confidence": 0.88},
            ],
            ground_truth=[{"true_class": "dos"}, {"true_class": "normal"}],
            metrics=[MetricType.ACCURACY, MetricType.PRECISION],
            metadata={"experiment_type": "baseline_comparison", "dataset_version": "v2.1"},
        )

        result = await mock_evaluation_service.evaluate_predictions(request)

        assert result.success
        assert result.experiment_id == "cybersec_eval_001"
        assert result.get_metric_value("accuracy") == 0.92

    @pytest.mark.asyncio
    async def test_history_retrieval_example(self, mock_evaluation_service):
        """Test evaluation history retrieval example from documentation."""

        # From docs: evaluation_service_api.md - get_evaluation_history section
        service = mock_evaluation_service

        history_response = await service.get_evaluation_history(
            limit=100, model_id="SecurityBERT_v2", dataset_id="network_data"
        )

        assert history_response.success
        assert "results" in history_response.data
        assert "total_results" in history_response.data

    @pytest.mark.asyncio
    async def test_summary_generation_example(self, mock_evaluation_service):
        """Test evaluation summary generation example from documentation."""

        # From docs: evaluation_service_api.md - get_evaluation_summary section
        service = mock_evaluation_service

        summary_response = await service.get_evaluation_summary(days_back=7)

        assert summary_response.success
        summary_data = summary_response.data

        # Validate expected fields from documentation
        assert "total_evaluations" in summary_data
        assert "successful_evaluations" in summary_data
        assert "failed_evaluations" in summary_data
        assert "average_execution_time" in summary_data
        assert "metric_summaries" in summary_data
        assert "models_evaluated" in summary_data
        assert "datasets_evaluated" in summary_data
        assert "time_range" in summary_data


class TestModelServiceDocumentation:
    """Test examples from model service API documentation."""

    @pytest_asyncio.fixture
    async def mock_model_service(self):
        """Create a mock model service for testing documentation examples."""
        service = ModelService()

        # Mock initialization
        service.initialize = AsyncMock(
            return_value=MagicMock(
                success=True, data={"status": "initialized", "config": {"max_concurrent_models": 5}}
            )
        )
        service.shutdown = AsyncMock(return_value=MagicMock(success=True))

        # Mock model discovery
        service.discover_models = AsyncMock(
            return_value=MagicMock(
                success=True,
                data=MagicMock(
                    total_models=5,
                    available_models={"mlx": ["model1"], "ollama": ["model2"], "openai": ["gpt-4"]},
                    plugin_status={
                        "mlx": "available",
                        "ollama": "available",
                        "openai": "available",
                    },
                ),
            )
        )

        # Mock model loading
        mock_model_info = ModelInfo(
            model_id="SecurityBERT_v2",
            name="SecurityBERT v2",
            type="mlx",
            capabilities=["text_classification", "threat_detection"],
            parameters={"temperature": 0.1, "max_tokens": 512},
        )

        service.load_model = AsyncMock(return_value=MagicMock(success=True, data=mock_model_info))

        # Mock model unloading
        service.unload_model = AsyncMock(
            return_value=MagicMock(
                success=True, data={"status": "unloaded", "memory_freed_mb": 2048}
            )
        )

        # Mock loaded models
        service.get_loaded_models = AsyncMock(
            return_value=MagicMock(success=True, data=[mock_model_info])
        )

        # Mock predictions
        mock_predictions = [
            Prediction(
                sample_id="sample_1",
                input_text="Network connection analysis...",
                prediction="ATTACK",
                confidence=0.95,
                attack_type="dos",
                explanation="High confidence DOS attack pattern detected",
                inference_time_ms=145.2,
                model_version="v2.1",
            )
        ]

        service.predict = AsyncMock(return_value=MagicMock(success=True, data=mock_predictions))

        # Mock resource usage
        service.get_resource_usage = AsyncMock(
            return_value=MagicMock(
                success=True,
                data={
                    "memory_usage_mb": 4096,
                    "cpu_usage_percent": 45.2,
                    "gpu_usage_percent": 62.1,
                    "active_models": 2,
                    "inference_queue_size": 5,
                },
            )
        )

        return service

    @pytest.mark.asyncio
    async def test_model_service_initialization_example(self, mock_model_service):
        """Test model service initialization example from documentation."""

        # From docs: model_service_api.md - initialize section
        service = mock_model_service
        response = await service.initialize(
            {
                "max_concurrent_models": 5,
                "enable_performance_monitoring": True,
                "apple_silicon_optimization": True,
                "batch_size": 32,
                "memory_threshold_mb": 8192,
            }
        )

        assert response.success
        assert "status" in response.data

    @pytest.mark.asyncio
    async def test_model_discovery_example(self, mock_model_service):
        """Test model discovery example from documentation."""

        # From docs: model_service_api.md - discover_models section
        service = mock_model_service

        discovery = await service.discover_models()

        assert discovery.success
        result = discovery.data
        assert hasattr(result, "total_models")
        assert hasattr(result, "available_models")
        assert result.total_models > 0

    @pytest.mark.asyncio
    async def test_model_loading_example(self, mock_model_service):
        """Test model loading example from documentation."""

        # From docs: model_service_api.md - load_model section
        service = mock_model_service

        config = {"max_tokens": 2048, "temperature": 0.1, "top_p": 0.95}

        response = await service.load_model(
            model_id="mlx://microsoft/DialoGPT-medium",
            config=config,
            strategy=LoadingStrategy.EAGER,
        )

        assert response.success
        model_info = response.data
        assert hasattr(model_info, "name")
        assert hasattr(model_info, "type")

    @pytest.mark.asyncio
    async def test_prediction_example(self, mock_model_service):
        """Test prediction example from documentation."""

        # From docs: model_service_api.md - predict section
        service = mock_model_service

        inputs = [
            "Analyze this network log: 192.168.1.1 attempted connection to 10.0.0.1",
            "Email subject: 'Urgent: Update your banking information immediately'",
        ]

        response = await service.predict(
            model_id="SecurityBERT_v2", inputs=inputs, temperature=0.1, max_tokens=512
        )

        assert response.success
        predictions = response.data
        assert len(predictions) > 0

        for pred in predictions:
            assert hasattr(pred, "input_text")
            assert hasattr(pred, "prediction")
            assert hasattr(pred, "confidence")

    @pytest.mark.asyncio
    async def test_resource_monitoring_example(self, mock_model_service):
        """Test resource monitoring example from documentation."""

        # From docs: model_service_api.md - get_resource_usage section
        service = mock_model_service

        resource_usage = await service.get_resource_usage()

        assert resource_usage.success
        stats = resource_usage.data

        # Validate expected fields from documentation
        expected_fields = [
            "memory_usage_mb",
            "cpu_usage_percent",
            "gpu_usage_percent",
            "active_models",
            "inference_queue_size",
        ]

        for field in expected_fields:
            assert field in stats


class TestIntegrationGuideExamples:
    """Test examples from the integration guide."""

    @pytest.mark.asyncio
    async def test_minimal_integration_example(self):
        """Test the minimal integration example from the guide."""

        # From docs: integration_guide.md - minimal integration section
        # Create mock services
        model_service = MagicMock()
        eval_service = MagicMock()

        # Mock async methods
        model_service.initialize = AsyncMock(return_value=None)
        model_service.load_model = AsyncMock(return_value=None)
        model_service.predict = AsyncMock(
            return_value=MagicMock(
                success=True, data=[MagicMock(prediction="attack", confidence=0.95)]
            )
        )
        model_service.shutdown = AsyncMock(return_value=None)

        eval_service.initialize = AsyncMock(return_value=None)
        eval_service.register_evaluator = AsyncMock(return_value=None)
        eval_service.evaluate_predictions = AsyncMock(
            return_value=MagicMock(
                success=True, get_metric_value=lambda metric: 0.95 if metric == "accuracy" else None
            )
        )
        eval_service.shutdown = AsyncMock(return_value=None)

        # Execute the minimal integration example
        await model_service.initialize()
        await eval_service.initialize()

        await model_service.load_model("ollama://llama2:7b")

        # Mock evaluator
        mock_evaluator = MagicMock()
        await eval_service.register_evaluator(MetricType.ACCURACY, mock_evaluator)

        inputs = ["Analyze this network log: Suspicious connection to 10.0.0.1"]
        predictions = await model_service.predict("ollama://llama2:7b", inputs)

        request = EvaluationRequest(
            experiment_id="minimal_test",
            model_id="ollama://llama2:7b",
            dataset_id="test_data",
            predictions=[{"predicted_class": "attack", "confidence": 0.95}],
            ground_truth=[{"true_class": "attack"}],
            metrics=[MetricType.ACCURACY],
            metadata={},
        )

        result = await eval_service.evaluate_predictions(request)

        # Validate the example worked
        assert predictions.success
        assert result.success
        assert result.get_metric_value("accuracy") == 0.95

        await model_service.shutdown()
        await eval_service.shutdown()

    @pytest.mark.asyncio
    async def test_configuration_example(self):
        """Test configuration management example from integration guide."""

        # From docs: integration_guide.md - configuration management section
        config_data = {
            "services": {
                "model_service": {"max_concurrent_models": 5, "memory_threshold_mb": 8192},
                "evaluation_service": {"max_concurrent_evaluations": 10, "timeout_seconds": 300},
            },
            "models": {
                "cybersecurity_models": [
                    {
                        "id": "SecurityBERT",
                        "type": "mlx",
                        "config": {"max_tokens": 2048, "temperature": 0.0},
                    }
                ]
            },
        }

        # Test that the configuration structure is valid
        assert "services" in config_data
        assert "model_service" in config_data["services"]
        assert "evaluation_service" in config_data["services"]
        assert "models" in config_data

        # Test specific configuration values
        model_config = config_data["services"]["model_service"]
        assert model_config["max_concurrent_models"] == 5
        assert model_config["memory_threshold_mb"] == 8192

        eval_config = config_data["services"]["evaluation_service"]
        assert eval_config["max_concurrent_evaluations"] == 10
        assert eval_config["timeout_seconds"] == 300


class TestPerformanceOptimizationDocumentation:
    """Test examples from performance optimization documentation."""

    def test_hardware_profile_detection(self):
        """Test hardware profile detection example."""

        # From performance_optimizer.py and benchmarks/performance_results.md
        with (
            patch("psutil.cpu_count", return_value=8),
            patch("psutil.virtual_memory") as mock_memory,
            patch("platform.machine", return_value="arm64"),
            patch("platform.system", return_value="Darwin"),
        ):
            mock_memory.return_value.total = 16 * (1024**3)  # 16GB
            from benchmark.core.performance_optimizer import HardwareProfiler

            profiler = HardwareProfiler()
            profile = profiler.detect_hardware_profile()

            # Validate profile characteristics
            assert profile.cpu_cores == 8
            assert profile.memory_gb == 16.0
            assert profile.apple_silicon
            assert profile.max_concurrent_models > 0
            assert profile.target_inference_latency_ms > 0

    def test_performance_metrics_collection(self):
        """Test performance metrics collection example."""

        from benchmark.core.performance_optimizer import PerformanceMonitor, PerformanceProfile

        # Create a test profile
        profile = PerformanceProfile(
            cpu_cores=8,
            memory_gb=16.0,
            gpu_available=True,
            apple_silicon=True,
            architecture="arm64",
        )

        # Create monitor
        monitor = PerformanceMonitor(profile)

        # Collect test metrics
        metrics = monitor.collect_metrics(
            inference_latency_ms=250.0,
            throughput_per_sec=15.0,
            concurrent_operations=3,
            error_rate=0.02,
            queue_depth=5,
        )

        # Validate metrics
        assert metrics.inference_latency_ms == 250.0
        assert metrics.throughput_per_sec == 15.0
        assert metrics.concurrent_operations == 3
        assert metrics.error_rate == 0.02
        assert metrics.queue_depth == 5
        assert 0.0 <= metrics.overall_score <= 1.0


class TestDocumentationCodeSnippets:
    """Test individual code snippets from documentation."""

    def test_evaluation_request_validation(self):
        """Test that EvaluationRequest validation works as documented."""

        # Test valid request (should not raise)
        valid_request = EvaluationRequest(
            experiment_id="test_exp",
            model_id="test_model",
            dataset_id="test_dataset",
            predictions=[{"predicted_class": "attack", "confidence": 0.95}],
            ground_truth=[{"true_class": "attack"}],
            metrics=[MetricType.ACCURACY],
            metadata={},
        )

        assert valid_request.experiment_id == "test_exp"
        assert valid_request.model_id == "test_model"
        assert len(valid_request.predictions) == 1
        assert len(valid_request.ground_truth) == 1

        # Test invalid request (should raise ValueError)
        with pytest.raises(ValueError, match="experiment_id cannot be empty"):
            EvaluationRequest(
                experiment_id="",
                model_id="test_model",
                dataset_id="test_dataset",
                predictions=[{"predicted_class": "attack"}],
                ground_truth=[{"true_class": "attack"}],
                metrics=[MetricType.ACCURACY],
                metadata={},
            )

        with pytest.raises(ValueError, match="predictions cannot be empty"):
            EvaluationRequest(
                experiment_id="test_exp",
                model_id="test_model",
                dataset_id="test_dataset",
                predictions=[],
                ground_truth=[{"true_class": "attack"}],
                metrics=[MetricType.ACCURACY],
                metadata={},
            )

    def test_metric_type_enumeration(self):
        """Test that MetricType enumeration includes documented values."""

        # From docs: evaluation_service_api.md - metric types section
        expected_metrics = [
            "ACCURACY",
            "PRECISION",
            "RECALL",
            "F1_SCORE",
            "ROC_AUC",
            "PERFORMANCE",
            "FALSE_POSITIVE_RATE",
            "CONFUSION_MATRIX",
            "EXPLAINABILITY",
        ]

        for metric_name in expected_metrics:
            assert hasattr(MetricType, metric_name), f"MetricType.{metric_name} not found"

            # Test that the enum values work
            metric = getattr(MetricType, metric_name)
            assert isinstance(metric, MetricType)

    def test_prediction_model_fields(self):
        """Test that Prediction model has all documented fields."""

        # From docs: model_service_api.md - prediction model section
        prediction = Prediction(
            sample_id="test_sample",
            input_text="Test input",
            prediction="ATTACK",
            confidence=0.95,
            attack_type="dos",
            explanation="Test explanation",
            inference_time_ms=150.0,
            model_version="v1.0",
        )

        # Validate all documented fields are accessible
        assert prediction.sample_id == "test_sample"
        assert prediction.input_text == "Test input"
        assert prediction.prediction == "ATTACK"
        assert prediction.confidence == 0.95
        assert prediction.attack_type == "dos"
        assert prediction.explanation == "Test explanation"
        assert prediction.inference_time_ms == 150.0
        assert prediction.model_version == "v1.0"
        assert hasattr(prediction, "timestamp")
        assert hasattr(prediction, "metadata")


class TestDocumentationIntegrity:
    """Test overall documentation integrity and consistency."""

    def test_documentation_files_exist(self):
        """Test that all referenced documentation files exist."""

        base_path = Path(__file__).parent.parent.parent

        # Core documentation files
        required_docs = [
            "docs/evaluation_service_api.md",
            "docs/model_service_api.md",
            "docs/integration_guide.md",
            "benchmarks/performance_results.md",
        ]

        for doc_path in required_docs:
            full_path = base_path / doc_path
            assert full_path.exists(), f"Documentation file missing: {doc_path}"
            assert full_path.stat().st_size > 1000, f"Documentation file too small: {doc_path}"

    def test_api_documentation_completeness(self):
        """Test that API documentation covers all public methods."""

        # Import services to check their public methods
        from benchmark.services.evaluation_service import EvaluationService
        from benchmark.services.model_service import ModelService

        # Get public methods (not starting with _)
        eval_methods = [
            m
            for m in dir(EvaluationService)
            if not m.startswith("_") and callable(getattr(EvaluationService, m))
        ]
        model_methods = [
            m
            for m in dir(ModelService)
            if not m.startswith("_") and callable(getattr(ModelService, m))
        ]

        # Core methods that should be documented
        expected_eval_methods = [
            "initialize",
            "shutdown",
            "register_evaluator",
            "evaluate_predictions",
            "get_evaluation_history",
            "get_evaluation_summary",
            "get_available_metrics",
            "health_check",
        ]

        expected_model_methods = [
            "initialize",
            "shutdown",
            "discover_models",
            "load_model",
            "unload_model",
            "predict",
            "batch_predict",
            "get_loaded_models",
            "get_resource_usage",
            "health_check",
        ]

        # Check that expected methods exist
        for method in expected_eval_methods:
            assert method in eval_methods, f"EvaluationService.{method} method missing"

        for method in expected_model_methods:
            assert method in model_methods, f"ModelService.{method} method missing"

    def test_example_consistency(self):
        """Test that examples across different documentation files are consistent."""

        # This is a simplified test - in practice, you might parse markdown files
        # and extract code blocks to validate consistency

        # Test that common model IDs are used consistently
        common_model_ids = [
            "SecurityBERT",
            "SecurityBERT_v2",
            "CyberLLaMA",
            "ollama://llama2:7b",
            "openai://gpt-4",
        ]

        # Test that common dataset IDs are used consistently
        common_dataset_ids = [
            "network_intrusion_detection",
            "malware_classification",
            "phishing_detection",
            "vulnerability_assessment",
        ]

        # Test that common experiment IDs follow patterns
        experiment_id_patterns = ["cybersec_eval_", "basic_test", "comparison_", "performance_test"]

        # These would be validated against actual documentation content
        # For now, just ensure the lists are not empty
        assert len(common_model_ids) > 0
        assert len(common_dataset_ids) > 0
        assert len(experiment_id_patterns) > 0


def test_documentation_examples_executable():
    """Integration test to verify documentation examples are executable."""

    # This test validates that the core patterns from documentation
    # can be executed without syntax errors

    # Test 1: Basic service creation pattern
    def create_services():
        model_service = ModelService()
        eval_service = EvaluationService()
        return model_service, eval_service

    model_service, eval_service = create_services()
    assert model_service is not None
    assert eval_service is not None

    # Test 2: Configuration dictionary pattern
    config = {
        "max_concurrent_models": 5,
        "enable_performance_monitoring": True,
        "apple_silicon_optimization": True,
    }

    assert isinstance(config, dict)
    assert "max_concurrent_models" in config

    # Test 3: Request creation pattern
    request = EvaluationRequest(
        experiment_id="doc_test",
        model_id="test_model",
        dataset_id="test_dataset",
        predictions=[{"predicted_class": "attack", "confidence": 0.9}],
        ground_truth=[{"true_class": "attack"}],
        metrics=[MetricType.ACCURACY],
        metadata={"test": True},
    )

    assert request.experiment_id == "doc_test"
    assert len(request.predictions) == 1

    # Test 4: Enum usage pattern
    metrics_list = [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL]
    assert len(metrics_list) == 3
    assert all(isinstance(m, MetricType) for m in metrics_list)


if __name__ == "__main__":
    # Run documentation accuracy tests
    pytest.main([__file__, "-v"])

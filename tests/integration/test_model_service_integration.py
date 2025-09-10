"""
Integration tests for the unified model service management.

This module tests the integration of all model plugins with the model service,
unified interfaces, performance comparison, cost estimation, and resource optimization.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from benchmark.core.exceptions import BenchmarkError
from benchmark.interfaces.model_interfaces import (
    CostEstimate,
    EnhancedModelInfo,
    LoadingStrategy,
    ModelDiscoveryResult,
    PerformanceComparison,
)
from benchmark.services.model_service import ModelService


@pytest_asyncio.fixture
async def model_service():
    """Create and initialize a model service for testing."""
    service = ModelService(max_models=5, max_memory_mb=4096)

    # Initialize the service (this will register plugins)
    await service.initialize()

    yield service

    # Cleanup
    await service.shutdown()


@pytest_asyncio.fixture
async def mock_plugins():
    """Mock all plugin classes for testing."""
    with (
        patch("benchmark.models.plugins.openai_api.OpenAIModelPlugin") as mock_openai,
        patch("benchmark.models.plugins.anthropic_api.AnthropicModelPlugin") as mock_anthropic,
        patch("benchmark.models.plugins.mlx_local.MLXModelPlugin") as mock_mlx,
        patch("benchmark.models.plugins.ollama_local.OllamaModelPlugin") as mock_ollama,
    ):
        # Setup common mock behavior
        for mock_plugin_class in [mock_openai, mock_anthropic, mock_mlx, mock_ollama]:
            mock_plugin = MagicMock()
            mock_plugin.get_supported_models.return_value = ["test-model-1", "test-model-2"]
            mock_plugin.get_model_specs.return_value = {"context_window": 4096, "memory_gb": 8}
            mock_plugin.cost_tracker.pricing = {"test-model-1": {"input": 0.001, "output": 0.002}}
            mock_plugin.cleanup = AsyncMock()
            mock_plugin_class.return_value = mock_plugin

        yield {
            "openai": mock_openai,
            "anthropic": mock_anthropic,
            "mlx": mock_mlx,
            "ollama": mock_ollama,
        }


class TestModelServiceInitialization:
    """Test model service initialization and plugin registration."""

    @pytest.mark.asyncio
    async def test_service_initialization_registers_plugins(self, model_service):
        """Test that service initialization registers all plugins."""
        assert "openai_api" in model_service.plugins
        assert "anthropic_api" in model_service.plugins
        assert "mlx_local" in model_service.plugins
        assert "ollama" in model_service.plugins
        assert len(model_service.plugins) == 4

    @pytest.mark.asyncio
    async def test_service_initialization_response(self):
        """Test service initialization response."""
        service = ModelService()
        response = await service.initialize()

        assert response.success is True
        assert "registered_plugins" in response.data
        assert len(response.data["registered_plugins"]) == 4

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_auto_registration_of_all_plugins(self, model_service):
        """Test that all available plugins are auto-registered."""
        expected_plugins = {
            "openai_api": "OpenAIModelPlugin",
            "anthropic_api": "AnthropicModelPlugin",
            "mlx_local": "MLXModelPlugin",
            "ollama": "OllamaModelPlugin",
        }

        for plugin_name, expected_class_name in expected_plugins.items():
            assert plugin_name in model_service.plugins
            assert expected_class_name in str(model_service.plugins[plugin_name])


class TestModelDiscovery:
    """Test model discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_available_models_success(self, model_service, mock_plugins):
        """Test successful model discovery across all plugins."""
        discovery_result = await model_service.discover_available_models()

        assert isinstance(discovery_result, ModelDiscoveryResult)
        assert discovery_result.total_models > 0
        assert len(discovery_result.models_by_plugin) == 4
        assert discovery_result.discovery_time_ms > 0
        assert len(discovery_result.errors) == 0

    @pytest.mark.asyncio
    async def test_discover_models_by_plugin_type(self, model_service, mock_plugins):
        """Test discovery results are properly grouped by plugin."""
        discovery_result = await model_service.discover_available_models()

        expected_plugins = ["openai_api", "anthropic_api", "mlx_local", "ollama"]
        for plugin_name in expected_plugins:
            assert plugin_name in discovery_result.models_by_plugin
            assert len(discovery_result.models_by_plugin[plugin_name]) > 0

    @pytest.mark.asyncio
    async def test_enhanced_model_info_creation(self, model_service, mock_plugins):
        """Test creation of enhanced model info with metadata."""
        discovery_result = await model_service.discover_available_models()

        if discovery_result.available_models:
            model_info = discovery_result.available_models[0]
            assert isinstance(model_info, EnhancedModelInfo)
            assert model_info.plugin_type is not None
            assert model_info.model_id is not None
            assert model_info.model_name is not None
            assert model_info.deployment_type in ["api", "local"]
            assert isinstance(model_info.supports_batching, bool)
            assert isinstance(model_info.supports_explanations, bool)

    @pytest.mark.asyncio
    async def test_list_available_models_shorthand(self, model_service, mock_plugins):
        """Test the shorthand method for listing available models."""
        models = await model_service.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, EnhancedModelInfo) for model in models)

    @pytest.mark.asyncio
    async def test_discovery_with_plugin_errors(self, model_service):
        """Test discovery gracefully handles plugin errors."""
        # Mock _discover_plugin_models to fail for all plugins
        with patch.object(
            model_service, "_discover_plugin_models", side_effect=Exception("Test error")
        ):
            discovery_result = await model_service.discover_available_models()

            # Should have errors for all plugins
            assert len(discovery_result.errors) == 4  # All 4 plugins should error
            assert discovery_result.total_models == 0

            # All plugins should have error status
            for plugin_name in ["openai_api", "anthropic_api", "mlx_local", "ollama"]:
                assert plugin_name in discovery_result.plugin_status
                assert discovery_result.plugin_status[plugin_name]["status"] == "error"


class TestUnifiedInterface:
    """Test unified interface for all model types."""

    @pytest.mark.asyncio
    async def test_load_different_model_types(self, model_service):
        """Test loading models from different plugin types."""
        # Test with non-existent model to check error handling
        config = {"type": "nonexistent_type", "model_name": "test", "name": "test"}

        try:
            await model_service.load_model(config)
            raise AssertionError("Should have raised an error for non-existent plugin type")
        except Exception as e:
            assert "no plugin registered" in str(e).lower() or "nonexistent_type" in str(e).lower()

    @pytest.mark.asyncio
    async def test_unified_prediction_interface(self, model_service):
        """Test unified prediction interface across model types."""
        # This would require setting up mock models with prediction capabilities
        # For now, test the interface exists and handles unloaded models correctly

        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.predict_batch("nonexistent-model", ["test sample"])

        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.error_code.name == "MODEL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_unified_explanation_interface(self, model_service):
        """Test unified explanation interface across model types."""
        # Test with non-existent model
        explanation = await model_service.explain_prediction("nonexistent-model", "test sample")
        assert "not available" in explanation.lower()


class TestPerformanceComparison:
    """Test model performance comparison utilities."""

    @pytest.mark.asyncio
    async def test_compare_model_performance_insufficient_models(self, model_service):
        """Test performance comparison with insufficient models."""
        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.compare_model_performance(["model1"])

        assert "at least 2 models" in str(exc_info.value).lower()
        assert exc_info.value.error_code.name == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_compare_model_performance_no_loaded_models(self, model_service):
        """Test performance comparison with no loaded models."""
        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.compare_model_performance(["model1", "model2"])

        assert "insufficient models" in str(exc_info.value).lower()
        assert exc_info.value.error_code.name == "INSUFFICIENT_DATA"

    @pytest.mark.asyncio
    async def test_performance_comparison_data_structure(self, model_service):
        """Test that performance comparison returns correct data structure."""
        # Mock loaded models with performance data
        model_service.loaded_models["test1"] = MagicMock()
        model_service.loaded_models["test2"] = MagicMock()

        mock_performance = {
            "basic_metrics": {
                "predictions_per_second": 10.0,
                "success_rate": 0.95,
                "average_inference_time_ms": 100.0,
            },
            "average_batch_throughput": 8.0,
        }

        with patch.object(model_service, "get_model_performance", return_value=mock_performance):
            comparison = await model_service.compare_model_performance(["test1", "test2"])

            assert isinstance(comparison, PerformanceComparison)
            assert len(comparison.model_ids) == 2
            assert "test1" in comparison.metrics
            assert "test2" in comparison.metrics
            assert "overall_rankings" in comparison.summary
            assert comparison.summary["best_performer"] is not None


class TestResourceOptimization:
    """Test resource optimization functionality."""

    @pytest.mark.asyncio
    async def test_optimize_model_loading_empty_configs(self, model_service):
        """Test loading optimization with empty configurations."""
        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.optimize_model_loading([])

        assert "no model configurations" in str(exc_info.value).lower()
        assert exc_info.value.error_code.name == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_optimize_model_loading_strategy(self, model_service):
        """Test model loading optimization strategy."""
        configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "fast-api"},
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "small-local"},
            {"type": "mlx_local", "model_name": "llama2-70b", "name": "large-local"},
        ]

        strategy = await model_service.optimize_model_loading(configs)

        assert isinstance(strategy, LoadingStrategy)
        assert len(strategy.loading_order) == 3
        assert len(strategy.model_configs) == 3
        assert strategy.estimated_total_memory_mb > 0
        assert strategy.estimated_loading_time_seconds > 0
        assert len(strategy.optimization_notes) > 0

        # API models should be prioritized (loaded first)
        assert "gpt-4o-mini" in strategy.loading_order[0]

        # Should have resource allocation for each model
        assert len(strategy.resource_allocation) == 3

    @pytest.mark.asyncio
    async def test_parallel_loading_identification(self, model_service):
        """Test identification of parallel loading opportunities."""
        configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "api1"},
            {"type": "anthropic_api", "model_name": "claude-3-haiku-20240307", "name": "api2"},
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "local1"},
            {"type": "mlx_local", "model_name": "llama2-13b", "name": "local2"},
        ]

        strategy = await model_service.optimize_model_loading(configs)

        # Should identify parallel loading groups
        assert len(strategy.parallel_loading_groups) > 0

        # API models should be in the same parallel group
        api_group_found = False
        for group in strategy.parallel_loading_groups:
            if "gpt-4o-mini" in group and "claude-3-haiku-20240307" in group:
                api_group_found = True
                break

        assert api_group_found, "API models should be grouped for parallel loading"


class TestCostEstimation:
    """Test cost estimation functionality."""

    @pytest.mark.asyncio
    async def test_cost_estimation_empty_configs(self, model_service):
        """Test cost estimation with empty configurations."""
        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.get_cost_estimates([], 1000)

        assert "no model configurations" in str(exc_info.value).lower()
        assert exc_info.value.error_code.name == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_cost_estimation_invalid_samples(self, model_service):
        """Test cost estimation with invalid sample count."""
        configs = [{"type": "openai_api", "model_name": "gpt-4o-mini", "name": "test"}]

        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.get_cost_estimates(configs, 0)

        assert "must be positive" in str(exc_info.value).lower()
        assert exc_info.value.error_code.name == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_cost_estimation_api_models(self, model_service):
        """Test cost estimation for API models."""
        configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "openai-test"},
            {
                "type": "anthropic_api",
                "model_name": "claude-3-haiku-20240307",
                "name": "anthropic-test",
            },
        ]

        estimate = await model_service.get_cost_estimates(configs, 1000)

        assert isinstance(estimate, CostEstimate)
        assert estimate.estimated_samples == 1000
        assert estimate.total_estimated_cost_usd > 0
        assert len(estimate.cost_by_model) == 2
        assert estimate.api_costs > 0
        assert estimate.local_compute_costs == 0  # No local models
        assert len(estimate.recommendations) > 0
        assert "assumptions" in estimate.model_dump()

    @pytest.mark.asyncio
    async def test_cost_estimation_local_models(self, model_service):
        """Test cost estimation for local models."""
        configs = [
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "local-small"},
            {"type": "ollama", "model_name": "llama2-13b", "name": "local-medium"},
        ]

        estimate = await model_service.get_cost_estimates(configs, 500)

        assert isinstance(estimate, CostEstimate)
        assert estimate.estimated_samples == 500
        assert estimate.total_estimated_cost_usd > 0
        assert len(estimate.cost_by_model) == 2
        assert estimate.api_costs == 0  # No API models
        assert estimate.local_compute_costs > 0
        assert len(estimate.recommendations) > 0

    @pytest.mark.asyncio
    async def test_cost_estimation_mixed_models(self, model_service):
        """Test cost estimation for mixed API and local models."""
        configs = [
            {"type": "openai_api", "model_name": "gpt-4o-mini", "name": "api"},
            {"type": "mlx_local", "model_name": "llama2-7b", "name": "local"},
        ]

        estimate = await model_service.get_cost_estimates(configs, 1000)

        assert isinstance(estimate, CostEstimate)
        assert estimate.api_costs > 0
        assert estimate.local_compute_costs > 0
        assert (
            estimate.total_estimated_cost_usd == estimate.api_costs + estimate.local_compute_costs
        )

        # Should have recommendations about mixed model usage
        recommendations_text = " ".join(estimate.recommendations).lower()
        assert any(keyword in recommendations_text for keyword in ["api", "local", "cost"])

    @pytest.mark.asyncio
    async def test_cost_recommendations_generation(self, model_service):
        """Test generation of cost optimization recommendations."""
        # High-cost scenario
        configs = [{"type": "openai_api", "model_name": "gpt-4", "name": "expensive"}]
        estimate = await model_service.get_cost_estimates(configs, 10000)  # Large sample

        assert len(estimate.recommendations) > 0
        recommendations_text = " ".join(estimate.recommendations)

        # Should contain helpful recommendations
        assert any(
            word in recommendations_text.lower()
            for word in ["cost", "sample", "batch", "efficiency"]
        )


class TestBatchProcessingOptimization:
    """Test batch processing optimization features."""

    @pytest.mark.asyncio
    async def test_batch_size_recommendations(self, model_service):
        """Test that models get appropriate batch size recommendations."""
        discovery_result = await model_service.discover_available_models()

        for model in discovery_result.available_models:
            assert model.recommended_batch_size is not None
            assert model.recommended_batch_size > 0

            # API models should have smaller batch sizes
            if model.deployment_type == "api":
                assert model.recommended_batch_size <= 8

            # Large local models should have smaller batch sizes
            if model.parameters and model.parameters > 30_000_000_000:  # 30B+ parameters
                assert model.recommended_batch_size <= 8

    @pytest.mark.asyncio
    async def test_memory_requirement_estimation(self, model_service):
        """Test memory requirement estimation for different model sizes."""
        discovery_result = await model_service.discover_available_models()

        for model in discovery_result.available_models:
            if model.memory_requirement_gb is not None:
                assert model.memory_requirement_gb > 0

                # Larger models should require more memory
                if model.parameters:
                    if model.parameters > 70_000_000_000:  # 70B+ parameters
                        assert model.memory_requirement_gb > 100
                    elif model.parameters > 30_000_000_000:  # 30B+ parameters
                        assert model.memory_requirement_gb > 50
                    elif model.parameters < 10_000_000_000:  # <10B parameters
                        assert model.memory_requirement_gb < 50


class TestAdvancedIntegration:
    """Test advanced integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow_discovery_to_optimization(self, model_service, mock_plugins):
        """Test complete workflow from discovery to optimization."""
        # 1. Discover available models
        discovery = await model_service.discover_available_models()
        assert discovery.total_models > 0

        # 2. Create configs from discovered models (select a few)
        sample_models = discovery.available_models[:3]  # Take first 3
        configs = []

        for model in sample_models:
            configs.append(
                {
                    "type": model.plugin_type,
                    "model_name": model.model_name,
                    "name": f"test-{model.model_name}",
                }
            )

        # 3. Get cost estimates
        if configs:
            cost_estimate = await model_service.get_cost_estimates(configs, 1000)
            assert cost_estimate.total_estimated_cost_usd >= 0

            # 4. Optimize loading strategy
            loading_strategy = await model_service.optimize_model_loading(configs)
            assert len(loading_strategy.loading_order) == len(configs)

            # 5. Verify optimization makes sense
            assert loading_strategy.estimated_total_memory_mb > 0
            assert loading_strategy.estimated_loading_time_seconds > 0

    @pytest.mark.asyncio
    async def test_service_health_with_all_features(self, model_service):
        """Test service health check includes all unified management features."""
        health = await model_service.health_check()

        assert health.status in ["healthy", "degraded", "error"]
        assert "loaded_models" in health.checks
        assert "registered_plugins" in health.checks
        assert health.checks["registered_plugins"] == 4  # All plugins registered

    @pytest.mark.asyncio
    async def test_service_stats_includes_plugin_info(self, model_service):
        """Test service statistics include plugin information."""
        stats = await model_service.get_service_stats()

        assert "registered_plugins" in stats
        assert len(stats["registered_plugins"]) == 4
        assert "openai_api" in stats["registered_plugins"]
        assert "anthropic_api" in stats["registered_plugins"]
        assert "mlx_local" in stats["registered_plugins"]
        assert "ollama" in stats["registered_plugins"]

    @pytest.mark.asyncio
    async def test_error_handling_across_features(self, model_service):
        """Test error handling is consistent across all features."""
        # Test discovery with plugin failures
        with patch.object(
            model_service, "_discover_plugin_models", side_effect=Exception("Test error")
        ):
            discovery = await model_service.discover_available_models()
            assert len(discovery.errors) == 4  # All plugins should error

        # Test cost estimation with invalid data
        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.get_cost_estimates([{"invalid": "config"}], -1)
        assert exc_info.value.error_code.name == "INVALID_PARAMETER"

        # Test optimization with invalid data
        with pytest.raises(BenchmarkError) as exc_info:
            await model_service.optimize_model_loading([])
        assert exc_info.value.error_code.name == "INVALID_PARAMETER"


class TestModelTagsAndMetadata:
    """Test model tagging and metadata features."""

    @pytest.mark.asyncio
    async def test_model_tags_generation(self, model_service, mock_plugins):
        """Test that models get appropriate tags for categorization."""
        discovery = await model_service.discover_available_models()

        for model in discovery.available_models:
            assert len(model.tags) > 0

            # Should include plugin type tag
            plugin_tag = model.plugin_type.replace("_", "-")
            assert plugin_tag in model.tags

            # Size-based tags
            if "7b" in model.model_name.lower() or "mini" in model.model_name.lower():
                assert "small" in model.tags
            elif "70b" in model.model_name.lower():
                assert "large" in model.tags

            # Capability tags
            if "chat" in model.model_name.lower():
                assert "conversational" in model.tags
            if "code" in model.model_name.lower():
                assert "code-generation" in model.tags

    @pytest.mark.asyncio
    async def test_performance_tier_assignment(self, model_service, mock_plugins):
        """Test performance tier assignment for different models."""
        discovery = await model_service.discover_available_models()

        for model in discovery.available_models:
            assert model.performance_tier in ["fast", "standard", "premium"]

            # Fast models
            if any(keyword in model.model_name.lower() for keyword in ["mini", "haiku", "small"]):
                assert model.performance_tier == "fast"

            # Premium models
            elif any(keyword in model.model_name.lower() for keyword in ["gpt-4", "opus", "large"]):
                assert model.performance_tier == "premium"

            # Standard models (everything else)
            else:
                assert model.performance_tier == "standard"

    @pytest.mark.asyncio
    async def test_vendor_detection(self, model_service, mock_plugins):
        """Test vendor detection from plugin types."""
        discovery = await model_service.discover_available_models()

        vendor_mapping = {
            "openai_api": "OpenAI",
            "anthropic_api": "Anthropic",
            "mlx_local": "MLX",
            "ollama": "Ollama",
        }

        for model in discovery.available_models:
            expected_vendor = vendor_mapping.get(model.plugin_type)
            if expected_vendor:
                assert model.vendor == expected_vendor

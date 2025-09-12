"""
Unit tests for ModelResourceManager and related components.

This module provides comprehensive unit tests for the intelligent resource
management system, including memory monitoring, model lifecycle management,
and optimization algorithms.
"""

import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from benchmark.core.config import ModelConfig
from benchmark.models.resource_manager import (
    LoadedModelInfo,
    MemoryMonitor,
    ModelEstimate,
    ModelPriority,
    ModelResourceManager,
    ResourceCheckResult,
    SystemMemoryInfo,
    UnloadReason,
)


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    config = Mock(spec=ModelConfig)
    config.model_name = "test-model-7b"
    config.type = "mlx_local"
    config.parameters = {"temperature": 0.7}
    config.context_length = 4096
    return config


@pytest.fixture
def mock_api_config():
    """Create a mock API model configuration."""
    config = Mock(spec=ModelConfig)
    config.model_name = "gpt-4o-mini"
    config.type = "openai_api"
    config.parameters = {"temperature": 0.5}
    config.context_length = 8192
    return config


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest_asyncio.fixture
async def memory_monitor():
    """Create memory monitor instance."""
    monitor = MemoryMonitor()
    yield monitor


@pytest_asyncio.fixture
async def resource_manager(temp_cache_dir):
    """Create resource manager instance."""
    manager = ModelResourceManager(max_memory_gb=16.0, cache_dir=temp_cache_dir)
    await manager.initialize()
    yield manager
    await manager.shutdown()


class TestMemoryMonitor:
    """Test cases for MemoryMonitor class."""

    @pytest.mark.asyncio
    async def test_get_current_usage(self, memory_monitor):
        """Test memory usage retrieval."""
        usage = await memory_monitor.get_current_usage()
        assert isinstance(usage, float)
        assert usage >= 0.0

    @pytest.mark.asyncio
    async def test_get_system_memory_info(self, memory_monitor):
        """Test system memory information retrieval."""
        info = await memory_monitor.get_system_memory_info()

        assert isinstance(info, SystemMemoryInfo)
        assert info.total_gb > 0.0
        assert info.available_gb >= 0.0
        assert info.used_gb >= 0.0
        assert 0.0 <= info.percentage <= 100.0
        assert info.swap_used_gb >= 0.0

    @pytest.mark.asyncio
    async def test_baseline_memory_tracking(self, memory_monitor):
        """Test baseline memory tracking functionality."""
        # Initially no baseline
        assert memory_monitor._baseline_memory is None

        # Set baseline
        await memory_monitor.set_baseline()
        assert memory_monitor._baseline_memory is not None
        assert memory_monitor._baseline_memory >= 0.0

        # Get delta
        delta = await memory_monitor.get_memory_delta()
        assert isinstance(delta, float)

    def test_apple_silicon_detection(self, memory_monitor):
        """Test Apple Silicon detection."""
        with patch("platform.machine") as mock_machine, patch("platform.system") as mock_system:
            # Test Apple Silicon detection
            mock_machine.return_value = "arm64"
            mock_system.return_value = "Darwin"
            assert memory_monitor._detect_apple_silicon() is True

            # Test non-Apple Silicon
            mock_machine.return_value = "x86_64"
            mock_system.return_value = "Linux"
            assert memory_monitor._detect_apple_silicon() is False

    @pytest.mark.asyncio
    async def test_memory_monitor_error_handling(self, memory_monitor):
        """Test error handling in memory monitoring."""
        with patch.object(
            memory_monitor.process, "memory_info", side_effect=Exception("Test error")
        ):
            usage = await memory_monitor.get_current_usage()
            assert usage == 0.0


class TestModelResourceManager:
    """Test cases for ModelResourceManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_cache_dir):
        """Test resource manager initialization."""
        manager = ModelResourceManager(max_memory_gb=32.0, cache_dir=temp_cache_dir)

        assert manager.max_memory_gb == 32.0
        assert manager.cache_dir == temp_cache_dir
        assert len(manager.loaded_models) == 0
        assert not manager._running

        await manager.initialize()
        assert manager._running is True
        assert manager._cleanup_task is not None

        await manager.shutdown()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_model_memory_estimation(self, resource_manager, mock_model_config):
        """Test model memory estimation."""
        # Test specific model estimate
        mock_model_config.model_name = "llama2-7b"
        estimate = await resource_manager._estimate_model_memory(mock_model_config)

        assert isinstance(estimate, ModelEstimate)
        assert estimate.total_estimated_gb > 0.0
        assert 0.0 <= estimate.confidence <= 1.0
        assert estimate.base_memory_gb > 0.0

        # Test API model estimate
        mock_model_config.type = "openai_api"
        mock_model_config.model_name = "gpt-4"
        api_estimate = await resource_manager._estimate_model_memory(mock_model_config)

        assert api_estimate.total_estimated_gb < estimate.total_estimated_gb
        assert api_estimate.confidence > 0.6

    @pytest.mark.asyncio
    async def test_unknown_model_estimation(self, resource_manager, mock_model_config):
        """Test estimation for unknown models."""
        mock_model_config.model_name = "unknown-model-xyz"
        mock_model_config.type = "unknown_type"

        estimate = await resource_manager._estimate_model_memory(mock_model_config)

        assert isinstance(estimate, ModelEstimate)
        assert estimate.total_estimated_gb > 0.0
        assert estimate.confidence == 0.5  # Default confidence for unknown

    @pytest.mark.asyncio
    async def test_can_load_model_success(self, resource_manager, mock_model_config):
        """Test successful model load check."""
        # Mock memory usage to be low
        with (
            patch.object(resource_manager.memory_monitor, "get_current_usage", return_value=2.0),
            patch.object(
                resource_manager.memory_monitor,
                "get_system_memory_info",
                return_value=SystemMemoryInfo(32.0, 20.0, 12.0, 37.5),
            ),
        ):
            result = await resource_manager.can_load_model(mock_model_config)

            assert isinstance(result, ResourceCheckResult)
            assert result.can_load is True
            assert result.estimated_memory_gb > 0.0
            assert result.current_usage_gb == 2.0
            assert len(result.recommendations) >= 0

    @pytest.mark.asyncio
    async def test_can_load_model_failure(self, resource_manager, mock_model_config):
        """Test model load check when memory insufficient."""
        # Mock high memory usage
        with (
            patch.object(resource_manager.memory_monitor, "get_current_usage", return_value=15.0),
            patch.object(
                resource_manager.memory_monitor,
                "get_system_memory_info",
                return_value=SystemMemoryInfo(32.0, 2.0, 30.0, 93.8),
            ),
        ):
            # Set model config to large model
            mock_model_config.model_name = "llama2-70b"

            result = await resource_manager.can_load_model(mock_model_config)

            assert isinstance(result, ResourceCheckResult)
            assert result.can_load is False
            assert result.estimated_memory_gb > 0.0
            assert len(result.recommendations) > 0
            assert "Need to free" in result.recommendations[0]

    @pytest.mark.asyncio
    async def test_model_registration(self, resource_manager, mock_model_config):
        """Test model registration and tracking."""
        model_id = "test_model_123"
        actual_memory = 8.5
        plugin_type = "mlx_local"

        await resource_manager.register_model_load(
            model_id, mock_model_config, actual_memory, plugin_type
        )

        assert model_id in resource_manager.loaded_models

        model_info = resource_manager.loaded_models[model_id]
        assert isinstance(model_info, LoadedModelInfo)
        assert model_info.model_id == model_id
        assert model_info.memory_usage_gb == actual_memory
        assert model_info.plugin_type == plugin_type
        assert model_info.access_count == 1
        assert model_info.priority == ModelPriority.NORMAL

    @pytest.mark.asyncio
    async def test_model_access_tracking(self, resource_manager, mock_model_config):
        """Test model access tracking."""
        model_id = "test_model_access"

        # Register model first
        await resource_manager.register_model_load(model_id, mock_model_config, 4.0, "test_plugin")

        initial_count = resource_manager.loaded_models[model_id].access_count
        initial_time = resource_manager.loaded_models[model_id].last_used

        # Wait a bit to ensure time difference
        await asyncio.sleep(0.01)

        # Register access
        await resource_manager.register_model_access(model_id)

        model_info = resource_manager.loaded_models[model_id]
        assert model_info.access_count == initial_count + 1
        assert model_info.last_used > initial_time

    @pytest.mark.asyncio
    async def test_model_unregistration(self, resource_manager, mock_model_config):
        """Test model unregistration."""
        model_id = "test_model_unregister"

        # Register model first
        await resource_manager.register_model_load(model_id, mock_model_config, 6.0, "test_plugin")

        assert model_id in resource_manager.loaded_models

        # Unregister model
        await resource_manager.unregister_model(model_id, UnloadReason.USER_REQUEST)

        assert model_id not in resource_manager.loaded_models

    @pytest.mark.asyncio
    async def test_loading_order_optimization(self, resource_manager):
        """Test model loading order optimization."""
        # Create multiple model configs with different characteristics
        configs = []

        # Small API model
        api_config = Mock(spec=ModelConfig)
        api_config.model_name = "gpt-4o-mini"
        api_config.type = "openai_api"
        configs.append(api_config)

        # Large local model
        large_config = Mock(spec=ModelConfig)
        large_config.model_name = "llama2-70b"
        large_config.type = "mlx_local"
        configs.append(large_config)

        # Medium local model
        medium_config = Mock(spec=ModelConfig)
        medium_config.model_name = "llama2-7b"
        medium_config.type = "mlx_local"
        configs.append(medium_config)

        optimized_order = await resource_manager.optimize_model_loading_order(configs)

        assert len(optimized_order) == 3
        assert isinstance(optimized_order, list)

        # API model should come first (highest priority)
        assert optimized_order[0].type == "openai_api"

    @pytest.mark.asyncio
    async def test_priority_score_calculation(self, resource_manager, mock_model_config):
        """Test priority score calculation for different model types."""
        # Test API model (should have high priority)
        mock_model_config.type = "openai_api"
        mock_model_config.model_name = "gpt-4"

        estimate = ModelEstimate(0.5, 0.2, 0.3, 1.0, 0.9)
        api_score = resource_manager._calculate_priority_score(mock_model_config, estimate)

        # Test large local model (should have lower priority)
        mock_model_config.type = "mlx_local"
        mock_model_config.model_name = "llama2-70b"

        large_estimate = ModelEstimate(70.0, 8.0, 7.0, 85.0, 0.8)
        large_score = resource_manager._calculate_priority_score(mock_model_config, large_estimate)

        assert api_score > large_score

    @pytest.mark.asyncio
    async def test_unload_candidates_suggestion(self, resource_manager, mock_model_config):
        """Test model unload candidate suggestions."""
        # Register multiple models with different usage patterns
        model_ids = []

        # Old, rarely used model (good candidate)
        old_model_id = "old_model"
        await resource_manager.register_model_load(old_model_id, mock_model_config, 8.0, "test")
        old_info = resource_manager.loaded_models[old_model_id]
        old_info.last_used = datetime.now() - timedelta(hours=5)
        old_info.access_count = 1
        model_ids.append(old_model_id)

        # Recent, frequently used model (poor candidate)
        recent_model_id = "recent_model"
        await resource_manager.register_model_load(recent_model_id, mock_model_config, 4.0, "test")
        recent_info = resource_manager.loaded_models[recent_model_id]
        recent_info.access_count = 20
        model_ids.append(recent_model_id)

        # Get unload candidates
        candidates = await resource_manager.suggest_model_unload_candidates(6.0)

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        # Old model should be suggested first
        assert old_model_id in candidates

    @pytest.mark.asyncio
    async def test_unload_score_calculation(self, resource_manager):
        """Test unload score calculation logic."""
        now = datetime.now()

        # Create model info with known characteristics
        old_model = LoadedModelInfo(
            model_id="old_model",
            config=Mock(),
            memory_usage_gb=10.0,
            load_time=now - timedelta(hours=4),
            last_used=now - timedelta(hours=3),
            access_count=2,
            plugin_type="test",
            priority=ModelPriority.LOW,
        )

        recent_model = LoadedModelInfo(
            model_id="recent_model",
            config=Mock(),
            memory_usage_gb=5.0,
            load_time=now - timedelta(hours=1),
            last_used=now - timedelta(minutes=10),
            access_count=15,
            plugin_type="test",
            priority=ModelPriority.HIGH,
        )

        old_score = resource_manager._calculate_unload_score(old_model)
        recent_score = resource_manager._calculate_unload_score(recent_model)

        # Old model should have lower score (better candidate for unloading)
        assert old_score < recent_score

    @pytest.mark.asyncio
    async def test_idle_model_detection(self, resource_manager):
        """Test idle model detection."""
        now = datetime.now()

        # Create idle model
        idle_model = LoadedModelInfo(
            model_id="idle_model",
            config=Mock(),
            memory_usage_gb=5.0,
            load_time=now - timedelta(hours=2),
            last_used=now - timedelta(hours=1),
            access_count=1,
            plugin_type="test",
        )

        # Create active model
        active_model = LoadedModelInfo(
            model_id="active_model",
            config=Mock(),
            memory_usage_gb=3.0,
            load_time=now - timedelta(minutes=30),
            last_used=now - timedelta(minutes=5),
            access_count=10,
            plugin_type="test",
        )

        assert resource_manager._is_model_idle(idle_model) is True
        assert resource_manager._is_model_idle(active_model) is False

    @pytest.mark.asyncio
    async def test_resource_statistics(self, resource_manager, mock_model_config):
        """Test resource statistics generation."""
        # Register a few models
        await resource_manager.register_model_load("model1", mock_model_config, 4.0, "test")
        await resource_manager.register_model_load("model2", mock_model_config, 6.0, "test")

        with (
            patch.object(resource_manager.memory_monitor, "get_current_usage", return_value=12.0),
            patch.object(
                resource_manager.memory_monitor,
                "get_system_memory_info",
                return_value=SystemMemoryInfo(32.0, 20.0, 12.0, 37.5, apple_silicon_unified=True),
            ),
        ):
            stats = await resource_manager.get_resource_statistics()

            assert isinstance(stats, dict)
            assert "memory" in stats
            assert "models" in stats
            assert "performance" in stats

            assert stats["memory"]["current_usage_gb"] == 12.0
            assert stats["models"]["loaded_count"] == 2
            assert stats["models"]["total_memory_gb"] == 10.0
            assert stats["memory"]["apple_silicon"] is True

    @pytest.mark.asyncio
    async def test_force_cleanup(self, resource_manager, mock_model_config):
        """Test force cleanup of idle models."""
        now = datetime.now()

        # Register models with different ages
        await resource_manager.register_model_load("model1", mock_model_config, 4.0, "test")
        await resource_manager.register_model_load("model2", mock_model_config, 6.0, "test")

        # Make one model idle
        idle_model = resource_manager.loaded_models["model1"]
        idle_model.last_used = now - timedelta(hours=2)

        cleanup_result = await resource_manager.force_cleanup()

        assert isinstance(cleanup_result, dict)
        assert "unloaded_models" in cleanup_result
        assert "freed_memory_gb" in cleanup_result
        assert len(cleanup_result["unloaded_models"]) >= 0

    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, resource_manager, mock_api_config):
        """Test optimization recommendations generation."""
        recommendations = await resource_manager._get_optimization_recommendations(mock_api_config)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # API models should get specific recommendations
        api_recs = [r for r in recommendations if "API model" in r]
        assert len(api_recs) > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_resource_check(self, resource_manager, mock_model_config):
        """Test error handling in resource availability check."""
        with patch.object(
            resource_manager.memory_monitor,
            "get_current_usage",
            side_effect=Exception("Memory error"),
        ):
            result = await resource_manager.can_load_model(mock_model_config)

            assert isinstance(result, ResourceCheckResult)
            assert result.can_load is False
            assert len(result.recommendations) > 0
            assert "Error checking resources" in result.recommendations[0]

    @pytest.mark.asyncio
    async def test_background_cleanup_task(self, resource_manager, mock_model_config):
        """Test background cleanup task functionality."""
        # Verify cleanup task is running
        assert resource_manager._cleanup_task is not None
        assert not resource_manager._cleanup_task.done()

        # Register an idle model
        await resource_manager.register_model_load("idle_test", mock_model_config, 5.0, "test")
        idle_model = resource_manager.loaded_models["idle_test"]
        idle_model.last_used = datetime.now() - timedelta(hours=2)

        # Force a cleanup cycle by setting high memory pressure
        with patch.object(resource_manager.memory_monitor, "get_current_usage", return_value=15.0):
            # Wait a short time for background task
            await asyncio.sleep(0.1)


class TestModelEstimate:
    """Test cases for ModelEstimate dataclass."""

    def test_model_estimate_creation(self):
        """Test ModelEstimate creation and attributes."""
        estimate = ModelEstimate(
            base_memory_gb=7.0,
            context_memory_gb=1.0,
            overhead_memory_gb=1.0,
            total_estimated_gb=9.0,
            confidence=0.9,
        )

        assert estimate.base_memory_gb == 7.0
        assert estimate.total_estimated_gb == 9.0
        assert estimate.confidence == 0.9

        # Test dict conversion
        estimate_dict = asdict(estimate)
        assert estimate_dict["base_memory_gb"] == 7.0
        assert estimate_dict["confidence"] == 0.9


class TestLoadedModelInfo:
    """Test cases for LoadedModelInfo dataclass."""

    def test_loaded_model_info_creation(self):
        """Test LoadedModelInfo creation and attributes."""
        config = Mock(spec=ModelConfig)
        now = datetime.now()

        model_info = LoadedModelInfo(
            model_id="test_model",
            config=config,
            memory_usage_gb=8.5,
            load_time=now,
            last_used=now,
            access_count=1,
            plugin_type="mlx_local",
        )

        assert model_info.model_id == "test_model"
        assert model_info.memory_usage_gb == 8.5
        assert model_info.priority == ModelPriority.NORMAL
        assert model_info.estimated_memory_gb == 0.0  # default value


class TestResourceCheckResult:
    """Test cases for ResourceCheckResult dataclass."""

    def test_resource_check_result_creation(self):
        """Test ResourceCheckResult creation and attributes."""
        result = ResourceCheckResult(
            can_load=True,
            estimated_memory_gb=4.5,
            current_usage_gb=8.0,
            available_memory_gb=16.0,
            recommendations=["Model can be loaded safely"],
            required_unloads=[],
            optimization_suggestions=["Use batch processing"],
        )

        assert result.can_load is True
        assert result.estimated_memory_gb == 4.5
        assert len(result.recommendations) == 1
        assert len(result.optimization_suggestions) == 1
        assert result.required_unloads == []


class TestSystemMemoryInfo:
    """Test cases for SystemMemoryInfo dataclass."""

    def test_system_memory_info_creation(self):
        """Test SystemMemoryInfo creation and attributes."""
        info = SystemMemoryInfo(
            total_gb=32.0,
            available_gb=20.0,
            used_gb=12.0,
            percentage=37.5,
            swap_used_gb=0.5,
            apple_silicon_unified=True,
        )

        assert info.total_gb == 32.0
        assert info.percentage == 37.5
        assert info.apple_silicon_unified is True
        assert info.swap_used_gb == 0.5


class TestEnums:
    """Test cases for enum classes."""

    def test_model_priority_enum(self):
        """Test ModelPriority enum values."""
        assert ModelPriority.LOW.value == 1
        assert ModelPriority.NORMAL.value == 2
        assert ModelPriority.HIGH.value == 3
        assert ModelPriority.CRITICAL.value == 4

    def test_unload_reason_enum(self):
        """Test UnloadReason enum values."""
        assert UnloadReason.MEMORY_PRESSURE.value == "memory_pressure"
        assert UnloadReason.IDLE_TIMEOUT.value == "idle_timeout"
        assert UnloadReason.USER_REQUEST.value == "user_request"
        assert UnloadReason.OPTIMIZATION.value == "optimization"

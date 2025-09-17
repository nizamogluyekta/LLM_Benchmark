"""Unit tests for orchestration service."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from benchmark.core.base import HealthCheck, ServiceResponse, ServiceStatus
from benchmark.interfaces.orchestration_interfaces import ExperimentProgress, ExperimentStatus
from benchmark.services.orchestration_service import ExperimentContext, OrchestrationService


class TestOrchestrationService:
    """Test orchestration service functionality."""

    @pytest.fixture
    def orchestration_service(self):
        """Create orchestration service for testing."""
        service = OrchestrationService()
        return service

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        services = {}

        for service_name in ["config", "data", "model", "evaluation"]:
            mock_service = AsyncMock()
            mock_service.initialize.return_value = ServiceResponse(
                success=True, message=f"{service_name} initialized"
            )
            mock_service.health_check.return_value = HealthCheck(
                status=ServiceStatus.HEALTHY, message=f"{service_name} healthy"
            )
            mock_service.shutdown.return_value = ServiceResponse(
                success=True, message=f"{service_name} shutdown"
            )
            services[service_name] = mock_service

        return services

    @pytest.fixture
    def sample_config(self):
        """Sample experiment configuration."""
        return {
            "name": "Test Experiment",
            "models": [
                {"id": "test_model_1", "plugin_type": "local"},
                {"id": "test_model_2", "plugin_type": "api"},
            ],
            "datasets": [
                {"id": "test_dataset_1", "loader_type": "local"},
                {"id": "test_dataset_2", "loader_type": "kaggle"},
            ],
            "evaluation": {"metrics": ["accuracy", "precision", "recall"]},
        }

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_service_initialization(self, orchestration_service, mock_services):
        """Test service initialization."""
        with patch.multiple(
            "benchmark.services.orchestration_service",
            ConfigurationService=lambda: mock_services["config"],
            DataService=lambda: mock_services["data"],
            ModelService=lambda: mock_services["model"],
            EvaluationService=lambda: mock_services["evaluation"],
        ):
            response = await orchestration_service.initialize()

            assert response.success
            assert response.data["initialized_services"] == 4
            assert orchestration_service.is_initialized()

            # Verify all services were initialized
            for service in mock_services.values():
                service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_initialization_failure(self, orchestration_service):
        """Test service initialization with failure."""
        mock_config_service = AsyncMock()
        mock_config_service.initialize.return_value = ServiceResponse(
            success=False, message="Config service failed", error="Config service failed"
        )

        with patch(
            "benchmark.services.orchestration_service.ConfigurationService",
            return_value=mock_config_service,
        ):
            response = await orchestration_service.initialize()

            assert not response.success
            assert "Config service failed" in response.error
            assert not orchestration_service.is_initialized()

    @pytest.mark.asyncio
    async def test_create_experiment(self, orchestration_service, mock_services, sample_config):
        """Test experiment creation."""
        # Setup mocks
        orchestration_service.services = mock_services
        mock_services["config"].load_experiment_config.return_value = ServiceResponse(
            success=True, message="Config loaded", data=sample_config
        )

        experiment_id = await orchestration_service.create_experiment(
            config_path="/path/to/config.yaml", experiment_name="My Test Experiment"
        )

        assert experiment_id.startswith("exp_")
        assert experiment_id in orchestration_service.experiments

        context = orchestration_service.experiments[experiment_id]
        assert context.name == "My Test Experiment"
        assert context.status == ExperimentStatus.CREATED
        assert context.config == sample_config

    @pytest.mark.asyncio
    async def test_create_experiment_config_failure(self, orchestration_service, mock_services):
        """Test experiment creation with config loading failure."""
        orchestration_service.services = mock_services
        mock_services["config"].load_experiment_config.return_value = ServiceResponse(
            success=False, message="Config not found", error="Config not found"
        )

        with pytest.raises(Exception, match="Failed to load configuration"):
            await orchestration_service.create_experiment(config_path="/invalid/path.yaml")

    @pytest.mark.asyncio
    async def test_start_experiment_background(
        self, orchestration_service, mock_services, sample_config
    ):
        """Test starting experiment in background."""
        # Setup
        orchestration_service.services = mock_services
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        orchestration_service.experiments[experiment_id] = context

        # Mock workflow execution
        with patch.object(orchestration_service, "_run_experiment_workflow") as mock_workflow:
            mock_workflow.return_value = {"status": "completed"}

            result = await orchestration_service.start_experiment(experiment_id, background=True)

            assert "message" in result
            assert "background" in result["message"]
            # Task should be started but not awaited
            assert experiment_id in orchestration_service._running_tasks

    @pytest.mark.asyncio
    async def test_start_experiment_synchronous(
        self, orchestration_service, mock_services, sample_config
    ):
        """Test starting experiment synchronously."""
        # Setup
        orchestration_service.services = mock_services
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        orchestration_service.experiments[experiment_id] = context

        # Mock workflow execution
        with patch.object(orchestration_service, "_run_experiment_workflow") as mock_workflow:
            expected_result = {"experiment_id": experiment_id, "status": "completed"}
            mock_workflow.return_value = expected_result

            result = await orchestration_service.start_experiment(experiment_id, background=False)

            assert result == expected_result
            mock_workflow.assert_called_once_with(experiment_id)

    @pytest.mark.asyncio
    async def test_start_experiment_not_found(self, orchestration_service):
        """Test starting non-existent experiment."""
        with pytest.raises(ValueError, match="Experiment not_found not found"):
            await orchestration_service.start_experiment("not_found")

    @pytest.mark.asyncio
    async def test_start_experiment_wrong_status(
        self, orchestration_service, mock_services, sample_config
    ):
        """Test starting experiment in wrong status."""
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        context.status = ExperimentStatus.RUNNING  # Wrong status
        orchestration_service.experiments[experiment_id] = context

        with pytest.raises(ValueError, match="not in CREATED status"):
            await orchestration_service.start_experiment(experiment_id)

    @pytest.mark.asyncio
    async def test_get_experiment_progress(
        self, orchestration_service, mock_services, sample_config
    ):
        """Test getting experiment progress."""
        # Setup experiment
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        context.status = ExperimentStatus.RUNNING
        context.started_at = datetime.now()
        context.total_steps = 4
        context.completed_steps = 2
        context.current_step = "model_loading"
        orchestration_service.experiments[experiment_id] = context

        progress = await orchestration_service.get_experiment_progress(experiment_id)

        assert isinstance(progress, ExperimentProgress)
        assert progress.experiment_id == experiment_id
        assert progress.status == ExperimentStatus.RUNNING
        assert progress.current_step == "model_loading"
        assert progress.total_steps == 4
        assert progress.completed_steps == 2
        assert progress.percentage == 50.0
        assert progress.elapsed_time_seconds > 0

    @pytest.mark.asyncio
    async def test_get_experiment_progress_not_found(self, orchestration_service):
        """Test getting progress for non-existent experiment."""
        with pytest.raises(ValueError, match="Experiment not_found not found"):
            await orchestration_service.get_experiment_progress("not_found")

    @pytest.mark.asyncio
    async def test_cancel_experiment(self, orchestration_service, mock_services, sample_config):
        """Test experiment cancellation."""
        # Setup running experiment
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        context.status = ExperimentStatus.RUNNING
        orchestration_service.experiments[experiment_id] = context

        # Mock running task
        mock_task = AsyncMock()
        orchestration_service._running_tasks[experiment_id] = mock_task

        # Mock cleanup
        with patch.object(orchestration_service, "_cleanup_experiment_resources") as mock_cleanup:
            result = await orchestration_service.cancel_experiment(experiment_id)

            assert result is True
            assert context.status == ExperimentStatus.CANCELLED
            assert context.cancel_requested is True
            mock_task.cancel.assert_called_once()
            mock_cleanup.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_cancel_experiment_wrong_status(
        self, orchestration_service, mock_services, sample_config
    ):
        """Test cancelling experiment in wrong status."""
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        context.status = ExperimentStatus.COMPLETED  # Cannot cancel completed
        orchestration_service.experiments[experiment_id] = context

        with pytest.raises(ValueError, match="cannot be cancelled"):
            await orchestration_service.cancel_experiment(experiment_id)

    @pytest.mark.asyncio
    async def test_list_experiments(self, orchestration_service, mock_services, sample_config):
        """Test listing experiments."""
        # Create multiple experiments
        for i in range(3):
            experiment_id = f"test_exp_{i}"
            context = ExperimentContext(
                experiment_id=experiment_id,
                name=f"Test Experiment {i}",
                config=sample_config,
                services=mock_services,
            )
            context.status = ExperimentStatus.RUNNING if i == 0 else ExperimentStatus.COMPLETED
            orchestration_service.experiments[experiment_id] = context

        # List all experiments
        all_experiments = await orchestration_service.list_experiments()
        assert len(all_experiments) == 3

        # List only running experiments
        running_experiments = await orchestration_service.list_experiments(
            status_filter=ExperimentStatus.RUNNING
        )
        assert len(running_experiments) == 1
        assert running_experiments[0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_experiment_result(self, orchestration_service, mock_services, sample_config):
        """Test getting experiment result."""
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        context.status = ExperimentStatus.COMPLETED
        context.started_at = datetime.now()
        context.completed_at = datetime.now()
        context.results = {"accuracy": 0.95}
        context.loaded_models = {"model1": {}, "model2": {}}
        context.loaded_datasets = {"dataset1": {}}
        orchestration_service.experiments[experiment_id] = context

        result = await orchestration_service.get_experiment_result(experiment_id)

        assert result.experiment_id == experiment_id
        assert result.experiment_name == "Test Experiment"
        assert result.status == ExperimentStatus.COMPLETED
        assert result.models_evaluated == ["model1", "model2"]
        assert result.datasets_used == ["dataset1"]
        assert result.evaluation_results == {"accuracy": 0.95}

    @pytest.mark.asyncio
    async def test_health_check(self, orchestration_service, mock_services):
        """Test health check."""
        orchestration_service.services = mock_services

        health = await orchestration_service.health_check()

        assert health.status == ServiceStatus.HEALTHY.value
        assert "dependent_services" in health.checks
        assert "active_experiments" in health.checks
        assert health.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_health_check_with_unhealthy_service(self, orchestration_service, mock_services):
        """Test health check with unhealthy dependent service."""
        orchestration_service.services = mock_services

        # Make one service unhealthy
        mock_services["model"].health_check.return_value = HealthCheck(
            status=ServiceStatus.UNHEALTHY, message="Model service down"
        )

        health = await orchestration_service.health_check()

        assert health.status == ServiceStatus.UNHEALTHY.value

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestration_service, mock_services, sample_config):
        """Test graceful shutdown."""
        orchestration_service.services = mock_services

        # Add running experiment
        experiment_id = "test_exp_123"
        context = ExperimentContext(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=sample_config,
            services=mock_services,
        )
        context.status = ExperimentStatus.RUNNING
        orchestration_service.experiments[experiment_id] = context

        # Mock running task
        mock_task = AsyncMock()
        orchestration_service._running_tasks[experiment_id] = mock_task

        with patch.object(orchestration_service, "cancel_experiment") as mock_cancel:
            response = await orchestration_service.shutdown()

            assert response.success
            mock_cancel.assert_called_once_with(experiment_id)

            # Verify all services were shut down
            for service in mock_services.values():
                service.shutdown.assert_called_once()

    def test_estimate_remaining_time(self, orchestration_service, mock_services, sample_config):
        """Test remaining time estimation."""
        context = ExperimentContext(
            experiment_id="test_exp",
            name="Test",
            config=sample_config,
            services=mock_services,
        )

        # Test with no progress
        context.completed_steps = 0
        context.total_steps = 4
        result = orchestration_service._estimate_remaining_time(context)
        assert result is None

        # Test with progress
        context.started_at = datetime.now()
        context.completed_steps = 1
        context.total_steps = 4

        # Mock some elapsed time
        with patch("benchmark.services.orchestration_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = context.started_at.replace(
                second=context.started_at.second + 60
            )

            result = orchestration_service._estimate_remaining_time(context)

            # Should estimate 3 more minutes (3 * 60 seconds)
            assert result is not None
            assert result > 0

    @pytest.mark.asyncio
    async def test_cleanup_experiment_resources(
        self, orchestration_service, mock_services, sample_config
    ):
        """Test experiment resource cleanup."""
        context = ExperimentContext(
            experiment_id="test_exp",
            name="Test",
            config=sample_config,
            services=mock_services,
        )
        context.loaded_models = {"model1": {}, "model2": {}}
        context.loaded_datasets = {"dataset1": {}, "dataset2": {}}

        # Mock model service cleanup
        mock_services["model"].cleanup_model = AsyncMock()

        await orchestration_service._cleanup_experiment_resources(context)

        # Verify model cleanup was called
        assert mock_services["model"].cleanup_model.call_count == 2

        # Verify resources were cleared
        assert len(context.loaded_models) == 0
        assert len(context.loaded_datasets) == 0

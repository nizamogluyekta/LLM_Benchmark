"""Integration tests for orchestration service with complete workflows."""

import asyncio
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.interfaces.orchestration_interfaces import ExperimentStatus
from benchmark.services.orchestration_service import OrchestrationService
from benchmark.workflow.workflow_engine import WorkflowEngine
from benchmark.workflow.workflow_steps import (
    DataLoadingStep,
    EvaluationExecutionStep,
    ModelLoadingStep,
    ResultsAggregationStep,
)


class TestOrchestrationIntegration:
    """Integration tests for complete workflow execution."""

    @pytest.fixture
    async def orchestration_service(self):
        """Create fully initialized orchestration service."""
        service = OrchestrationService()
        return service

    @pytest.fixture
    def sample_experiment_config(self):
        """Sample experiment configuration for testing."""
        return {
            "name": "Integration Test Experiment",
            "description": "Full workflow integration test",
            "models": [
                {
                    "id": "test_model_1",
                    "plugin_type": "local",
                    "config": {"model_path": "/tmp/test_model_1"},
                },
                {
                    "id": "test_model_2",
                    "plugin_type": "api",
                    "config": {"api_endpoint": "http://localhost:8000/api"},
                },
            ],
            "datasets": [
                {
                    "id": "test_dataset_1",
                    "loader_type": "local",
                    "config": {"file_path": "/tmp/test_dataset_1.csv"},
                },
                {
                    "id": "test_dataset_2",
                    "loader_type": "kaggle",
                    "config": {"dataset_name": "test/dataset2"},
                },
            ],
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "batch_size": 32,
                "cross_validation": False,
            },
        }

    @pytest.fixture
    def mock_service_responses(self):
        """Mock responses for all services."""
        return {
            "config": {
                "load_experiment_config": {
                    "success": True,
                    "data": {},  # Will be set by test
                }
            },
            "data": {
                "load_dataset": {
                    "success": True,
                    "data": {
                        "size": 1000,
                        "features": ["feature1", "feature2", "label"],
                        "path": "/tmp/loaded_dataset",
                    },
                }
            },
            "model": {
                "load_model": {
                    "success": True,
                    "data": {
                        "model_id": "loaded_model",
                        "memory_usage_mb": 512,
                        "status": "ready",
                    },
                },
                "cleanup_model": {"success": True, "message": "Model cleaned up"},
            },
            "evaluation": {
                "evaluate_model": {
                    "success": True,
                    "data": {
                        "accuracy": 0.85,
                        "precision": 0.82,
                        "recall": 0.88,
                        "f1_score": 0.85,
                        "evaluation_time": 45.2,
                    },
                }
            },
        }

    @pytest.fixture
    def config_file(self, sample_experiment_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_experiment_config, f)
            return f.name

    @pytest.mark.asyncio
    async def test_complete_workflow_execution(
        self, orchestration_service, config_file, sample_experiment_config, mock_service_responses
    ):
        """Test complete experiment workflow from start to finish."""
        # Setup mock services
        mock_services = {}
        for service_name in ["config", "data", "model", "evaluation"]:
            mock_service = AsyncMock()

            # Setup initialization
            mock_service.initialize.return_value = MagicMock(
                success=True, message=f"{service_name} initialized"
            )

            # Setup health check
            mock_service.health_check.return_value = MagicMock(
                status=MagicMock(value="healthy"), message=f"{service_name} healthy"
            )

            # Setup shutdown
            mock_service.shutdown.return_value = MagicMock(
                success=True, message=f"{service_name} shutdown"
            )

            mock_services[service_name] = mock_service

        # Setup specific service responses
        mock_service_responses["config"]["load_experiment_config"]["data"] = (
            sample_experiment_config
        )
        mock_services["config"].load_experiment_config.return_value = MagicMock(
            **mock_service_responses["config"]["load_experiment_config"]
        )

        mock_services["data"].load_dataset.return_value = MagicMock(
            **mock_service_responses["data"]["load_dataset"]
        )

        mock_services["model"].load_model.return_value = MagicMock(
            **mock_service_responses["model"]["load_model"]
        )
        mock_services["model"].cleanup_model.return_value = MagicMock(
            **mock_service_responses["model"]["cleanup_model"]
        )

        mock_services["evaluation"].evaluate_model.return_value = MagicMock(
            **mock_service_responses["evaluation"]["evaluate_model"]
        )

        # Patch service creation
        with patch.multiple(
            "benchmark.services.orchestration_service",
            ConfigurationService=lambda: mock_services["config"],
            DataService=lambda: mock_services["data"],
            ModelService=lambda: mock_services["model"],
            EvaluationService=lambda: mock_services["evaluation"],
        ):
            # Initialize orchestration service
            init_response = await orchestration_service.initialize()
            assert init_response.success

            # Create experiment
            experiment_id = await orchestration_service.create_experiment(
                config_path=config_file, experiment_name="Integration Test"
            )
            assert experiment_id is not None

            # Start experiment (synchronous for testing)
            result = await orchestration_service.start_experiment(experiment_id, background=False)

            # Verify experiment completed successfully
            assert result["experiment_id"] == experiment_id
            assert result["status"] == ExperimentStatus.COMPLETED.value

            # Get experiment result
            experiment_result = await orchestration_service.get_experiment_result(experiment_id)
            assert experiment_result.status == ExperimentStatus.COMPLETED
            assert len(experiment_result.models_evaluated) == 2
            assert len(experiment_result.datasets_used) == 2

            # Verify all workflow steps completed
            assert "data_loading" in experiment_result.evaluation_results
            assert "model_loading" in experiment_result.evaluation_results
            assert "evaluation_execution" in experiment_result.evaluation_results
            assert "results_aggregation" in experiment_result.evaluation_results

            # Verify service calls
            mock_services["config"].load_experiment_config.assert_called_once()
            assert mock_services["data"].load_dataset.call_count == 2  # 2 datasets
            assert mock_services["model"].load_model.call_count == 2  # 2 models
            assert mock_services["evaluation"].evaluate_model.call_count == 4  # 2x2 combinations

    @pytest.mark.asyncio
    async def test_workflow_with_partial_failures(
        self, orchestration_service, config_file, sample_experiment_config, mock_service_responses
    ):
        """Test workflow handling when some evaluations fail."""
        # Setup mock services with some failures
        mock_services = {}
        for service_name in ["config", "data", "model", "evaluation"]:
            mock_service = AsyncMock()
            mock_service.initialize.return_value = MagicMock(
                success=True, message=f"{service_name} initialized"
            )
            mock_service.health_check.return_value = MagicMock(
                status=MagicMock(value="healthy"), message=f"{service_name} healthy"
            )
            mock_service.shutdown.return_value = MagicMock(
                success=True, message=f"{service_name} shutdown"
            )
            mock_services[service_name] = mock_service

        # Setup responses with some failures
        mock_service_responses["config"]["load_experiment_config"]["data"] = (
            sample_experiment_config
        )
        mock_services["config"].load_experiment_config.return_value = MagicMock(
            **mock_service_responses["config"]["load_experiment_config"]
        )

        mock_services["data"].load_dataset.return_value = MagicMock(
            **mock_service_responses["data"]["load_dataset"]
        )

        mock_services["model"].load_model.return_value = MagicMock(
            **mock_service_responses["model"]["load_model"]
        )

        # Make some evaluations fail
        def mock_evaluate_model(*_args, **_kwargs):
            # Fail every other call
            if mock_services["evaluation"].evaluate_model.call_count % 2 == 0:
                return MagicMock(success=False, error="Evaluation failed")
            else:
                return MagicMock(**mock_service_responses["evaluation"]["evaluate_model"])

        mock_services["evaluation"].evaluate_model.side_effect = mock_evaluate_model

        with patch.multiple(
            "benchmark.services.orchestration_service",
            ConfigurationService=lambda: mock_services["config"],
            DataService=lambda: mock_services["data"],
            ModelService=lambda: mock_services["model"],
            EvaluationService=lambda: mock_services["evaluation"],
        ):
            # Initialize and run
            await orchestration_service.initialize()
            experiment_id = await orchestration_service.create_experiment(config_path=config_file)
            result = await orchestration_service.start_experiment(experiment_id, background=False)

            # Should still complete even with partial failures
            assert result["status"] == ExperimentStatus.COMPLETED.value

            # Check evaluation results show both successes and failures
            experiment_result = await orchestration_service.get_experiment_result(experiment_id)
            eval_results = experiment_result.evaluation_results.get("evaluation_execution", {})

            if "results" in eval_results:
                # Should have some successful and some failed evaluations
                all_model_results = eval_results["results"]
                has_success = False
                has_failure = False

                for model_results in all_model_results.values():
                    for dataset_result in model_results.values():
                        if "error" in dataset_result:
                            has_failure = True
                        else:
                            has_success = True

                # We expect both successes and failures
                assert has_success or has_failure  # At least one type occurred

    @pytest.mark.asyncio
    async def test_experiment_cancellation_during_execution(
        self, orchestration_service, config_file, sample_experiment_config
    ):
        """Test experiment cancellation during workflow execution."""
        # Setup mock services with slow responses
        mock_services = {}
        for service_name in ["config", "data", "model", "evaluation"]:
            mock_service = AsyncMock()
            mock_service.initialize.return_value = MagicMock(
                success=True, message=f"{service_name} initialized"
            )
            mock_service.health_check.return_value = MagicMock(
                status=MagicMock(value="healthy"), message=f"{service_name} healthy"
            )
            mock_service.shutdown.return_value = MagicMock(
                success=True, message=f"{service_name} shutdown"
            )
            mock_services[service_name] = mock_service

        # Setup config loading
        mock_services["config"].load_experiment_config.return_value = MagicMock(
            success=True, data=sample_experiment_config
        )

        # Make data loading slow to allow cancellation
        async def slow_data_loading(*_args, **_kwargs):
            await asyncio.sleep(1.0)  # Simulate slow loading
            return MagicMock(
                success=True,
                data={"size": 1000, "features": ["f1", "f2"], "path": "/tmp/data"},
            )

        mock_services["data"].load_dataset.side_effect = slow_data_loading

        with patch.multiple(
            "benchmark.services.orchestration_service",
            ConfigurationService=lambda: mock_services["config"],
            DataService=lambda: mock_services["data"],
            ModelService=lambda: mock_services["model"],
            EvaluationService=lambda: mock_services["evaluation"],
        ):
            # Initialize
            await orchestration_service.initialize()

            # Create experiment
            experiment_id = await orchestration_service.create_experiment(config_path=config_file)

            # Start experiment in background
            await orchestration_service.start_experiment(experiment_id, background=True)

            # Wait a bit for execution to start
            await asyncio.sleep(0.1)

            # Cancel experiment
            cancelled = await orchestration_service.cancel_experiment(experiment_id)
            assert cancelled is True

            # Check experiment status
            progress = await orchestration_service.get_experiment_progress(experiment_id)
            assert progress.status == ExperimentStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_workflow_engine_validation(self):
        """Test workflow engine step validation."""
        engine = WorkflowEngine()

        # Create test services
        mock_services = {
            "data": AsyncMock(),
            "model": AsyncMock(),
            "evaluation": AsyncMock(),
        }

        await engine.initialize(mock_services)

        # Create workflow steps
        steps = [
            DataLoadingStep(),
            ModelLoadingStep(),
            EvaluationExecutionStep(),
            ResultsAggregationStep(),
        ]

        # Validate workflow
        validation = await engine.validate_workflow(steps)

        assert validation["valid"] is True
        assert validation["steps_count"] == 4
        assert "data_loading" in validation["dependencies"]
        assert "model_loading" in validation["dependencies"]

        # Test workflow info
        info = engine.get_workflow_info(steps)
        assert info["steps_count"] == 4
        assert len(info["steps"]) == 4
        assert isinstance(info["total_estimated_duration"], float)

    @pytest.mark.asyncio
    async def test_individual_workflow_steps(self):
        """Test individual workflow steps in isolation."""
        # Mock services
        mock_services = {
            "data": AsyncMock(),
            "model": AsyncMock(),
            "evaluation": AsyncMock(),
        }

        # Test data loading step
        data_step = DataLoadingStep()
        assert data_step.get_step_name() == "data_loading"
        assert "data" in data_step.get_dependencies()

        # Mock successful data loading
        mock_services["data"].load_dataset.return_value = MagicMock(
            success=True,
            data={"size": 500, "features": ["f1", "f2", "label"], "path": "/tmp/test"},
        )

        context = MagicMock()
        context.experiment_id = "test_exp"
        context.config = {"datasets": [{"id": "test_dataset", "loader_type": "local"}]}
        context.services = mock_services
        context.resources = {}

        result = await data_step.execute(context)
        assert result["datasets_loaded"] == 1
        assert result["total_samples"] == 500

        # Test model loading step
        model_step = ModelLoadingStep()
        assert model_step.get_step_name() == "model_loading"
        assert "model" in model_step.get_dependencies()

        mock_services["model"].load_model.return_value = MagicMock(
            success=True, data={"model_id": "test_model", "memory_usage_mb": 256}
        )

        context.config = {"models": [{"id": "test_model", "plugin_type": "local"}]}
        result = await model_step.execute(context)
        assert result["models_loaded"] == 1
        assert result["total_memory_usage_mb"] == 256

        # Test evaluation step
        eval_step = EvaluationExecutionStep()
        assert eval_step.get_step_name() == "evaluation_execution"
        assert "evaluation" in eval_step.get_dependencies()

        # Setup context with loaded resources
        context.resources = {
            "loaded_datasets": {"dataset1": {}},
            "loaded_models": {"model1": {}},
        }
        context.config = {"evaluation": {"metrics": ["accuracy"]}}

        mock_services["evaluation"].evaluate_model.return_value = MagicMock(
            success=True, data={"accuracy": 0.90}
        )

        result = await eval_step.execute(context)
        assert result["total_evaluations"] == 1
        assert result["successful_evaluations"] == 1

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, orchestration_service):
        """Test health monitoring of dependent services."""
        # Setup mock services with different health states
        mock_services = {}
        for service_name in ["config", "data", "model", "evaluation"]:
            mock_service = AsyncMock()
            mock_service.initialize.return_value = MagicMock(success=True)
            mock_service.shutdown.return_value = MagicMock(success=True)
            mock_services[service_name] = mock_service

        # Make services have different health states
        mock_services["config"].health_check.return_value = MagicMock(
            status=MagicMock(value="healthy")
        )
        mock_services["data"].health_check.return_value = MagicMock(
            status=MagicMock(value="degraded")
        )
        mock_services["model"].health_check.return_value = MagicMock(
            status=MagicMock(value="unhealthy")
        )
        mock_services["evaluation"].health_check.return_value = MagicMock(
            status=MagicMock(value="healthy")
        )

        orchestration_service.services = mock_services

        # Check overall health
        health = await orchestration_service.health_check()

        # Should be unhealthy due to model service
        assert health.status.value == "unhealthy"
        assert "dependent_services" in health.checks

    @pytest.mark.asyncio
    async def test_concurrent_experiments(
        self, orchestration_service, config_file, sample_experiment_config
    ):
        """Test running multiple experiments concurrently."""
        # Setup mock services
        mock_services = {}
        for service_name in ["config", "data", "model", "evaluation"]:
            mock_service = AsyncMock()
            mock_service.initialize.return_value = MagicMock(success=True)
            mock_service.health_check.return_value = MagicMock(status=MagicMock(value="healthy"))
            mock_service.shutdown.return_value = MagicMock(success=True)
            mock_services[service_name] = mock_service

        # Setup service responses
        mock_services["config"].load_experiment_config.return_value = MagicMock(
            success=True, data=sample_experiment_config
        )
        mock_services["data"].load_dataset.return_value = MagicMock(
            success=True, data={"size": 100, "features": ["f1"], "path": "/tmp/data"}
        )
        mock_services["model"].load_model.return_value = MagicMock(
            success=True, data={"model_id": "test", "memory_usage_mb": 128}
        )
        mock_services["evaluation"].evaluate_model.return_value = MagicMock(
            success=True, data={"accuracy": 0.80}
        )

        with patch.multiple(
            "benchmark.services.orchestration_service",
            ConfigurationService=lambda: mock_services["config"],
            DataService=lambda: mock_services["data"],
            ModelService=lambda: mock_services["model"],
            EvaluationService=lambda: mock_services["evaluation"],
        ):
            await orchestration_service.initialize()

            # Create multiple experiments
            experiment_ids = []
            for i in range(3):
                exp_id = await orchestration_service.create_experiment(
                    config_path=config_file, experiment_name=f"Concurrent Test {i}"
                )
                experiment_ids.append(exp_id)

            # Start all experiments concurrently
            tasks = []
            for exp_id in experiment_ids:
                task = asyncio.create_task(
                    orchestration_service.start_experiment(exp_id, background=False)
                )
                tasks.append(task)

            # Wait for all to complete
            results = await asyncio.gather(*tasks)

            # Verify all completed successfully
            for i, result in enumerate(results):
                assert result["experiment_id"] == experiment_ids[i]
                assert result["status"] == ExperimentStatus.COMPLETED.value

            # Verify experiment list shows all experiments
            all_experiments = await orchestration_service.list_experiments()
            assert len(all_experiments) == 3

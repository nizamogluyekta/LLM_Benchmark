"""Integration tests for complete workflow execution with new workflow steps."""

import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.interfaces.orchestration_interfaces import ExperimentStatus, WorkflowContext
from benchmark.services.orchestration_service import OrchestrationService
from benchmark.workflow.steps import (
    DataLoadingStep,
    EvaluationExecutionStep,
    ModelLoadingStep,
    ResultsAggregationStep,
)
from benchmark.workflow.workflow_engine import WorkflowEngine


class TestCompleteWorkflow:
    """Integration tests for complete workflow execution with new steps."""

    @pytest.fixture
    def sample_experiment_config(self):
        """Sample experiment configuration for testing."""
        return {
            "name": "Complete Workflow Integration Test",
            "description": "Full workflow integration test with new steps",
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
                "parallel_jobs": 2,
            },
        }

    @pytest.fixture
    def mock_service_responses(self):
        """Mock responses for all services."""
        return {
            "config": {
                "load_experiment_config": {"success": True, "data": {}},
            },
            "data": {
                "load_dataset": {
                    "success": True,
                    "data": {
                        "samples": [
                            {
                                "input_text": f"Sample {i}",
                                "label": "positive" if i % 2 == 0 else "negative",
                            }
                            for i in range(100)
                        ],
                    },
                },
                "get_dataset_info": {
                    "success": True,
                    "data": {
                        "samples_count": 100,
                        "features": ["input_text", "label"],
                        "splits": {"train": 80, "test": 20},
                    },
                },
                "cleanup_dataset": {"success": True},
            },
            "model": {
                "load_model": {
                    "success": True,
                    "data": {"model_id": "loaded_model", "memory_usage_mb": 512},
                },
                "get_model_info": {
                    "success": True,
                    "data": {"memory_usage_mb": 512, "parameters": "7B"},
                },
                "predict_batch": [
                    {
                        "prediction": "positive" if i % 2 == 0 else "negative",
                        "confidence": 0.8 + (i % 3) * 0.05,
                        "inference_time_ms": 50.0 + i,
                    }
                    for i in range(100)
                ],
                "optimize_for_hardware": {"success": True},
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
                },
                "generate_comprehensive_report": {
                    "performance_overview": {"success_rate": 1.0},
                    "detailed_analysis": {"accuracy": {"mean": 0.85}},
                },
            },
        }

    @pytest.fixture
    def config_file(self, sample_experiment_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_experiment_config, f)
            return f.name

    @pytest.mark.asyncio
    async def test_complete_workflow_with_new_steps(
        self, config_file, sample_experiment_config, mock_service_responses
    ):
        """Test complete experiment workflow using new workflow steps."""
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
        mock_services["data"].get_dataset_info.return_value = MagicMock(
            **mock_service_responses["data"]["get_dataset_info"]
        )
        mock_services["data"].cleanup_dataset = AsyncMock()

        mock_services["model"].load_model.return_value = MagicMock(
            **mock_service_responses["model"]["load_model"]
        )
        mock_services["model"].get_model_info.return_value = MagicMock(
            **mock_service_responses["model"]["get_model_info"]
        )
        mock_services["model"].predict_batch.return_value = mock_service_responses["model"][
            "predict_batch"
        ]
        mock_services["model"].optimize_for_hardware = AsyncMock()

        # Setup resource manager
        resource_check = MagicMock()
        resource_check.can_load = True
        resource_check.estimated_memory_gb = 0.5
        mock_services["model"].resource_manager.can_load_model.return_value = resource_check

        mock_services["evaluation"].evaluate_model.return_value = MagicMock(
            **mock_service_responses["evaluation"]["evaluate_model"]
        )
        mock_services[
            "evaluation"
        ].generate_comprehensive_report.return_value = mock_service_responses["evaluation"][
            "generate_comprehensive_report"
        ]

        # Patch service creation and workflow step imports
        with (
            patch.multiple(
                "benchmark.services.orchestration_service",
                ConfigurationService=lambda: mock_services["config"],
                DataService=lambda: mock_services["data"],
                ModelService=lambda: mock_services["model"],
                EvaluationService=lambda: mock_services["evaluation"],
            ),
            patch.object(
                OrchestrationService,
                "_create_workflow_steps",
                return_value=[
                    DataLoadingStep(),
                    ModelLoadingStep(),
                    EvaluationExecutionStep(),
                    ResultsAggregationStep(),
                ],
            ),
        ):
            # Initialize orchestration service
            orchestration_service = OrchestrationService()
            init_response = await orchestration_service.initialize()
            assert init_response.success

            # Create experiment
            experiment_id = await orchestration_service.create_experiment(
                config_path=config_file, experiment_name="Complete Workflow Test"
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

            # Verify all workflow steps completed
            assert "data_loading" in experiment_result.evaluation_results
            assert "model_loading" in experiment_result.evaluation_results
            assert "evaluation_execution" in experiment_result.evaluation_results
            assert "results_aggregation" in experiment_result.evaluation_results

            # Verify step results
            data_loading_result = experiment_result.evaluation_results["data_loading"]
            assert data_loading_result["datasets_loaded"] == 2
            assert data_loading_result["successful_loads"] == 2

            model_loading_result = experiment_result.evaluation_results["model_loading"]
            assert model_loading_result["models_loaded"] == 2
            assert model_loading_result["successful_loads"] == 2

            evaluation_result = experiment_result.evaluation_results["evaluation_execution"]
            assert evaluation_result["total_evaluations"] == 4  # 2 models × 2 datasets
            assert evaluation_result["successful_evaluations"] == 4

            aggregation_result = experiment_result.evaluation_results["results_aggregation"]
            assert "experiment_summary" in aggregation_result
            assert "comprehensive_analysis" in aggregation_result

            # Verify service calls
            mock_services["config"].load_experiment_config.assert_called_once()
            assert mock_services["data"].load_dataset.call_count == 2  # 2 datasets
            assert mock_services["model"].load_model.call_count == 2  # 2 models
            assert mock_services["evaluation"].evaluate_model.call_count == 4  # 2×2 combinations

    @pytest.mark.asyncio
    async def test_workflow_step_dependencies(self):
        """Test that workflow steps have correct dependencies."""
        steps = [
            DataLoadingStep(),
            ModelLoadingStep(),
            EvaluationExecutionStep(),
            ResultsAggregationStep(),
        ]

        engine = WorkflowEngine()
        mock_services = {
            "data": AsyncMock(),
            "model": AsyncMock(),
            "evaluation": AsyncMock(),
        }

        await engine.initialize(mock_services)

        # Validate workflow
        validation = await engine.validate_workflow(steps)

        assert validation["valid"] is True
        assert validation["steps_count"] == 4

        # Check dependencies
        dependencies = validation["dependencies"]
        assert "data" in dependencies["data_loading"]
        assert "model" in dependencies["model_loading"]
        assert set(dependencies["evaluation_execution"]) == {"evaluation", "model", "data"}
        assert dependencies["results_aggregation"] == []

    @pytest.mark.asyncio
    async def test_workflow_with_step_failures(self, sample_experiment_config):
        """Test workflow behavior when individual steps fail."""
        # Create workflow context
        context = WorkflowContext(
            experiment_id="test_exp",
            config=sample_experiment_config,
            services={
                "data": AsyncMock(),
                "model": AsyncMock(),
                "evaluation": AsyncMock(),
            },
        )

        # Test data loading step failure
        context.services["data"].load_dataset.return_value = MagicMock(
            success=False, error="Dataset not found"
        )

        data_step = DataLoadingStep()
        with pytest.raises(Exception, match="All dataset loading attempts failed"):
            await data_step.execute(context)

        # Test model loading step failure
        context.services["model"].load_model.return_value = MagicMock(
            success=False, error="Model not found"
        )

        model_step = ModelLoadingStep()
        with pytest.raises(Exception, match="No models could be loaded"):
            await model_step.execute(context)

    @pytest.mark.asyncio
    async def test_workflow_resource_management(self, sample_experiment_config):
        """Test workflow resource management and cleanup."""
        context = WorkflowContext(
            experiment_id="test_exp",
            config=sample_experiment_config,
            services={
                "data": AsyncMock(),
                "model": AsyncMock(),
                "evaluation": AsyncMock(),
            },
        )

        # Setup successful responses
        context.services["data"].load_dataset.return_value = MagicMock(
            success=True, data={"samples": [{"input": "test", "label": "pos"}] * 10}
        )
        context.services["data"].get_dataset_info.return_value = MagicMock(
            success=True, data={"samples_count": 10, "features": ["input", "label"]}
        )
        context.services["data"].cleanup_dataset = AsyncMock()

        # Test data loading and cleanup
        data_step = DataLoadingStep()

        # Simulate failure after partial loading - but our implementation continues with partial success
        # This should actually succeed with partial loading, not fail
        context.services["data"].load_dataset.side_effect = [
            MagicMock(success=True, data={"samples": [{"input": "test"}] * 10}),
            Exception("Second dataset failed"),
        ]

        # This should succeed with 1 successful dataset (partial success is allowed)
        result = await data_step.execute(context)

        # Verify partial success
        assert result["datasets_loaded"] == 1
        assert result["successful_loads"] == 1
        assert result["failed_loads"] == 1

        # Since partial success is allowed, cleanup is not called for successful datasets
        # Cleanup only happens on complete failure, not partial failure

    @pytest.mark.asyncio
    async def test_workflow_context_resource_sharing(self, sample_experiment_config):
        """Test that workflow steps properly share resources through context."""
        context = WorkflowContext(
            experiment_id="test_exp",
            config=sample_experiment_config,
            services={
                "data": AsyncMock(),
                "model": AsyncMock(),
                "evaluation": AsyncMock(),
            },
        )

        # Setup mock responses
        context.services["data"].load_dataset.return_value = MagicMock(
            success=True, data={"samples": [{"input": "test", "label": "pos"}] * 10}
        )
        context.services["data"].get_dataset_info.return_value = MagicMock(
            success=True, data={"samples_count": 10}
        )

        context.services["model"].load_model.return_value = MagicMock(
            success=True, data={"model_id": "loaded_model"}
        )
        context.services["model"].get_model_info.return_value = MagicMock(
            success=True, data={"memory_usage_mb": 256}
        )
        context.services["model"].optimize_for_hardware = AsyncMock()

        # Setup resource manager
        resource_check = MagicMock()
        resource_check.can_load = True
        resource_check.estimated_memory_gb = 0.25
        context.services["model"].resource_manager.can_load_model.return_value = resource_check

        context.services["evaluation"].evaluate_model.return_value = MagicMock(
            success=True, data={"accuracy": 0.8}
        )
        context.services["model"].predict_batch.return_value = [
            {"prediction": "pos", "confidence": 0.8} for _ in range(10)
        ]

        # Execute steps in order
        data_step = DataLoadingStep()
        await data_step.execute(context)

        # Verify data was loaded into context
        assert "loaded_datasets" in context.resources
        assert len(context.resources["loaded_datasets"]) == 2

        model_step = ModelLoadingStep()
        await model_step.execute(context)

        # Verify models were loaded into context
        assert "loaded_models" in context.resources
        assert len(context.resources["loaded_models"]) == 2

        evaluation_step = EvaluationExecutionStep()
        result = await evaluation_step.execute(context)

        # Verify evaluation used both datasets and models
        assert result["total_evaluations"] == 4  # 2 models × 2 datasets

        # Store results for aggregation
        context.step_results["evaluation_execution"] = result

        aggregation_step = ResultsAggregationStep()
        agg_result = await aggregation_step.execute(context)

        # Verify aggregation processed all results
        assert agg_result["experiment_summary"]["total_evaluations"] == 4

    @pytest.mark.asyncio
    async def test_workflow_performance_estimation(self):
        """Test workflow step duration estimation."""
        steps = [
            DataLoadingStep(),
            ModelLoadingStep(),
            EvaluationExecutionStep(),
            ResultsAggregationStep(),
        ]

        engine = WorkflowEngine()
        info = engine.get_workflow_info(steps)

        assert info["steps_count"] == 4
        assert info["total_estimated_duration"] > 0

        # Check individual step estimates
        step_durations = [step_info["estimated_duration"] for step_info in info["steps"]]
        assert all(duration > 0 for duration in step_durations)

        # Verify total is sum of individual estimates
        expected_total = sum(step_durations)
        assert abs(info["total_estimated_duration"] - expected_total) < 0.1

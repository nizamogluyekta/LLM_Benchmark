"""Unit tests for workflow steps."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmark.core.base import ServiceResponse
from benchmark.interfaces.orchestration_interfaces import WorkflowContext
from benchmark.workflow.steps.data_loading import DataLoadingStep
from benchmark.workflow.steps.evaluation_execution import EvaluationExecutionStep
from benchmark.workflow.steps.model_loading import ModelLoadingStep
from benchmark.workflow.steps.results_aggregation import ResultsAggregationStep


class TestDataLoadingStep:
    """Test data loading workflow step."""

    @pytest.fixture
    def data_loading_step(self):
        """Create data loading step for testing."""
        return DataLoadingStep()

    @pytest.fixture
    def mock_context(self):
        """Create mock workflow context."""
        context = MagicMock(spec=WorkflowContext)
        context.experiment_id = "test_exp_123"
        context.services = {}
        context.config = {}
        context.resources = {}
        context.step_results = {}
        return context

    @pytest.fixture
    def mock_data_service(self):
        """Create mock data service."""
        service = AsyncMock()
        service.load_dataset.return_value = ServiceResponse(
            success=True,
            message="Dataset loaded successfully",
            data={"samples": [{"input": "test", "label": "positive"}] * 100},
        )
        service.get_dataset_info.return_value = ServiceResponse(
            success=True,
            message="Dataset info retrieved",
            data={"samples_count": 100, "features": ["input", "label"]},
        )
        service.cleanup_dataset = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_data_loading_step_basic(
        self, data_loading_step, mock_context, mock_data_service
    ):
        """Test basic data loading functionality."""
        mock_context.services["data"] = mock_data_service
        mock_context.config = {
            "datasets": [
                {"id": "test_dataset", "loader_type": "local", "config": {"path": "/tmp/test"}}
            ]
        }

        result = await data_loading_step.execute(mock_context)

        assert result["datasets_loaded"] == 1
        assert result["successful_loads"] == 1
        assert result["failed_loads"] == 0
        assert result["total_samples"] == 100
        assert "test_dataset" in mock_context.resources["loaded_datasets"]

        mock_data_service.load_dataset.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_loading_step_no_datasets(
        self, data_loading_step, mock_context, mock_data_service
    ):
        """Test data loading with no datasets configured."""
        mock_context.services["data"] = mock_data_service
        mock_context.config = {"datasets": []}

        result = await data_loading_step.execute(mock_context)

        assert result["datasets_loaded"] == 0
        assert result["loading_results"] == {}
        mock_data_service.load_dataset.assert_not_called()

    @pytest.mark.asyncio
    async def test_data_loading_step_service_failure(
        self, data_loading_step, mock_context, mock_data_service
    ):
        """Test data loading with service failure."""
        mock_context.services["data"] = mock_data_service
        mock_context.config = {"datasets": [{"id": "test_dataset", "loader_type": "local"}]}

        mock_data_service.load_dataset.return_value = ServiceResponse(
            success=False, message="Dataset not found", error="Dataset not found"
        )

        with pytest.raises(Exception, match="All dataset loading attempts failed"):
            await data_loading_step.execute(mock_context)

    @pytest.mark.asyncio
    async def test_data_loading_step_missing_service(self, data_loading_step, mock_context):
        """Test data loading without data service."""
        mock_context.services = {}

        with pytest.raises(Exception, match="Data service not available"):
            await data_loading_step.execute(mock_context)

    @pytest.mark.asyncio
    async def test_data_loading_step_multiple_datasets(
        self, data_loading_step, mock_context, mock_data_service
    ):
        """Test loading multiple datasets."""
        mock_context.services["data"] = mock_data_service
        mock_context.config = {
            "datasets": [
                {"id": "dataset1", "loader_type": "local"},
                {"id": "dataset2", "loader_type": "kaggle"},
            ]
        }

        result = await data_loading_step.execute(mock_context)

        assert result["datasets_loaded"] == 2
        assert result["successful_loads"] == 2
        assert result["total_samples"] == 200  # 2 datasets × 100 samples each
        assert mock_data_service.load_dataset.call_count == 2

    def test_data_loading_step_metadata(self, data_loading_step):
        """Test data loading step metadata."""
        assert data_loading_step.get_step_name() == "data_loading"
        assert data_loading_step.get_dependencies() == ["data"]
        assert data_loading_step.get_estimated_duration_seconds() == 180.0


class TestModelLoadingStep:
    """Test model loading workflow step."""

    @pytest.fixture
    def model_loading_step(self):
        """Create model loading step for testing."""
        return ModelLoadingStep()

    @pytest.fixture
    def mock_context(self):
        """Create mock workflow context."""
        context = MagicMock(spec=WorkflowContext)
        context.experiment_id = "test_exp_123"
        context.services = {}
        context.config = {}
        context.resources = {}
        context.step_results = {}
        return context

    @pytest.fixture
    def mock_model_service(self):
        """Create mock model service."""
        service = AsyncMock()
        service.load_model.return_value = ServiceResponse(
            success=True, message="Model loaded successfully", data={"model_id": "loaded_model_123"}
        )
        service.get_model_info.return_value = ServiceResponse(
            success=True,
            message="Model info retrieved",
            data={"memory_usage_mb": 512, "parameters": "7B"},
        )
        service.optimize_for_hardware = AsyncMock()

        # Mock resource manager
        resource_check = MagicMock()
        resource_check.can_load = True
        resource_check.estimated_memory_gb = 0.5
        service.resource_manager.can_load_model.return_value = resource_check

        return service

    @pytest.mark.asyncio
    async def test_model_loading_step_basic(
        self, model_loading_step, mock_context, mock_model_service
    ):
        """Test basic model loading functionality."""
        mock_context.services["model"] = mock_model_service
        mock_context.config = {
            "models": [
                {"id": "test_model", "plugin_type": "local", "config": {"path": "/tmp/model"}}
            ]
        }

        result = await model_loading_step.execute(mock_context)

        assert result["models_loaded"] == 1
        assert result["successful_loads"] == 1
        assert result["failed_loads"] == 0
        assert result["optimization_applied"] is True
        assert "test_model" in mock_context.resources["loaded_models"]

        mock_model_service.load_model.assert_called_once()
        mock_model_service.optimize_for_hardware.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_loading_step_no_models(
        self, model_loading_step, mock_context, mock_model_service
    ):
        """Test model loading with no models configured."""
        mock_context.services["model"] = mock_model_service
        mock_context.config = {"models": []}

        with pytest.raises(Exception, match="No models specified"):
            await model_loading_step.execute(mock_context)

    @pytest.mark.asyncio
    async def test_model_loading_step_service_failure(
        self, model_loading_step, mock_context, mock_model_service
    ):
        """Test model loading with service failure."""
        mock_context.services["model"] = mock_model_service
        mock_context.config = {"models": [{"id": "test_model", "plugin_type": "local"}]}

        mock_model_service.load_model.return_value = ServiceResponse(
            success=False, message="Model not found", error="Model not found"
        )

        with pytest.raises(Exception, match="No models could be loaded"):
            await model_loading_step.execute(mock_context)

    @pytest.mark.asyncio
    async def test_model_loading_step_partial_failure(
        self, model_loading_step, mock_context, mock_model_service
    ):
        """Test model loading with partial failures."""
        mock_context.services["model"] = mock_model_service
        mock_context.config = {
            "models": [
                {"id": "model1", "plugin_type": "local"},
                {"id": "model2", "plugin_type": "api"},
            ]
        }

        # First model succeeds, second fails
        responses = [
            ServiceResponse(
                success=True, message="Model 1 loaded", data={"model_id": "loaded_model_1"}
            ),
            ServiceResponse(success=False, message="Model 2 failed", error="Model 2 failed"),
        ]
        mock_model_service.load_model.side_effect = responses

        result = await model_loading_step.execute(mock_context)

        assert result["models_loaded"] == 1
        assert result["successful_loads"] == 1
        assert result["failed_loads"] == 1
        assert mock_model_service.load_model.call_count == 2

    def test_model_loading_step_metadata(self, model_loading_step):
        """Test model loading step metadata."""
        assert model_loading_step.get_step_name() == "model_loading"
        assert model_loading_step.get_dependencies() == ["model"]
        assert model_loading_step.get_estimated_duration_seconds() == 360.0


class TestEvaluationExecutionStep:
    """Test evaluation execution workflow step."""

    @pytest.fixture
    def evaluation_step(self):
        """Create evaluation execution step for testing."""
        return EvaluationExecutionStep()

    @pytest.fixture
    def mock_context(self):
        """Create mock workflow context."""
        context = MagicMock(spec=WorkflowContext)
        context.experiment_id = "test_exp_123"
        context.services = {}
        context.config = {}
        context.resources = {
            "loaded_datasets": {"dataset1": {"samples": [{"input": "test1", "label": "pos"}] * 10}},
            "loaded_models": {"model1": {"model_id": "loaded_model_1"}},
        }
        context.step_results = {}
        return context

    @pytest.fixture
    def mock_evaluation_service(self):
        """Create mock evaluation service."""
        service = AsyncMock()
        service.evaluate_model.return_value = ServiceResponse(
            success=True,
            message="Evaluation completed",
            data={"accuracy": 0.85, "precision": 0.8, "recall": 0.9},
        )
        return service

    @pytest.fixture
    def mock_model_service(self):
        """Create mock model service."""
        service = AsyncMock()
        service.predict_batch.return_value = [
            {"prediction": "pos", "confidence": 0.9} for _ in range(10)
        ]
        return service

    @pytest.mark.asyncio
    async def test_evaluation_execution_basic(
        self, evaluation_step, mock_context, mock_evaluation_service, mock_model_service
    ):
        """Test basic evaluation execution."""
        mock_context.services["evaluation"] = mock_evaluation_service
        mock_context.services["model"] = mock_model_service
        mock_context.config = {"evaluation": {"metrics": ["accuracy", "precision"]}}

        result = await evaluation_step.execute(mock_context)

        assert result["total_evaluations"] == 1
        assert result["successful_evaluations"] == 1
        assert result["failed_evaluations"] == 0
        assert result["success_rate"] == 1.0

        mock_evaluation_service.evaluate_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_execution_no_datasets(
        self, evaluation_step, mock_context, mock_evaluation_service
    ):
        """Test evaluation with no datasets."""
        mock_context.services["evaluation"] = mock_evaluation_service
        mock_context.resources["loaded_datasets"] = {}

        with pytest.raises(Exception, match="No datasets available"):
            await evaluation_step.execute(mock_context)

    @pytest.mark.asyncio
    async def test_evaluation_execution_no_models(
        self, evaluation_step, mock_context, mock_evaluation_service
    ):
        """Test evaluation with no models."""
        mock_context.services["evaluation"] = mock_evaluation_service
        mock_context.resources["loaded_models"] = {}

        with pytest.raises(Exception, match="No models available"):
            await evaluation_step.execute(mock_context)

    @pytest.mark.asyncio
    async def test_evaluation_execution_multiple_combinations(
        self, evaluation_step, mock_context, mock_evaluation_service, mock_model_service
    ):
        """Test evaluation with multiple model-dataset combinations."""
        mock_context.services["evaluation"] = mock_evaluation_service
        mock_context.services["model"] = mock_model_service
        mock_context.resources["loaded_datasets"] = {
            "dataset1": {"samples": [{"input": "test1", "label": "pos"}] * 5},
            "dataset2": {"samples": [{"input": "test2", "label": "neg"}] * 5},
        }
        mock_context.resources["loaded_models"] = {
            "model1": {"model_id": "loaded_model_1"},
            "model2": {"model_id": "loaded_model_2"},
        }
        mock_context.config = {"evaluation": {"metrics": ["accuracy"]}}

        result = await evaluation_step.execute(mock_context)

        assert result["total_evaluations"] == 4  # 2 models × 2 datasets
        assert mock_evaluation_service.evaluate_model.call_count == 4

    def test_evaluation_execution_step_metadata(self, evaluation_step):
        """Test evaluation execution step metadata."""
        assert evaluation_step.get_step_name() == "evaluation_execution"
        assert evaluation_step.get_dependencies() == ["evaluation", "model", "data"]
        assert evaluation_step.get_estimated_duration_seconds() == 900.0


class TestResultsAggregationStep:
    """Test results aggregation workflow step."""

    @pytest.fixture
    def aggregation_step(self):
        """Create results aggregation step for testing."""
        return ResultsAggregationStep()

    @pytest.fixture
    def mock_context(self):
        """Create mock workflow context."""
        context = MagicMock(spec=WorkflowContext)
        context.experiment_id = "test_exp_123"
        context.services = {}
        context.config = {"name": "Test Experiment", "models": ["model1"], "datasets": ["dataset1"]}
        context.resources = {}
        context.step_results = {
            "evaluation_execution": {
                "results": {
                    "model1_dataset1": {
                        "status": "success",
                        "model_id": "model1",
                        "dataset_id": "dataset1",
                        "evaluation_data": {"accuracy": 0.85, "precision": 0.8},
                    }
                }
            }
        }
        return context

    @pytest.fixture
    def mock_evaluation_service(self):
        """Create mock evaluation service."""
        service = AsyncMock()
        service.generate_comprehensive_report.return_value = {
            "performance_overview": {"success_rate": 1.0},
            "detailed_analysis": {"accuracy": {"mean": 0.85}},
        }
        return service

    @pytest.mark.asyncio
    async def test_results_aggregation_basic(
        self, aggregation_step, mock_context, mock_evaluation_service
    ):
        """Test basic results aggregation."""
        mock_context.services["evaluation"] = mock_evaluation_service

        result = await aggregation_step.execute(mock_context)

        assert "experiment_summary" in result
        assert "comprehensive_analysis" in result
        assert "model_comparisons" in result
        assert "summary_statistics" in result
        assert "insights_and_recommendations" in result

        assert result["experiment_summary"]["experiment_id"] == "test_exp_123"
        assert result["experiment_summary"]["total_evaluations"] == 1

    @pytest.mark.asyncio
    async def test_results_aggregation_no_results(self, aggregation_step, mock_context):
        """Test results aggregation with no evaluation results."""
        mock_context.step_results = {"evaluation_execution": {"results": {}}}

        result = await aggregation_step.execute(mock_context)

        assert result["message"] == "No results to aggregate"
        assert result["summary"] == {}

    @pytest.mark.asyncio
    async def test_results_aggregation_comprehensive_analysis(self, aggregation_step, mock_context):
        """Test comprehensive analysis generation."""
        mock_context.step_results = {
            "evaluation_execution": {
                "results": {
                    "model1_dataset1": {
                        "status": "success",
                        "model_id": "model1",
                        "dataset_id": "dataset1",
                        "evaluation_data": {"accuracy": 0.95, "f1_score": 0.93},
                    },
                    "model2_dataset1": {
                        "status": "success",
                        "model_id": "model2",
                        "dataset_id": "dataset1",
                        "evaluation_data": {"accuracy": 0.87, "f1_score": 0.85},
                    },
                }
            }
        }

        result = await aggregation_step.execute(mock_context)

        # Check model comparisons
        model_comparisons = result["model_comparisons"]
        assert "model_summaries" in model_comparisons
        assert "best_performing_model" in model_comparisons
        assert len(model_comparisons["model_summaries"]) == 2

        # Check summary statistics
        summary_stats = result["summary_statistics"]
        assert "metrics_summary" in summary_stats
        assert "accuracy" in summary_stats["metrics_summary"]
        assert summary_stats["metrics_summary"]["accuracy"]["count"] == 2

        # Check insights
        insights = result["insights_and_recommendations"]
        assert isinstance(insights, list)
        assert len(insights) > 0

    def test_results_aggregation_step_metadata(self, aggregation_step):
        """Test results aggregation step metadata."""
        assert aggregation_step.get_step_name() == "results_aggregation"
        assert aggregation_step.get_dependencies() == []
        assert aggregation_step.get_estimated_duration_seconds() == 60.0

    def test_calculate_std(self, aggregation_step):
        """Test standard deviation calculation."""
        # Test with normal values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std = aggregation_step._calculate_std(values)
        assert abs(std - 1.5811388300841898) < 0.001

        # Test with single value
        values = [5.0]
        std = aggregation_step._calculate_std(values)
        assert std == 0.0

        # Test with empty list
        values = []
        std = aggregation_step._calculate_std(values)
        assert std == 0.0

    def test_calculate_overall_model_score(self, aggregation_step):
        """Test overall model score calculation."""
        # Test with priority metrics
        metrics_summary = {
            "accuracy": {"average": 0.85},
            "f1_score": {"average": 0.82},
            "precision": {"average": 0.80},
        }
        score = aggregation_step._calculate_overall_model_score(metrics_summary)
        assert abs(score - 0.8233333333333334) < 0.001

        # Test with no metrics
        score = aggregation_step._calculate_overall_model_score({})
        assert score == 0.0

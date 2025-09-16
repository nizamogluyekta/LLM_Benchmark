"""
Comprehensive test suite for EvaluationWorkflow.

Tests workflow orchestration, progress tracking, state management,
and end-to-end workflow execution.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmark.evaluation.evaluation_workflow import (
    EvaluationWorkflow,
    WorkflowProgress,
    WorkflowState,
    WorkflowStep,
)
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage
from benchmark.services.evaluation_service import EvaluationService


class TestWorkflowProgress:
    """Test WorkflowProgress functionality."""

    def test_progress_initialization(self):
        """Test workflow progress initialization."""
        progress = WorkflowProgress("test_workflow", 5)

        assert progress.workflow_id == "test_workflow"
        assert progress.total_steps == 5
        assert progress.completed_steps == 0
        assert progress.current_step is None
        assert progress.state == WorkflowState.PENDING
        assert progress.get_progress_percentage() == 0.0

    def test_step_lifecycle(self):
        """Test complete step lifecycle."""
        progress = WorkflowProgress("test_workflow", 3)

        # Start step
        progress.start_step(WorkflowStep.INITIALIZATION, 2)
        assert progress.current_step == WorkflowStep.INITIALIZATION
        assert "initialization" in progress.step_progress
        assert progress.step_progress["initialization"]["status"] == "running"
        assert progress.step_progress["initialization"]["substeps_total"] == 2

        # Update progress
        progress.update_step_progress(1)
        assert progress.step_progress["initialization"]["substeps_completed"] == 1

        # Complete step
        progress.complete_step(WorkflowStep.INITIALIZATION, {"result": "success"})
        assert progress.step_progress["initialization"]["status"] == "completed"
        assert progress.completed_steps == 1
        assert progress.get_progress_percentage() == pytest.approx(33.33, rel=1e-2)

    def test_step_failure(self):
        """Test step failure handling."""
        progress = WorkflowProgress("test_workflow", 3)

        progress.start_step(WorkflowStep.MODEL_LOADING)
        progress.fail_step(WorkflowStep.MODEL_LOADING, "Model not found")

        assert progress.step_progress["model_loading"]["status"] == "failed"
        assert "Model not found" in progress.step_progress["model_loading"]["errors"]
        assert progress.state == WorkflowState.FAILED
        assert "model_loading: Model not found" in progress.errors

    def test_estimated_completion(self):
        """Test completion time estimation."""
        progress = WorkflowProgress("test_workflow", 4)

        # No progress yet
        assert progress.get_estimated_completion() is None

        # Simulate some progress
        progress.completed_steps = 2
        estimated = progress.get_estimated_completion()

        assert estimated is not None
        assert estimated > progress.start_time


class TestEvaluationWorkflow:
    """Test EvaluationWorkflow functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def mock_evaluation_service(self, temp_storage):
        """Create mock evaluation service."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage

        # Mock async methods
        service.list_available_models = AsyncMock(return_value=["model_a", "model_b", "model_c"])
        service.prepare_model = AsyncMock(return_value={"status": "ready"})
        service.prepare_task_data = AsyncMock(return_value={"data_ready": True})
        service.run_evaluation = AsyncMock(return_value=self._create_mock_result())

        return service

    @pytest.fixture
    def workflow(self, mock_evaluation_service):
        """Create workflow instance."""
        return EvaluationWorkflow(mock_evaluation_service)

    def _create_mock_result(self, model_name="test_model", task_type="test_task"):
        """Create mock evaluation result."""
        return EvaluationResult(
            evaluation_id="eval_001",
            model_name=model_name,
            task_type=task_type,
            dataset_name="test_dataset",
            metrics={"accuracy": 0.85, "f1_score": 0.82},
            timestamp=datetime.now(),
            configuration={"learning_rate": 0.001},
            raw_responses=[],
            processing_time=10.5,
        )

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.service is not None
        assert isinstance(workflow.workflow_states, dict)
        assert workflow.analyzer is not None
        assert workflow.comparison_engine is not None

    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_workflow(self, workflow):
        """Test complete comprehensive evaluation workflow."""
        config = {
            "models": ["model_a", "model_b"],
            "tasks": ["task_1", "task_2"],
            "evaluation_params": {"batch_size": 16},
        }

        workflow_id = await workflow.run_comprehensive_evaluation(config)

        assert workflow_id is not None
        assert workflow_id in workflow.workflow_states

        progress = workflow.workflow_states[workflow_id]
        assert progress.state == WorkflowState.COMPLETED
        assert progress.completed_steps == progress.total_steps
        assert progress.end_time is not None
        assert "evaluation_results" in progress.results

        # Verify service calls
        workflow.service.list_available_models.assert_called_once()
        assert workflow.service.prepare_model.call_count == 2  # 2 models
        assert workflow.service.prepare_task_data.call_count == 2  # 2 tasks
        assert workflow.service.run_evaluation.call_count == 4  # 2 models × 2 tasks

    @pytest.mark.asyncio
    async def test_model_comparison_workflow(self, workflow):
        """Test model comparison workflow."""
        models = ["model_a", "model_b", "model_c"]
        tasks = ["classification", "regression"]

        result = await workflow.run_model_comparison(models, tasks)

        assert result["models_compared"] == models
        assert result["tasks_evaluated"] == tasks
        assert "comparison_results" in result
        assert "analysis_summary" in result
        assert result["evaluation_count"] == 6  # 3 models × 2 tasks

    @pytest.mark.asyncio
    async def test_baseline_evaluation_workflow(self, workflow):
        """Test baseline evaluation workflow."""
        # Setup mock baseline detection
        workflow.service.storage.query_results = MagicMock(
            return_value=[
                self._create_mock_result("baseline_1", "task_1"),
                self._create_mock_result("baseline_2", "task_1"),
            ]
        )

        result = await workflow.run_baseline_evaluation(
            "new_model", baseline_models=["baseline_1", "baseline_2"], config={"tasks": ["task_1"]}
        )

        assert result["new_model"] == "new_model"
        assert result["baseline_models"] == ["baseline_1", "baseline_2"]
        assert "baseline_comparison" in result
        assert "analysis_results" in result
        assert "performance_summary" in result

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow):
        """Test workflow error handling."""
        # Make model preparation fail
        workflow.service.prepare_model = AsyncMock(side_effect=Exception("Model loading failed"))

        config = {"models": ["failing_model"], "tasks": ["task_1"]}

        with pytest.raises(Exception, match="Model loading failed"):
            await workflow.run_comprehensive_evaluation(config)

        # Check workflow state reflects the failure
        workflow_id = list(workflow.workflow_states.keys())[0]
        progress = workflow.workflow_states[workflow_id]
        assert progress.state == WorkflowState.FAILED
        assert len(progress.errors) > 0

    def test_workflow_progress_tracking(self, workflow):
        """Test workflow progress tracking."""
        # Create a workflow with some progress
        workflow_id = "test_workflow_123"
        progress = WorkflowProgress(workflow_id, 5)
        progress.state = WorkflowState.RUNNING
        progress.completed_steps = 2
        progress.start_step(WorkflowStep.EVALUATION_EXECUTION)
        workflow.workflow_states[workflow_id] = progress

        status = workflow.track_workflow_progress(workflow_id)

        assert status["workflow_id"] == workflow_id
        assert status["state"] == "running"
        assert status["progress_percentage"] == 40.0
        assert status["completed_steps"] == 2
        assert status["total_steps"] == 5
        assert status["current_step"] == "evaluation_execution"

    def test_workflow_progress_tracking_nonexistent(self, workflow):
        """Test tracking nonexistent workflow."""
        status = workflow.track_workflow_progress("nonexistent")
        assert "error" in status

    def test_workflow_cancellation(self, workflow):
        """Test workflow cancellation."""
        workflow_id = "test_workflow_456"
        progress = WorkflowProgress(workflow_id, 3)
        progress.state = WorkflowState.RUNNING
        workflow.workflow_states[workflow_id] = progress

        success = workflow.cancel_workflow(workflow_id)
        assert success
        assert progress.state == WorkflowState.CANCELLED
        assert progress.end_time is not None

        # Cannot cancel already cancelled workflow
        success = workflow.cancel_workflow(workflow_id)
        assert not success

    def test_workflow_pause_resume(self, workflow):
        """Test workflow pause and resume."""
        workflow_id = "test_workflow_789"
        progress = WorkflowProgress(workflow_id, 3)
        progress.state = WorkflowState.RUNNING
        workflow.workflow_states[workflow_id] = progress

        # Pause workflow
        success = workflow.pause_workflow(workflow_id)
        assert success
        assert progress.state == WorkflowState.PAUSED

        # Resume workflow
        success = workflow.resume_workflow(workflow_id)
        assert success
        assert progress.state == WorkflowState.RUNNING

        # Cannot pause non-running workflow
        progress.state = WorkflowState.COMPLETED
        success = workflow.pause_workflow(workflow_id)
        assert not success

    def test_list_workflows(self, workflow):
        """Test listing workflows with filtering."""
        # Create several workflows in different states
        workflows_data = [
            ("wf_1", WorkflowState.RUNNING),
            ("wf_2", WorkflowState.COMPLETED),
            ("wf_3", WorkflowState.RUNNING),
            ("wf_4", WorkflowState.FAILED),
        ]

        for wf_id, state in workflows_data:
            progress = WorkflowProgress(wf_id, 3)
            progress.state = state
            workflow.workflow_states[wf_id] = progress

        # List all workflows
        all_workflows = workflow.list_workflows()
        assert len(all_workflows) == 4

        # Filter by state
        running_workflows = workflow.list_workflows(state_filter=WorkflowState.RUNNING)
        assert len(running_workflows) == 2
        assert all(wf["state"] == "running" for wf in running_workflows)

        # Test limit
        limited_workflows = workflow.list_workflows(limit=2)
        assert len(limited_workflows) == 2

    def test_cleanup_completed_workflows(self, workflow):
        """Test cleanup of old completed workflows."""
        # Create workflows with different completion times
        old_time = datetime.now() - timedelta(hours=25)
        recent_time = datetime.now() - timedelta(hours=1)

        workflows_data = [
            ("old_completed", WorkflowState.COMPLETED, old_time),
            ("recent_completed", WorkflowState.COMPLETED, recent_time),
            ("old_failed", WorkflowState.FAILED, old_time),
            ("running", WorkflowState.RUNNING, None),
        ]

        for wf_id, state, end_time in workflows_data:
            progress = WorkflowProgress(wf_id, 3)
            progress.state = state
            progress.end_time = end_time
            workflow.workflow_states[wf_id] = progress

        # Cleanup workflows older than 24 hours
        cleaned_count = workflow.cleanup_completed_workflows(older_than_hours=24)

        assert cleaned_count == 2  # old_completed and old_failed
        assert "old_completed" not in workflow.workflow_states
        assert "old_failed" not in workflow.workflow_states
        assert "recent_completed" in workflow.workflow_states
        assert "running" in workflow.workflow_states

    @pytest.mark.asyncio
    async def test_validation_methods(self, workflow):
        """Test configuration validation methods."""
        # Test valid configuration
        valid_config = {"models": ["model_a", "model_b"], "tasks": ["task_1", "task_2"]}

        await workflow._validate_workflow_config(valid_config)  # Should not raise

        # Test invalid configuration - missing models
        invalid_config = {"tasks": ["task_1"]}

        with pytest.raises(ValueError, match="Missing required field: models"):
            await workflow._validate_workflow_config(invalid_config)

        # Test invalid configuration - unavailable model
        workflow.service.list_available_models = AsyncMock(return_value=["model_a"])
        unavailable_config = {"models": ["model_a", "unavailable_model"], "tasks": ["task_1"]}

        with pytest.raises(ValueError, match="Model unavailable_model not available"):
            await workflow._validate_workflow_config(unavailable_config)

    @pytest.mark.asyncio
    async def test_comparison_validation(self, workflow):
        """Test model comparison validation."""
        # Valid comparison
        await workflow._validate_comparison_config(["model_a", "model_b"], ["task_1"], {})

        # Invalid - too few models
        with pytest.raises(ValueError, match="Need at least 2 models"):
            await workflow._validate_comparison_config(["model_a"], ["task_1"], {})

        # Invalid - no tasks
        with pytest.raises(ValueError, match="Need at least 1 task"):
            await workflow._validate_comparison_config(["model_a", "model_b"], [], {})

    @pytest.mark.asyncio
    async def test_baseline_detection(self, workflow):
        """Test automatic baseline model detection."""
        # Mock recent evaluation results
        mock_results = [
            self._create_mock_result("model_frequent", "task_1"),
            self._create_mock_result("model_frequent", "task_2"),
            self._create_mock_result("model_frequent", "task_1"),
            self._create_mock_result("model_frequent", "task_2"),
            self._create_mock_result("model_frequent", "task_1"),
            self._create_mock_result("model_rare", "task_1"),
            self._create_mock_result("model_medium", "task_1"),
            self._create_mock_result("model_medium", "task_2"),
            self._create_mock_result("model_medium", "task_1"),
        ]

        workflow.service.storage.query_results = MagicMock(return_value=mock_results)

        baselines = await workflow._detect_baseline_models({"min_baseline_evaluations": 3})

        # Should detect models with sufficient evaluation history
        assert "model_frequent" in baselines  # 5 evaluations
        assert "model_medium" in baselines  # 3 evaluations
        assert "model_rare" not in baselines  # Only 1 evaluation

    @pytest.mark.asyncio
    async def test_step_execution_methods(self, workflow):
        """Test individual step execution methods."""
        # Test model preparation
        models_info = await workflow._prepare_models(
            ["model_a", "model_b"], WorkflowProgress("test", 5)
        )
        assert len(models_info) == 2
        assert "model_a" in models_info
        assert "model_b" in models_info

        # Test data preparation
        data_info = await workflow._prepare_datasets(
            ["task_1", "task_2"], WorkflowProgress("test", 5)
        )
        assert len(data_info) == 2
        assert "task_1" in data_info
        assert "task_2" in data_info

    @pytest.mark.asyncio
    async def test_evaluation_execution(self, workflow):
        """Test evaluation execution with different configurations."""
        config = {
            "models": ["model_a", "model_b"],
            "tasks": ["task_1"],
            "evaluation_params": {"batch_size": 32},
        }

        results = await workflow._run_evaluations(config, WorkflowProgress("test", 5))

        assert len(results) == 2  # 2 models × 1 task
        assert workflow.service.run_evaluation.call_count == 2

        # Verify evaluation config passed correctly
        call_args = workflow.service.run_evaluation.call_args_list
        for call in call_args:
            config_arg = call[0][0]  # First positional argument
            assert "model_name" in config_arg
            assert "task_type" in config_arg
            assert config_arg.get("batch_size") == 32

    @pytest.mark.asyncio
    async def test_analysis_and_comparison_methods(self, workflow):
        """Test analysis and comparison step methods."""
        # Create mock evaluation results
        mock_results = [
            self._create_mock_result("model_a", "task_1"),
            self._create_mock_result("model_b", "task_1"),
            self._create_mock_result("model_a", "task_2"),
        ]

        # Store results in mock storage for analysis
        for result in mock_results:
            workflow.service.storage.store_evaluation_result(result)

        # Test analysis
        analysis_results = await workflow._perform_analysis(
            mock_results, {"models": ["model_a", "model_b"]}, WorkflowProgress("test", 5)
        )

        assert "model_a" in analysis_results
        assert "model_b" in analysis_results
        assert "overall" in analysis_results

        # Test comparisons
        comparison_results = await workflow._perform_comparisons(
            {"models": ["model_a", "model_b"], "metrics": ["accuracy"]}, WorkflowProgress("test", 5)
        )

        assert "multi_model" in comparison_results

    @pytest.mark.asyncio
    async def test_report_generation(self, workflow):
        """Test comprehensive report generation."""
        mock_results = [self._create_mock_result("model_a", "task_1")]
        analysis_results = {
            "overall": {"summary": {}, "bottlenecks": {"recommendations": ["test rec"]}}
        }
        comparison_results = {
            "multi_model": {
                "ranking_analysis": {
                    "rankings": [{"model_name": "model_a", "mean_performance": 0.85}]
                }
            }
        }
        config = {"models": ["model_a"], "tasks": ["task_1"]}

        report = await workflow._generate_comprehensive_report(
            mock_results, analysis_results, comparison_results, config
        )

        assert "evaluation_summary" in report
        assert "performance_analysis" in report
        assert "model_comparisons" in report
        assert "recommendations" in report
        assert "next_steps" in report
        assert report["evaluation_summary"]["total_evaluations"] == 1

    @pytest.mark.asyncio
    async def test_workflow_with_custom_id(self, workflow):
        """Test workflow execution with custom workflow ID."""
        config = {"models": ["model_a"], "tasks": ["task_1"]}

        custom_id = "custom_workflow_123"
        returned_id = await workflow.run_comprehensive_evaluation(config, workflow_id=custom_id)

        assert returned_id == custom_id
        assert custom_id in workflow.workflow_states

    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, workflow):
        """Test running multiple workflows concurrently."""
        config1 = {"models": ["model_a"], "tasks": ["task_1"]}
        config2 = {"models": ["model_b"], "tasks": ["task_2"]}

        # Run workflows concurrently
        task1 = asyncio.create_task(workflow.run_comprehensive_evaluation(config1))
        task2 = asyncio.create_task(workflow.run_comprehensive_evaluation(config2))

        workflow_id1, workflow_id2 = await asyncio.gather(task1, task2)

        assert workflow_id1 != workflow_id2
        assert workflow_id1 in workflow.workflow_states
        assert workflow_id2 in workflow.workflow_states

        # Both should be completed
        assert workflow.workflow_states[workflow_id1].state == WorkflowState.COMPLETED
        assert workflow.workflow_states[workflow_id2].state == WorkflowState.COMPLETED

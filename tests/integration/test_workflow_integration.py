"""
Integration tests for evaluation workflows.

Tests end-to-end workflow execution, integration between components,
and real-world scenario validation.
"""

import asyncio
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmark.evaluation import (
    BatchEvaluator,
    EvaluationWorkflow,
    ResourceLimits,
)
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage
from benchmark.services.evaluation_service import EvaluationService


class TestWorkflowIntegration:
    """Test complete workflow integration scenarios."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultsStorage(temp_dir)
            yield storage

    @pytest.fixture
    def mock_evaluation_service(self, temp_storage):
        """Create realistic mock evaluation service."""
        service = MagicMock(spec=EvaluationService)
        service.storage = temp_storage

        # Mock available models
        service.list_available_models = AsyncMock(
            return_value=["gpt_4", "claude_3", "llama_2", "bert_large", "roberta_base"]
        )

        # Mock model preparation
        service.prepare_model = AsyncMock(return_value={"status": "ready", "memory_usage": "2GB"})

        # Mock task data preparation
        service.prepare_task_data = AsyncMock(return_value={"samples": 1000, "status": "prepared"})

        # Mock evaluation execution with realistic variation
        async def mock_evaluation(config):
            # Simulate different performance levels for different models
            model_performance = {
                "gpt_4": 0.92,
                "claude_3": 0.90,
                "llama_2": 0.85,
                "bert_large": 0.82,
                "roberta_base": 0.80,
            }

            base_accuracy = model_performance.get(config["model_name"], 0.75)

            # Add some random variation
            import random

            variation = random.uniform(-0.05, 0.05)
            accuracy = max(0.5, min(0.98, base_accuracy + variation))

            # Simulate processing time variation
            processing_time = random.uniform(8.0, 15.0)

            result = EvaluationResult(
                evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                model_name=config["model_name"],
                task_type=config["task_type"],
                dataset_name=config.get("dataset_name", "default_dataset"),
                metrics={
                    "accuracy": accuracy,
                    "f1_score": accuracy * 0.95,
                    "precision": accuracy * 0.98,
                    "recall": accuracy * 0.92,
                    "latency": processing_time * 0.1,
                },
                timestamp=datetime.now(),
                configuration=config.get("evaluation_params", {}),
                raw_responses=[],
                processing_time=processing_time,
                experiment_name=config.get("experiment_name", "integration_test"),
                tags=["integration", "test"],
            )

            # Store the result
            temp_storage.store_evaluation_result(result)
            return result

        service.run_evaluation = mock_evaluation

        return service

    @pytest.fixture
    def workflow(self, mock_evaluation_service):
        """Create workflow instance."""
        return EvaluationWorkflow(mock_evaluation_service)

    @pytest.fixture
    def batch_evaluator(self, mock_evaluation_service):
        """Create batch evaluator instance."""
        limits = ResourceLimits(
            max_concurrent_evaluations=4, max_memory_mb=8000, timeout_seconds=60
        )
        return BatchEvaluator(mock_evaluation_service, limits)

    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_workflow_integration(self, workflow):
        """Test complete comprehensive evaluation workflow."""
        config = {
            "models": ["gpt_4", "claude_3", "llama_2"],
            "tasks": ["text_classification", "sentiment_analysis", "question_answering"],
            "datasets": ["imdb", "sst", "squad"],
            "evaluation_params": {"batch_size": 32, "max_length": 512, "temperature": 0.7},
            "metrics": ["accuracy", "f1_score", "precision", "recall"],
        }

        # Run comprehensive evaluation
        workflow_id = await workflow.run_comprehensive_evaluation(config)

        # Verify workflow completion
        assert workflow_id is not None
        progress = workflow.track_workflow_progress(workflow_id)

        assert progress["state"] == "completed"
        assert progress["progress_percentage"] == 100.0
        assert progress["errors"] == []

        # Verify results were stored
        results = workflow.service.storage.query_results()
        expected_count = len(config["models"]) * len(config["tasks"])
        assert len(results) == expected_count

        # Verify all models and tasks were covered
        model_names = {r.model_name for r in results}
        task_types = {r.task_type for r in results}

        assert model_names == set(config["models"])
        assert task_types == set(config["tasks"])

        # Check that all results have required metrics
        for result in results:
            assert "accuracy" in result.metrics
            assert "f1_score" in result.metrics
            assert result.metrics["accuracy"] > 0.5
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_model_comparison_workflow_integration(self, workflow):
        """Test model comparison workflow integration."""
        models = ["gpt_4", "claude_3", "llama_2", "bert_large"]
        tasks = ["text_classification", "sentiment_analysis"]

        comparison_result = await workflow.run_model_comparison(
            models, tasks, config={"metrics": ["accuracy", "f1_score"]}
        )

        # Verify comparison structure
        assert comparison_result["models_compared"] == models
        assert comparison_result["tasks_evaluated"] == tasks
        assert comparison_result["evaluation_count"] == len(models) * len(tasks)

        # Verify comparison results contain statistical analysis
        comparison_data = comparison_result["comparison_results"]
        assert "overall" in comparison_data
        assert "by_task" in comparison_data

        # Check that rankings are provided
        overall_comparison = comparison_data["overall"]
        assert "ranking_analysis" in overall_comparison
        assert "statistical_comparison" in overall_comparison

        rankings = overall_comparison["ranking_analysis"]["rankings"]
        assert len(rankings) == len(models)

        # Verify rankings are properly ordered
        for i in range(1, len(rankings)):
            assert rankings[i - 1]["mean_performance"] >= rankings[i]["mean_performance"]

    @pytest.mark.asyncio
    async def test_baseline_evaluation_workflow_integration(self, workflow):
        """Test baseline evaluation workflow integration."""
        # Pre-populate storage with baseline model results
        baseline_models = ["bert_large", "roberta_base"]
        baseline_tasks = ["text_classification", "sentiment_analysis"]

        # Create some historical baseline results
        for model in baseline_models:
            for task in baseline_tasks:
                for _ in range(5):  # 5 evaluations per model/task
                    config = {"model_name": model, "task_type": task}
                    await workflow.service.run_evaluation(config)

        # Now test the new model against baselines
        new_model = "gpt_4"

        baseline_result = await workflow.run_baseline_evaluation(
            new_model, baseline_models=baseline_models, config={"tasks": baseline_tasks}
        )

        # Verify baseline comparison structure
        assert baseline_result["new_model"] == new_model
        assert baseline_result["baseline_models"] == baseline_models

        # Check baseline comparison results
        comparison = baseline_result["baseline_comparison"]
        assert "comparisons" in comparison
        assert "summary" in comparison

        # Verify comparisons exist for each baseline
        for baseline in baseline_models:
            assert baseline in comparison["comparisons"]

        # Check performance summary
        summary = baseline_result["performance_summary"]
        assert "beats_baselines" in summary
        assert "total_baselines" in summary
        assert summary["total_baselines"] == len(baseline_models)

    @pytest.mark.asyncio
    async def test_batch_evaluation_integration(self, batch_evaluator):
        """Test batch evaluation integration."""
        batch_config = {
            "models": ["gpt_4", "claude_3", "llama_2"],
            "tasks": ["text_classification", "sentiment_analysis"],
            "datasets": ["imdb", "sst"],
            "evaluation_params": {"batch_size": 16, "max_length": 256},
        }

        # Run batch evaluation
        batch_result = await batch_evaluator.evaluate_batch(batch_config)

        # Verify batch completion
        assert batch_result.status.value == "completed"
        assert batch_result.total_evaluations == 12  # 3 models × 2 tasks × 2 datasets
        assert batch_result.successful_evaluations == batch_result.total_evaluations
        assert batch_result.failed_evaluations == 0

        # Verify performance metrics
        metrics = batch_result.performance_metrics
        assert metrics["success_rate"] == 1.0
        assert metrics["evaluations_per_second"] > 0
        assert metrics["avg_evaluation_time"] > 0

        # Verify results were generated
        assert len(batch_result.results) == batch_result.successful_evaluations

        # Check result diversity
        model_names = {r.model_name for r in batch_result.results}
        task_types = {r.task_type for r in batch_result.results}
        dataset_names = {r.dataset_name for r in batch_result.results}

        assert len(model_names) == 3
        assert len(task_types) == 2
        assert len(dataset_names) == 2

    @pytest.mark.asyncio
    async def test_parallel_processing_performance(self, batch_evaluator):
        """Test parallel processing performance and efficiency."""
        # Create large batch for parallel processing
        evaluation_configs = [
            {
                "model_name": f"model_{i % 3}",  # Cycle through 3 models
                "task_type": f"task_{i % 2}",  # Cycle through 2 tasks
                "dataset_name": "test_dataset",
            }
            for i in range(20)  # 20 evaluations total
        ]

        # Test sequential-like processing (1 worker)
        import time

        start_time = time.time()
        sequential_results = await batch_evaluator.parallel_evaluation(
            evaluation_configs[:5],  # Smaller batch for comparison
            max_workers=1,
        )
        sequential_time = time.time() - start_time

        # Test parallel processing (4 workers)
        start_time = time.time()
        parallel_results = await batch_evaluator.parallel_evaluation(
            evaluation_configs, max_workers=4
        )
        parallel_time = time.time() - start_time

        # Verify all evaluations completed
        assert len(sequential_results) == 5
        assert len(parallel_results) == 20

        # Count successful evaluations
        sequential_success = len([r for r in sequential_results if r.get("status") == "success"])
        parallel_success = len([r for r in parallel_results if r.get("status") == "success"])

        assert sequential_success == 5
        assert parallel_success == 20

        # Parallel processing should be more efficient per evaluation
        # (Though exact timing depends on system and mock behavior)
        sequential_per_eval = sequential_time / 5
        parallel_per_eval = parallel_time / 20

        # Log performance metrics for analysis
        print(f"Sequential: {sequential_per_eval:.3f}s per evaluation")
        print(f"Parallel: {parallel_per_eval:.3f}s per evaluation")

    @pytest.mark.asyncio
    async def test_workflow_error_recovery_integration(self, workflow):
        """Test workflow error recovery and partial completion."""
        # Configure service to fail for specific model
        original_run_evaluation = workflow.service.run_evaluation

        async def failing_evaluation(config):
            if config["model_name"] == "failing_model":
                raise Exception("Simulated model failure")
            return await original_run_evaluation(config)

        workflow.service.run_evaluation = failing_evaluation

        config = {
            "models": ["gpt_4", "failing_model", "claude_3"],
            "tasks": ["text_classification"],
        }

        # Workflow should complete despite partial failures
        workflow_id = await workflow.run_comprehensive_evaluation(config)
        progress = workflow.track_workflow_progress(workflow_id)

        # Workflow should still complete
        assert progress["state"] == "completed"

        # Check that successful evaluations were stored
        results = workflow.service.storage.query_results()
        successful_models = {r.model_name for r in results}

        # Should have results for working models
        assert "gpt_4" in successful_models
        assert "claude_3" in successful_models
        assert "failing_model" not in successful_models

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, workflow):
        """Test running multiple workflows concurrently."""
        # Define different workflow configurations
        workflow_configs = [
            {
                "models": ["gpt_4", "claude_3"],
                "tasks": ["text_classification"],
                "name": "classification_comparison",
            },
            {
                "models": ["llama_2", "bert_large"],
                "tasks": ["sentiment_analysis"],
                "name": "sentiment_analysis",
            },
            {"models": ["roberta_base"], "tasks": ["question_answering"], "name": "qa_evaluation"},
        ]

        # Run workflows concurrently
        tasks = []
        for i, config in enumerate(workflow_configs):
            task = asyncio.create_task(
                workflow.run_comprehensive_evaluation(config, workflow_id=f"concurrent_{i}")
            )
            tasks.append(task)

        # Wait for all workflows to complete
        workflow_ids = await asyncio.gather(*tasks)

        # Verify all workflows completed successfully
        assert len(workflow_ids) == 3

        for workflow_id in workflow_ids:
            progress = workflow.track_workflow_progress(workflow_id)
            assert progress["state"] == "completed"
            assert progress["progress_percentage"] == 100.0

        # Verify results from all workflows were stored
        all_results = workflow.service.storage.query_results()
        expected_total = sum(
            len(config["models"]) * len(config["tasks"]) for config in workflow_configs
        )
        assert len(all_results) == expected_total

    @pytest.mark.asyncio
    async def test_adaptive_batch_processing_integration(self, batch_evaluator):
        """Test adaptive batch processing with realistic scenarios."""
        # Create evaluation configs with varying complexity
        evaluation_configs = []

        # Simple evaluations (fast)
        for _ in range(10):
            evaluation_configs.append(
                {"model_name": "simple_model", "task_type": "simple_task", "complexity": "low"}
            )

        # Complex evaluations (slower)
        for _ in range(10):
            evaluation_configs.append(
                {"model_name": "complex_model", "task_type": "complex_task", "complexity": "high"}
            )

        # Mock different processing times based on complexity
        original_evaluation = batch_evaluator.service.run_evaluation

        async def variable_speed_evaluation(config):
            if config.get("complexity") == "high":
                await asyncio.sleep(0.1)  # Slower evaluation
            else:
                await asyncio.sleep(0.01)  # Faster evaluation
            return await original_evaluation(config)

        batch_evaluator.service.run_evaluation = variable_speed_evaluation

        # Run adaptive batch processing
        batch_results = await batch_evaluator.adaptive_batch_processing(
            evaluation_configs,
            initial_batch_size=5,
            target_duration_minutes=0.05,  # Very short for testing
            max_batch_size=15,
        )

        # Verify all evaluations were processed
        total_successful = sum(br.successful_evaluations for br in batch_results)
        assert total_successful == 20

        # Verify adaptive behavior occurred
        assert len(batch_results) >= 1

        # All batches should have completed successfully
        for batch_result in batch_results:
            assert batch_result.status.value == "completed"

    @pytest.mark.asyncio
    async def test_resource_monitoring_integration(self, batch_evaluator):
        """Test resource monitoring during batch processing."""
        batch_config = {
            "models": ["gpt_4", "claude_3"],
            "tasks": ["text_classification", "sentiment_analysis"],
            "evaluation_params": {"batch_size": 32},
        }

        # Set up resource limits
        resource_limits = ResourceLimits(
            max_concurrent_evaluations=2,
            max_memory_mb=4000,
            max_cpu_percent=80.0,
            timeout_seconds=30,
        )

        # Run batch with resource monitoring
        batch_result = await batch_evaluator.evaluate_batch(
            batch_config, resource_limits=resource_limits
        )

        # Verify resource monitoring data was collected
        assert "usage_history" in batch_result.resource_usage
        assert len(batch_result.resource_usage["usage_history"]) > 0

        # Verify performance metrics include resource information
        metrics = batch_result.performance_metrics
        assert "avg_cpu_usage" in metrics
        assert "avg_memory_usage_mb" in metrics

        # Resource usage should be reasonable (not excessive)
        if metrics["avg_memory_usage_mb"] > 0:
            assert metrics["avg_memory_usage_mb"] < resource_limits.max_memory_mb

    @pytest.mark.asyncio
    async def test_end_to_end_analysis_workflow(self, workflow):
        """Test complete end-to-end workflow with analysis."""
        # Run comprehensive evaluation
        config = {
            "models": ["gpt_4", "claude_3", "llama_2"],
            "tasks": ["text_classification", "sentiment_analysis"],
            "metrics": ["accuracy", "f1_score", "precision", "recall"],
        }

        workflow_id = await workflow.run_comprehensive_evaluation(config)

        # Verify workflow completed with analysis
        progress = workflow.track_workflow_progress(workflow_id)
        assert progress["state"] == "completed"

        # Check that analysis results are available
        results = progress["results"]
        assert "analysis_results" in results
        assert "comparison_results" in results
        assert "report" in results

        # Verify report contains expected sections
        report = results["report"]
        assert "evaluation_summary" in report
        assert "performance_analysis" in report
        assert "model_comparisons" in report
        assert "recommendations" in report

        # Verify evaluation summary
        summary = report["evaluation_summary"]
        assert summary["total_evaluations"] == 6  # 3 models × 2 tasks
        assert summary["models_evaluated"] == 3
        assert summary["tasks_covered"] == 2

        # Verify model comparison rankings
        if "multi_model" in results["comparison_results"]:
            multi_model = results["comparison_results"]["multi_model"]
            if "ranking_analysis" in multi_model:
                rankings = multi_model["ranking_analysis"]["rankings"]
                assert len(rankings) == 3

                # Rankings should be ordered by performance
                for i in range(1, len(rankings)):
                    assert rankings[i - 1]["mean_performance"] >= rankings[i]["mean_performance"]

#!/usr/bin/env python3
"""
Demonstration of the evaluation workflow system.

This script shows how to use the EvaluationWorkflow and BatchEvaluator
to orchestrate complex evaluation scenarios with progress tracking and resource management.
"""

import asyncio
import tempfile
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from benchmark.evaluation import (
    BatchEvaluator,
    EvaluationWorkflow,
    ResourceLimits,
)
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage
from benchmark.services.evaluation_service import EvaluationService


async def create_mock_service() -> AsyncGenerator[EvaluationService, None]:
    """Create a mock evaluation service for demonstration."""

    # Create temporary storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ResultsStorage(temp_dir)

        # Create mock service
        service = MagicMock(spec=EvaluationService)
        service.storage = storage

        # Mock async methods
        service.list_available_models = AsyncMock(
            return_value=["gpt_4", "claude_3", "llama_2", "bert_large"]
        )

        service.prepare_model = AsyncMock(return_value={"status": "ready"})
        service.prepare_task_data = AsyncMock(return_value={"data_ready": True})

        # Mock evaluation execution
        async def mock_evaluation(config: dict[str, Any]) -> EvaluationResult:
            import random
            from datetime import datetime

            # Simulate different performance levels
            model_performance = {
                "gpt_4": 0.92,
                "claude_3": 0.89,
                "llama_2": 0.85,
                "bert_large": 0.82,
            }

            base_accuracy = model_performance.get(config["model_name"], 0.75)
            accuracy = base_accuracy + random.uniform(-0.03, 0.03)

            result = EvaluationResult(
                evaluation_id=f"demo_{random.randint(1000, 9999)}",
                model_name=config["model_name"],
                task_type=config["task_type"],
                dataset_name=config.get("dataset_name", "demo_dataset"),
                metrics={
                    "accuracy": accuracy,
                    "f1_score": accuracy * 0.95,
                    "precision": accuracy * 0.98,
                    "recall": accuracy * 0.92,
                },
                timestamp=datetime.now(),
                configuration=config,
                raw_responses=[],
                processing_time=random.uniform(5.0, 15.0),
                experiment_name="workflow_demo",
            )

            # Store result
            storage.store_evaluation_result(result)
            return result

        service.run_evaluation = mock_evaluation

        yield service


async def demonstrate_workflow_orchestration() -> None:
    """Demonstrate complete workflow orchestration."""
    print("🚀 Evaluation Workflow Demonstration")
    print("=" * 60)

    async for service in create_mock_service():
        workflow = EvaluationWorkflow(service)

        # Example 1: Comprehensive Evaluation
        print("\n📊 Running Comprehensive Evaluation")
        print("-" * 40)

        config = {
            "models": ["gpt_4", "claude_3", "llama_2"],
            "tasks": ["text_classification", "sentiment_analysis"],
            "evaluation_params": {"batch_size": 32, "max_length": 512},
            "metrics": ["accuracy", "f1_score"],
        }

        try:
            workflow_id = await workflow.run_comprehensive_evaluation(config)
            print(f"✅ Workflow {workflow_id} completed successfully")

            # Track progress
            progress = workflow.track_workflow_progress(workflow_id)
            print(f"   Progress: {progress['progress_percentage']:.1f}%")
            print(f"   State: {progress['state']}")
            print(f"   Total evaluations: {progress['results'].get('evaluation_results', 0)}")

        except Exception as e:
            print(f"❌ Workflow failed: {e}")

        # Example 2: Model Comparison
        print("\n🔍 Running Model Comparison")
        print("-" * 40)

        try:
            comparison_result = await workflow.run_model_comparison(
                models=["gpt_4", "claude_3", "llama_2"],
                tasks=["text_classification"],
                config={"metrics": ["accuracy", "f1_score"]},
            )

            print(f"✅ Compared {len(comparison_result['models_compared'])} models")
            print(f"   Evaluations: {comparison_result['evaluation_count']}")

            # Show analysis summary
            analysis = comparison_result.get("analysis_summary", {})
            insights = analysis.get("comparison_insights", [])
            if insights:
                print("   Key insights:")
                for insight in insights[:2]:
                    print(f"     • {insight}")

        except Exception as e:
            print(f"❌ Comparison failed: {e}")

        # Example 3: Baseline Evaluation
        print("\n📈 Running Baseline Evaluation")
        print("-" * 40)

        try:
            baseline_result = await workflow.run_baseline_evaluation(
                new_model="gpt_4",
                baseline_models=["claude_3", "llama_2"],
                config={"tasks": ["text_classification"]},
            )

            summary = baseline_result["performance_summary"]
            print(
                f"✅ New model beats {summary['beats_baselines']}/{summary['total_baselines']} baselines"
            )
            print(f"   Average improvement: {summary['improvement_percentage']:.1f}%")
            print(
                f"   Performance verdict: {baseline_result['analysis_results']['performance_verdict']}"
            )

        except Exception as e:
            print(f"❌ Baseline evaluation failed: {e}")


async def demonstrate_batch_processing() -> None:
    """Demonstrate batch processing capabilities."""
    print("\n🔄 Batch Processing Demonstration")
    print("=" * 60)

    async for service in create_mock_service():
        # Create batch evaluator with resource limits
        resource_limits = ResourceLimits(
            max_concurrent_evaluations=3, max_memory_mb=4000, timeout_seconds=30
        )

        batch_evaluator = BatchEvaluator(service, resource_limits)

        # Example 1: Simple Batch Evaluation
        print("\n📦 Running Simple Batch Evaluation")
        print("-" * 40)

        batch_config = {
            "models": ["gpt_4", "claude_3"],
            "tasks": ["text_classification", "sentiment_analysis"],
            "evaluation_params": {"batch_size": 16},
        }

        try:
            batch_result = await batch_evaluator.evaluate_batch(batch_config)

            print(
                f"✅ Batch completed: {batch_result.successful_evaluations}/{batch_result.total_evaluations} successful"
            )
            print(f"   Duration: {batch_result.duration_seconds:.1f} seconds")
            print(f"   Success rate: {batch_result.performance_metrics.get('success_rate', 0):.1%}")
            print(
                f"   Evaluations/sec: {batch_result.performance_metrics.get('evaluations_per_second', 0):.2f}"
            )

        except Exception as e:
            print(f"❌ Batch evaluation failed: {e}")

        # Example 2: Parallel Evaluation
        print("\n⚡ Running Parallel Evaluation")
        print("-" * 40)

        evaluation_configs = [
            {"model_name": f"model_{i}", "task_type": "demo_task"} for i in range(8)
        ]

        try:
            start_time = asyncio.get_event_loop().time()
            parallel_results = await batch_evaluator.parallel_evaluation(
                evaluation_configs, max_workers=4
            )
            end_time = asyncio.get_event_loop().time()

            successful = len([r for r in parallel_results if r.get("status") == "success"])
            print(f"✅ Parallel processing: {successful}/{len(evaluation_configs)} successful")
            print(f"   Total time: {end_time - start_time:.1f} seconds")
            print(
                f"   Parallel efficiency: {len(evaluation_configs) / (end_time - start_time):.1f} evals/sec"
            )

        except Exception as e:
            print(f"❌ Parallel evaluation failed: {e}")

        # Example 3: Adaptive Batch Processing
        print("\n🧠 Running Adaptive Batch Processing")
        print("-" * 40)

        large_config_set = [
            {"model_name": f"model_{i % 3}", "task_type": f"task_{i % 2}"} for i in range(15)
        ]

        try:
            adaptive_results = await batch_evaluator.adaptive_batch_processing(
                large_config_set,
                initial_batch_size=5,
                target_duration_minutes=0.1,  # Very short for demo
                max_batch_size=10,
            )

            total_successful = sum(br.successful_evaluations for br in adaptive_results)
            print(f"✅ Adaptive processing: {total_successful} successful evaluations")
            print(f"   Batches used: {len(adaptive_results)}")
            print(f"   Average batch size: {len(large_config_set) / len(adaptive_results):.1f}")

            # Show batch adaptation
            print("   Batch sizes:", [br.total_evaluations for br in adaptive_results])

        except Exception as e:
            print(f"❌ Adaptive processing failed: {e}")


async def demonstrate_workflow_management() -> None:
    """Demonstrate workflow management features."""
    print("\n⚙️  Workflow Management Demonstration")
    print("=" * 60)

    async for service in create_mock_service():
        workflow = EvaluationWorkflow(service)

        # Start multiple workflows
        print("\n🎯 Managing Multiple Workflows")
        print("-" * 40)

        workflows = []
        configs = [
            {"models": ["gpt_4"], "tasks": ["task_1"], "name": "quick_eval"},
            {"models": ["claude_3", "llama_2"], "tasks": ["task_2"], "name": "comparison"},
        ]

        # Start workflows
        for i, config in enumerate(configs):
            try:
                workflow_id = await workflow.run_comprehensive_evaluation(
                    config, workflow_id=f"demo_workflow_{i}"
                )
                workflows.append(workflow_id)
                print(f"✅ Started workflow {workflow_id}")
            except Exception as e:
                print(f"❌ Failed to start workflow {i}: {e}")

        # List workflows
        active_workflows = workflow.list_workflows()
        print(f"\n📋 Active Workflows: {len(active_workflows)}")
        for wf in active_workflows:
            print(f"   • {wf['workflow_id']}: {wf['state']} ({wf['progress_percentage']:.0f}%)")

        # Demonstrate cleanup
        print("\n🧹 Cleaning up workflows")
        cleaned = workflow.cleanup_completed_workflows(older_than_hours=0)  # Clean all
        print(f"   Cleaned up {cleaned} completed workflows")


async def main() -> None:
    """Run all demonstrations."""
    print("🎬 Evaluation Workflow System Demo")
    print("🎬 " + "=" * 50)

    try:
        await demonstrate_workflow_orchestration()
        await demonstrate_batch_processing()
        await demonstrate_workflow_management()

        print("\n🎉 Demo completed successfully!")
        print("   The workflow system provides:")
        print("   • Comprehensive evaluation orchestration")
        print("   • Parallel batch processing with resource management")
        print("   • Progress tracking and workflow state management")
        print("   • Statistical analysis and model comparison")
        print("   • Adaptive batch sizing for optimal performance")

    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

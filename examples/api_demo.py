#!/usr/bin/env python3
"""
Demonstration of the EvaluationAPI interface.

This script shows how to use the clean API interface to interact with
the evaluation service, including starting evaluations, monitoring progress,
and retrieving results.
"""

import asyncio
import tempfile
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from benchmark.evaluation import (
    APIError,
    EvaluationAPI,
    EvaluationConfig,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultsResponse,
    EvaluationStatus,
    EvaluationStatusResponse,
    EvaluationType,
    ValidationLevel,
)
from benchmark.evaluation.api_models import ResourceLimits as APIResourceLimits
from benchmark.evaluation.result_models import EvaluationResult
from benchmark.evaluation.results_storage import ResultsStorage
from benchmark.services.evaluation_service import EvaluationService


async def create_demo_api() -> EvaluationAPI:
    """Create a demo API with mock evaluation service."""

    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    storage = ResultsStorage(temp_dir)

    # Create mock service with realistic behavior
    service = MagicMock(spec=EvaluationService)
    service.storage = storage

    # Mock available models
    service.list_available_models = AsyncMock(
        return_value=["gpt-4", "claude-3", "llama-2", "bert-large", "t5-base"]
    )

    # Mock evaluation with realistic performance data
    async def demo_evaluation(config: dict[str, Any]) -> EvaluationResult:
        import random

        # Simulate different model performances
        model_performance = {
            "gpt-4": {"base_accuracy": 0.92, "variance": 0.03},
            "claude-3": {"base_accuracy": 0.89, "variance": 0.025},
            "llama-2": {"base_accuracy": 0.85, "variance": 0.04},
            "bert-large": {"base_accuracy": 0.88, "variance": 0.02},
            "t5-base": {"base_accuracy": 0.83, "variance": 0.035},
        }

        # Task-specific performance modifiers
        task_modifiers = {
            "text_classification": 1.0,
            "sentiment_analysis": 1.05,
            "question_answering": 0.95,
            "named_entity_recognition": 0.98,
            "text_generation": 0.90,
        }

        model_name = config["model_name"]
        task_type = config["task_type"]

        # Calculate performance
        perf = model_performance.get(model_name, {"base_accuracy": 0.75, "variance": 0.05})
        modifier = task_modifiers.get(task_type, 1.0)

        base_accuracy = perf["base_accuracy"] * modifier
        variance = perf["variance"]

        accuracy = base_accuracy + random.uniform(-variance, variance)
        accuracy = max(0.5, min(0.98, accuracy))  # Clamp to reasonable range

        # Generate correlated metrics
        f1_score = accuracy * random.uniform(0.95, 0.99)
        precision = accuracy * random.uniform(0.96, 1.02)
        recall = accuracy * random.uniform(0.92, 0.98)

        # Simulate processing time
        processing_time = random.uniform(2.0, 8.0)
        await asyncio.sleep(processing_time * 0.01)  # Scaled down for demo

        result = EvaluationResult(
            evaluation_id=f"demo_{random.randint(1000, 9999)}",
            model_name=model_name,
            task_type=task_type,
            dataset_name=config.get("dataset_name", "demo_dataset"),
            metrics={
                "accuracy": accuracy,
                "f1_score": f1_score,
                "precision": min(1.0, precision),
                "recall": min(1.0, recall),
                "processing_time_ms": processing_time * 1000,
            },
            timestamp=datetime.now(),
            configuration=config,
            raw_responses=[],
            processing_time=processing_time,
            experiment_name="api_demo",
            tags=["demo", "api"],
        )

        # Store result
        storage.store_evaluation_result(result)
        return result

    service.run_evaluation = demo_evaluation

    # Create and return API
    return EvaluationAPI(
        evaluation_service=service, max_concurrent_evaluations=8, default_timeout_seconds=300
    )


async def demonstrate_basic_evaluation() -> None:
    """Demonstrate basic single model evaluation."""
    print("🔍 Basic Single Model Evaluation")
    print("=" * 50)

    api = await create_demo_api()

    # Create evaluation request
    request = EvaluationRequest(
        model_names=["gpt-4"],
        task_types=["text_classification"],
        dataset_names=["imdb"],
        evaluation_type=EvaluationType.SINGLE_MODEL,
        evaluation_config=EvaluationConfig(
            batch_size=32, temperature=0.7, validation_level=ValidationLevel.MODERATE
        ),
        metadata={"experiment": "demo_basic", "version": "1.0"},
    )

    print(f"📋 Request created: {request.request_id}")
    print(f"   Model: {request.model_names[0]}")
    print(f"   Task: {request.task_types[0]}")
    print(f"   Dataset: {request.dataset_names[0] if request.dataset_names else 'default'}")

    # Start evaluation
    response = await api.start_evaluation(request)

    if isinstance(response, EvaluationResponse):
        print(f"✅ Evaluation started: {response.evaluation_id}")
        print(f"   Status: {response.status.value}")
        print(f"   Message: {response.message}")

        # Monitor progress
        evaluation_id = response.evaluation_id
        print("\n⏳ Monitoring progress...")

        for _i in range(20):  # Max 2 seconds
            status = await api.get_evaluation_status(evaluation_id)

            if isinstance(status, EvaluationStatusResponse):
                print(f"   Status: {status.status.value}", end="")

                if status.progress:
                    print(f" ({status.progress.progress_percentage:.1f}%)")
                    print(f"   Current step: {status.progress.current_step_description}")
                else:
                    print()

                if status.status == EvaluationStatus.COMPLETED:
                    print("🎉 Evaluation completed!")
                    break
                elif status.status == EvaluationStatus.FAILED:
                    print(f"❌ Evaluation failed: {status.error_message}")
                    return

            await asyncio.sleep(0.1)

        # Get results
        results = await api.get_evaluation_results(evaluation_id)

        if isinstance(results, EvaluationResultsResponse):
            print("\n📊 Results:")
            for result in results.results:
                print(f"   Model: {result.model_name}")
                print(f"   Task: {result.task_type}")
                print(f"   Accuracy: {result.metrics['accuracy']:.3f}")
                print(f"   F1 Score: {result.metrics['f1_score']:.3f}")
                print(f"   Processing time: {result.processing_time:.2f}s")

            print("\n📈 Summary:")
            summary = results.summary
            print(f"   Total evaluations: {summary['total_evaluations']}")
            print(f"   Successful: {summary['successful_evaluations']}")
            print(f"   Models evaluated: {summary['models_evaluated']}")

    elif isinstance(response, APIError):
        print(f"❌ Failed to start evaluation: {response.error_message}")
        if response.validation_errors:
            print("   Validation errors:")
            for error in response.validation_errors:
                print(f"     - {error.field}: {error.message}")


async def demonstrate_model_comparison() -> None:
    """Demonstrate model comparison evaluation."""
    print("\n🏆 Model Comparison Evaluation")
    print("=" * 50)

    api = await create_demo_api()

    # Create comparison request
    request = EvaluationRequest(
        model_names=["gpt-4", "claude-3", "llama-2", "bert-large"],
        task_types=["text_classification", "sentiment_analysis"],
        evaluation_type=EvaluationType.MODEL_COMPARISON,
        evaluation_config=EvaluationConfig(
            batch_size=64, temperature=0.5, validation_level=ValidationLevel.STRICT
        ),
        resource_limits=APIResourceLimits(
            max_concurrent_evaluations=6, max_memory_mb=8000, max_execution_time_seconds=120
        ),
        metadata={"experiment": "model_comparison_demo"},
    )

    print(f"📋 Comparing {len(request.model_names)} models on {len(request.task_types)} tasks")
    print(f"   Models: {', '.join(request.model_names)}")
    print(f"   Tasks: {', '.join(request.task_types)}")

    # Start evaluation
    response = await api.start_evaluation(request)

    if isinstance(response, EvaluationResponse):
        print(f"✅ Comparison started: {response.evaluation_id}")
        evaluation_id = response.evaluation_id

        # Monitor with progress updates
        print("\n⏳ Running comparison...")
        last_progress = 0.0

        for _i in range(50):  # Max 5 seconds
            status = await api.get_evaluation_status(evaluation_id)

            if isinstance(status, EvaluationStatusResponse):
                if status.progress and status.progress.progress_percentage != last_progress:
                    progress = status.progress.progress_percentage
                    print(
                        f"   Progress: {progress:.1f}% - {status.progress.current_step_description}"
                    )
                    last_progress = progress

                if status.status == EvaluationStatus.COMPLETED:
                    break
                elif status.status == EvaluationStatus.FAILED:
                    print(f"❌ Comparison failed: {status.error_message}")
                    return

            await asyncio.sleep(0.1)

        # Get and display results
        results = await api.get_evaluation_results(evaluation_id)

        if isinstance(results, EvaluationResultsResponse):
            print("\n📊 Comparison Results:")

            # Group results by task
            results_by_task: dict[str, list[EvaluationResult]] = {}
            for result in results.results:
                task = result.task_type
                if task not in results_by_task:
                    results_by_task[task] = []
                results_by_task[task].append(result)

            for task, task_results in results_by_task.items():
                print(f"\n   📋 {task}:")

                # Sort by accuracy
                task_results.sort(key=lambda r: r.metrics.get("accuracy", 0), reverse=True)

                for i, result in enumerate(task_results, 1):
                    accuracy = result.metrics.get("accuracy", 0)
                    f1_score = result.metrics.get("f1_score", 0)
                    print(
                        f"      {i}. {result.model_name:<12} - Accuracy: {accuracy:.3f}, F1: {f1_score:.3f}"
                    )

            # Display summary
            summary = results.summary
            print("\n📈 Overall Summary:")
            print(f"   Total evaluations: {summary['total_evaluations']}")
            print(f"   Models compared: {summary['models_evaluated']}")
            print(f"   Tasks evaluated: {summary['tasks_evaluated']}")

            if "metric_summaries" in summary:
                metrics = summary["metric_summaries"]
                if "accuracy" in metrics:
                    acc_stats = metrics["accuracy"]
                    print(f"   Average accuracy: {acc_stats['mean']:.3f}")
                    print(f"   Best accuracy: {acc_stats['max']:.3f}")


async def demonstrate_batch_processing() -> None:
    """Demonstrate batch processing capabilities."""
    print("\n⚡ Batch Processing Demonstration")
    print("=" * 50)

    api = await create_demo_api()

    # Create batch processing request
    request = EvaluationRequest(
        model_names=["gpt-4", "claude-3", "llama-2"],
        task_types=["text_classification", "sentiment_analysis", "named_entity_recognition"],
        dataset_names=["dataset_1", "dataset_2"],
        evaluation_type=EvaluationType.BATCH_PROCESSING,
        resource_limits=APIResourceLimits(max_concurrent_evaluations=8, max_memory_mb=12000),
        metadata={"experiment": "batch_demo", "batch_size": "large"},
    )

    total_evaluations = (
        len(request.model_names) * len(request.task_types) * len(request.dataset_names or [])
    )
    print("📦 Starting batch processing:")
    print(
        f"   {len(request.model_names)} models × {len(request.task_types)} tasks × {len(request.dataset_names or [])} datasets"
    )
    print(f"   = {total_evaluations} total evaluations")

    # Start batch evaluation
    start_time = asyncio.get_event_loop().time()
    response = await api.start_evaluation(request)

    if isinstance(response, EvaluationResponse):
        print(f"✅ Batch started: {response.evaluation_id}")
        evaluation_id = response.evaluation_id

        # Monitor batch progress
        print("\n⏳ Processing batch...")

        for i in range(100):  # Max 10 seconds
            status = await api.get_evaluation_status(evaluation_id)

            if isinstance(status, EvaluationStatusResponse):
                if status.progress:
                    progress = status.progress.progress_percentage
                    completed = status.progress.completed_steps
                    total = status.progress.total_steps

                    if i % 10 == 0 or progress == 100:  # Update every second or on completion
                        print(f"   Progress: {progress:.1f}% ({completed}/{total} evaluations)")

                if status.status == EvaluationStatus.COMPLETED:
                    break
                elif status.status == EvaluationStatus.FAILED:
                    print(f"❌ Batch failed: {status.error_message}")
                    return

            await asyncio.sleep(0.1)

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # Get results and performance metrics
        results = await api.get_evaluation_results(evaluation_id)

        if isinstance(results, EvaluationResultsResponse):
            print(f"\n🎉 Batch completed in {total_time:.2f} seconds!")
            print(f"   Throughput: {len(results.results) / total_time:.1f} evaluations/second")

            # Performance breakdown
            model_performance: dict[str, list[float]] = {}
            task_performance: dict[str, list[float]] = {}

            for result in results.results:
                model = result.model_name
                task = result.task_type
                accuracy = result.metrics.get("accuracy", 0)

                if model not in model_performance:
                    model_performance[model] = []
                model_performance[model].append(accuracy)

                if task not in task_performance:
                    task_performance[task] = []
                task_performance[task].append(accuracy)

            print("\n📊 Performance by Model:")
            for model, accuracies in model_performance.items():
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"   {model:<12}: {avg_acc:.3f} (avg across {len(accuracies)} evaluations)")

            print("\n📋 Performance by Task:")
            for task, accuracies in task_performance.items():
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"   {task:<20}: {avg_acc:.3f} (avg across {len(accuracies)} models)")


async def demonstrate_error_handling() -> None:
    """Demonstrate API error handling."""
    print("\n⚠️  Error Handling Demonstration")
    print("=" * 50)

    api = await create_demo_api()

    # Test validation errors
    print("🔍 Testing validation errors...")

    invalid_request = EvaluationRequest(
        model_names=[],  # Empty - invalid
        task_types=["text_classification"],
        evaluation_config=EvaluationConfig(
            batch_size=-10,  # Invalid
            temperature=5.0,  # Invalid
        ),
    )

    response = await api.start_evaluation(invalid_request)

    if isinstance(response, APIError):
        print(f"✅ Caught validation error: {response.error_code}")
        print("   Validation issues:")
        for error in response.validation_errors:
            print(f"     - {error.message}")

    # Test capacity limits
    print(f"\n🔍 Testing capacity limits (max: {api.max_concurrent_evaluations})...")

    # Fill up capacity
    evaluation_ids = []
    for _i in range(api.max_concurrent_evaluations + 2):  # Try to exceed capacity
        request = EvaluationRequest(model_names=["gpt-4"], task_types=["text_classification"])

        response = await api.start_evaluation(request)

        if isinstance(response, EvaluationResponse):
            evaluation_ids.append(response.evaluation_id)
        elif isinstance(response, APIError) and response.error_code == "CAPACITY_EXCEEDED":
            print(f"✅ Capacity limit enforced after {len(evaluation_ids)} evaluations")
            break

    # Test non-existent evaluation
    print("\n🔍 Testing non-existent evaluation access...")

    fake_id = "nonexistent_evaluation_id"
    status = await api.get_evaluation_status(fake_id)

    if isinstance(status, APIError):
        print(f"✅ Properly handled non-existent ID: {status.error_code}")

    # Show active evaluations
    active_list = await api.list_active_evaluations()
    print(f"\n📋 Active evaluations: {len(active_list)}")
    for eval_info in active_list:
        print(
            f"   {eval_info['evaluation_id']}: {eval_info['status']} ({eval_info['progress_percentage']:.0f}%)"
        )


async def demonstrate_api_management() -> None:
    """Demonstrate API management features."""
    print("\n⚙️  API Management Features")
    print("=" * 50)

    api = await create_demo_api()

    # Show available evaluators
    print("🔍 Available evaluators:")
    evaluators = await api.list_available_evaluators()

    from benchmark.evaluation import AvailableEvaluatorsResponse

    if isinstance(evaluators, AvailableEvaluatorsResponse):
        print("   Evaluation types:")
        for eval_type in evaluators.evaluation_types:
            print(f"     - {eval_type}")

        print("   Supported tasks:")
        for task, metrics in evaluators.supported_tasks.items():
            print(f"     - {task}: {', '.join(metrics[:3])}")  # Show first 3 metrics

    # Start several evaluations
    print("\n🚀 Starting multiple evaluations for management demo...")

    for i in range(3):
        request = EvaluationRequest(
            model_names=[f"model_{i}"], task_types=["text_classification"], metadata={"demo_id": i}
        )
        await api.start_evaluation(request)

    # List active evaluations
    print("\n📋 Active evaluations:")
    active_list = await api.list_active_evaluations()

    for eval_info in active_list:
        print(f"   {eval_info['evaluation_id'][:12]}... ({eval_info['status']})")
        print(f"      Models: {eval_info['model_count']}, Tasks: {eval_info['task_count']}")
        print(f"      Progress: {eval_info['progress_percentage']:.0f}%")

    # Wait a bit and then cleanup
    await asyncio.sleep(0.3)

    print("\n🧹 Cleaning up completed evaluations...")
    cleaned = await api.cleanup_completed_evaluations(older_than_hours=0)
    print(f"   Cleaned up {cleaned} evaluations")

    remaining = await api.list_active_evaluations()
    print(f"   Remaining active: {len(remaining)}")


async def main() -> None:
    """Run all API demonstrations."""
    print("🎬 EvaluationAPI Demonstration")
    print("🎬 " + "=" * 40)
    print("This demo shows how to use the clean API interface for")
    print("evaluation service integration, including:")
    print("• Basic single model evaluation")
    print("• Model comparison workflows")
    print("• Batch processing capabilities")
    print("• Error handling and validation")
    print("• API management features")
    print()

    try:
        await demonstrate_basic_evaluation()
        await demonstrate_model_comparison()
        await demonstrate_batch_processing()
        await demonstrate_error_handling()
        await demonstrate_api_management()

        print("\n🎉 API Demo completed successfully!")
        print("\n💡 Key takeaways:")
        print("   • The API provides clean, type-safe interfaces")
        print("   • Comprehensive validation catches errors early")
        print("   • Real-time progress monitoring keeps users informed")
        print("   • Resource management prevents system overload")
        print("   • Error handling provides clear feedback")
        print("   • Management features enable monitoring and cleanup")

    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Demonstration of the analysis tools for evaluation results.

This script shows how to use the ResultsAnalyzer and ModelComparisonEngine
to perform comprehensive analysis of model evaluation results.
"""

import tempfile
from datetime import datetime, timedelta

import numpy as np

from benchmark.evaluation import (
    EvaluationResult,
    ModelComparisonEngine,
    ResultsAnalyzer,
    ResultsStorage,
)


def create_sample_data(storage: ResultsStorage) -> None:
    """Create sample evaluation data for demonstration."""
    models = ["gpt_4", "claude_3", "llama_2", "bert_large"]
    tasks = ["text_classification", "sentiment_analysis", "question_answering"]
    datasets = ["imdb", "sst", "squad", "glue"]

    print("Creating sample evaluation data...")

    eval_id = 0
    for model_idx, model in enumerate(models):
        # Different models have different base performance levels
        base_performance = 0.85 - model_idx * 0.05

        for task in tasks:
            for dataset in datasets:
                for i in range(8):  # 8 evaluations per model/task/dataset combo
                    # Add realistic variation
                    noise = np.random.normal(0, 0.03)
                    accuracy = max(0.5, min(0.99, base_performance + noise))

                    # Create evaluation result
                    result = EvaluationResult(
                        evaluation_id=f"eval_{eval_id:04d}",
                        model_name=model,
                        task_type=task,
                        dataset_name=dataset,
                        metrics={
                            "accuracy": accuracy,
                            "f1_score": accuracy * 0.96,
                            "precision": accuracy * 0.98,
                            "recall": accuracy * 0.94,
                        },
                        timestamp=datetime.now() - timedelta(days=30 - eval_id % 30),
                        configuration={
                            "learning_rate": 0.0001 * (1 + model_idx),
                            "batch_size": 16 * (1 + i % 2),
                            "temperature": 0.7 + model_idx * 0.1,
                        },
                        raw_responses=[],
                        processing_time=10.0 + np.random.normal(0, 2),
                        experiment_name="comprehensive_evaluation",
                        tags=["demo", model.replace("_", "-")],
                        notes=f"Evaluation {eval_id} for {model} on {task}/{dataset}",
                    )

                    storage.store_evaluation_result(result)
                    eval_id += 1

    print(f"Created {eval_id} evaluation results across {len(models)} models")


def demonstrate_results_analyzer(storage: ResultsStorage) -> None:
    """Demonstrate the ResultsAnalyzer capabilities."""
    print("\n" + "=" * 60)
    print("RESULTS ANALYZER DEMONSTRATION")
    print("=" * 60)

    analyzer = ResultsAnalyzer(storage)

    # Analyze individual model performance
    print("\n1. Individual Model Performance Analysis")
    print("-" * 40)

    model_analysis = analyzer.analyze_model_performance("gpt_4")
    print(f"Model: {model_analysis['model_name']}")

    if "error" not in model_analysis:
        # Use the actual structure
        if "metric_analysis" in model_analysis:
            metrics = model_analysis["metric_analysis"]
            if "accuracy" in metrics:
                acc_data = metrics["accuracy"]
                print(f"Accuracy: {acc_data.get('mean', 'N/A'):.3f} ± {acc_data.get('std', 0):.3f}")
                print(f"Evaluations: {acc_data.get('count', 0)}")

        if "summary" in model_analysis:
            summary = model_analysis["summary"]
            print(f"Total evaluations: {summary.get('total_evaluations', 0)}")
            print(f"Tasks covered: {summary.get('unique_tasks', 0)}")
            print(f"Datasets covered: {summary.get('unique_datasets', 0)}")

    # Identify performance trends
    print("\n2. Performance Trend Analysis")
    print("-" * 40)

    trends = analyzer.identify_performance_trends("gpt_4")
    if "error" not in trends and "accuracy" in trends["metric_trends"]:
        accuracy_trend = trends["metric_trends"]["accuracy"]
        direction = accuracy_trend["trend_direction"]
        correlation = accuracy_trend["correlation"]
        print(f"Trend Direction: {direction}")
        print(f"Trend Strength: {correlation:.3f}")
        print(f"Statistically Significant: {accuracy_trend['is_significant']}")

    # Find performance bottlenecks
    print("\n3. Performance Bottleneck Detection")
    print("-" * 40)

    all_results = storage.query_results()
    bottlenecks = analyzer.find_performance_bottlenecks(all_results)

    print(f"Total evaluations analyzed: {bottlenecks['summary']['total_evaluations']}")
    print("Summary keys:", list(bottlenecks["summary"].keys()))
    if "unique_models" in bottlenecks["summary"]:
        print(f"Models analyzed: {bottlenecks['summary']['unique_models']}")
    if "time_span_days" in bottlenecks["summary"]:
        print(f"Time span: {bottlenecks['summary']['time_span_days']} days")

    recommendations = bottlenecks["recommendations"]
    if recommendations:
        print("\nTop Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")

    # Generate performance summary
    print("\n4. Executive Performance Summary")
    print("-" * 40)

    summary = analyzer.generate_performance_summary(all_results)
    exec_summary = summary["executive_summary"]

    print(f"Models Evaluated: {exec_summary['models_evaluated']}")
    print(f"Total Evaluations: {exec_summary['total_evaluations']}")
    print(f"Overall Success Rate: {exec_summary['overall_success_rate']:.1%}")
    print(f"Average Performance: {exec_summary['average_performance']:.3f}")

    if summary["model_ranking"]:
        print("\nTop 3 Models:")
        for i, model in enumerate(summary["model_ranking"][:3], 1):
            print(f"  {i}. {model['model_name']}: {model['average_score']:.3f}")


def demonstrate_comparison_engine(storage: ResultsStorage) -> None:
    """Demonstrate the ModelComparisonEngine capabilities."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON ENGINE DEMONSTRATION")
    print("=" * 60)

    engine = ModelComparisonEngine(storage)

    # Compare multiple models
    print("\n1. Multi-Model Comparison")
    print("-" * 40)

    models_to_compare = ["gpt_4", "claude_3", "llama_2"]
    comparison = engine.compare_models(model_names=models_to_compare, metric_name="accuracy")

    print(f"Comparing {len(models_to_compare)} models")
    print(f"Total evaluations: {comparison['comparison_metadata']['total_evaluations']}")

    # Show performance summary
    perf_summary = comparison["performance_summary"]
    print("\nPerformance Summary:")
    for model, stats in perf_summary.items():
        print(f"  {model}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")

    # Show ranking
    ranking = comparison["ranking_analysis"]["rankings"]
    print("\nRankings:")
    for rank_info in ranking:
        print(
            f"  #{rank_info['rank']}: {rank_info['model_name']} ({rank_info['mean_performance']:.3f})"
        )

    # Statistical significance testing
    print("\n2. Statistical Significance Testing")
    print("-" * 40)

    model_a_results = storage.query_results({"model_name": "gpt_4"})
    model_b_results = storage.query_results({"model_name": "claude_3"})

    if model_a_results and model_b_results:
        test_result = engine.statistical_significance_test(
            model_a_results, model_b_results, "accuracy"
        )

        if "error" not in test_result:
            test_stats = test_result["test_results"]
            print(f"Test: {test_stats['test_name']}")
            print(f"P-value: {test_stats['p_value']:.4f}")
            print(f"Significant: {test_stats['is_significant']}")
            print(
                f"Effect Size: {test_stats['effect_size']:.3f} ({test_stats['effect_size_interpretation']})"
            )
            print(f"Interpretation: {test_result['interpretation']}")

    # Model ranking with multiple metrics
    print("\n3. Multi-Metric Model Ranking")
    print("-" * 40)

    ranking = engine.rank_models(
        model_names=models_to_compare,
        metrics=["accuracy", "f1_score", "precision"],
        weights={"accuracy": 0.5, "f1_score": 0.3, "precision": 0.2},
    )

    print("Weighted Rankings:")
    for rank_info in ranking["rankings"]:
        print(
            f"  #{rank_info['rank']}: {rank_info['model_name']} (score: {rank_info['overall_score']:.3f})"
        )


def main() -> None:
    """Run the analysis demonstration."""
    print("LLM Benchmark Analysis Tools Demonstration")
    print("=" * 60)

    # Create temporary storage for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ResultsStorage(temp_dir)

        # Create sample data
        create_sample_data(storage)

        # Demonstrate analysis capabilities
        demonstrate_results_analyzer(storage)
        demonstrate_comparison_engine(storage)

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThe analysis tools provide comprehensive insights into:")
        print("• Individual model performance and trends")
        print("• Performance bottlenecks and improvement opportunities")
        print("• Statistical comparisons between models")
        print("• Multi-metric model rankings")
        print("• Executive summaries for stakeholders")


if __name__ == "__main__":
    main()

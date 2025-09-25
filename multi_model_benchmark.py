#!/usr/bin/env python3
"""
Multi-Model NSL-KDD Benchmark with OpenRouter

Tests multiple free models with configurable sample sizes to compare performance.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import our existing benchmark
import sys
sys.path.append('.')
from nsl_kdd_openrouter_benchmark import NSLKDDBenchmarkWithOpenRouter


# Model configurations
FREE_MODELS = {
    "meta-llama/llama-3.2-3b-instruct:free": {
        "name": "Llama 3.2 3B",
        "description": "Meta's efficient 3B parameter model"
    },
    "openai/gpt-oss-120b:free": {
        "name": "GPT-OSS 120B",
        "description": "OpenAI's open-source large model"
    },
    "openai/gpt-oss-20b:free": {
        "name": "GPT-OSS 20B",
        "description": "OpenAI's open-source medium model"
    },
    "z-ai/glm-4.5-air:free": {
        "name": "GLM-4.5 Air",
        "description": "Zhipu AI's efficient model"
    },
    "qwen/qwen3-coder:free": {
        "name": "Qwen3 Coder",
        "description": "Alibaba's coding-focused model"
    },
    "nvidia/nemotron-nano-9b-v2:free": {
        "name": "Nemotron Nano 9B",
        "description": "NVIDIA's nano language model"
    },
    "moonshotai/kimi-k2:free": {
        "name": "Kimi K2",
        "description": "Moonshot AI's conversational model"
    }
}


class MultiModelBenchmark:
    """Benchmark multiple models with configurable parameters."""

    def __init__(self, models_to_test=None, sample_size=5, attack_ratio=0.3, dataset_file="KDDTrain+.txt"):
        self.models_to_test = models_to_test or list(FREE_MODELS.keys())
        self.sample_size = sample_size
        self.attack_ratio = attack_ratio
        self.dataset_file = dataset_file
        self.results = []

    async def test_model(self, model_name: str) -> dict:
        """Test a single model and return results."""
        print(f"\n🚀 Testing {FREE_MODELS[model_name]['name']} ({model_name})")
        print("=" * 60)

        try:
            # Initialize benchmark for this model
            benchmark = NSLKDDBenchmarkWithOpenRouter(
                api_key=None,  # Will use .env
                model_name=model_name
            )

            # Load data
            connections = benchmark.load_nsl_kdd_data(
                filename=self.dataset_file,
                max_samples=self.sample_size,
                attack_ratio=self.attack_ratio
            )

            # Run benchmark
            start_time = time.time()
            report = await benchmark.run_benchmark(connections)
            total_time = time.time() - start_time

            # Extract key metrics
            metrics = report["metrics"]
            result = {
                "model_name": model_name,
                "display_name": FREE_MODELS[model_name]["name"],
                "description": FREE_MODELS[model_name]["description"],
                "status": "SUCCESS",
                "total_samples": len(connections),
                "attack_samples": report["benchmark_info"]["attack_samples"],
                "normal_samples": report["benchmark_info"]["normal_samples"],
                "overall_accuracy": metrics["classification_metrics"]["overall_accuracy"],
                "balanced_accuracy": metrics["classification_metrics"]["balanced_accuracy"],
                "precision": metrics["classification_metrics"]["precision"],
                "recall": metrics["classification_metrics"]["recall_sensitivity"],
                "f1_score": metrics["classification_metrics"]["f1_score"],
                "avg_response_time": metrics["performance_metrics"]["average_response_time"],
                "total_processing_time": total_time,
                "predictions_per_second": metrics["performance_metrics"]["predictions_per_second"],
                "timestamp": datetime.now().isoformat(),
                "full_report": report
            }

            print(f"✅ {FREE_MODELS[model_name]['name']}: {result['overall_accuracy']:.1%} accuracy")
            return result

        except Exception as e:
            print(f"❌ {FREE_MODELS[model_name]['name']} failed: {str(e)[:100]}...")
            return {
                "model_name": model_name,
                "display_name": FREE_MODELS[model_name]["name"],
                "description": FREE_MODELS[model_name]["description"],
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def run_all_models(self) -> list:
        """Test all models and return aggregated results."""
        print("🔬 Multi-Model NSL-KDD Cybersecurity Benchmark")
        print("=" * 60)
        print(f"Dataset: {self.dataset_file}")
        print(f"Sample size: {self.sample_size}")
        print(f"Attack ratio: {self.attack_ratio}")
        print(f"Models to test: {len(self.models_to_test)}")
        print("=" * 60)

        results = []
        for i, model_name in enumerate(self.models_to_test, 1):
            print(f"\n[{i}/{len(self.models_to_test)}] Testing {FREE_MODELS[model_name]['name']}")

            result = await self.test_model(model_name)
            results.append(result)

            # Add delay between models to respect rate limits
            if i < len(self.models_to_test):
                print("⏳ Waiting 30 seconds before next model...")
                await asyncio.sleep(30)

        self.results = results
        return results

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary of all model results."""

        successful_results = [r for r in self.results if r.get("status") == "SUCCESS"]
        failed_results = [r for r in self.results if r.get("status") == "FAILED"]

        if not successful_results:
            return "❌ No models completed successfully."

        # Sort by accuracy
        successful_results.sort(key=lambda x: x["overall_accuracy"], reverse=True)

        report = []
        report.append("🏆 MULTI-MODEL CYBERSECURITY BENCHMARK RESULTS")
        report.append("=" * 70)
        report.append(f"📊 Dataset: {self.dataset_file}")
        report.append(f"📈 Sample Size: {self.sample_size} ({self.attack_ratio:.1%} attacks)")
        report.append(f"✅ Successful: {len(successful_results)}/{len(self.results)} models")
        report.append("")

        # Rankings
        report.append("🥇 MODEL RANKINGS (by Overall Accuracy)")
        report.append("=" * 50)
        for i, result in enumerate(successful_results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
            report.append(f"{emoji} {i}. {result['display_name']}")
            report.append(f"   Accuracy: {result['overall_accuracy']:.1%} | F1: {result['f1_score']:.3f} | Speed: {result['avg_response_time']:.2f}s")

        # Detailed metrics
        report.append("\n📊 DETAILED METRICS")
        report.append("=" * 50)
        report.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Speed':<10}")
        report.append("-" * 70)

        for result in successful_results:
            report.append(f"{result['display_name'][:19]:<20} "
                         f"{result['overall_accuracy']:.1%} "
                         f"{result['precision']:.3f}"[:9].ljust(10) + " "
                         f"{result['recall']:.3f}"[:9].ljust(10) + " "
                         f"{result['f1_score']:.3f}"[:9].ljust(10) + " "
                         f"{result['avg_response_time']:.2f}s"[:9].ljust(10))

        # Performance insights
        report.append("\n🎯 KEY INSIGHTS")
        report.append("=" * 50)

        if successful_results:
            best_accuracy = successful_results[0]
            fastest_model = min(successful_results, key=lambda x: x["avg_response_time"])
            best_f1 = max(successful_results, key=lambda x: x["f1_score"])

            report.append(f"🎯 Most Accurate: {best_accuracy['display_name']} ({best_accuracy['overall_accuracy']:.1%})")
            report.append(f"⚡ Fastest: {fastest_model['display_name']} ({fastest_model['avg_response_time']:.2f}s avg)")
            report.append(f"🏆 Best F1-Score: {best_f1['display_name']} ({best_f1['f1_score']:.3f})")

            avg_accuracy = sum(r["overall_accuracy"] for r in successful_results) / len(successful_results)
            report.append(f"📊 Average Accuracy: {avg_accuracy:.1%}")

        if failed_results:
            report.append(f"\n❌ Failed Models: {', '.join(r['display_name'] for r in failed_results)}")

        return "\n".join(report)

    def save_results(self, output_file: str = None) -> str:
        """Save detailed results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"multi_model_benchmark_{timestamp}.json"

        results_data = {
            "benchmark_info": {
                "dataset_file": self.dataset_file,
                "sample_size": self.sample_size,
                "attack_ratio": self.attack_ratio,
                "models_tested": len(self.results),
                "timestamp": datetime.now().isoformat()
            },
            "summary": {
                "successful_models": len([r for r in self.results if r.get("status") == "SUCCESS"]),
                "failed_models": len([r for r in self.results if r.get("status") == "FAILED"]),
            },
            "results": self.results
        }

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"📄 Detailed results saved to: {output_file}")
        return output_file


async def main():
    """Main function to run multi-model benchmark."""

    import argparse
    parser = argparse.ArgumentParser(description="Multi-Model NSL-KDD Cybersecurity Benchmark")
    parser.add_argument("--models", nargs="+", help="Specific models to test (default: all)")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per model (default: 5)")
    parser.add_argument("--attack-ratio", type=float, default=0.3, help="Ratio of attack samples (default: 0.3)")
    parser.add_argument("--dataset", default="KDDTrain+.txt", help="Dataset file to use")
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with 3 samples")

    args = parser.parse_args()

    if args.quick_test:
        args.samples = 3
        args.attack_ratio = 0.3

    # Validate models
    if args.models:
        invalid_models = [m for m in args.models if m not in FREE_MODELS]
        if invalid_models:
            print(f"❌ Invalid models: {invalid_models}")
            print("Available models:")
            for model, info in FREE_MODELS.items():
                print(f"  - {model} ({info['name']})")
            return 1

    # Initialize benchmark
    benchmark = MultiModelBenchmark(
        models_to_test=args.models,
        sample_size=args.samples,
        attack_ratio=args.attack_ratio,
        dataset_file=args.dataset
    )

    try:
        # Run benchmark
        results = await benchmark.run_all_models()

        # Generate and display report
        summary = benchmark.generate_summary_report()
        print(f"\n{summary}")

        # Save results
        output_file = benchmark.save_results(args.output)

        return 0

    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
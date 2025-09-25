#!/usr/bin/env python3
"""
Local LLM Benchmark Script for LM Studio Integration

This script provides a complete benchmarking solution using local LLMs running on LM Studio.
It includes data loading, model evaluation, explainability analysis, and comprehensive reporting.

Requirements:
- LM Studio running locally (default: http://localhost:1234)
- Models loaded in LM Studio
- Python dependencies: openai, pandas, matplotlib, seaborn

Usage:
    python local_llm_benchmark.py --config config.yaml
    python local_llm_benchmark.py --interactive
    python local_llm_benchmark.py --quick-benchmark
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add src to Python path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

# Import benchmark modules
from benchmark.core.logging import get_logger  # noqa: E402
from benchmark.evaluation.explainability.advanced_analysis import (  # noqa: E402
    AdvancedExplainabilityAnalyzer,
)
from benchmark.evaluation.explainability.explanation_templates import (  # noqa: E402
    ExplanationTemplateGenerator,
)
from benchmark.services.data_service import DataService  # noqa: E402
from benchmark.services.evaluation_service import EvaluationService  # noqa: E402
from benchmark.services.model_service import ModelService  # noqa: E402


class LocalLLMBenchmark:
    """Complete benchmarking solution for local LLMs running on LM Studio."""

    def __init__(self, lm_studio_url: str = "http://localhost:1234"):
        """Initialize the benchmark with LM Studio configuration.

        Args:
            lm_studio_url: URL of your LM Studio server (default: http://localhost:1234)
        """
        self.lm_studio_url = lm_studio_url
        self.logger = get_logger("local_llm_benchmark")

        # Initialize services
        self.model_service: ModelService | None = None
        self.data_service: DataService | None = None
        self.evaluation_service: EvaluationService | None = None
        self.explainability_analyzer: AdvancedExplainabilityAnalyzer | None = None
        self.template_generator: ExplanationTemplateGenerator | None = None

        # Results storage
        self.results: dict[str, Any] = {
            "benchmark_start": None,
            "benchmark_end": None,
            "models": {},
            "datasets": {},
            "predictions": {},
            "evaluations": {},
            "explainability": {},
            "performance": {},
            "costs": {},
        }

    async def initialize_services(self) -> None:
        """Initialize all benchmark services."""
        try:
            self.logger.info("Initializing benchmark services...")

            # Initialize services
            self.model_service = ModelService()
            self.data_service = DataService()
            self.evaluation_service = EvaluationService()

            # Initialize services
            if self.model_service:
                await self.model_service.initialize()
            if self.data_service:
                await self.data_service.initialize()
            if self.evaluation_service:
                await self.evaluation_service.initialize()

            # Initialize explainability components
            self.explainability_analyzer = AdvancedExplainabilityAnalyzer()
            self.template_generator = ExplanationTemplateGenerator()

            self.logger.info("All services initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise

    def create_lm_studio_config(
        self, model_name: str, display_name: str, max_tokens: int = 512, temperature: float = 0.1
    ) -> dict[str, Any]:
        """Create configuration for LM Studio model.

        Args:
            model_name: Name of the model in LM Studio
            display_name: Display name for the benchmark
            max_tokens: Maximum tokens per response
            temperature: Temperature setting

        Returns:
            Model configuration dictionary
        """
        return {
            "name": display_name,
            "type": "openai_api",  # LM Studio uses OpenAI-compatible API
            "model_name": model_name,
            "base_url": f"{self.lm_studio_url}/v1",  # LM Studio OpenAI-compatible endpoint
            "api_key": "lm-studio",  # LM Studio doesn't require real API key
            "max_tokens": max_tokens,
            "temperature": temperature,
            "requests_per_minute": 300,  # Local models can handle more requests
            "tokens_per_minute": 150000,  # Higher token limit for local
            "config": {
                "api_key": "lm-studio"  # Dummy key for local usage
            },
        }

    def create_cybersecurity_dataset_config(
        self, name: str, samples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create dataset configuration for cybersecurity samples.

        Args:
            name: Dataset name
            samples: List of sample dictionaries with 'text' and 'label' keys

        Returns:
            Dataset configuration dictionary
        """
        return {
            "name": name,
            "source": "memory",  # Load from memory
            "samples": samples,
            "max_samples": len(samples),
            "test_split": 0.2,
            "validation_split": 0.1,
        }

    async def load_model(self, model_config: dict[str, Any]) -> str:
        """Load a model into the model service.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Model ID
        """
        try:
            self.logger.info(f"Loading model: {model_config['name']}")

            if not self.model_service:
                raise Exception("Model service not initialized")
            response = await self.model_service.load_model(model_config)
            if not response.success:
                raise Exception(f"Failed to load model: {response.error}")

            model_id: str = model_config["name"]
            self.results["models"][model_id] = {
                "config": model_config,
                "loaded_at": datetime.now().isoformat(),
                "status": "loaded",
            }

            self.logger.info(f"Model loaded successfully: {model_id}")
            return model_id

        except Exception as e:
            self.logger.error(f"Failed to load model {model_config['name']}: {e}")
            raise

    async def load_dataset(self, dataset_config: dict[str, Any]) -> str:
        """Load a dataset into the data service.

        Args:
            dataset_config: Dataset configuration dictionary

        Returns:
            Dataset ID
        """
        try:
            self.logger.info(f"Loading dataset: {dataset_config['name']}")

            # For memory-based datasets, we'll simulate loading
            dataset_id: str = dataset_config["name"]
            self.results["datasets"][dataset_id] = {
                "config": dataset_config,
                "loaded_at": datetime.now().isoformat(),
                "sample_count": len(dataset_config.get("samples", [])),
                "status": "loaded",
            }

            self.logger.info(f"Dataset loaded successfully: {dataset_id}")
            return dataset_id

        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_config['name']}: {e}")
            raise

    async def run_predictions(
        self, model_id: str, dataset_id: str, samples: list[str]
    ) -> list[dict[str, Any]]:
        """Run predictions on a dataset using a model.

        Args:
            model_id: Model identifier
            dataset_id: Dataset identifier
            samples: List of text samples to predict on

        Returns:
            List of prediction dictionaries
        """
        try:
            self.logger.info(f"Running predictions: {model_id} on {dataset_id}")

            start_time = time.time()

            # Get predictions from model service
            if not self.model_service:
                raise Exception("Model service not initialized")
            predictions = await self.model_service.predict(model_id, samples)

            end_time = time.time()
            total_time = end_time - start_time

            # Convert predictions to dictionaries for JSON serialization
            prediction_dicts = [asdict(pred) for pred in predictions]

            # Store results
            key = f"{model_id}_{dataset_id}"
            self.results["predictions"][key] = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "predictions": prediction_dicts,
                "total_time": total_time,
                "samples_per_second": len(samples) / total_time if total_time > 0 else 0,
                "created_at": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Predictions completed: {len(predictions)} samples in {total_time:.2f}s"
            )
            return prediction_dicts

        except Exception as e:
            self.logger.error(f"Failed to run predictions: {e}")
            raise

    async def run_evaluation(
        self,
        model_id: str,
        dataset_id: str,
        predictions: list[dict[str, Any]],
        ground_truth: list[str],
    ) -> dict[str, Any]:
        """Run evaluation on predictions.

        Args:
            model_id: Model identifier
            dataset_id: Dataset identifier
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth labels

        Returns:
            Evaluation results dictionary
        """
        try:
            self.logger.info(f"Running evaluation: {model_id} on {dataset_id}")

            # Run basic metrics evaluation
            metrics = self._calculate_metrics(predictions, ground_truth)

            # Store results
            key = f"{model_id}_{dataset_id}"
            self.results["evaluations"][key] = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "metrics": metrics,
                "evaluation_time": datetime.now().isoformat(),
            }

            self.logger.info(f"Evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Failed to run evaluation: {e}")
            raise

    async def run_explainability_analysis(
        self, model_id: str, dataset_id: str, predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run advanced explainability analysis on predictions.

        Args:
            model_id: Model identifier
            dataset_id: Dataset identifier
            predictions: List of prediction dictionaries

        Returns:
            Explainability analysis results
        """
        try:
            self.logger.info(f"Running explainability analysis: {model_id} on {dataset_id}")

            # Extract explanations from predictions
            explanations = []
            for pred in predictions:
                explanation = pred.get("explanation", "")
                if explanation:
                    explanations.append(
                        {
                            "sample_id": pred.get("sample_id", ""),
                            "explanation": explanation,
                            "prediction": pred.get("prediction", ""),
                            "attack_type": pred.get("attack_type", ""),
                        }
                    )

            if not explanations:
                self.logger.warning("No explanations found in predictions")
                return {"error": "No explanations available for analysis"}

            # Run pattern analysis
            if not self.explainability_analyzer:
                raise Exception("Explainability analyzer not initialized")
            pattern_results = await self.explainability_analyzer.analyze_explanation_patterns(
                explanations
            )

            # Run template evaluation
            if not self.template_generator:
                raise Exception("Template generator not initialized")
            template_results = self.template_generator.evaluate_explanations_batch(explanations)

            # Combine results
            explainability_results = {
                "pattern_analysis": pattern_results,
                "template_evaluation": template_results,
                "total_explanations": len(explanations),
                "analysis_time": datetime.now().isoformat(),
            }

            # Store results
            key = f"{model_id}_{dataset_id}"
            self.results["explainability"][key] = explainability_results

            self.logger.info(
                f"Explainability analysis completed: {len(explanations)} explanations analyzed"
            )
            return explainability_results

        except Exception as e:
            self.logger.error(f"Failed to run explainability analysis: {e}")
            return {"error": str(e)}

    def _calculate_metrics(
        self, predictions: list[dict[str, Any]], ground_truth: list[str]
    ) -> dict[str, Any]:
        """Calculate evaluation metrics.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth labels

        Returns:
            Dictionary of metrics
        """
        try:
            # Extract prediction labels
            pred_labels = [pred.get("prediction", "BENIGN") for pred in predictions]

            # Ensure same length
            min_len = min(len(pred_labels), len(ground_truth))
            pred_labels = pred_labels[:min_len]
            ground_truth = ground_truth[:min_len]

            # Calculate basic metrics
            correct = sum(1 for p, g in zip(pred_labels, ground_truth, strict=False) if p == g)
            total = len(ground_truth)
            accuracy = correct / total if total > 0 else 0.0

            # Calculate attack detection metrics
            attack_predictions = [1 if p == "ATTACK" else 0 for p in pred_labels]
            attack_ground_truth = [1 if g == "ATTACK" else 0 for g in ground_truth]

            tp = sum(
                1
                for p, g in zip(attack_predictions, attack_ground_truth, strict=False)
                if p == 1 and g == 1
            )
            fp = sum(
                1
                for p, g in zip(attack_predictions, attack_ground_truth, strict=False)
                if p == 1 and g == 0
            )
            fn = sum(
                1
                for p, g in zip(attack_predictions, attack_ground_truth, strict=False)
                if p == 0 and g == 1
            )
            tn = sum(
                1
                for p, g in zip(attack_predictions, attack_ground_truth, strict=False)
                if p == 0 and g == 0
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": float(tp),
                "false_positives": float(fp),
                "false_negatives": float(fn),
                "true_negatives": float(tn),
                "total_samples": float(total),
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {e}")
            return {"error": str(e)}

    def generate_sample_cybersecurity_data(self) -> list[dict[str, Any]]:
        """Generate sample cybersecurity data for testing.

        Returns:
            List of sample dictionaries with text and label
        """
        samples = [
            # Attack samples
            {
                "text": "192.168.1.100 attempted SQL injection: ' OR '1'='1' -- on login form",
                "label": "ATTACK",
            },
            {
                "text": "Multiple failed login attempts from 203.0.113.42: admin/password123, admin/admin, root/toor",
                "label": "ATTACK",
            },
            {
                "text": "Suspicious file hash detected: 5d41402abc4b2a76b9719d911017c592 - known malware signature",
                "label": "ATTACK",
            },
            {
                "text": "DDoS attack detected: 10,000 requests/second from botnet IPs targeting server 192.168.1.50",
                "label": "ATTACK",
            },
            {
                "text": "Phishing email detected: fake PayPal login page at http://paypal-secure.malicious.com",
                "label": "ATTACK",
            },
            {
                "text": "Port scan detected from 198.51.100.25: scanning ports 22, 80, 443, 3389 on target 10.0.0.15",
                "label": "ATTACK",
            },
            {
                "text": "Data exfiltration attempt: 500MB upload to external FTP server ftp.malicious.com",
                "label": "ATTACK",
            },
            {
                "text": "Buffer overflow exploit attempt in HTTP request: AAAA...AAAA (1024 characters)",
                "label": "ATTACK",
            },
            # Benign samples
            {
                "text": "User john.doe successfully logged in from 192.168.1.55 at 09:15 AM",
                "label": "BENIGN",
            },
            {
                "text": "Scheduled backup completed successfully: 2.5GB backed up to network storage",
                "label": "BENIGN",
            },
            {
                "text": "Software update installed: Security patch KB4052623 applied to Windows servers",
                "label": "BENIGN",
            },
            {
                "text": "Normal web traffic: GET /index.html HTTP/1.1 from legitimate user browser",
                "label": "BENIGN",
            },
            {
                "text": "Database maintenance: Index rebuild completed on customer_data table",
                "label": "BENIGN",
            },
            {
                "text": "Email sent successfully: Weekly report from accounting@company.com to management team",
                "label": "BENIGN",
            },
            {
                "text": "System health check: All services running normally, CPU usage 25%, Memory 60%",
                "label": "BENIGN",
            },
            {
                "text": "File transfer completed: quarterly_report.pdf uploaded to shared drive",
                "label": "BENIGN",
            },
        ]

        return samples

    async def run_quick_benchmark(self, model_name: str = "local-model") -> None:
        """Run a quick benchmark with sample data.

        Args:
            model_name: Name of the model in LM Studio
        """
        try:
            self.logger.info("Starting quick benchmark...")
            self.results["benchmark_start"] = datetime.now().isoformat()

            # Create model configuration
            model_config = self.create_lm_studio_config(
                model_name=model_name,
                display_name="LM_Studio_Local_Model",
                max_tokens=256,
                temperature=0.1,
            )

            # Generate sample data
            sample_data = self.generate_sample_cybersecurity_data()
            dataset_config = self.create_cybersecurity_dataset_config(
                name="cybersecurity_samples", samples=sample_data
            )

            # Load model and dataset
            model_id = await self.load_model(model_config)
            dataset_id = await self.load_dataset(dataset_config)

            # Extract samples and labels
            samples = [item["text"] for item in sample_data]
            ground_truth = [item["label"] for item in sample_data]

            # Run predictions
            predictions = await self.run_predictions(model_id, dataset_id, samples)

            # Run evaluation
            await self.run_evaluation(model_id, dataset_id, predictions, ground_truth)

            # Run explainability analysis
            await self.run_explainability_analysis(model_id, dataset_id, predictions)

            self.results["benchmark_end"] = datetime.now().isoformat()

            # Generate report
            self.generate_benchmark_report()

            self.logger.info("Quick benchmark completed successfully!")

        except Exception as e:
            self.logger.error(f"Quick benchmark failed: {e}")
            raise

    async def run_full_benchmark(self, config_file: str) -> None:
        """Run a full benchmark from configuration file.

        Args:
            config_file: Path to configuration file
        """
        try:
            self.logger.info(f"Starting full benchmark from config: {config_file}")
            self.results["benchmark_start"] = datetime.now().isoformat()

            # Load configuration
            with open(config_file) as f:
                config = yaml.safe_load(f)

            # Process models
            for model_config in config.get("models", []):
                # Convert to LM Studio configuration
                lm_studio_config = self.create_lm_studio_config(
                    model_name=model_config.get("model_name", "local-model"),
                    display_name=model_config.get("name", "local-model"),
                    max_tokens=model_config.get("max_tokens", 512),
                    temperature=model_config.get("temperature", 0.1),
                )

                model_id = await self.load_model(lm_studio_config)

                # Process datasets for this model
                for dataset_config in config.get("datasets", []):
                    dataset_id = await self.load_dataset(dataset_config)

                    # Load actual data (this would be customized based on your data format)
                    samples, ground_truth = self._load_dataset_samples(dataset_config)

                    # Run predictions
                    predictions = await self.run_predictions(model_id, dataset_id, samples)

                    # Run evaluation
                    await self.run_evaluation(model_id, dataset_id, predictions, ground_truth)

                    # Run explainability analysis
                    await self.run_explainability_analysis(model_id, dataset_id, predictions)

            self.results["benchmark_end"] = datetime.now().isoformat()

            # Generate comprehensive report
            self.generate_benchmark_report()

            self.logger.info("Full benchmark completed successfully!")

        except Exception as e:
            self.logger.error(f"Full benchmark failed: {e}")
            raise

    def _load_dataset_samples(self, dataset_config: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Load dataset samples from configuration.

        Args:
            dataset_config: Dataset configuration

        Returns:
            Tuple of (samples, ground_truth_labels)
        """
        # This is a placeholder - customize based on your data format
        if "samples" in dataset_config:
            # Memory-based dataset
            samples = [item["text"] for item in dataset_config["samples"]]
            ground_truth = [item["label"] for item in dataset_config["samples"]]
        elif "path" in dataset_config:
            # File-based dataset - implement your file loading logic here
            samples = []
            ground_truth = []
            # TODO: Add file loading logic based on your data format
        else:
            # Default to sample data
            sample_data = self.generate_sample_cybersecurity_data()
            samples = [item["text"] for item in sample_data]
            ground_truth = [item["label"] for item in sample_data]

        return samples, ground_truth

    def generate_benchmark_report(self) -> None:
        """Generate a comprehensive benchmark report."""
        try:
            report_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Save detailed results
            with open(report_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            # Generate summary report
            summary_file = f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(summary_file, "w") as f:
                f.write("=== LM Studio Local LLM Benchmark Report ===\n\n")

                # Benchmark info
                f.write(f"Benchmark Start: {self.results['benchmark_start']}\n")
                f.write(f"Benchmark End: {self.results['benchmark_end']}\n")
                f.write(f"LM Studio URL: {self.lm_studio_url}\n\n")

                # Models summary
                f.write(f"Models Tested: {len(self.results['models'])}\n")
                for model_id, model_info in self.results["models"].items():
                    f.write(f"  - {model_id}: {model_info['status']}\n")
                f.write("\n")

                # Datasets summary
                f.write(f"Datasets Used: {len(self.results['datasets'])}\n")
                for dataset_id, dataset_info in self.results["datasets"].items():
                    f.write(f"  - {dataset_id}: {dataset_info['sample_count']} samples\n")
                f.write("\n")

                # Evaluation results
                f.write("=== EVALUATION RESULTS ===\n")
                for key, eval_result in self.results["evaluations"].items():
                    metrics = eval_result["metrics"]
                    f.write(f"\n{key}:\n")
                    f.write(f"  Accuracy: {metrics.get('accuracy', 0):.3f}\n")
                    f.write(f"  Precision: {metrics.get('precision', 0):.3f}\n")
                    f.write(f"  Recall: {metrics.get('recall', 0):.3f}\n")
                    f.write(f"  F1-Score: {metrics.get('f1_score', 0):.3f}\n")

                # Explainability results
                f.write("\n=== EXPLAINABILITY ANALYSIS ===\n")
                for key, exp_result in self.results["explainability"].items():
                    if "error" not in exp_result:
                        f.write(f"\n{key}:\n")
                        f.write(
                            f"  Total Explanations: {exp_result.get('total_explanations', 0)}\n"
                        )

                        # Pattern analysis
                        pattern_analysis = exp_result.get("pattern_analysis", {})
                        if pattern_analysis:
                            f.write(
                                f"  Pattern Analysis: {len(pattern_analysis.get('patterns_by_type', {}))} attack types analyzed\n"
                            )

                        # Template evaluation
                        template_eval = exp_result.get("template_evaluation", {})
                        if template_eval:
                            avg_score = template_eval.get("average_score", 0)
                            f.write(f"  Template Evaluation: {avg_score:.3f} average score\n")

                # Performance summary
                f.write("\n=== PERFORMANCE SUMMARY ===\n")
                for key, pred_result in self.results["predictions"].items():
                    f.write(f"\n{key}:\n")
                    f.write(f"  Processing Time: {pred_result.get('total_time', 0):.2f} seconds\n")
                    f.write(f"  Samples/Second: {pred_result.get('samples_per_second', 0):.2f}\n")

            self.logger.info(f"Benchmark report saved: {report_file}")
            self.logger.info(f"Summary report saved: {summary_file}")

            # Print summary to console
            print("\n=== BENCHMARK COMPLETED ===")
            print(f"Detailed Report: {report_file}")
            print(f"Summary Report: {summary_file}")
            print(f"Models Tested: {len(self.results['models'])}")
            print(f"Datasets Used: {len(self.results['datasets'])}")
            print(f"LM Studio URL: {self.lm_studio_url}")

        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.model_service:
                await self.model_service.shutdown()
            if self.data_service:
                await self.data_service.shutdown()
            if self.evaluation_service:
                await self.evaluation_service.shutdown()

            self.logger.info("Cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


async def main() -> None:
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Local LLM Benchmark for LM Studio")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--lm-studio-url",
        type=str,
        default="http://localhost:1234",
        help="LM Studio server URL (default: http://localhost:1234)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="local-model",
        help="Model name in LM Studio (default: local-model)",
    )
    parser.add_argument(
        "--quick-benchmark", action="store_true", help="Run quick benchmark with sample data"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode for configuration"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = get_logger("main")

    # Create benchmark instance
    benchmark = LocalLLMBenchmark(lm_studio_url=args.lm_studio_url)

    try:
        # Initialize services
        await benchmark.initialize_services()

        if args.quick_benchmark:
            # Run quick benchmark
            await benchmark.run_quick_benchmark(model_name=args.model_name)

        elif args.config:
            # Run full benchmark from config
            await benchmark.run_full_benchmark(args.config)

        elif args.interactive:
            # Interactive mode
            print("=== Interactive LM Studio Benchmark ===")
            print(f"LM Studio URL: {args.lm_studio_url}")

            model_name = input(f"Enter model name in LM Studio [{args.model_name}]: ").strip()
            if not model_name:
                model_name = args.model_name

            print("\nRunning benchmark with your settings...")
            await benchmark.run_quick_benchmark(model_name=model_name)

        else:
            # Default: quick benchmark
            print("No specific mode selected. Running quick benchmark...")
            print("Use --help for more options")
            await benchmark.run_quick_benchmark(model_name=args.model_name)

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
    finally:
        # Cleanup
        await benchmark.cleanup()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

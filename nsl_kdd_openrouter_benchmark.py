#!/usr/bin/env python3
"""
NSL-KDD Network Intrusion Detection Benchmark with OpenRouter API

This script provides a specialized benchmarking solution for network intrusion detection
using the NSL-KDD dataset and OpenRouter API (x-ai/grok-4-fast:free).

Requirements:
- OpenRouter API key (OPENROUTER_API_KEY environment variable)
- NSL-KDD dataset in the NSL-KDD folder
- Python dependencies: openai, pandas, numpy, scikit-learn

Usage:
    export OPENROUTER_API_KEY="your-api-key"
    python nsl_kdd_openrouter_benchmark.py --quick-test
    python nsl_kdd_openrouter_benchmark.py --max-samples 10
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class NetworkConnection:
    """Represents a network connection from NSL-KDD dataset."""

    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float
    attack_type: str

    def to_readable_text(self) -> str:
        """Convert network connection to human-readable text for LLM analysis."""

        # Build readable description
        description_parts = []

        # Basic connection info
        description_parts.append(f"Network connection: {self.protocol_type.upper()} protocol")
        description_parts.append(f"Service: {self.service}")
        description_parts.append(f"Connection state: {self.flag}")

        # Data transfer information
        if self.src_bytes > 0 or self.dst_bytes > 0:
            description_parts.append(f"Data transfer: {self.src_bytes} bytes sent, {self.dst_bytes} bytes received")

        # Duration
        if self.duration > 0:
            description_parts.append(f"Duration: {self.duration} seconds")

        # Connection counts
        if self.count > 1:
            description_parts.append(f"Connection count in time window: {self.count}")
        if self.srv_count > 1:
            description_parts.append(f"Same service connections: {self.srv_count}")

        # Error rates (significant indicators)
        if self.serror_rate > 0:
            description_parts.append(f"SYN error rate: {self.serror_rate:.2f}")
        if self.rerror_rate > 0:
            description_parts.append(f"REJ error rate: {self.rerror_rate:.2f}")

        # Login attempts and failures
        if self.num_failed_logins > 0:
            description_parts.append(f"Failed login attempts: {self.num_failed_logins}")
        if self.logged_in == 1:
            description_parts.append("Successful login")

        # Suspicious activities
        if self.num_compromised > 0:
            description_parts.append(f"Compromised conditions: {self.num_compromised}")
        if self.root_shell > 0:
            description_parts.append(f"Root shell access: {self.root_shell}")
        if self.su_attempted > 0:
            description_parts.append(f"Su command attempts: {self.su_attempted}")

        # Host-based features
        if self.dst_host_count > 0:
            description_parts.append(f"Destination host connections: {self.dst_host_count}")

        # Special flags
        if self.land == 1:
            description_parts.append("Land attack indicator detected")
        if self.wrong_fragment > 0:
            description_parts.append(f"Wrong fragments: {self.wrong_fragment}")
        if self.urgent > 0:
            description_parts.append(f"Urgent packets: {self.urgent}")
        if self.hot > 0:
            description_parts.append(f"Hot indicators: {self.hot}")

        return ". ".join(description_parts) + "."


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(self, api_key: str = None, model_name: str = "x-ai/grok-4-fast:free"):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    async def predict_intrusion(self, connection: NetworkConnection) -> dict[str, Any]:
        """Predict if a network connection is an intrusion."""

        # Create prompt for network intrusion detection
        prompt = self._create_intrusion_detection_prompt(connection)

        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Generate response using OpenRouter
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a cybersecurity expert analyzing network traffic for intrusion detection. Respond with NORMAL or ATTACK followed by a detailed explanation.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0.1,
                )

                end_time = time.time()
                response_time = end_time - start_time

                # Extract response text
                response_text = response.choices[0].message.content.strip()

                # Parse prediction
                prediction_result = self._parse_prediction(response_text, response_time)

                return prediction_result

            except Exception as e:
                error_message = str(e)
                if "429" in error_message or "rate" in error_message.lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error("Rate limit exceeded, max retries reached")
                else:
                    logger.error(f"Error in OpenRouter prediction: {e}")

                return {
                    "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "response_time": 0.0,
                    "raw_response": "",
                }

    def _create_intrusion_detection_prompt(self, connection: NetworkConnection) -> str:
        """Create a cybersecurity-focused prompt for network intrusion detection."""

        readable_connection = connection.to_readable_text()

        prompt = f"""You are a cybersecurity expert analyzing network traffic for intrusion detection.

Analyze the following network connection for potential security threats:

{readable_connection}

Task: Determine if this network connection represents:
- NORMAL: Legitimate network activity
- ATTACK: Malicious or suspicious activity

Please respond with your classification (NORMAL or ATTACK) followed by detailed reasoning explaining:
1. Key indicators that support your decision
2. Specific security concerns or normal patterns observed
3. Attack type if malicious (e.g., DoS, Probe, R2L, U2R)

Response format:
PREDICTION: [NORMAL/ATTACK]
REASONING: [Your detailed analysis]"""

        return prompt

    def _parse_prediction(self, response_text: str, response_time: float) -> dict[str, Any]:
        """Parse the LLM response and extract structured prediction data."""

        # Clean response
        response_text = response_text.strip()

        # Extract prediction
        prediction = "UNKNOWN"
        confidence = 0.5
        reasoning = response_text

        # Look for explicit prediction markers
        lines = response_text.split('\n')
        for line in lines:
            line_upper = line.upper()
            if 'PREDICTION:' in line_upper:
                if 'ATTACK' in line_upper:
                    prediction = "ATTACK"
                elif 'NORMAL' in line_upper:
                    prediction = "NORMAL"
            elif line_upper.startswith('REASONING:'):
                reasoning = line[10:].strip()

        # If no explicit markers, look for ATTACK or NORMAL at start of response
        if prediction == "UNKNOWN":
            if response_text.upper().startswith("ATTACK"):
                prediction = "ATTACK"
            elif response_text.upper().startswith("NORMAL"):
                prediction = "NORMAL"
            elif "ATTACK" in response_text.upper()[:50]:
                prediction = "ATTACK"
            elif "NORMAL" in response_text.upper()[:50]:
                prediction = "NORMAL"

        # Estimate confidence based on response characteristics
        confidence_indicators = 0
        if "definitely" in response_text.lower() or "clearly" in response_text.lower():
            confidence_indicators += 0.2
        if "suspicious" in response_text.lower() or "malicious" in response_text.lower():
            confidence_indicators += 0.1
        if "indicators" in response_text.lower() or "evidence" in response_text.lower():
            confidence_indicators += 0.1

        if prediction == "ATTACK":
            confidence = min(0.9, 0.6 + confidence_indicators)
        elif prediction == "NORMAL":
            confidence = min(0.9, 0.7 + confidence_indicators)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": reasoning,
            "response_time": response_time,
            "raw_response": response_text,
        }


class NSLKDDBenchmarkWithOpenRouter:
    """Main benchmarking class for NSL-KDD intrusion detection with OpenRouter API."""

    def __init__(self, api_key: str = None, model_name: str = "x-ai/grok-4-fast:free"):
        self.openrouter_client = OpenRouterClient(api_key, model_name)
        self.model_name = model_name

        # NSL-KDD feature names (standard format without difficulty column)
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
        ]

    def load_nsl_kdd_data(self, filename: str = "KDDTrain+.txt", max_samples: int = 100, attack_ratio: float = 0.3) -> list[NetworkConnection]:
        """Load and sample data from NSL-KDD dataset."""

        file_path = Path("NSL-KDD") / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading NSL-KDD data from {file_path}")

        connections = []
        attack_connections = []
        normal_connections = []

        # Read data
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) != len(self.feature_names):
                    continue

                # Parse numeric values
                parsed_row = []
                for i, value in enumerate(row):
                    if i in [1, 2, 3, 41]:  # String fields: protocol_type, service, flag, attack_type (index 41 for 42 columns)
                        parsed_row.append(value.strip())
                    else:
                        try:
                            if '.' in value:
                                parsed_row.append(float(value))
                            else:
                                parsed_row.append(int(value))
                        except ValueError:
                            parsed_row.append(0)

                connection = NetworkConnection(*parsed_row)

                # Separate normal and attack connections
                if connection.attack_type.lower() == 'normal':
                    normal_connections.append(connection)
                else:
                    attack_connections.append(connection)

        # Sample data based on requested ratio
        target_attacks = int(max_samples * attack_ratio)
        target_normal = max_samples - target_attacks

        # Sample connections
        sampled_attacks = random.sample(attack_connections, min(target_attacks, len(attack_connections)))
        sampled_normal = random.sample(normal_connections, min(target_normal, len(normal_connections)))

        connections = sampled_attacks + sampled_normal
        random.shuffle(connections)

        logger.info(f"Loaded {len(connections)} connections: {len(sampled_attacks)} attacks, {len(sampled_normal)} normal")
        return connections

    async def run_benchmark(self, connections: list[NetworkConnection]) -> dict[str, Any]:
        """Run the benchmark on a list of network connections."""

        logger.info(f"Starting benchmark with {len(connections)} connections using OpenRouter {self.model_name}")

        predictions = []
        actual_labels = []
        response_times = []
        attack_types = {}

        start_time = time.time()

        for i, connection in enumerate(connections):
            logger.info(f"Processing connection {i+1}/{len(connections)}")

            # Get prediction
            prediction_result = await self.openrouter_client.predict_intrusion(connection)

            # Store results
            predictions.append(prediction_result)

            # Convert actual label for evaluation
            actual_label = "NORMAL" if connection.attack_type.lower() == "normal" else "ATTACK"
            actual_labels.append(actual_label)

            response_times.append(prediction_result["response_time"])

            # Track attack types
            if actual_label == "ATTACK":
                attack_type = connection.attack_type
                if attack_type not in attack_types:
                    attack_types[attack_type] = {"correct": 0, "total": 0}
                attack_types[attack_type]["total"] += 1
                if prediction_result["prediction"] == "ATTACK":
                    attack_types[attack_type]["correct"] += 1

            # Add delay to respect API rate limits
            await asyncio.sleep(0.5)  # Reduced delay for faster testing

        total_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, actual_labels, response_times, attack_types, total_time)

        # Generate report
        report = {
            "benchmark_info": {
                "model_name": self.model_name,
                "total_samples": len(connections),
                "attack_samples": sum(1 for label in actual_labels if label == "ATTACK"),
                "normal_samples": sum(1 for label in actual_labels if label == "NORMAL"),
                "attack_types": list(attack_types.keys()),
                "timestamp": datetime.now().isoformat(),
                "total_processing_time": total_time,
            },
            "metrics": metrics,
            "predictions": [
                {
                    "prediction": pred["prediction"],
                    "actual": actual,
                    "confidence": pred["confidence"],
                    "reasoning": pred["reasoning"],
                    "response_time": pred["response_time"],
                }
                for pred, actual in zip(predictions, actual_labels)
            ],
            "attack_type_performance": attack_types,
        }

        return report

    def _calculate_metrics(self, predictions: list[dict], actual_labels: list[str], response_times: list[float], attack_types: dict, total_time: float) -> dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""

        # Extract predictions
        pred_labels = [pred["prediction"] for pred in predictions]

        # Basic counts
        tp = sum(1 for pred, actual in zip(pred_labels, actual_labels) if pred == "ATTACK" and actual == "ATTACK")
        fp = sum(1 for pred, actual in zip(pred_labels, actual_labels) if pred == "ATTACK" and actual == "NORMAL")
        tn = sum(1 for pred, actual in zip(pred_labels, actual_labels) if pred == "NORMAL" and actual == "NORMAL")
        fn = sum(1 for pred, actual in zip(pred_labels, actual_labels) if pred == "NORMAL" and actual == "ATTACK")

        total = len(predictions)

        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Advanced metrics
        balanced_accuracy = (recall + specificity) / 2
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

        # Performance metrics
        avg_response_time = np.mean(response_times) if response_times else 0
        median_response_time = np.median(response_times) if response_times else 0
        min_response_time = np.min(response_times) if response_times else 0
        max_response_time = np.max(response_times) if response_times else 0
        predictions_per_second = total / total_time if total_time > 0 else 0

        return {
            "classification_metrics": {
                "overall_accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "f1_score": f1_score,
            },
            "advanced_metrics": {
                "matthews_correlation": mcc,
            },
            "error_rates": {
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
            },
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_samples": total,
            },
            "performance_metrics": {
                "average_response_time": avg_response_time,
                "median_response_time": median_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "predictions_per_second": predictions_per_second,
                "total_processing_time": total_time,
            },
        }

    def print_results_summary(self, report: dict[str, Any]) -> None:
        """Print a comprehensive summary of benchmark results."""

        info = report["benchmark_info"]
        metrics = report["metrics"]
        attack_performance = report.get("attack_type_performance", {})

        print("\n" + "="*60)
        print("NSL-KDD OPENROUTER INTRUSION DETECTION BENCHMARK SUMMARY")
        print("="*60)

        print(f"\nModel: {info['model_name']}")
        print(f"Dataset: NSL-KDD")
        print(f"Sample size: {info['total_samples']} connections")
        print(f"Normal connections: {info['normal_samples']}")
        print(f"Attack connections: {info['attack_samples']}")
        if info['attack_types']:
            print(f"Attack types: {', '.join(info['attack_types'][:5])}")
            if len(info['attack_types']) > 5:
                print(f"... and {len(info['attack_types']) - 5} more")

        print("\n" + "="*50)
        print("CLASSIFICATION METRICS")
        print("="*50)
        cm = metrics["classification_metrics"]
        print(f"Overall Accuracy:        {cm['overall_accuracy']:.3f}")
        print(f"Balanced Accuracy:       {cm['balanced_accuracy']:.3f}")
        print(f"Precision:               {cm['precision']:.3f}")
        print(f"Recall (Sensitivity):    {cm['recall_sensitivity']:.3f}")
        print(f"Specificity:             {cm['specificity']:.3f}")
        print(f"F1-Score:                {cm['f1_score']:.3f}")

        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        pm = metrics["performance_metrics"]
        print(f"Average Response Time:   {pm['average_response_time']:.3f}s")
        print(f"Median Response Time:    {pm['median_response_time']:.3f}s")
        print(f"Predictions per Second:  {pm['predictions_per_second']:.2f}")
        print(f"Total Processing Time:   {pm['total_processing_time']:.2f}s")
        print("\n" + "="*60)

    def save_results(self, report: dict[str, Any], output_file: str = None) -> str:
        """Save benchmark results to JSON file."""

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_clean = self.model_name.replace("/", "_").replace("-", "_").replace(":", "_")
            output_file = f"nsl_kdd_benchmark_{model_name_clean}_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")
        return output_file


async def main():
    """Main function to run the NSL-KDD benchmark with OpenRouter."""

    parser = argparse.ArgumentParser(description="NSL-KDD Intrusion Detection Benchmark with OpenRouter API")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model-name", default="x-ai/grok-4-fast:free", help="OpenRouter model to use (default: x-ai/grok-4-fast:free)")
    parser.add_argument("--dataset-file", default="KDDTrain+.txt", help="NSL-KDD dataset file to use")
    parser.add_argument("--max-samples", type=int, default=2, help="Maximum number of samples to test (default: 2 for testing)")
    parser.add_argument("--attack-ratio", type=float, default=0.5, help="Ratio of attack samples (0.0-1.0)")
    parser.add_argument("--output-file", help="Output file for results (auto-generated if not specified)")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with 2 samples")

    args = parser.parse_args()

    # Handle quick test flag
    if args.quick_test:
        args.max_samples = 2
        args.attack_ratio = 0.5

    print("NSL-KDD Intrusion Detection Benchmark with OpenRouter")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Max samples: {args.max_samples}")
    print(f"Attack ratio: {args.attack_ratio}")
    print("=" * 60)

    try:
        # Initialize benchmark
        benchmark = NSLKDDBenchmarkWithOpenRouter(
            api_key=args.api_key,
            model_name=args.model_name
        )

        # Load data
        connections = benchmark.load_nsl_kdd_data(
            filename=args.dataset_file,
            max_samples=args.max_samples,
            attack_ratio=args.attack_ratio
        )

        # Run benchmark
        report = await benchmark.run_benchmark(connections)

        # Print results
        benchmark.print_results_summary(report)

        # Save results
        output_file = benchmark.save_results(report, args.output_file)
        print(f"\n✅ Results saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\n❌ Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
#!/usr/bin/env python3
"""
NSL-KDD Network Intrusion Detection Benchmark for LM Studio

This script provides a specialized benchmarking solution for network intrusion detection
using the NSL-KDD dataset and local LLMs running on LM Studio.

Requirements:
- LM Studio running locally (default: http://localhost:1234)
- Models loaded in LM Studio
- NSL-KDD dataset in the NSL-KDD folder
- Python dependencies: openai, pandas, numpy, scikit-learn

Usage:
    python nsl_kdd_lm_studio_benchmark.py --model-name "your-model"
    python nsl_kdd_lm_studio_benchmark.py --quick-test
    python nsl_kdd_lm_studio_benchmark.py --full-benchmark --max-samples 1000
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

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
    difficulty: int

    def to_readable_text(self) -> str:
        """Convert network connection to human-readable text for LLM analysis."""

        # Build readable description
        description_parts = []

        # Basic connection info
        description_parts.append(f"Network connection: {self.protocol_type.upper()} protocol")
        description_parts.append(f"Service: {self.service}")
        description_parts.append(f"Connection state: {self.flag}")

        # Data transfer
        if self.src_bytes > 0 or self.dst_bytes > 0:
            description_parts.append(
                f"Data transfer: {self.src_bytes} bytes sent, {self.dst_bytes} bytes received"
            )

        # Duration
        if self.duration > 0:
            description_parts.append(f"Duration: {self.duration} seconds")

        # Security-relevant features
        if self.num_failed_logins > 0:
            description_parts.append(f"Failed login attempts: {self.num_failed_logins}")

        if self.hot > 0:
            description_parts.append(f"Hot indicators: {self.hot}")

        if self.num_compromised > 0:
            description_parts.append(f"Compromised conditions: {self.num_compromised}")

        if self.root_shell > 0:
            description_parts.append("Root shell obtained")

        if self.su_attempted > 0:
            description_parts.append("Su attempted")

        # Connection patterns
        if self.count > 1:
            description_parts.append(f"Connection count in time window: {self.count}")

        if self.srv_count > 1:
            description_parts.append(f"Same service connections: {self.srv_count}")

        # Error rates
        if self.serror_rate > 0:
            description_parts.append(f"SYN error rate: {self.serror_rate:.2f}")

        if self.rerror_rate > 0:
            description_parts.append(f"REJ error rate: {self.rerror_rate:.2f}")

        # Host-based features
        if self.dst_host_count > 1:
            description_parts.append(f"Destination host connections: {self.dst_host_count}")

        if self.dst_host_serror_rate > 0:
            description_parts.append(f"Host SYN error rate: {self.dst_host_serror_rate:.2f}")

        return ". ".join(description_parts) + "."


class NSLKDDLoader:
    """Loads and preprocesses NSL-KDD dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.feature_names = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "attack_type",
            "difficulty",
        ]

    def load_data(self, filename: str, max_samples: int | None = None) -> list[NetworkConnection]:
        """Load NSL-KDD data from CSV file."""
        file_path = self.dataset_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        connections = []

        logger.info(f"Loading data from {file_path}")

        with open(file_path) as f:
            reader = csv.reader(f)

            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break

                try:
                    # Parse row data
                    connection_data: dict[str, Any] = {}

                    # Convert numeric fields
                    connection_data["duration"] = float(row[0])
                    connection_data["protocol_type"] = str(row[1])
                    connection_data["service"] = str(row[2])
                    connection_data["flag"] = str(row[3])
                    connection_data["src_bytes"] = int(row[4])
                    connection_data["dst_bytes"] = int(row[5])
                    connection_data["land"] = int(row[6])
                    connection_data["wrong_fragment"] = int(row[7])
                    connection_data["urgent"] = int(row[8])
                    connection_data["hot"] = int(row[9])
                    connection_data["num_failed_logins"] = int(row[10])
                    connection_data["logged_in"] = int(row[11])
                    connection_data["num_compromised"] = int(row[12])
                    connection_data["root_shell"] = int(row[13])
                    connection_data["su_attempted"] = int(row[14])
                    connection_data["num_root"] = int(row[15])
                    connection_data["num_file_creations"] = int(row[16])
                    connection_data["num_shells"] = int(row[17])
                    connection_data["num_access_files"] = int(row[18])
                    connection_data["num_outbound_cmds"] = int(row[19])
                    connection_data["is_host_login"] = int(row[20])
                    connection_data["is_guest_login"] = int(row[21])
                    connection_data["count"] = int(row[22])
                    connection_data["srv_count"] = int(row[23])
                    connection_data["serror_rate"] = float(row[24])
                    connection_data["srv_serror_rate"] = float(row[25])
                    connection_data["rerror_rate"] = float(row[26])
                    connection_data["srv_rerror_rate"] = float(row[27])
                    connection_data["same_srv_rate"] = float(row[28])
                    connection_data["diff_srv_rate"] = float(row[29])
                    connection_data["srv_diff_host_rate"] = float(row[30])
                    connection_data["dst_host_count"] = int(row[31])
                    connection_data["dst_host_srv_count"] = int(row[32])
                    connection_data["dst_host_same_srv_rate"] = float(row[33])
                    connection_data["dst_host_diff_srv_rate"] = float(row[34])
                    connection_data["dst_host_same_src_port_rate"] = float(row[35])
                    connection_data["dst_host_srv_diff_host_rate"] = float(row[36])
                    connection_data["dst_host_serror_rate"] = float(row[37])
                    connection_data["dst_host_srv_serror_rate"] = float(row[38])
                    connection_data["dst_host_rerror_rate"] = float(row[39])
                    connection_data["dst_host_srv_rerror_rate"] = float(row[40])
                    connection_data["attack_type"] = str(row[41])
                    connection_data["difficulty"] = int(row[42])

                    connection = NetworkConnection(**connection_data)
                    connections.append(connection)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed row {i}: {e}")
                    continue

        logger.info(f"Loaded {len(connections)} network connections")
        return connections

    def get_balanced_sample(
        self, connections: list[NetworkConnection], sample_size: int, attack_ratio: float = 0.3
    ) -> list[NetworkConnection]:
        """Get a balanced sample of normal and attack connections."""

        # Separate normal and attack connections
        normal_connections = [c for c in connections if c.attack_type == "normal"]
        attack_connections = [c for c in connections if c.attack_type != "normal"]

        # Calculate sample sizes
        attack_samples = min(int(sample_size * attack_ratio), len(attack_connections))
        normal_samples = min(sample_size - attack_samples, len(normal_connections))

        # Random sampling
        random.shuffle(normal_connections)
        random.shuffle(attack_connections)

        balanced_sample = normal_connections[:normal_samples] + attack_connections[:attack_samples]

        # Shuffle the final sample
        random.shuffle(balanced_sample)

        logger.info(
            f"Created balanced sample: {normal_samples} normal, {attack_samples} attack connections"
        )
        return balanced_sample


class LMStudioClient:
    """Client for interacting with LM Studio API."""

    def __init__(self, base_url: str = "http://localhost:1234"):
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="lm-studio",  # LM Studio doesn't require real API key
        )
        self.base_url = base_url

    async def predict_intrusion(self, connection: NetworkConnection) -> dict[str, Any]:
        """Predict if a network connection is an intrusion."""

        # Create prompt for network intrusion detection
        prompt = self._create_intrusion_detection_prompt(connection)

        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model="local-model",  # LM Studio uses this as default
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

            # Parse response
            content = response.choices[0].message.content.strip()

            # Extract prediction and explanation
            if content.upper().startswith("NORMAL"):
                prediction = "NORMAL"
                explanation = content[6:].strip()
            elif content.upper().startswith("ATTACK"):
                prediction = "ATTACK"
                explanation = content[6:].strip()
            else:
                # Fallback parsing
                if "normal" in content.lower() and "attack" not in content.lower():
                    prediction = "NORMAL"
                elif "attack" in content.lower():
                    prediction = "ATTACK"
                else:
                    prediction = "UNKNOWN"
                explanation = content

            return {
                "prediction": prediction,
                "explanation": explanation,
                "response_time": end_time - start_time,
                "model_response": content,
                "ground_truth": "NORMAL" if connection.attack_type == "normal" else "ATTACK",
                "attack_type": connection.attack_type,
                "difficulty": connection.difficulty,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "prediction": "ERROR",
                "explanation": f"Error during prediction: {str(e)}",
                "response_time": 0,
                "model_response": "",
                "ground_truth": "NORMAL" if connection.attack_type == "normal" else "ATTACK",
                "attack_type": connection.attack_type,
                "difficulty": connection.difficulty,
            }

    def _create_intrusion_detection_prompt(self, connection: NetworkConnection) -> str:
        """Create a prompt for intrusion detection analysis."""

        readable_text = connection.to_readable_text()

        prompt = f"""Analyze the following network connection for potential security threats:

{readable_text}

Task: Determine if this network connection represents:
- NORMAL: Legitimate network activity
- ATTACK: Malicious or suspicious activity (intrusion attempt)

Please respond with your classification (NORMAL or ATTACK) followed by a detailed explanation of your reasoning, including:
1. Key indicators that influenced your decision
2. Specific features that are suspicious or normal
3. The type of potential attack if you classify it as ATTACK
4. Your confidence level in the assessment

Response format: [NORMAL/ATTACK] - [Your detailed explanation]"""

        return prompt


class NSLKDDBenchmark:
    """Main benchmark class for NSL-KDD intrusion detection evaluation."""

    def __init__(self, lm_studio_url: str = "http://localhost:1234"):
        self.lm_studio_client = LMStudioClient(lm_studio_url)
        self.data_loader = NSLKDDLoader("NSL-KDD")
        self.results: dict[str, Any] = {
            "benchmark_start": None,
            "benchmark_end": None,
            "model_url": lm_studio_url,
            "dataset_info": {},
            "predictions": [],
            "metrics": {},
            "performance": {},
        }

    async def run_benchmark(
        self, filename: str = "KDDTrain+.txt", max_samples: int = 100, attack_ratio: float = 0.3
    ) -> dict[str, Any]:
        """Run the complete NSL-KDD benchmark."""

        logger.info("Starting NSL-KDD intrusion detection benchmark")
        self.results["benchmark_start"] = datetime.now().isoformat()

        # Load data
        logger.info(f"Loading NSL-KDD data from {filename}")
        all_connections = self.data_loader.load_data(filename)

        # Get balanced sample
        connections = self.data_loader.get_balanced_sample(
            all_connections, max_samples, attack_ratio
        )

        # Store dataset info
        self.results["dataset_info"] = {
            "filename": filename,
            "total_connections": len(all_connections),
            "sample_size": len(connections),
            "attack_ratio": attack_ratio,
            "normal_count": len([c for c in connections if c.attack_type == "normal"]),
            "attack_count": len([c for c in connections if c.attack_type != "normal"]),
            "attack_types": list({c.attack_type for c in connections if c.attack_type != "normal"}),
        }

        # Run predictions
        logger.info(f"Running predictions on {len(connections)} connections")
        predictions: list[dict[str, Any]] = []

        for i, connection in enumerate(connections):
            if i % 10 == 0:
                logger.info(f"Processing connection {i + 1}/{len(connections)}")

            prediction = await self.lm_studio_client.predict_intrusion(connection)
            prediction["sample_id"] = i
            predictions.append(prediction)

        self.results["predictions"] = predictions

        # Calculate metrics
        logger.info("Calculating evaluation metrics")
        metrics = self._calculate_metrics(predictions)
        self.results["metrics"] = metrics

        # Calculate performance stats
        performance = self._calculate_performance(predictions)
        self.results["performance"] = performance

        self.results["benchmark_end"] = datetime.now().isoformat()

        logger.info("Benchmark completed successfully")
        return self.results

    def _calculate_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate evaluation metrics."""

        # Extract predictions and ground truth
        pred_labels = [p["prediction"] for p in predictions]
        true_labels = [p["ground_truth"] for p in predictions]

        # Remove error predictions
        valid_predictions = [
            (p, t) for p, t in zip(pred_labels, true_labels, strict=False) if p != "ERROR"
        ]

        if not valid_predictions:
            return {"error": "No valid predictions to evaluate"}

        pred_labels_list, true_labels_list = zip(*valid_predictions, strict=False)
        pred_labels = list(pred_labels_list)
        true_labels = list(true_labels_list)

        # Calculate confusion matrix
        tp = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "ATTACK" and t == "ATTACK"
        )
        fp = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "ATTACK" and t == "NORMAL"
        )
        fn = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "NORMAL" and t == "ATTACK"
        )
        tn = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "NORMAL" and t == "NORMAL"
        )

        # Calculate basic metrics
        total_samples = tp + fp + fn + tn
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also known as sensitivity
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

        # Matthews Correlation Coefficient (MCC)
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0

        # R-squared (coefficient of determination) - for binary classification
        # Calculate predicted probabilities based on binary predictions
        y_true_binary = [1 if t == "ATTACK" else 0 for t in true_labels]
        y_pred_binary = [1 if p == "ATTACK" else 0 for p in pred_labels]

        # Mean of actual values
        y_mean = sum(y_true_binary) / len(y_true_binary) if len(y_true_binary) > 0 else 0

        # Total sum of squares and residual sum of squares
        ss_tot = sum((y - y_mean) ** 2 for y in y_true_binary)
        ss_res = sum(
            (y_true - y_pred) ** 2
            for y_true, y_pred in zip(y_true_binary, y_pred_binary, strict=False)
        )

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2

        # Attack type breakdown
        attack_type_performance: dict[str, dict[str, Any]] = {}
        for prediction in predictions:
            if prediction["prediction"] != "ERROR" and prediction["attack_type"] != "normal":
                attack_type = prediction["attack_type"]
                if attack_type not in attack_type_performance:
                    attack_type_performance[attack_type] = {
                        "correct": 0,
                        "total": 0,
                        "accuracy": 0.0,
                    }

                attack_type_performance[attack_type]["total"] += 1
                if prediction["prediction"] == "ATTACK":
                    attack_type_performance[attack_type]["correct"] += 1

        # Calculate per-attack-type accuracy
        for attack_type in attack_type_performance:
            stats = attack_type_performance[attack_type]
            stats["accuracy"] = (
                float(stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
            )

        return {
            # Basic metrics
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            # Advanced metrics
            "matthews_correlation_coefficient": mcc,
            "r_squared": r_squared,
            # Error rates
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            # Confusion matrix
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "total_samples": int(total_samples),
            # Additional info
            "total_predictions": len(valid_predictions),
            "error_predictions": len(predictions) - len(valid_predictions),
            "attack_type_performance": attack_type_performance,
        }

    def _calculate_performance(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate performance statistics."""

        response_times = [p["response_time"] for p in predictions if p["response_time"] > 0]

        if not response_times:
            return {"error": "No valid response times recorded"}

        return {
            "total_predictions": len(predictions),
            "avg_response_time": np.mean(response_times),
            "median_response_time": np.median(response_times),
            "min_response_time": np.min(response_times),
            "max_response_time": np.max(response_times),
            "std_response_time": np.std(response_times),
            "predictions_per_second": 1 / np.mean(response_times)
            if np.mean(response_times) > 0
            else 0,
            "total_time": sum(response_times),
        }

    def save_results(self, output_file: str | None = None) -> str:
        """Save benchmark results to file."""

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"nsl_kdd_benchmark_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def print_summary(self) -> None:
        """Print benchmark summary to console."""

        print("\n" + "=" * 60)
        print("NSL-KDD INTRUSION DETECTION BENCHMARK SUMMARY")
        print("=" * 60)

        # Dataset info
        dataset_info = self.results["dataset_info"]
        if isinstance(dataset_info, dict):
            print(f"\nDataset: {dataset_info.get('filename', 'Unknown')}")
            print(f"Sample size: {dataset_info.get('sample_size', 0)} connections")
            print(f"Normal connections: {dataset_info.get('normal_count', 0)}")
            print(f"Attack connections: {dataset_info.get('attack_count', 0)}")
            attack_types = dataset_info.get("attack_types", [])
            if isinstance(attack_types, list):
                print(f"Attack types: {', '.join(str(t) for t in attack_types)}")

        # Performance metrics
        metrics = self.results.get("metrics", {})
        if isinstance(metrics, dict) and "error" not in metrics:
            print("\n" + "=" * 50)
            print("CLASSIFICATION METRICS")
            print("=" * 50)
            print(f"Overall Accuracy:        {metrics.get('accuracy', 0):.3f}")
            print(f"Balanced Accuracy:       {metrics.get('balanced_accuracy', 0):.3f}")
            print(f"Precision:               {metrics.get('precision', 0):.3f}")
            print(f"Recall (Sensitivity):    {metrics.get('recall', 0):.3f}")
            print(f"Specificity:             {metrics.get('specificity', 0):.3f}")
            print(f"F1-Score:                {metrics.get('f1_score', 0):.3f}")

            print("\n" + "=" * 50)
            print("ADVANCED METRICS")
            print("=" * 50)
            print(
                f"Matthews Correlation:    {metrics.get('matthews_correlation_coefficient', 0):.3f}"
            )
            print(f"R-Squared:               {metrics.get('r_squared', 0):.3f}")

            print("\n" + "=" * 50)
            print("ERROR RATES")
            print("=" * 50)
            print(f"False Positive Rate:     {metrics.get('false_positive_rate', 0):.3f}")
            print(f"False Negative Rate:     {metrics.get('false_negative_rate', 0):.3f}")

            print("\n" + "=" * 50)
            print("CONFUSION MATRIX")
            print("=" * 50)
            print(f"True Positives:          {metrics.get('true_positives', 0)}")
            print(f"False Positives:         {metrics.get('false_positives', 0)}")
            print(f"False Negatives:         {metrics.get('false_negatives', 0)}")
            print(f"True Negatives:          {metrics.get('true_negatives', 0)}")
            print(f"Total Samples:           {metrics.get('total_samples', 0)}")

            # Attack type performance
            attack_type_perf = metrics.get("attack_type_performance", {})
            if isinstance(attack_type_perf, dict) and attack_type_perf:
                print("\n" + "=" * 50)
                print("ATTACK TYPE DETECTION PERFORMANCE")
                print("=" * 50)
                for attack_type, stats in attack_type_perf.items():
                    if isinstance(stats, dict):
                        correct = stats.get("correct", 0)
                        total = stats.get("total", 0)
                        accuracy = stats.get("accuracy", 0)
                        print(f"{attack_type:15}: {correct:3}/{total:3} ({accuracy:.3f})")

        # Performance stats
        perf = self.results.get("performance", {})
        if isinstance(perf, dict) and "error" not in perf:
            print("\n" + "=" * 50)
            print("PERFORMANCE METRICS")
            print("=" * 50)
            print(f"Average Response Time:   {perf.get('avg_response_time', 0):.3f}s")
            print(f"Median Response Time:    {perf.get('median_response_time', 0):.3f}s")
            print(f"Min Response Time:       {perf.get('min_response_time', 0):.3f}s")
            print(f"Max Response Time:       {perf.get('max_response_time', 0):.3f}s")
            print(f"Predictions per Second:  {perf.get('predictions_per_second', 0):.2f}")
            print(f"Total Processing Time:   {perf.get('total_time', 0):.2f}s")

        print(f"\nModel URL: {self.results.get('model_url', 'Unknown')}")
        print(
            f"Benchmark duration: {self.results.get('benchmark_start', 'Unknown')} to {self.results.get('benchmark_end', 'Unknown')}"
        )


async def main() -> int | None:
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="NSL-KDD Network Intrusion Detection Benchmark for LM Studio"
    )
    parser.add_argument(
        "--lm-studio-url", default="http://localhost:1234", help="LM Studio server URL"
    )
    parser.add_argument("--model-name", default="local-model", help="Model name in LM Studio")
    parser.add_argument(
        "--dataset-file", default="KDDTrain+.txt", help="NSL-KDD dataset file to use"
    )
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Maximum number of samples to test"
    )
    parser.add_argument(
        "--attack-ratio", type=float, default=0.3, help="Ratio of attack samples in test set"
    )
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with 20 samples")
    parser.add_argument(
        "--full-benchmark", action="store_true", help="Run full benchmark with 1000+ samples"
    )
    parser.add_argument("--output-file", type=str, help="Output file for results")

    args = parser.parse_args()

    # Adjust parameters based on mode
    if args.quick_test:
        max_samples = 20
        attack_ratio = 0.4
        print("Running quick test with 20 samples...")
    elif args.full_benchmark:
        max_samples = 1000
        attack_ratio = args.attack_ratio
        print("Running full benchmark with 1000 samples...")
    else:
        max_samples = args.max_samples
        attack_ratio = args.attack_ratio

    # Check if NSL-KDD dataset exists
    dataset_path = Path("NSL-KDD") / args.dataset_file
    if not dataset_path.exists():
        print(f"Error: NSL-KDD dataset file not found: {dataset_path}")
        print("Please ensure the NSL-KDD folder contains the dataset files.")
        return 1

    # Initialize benchmark
    benchmark = NSLKDDBenchmark(args.lm_studio_url)

    try:
        # Run benchmark
        print(f"Starting NSL-KDD benchmark with LM Studio at {args.lm_studio_url}")
        await benchmark.run_benchmark(
            filename=args.dataset_file, max_samples=max_samples, attack_ratio=attack_ratio
        )

        # Save results
        output_file = benchmark.save_results(args.output_file)

        # Print summary
        benchmark.print_summary()

        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    asyncio.run(main())

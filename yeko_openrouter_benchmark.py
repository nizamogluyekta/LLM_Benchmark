#!/usr/bin/env python3
"""
Yeko's Advanced NSL-KDD Cybersecurity Benchmark with OpenRouter

This script provides comprehensive benchmarking for cybersecurity models using NSL-KDD dataset.
Features:
- Balanced sampling across all attack types
- Detailed progress logging
- Comprehensive reporting
- OpenRouter API integration

Requirements:
- NSL-KDD dataset in the NSL-KDD folder
- Python dependencies: openai, pandas, numpy, scikit-learn

Usage:
    python yeko_openrouter_benchmark.py --model "meta-llama/llama-3.2-3b-instruct:free"
    python yeko_openrouter_benchmark.py --model "x-ai/grok-4-fast:free" --dataset "KDDTest+.txt"
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"yeko_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
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
        description_parts = []

        # Basic connection info
        description_parts.append(f"Network connection: {self.protocol_type.upper()} protocol")
        description_parts.append(f"Service: {self.service}")
        description_parts.append(f"Connection state: {self.flag}")

        # Data transfer information
        if self.src_bytes > 0 or self.dst_bytes > 0:
            description_parts.append(
                f"Data transfer: {self.src_bytes} bytes sent, {self.dst_bytes} bytes received"
            )

        # Duration and timing
        if self.duration > 0:
            description_parts.append(f"Duration: {self.duration} seconds")

        # Connection patterns
        if self.count > 1:
            description_parts.append(f"Connection count in time window: {self.count}")
        if self.srv_count > 1:
            description_parts.append(f"Same service connections: {self.srv_count}")

        # Security indicators
        if self.num_failed_logins > 0:
            description_parts.append(f"Failed login attempts: {self.num_failed_logins}")
        if self.logged_in == 1:
            description_parts.append("Successful login")
        if self.num_compromised > 0:
            description_parts.append(f"Compromised conditions: {self.num_compromised}")
        if self.root_shell > 0:
            description_parts.append(f"Root shell access: {self.root_shell}")
        if self.su_attempted > 0:
            description_parts.append(f"Su command attempts: {self.su_attempted}")

        # Error patterns
        if self.serror_rate > 0:
            description_parts.append(f"SYN error rate: {self.serror_rate:.2f}")
        if self.rerror_rate > 0:
            description_parts.append(f"REJ error rate: {self.rerror_rate:.2f}")

        # Host-based features
        if self.dst_host_count > 0:
            description_parts.append(f"Destination host connections: {self.dst_host_count}")

        # Attack indicators
        if self.land == 1:
            description_parts.append("Land attack indicator detected")
        if self.wrong_fragment > 0:
            description_parts.append(f"Wrong fragments: {self.wrong_fragment}")
        if self.urgent > 0:
            description_parts.append(f"Urgent packets: {self.urgent}")
        if self.hot > 0:
            description_parts.append(f"Hot indicators: {self.hot}")

        return ". ".join(description_parts) + "."


class YekoOpenRouterClient:
    """Enhanced OpenRouter client with detailed logging and error handling."""

    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name
        self.request_count = 0
        self.total_tokens = 0
        self.error_count = 0

        logger.info(f"🚀 Initialized OpenRouter client with model: {model_name}")

    async def predict_intrusion(
        self, connection: NetworkConnection, sample_id: int
    ) -> dict[str, Any]:
        """Predict if a network connection is an intrusion with detailed logging."""

        self.request_count += 1
        logger.info(f"📡 Processing sample {sample_id}: {connection.attack_type} connection")

        prompt = self._create_advanced_prompt(connection)

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                logger.debug(f"🔄 API call attempt {attempt + 1} for sample {sample_id}")

                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cybersecurity analyst specializing in network intrusion detection. Provide detailed, technical analysis of network traffic patterns.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=512,
                    temperature=0.1,
                )

                end_time = time.time()
                response_time = end_time - start_time

                # Track token usage if available
                if hasattr(response, "usage") and response.usage:
                    self.total_tokens += response.usage.total_tokens
                    logger.debug(f"🎯 Tokens used: {response.usage.total_tokens}")

                response_text = response.choices[0].message.content
                if response_text is None:
                    response_text = ""
                response_text = response_text.strip()
                logger.debug(f"✅ Sample {sample_id} processed in {response_time:.2f}s")

                return self._parse_detailed_prediction(
                    response_text, response_time, sample_id, connection.attack_type
                )

            except Exception as e:
                self.error_count += 1
                error_msg = str(e)

                if "429" in error_msg or "rate" in error_msg.lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"⚠️ Rate limit hit for sample {sample_id}, retrying in {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(
                            f"❌ Rate limit exceeded for sample {sample_id}, max retries reached"
                        )
                else:
                    logger.error(f"❌ API error for sample {sample_id}: {error_msg}")

        return {
            "prediction": "ERROR",
            "confidence": 0.0,
            "reasoning": f"API Error: {error_msg}",
            "response_time": 0.0,
            "raw_response": "",
            "sample_id": sample_id,
            "actual_attack_type": connection.attack_type,
            "error_type": "api_error",
        }

    def _create_advanced_prompt(self, connection: NetworkConnection) -> str:
        """Create an advanced cybersecurity analysis prompt."""

        readable_connection = connection.to_readable_text()

        prompt = f"""CYBERSECURITY NETWORK TRAFFIC ANALYSIS

Connection Details:
{readable_connection}

ANALYSIS TASK:
As a cybersecurity expert, analyze this network connection for potential security threats.

CLASSIFICATION REQUIRED:
- NORMAL: Legitimate network activity with standard patterns
- ATTACK: Malicious or suspicious activity indicating a security threat

DETAILED ANALYSIS REQUIRED:
1. TRAFFIC PATTERNS: Analyze connection characteristics, data volumes, timing patterns
2. PROTOCOL ANALYSIS: Examine protocol usage, service interactions, connection states
3. BEHAVIORAL INDICATORS: Identify suspicious patterns, anomalies, or attack signatures
4. ATTACK CATEGORIZATION: If malicious, specify attack type (DoS, Probe, R2L, U2R, etc.)
5. CONFIDENCE ASSESSMENT: Rate your confidence level and explain reasoning
6. RISK ASSESSMENT: Evaluate potential impact and urgency level

RESPONSE FORMAT:
PREDICTION: [NORMAL/ATTACK]
ATTACK_TYPE: [If attack: specify type, else: N/A]
CONFIDENCE: [High/Medium/Low]
REASONING: [Your comprehensive technical analysis]
RISK_LEVEL: [Critical/High/Medium/Low]"""

        return prompt

    def _parse_detailed_prediction(
        self, response_text: str, response_time: float, sample_id: int, actual_attack_type: str
    ) -> dict[str, Any]:
        """Parse LLM response with enhanced detail extraction."""

        response_lines = response_text.split("\n")

        # Initialize defaults
        prediction = "UNKNOWN"
        predicted_attack_type = "unknown"
        confidence_level = "unknown"
        reasoning = response_text
        risk_level = "unknown"
        confidence_score = 0.5

        # Parse structured response
        for line in response_lines:
            line_upper = line.upper().strip()
            if line_upper.startswith("PREDICTION:"):
                if "ATTACK" in line_upper:
                    prediction = "ATTACK"
                elif "NORMAL" in line_upper:
                    prediction = "NORMAL"
            elif line_upper.startswith("ATTACK_TYPE:"):
                predicted_attack_type = line.split(":", 1)[1].strip().lower()
            elif line_upper.startswith("CONFIDENCE:"):
                confidence_level = line.split(":", 1)[1].strip().lower()
            elif line_upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line_upper.startswith("RISK_LEVEL:"):
                risk_level = line.split(":", 1)[1].strip().lower()

        # Enhanced fallback parsing for different model response styles
        if prediction == "UNKNOWN":
            response_upper = response_text.upper()

            # Debug logging for Tongyi responses
            if "tongyi" in self.model_name.lower():
                logger.info(f"🔍 Tongyi response for sample {sample_id}: {response_text[:200]}...")

            # Standard format checking
            if (
                response_text.upper().startswith("ATTACK")
                or "ATTACK" in response_text.upper()[:100]
            ):
                prediction = "ATTACK"
            elif (
                response_text.upper().startswith("NORMAL")
                or "NORMAL" in response_text.upper()[:100]
            ):
                prediction = "NORMAL"
            # Extended pattern matching for various model styles
            elif any(
                word in response_upper
                for word in ["MALICIOUS", "SUSPICIOUS", "INTRUSION", "THREAT", "ANOMAL"]
            ):
                prediction = "ATTACK"
            elif any(
                word in response_upper
                for word in ["LEGITIMATE", "BENIGN", "SAFE", "REGULAR", "CLEAN"]
            ):
                prediction = "NORMAL"
            # Check for attack type indicators
            elif any(
                attack in response_upper
                for attack in ["DOS", "PROBE", "R2L", "U2R", "NEPTUNE", "SMURF", "BACK"]
            ):
                prediction = "ATTACK"

        # Convert confidence level to score
        if confidence_level == "high":
            confidence_score = 0.9
        elif confidence_level == "medium":
            confidence_score = 0.7
        elif confidence_level == "low":
            confidence_score = 0.5

        # Determine correctness
        actual_label = "NORMAL" if actual_attack_type.lower() == "normal" else "ATTACK"
        is_correct = prediction == actual_label

        logger.info(
            f"🎯 Sample {sample_id}: Predicted={prediction}, Actual={actual_label}, Correct={is_correct}"
        )

        return {
            "prediction": prediction,
            "predicted_attack_type": predicted_attack_type,
            "confidence_level": confidence_level,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            "risk_level": risk_level,
            "response_time": response_time,
            "raw_response": response_text,
            "sample_id": sample_id,
            "actual_attack_type": actual_attack_type,
            "actual_label": actual_label,
            "is_correct": is_correct,
        }


class YekoNSLKDDBenchmark:
    """Advanced NSL-KDD benchmarking system with balanced attack type sampling."""

    def __init__(self, api_key: str, model_name: str):
        self.openrouter_client = YekoOpenRouterClient(api_key, model_name)
        self.model_name = model_name

        # NSL-KDD feature names
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

        logger.info("🔬 Initialized Yeko's NSL-KDD Benchmark System")

    def load_balanced_nsl_kdd_data(
        self, filename: str = "KDDTrain+.txt", target_samples: int = 1000
    ) -> list[NetworkConnection]:
        """Load NSL-KDD data with balanced attack type sampling."""

        file_path = Path("NSL-KDD") / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"📊 Loading NSL-KDD data from {file_path}")
        logger.info(f"🎯 Target samples: {target_samples} with balanced attack types")

        # First pass: collect all data by attack type
        attack_type_data = defaultdict(list)
        total_rows = 0

        with open(file_path, encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            for row_num, row in enumerate(csv_reader):
                total_rows += 1
                if len(row) != len(self.feature_names):
                    continue

                try:
                    # Parse row data
                    parsed_row: list[Any] = []
                    for i, value in enumerate(row):
                        if i in [
                            1,
                            2,
                            3,
                            41,
                        ]:  # String fields: protocol_type, service, flag, attack_type
                            parsed_row.append(value.strip())
                        else:
                            if "." in value:
                                parsed_row.append(float(value))
                            else:
                                parsed_row.append(int(value))

                    connection = NetworkConnection(*parsed_row)
                    attack_type_data[connection.attack_type].append(connection)

                except (ValueError, TypeError) as e:
                    if row_num < 5:  # Debug first few errors
                        logger.warning(f"Parse error on row {row_num}: {e}")
                    logger.debug(f"Skipping malformed row {row_num}: {e}")
                    continue

        # Log attack type distribution
        logger.info("📈 Dataset Analysis:")
        logger.info(f"   Total rows processed: {total_rows}")
        logger.info(
            f"   Total valid connections: {sum(len(connections) for connections in attack_type_data.values())}"
        )
        logger.info(f"   Attack types found: {len(attack_type_data)}")

        for attack_type, connections in attack_type_data.items():
            logger.info(f"   - {attack_type}: {len(connections)} samples")

        # Calculate balanced sampling
        attack_types = list(attack_type_data.keys())
        samples_per_type = target_samples // len(attack_types)
        remainder = target_samples % len(attack_types)

        logger.info("⚖️ Balanced Sampling Strategy:")
        logger.info(f"   Base samples per type: {samples_per_type}")
        logger.info(f"   Extra samples for first {remainder} types: 1 each")

        # Sample balanced data
        selected_connections = []
        for i, attack_type in enumerate(attack_types):
            available = len(attack_type_data[attack_type])
            target_for_type = samples_per_type + 1 if i < remainder else samples_per_type

            actual_samples = min(target_for_type, available)
            sampled = random.sample(attack_type_data[attack_type], actual_samples)
            selected_connections.extend(sampled)

            logger.info(f"   ✅ {attack_type}: selected {actual_samples}/{available} samples")

        # Shuffle final dataset
        random.shuffle(selected_connections)

        logger.info(f"🎲 Final dataset: {len(selected_connections)} samples, shuffled and ready")
        return selected_connections

    async def run_comprehensive_benchmark(
        self, connections: list[NetworkConnection]
    ) -> dict[str, Any]:
        """Run comprehensive benchmark with detailed progress tracking."""

        logger.info("🚀 Starting Comprehensive Cybersecurity Benchmark")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Total samples: {len(connections)}")
        logger.info(f"   Expected duration: ~{len(connections) * 2:.1f} seconds")

        start_time = time.time()
        predictions = []

        # Progress tracking
        processed = 0
        correct_predictions = 0

        for i, connection in enumerate(connections):
            sample_id = i + 1

            # Progress logging
            if sample_id % 50 == 0 or sample_id <= 10:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(connections) - processed) / rate if rate > 0 else 0

                logger.info(
                    f"📊 Progress: {sample_id}/{len(connections)} ({sample_id / len(connections) * 100:.1f}%) | "
                    f"Rate: {rate:.1f} samples/min | ETA: {eta / 60:.1f} min | "
                    f"Accuracy so far: {correct_predictions / processed * 100:.1f}%"
                    if processed > 0
                    else "Starting..."
                )

            # Get prediction
            prediction_result = await self.openrouter_client.predict_intrusion(
                connection, sample_id
            )
            predictions.append(prediction_result)

            # Update tracking
            processed += 1
            if prediction_result.get("is_correct", False):
                correct_predictions += 1

            # Rate limiting
            await asyncio.sleep(0.5)

        total_time = time.time() - start_time

        logger.info(f"✅ Benchmark completed in {total_time:.2f} seconds")
        logger.info(f"📈 Final accuracy: {correct_predictions / processed * 100:.1f}%")

        # Generate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(predictions, total_time)

        # Create detailed report
        report = self._generate_detailed_report(connections, predictions, metrics, total_time)

        return report

    def _calculate_comprehensive_metrics(
        self, predictions: list[dict[str, Any]], total_time: float
    ) -> dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""

        logger.info("📊 Calculating comprehensive metrics...")

        # Basic statistics
        total_samples = len(predictions)
        error_predictions = [p for p in predictions if p["prediction"] == "ERROR"]
        valid_predictions = [p for p in predictions if p["prediction"] != "ERROR"]

        if not valid_predictions:
            logger.error("❌ No valid predictions found!")
            return {"error": "No valid predictions"}

        # Classification metrics
        tp = fp = tn = fn = 0
        correct = 0

        for pred in valid_predictions:
            actual = pred["actual_label"]
            predicted = pred["prediction"]

            if pred["is_correct"]:
                correct += 1

            if predicted == "ATTACK" and actual == "ATTACK":
                tp += 1
            elif predicted == "ATTACK" and actual == "NORMAL":
                fp += 1
            elif predicted == "NORMAL" and actual == "NORMAL":
                tn += 1
            elif predicted == "NORMAL" and actual == "ATTACK":
                fn += 1

        # Calculate metrics
        accuracy = correct / len(valid_predictions) if valid_predictions else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )
        balanced_accuracy = (recall + specificity) / 2

        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

        # Performance metrics
        response_times = [p["response_time"] for p in valid_predictions if p["response_time"] > 0]
        avg_response_time = np.mean(response_times) if response_times else 0

        # Attack type specific metrics
        attack_type_metrics = self._calculate_attack_type_metrics(valid_predictions)

        logger.info(
            f"✅ Metrics calculated: Accuracy={accuracy:.3f}, F1={f1_score:.3f}, MCC={mcc:.3f}"
        )

        return {
            "overall_metrics": {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1_score,
                "matthews_correlation": mcc,
            },
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "total_valid": len(valid_predictions),
            },
            "performance_metrics": {
                "total_samples": total_samples,
                "valid_predictions": len(valid_predictions),
                "error_predictions": len(error_predictions),
                "avg_response_time": avg_response_time,
                "total_processing_time": total_time,
                "samples_per_second": total_samples / total_time if total_time > 0 else 0,
            },
            "attack_type_metrics": attack_type_metrics,
        }

    def _calculate_attack_type_metrics(
        self, valid_predictions: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Calculate metrics for each attack type."""

        attack_type_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"correct": 0, "total": 0, "predictions": []}
        )

        for pred in valid_predictions:
            attack_type = pred["actual_attack_type"]
            attack_type_stats[attack_type]["total"] += 1
            attack_type_stats[attack_type]["predictions"].append(pred)

            if pred["is_correct"]:
                attack_type_stats[attack_type]["correct"] += 1

        # Calculate accuracy for each type
        for attack_type in attack_type_stats:
            stats = attack_type_stats[attack_type]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            # Remove detailed predictions to reduce size
            del stats["predictions"]

        return dict(attack_type_stats)

    def _generate_detailed_report(
        self,
        connections: list[NetworkConnection],
        predictions: list[dict[str, Any]],
        metrics: dict[str, Any],
        total_time: float,
    ) -> dict[str, Any]:
        """Generate comprehensive benchmark report."""

        logger.info("📄 Generating comprehensive report...")

        # Dataset analysis
        attack_type_distribution = Counter(conn.attack_type for conn in connections)

        report = {
            "benchmark_info": {
                "model_name": self.model_name,
                "benchmark_type": "Yeko's Advanced NSL-KDD Cybersecurity Benchmark",
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(connections),
                "total_processing_time": total_time,
                "dataset_distribution": dict(attack_type_distribution),
            },
            "model_performance": {
                "api_requests": self.openrouter_client.request_count,
                "total_tokens": self.openrouter_client.total_tokens,
                "error_count": self.openrouter_client.error_count,
                "success_rate": 1
                - (self.openrouter_client.error_count / self.openrouter_client.request_count)
                if self.openrouter_client.request_count > 0
                else 0,
            },
            "evaluation_metrics": metrics,
            "detailed_predictions": predictions[:100],  # First 100 for size management
            "prediction_summary": {
                "total_predictions": len(predictions),
                "correct_predictions": sum(1 for p in predictions if p.get("is_correct", False)),
                "error_predictions": sum(1 for p in predictions if p["prediction"] == "ERROR"),
            },
        }

        logger.info("✅ Comprehensive report generated successfully")
        return report

    def save_detailed_results(self, report: dict[str, Any], output_file: str | None = None) -> str:
        """Save detailed benchmark results."""

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_clean = self.model_name.replace("/", "_").replace("-", "_").replace(":", "_")
            output_file = f"yeko_benchmark_{model_name_clean}_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Detailed results saved to: {output_file}")
        return output_file

    def print_comprehensive_summary(self, report: dict[str, Any]) -> None:
        """Print comprehensive benchmark summary."""

        info = report["benchmark_info"]
        metrics = report["evaluation_metrics"]["overall_metrics"]
        confusion = report["evaluation_metrics"]["confusion_matrix"]
        performance = report["evaluation_metrics"]["performance_metrics"]
        attack_metrics = report["evaluation_metrics"]["attack_type_metrics"]

        print("\n" + "=" * 80)
        print("🏆 YEKO'S ADVANCED NSL-KDD CYBERSECURITY BENCHMARK RESULTS")
        print("=" * 80)

        print("\n🔬 BENCHMARK INFORMATION")
        print(f"   Model: {info['model_name']}")
        print(f"   Timestamp: {info['timestamp']}")
        print(f"   Total Samples: {info['total_samples']}")
        print(f"   Processing Time: {info['total_processing_time']:.2f} seconds")
        print(f"   Samples per Second: {performance['samples_per_second']:.2f}")

        print("\n📊 DATASET DISTRIBUTION")
        for attack_type, count in info["dataset_distribution"].items():
            percentage = (count / info["total_samples"]) * 100
            print(f"   - {attack_type}: {count} samples ({percentage:.1f}%)")

        print("\n🎯 CLASSIFICATION PERFORMANCE")
        print(
            f"   Overall Accuracy:        {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)"
        )
        print(f"   Balanced Accuracy:       {metrics['balanced_accuracy']:.3f}")
        print(f"   Precision:               {metrics['precision']:.3f}")
        print(f"   Recall (Sensitivity):    {metrics['recall']:.3f}")
        print(f"   Specificity:             {metrics['specificity']:.3f}")
        print(f"   F1-Score:                {metrics['f1_score']:.3f}")
        print(f"   Matthews Correlation:    {metrics['matthews_correlation']:.3f}")

        print("\n🔍 CONFUSION MATRIX")
        print(
            f"   True Positives:  {confusion['true_positives']:4d}   False Positives: {confusion['false_positives']:4d}"
        )
        print(
            f"   False Negatives: {confusion['false_negatives']:4d}   True Negatives:  {confusion['true_negatives']:4d}"
        )
        print(f"   Total Valid Predictions: {confusion['total_valid']}")

        print("\n⚡ PERFORMANCE METRICS")
        print(f"   Average Response Time:   {performance['avg_response_time']:.3f}s")
        print(
            f"   Valid Predictions:       {performance['valid_predictions']}/{performance['total_samples']}"
        )
        print(
            f"   Error Rate:              {performance['error_predictions']}/{performance['total_samples']} ({performance['error_predictions'] / performance['total_samples'] * 100:.1f}%)"
        )

        print("\n🏹 ATTACK TYPE PERFORMANCE")
        for attack_type, stats in sorted(
            attack_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
        ):
            accuracy_pct = stats["accuracy"] * 100
            print(
                f"   {attack_type:15}: {stats['correct']:3d}/{stats['total']:3d} ({accuracy_pct:5.1f}%)"
            )

        print("\n" + "=" * 80)


async def main() -> int:
    """Main function to run Yeko's advanced NSL-KDD benchmark."""

    parser = argparse.ArgumentParser(description="Yeko's Advanced NSL-KDD Cybersecurity Benchmark")
    parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model name (e.g., 'meta-llama/llama-3.2-3b-instruct:free')",
    )
    parser.add_argument("--dataset", default="KDDTrain+.txt", help="NSL-KDD dataset file to use")
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples to test (default: 1000)"
    )
    parser.add_argument("--output", help="Output file name for results")
    parser.add_argument(
        "--api-key", help="OpenRouter API key (optional, will use hardcoded if not provided)"
    )

    args = parser.parse_args()

    # Hardcoded API key - replace with your actual key
    HARDCODED_API_KEY = "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"  # Replace this with your actual OpenRouter API key

    api_key = args.api_key if args.api_key else HARDCODED_API_KEY

    if api_key == "YOUR_API_KEY_HERE":
        logger.error(
            "❌ Please replace the hardcoded API key in the script with your actual OpenRouter API key"
        )
        return 1

    print("🚀 YEKO'S ADVANCED NSL-KDD CYBERSECURITY BENCHMARK")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Target Samples: {args.samples}")
    print("Balanced Attack Types: Yes")
    print("=" * 60)

    try:
        # Initialize benchmark system
        benchmark = YekoNSLKDDBenchmark(api_key, args.model)

        # Load balanced dataset
        connections = benchmark.load_balanced_nsl_kdd_data(
            filename=args.dataset, target_samples=args.samples
        )

        # Run comprehensive benchmark
        report = await benchmark.run_comprehensive_benchmark(connections)

        # Print detailed summary
        benchmark.print_comprehensive_summary(report)

        # Save results
        output_file = benchmark.save_detailed_results(report, args.output)
        print(f"\n💾 Detailed results saved to: {output_file}")

        return 0

    except KeyboardInterrupt:
        logger.info("❌ Benchmark interrupted by user")
        print("\n❌ Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))

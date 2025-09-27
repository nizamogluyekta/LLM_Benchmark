#!/usr/bin/env python3
"""
Yeko's Multi-Language Model NSL-KDD Cybersecurity Benchmark with OpenRouter

This script provides comprehensive benchmarking for multiple language models using NSL-KDD dataset.
Features:
- Two-part testing: Attack Detection + Attack Type Classification
- Multi-model comparison with hardcoded model list
- Homogeneous dataset sampling across all attack types
- Comprehensive comparative reporting

Requirements:
- NSL-KDD dataset in the NSL-KDD folder
- Python dependencies: openai, pandas, numpy, scikit-learn

Usage:
    python yeko_openrouter_multilang_benchmark.py --samples 1000
    python yeko_openrouter_multilang_benchmark.py --samples 500 --dataset "KDDTest+.txt"
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

from openai import OpenAI

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"yeko_multilang_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
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


class MultiLangOpenRouterClient:
    """Enhanced OpenRouter client for multi-model testing with two-part analysis."""

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

    async def test_attack_detection(
        self, connection: NetworkConnection, sample_id: int
    ) -> dict[str, Any]:
        """Test 1: Binary attack detection (NORMAL vs ATTACK)."""

        self.request_count += 1
        logger.debug(
            f"📡 Part 1 - Attack Detection for sample {sample_id}: {connection.attack_type}"
        )

        prompt = self._create_attack_detection_prompt(connection)
        return await self._make_api_call(prompt, sample_id, "attack_detection")

    async def test_attack_classification(
        self, connection: NetworkConnection, sample_id: int
    ) -> dict[str, Any]:
        """Test 2: Attack type classification (specific attack type identification)."""

        self.request_count += 1
        logger.debug(
            f"📡 Part 2 - Attack Classification for sample {sample_id}: {connection.attack_type}"
        )

        prompt = self._create_attack_classification_prompt(connection)
        return await self._make_api_call(prompt, sample_id, "attack_classification")

    def _create_attack_detection_prompt(self, connection: NetworkConnection) -> str:
        """Create prompt for binary attack detection."""

        readable_connection = connection.to_readable_text()

        prompt = f"""CYBERSECURITY ATTACK DETECTION - PART 1

Connection Details:
{readable_connection}

TASK: Determine if this network connection represents normal traffic or a cyber attack.

CLASSIFICATION OPTIONS:
- NORMAL: Legitimate network activity with standard patterns
- ATTACK: Malicious or suspicious activity indicating a security threat

ANALYSIS REQUIRED:
1. Examine connection characteristics and behavioral patterns
2. Identify any suspicious indicators or anomalies
3. Make a binary classification decision

RESPONSE FORMAT:
DETECTION: [NORMAL/ATTACK]
CONFIDENCE: [High/Medium/Low]
REASONING: [Brief explanation of your decision]"""

        return prompt

    def _create_attack_classification_prompt(self, connection: NetworkConnection) -> str:
        """Create prompt for specific attack type classification."""

        readable_connection = connection.to_readable_text()

        prompt = f"""CYBERSECURITY ATTACK CLASSIFICATION - PART 2

Connection Details:
{readable_connection}

TASK: If this is an attack, identify the specific type of cyber attack.

ATTACK CATEGORIES:
- DoS (Denial of Service): neptune, smurf, pod, back, land, teardrop
- Probe (Reconnaissance): satan, ipsweep, nmap, portsweep
- R2L (Remote to Local): warezclient, guess_passwd, ftp_write, imap, phf, multihop, warezmaster, spy
- U2R (User to Root): buffer_overflow, loadmodule, perl, rootkit
- NORMAL: Legitimate traffic (not an attack)

ANALYSIS REQUIRED:
1. Determine the attack category and specific type
2. Identify attack signatures and patterns
3. Provide technical reasoning

RESPONSE FORMAT:
CLASSIFICATION: [Specific attack name or NORMAL]
CATEGORY: [DoS/Probe/R2L/U2R/NORMAL]
CONFIDENCE: [High/Medium/Low]
REASONING: [Technical explanation of attack characteristics]"""

        return prompt

    async def _make_api_call(self, prompt: str, sample_id: int, test_type: str) -> dict[str, Any]:
        """Make API call with retry logic."""

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cybersecurity analyst specializing in network intrusion detection. Provide precise, technical analysis following the exact format requested.",
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

                response_text = response.choices[0].message.content
                if response_text is None:
                    response_text = ""
                response_text = response_text.strip()

                return self._parse_response(response_text, response_time, sample_id, test_type)

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
            "test_type": test_type,
            "error_type": "api_error",
        }

    def _parse_response(
        self, response_text: str, response_time: float, sample_id: int, test_type: str
    ) -> dict[str, Any]:
        """Parse LLM response for both test types."""

        response_lines = response_text.split("\n")

        # Initialize defaults
        prediction = "UNKNOWN"
        category = "unknown"
        confidence_level = "unknown"
        reasoning = response_text
        confidence_score = 0.5

        # Parse structured response
        for line in response_lines:
            line_upper = line.upper().strip()
            if line_upper.startswith("DETECTION:") or line_upper.startswith("CLASSIFICATION:"):
                content = line.split(":", 1)[1].strip().upper()
                if any(
                    word in content
                    for word in ["ATTACK", "NEPTUNE", "SMURF", "SATAN", "BACK", "DOS", "PROBE"]
                ):
                    prediction = content.strip()
                elif "NORMAL" in content:
                    prediction = "NORMAL"
            elif line_upper.startswith("CATEGORY:"):
                category = line.split(":", 1)[1].strip().lower()
            elif line_upper.startswith("CONFIDENCE:"):
                confidence_level = line.split(":", 1)[1].strip().lower()
            elif line_upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Enhanced fallback parsing
        if prediction == "UNKNOWN":
            response_upper = response_text.upper()

            # Attack type specific detection
            attack_types = [
                "NEPTUNE",
                "SMURF",
                "POD",
                "BACK",
                "LAND",
                "TEARDROP",  # DoS
                "SATAN",
                "IPSWEEP",
                "NMAP",
                "PORTSWEEP",  # Probe
                "WAREZCLIENT",
                "GUESS_PASSWD",
                "FTP_WRITE",
                "IMAP",
                "PHF",
                "MULTIHOP",
                "WAREZMASTER",
                "SPY",  # R2L
                "BUFFER_OVERFLOW",
                "LOADMODULE",
                "PERL",
                "ROOTKIT",  # U2R
            ]

            for attack_type in attack_types:
                if attack_type in response_upper:
                    prediction = attack_type
                    break

            if prediction == "UNKNOWN":
                if any(
                    word in response_upper
                    for word in ["ATTACK", "MALICIOUS", "SUSPICIOUS", "INTRUSION", "THREAT"]
                ):
                    prediction = "ATTACK"
                elif any(
                    word in response_upper
                    for word in ["NORMAL", "LEGITIMATE", "BENIGN", "SAFE", "CLEAN"]
                ):
                    prediction = "NORMAL"

        # Convert confidence level to score
        if confidence_level == "high":
            confidence_score = 0.9
        elif confidence_level == "medium":
            confidence_score = 0.7
        elif confidence_level == "low":
            confidence_score = 0.5

        logger.debug(
            f"✅ Sample {sample_id} ({test_type}) processed in {response_time:.2f}s: {prediction}"
        )

        return {
            "prediction": prediction,
            "category": category,
            "confidence_level": confidence_level,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            "response_time": response_time,
            "raw_response": response_text,
            "sample_id": sample_id,
            "test_type": test_type,
        }


class YekoMultiLangBenchmark:
    """Multi-language model NSL-KDD benchmarking system with two-part testing."""

    def __init__(self, api_key: str):
        self.api_key = api_key

        # Hardcoded list of models to test
        self.models_to_test = [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-pro-1.5",
            "anthropic/claude-3-opus",
            "cohere/command-r-plus",
            "meta-llama/llama-3.2-3b-instruct:free",
            "x-ai/grok-4-fast:free",
        ]

        self.results: dict[str, Any] = {}

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

        logger.info("🔬 Initialized Yeko's Multi-Language NSL-KDD Benchmark System")
        logger.info(f"📋 Models to test: {len(self.models_to_test)}")
        for i, model in enumerate(self.models_to_test, 1):
            logger.info(f"   {i}. {model}")

    def load_homogeneous_dataset(
        self, filename: str = "KDDTrain+.txt", target_samples: int = 1000
    ) -> list[NetworkConnection]:
        """Load NSL-KDD data with homogeneous sampling across all attack types."""

        file_path = Path("NSL-KDD") / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"📊 Loading homogeneous NSL-KDD dataset from {file_path}")
        logger.info(
            f"🎯 Target samples: {target_samples} with equal representation across attack types"
        )

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
                    continue

        # Log attack type distribution
        logger.info("📈 Dataset Analysis:")
        logger.info(f"   Total rows processed: {total_rows}")
        logger.info(
            f"   Total valid connections: {sum(len(connections) for connections in attack_type_data.values())}"
        )
        logger.info(f"   Attack types found: {len(attack_type_data)}")

        # Calculate homogeneous sampling (equal samples per attack type)
        attack_types = list(attack_type_data.keys())
        samples_per_type = target_samples // len(attack_types)
        remainder = target_samples % len(attack_types)

        logger.info("⚖️ Homogeneous Sampling Strategy:")
        logger.info(f"   Equal samples per type: {samples_per_type}")
        logger.info(f"   Extra samples for first {remainder} types: 1 each")

        # Sample homogeneous data
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

        logger.info(
            f"🎲 Final homogeneous dataset: {len(selected_connections)} samples, ready for multi-model testing"
        )
        return selected_connections

    async def run_two_part_benchmark(
        self, model_name: str, connections: list[NetworkConnection]
    ) -> dict[str, Any]:
        """Run two-part benchmark for a single model."""

        logger.info(f"🚀 Starting Two-Part Benchmark for {model_name}")
        logger.info("   Part 1: Attack Detection (Binary Classification)")
        logger.info("   Part 2: Attack Classification (Multi-class)")
        logger.info(f"   Total samples: {len(connections)}")

        client = MultiLangOpenRouterClient(self.api_key, model_name)

        start_time = time.time()
        part1_results = []
        part2_results = []

        # Progress tracking using enumerate for index
        for i, connection in enumerate(connections):
            sample_id = i + 1

            # Progress logging
            if sample_id % 25 == 0 or sample_id <= 5:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(connections) * 2 - i * 2) / rate if rate > 0 else 0

                logger.info(
                    f"📊 {model_name} Progress: {sample_id}/{len(connections)} | "
                    f"Rate: {rate:.1f} tests/min | ETA: {eta / 60:.1f} min"
                )

            # Part 1: Attack Detection
            part1_result = await client.test_attack_detection(connection, sample_id)
            part1_result["actual_attack_type"] = connection.attack_type
            part1_result["actual_label"] = (
                "NORMAL" if connection.attack_type.lower() == "normal" else "ATTACK"
            )
            part1_results.append(part1_result)

            # Small delay between parts
            await asyncio.sleep(0.2)

            # Part 2: Attack Classification
            part2_result = await client.test_attack_classification(connection, sample_id)
            part2_result["actual_attack_type"] = connection.attack_type
            part2_result["actual_label"] = connection.attack_type.lower()
            part2_results.append(part2_result)

            # Rate limiting between samples
            await asyncio.sleep(0.5)

        total_time = time.time() - start_time

        logger.info(f"✅ {model_name} completed in {total_time:.2f} seconds")

        # Calculate metrics for both parts
        part1_metrics = self._calculate_binary_metrics(part1_results)
        part2_metrics = self._calculate_multiclass_metrics(part2_results)

        return {
            "model_name": model_name,
            "part1_results": part1_results,
            "part2_results": part2_results,
            "part1_metrics": part1_metrics,
            "part2_metrics": part2_metrics,
            "processing_time": total_time,
            "api_stats": {
                "total_requests": client.request_count,
                "total_tokens": client.total_tokens,
                "error_count": client.error_count,
            },
        }

    def _calculate_binary_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate metrics for binary attack detection (Part 1)."""

        valid_results = [r for r in results if r["prediction"] != "ERROR"]
        if not valid_results:
            return {"error": "No valid predictions"}

        tp = fp = tn = fn = 0
        correct = 0

        for result in valid_results:
            actual = result["actual_label"]
            predicted = (
                "ATTACK"
                if result["prediction"] != "NORMAL" and result["prediction"] != "UNKNOWN"
                else result["prediction"]
            )

            if predicted == actual:
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
        accuracy = correct / len(valid_results) if valid_results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "total_valid": len(valid_results),
        }

    def _calculate_multiclass_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate metrics for attack type classification (Part 2)."""

        valid_results = [r for r in results if r["prediction"] != "ERROR"]
        if not valid_results:
            return {"error": "No valid predictions"}

        correct = 0
        attack_type_accuracy: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )

        for result in valid_results:
            actual = result["actual_attack_type"].lower()
            predicted = result["prediction"].lower()

            attack_type_accuracy[actual]["total"] += 1

            if predicted == actual:
                correct += 1
                attack_type_accuracy[actual]["correct"] += 1

        # Calculate per-attack-type accuracy
        for attack_type in attack_type_accuracy:
            stats = attack_type_accuracy[attack_type]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        overall_accuracy = correct / len(valid_results) if valid_results else 0

        return {
            "overall_accuracy": overall_accuracy,
            "attack_type_accuracy": dict(attack_type_accuracy),
            "total_valid": len(valid_results),
        }

    async def run_all_models(self, connections: list[NetworkConnection]) -> dict[str, Any]:
        """Run benchmark on all models and return comparative results."""

        logger.info("🌍 Starting Multi-Language Model Benchmark")
        logger.info(f"📊 Dataset: {len(connections)} homogeneous samples")
        logger.info(f"🤖 Models: {len(self.models_to_test)} language models")
        logger.info("=" * 80)

        all_results = {}

        for i, model_name in enumerate(self.models_to_test, 1):
            logger.info(f"\n[{i}/{len(self.models_to_test)}] Testing {model_name}")
            logger.info("-" * 60)

            try:
                model_results = await self.run_two_part_benchmark(model_name, connections)
                all_results[model_name] = model_results

                # Quick summary
                part1_acc = model_results["part1_metrics"].get("accuracy", 0)
                part2_acc = model_results["part2_metrics"].get("overall_accuracy", 0)
                logger.info(f"✅ {model_name} completed:")
                logger.info(f"   Part 1 (Detection): {part1_acc:.1%} accuracy")
                logger.info(f"   Part 2 (Classification): {part2_acc:.1%} accuracy")

            except Exception as e:
                logger.error(f"❌ {model_name} failed: {str(e)}")
                all_results[model_name] = {
                    "model_name": model_name,
                    "error": str(e),
                    "status": "FAILED",
                }

            # Delay between models
            if i < len(self.models_to_test):
                logger.info("⏳ Waiting 30 seconds before next model...")
                await asyncio.sleep(30)

        return all_results

    def generate_comparative_report(
        self, all_results: dict[str, Any], connections: list[NetworkConnection]
    ) -> dict[str, Any]:
        """Generate comprehensive comparative report."""

        logger.info("📄 Generating comprehensive comparative report...")

        successful_results = {k: v for k, v in all_results.items() if "error" not in v}
        failed_results = {k: v for k, v in all_results.items() if "error" in v}

        # Dataset analysis
        attack_type_distribution = Counter(conn.attack_type for conn in connections)

        report = {
            "benchmark_info": {
                "benchmark_type": "Yeko's Multi-Language NSL-KDD Cybersecurity Benchmark",
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(connections),
                "models_tested": len(self.models_to_test),
                "successful_models": len(successful_results),
                "failed_models": len(failed_results),
                "dataset_distribution": dict(attack_type_distribution),
            },
            "part1_comparison": self._generate_part1_comparison(successful_results),
            "part2_comparison": self._generate_part2_comparison(successful_results),
            "model_rankings": self._generate_model_rankings(successful_results),
            "detailed_results": all_results,
            "failed_models": failed_results,
        }

        logger.info("✅ Comparative report generated successfully")
        return report

    def _generate_part1_comparison(self, successful_results: dict[str, Any]) -> dict[str, Any]:
        """Generate Part 1 (Attack Detection) comparison."""

        comparison = {}

        for model_name, results in successful_results.items():
            metrics = results.get("part1_metrics", {})
            if "error" not in metrics:
                comparison[model_name] = {
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "processing_time": results.get("processing_time", 0),
                }

        # Sort by F1 score
        sorted_comparison = dict(
            sorted(comparison.items(), key=lambda x: x[1]["f1_score"], reverse=True)
        )

        return {
            "description": "Binary Attack Detection Performance",
            "metrics": sorted_comparison,
            "best_model": list(sorted_comparison.keys())[0] if sorted_comparison else None,
        }

    def _generate_part2_comparison(self, successful_results: dict[str, Any]) -> dict[str, Any]:
        """Generate Part 2 (Attack Classification) comparison."""

        comparison = {}

        for model_name, results in successful_results.items():
            metrics = results.get("part2_metrics", {})
            if "error" not in metrics:
                comparison[model_name] = {
                    "overall_accuracy": metrics.get("overall_accuracy", 0),
                    "attack_type_accuracy": metrics.get("attack_type_accuracy", {}),
                    "processing_time": results.get("processing_time", 0),
                }

        # Sort by overall accuracy
        sorted_comparison = dict(
            sorted(comparison.items(), key=lambda x: x[1]["overall_accuracy"], reverse=True)
        )

        return {
            "description": "Multi-class Attack Type Classification Performance",
            "metrics": sorted_comparison,
            "best_model": list(sorted_comparison.keys())[0] if sorted_comparison else None,
        }

    def _generate_model_rankings(self, successful_results: dict[str, Any]) -> dict[str, Any]:
        """Generate overall model rankings combining both parts."""

        rankings = {}

        for model_name, results in successful_results.items():
            part1_metrics = results.get("part1_metrics", {})
            part2_metrics = results.get("part2_metrics", {})

            if "error" not in part1_metrics and "error" not in part2_metrics:
                # Composite score: weighted average of Part 1 F1 and Part 2 accuracy
                part1_score = part1_metrics.get("f1_score", 0)
                part2_score = part2_metrics.get("overall_accuracy", 0)
                composite_score = (part1_score * 0.6) + (
                    part2_score * 0.4
                )  # Weight detection higher

                rankings[model_name] = {
                    "composite_score": composite_score,
                    "part1_f1": part1_score,
                    "part2_accuracy": part2_score,
                    "processing_time": results.get("processing_time", 0),
                }

        # Sort by composite score
        sorted_rankings = dict(
            sorted(rankings.items(), key=lambda x: x[1]["composite_score"], reverse=True)
        )

        return {
            "description": "Overall Model Rankings (Composite Score)",
            "rankings": sorted_rankings,
            "champion": list(sorted_rankings.keys())[0] if sorted_rankings else None,
        }

    def save_comparative_report(
        self, report: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Save comprehensive comparative report."""

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"yeko_openrouter_multilang_report_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Comparative report saved to: {output_file}")
        return output_file

    def print_executive_summary(self, report: dict[str, Any]) -> None:
        """Print executive summary of comparative benchmark."""

        info = report["benchmark_info"]
        part1_comp = report["part1_comparison"]
        part2_comp = report["part2_comparison"]
        rankings = report["model_rankings"]

        print("\n" + "=" * 100)
        print("🏆 YEKO'S MULTI-LANGUAGE CYBERSECURITY BENCHMARK - EXECUTIVE SUMMARY")
        print("=" * 100)

        print("\n🔬 BENCHMARK OVERVIEW")
        print(f"   Timestamp: {info['timestamp']}")
        print(f"   Total Samples: {info['total_samples']} (homogeneous across attack types)")
        print(f"   Models Tested: {info['models_tested']}")
        print(f"   Successful: {info['successful_models']}/{info['models_tested']}")
        print("   Two-Part Testing: Attack Detection + Attack Classification")

        print("\n🥇 OVERALL CHAMPION")
        champion = rankings.get("champion")
        if champion:
            champion_data = rankings["rankings"][champion]
            print(f"   🏆 {champion}")
            print(f"   Composite Score: {champion_data['composite_score']:.3f}")
            print(f"   Attack Detection F1: {champion_data['part1_f1']:.3f}")
            print(f"   Attack Classification: {champion_data['part2_accuracy']:.1%}")

        print("\n📊 PART 1: ATTACK DETECTION RANKINGS")
        part1_metrics = part1_comp.get("metrics", {})
        for i, (model, metrics) in enumerate(list(part1_metrics.items())[:5], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
            print(f"   {emoji} {i}. {model}")
            print(
                f"      F1-Score: {metrics['f1_score']:.3f} | Accuracy: {metrics['accuracy']:.1%}"
            )

        print("\n🎯 PART 2: ATTACK CLASSIFICATION RANKINGS")
        part2_metrics = part2_comp.get("metrics", {})
        for i, (model, metrics) in enumerate(list(part2_metrics.items())[:5], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
            print(f"   {emoji} {i}. {model}")
            print(f"      Accuracy: {metrics['overall_accuracy']:.1%}")

        if report["failed_models"]:
            print("\n❌ FAILED MODELS")
            for model, data in report["failed_models"].items():
                print(f"   - {model}: {data.get('error', 'Unknown error')}")

        print("\n" + "=" * 100)


async def main() -> int:
    """Main function to run multi-language model benchmark."""

    parser = argparse.ArgumentParser(
        description="Yeko's Multi-Language NSL-KDD Cybersecurity Benchmark"
    )
    parser.add_argument("--dataset", default="KDDTrain+.txt", help="NSL-KDD dataset file to use")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples for homogeneous testing (default: 1000)",
    )
    parser.add_argument("--output", help="Output file name for comparative report")
    parser.add_argument(
        "--api-key", help="OpenRouter API key (optional, will use hardcoded if not provided)"
    )

    args = parser.parse_args()

    # Hardcoded API key - replace with your actual key
    HARDCODED_API_KEY = "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"  # Replace this with your actual OpenRouter API key

    api_key = args.api_key if args.api_key else HARDCODED_API_KEY

    if api_key == "YOUR_KEY_HERE":
        logger.error(
            "❌ Please replace the hardcoded API key in the script with your actual OpenRouter API key"
        )
        return 1

    print("🌍 YEKO'S MULTI-LANGUAGE NSL-KDD CYBERSECURITY BENCHMARK")
    print("=" * 80)
    print("🧪 Two-Part Testing: Attack Detection + Attack Classification")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 Samples: {args.samples} (homogeneous across attack types)")
    print("🤖 Multi-Model Comparison with Comprehensive Reporting")
    print("=" * 80)

    try:
        # Initialize benchmark system
        benchmark = YekoMultiLangBenchmark(api_key)

        # Load homogeneous dataset
        connections = benchmark.load_homogeneous_dataset(
            filename=args.dataset, target_samples=args.samples
        )

        # Run comprehensive multi-model benchmark
        all_results = await benchmark.run_all_models(connections)

        # Generate comparative report
        report = benchmark.generate_comparative_report(all_results, connections)

        # Print executive summary
        benchmark.print_executive_summary(report)

        # Save comprehensive report
        output_file = benchmark.save_comparative_report(report, args.output)
        print(f"\n💾 Detailed comparative report saved to: {output_file}")

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

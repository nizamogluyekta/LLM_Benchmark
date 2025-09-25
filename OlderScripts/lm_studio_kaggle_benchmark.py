#!/usr/bin/env python3
"""
LM Studio Few-Shot Learning Cybersecurity Benchmark with Kaggle Dataset Support

This script extends the few-shot learning benchmark to support loading datasets from:
1. Kaggle cybersecurity datasets (CSV/JSON format)
2. Custom CSV files with cybersecurity data
3. Built-in synthetic dataset generation

Supported Kaggle datasets:
- Network Intrusion Detection datasets
- Malware Detection datasets
- Phishing Detection datasets
- Any cybersecurity dataset with text and binary labels

Usage:
    # Use Kaggle dataset
    python lm_studio_kaggle_benchmark.py --model-name "llama-model" --kaggle-dataset "path/to/dataset.csv"

    # Use built-in synthetic data
    python lm_studio_kaggle_benchmark.py --model-name "llama-model" --use-synthetic

    # Specify custom columns
    python lm_studio_kaggle_benchmark.py --model-name "llama-model" --kaggle-dataset "data.csv" --text-column "description" --label-column "is_attack"
"""

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
import pandas as pd


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    dataset_source: str
    total_samples: int
    training_samples: int
    test_samples: int
    few_shot_examples: int
    zero_shot_accuracy: float
    few_shot_accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    samples_per_second: float
    predictions: list[dict[str, Any]]
    evaluation_details: dict[str, Any]
    training_examples_used: list[dict[str, Any]]


class KaggleDatasetLoader:
    """Loads and preprocesses Kaggle cybersecurity datasets."""

    def __init__(self) -> None:
        self.supported_formats = [".csv", ".json", ".jsonl"]
        self.logger = logging.getLogger("kaggle_loader")

    def load_dataset(
        self,
        file_path: str,
        text_column: str = "text",
        label_column: str = "label",
        max_samples: int | None = None,
    ) -> list[dict[str, str]]:
        """Load dataset from file.

        Args:
            file_path: Path to the dataset file
            text_column: Name of the column containing text data
            label_column: Name of the column containing labels
            max_samples: Maximum number of samples to load

        Returns:
            List of samples with 'text' and 'label' keys
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self.logger.info(f"Loading dataset from: {file_path}")

        # Load based on file extension
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(file_path)
        elif path.suffix.lower() == ".jsonl":
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.logger.info(f"Loaded {len(df)} rows from dataset")
        self.logger.info(f"Columns available: {list(df.columns)}")

        # Validate required columns
        if text_column not in df.columns:
            raise ValueError(
                f"Text column '{text_column}' not found. Available: {list(df.columns)}"
            )
        if label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found. Available: {list(df.columns)}"
            )

        # Convert to standard format
        samples = []
        for _, row in df.iterrows():
            text = str(row[text_column]).strip()
            label = str(row[label_column]).strip()

            # Normalize labels to ATTACK/BENIGN
            normalized_label = self._normalize_label(label)

            if text and normalized_label:
                samples.append({"text": text, "label": normalized_label, "original_label": label})

        # Limit samples if requested
        if max_samples and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
            self.logger.info(f"Randomly sampled {max_samples} examples")

        # Balance dataset
        samples = self._balance_dataset(samples)

        self.logger.info(f"Final dataset: {len(samples)} samples")
        attack_count = sum(1 for s in samples if s["label"] == "ATTACK")
        benign_count = len(samples) - attack_count
        self.logger.info(f"Distribution: {attack_count} ATTACK, {benign_count} BENIGN")

        return samples

    def _normalize_label(self, label: str) -> str | None:
        """Normalize various label formats to ATTACK/BENIGN.

        Args:
            label: Original label from dataset

        Returns:
            Normalized label or None if not recognized
        """
        label_lower = label.lower().strip()

        # Attack indicators
        attack_keywords = [
            "attack",
            "malicious",
            "malware",
            "phishing",
            "intrusion",
            "anomaly",
            "suspicious",
            "threat",
            "exploit",
            "vulnerability",
            "dos",
            "ddos",
            "injection",
            "backdoor",
            "trojan",
            "virus",
            "botnet",
            "ransomware",
            "spam",
            "fraud",
            "breach",
            "compromise",
            "1",
            "true",
            "yes",
            "positive",
        ]

        # Benign indicators
        benign_keywords = [
            "normal",
            "benign",
            "legitimate",
            "clean",
            "safe",
            "regular",
            "standard",
            "baseline",
            "ok",
            "good",
            "valid",
            "authorized",
            "0",
            "false",
            "no",
            "negative",
        ]

        if any(keyword in label_lower for keyword in attack_keywords):
            return "ATTACK"
        elif any(keyword in label_lower for keyword in benign_keywords):
            return "BENIGN"
        else:
            self.logger.warning(f"Could not normalize label: {label}")
            return None

    def _balance_dataset(self, samples: list[dict[str, str]]) -> list[dict[str, str]]:
        """Balance the dataset to have equal ATTACK and BENIGN samples.

        Args:
            samples: List of samples

        Returns:
            Balanced list of samples
        """
        attack_samples = [s for s in samples if s["label"] == "ATTACK"]
        benign_samples = [s for s in samples if s["label"] == "BENIGN"]

        if not attack_samples or not benign_samples:
            self.logger.warning("Dataset is not balanced - missing one class")
            return samples

        # Use the smaller class size as the target
        min_count = min(len(attack_samples), len(benign_samples))

        balanced_samples = random.sample(attack_samples, min_count) + random.sample(
            benign_samples, min_count
        )

        random.shuffle(balanced_samples)

        self.logger.info(f"Balanced dataset: {min_count} samples per class")
        return balanced_samples


class LMStudioKaggleBenchmark:
    """LM Studio benchmark with Kaggle dataset support."""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        """Initialize the benchmark."""
        self.base_url = base_url
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        self.logger = self._setup_logging()
        self.dataset_loader: KaggleDatasetLoader = KaggleDatasetLoader()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger("kaggle_benchmark")

    def generate_synthetic_cybersecurity_data(
        self, total_samples: int = 200
    ) -> list[dict[str, str]]:
        """Generate synthetic cybersecurity data for testing."""
        attack_templates = [
            "SQL injection attempt detected: {query} from IP {ip}",
            "Brute force login attack: {attempts} failed attempts from {ip}",
            "Malware signature found: {hash} in file {filename}",
            "DDoS attack: {rps} requests/sec from {sources} IPs",
            "Phishing email: fake {service} login at {url}",
            "Port scan detected: {ports} scanned by {ip}",
            "Data exfiltration: {size} uploaded to {server}",
            "Privilege escalation attempt by user {user}",
            "Suspicious PowerShell execution: {command}",
            "Network anomaly: unusual traffic pattern from {subnet}",
        ]

        benign_templates = [
            "User {user} logged in successfully from {ip}",
            "Backup completed: {size} backed up to {location}",
            "Software update: {patch} installed on {systems}",
            "Normal web request: {method} {url} from {browser}",
            "Database query: SELECT operation on {table}",
            "Email sent: {subject} from {sender} to {recipient}",
            "System health check: all services running normally",
            "File transfer: {file} uploaded to secure storage",
            "VPN connection established for user {user}",
            "Certificate renewal completed for {domain}",
        ]

        samples = []
        attack_count = total_samples // 2
        benign_count = total_samples - attack_count

        # Generate attack samples
        for _ in range(attack_count):
            template = random.choice(attack_templates)
            values = {
                "query": "' OR '1'='1' --",
                "ip": f"192.168.1.{random.randint(1, 254)}",
                "attempts": random.randint(50, 500),
                "hash": "".join(random.choices("0123456789abcdef", k=32)),
                "filename": random.choice(["document.pdf", "update.exe", "script.ps1"]),
                "rps": random.randint(1000, 50000),
                "sources": random.randint(10, 500),
                "service": random.choice(["PayPal", "Amazon", "Microsoft"]),
                "url": "http://malicious-site.com/login",
                "ports": "22,80,443,3389",
                "size": f"{random.randint(100, 1000)}MB",
                "server": "ftp.suspicious.com",
                "user": random.choice(["admin", "root", "guest"]),
                "command": "Invoke-Expression (New-Object Net.WebClient).DownloadString(...)",
                "subnet": "10.0.0.0/24",
            }

            try:
                text = template.format(**values)
            except KeyError:
                text = template

            samples.append({"text": text, "label": "ATTACK", "original_label": "attack"})

        # Generate benign samples
        for _ in range(benign_count):
            template = random.choice(benign_templates)
            values = {
                "user": random.choice(["john.doe", "alice.smith", "bob.jones"]),
                "ip": f"192.168.1.{random.randint(1, 254)}",
                "size": f"{random.randint(1, 100)}GB",
                "location": "secure backup server",
                "patch": f"KB{random.randint(1000000, 9999999)}",
                "systems": f"{random.randint(5, 50)} servers",
                "method": random.choice(["GET", "POST", "PUT"]),
                "url": "/api/data",
                "browser": "Mozilla/5.0 (legitimate browser)",
                "table": random.choice(["users", "orders", "products"]),
                "subject": "Weekly Security Report",
                "sender": "security@company.com",
                "recipient": "management@company.com",
                "file": "quarterly_report.pdf",
                "domain": "company.com",
            }

            try:
                text = template.format(**values)
            except KeyError:
                text = template

            samples.append({"text": text, "label": "BENIGN", "original_label": "normal"})

        random.shuffle(samples)
        return samples

    def split_dataset(
        self, dataset: list[dict[str, str]], train_ratio: float = 0.75
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Split dataset into training and testing sets while maintaining balance."""

        # Separate by label
        attack_samples = [s for s in dataset if s["label"] == "ATTACK"]
        benign_samples = [s for s in dataset if s["label"] == "BENIGN"]

        # Split each class
        attack_train_size = int(len(attack_samples) * train_ratio)
        benign_train_size = int(len(benign_samples) * train_ratio)

        attack_train = attack_samples[:attack_train_size]
        attack_test = attack_samples[attack_train_size:]

        benign_train = benign_samples[:benign_train_size]
        benign_test = benign_samples[benign_train_size:]

        # Combine and shuffle
        train_set = attack_train + benign_train
        test_set = attack_test + benign_test

        random.shuffle(train_set)
        random.shuffle(test_set)

        return train_set, test_set

    def create_few_shot_prompt(
        self, training_examples: list[dict[str, str]], test_sample: str, num_examples: int = 10
    ) -> tuple[str, list[dict[str, str]]]:
        """Create a few-shot learning prompt with training examples."""

        # Select balanced examples
        attack_examples = [ex for ex in training_examples if ex["label"] == "ATTACK"]
        benign_examples = [ex for ex in training_examples if ex["label"] == "BENIGN"]

        # Take equal numbers from each class
        examples_per_class = num_examples // 2
        selected_attack = random.sample(
            attack_examples, min(examples_per_class, len(attack_examples))
        )
        selected_benign = random.sample(
            benign_examples, min(examples_per_class, len(benign_examples))
        )

        examples = selected_attack + selected_benign
        random.shuffle(examples)

        prompt = """You are a cybersecurity expert analyzing network logs and security events. Based on the training examples below, classify new events as either ATTACK or BENIGN.

TRAINING EXAMPLES:
"""

        for i, example in enumerate(examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Event: {example['text']}\n"
            prompt += f"Classification: {example['label']}\n"

        prompt += f"""

Now classify this new event:

Event: {test_sample}

Provide your analysis in this format:
Classification: [ATTACK or BENIGN]
Confidence: [0.0 to 1.0]
Explanation: [Brief explanation of your reasoning based on the training examples]

Analysis:"""

        return prompt, examples

    async def run_few_shot_prediction(
        self,
        sample: str,
        training_examples: list[dict[str, str]],
        model_name: str,
        num_examples: int = 10,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Run few-shot prediction with training examples."""
        prompt, used_examples = self.create_few_shot_prompt(training_examples, sample, num_examples)

        try:
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert. Learn from the training examples to classify new events.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            inference_time = time.time() - start_time
            content = response.choices[0].message.content
            parsed = self._parse_response(content or "")
            parsed["inference_time_ms"] = inference_time * 1000
            parsed["model_response"] = content
            parsed["tokens_used"] = response.usage.total_tokens if response.usage else 0

            return parsed, used_examples

        except Exception as e:
            return {
                "classification": "ERROR",
                "confidence": 0.0,
                "explanation": str(e),
                "inference_time_ms": 0.0,
            }, used_examples

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse model response to extract structured information."""
        import re

        parsed = {
            "classification": "BENIGN",
            "confidence": 0.5,
            "explanation": "",
        }

        try:
            # Extract classification
            classification_match = re.search(
                r"Classification:\s*(ATTACK|BENIGN)", response, re.IGNORECASE
            )
            if classification_match:
                parsed["classification"] = classification_match.group(1).upper()

            # Extract confidence
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                parsed["confidence"] = max(0.0, min(1.0, confidence))

            # Extract explanation
            explanation_match = re.search(
                r"Explanation:\s*(.+?)(?=\n\s*$|$)", response, re.IGNORECASE | re.DOTALL
            )
            if explanation_match:
                parsed["explanation"] = explanation_match.group(1).strip()

        except Exception as e:
            parsed["explanation"] = f"Parse error: {str(e)}"

        return parsed

    def calculate_metrics(
        self, predictions: list[dict[str, Any]], ground_truth: list[str]
    ) -> dict[str, float]:
        """Calculate evaluation metrics."""
        if not predictions:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "true_positives": 0.0,
                "false_positives": 0.0,
                "false_negatives": 0.0,
                "true_negatives": 0.0,
                "total_samples": 0.0,
                "error_count": 0.0,
            }

        # Extract valid predictions
        valid_predictions = [
            (p, g)
            for p, g in zip(predictions, ground_truth, strict=False)
            if p.get("classification") != "ERROR"
        ]
        if not valid_predictions:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "true_positives": 0.0,
                "false_positives": 0.0,
                "false_negatives": 0.0,
                "true_negatives": 0.0,
                "total_samples": 0.0,
                "error_count": float(len(predictions)),
            }

        pred_labels = [p["classification"] for p, _ in valid_predictions]
        true_labels = [g for _, g in valid_predictions]

        # Calculate metrics
        correct = sum(1 for p, t in zip(pred_labels, true_labels, strict=False) if p == t)
        total = len(true_labels)
        accuracy = correct / total if total > 0 else 0.0

        # Binary classification metrics
        tp = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "ATTACK" and t == "ATTACK"
        )
        fp = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "ATTACK" and t == "BENIGN"
        )
        fn = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "BENIGN" and t == "ATTACK"
        )
        tn = sum(
            1
            for p, t in zip(pred_labels, true_labels, strict=False)
            if p == "BENIGN" and t == "BENIGN"
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
            "error_count": float(len(predictions) - len(valid_predictions)),
        }

    async def run_benchmark(
        self,
        model_name: str,
        dataset_path: str | None = None,
        text_column: str = "text",
        label_column: str = "label",
        max_samples: int | None = None,
        few_shot_examples: int = 10,
    ) -> BenchmarkResult:
        """Run complete benchmark with Kaggle dataset or synthetic data."""

        self.logger.info(f"Starting benchmark for model: {model_name}")

        # Load dataset
        if dataset_path:
            self.logger.info(f"Loading Kaggle dataset from: {dataset_path}")
            dataset = self.dataset_loader.load_dataset(
                dataset_path, text_column, label_column, max_samples
            )
            dataset_source = f"Kaggle: {Path(dataset_path).name}"
        else:
            self.logger.info("Using synthetic cybersecurity dataset")
            dataset = self.generate_synthetic_cybersecurity_data(max_samples or 200)
            dataset_source = "Synthetic"

        attack_count = sum(1 for s in dataset if s["label"] == "ATTACK")
        benign_count = sum(1 for s in dataset if s["label"] == "BENIGN")
        self.logger.info(
            f"Dataset loaded: {len(dataset)} samples ({attack_count} ATTACK, {benign_count} BENIGN)"
        )

        # Split into train/test
        train_set, test_set = self.split_dataset(dataset, train_ratio=0.75)
        self.logger.info(f"Split: {len(train_set)} training, {len(test_set)} testing samples")

        # Run few-shot predictions on test set
        self.logger.info("Running few-shot predictions...")
        start_time = time.time()

        few_shot_predictions = []
        all_training_examples = []

        for i, sample in enumerate(test_set):
            if i % 10 == 0:
                self.logger.info(f"Processing test sample {i + 1}/{len(test_set)}")

            pred, training_examples = await self.run_few_shot_prediction(
                sample["text"], train_set, model_name, few_shot_examples
            )

            pred["sample_id"] = f"test_{i}"
            pred["ground_truth"] = sample["label"]
            pred["input_text"] = sample["text"]
            pred["original_label"] = sample.get("original_label", "")

            few_shot_predictions.append(pred)

            if i == 0:  # Store training examples used for the first sample
                all_training_examples = training_examples

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate metrics
        few_shot_metrics = self.calculate_metrics(
            few_shot_predictions, [s["label"] for s in test_set]
        )

        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            dataset_source=dataset_source,
            total_samples=len(dataset),
            training_samples=len(train_set),
            test_samples=len(test_set),
            few_shot_examples=few_shot_examples,
            zero_shot_accuracy=0.0,  # Not calculated in this version
            few_shot_accuracy=few_shot_metrics.get("accuracy", 0.0),
            precision=few_shot_metrics.get("precision", 0.0),
            recall=few_shot_metrics.get("recall", 0.0),
            f1_score=few_shot_metrics.get("f1_score", 0.0),
            processing_time=processing_time,
            samples_per_second=len(test_set) / processing_time if processing_time > 0 else 0.0,
            predictions=few_shot_predictions,
            evaluation_details=few_shot_metrics,
            training_examples_used=all_training_examples,
        )

        self.logger.info("Benchmark completed successfully!")
        return result

    def generate_report(self, result: BenchmarkResult) -> None:
        """Generate comprehensive benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate detailed JSON report
        json_report = f"kaggle_benchmark_{timestamp}.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": result.model_name,
            "dataset_source": result.dataset_source,
            "methodology": "few_shot_learning_with_kaggle_data",
            "dataset_info": {
                "total_samples": result.total_samples,
                "training_samples": result.training_samples,
                "test_samples": result.test_samples,
                "few_shot_examples": result.few_shot_examples,
            },
            "metrics": {
                "accuracy": result.few_shot_accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "processing_time": result.processing_time,
                "samples_per_second": result.samples_per_second,
            },
            "detailed_metrics": result.evaluation_details,
            "training_examples_used": result.training_examples_used,
            "predictions": result.predictions,
        }

        with open(json_report, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate human-readable summary
        summary_report = f"kaggle_summary_{timestamp}.txt"
        with open(summary_report, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("LM Studio Kaggle Dataset Cybersecurity Benchmark Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {result.model_name}\n")
            f.write(f"Dataset Source: {result.dataset_source}\n")
            f.write(f"Methodology: Few-Shot Learning with {result.few_shot_examples} examples\n\n")

            f.write("DATASET INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Samples: {result.total_samples}\n")
            f.write(f"Training Samples: {result.training_samples}\n")
            f.write(f"Test Samples: {result.test_samples}\n")
            f.write(f"Examples per prediction: {result.few_shot_examples}\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {result.few_shot_accuracy:.1%}\n")
            f.write(f"Precision: {result.precision:.1%}\n")
            f.write(f"Recall: {result.recall:.1%}\n")
            f.write(f"F1-Score: {result.f1_score:.1%}\n\n")

            f.write("PROCESSING PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Time: {result.processing_time:.2f} seconds\n")
            f.write(f"Speed: {result.samples_per_second:.2f} samples/second\n")
            f.write(
                f"Avg Time/Sample: {result.processing_time / result.test_samples:.2f} seconds\n\n"
            )

            # Sample predictions
            f.write("SAMPLE PREDICTIONS:\n")
            f.write("-" * 30 + "\n")
            for i, pred in enumerate(result.predictions[:5], 1):
                f.write(f"Sample {i}:\n")
                f.write(f"  Input: {pred['input_text'][:80]}...\n")
                f.write(f"  Expected: {pred['ground_truth']}\n")
                f.write(f"  Predicted: {pred['classification']}\n")
                f.write(f"  Confidence: {pred.get('confidence', 0):.2f}\n")
                f.write(f"  Original Label: {pred.get('original_label', 'N/A')}\n\n")

        # Print summary to console
        print("\n" + "=" * 70)
        print("🎉 KAGGLE DATASET BENCHMARK COMPLETED!")
        print("=" * 70)
        print(f"📊 Model: {result.model_name}")
        print(f"📁 Dataset: {result.dataset_source}")
        print(f"📄 Detailed Report: {json_report}")
        print(f"📝 Summary Report: {summary_report}")
        print("-" * 70)
        print("📈 RESULTS:")
        print(f"   🎯 Accuracy: {result.few_shot_accuracy:.1%}")
        print(f"   📏 Precision: {result.precision:.1%}")
        print(f"   🔍 Recall: {result.recall:.1%}")
        print(f"   ⚖️  F1-Score: {result.f1_score:.1%}")
        print(f"   ⚡ Speed: {result.samples_per_second:.1f} samples/sec")
        print(f"   📊 Total Samples: {result.total_samples}")
        print("=" * 70)


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="LM Studio Cybersecurity Benchmark with Kaggle Dataset Support"
    )
    parser.add_argument("--model-name", type=str, required=True, help="Model name in LM Studio")
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:1234/v1", help="LM Studio base URL"
    )
    parser.add_argument("--kaggle-dataset", type=str, help="Path to Kaggle dataset file (CSV/JSON)")
    parser.add_argument(
        "--text-column", type=str, default="text", help="Name of text column in dataset"
    )
    parser.add_argument(
        "--label-column", type=str, default="label", help="Name of label column in dataset"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to use from dataset"
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=10,
        help="Number of training examples to use per prediction (default: 10)",
    )
    parser.add_argument(
        "--use-synthetic", action="store_true", help="Use synthetic data instead of Kaggle dataset"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)

    # Run benchmark
    benchmark = LMStudioKaggleBenchmark(base_url=args.base_url)

    try:
        if args.use_synthetic or not args.kaggle_dataset:
            # Use synthetic data
            result = await benchmark.run_benchmark(
                args.model_name,
                dataset_path=None,
                max_samples=args.max_samples,
                few_shot_examples=args.few_shot_examples,
            )
        else:
            # Use Kaggle dataset
            result = await benchmark.run_benchmark(
                args.model_name,
                dataset_path=args.kaggle_dataset,
                text_column=args.text_column,
                label_column=args.label_column,
                max_samples=args.max_samples,
                few_shot_examples=args.few_shot_examples,
            )

        benchmark.generate_report(result)

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logging.error(f"Benchmark failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
LM Studio Benchmark Script - Direct API Version

This script runs a complete cybersecurity benchmark using LM Studio models
via direct OpenAI-compatible API calls, bypassing the model service integration.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import openai


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    total_samples: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    samples_per_second: float
    predictions: list[dict[str, Any]]
    evaluation_details: dict[str, Any]


class LMStudioBenchmark:
    """Complete benchmarking solution for LM Studio models."""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        """Initialize the benchmark."""
        self.base_url = base_url
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key="lm-studio",  # Dummy key for local
        )
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger("lm_studio_benchmark")

    def generate_cybersecurity_samples(self) -> list[dict[str, str]]:
        """Generate comprehensive cybersecurity test samples."""
        return [
            # SQL Injection Attacks
            {
                "text": "192.168.1.100 attempted SQL injection: ' OR '1'='1' -- on login form at /admin/login.php",
                "label": "ATTACK",
                "attack_type": "sql_injection",
            },
            {
                "text": "Database query detected: SELECT * FROM users WHERE username='admin' AND password='' OR '1'='1'--",
                "label": "ATTACK",
                "attack_type": "sql_injection",
            },
            # Brute Force Attacks
            {
                "text": "Multiple failed login attempts from 203.0.113.42: admin/password123, admin/admin, root/toor, admin/12345",
                "label": "ATTACK",
                "attack_type": "brute_force",
            },
            {
                "text": "SSH brute force detected: 847 failed attempts from 198.51.100.25 in 10 minutes targeting root account",
                "label": "ATTACK",
                "attack_type": "brute_force",
            },
            # Malware Detection
            {
                "text": "Suspicious file hash detected: 5d41402abc4b2a76b9719d911017c592 - matches known Trojan.Win32.Malware signature",
                "label": "ATTACK",
                "attack_type": "malware",
            },
            {
                "text": "Antivirus alert: Suspicious PowerShell execution attempting to download payload from malicious-domain.com",
                "label": "ATTACK",
                "attack_type": "malware",
            },
            # DDoS Attacks
            {
                "text": "DDoS attack detected: 10,000 requests/second from botnet IPs targeting server 192.168.1.50 port 80",
                "label": "ATTACK",
                "attack_type": "ddos",
            },
            {
                "text": "Network anomaly: Unusual traffic spike 50x normal baseline, 500+ unique source IPs, SYN flood pattern",
                "label": "ATTACK",
                "attack_type": "ddos",
            },
            # Phishing Attempts
            {
                "text": "Phishing email detected: fake PayPal login page at http://paypal-secure.malicious-site.com/login.html",
                "label": "ATTACK",
                "attack_type": "phishing",
            },
            {
                "text": "Suspicious email: 'Urgent: Verify your bank account' with link to fake-bank-security.evil.com",
                "label": "ATTACK",
                "attack_type": "phishing",
            },
            # Network Scanning
            {
                "text": "Port scan detected from 198.51.100.25: scanning ports 22, 80, 443, 3389, 5432 on target 10.0.0.15",
                "label": "ATTACK",
                "attack_type": "network_scan",
            },
            {
                "text": "Nmap scan identified: TCP SYN stealth scan targeting internal subnet 192.168.1.0/24 from external IP",
                "label": "ATTACK",
                "attack_type": "network_scan",
            },
            # Data Exfiltration
            {
                "text": "Data exfiltration attempt: 500MB upload to external FTP server ftp.suspicious-domain.com during off-hours",
                "label": "ATTACK",
                "attack_type": "data_exfiltration",
            },
            {
                "text": "Unusual data transfer: 2GB sensitive customer database copied to external cloud storage at 3:00 AM",
                "label": "ATTACK",
                "attack_type": "data_exfiltration",
            },
            # Buffer Overflow
            {
                "text": "Buffer overflow exploit attempt in HTTP request: AAAAAAAA...AAAA (1024 'A' characters) targeting /cgi-bin/vulnerable.cgi",
                "label": "ATTACK",
                "attack_type": "buffer_overflow",
            },
            {
                "text": "Memory corruption attempt detected: Shellcode pattern in input field exceeding 512-byte buffer limit",
                "label": "ATTACK",
                "attack_type": "buffer_overflow",
            },
            # Benign Activities
            {
                "text": "User john.doe successfully logged in from 192.168.1.55 at 09:15 AM using valid credentials",
                "label": "BENIGN",
                "attack_type": "none",
            },
            {
                "text": "Scheduled backup completed successfully: 2.5GB customer data backed up to secure network storage",
                "label": "BENIGN",
                "attack_type": "none",
            },
            {
                "text": "Software update installed: Security patch KB4052623 applied to all Windows servers during maintenance window",
                "label": "BENIGN",
                "attack_type": "none",
            },
            {
                "text": "Normal web traffic: GET /products/laptop HTTP/1.1 from legitimate customer browser with valid session",
                "label": "BENIGN",
                "attack_type": "none",
            },
            {
                "text": "Database maintenance: Index rebuild completed on customer_orders table, performance improved by 15%",
                "label": "BENIGN",
                "attack_type": "none",
            },
            {
                "text": "Email sent successfully: Weekly security report from security@company.com to management team",
                "label": "BENIGN",
                "attack_type": "none",
            },
            {
                "text": "System health check: All services running normally, CPU usage 25%, Memory 60%, Disk space 70% available",
                "label": "BENIGN",
                "attack_type": "none",
            },
            {
                "text": "File transfer completed: quarterly_financial_report.pdf uploaded to secure document management system",
                "label": "BENIGN",
                "attack_type": "none",
            },
        ]

    async def run_prediction(self, sample: dict[str, str], model_name: str) -> dict[str, Any]:
        """Run prediction on a single sample."""
        try:
            start_time = time.time()

            # Create cybersecurity analysis prompt
            prompt = f"""Analyze the following network log entry or security event for potential cybersecurity threats:

Event: {sample["text"]}

Provide your analysis in the following structured format:
Classification: [ATTACK or BENIGN]
Confidence: [0.0 to 1.0]
Attack_Type: [sql_injection, brute_force, malware, ddos, phishing, network_scan, data_exfiltration, buffer_overflow, or none if benign]
Explanation: [Brief explanation of your reasoning and what indicators led to this classification]
IOCs: [List any indicators of compromise found, comma-separated, or None if no IOCs]

Analysis:"""

            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert analyzing network logs and security events for potential threats. Provide structured analysis with clear classifications.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.1,
            )

            inference_time = time.time() - start_time
            model_response = response.choices[0].message.content

            # Parse structured response
            parsed = self._parse_response(model_response or "")

            return {
                "sample_id": f"sample_{hash(sample['text']) % 10000}",
                "input_text": sample["text"],
                "ground_truth": sample["label"],
                "ground_truth_attack_type": sample.get("attack_type", "none"),
                "prediction": parsed["classification"],
                "confidence": parsed["confidence"],
                "predicted_attack_type": parsed.get("attack_type"),
                "explanation": parsed.get("explanation", ""),
                "iocs": parsed.get("iocs", []),
                "inference_time_ms": inference_time * 1000,
                "model_response": model_response,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "model_version": response.model,
            }

        except Exception as e:
            self.logger.error(f"Prediction failed for sample: {e}")
            return {
                "sample_id": f"sample_{hash(sample['text']) % 10000}",
                "input_text": sample["text"],
                "ground_truth": sample["label"],
                "prediction": "ERROR",
                "confidence": 0.0,
                "explanation": f"Prediction failed: {str(e)}",
                "inference_time_ms": 0.0,
                "error": str(e),
            }

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse structured response from the model."""
        import re

        parsed = {
            "classification": "BENIGN",
            "confidence": 0.5,
            "attack_type": None,
            "explanation": "",
            "iocs": [],
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

            # Extract attack type
            attack_type_match = re.search(
                r"Attack_Type:\s*(sql_injection|brute_force|malware|ddos|phishing|network_scan|data_exfiltration|buffer_overflow|none)",
                response,
                re.IGNORECASE,
            )
            if attack_type_match and attack_type_match.group(1).lower() != "none":
                parsed["attack_type"] = attack_type_match.group(1).lower()

            # Extract explanation
            explanation_match = re.search(
                r"Explanation:\s*(.+?)(?=\n\s*IOCs:|$)", response, re.IGNORECASE | re.DOTALL
            )
            if explanation_match:
                parsed["explanation"] = explanation_match.group(1).strip()

            # Extract IOCs
            iocs_match = re.search(r"IOCs:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
            if iocs_match:
                iocs_text = iocs_match.group(1).strip()
                if iocs_text and iocs_text.lower() not in ["none", "n/a", "null"]:
                    iocs = [ioc.strip() for ioc in iocs_text.split(",") if ioc.strip()]
                    parsed["iocs"] = iocs

        except Exception as e:
            self.logger.warning(f"Failed to parse response: {e}")

        return parsed

    def calculate_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        if not predictions:
            return {}

        # Extract predictions and ground truth
        pred_labels = [p.get("prediction", "BENIGN") for p in predictions]
        true_labels = [p.get("ground_truth", "BENIGN") for p in predictions]

        # Filter out error predictions
        valid_predictions = [
            (p, t) for p, t in zip(pred_labels, true_labels, strict=False) if p != "ERROR"
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

        pred_labels = [p for p, _ in valid_predictions]
        true_labels = [t for _, t in valid_predictions]

        # Calculate basic metrics
        correct = sum(1 for p, t in zip(pred_labels, true_labels, strict=False) if p == t)
        total = len(true_labels)
        accuracy = correct / total if total > 0 else 0.0

        # Calculate binary classification metrics (ATTACK vs BENIGN)
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
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_samples": total,
            "error_count": len(predictions) - len(valid_predictions),
        }

    async def run_benchmark(self, model_name: str) -> BenchmarkResult:
        """Run complete benchmark on the model."""
        self.logger.info(f"Starting benchmark for model: {model_name}")

        # Generate test samples
        samples = self.generate_cybersecurity_samples()
        self.logger.info(f"Generated {len(samples)} cybersecurity test samples")

        # Run predictions
        self.logger.info("Running predictions...")
        start_time = time.time()

        predictions = []
        for i, sample in enumerate(samples):
            self.logger.info(f"Processing sample {i + 1}/{len(samples)}")
            prediction = await self.run_prediction(sample, model_name)
            predictions.append(prediction)

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate metrics
        self.logger.info("Calculating evaluation metrics...")
        metrics = self.calculate_metrics(predictions)

        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            total_samples=len(samples),
            correct_predictions=int(
                metrics.get("true_positives", 0) + metrics.get("true_negatives", 0)
            ),
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1_score=metrics.get("f1_score", 0.0),
            processing_time=processing_time,
            samples_per_second=len(samples) / processing_time if processing_time > 0 else 0.0,
            predictions=predictions,
            evaluation_details=metrics,
        )

        self.logger.info("Benchmark completed successfully!")
        return result

    def generate_report(self, result: BenchmarkResult) -> None:
        """Generate comprehensive benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate detailed JSON report
        json_report = f"lm_studio_benchmark_{timestamp}.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": result.model_name,
            "base_url": self.base_url,
            "summary": {
                "total_samples": result.total_samples,
                "correct_predictions": result.correct_predictions,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "processing_time": result.processing_time,
                "samples_per_second": result.samples_per_second,
            },
            "detailed_metrics": result.evaluation_details,
            "predictions": result.predictions,
        }

        with open(json_report, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate human-readable summary
        summary_report = f"lm_studio_summary_{timestamp}.txt"
        with open(summary_report, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("LM Studio Cybersecurity Benchmark Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {result.model_name}\n")
            f.write(f"LM Studio URL: {self.base_url}\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Samples: {result.total_samples}\n")
            f.write(f"Processing Time: {result.processing_time:.2f} seconds\n")
            f.write(f"Speed: {result.samples_per_second:.2f} samples/second\n\n")

            f.write("ACCURACY METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Accuracy: {result.accuracy:.1%}\n")
            f.write(f"Precision: {result.precision:.1%}\n")
            f.write(f"Recall: {result.recall:.1%}\n")
            f.write(f"F1-Score: {result.f1_score:.1%}\n\n")

            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 30 + "\n")
            tp = result.evaluation_details.get("true_positives", 0)
            fp = result.evaluation_details.get("false_positives", 0)
            fn = result.evaluation_details.get("false_negatives", 0)
            tn = result.evaluation_details.get("true_negatives", 0)
            f.write(f"True Positives: {tp}\n")
            f.write(f"False Positives: {fp}\n")
            f.write(f"False Negatives: {fn}\n")
            f.write(f"True Negatives: {tn}\n\n")

            # Sample predictions
            f.write("SAMPLE PREDICTIONS:\n")
            f.write("-" * 30 + "\n")
            for i, pred in enumerate(result.predictions[:5]):
                f.write(f"Sample {i + 1}:\n")
                f.write(f"  Input: {pred['input_text'][:80]}...\n")
                f.write(f"  Expected: {pred['ground_truth']}\n")
                f.write(f"  Predicted: {pred['prediction']}\n")
                f.write(f"  Confidence: {pred.get('confidence', 0):.2f}\n")
                f.write(f"  Explanation: {pred.get('explanation', '')[:100]}...\n\n")

        # Print summary to console
        print("\n" + "=" * 60)
        print("🎉 LM STUDIO BENCHMARK COMPLETED!")
        print("=" * 60)
        print(f"📊 Model: {result.model_name}")
        print(f"📁 Detailed Report: {json_report}")
        print(f"📄 Summary Report: {summary_report}")
        print("-" * 60)
        print("📈 RESULTS SUMMARY:")
        print(f"   🎯 Accuracy: {result.accuracy:.1%}")
        print(f"   📏 Precision: {result.precision:.1%}")
        print(f"   🔍 Recall: {result.recall:.1%}")
        print(f"   ⚖️  F1-Score: {result.f1_score:.1%}")
        print(f"   ⚡ Speed: {result.samples_per_second:.1f} samples/sec")
        print(f"   📊 Total Samples: {result.total_samples}")
        print("=" * 60)


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="LM Studio Cybersecurity Benchmark")
    parser.add_argument("--model-name", type=str, required=True, help="Model name in LM Studio")
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:1234/v1", help="LM Studio base URL"
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = LMStudioBenchmark(base_url=args.base_url)

    try:
        result = await benchmark.run_benchmark(args.model_name)
        benchmark.generate_report(result)

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logging.error(f"Benchmark failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

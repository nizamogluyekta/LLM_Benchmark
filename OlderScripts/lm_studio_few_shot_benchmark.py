#!/usr/bin/env python3
"""
LM Studio Few-Shot Learning Cybersecurity Benchmark

This script implements proper ML evaluation methodology:
1. Generates 200 balanced cybersecurity samples
2. Splits into 150 training + 50 testing samples
3. Uses few-shot learning (training examples in prompt)
4. Evaluates on unseen test data
5. Provides comprehensive performance analysis

This approach gives a more realistic assessment of model performance.
"""

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import openai


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
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


class DatasetGenerator:
    """Generates balanced cybersecurity datasets for training and testing."""

    def __init__(self) -> None:
        self.attack_templates = {
            "sql_injection": [
                "IP {ip} attempted SQL injection: ' OR '1'='1' -- on {endpoint}",
                "Database query detected: SELECT * FROM users WHERE username='{user}' AND password='' OR '1'='1'--",
                "SQL injection in parameter: /search.php?q=' UNION SELECT password FROM admin_users--",
                "Malicious SQL detected: '; DROP TABLE {table}; --",
                "SQLi attempt: /login.php POST data contains ' OR 1=1# in password field",
            ],
            "brute_force": [
                "Multiple failed login attempts from {ip}: {attempts} attempts in {timeframe}",
                "SSH brute force detected: {count} failed attempts from {ip} targeting {user} account",
                "FTP brute force: {ip} attempted {attempts} different passwords for user {user}",
                "Failed authentication spike: {count} login failures from {ip} across multiple services",
                "Credential stuffing detected: {ip} testing {count} username/password combinations",
            ],
            "malware": [
                "Suspicious file hash detected: {hash} - matches known {malware_type} signature",
                "Antivirus alert: {file} contains {malware_type} attempting to contact {domain}",
                "Malicious PowerShell execution: attempting to download payload from {domain}",
                "Trojan detected: {file} attempting to establish backdoor connection to {ip}",
                "Ransomware signature found: {file} encrypting files with extension .{ext}",
            ],
            "ddos": [
                "DDoS attack detected: {rps} requests/second from {source_count} IPs targeting {target}",
                "Network anomaly: Traffic spike {multiplier}x normal baseline from {ip_count} sources",
                "SYN flood detected: {packets} SYN packets from {ip} targeting port {port}",
                "UDP flood: {ip} sending {pps} packets/second to overwhelm {target}",
                "HTTP flood attack: {rps} GET requests from botnet targeting {endpoint}",
            ],
            "phishing": [
                "Phishing email detected: fake {service} login page at {url}",
                "Suspicious email: '{subject}' with link to {domain}",
                "Spear phishing attempt: targeted email to {user} from spoofed {sender}",
                "Credential harvesting: fake {bank} security alert redirecting to {url}",
                "Business email compromise: spoofed CEO email requesting urgent wire transfer",
            ],
            "network_scan": [
                "Port scan detected from {ip}: scanning ports {ports} on target {target}",
                "Nmap scan identified: {scan_type} scan targeting subnet {subnet}",
                "Vulnerability scan: {ip} probing {service} service for known exploits",
                "Service enumeration: {ip} fingerprinting {services} on {target}",
                "Network reconnaissance: {ip} performing {scan_type} across {range}",
            ],
            "data_exfiltration": [
                "Data exfiltration attempt: {size} uploaded to external server {server}",
                "Unusual data transfer: {amount} sensitive data copied to {destination}",
                "Database dump detected: {records} customer records exported to {file}",
                "Suspicious file access: {user} accessing {count} files outside normal pattern",
                "Large file transfer: {size} confidential documents sent to {email}",
            ],
            "privilege_escalation": [
                "Privilege escalation attempt: {user} trying to access admin-only {resource}",
                "Sudo abuse detected: {user} attempting unauthorized root commands",
                "Service account compromise: {account} performing unusual privileged operations",
                "Windows privilege escalation: exploiting {vulnerability} to gain SYSTEM access",
                "Container escape attempt: process trying to access host filesystem",
            ],
            "lateral_movement": [
                "Lateral movement detected: {user} accessing {count} systems in {timeframe}",
                "SMB enumeration: {ip} scanning network shares across multiple hosts",
                "RDP session anomaly: {user} connecting from {ip} to multiple internal servers",
                "Kerberos ticket abuse: {user} requesting unusual service tickets",
                "WMI remote execution: {ip} executing commands on {target_count} remote hosts",
            ],
            "persistence": [
                "Suspicious service creation: {service} installed with network capabilities",
                "Registry modification: startup entry added for {file} with suspicious permissions",
                "Scheduled task anomaly: {task} created to run {executable} every {interval}",
                "DLL hijacking attempt: {dll} loaded from unusual location by {process}",
                "Boot sector modification: MBR changes detected indicating rootkit installation",
            ],
        }

        self.benign_templates = [
            "User {user} successfully logged in from {ip} at {time}",
            "Scheduled backup completed: {size} backed up to {destination}",
            "Software update installed: {patch} applied to {systems}",
            "Normal web traffic: {method} {endpoint} from legitimate user browser",
            "Database maintenance: {operation} completed on {table}",
            "Email sent successfully: {subject} from {sender} to {recipients}",
            "System health check: All services running normally, CPU {cpu}%, Memory {memory}%",
            "File transfer completed: {file} uploaded to {destination}",
            "Certificate renewal: SSL certificate for {domain} updated successfully",
            "Firewall rule update: Policy {policy} applied to {interfaces}",
            "VPN connection established: {user} connected from {location}",
            "API request processed: {endpoint} returned {status} for client {client}",
            "Cache cleanup completed: {size} temporary files removed",
            "Load balancer health check: All {count} backend servers responding",
            "DNS resolution successful: {domain} resolved to {ip}",
            "Log rotation completed: {logs} archived to {storage}",
            "User password changed: {user} updated credentials via self-service portal",
            "Network scan completed: Infrastructure health check on {subnet}",
            "Database query executed: SELECT operation on {table} by {application}",
            "File sync completed: {count} files synchronized to backup location",
        ]

    def _generate_realistic_values(self) -> dict[str, str]:
        """Generate realistic values for template substitution."""
        ips = ["192.168.1.100", "10.0.0.25", "203.0.113.42", "198.51.100.15", "172.16.0.50"]
        users = ["admin", "root", "john.doe", "alice.smith", "service_account", "guest"]
        domains = ["malicious-site.com", "fake-bank.evil", "phishing-domain.net", "suspicious.org"]
        files = ["document.pdf", "update.exe", "script.ps1", "data.xlsx", "backup.zip"]
        endpoints = ["/login.php", "/admin/panel", "/api/users", "/search", "/upload"]

        return {
            "ip": random.choice(ips),
            "user": random.choice(users),
            "domain": random.choice(domains),
            "file": random.choice(files),
            "endpoint": random.choice(endpoints),
            "hash": "".join(random.choices("0123456789abcdef", k=32)),
            "size": f"{random.randint(1, 500)}MB",
            "count": str(random.randint(10, 1000)),
            "attempts": str(random.randint(50, 500)),
            "timeframe": f"{random.randint(5, 60)} minutes",
            "rps": f"{random.randint(1000, 50000)}",
            "source_count": f"{random.randint(10, 500)}",
            "target": "192.168.1.50",
            "ports": "22, 80, 443, 3389",
            "time": f"{random.randint(1, 12):02d}:{random.randint(0, 59):02d} {'AM' if random.random() > 0.5 else 'PM'}",
            "cpu": str(random.randint(15, 85)),
            "memory": str(random.randint(40, 90)),
            "table": random.choice(["users", "orders", "products", "logs"]),
            "operation": random.choice(["index rebuild", "optimization", "cleanup"]),
            "patch": f"KB{random.randint(1000000, 9999999)}",
            "systems": f"{random.randint(5, 50)} servers",
            "method": random.choice(["GET", "POST", "PUT"]),
            "status": random.choice(["200", "201", "204"]),
            "client": f"client_{random.randint(1000, 9999)}",
            "subject": random.choice(["Weekly Report", "Security Alert", "System Notification"]),
            "sender": random.choice(
                ["security@company.com", "admin@organization.org", "noreply@service.com"]
            ),
            "recipients": "management team",
            "destination": random.choice(["secure storage", "backup server", "cloud storage"]),
            "malware_type": random.choice(["Trojan", "Ransomware", "Backdoor", "Keylogger"]),
            "service": random.choice(["PayPal", "Amazon", "Microsoft", "Google"]),
            "bank": random.choice(["Chase", "Bank of America", "Wells Fargo"]),
            "url": f"http://{random.choice(domains)}/login.html",
            "subnet": "192.168.1.0/24",
            "scan_type": random.choice(["TCP SYN", "UDP", "stealth"]),
            "services": random.choice(["HTTP, SSH, FTP", "SMTP, DNS", "RDP, SMB"]),
            "range": "10.0.0.0/24",
            "server": f"ftp.{random.choice(domains)}",
            "amount": f"{random.randint(100, 1000)}MB",
            "records": f"{random.randint(1000, 100000):,}",
            "email": f"external@{random.choice(domains)}",
            "logs": f"{random.randint(10, 100)} log files",
            "storage": "archive server",
            "policy": f"POLICY_{random.randint(100, 999)}",
            "interfaces": "eth0, eth1",
            "location": random.choice(["New York", "London", "Tokyo", "Sydney"]),
            "application": random.choice(["webapp", "api_service", "scheduler"]),
            "multiplier": f"{random.randint(10, 100)}",
            "ip_count": f"{random.randint(50, 500)}",
            "packets": f"{random.randint(10000, 100000):,}",
            "port": str(random.choice([80, 443, 22, 3389, 25])),
            "pps": f"{random.randint(1000, 10000):,}",
            "vulnerability": random.choice(["CVE-2023-1234", "CVE-2023-5678", "privilege bug"]),
            "resource": random.choice(["configuration file", "admin panel", "database"]),
            "account": random.choice(["service_user", "backup_account", "monitoring_svc"]),
            "web_service": random.choice(["WebService", "DataProcessor", "SecurityMonitor"]),
            "task": f"Task_{random.randint(1000, 9999)}",
            "executable": random.choice(["script.exe", "updater.exe", "service.exe"]),
            "interval": random.choice(["5 minutes", "1 hour", "daily"]),
            "dll": random.choice(["system32.dll", "kernel.dll", "security.dll"]),
            "process": random.choice(["explorer.exe", "svchost.exe", "winlogon.exe"]),
            "ext": random.choice(["locked", "encrypted", "secure"]),
            "target_count": str(random.randint(5, 20)),
        }

    def generate_attack_sample(self, attack_type: str) -> dict[str, str]:
        """Generate a realistic attack sample."""
        template = random.choice(self.attack_templates[attack_type])
        values = self._generate_realistic_values()

        try:
            text = template.format(**values)
        except KeyError:
            # Fallback if template has missing keys
            text = template

        return {"text": text, "label": "ATTACK", "attack_type": attack_type}

    def generate_benign_sample(self) -> dict[str, str]:
        """Generate a realistic benign sample."""
        template = random.choice(self.benign_templates)
        values = self._generate_realistic_values()

        try:
            text = template.format(**values)
        except KeyError:
            # Fallback if template has missing keys
            text = template

        return {"text": text, "label": "BENIGN", "attack_type": "none"}

    def generate_balanced_dataset(self, total_samples: int = 200) -> list[dict[str, str]]:
        """Generate a balanced dataset with equal ATTACK/BENIGN samples."""
        samples = []
        attack_types = list(self.attack_templates.keys())

        # Generate attack samples (50% of total)
        attack_samples = total_samples // 2
        samples_per_type = attack_samples // len(attack_types)
        remainder = attack_samples % len(attack_types)

        for i, attack_type in enumerate(attack_types):
            count = samples_per_type + (1 if i < remainder else 0)
            for _ in range(count):
                samples.append(self.generate_attack_sample(attack_type))

        # Generate benign samples (50% of total)
        benign_samples = total_samples - len(samples)
        for _ in range(benign_samples):
            samples.append(self.generate_benign_sample())

        # Shuffle the dataset
        random.shuffle(samples)
        return samples


class LMStudioFewShotBenchmark:
    """LM Studio benchmark with proper train/test split and few-shot learning."""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        """Initialize the benchmark."""
        self.base_url = base_url
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        self.logger = self._setup_logging()
        self.dataset_generator: DatasetGenerator = DatasetGenerator()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger("few_shot_benchmark")

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
Attack_Type: [sql_injection, brute_force, malware, ddos, phishing, network_scan, data_exfiltration, privilege_escalation, lateral_movement, persistence, or none if benign]
Explanation: [Brief explanation of your reasoning based on the training examples]

Analysis:"""

        return prompt, examples

    async def run_zero_shot_prediction(self, sample: str, model_name: str) -> dict[str, Any]:
        """Run zero-shot prediction (no training examples)."""
        prompt = f"""You are a cybersecurity expert analyzing network logs. Classify this event as ATTACK or BENIGN.

Event: {sample}

Classification: [ATTACK or BENIGN]
Confidence: [0.0 to 1.0]
Explanation: [Brief reasoning]

Analysis:"""

        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.1,
            )

            content = response.choices[0].message.content
            return self._parse_response(content or "")

        except Exception as e:
            return {"classification": "ERROR", "confidence": 0.0, "explanation": str(e)}

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
            parsed["model_response"] = response.choices[0].message.content
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
            "attack_type": None,
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

            # Extract attack type
            attack_type_match = re.search(
                r"Attack_Type:\s*(sql_injection|brute_force|malware|ddos|phishing|network_scan|data_exfiltration|privilege_escalation|lateral_movement|persistence|none)",
                response,
                re.IGNORECASE,
            )
            if attack_type_match and attack_type_match.group(1).lower() != "none":
                parsed["attack_type"] = attack_type_match.group(1).lower()

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
            return {}

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
        self, model_name: str, total_samples: int = 200, few_shot_examples: int = 10
    ) -> BenchmarkResult:
        """Run complete few-shot learning benchmark."""

        self.logger.info(f"Starting few-shot benchmark for model: {model_name}")
        self.logger.info(f"Total samples: {total_samples}, Few-shot examples: {few_shot_examples}")

        # Generate balanced dataset
        self.logger.info("Generating balanced cybersecurity dataset...")
        dataset = self.dataset_generator.generate_balanced_dataset(total_samples)

        attack_count = sum(1 for s in dataset if s["label"] == "ATTACK")
        benign_count = sum(1 for s in dataset if s["label"] == "BENIGN")
        self.logger.info(
            f"Generated {len(dataset)} samples: {attack_count} ATTACK, {benign_count} BENIGN"
        )

        # Split into train/test
        train_set, test_set = self.split_dataset(dataset, train_ratio=0.75)
        self.logger.info(f"Split: {len(train_set)} training, {len(test_set)} testing samples")

        # Run zero-shot baseline on a subset
        self.logger.info("Running zero-shot baseline...")
        zero_shot_subset = test_set[:10]  # Small subset for comparison
        zero_shot_predictions = []

        for sample in zero_shot_subset:
            pred = await self.run_zero_shot_prediction(sample["text"], model_name)
            zero_shot_predictions.append(pred)

        zero_shot_metrics = self.calculate_metrics(
            zero_shot_predictions, [s["label"] for s in zero_shot_subset]
        )

        # Run few-shot predictions on full test set
        self.logger.info("Running few-shot predictions...")
        start_time = time.time()

        few_shot_predictions = []
        all_training_examples = []

        for i, sample in enumerate(test_set):
            self.logger.info(f"Processing test sample {i + 1}/{len(test_set)}")

            pred, training_examples = await self.run_few_shot_prediction(
                sample["text"], train_set, model_name, few_shot_examples
            )

            pred["sample_id"] = f"test_{i}"
            pred["ground_truth"] = sample["label"]
            pred["ground_truth_attack_type"] = sample.get("attack_type", "none")
            pred["input_text"] = sample["text"]

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
            training_samples=len(train_set),
            test_samples=len(test_set),
            few_shot_examples=few_shot_examples,
            zero_shot_accuracy=zero_shot_metrics.get("accuracy", 0.0),
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

        self.logger.info("Few-shot benchmark completed successfully!")
        return result

    def generate_report(self, result: BenchmarkResult) -> None:
        """Generate comprehensive benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate detailed JSON report
        json_report = f"few_shot_benchmark_{timestamp}.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": result.model_name,
            "methodology": "few_shot_learning",
            "dataset_info": {
                "total_samples": result.training_samples + result.test_samples,
                "training_samples": result.training_samples,
                "test_samples": result.test_samples,
                "few_shot_examples": result.few_shot_examples,
            },
            "performance_comparison": {
                "zero_shot_accuracy": result.zero_shot_accuracy,
                "few_shot_accuracy": result.few_shot_accuracy,
                "improvement": result.few_shot_accuracy - result.zero_shot_accuracy,
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
        summary_report = f"few_shot_summary_{timestamp}.txt"
        with open(summary_report, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("LM Studio Few-Shot Learning Cybersecurity Benchmark Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {result.model_name}\n")
            f.write(f"Methodology: Few-Shot Learning with {result.few_shot_examples} examples\n\n")

            f.write("DATASET INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Samples: {result.training_samples + result.test_samples}\n")
            f.write(f"Training Samples: {result.training_samples}\n")
            f.write(f"Test Samples: {result.test_samples}\n")
            f.write(f"Examples per prediction: {result.few_shot_examples}\n\n")

            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Zero-Shot Accuracy: {result.zero_shot_accuracy:.1%}\n")
            f.write(f"Few-Shot Accuracy: {result.few_shot_accuracy:.1%}\n")
            improvement = result.few_shot_accuracy - result.zero_shot_accuracy
            f.write(f"Improvement: {improvement:+.1%}\n\n")

            f.write("DETAILED METRICS:\n")
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

            # Training examples used
            f.write("TRAINING EXAMPLES USED:\n")
            f.write("-" * 30 + "\n")
            for i, example in enumerate(result.training_examples_used, 1):
                f.write(f"{i}. [{example['label']}] {example['text'][:80]}...\n")

        # Print summary to console
        print("\n" + "=" * 70)
        print("🎉 FEW-SHOT LEARNING BENCHMARK COMPLETED!")
        print("=" * 70)
        print(f"📊 Model: {result.model_name}")
        print(f"📁 Detailed Report: {json_report}")
        print(f"📄 Summary Report: {summary_report}")
        print("-" * 70)
        print("📈 RESULTS COMPARISON:")
        print(f"   🔹 Zero-Shot: {result.zero_shot_accuracy:.1%}")
        print(f"   🔸 Few-Shot:  {result.few_shot_accuracy:.1%}")
        improvement = result.few_shot_accuracy - result.zero_shot_accuracy
        print(f"   🚀 Improvement: {improvement:+.1%}")
        print("-" * 70)
        print("📊 FINAL METRICS:")
        print(f"   🎯 Accuracy: {result.few_shot_accuracy:.1%}")
        print(f"   📏 Precision: {result.precision:.1%}")
        print(f"   🔍 Recall: {result.recall:.1%}")
        print(f"   ⚖️  F1-Score: {result.f1_score:.1%}")
        print(f"   ⚡ Speed: {result.samples_per_second:.1f} samples/sec")
        print("=" * 70)


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="LM Studio Few-Shot Learning Cybersecurity Benchmark"
    )
    parser.add_argument("--model-name", type=str, required=True, help="Model name in LM Studio")
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:1234/v1", help="LM Studio base URL"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=200,
        help="Total number of samples to generate (default: 200)",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=10,
        help="Number of training examples to use per prediction (default: 10)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)

    # Run benchmark
    benchmark = LMStudioFewShotBenchmark(base_url=args.base_url)

    try:
        result = await benchmark.run_benchmark(
            args.model_name, args.total_samples, args.few_shot_examples
        )
        benchmark.generate_report(result)

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logging.error(f"Benchmark failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

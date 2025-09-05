"""
Cybersecurity data generators for creating realistic test data.

This module provides utilities to generate realistic cybersecurity samples,
predictions, and evaluation data for comprehensive testing.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any


class CybersecurityDataGenerator:
    """
    Generator for realistic cybersecurity test data.

    Provides methods to generate network logs, email samples, model predictions,
    explanations, and performance data for different attack types.
    """

    # Attack type constants
    ATTACK_TYPES = {
        "malware": ["trojan", "virus", "worm", "ransomware", "backdoor", "rootkit"],
        "intrusion": ["unauthorized_access", "privilege_escalation", "lateral_movement"],
        "dos": ["flooding", "amplification", "resource_exhaustion", "slowloris"],
        "phishing": ["spear_phishing", "whaling", "smishing", "vishing"],
        "injection": ["sql_injection", "xss", "command_injection", "ldap_injection"],
        "reconnaissance": ["port_scan", "network_scan", "vulnerability_scan", "enumeration"],
    }

    # Common ports and services
    COMMON_PORTS = {
        22: "SSH",
        23: "Telnet",
        25: "SMTP",
        53: "DNS",
        80: "HTTP",
        110: "POP3",
        143: "IMAP",
        443: "HTTPS",
        993: "IMAPS",
        995: "POP3S",
        21: "FTP",
        3389: "RDP",
        1433: "MSSQL",
        3306: "MySQL",
        5432: "PostgreSQL",
    }

    # User agents for web traffic
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]

    def __init__(self, seed: int | None = None):
        """
        Initialize the data generator.

        Args:
            seed: Optional random seed for reproducible data generation
        """
        self._random = random.Random(seed)

    def generate_ip_address(self, private: bool = True) -> str:
        """Generate a random IP address."""
        if private:
            # Generate private IP ranges
            ranges = [
                (10, 0, 0, 0, 8),  # 10.0.0.0/8
                (172, 16, 0, 0, 12),  # 172.16.0.0/12
                (192, 168, 0, 0, 16),  # 192.168.0.0/16
            ]
            range_choice = self._random.choice(ranges)

            if range_choice[4] == 8:  # 10.0.0.0/8
                return f"10.{self._random.randint(0, 255)}.{self._random.randint(0, 255)}.{self._random.randint(1, 254)}"
            elif range_choice[4] == 12:  # 172.16.0.0/12
                return f"172.{self._random.randint(16, 31)}.{self._random.randint(0, 255)}.{self._random.randint(1, 254)}"
            else:  # 192.168.0.0/16
                return f"192.168.{self._random.randint(0, 255)}.{self._random.randint(1, 254)}"
        else:
            # Generate public IP (avoiding private ranges)
            while True:
                ip = f"{self._random.randint(1, 223)}.{self._random.randint(0, 255)}.{self._random.randint(0, 255)}.{self._random.randint(1, 254)}"
                # Simple check to avoid private ranges
                if not (
                    ip.startswith("10.")
                    or ip.startswith("192.168.")
                    or any(ip.startswith(f"172.{i}.") for i in range(16, 32))
                ):
                    return ip

    def generate_timestamp(self, days_back: int = 30) -> str:
        """Generate a random timestamp within the last N days."""
        now = datetime.now()
        start_time = now - timedelta(days=days_back)
        random_time = start_time + timedelta(
            seconds=self._random.randint(0, int((now - start_time).total_seconds()))
        )
        return random_time.strftime("%Y-%m-%d %H:%M:%S")

    def generate_network_log(
        self, is_attack: bool = False, attack_type: str | None = None
    ) -> dict[str, Any]:
        """
        Generate realistic network log entries.

        Args:
            is_attack: Whether to generate an attack log
            attack_type: Specific attack type (malware, intrusion, dos, reconnaissance)

        Returns:
            Dictionary containing network log data
        """
        src_ip = self.generate_ip_address(private=False)
        dst_ip = self.generate_ip_address(private=True)
        timestamp = self.generate_timestamp()

        if is_attack:
            if not attack_type:
                attack_type = self._random.choice(list(self.ATTACK_TYPES.keys()))

            if attack_type == "malware":
                return self._generate_malware_log(src_ip, dst_ip, timestamp, attack_type)
            elif attack_type == "intrusion":
                return self._generate_intrusion_log(src_ip, dst_ip, timestamp, attack_type)
            elif attack_type == "dos":
                return self._generate_dos_log(src_ip, dst_ip, timestamp, attack_type)
            elif attack_type == "reconnaissance":
                return self._generate_recon_log(src_ip, dst_ip, timestamp, attack_type)
            else:
                return self._generate_generic_attack_log(src_ip, dst_ip, timestamp, attack_type)
        else:
            return self._generate_benign_network_log(src_ip, dst_ip, timestamp)

    def _generate_malware_log(
        self, src_ip: str, dst_ip: str, timestamp: str, attack_type: str
    ) -> dict[str, Any]:
        """Generate malware-related network log."""
        malware_types = self.ATTACK_TYPES["malware"]
        malware_subtype = self._random.choice(malware_types)

        suspicious_domains = [
            "malicious-cdn.com",
            "evil-payload.net",
            "c2-server.org",
            "botnet-command.io",
            "crypto-miner.xyz",
        ]

        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": self._random.randint(32768, 65535),
            "dst_port": self._random.choice([80, 443, 8080, 53]),
            "protocol": "TCP",
            "bytes_sent": self._random.randint(1024, 10240),
            "bytes_received": self._random.randint(512, 5120),
            "text": f"Suspicious {malware_subtype} communication detected from {src_ip} to {self._random.choice(suspicious_domains)}",
            "label": "ATTACK",
            "attack_type": "malware",
            "attack_subtype": malware_subtype,
            "severity": self._random.choice(["HIGH", "CRITICAL"]),
            "confidence": self._random.uniform(0.8, 0.98),
            "additional_data": {
                "suspicious_domain": self._random.choice(suspicious_domains),
                "file_hash": self._generate_file_hash(),
                "user_agent": self._random.choice(self.USER_AGENTS),
            },
        }

    def _generate_intrusion_log(
        self, src_ip: str, dst_ip: str, timestamp: str, attack_type: str
    ) -> dict[str, Any]:
        """Generate intrusion-related network log."""
        intrusion_types = self.ATTACK_TYPES["intrusion"]
        intrusion_subtype = self._random.choice(intrusion_types)

        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": self._random.randint(32768, 65535),
            "dst_port": self._random.choice([22, 23, 3389, 5985]),
            "protocol": "TCP",
            "bytes_sent": self._random.randint(256, 2048),
            "bytes_received": self._random.randint(128, 1024),
            "text": f"{intrusion_subtype} attempt detected from {src_ip} targeting {dst_ip}",
            "label": "ATTACK",
            "attack_type": "intrusion",
            "attack_subtype": intrusion_subtype,
            "severity": "HIGH",
            "confidence": self._random.uniform(0.75, 0.95),
            "additional_data": {
                "failed_login_attempts": self._random.randint(5, 50),
                "target_service": self._random.choice(["SSH", "RDP", "WinRM", "Telnet"]),
                "username_attempts": self._random.sample(
                    ["admin", "root", "administrator", "user", "guest"], self._random.randint(2, 4)
                ),
            },
        }

    def _generate_dos_log(
        self, src_ip: str, dst_ip: str, timestamp: str, attack_type: str
    ) -> dict[str, Any]:
        """Generate DoS attack network log."""
        dos_types = self.ATTACK_TYPES["dos"]
        dos_subtype = self._random.choice(dos_types)

        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": self._random.randint(1024, 65535),
            "dst_port": self._random.choice([80, 443, 53, 25]),
            "protocol": self._random.choice(["TCP", "UDP", "ICMP"]),
            "bytes_sent": self._random.randint(10240, 102400),
            "bytes_received": self._random.randint(0, 1024),
            "text": f"{dos_subtype} attack detected - high volume traffic from {src_ip}",
            "label": "ATTACK",
            "attack_type": "dos",
            "attack_subtype": dos_subtype,
            "severity": "HIGH",
            "confidence": self._random.uniform(0.85, 0.98),
            "additional_data": {
                "request_rate": self._random.randint(1000, 10000),
                "packet_size": self._random.randint(1, 65535),
                "duration_seconds": self._random.randint(60, 3600),
                "amplification_factor": self._random.uniform(1.0, 50.0)
                if dos_subtype == "amplification"
                else 1.0,
            },
        }

    def _generate_recon_log(
        self, src_ip: str, dst_ip: str, timestamp: str, attack_type: str
    ) -> dict[str, Any]:
        """Generate reconnaissance network log."""
        recon_types = self.ATTACK_TYPES["reconnaissance"]
        recon_subtype = self._random.choice(recon_types)

        scanned_ports = self._random.sample(
            list(self.COMMON_PORTS.keys()), self._random.randint(3, 8)
        )

        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": self._random.randint(32768, 65535),
            "dst_port": scanned_ports[0],
            "protocol": "TCP",
            "bytes_sent": self._random.randint(64, 512),
            "bytes_received": self._random.randint(0, 256),
            "text": f"{recon_subtype} detected from {src_ip} targeting {len(scanned_ports)} ports on {dst_ip}",
            "label": "ATTACK",
            "attack_type": "reconnaissance",
            "attack_subtype": recon_subtype,
            "severity": "MEDIUM",
            "confidence": self._random.uniform(0.7, 0.92),
            "additional_data": {
                "scanned_ports": scanned_ports,
                "scan_type": self._random.choice(["TCP SYN", "TCP Connect", "UDP", "Stealth"]),
                "scan_duration": self._random.randint(1, 300),
                "response_analysis": {
                    "open_ports": self._random.sample(
                        scanned_ports, self._random.randint(0, len(scanned_ports) // 2)
                    ),
                    "closed_ports": self._random.randint(
                        len(scanned_ports) // 2, len(scanned_ports)
                    ),
                },
            },
        }

    def _generate_generic_attack_log(
        self, src_ip: str, dst_ip: str, timestamp: str, attack_type: str
    ) -> dict[str, Any]:
        """Generate generic attack network log."""
        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": self._random.randint(1024, 65535),
            "dst_port": self._random.choice(list(self.COMMON_PORTS.keys())),
            "protocol": self._random.choice(["TCP", "UDP"]),
            "bytes_sent": self._random.randint(256, 4096),
            "bytes_received": self._random.randint(128, 2048),
            "text": f"Suspicious {attack_type} activity detected from {src_ip}",
            "label": "ATTACK",
            "attack_type": attack_type,
            "severity": self._random.choice(["MEDIUM", "HIGH"]),
            "confidence": self._random.uniform(0.6, 0.9),
        }

    def _generate_benign_network_log(
        self, src_ip: str, dst_ip: str, timestamp: str
    ) -> dict[str, Any]:
        """Generate benign network log."""
        benign_activities = [
            "Normal HTTP request to web server",
            "Successful database connection",
            "Regular email synchronization",
            "System backup data transfer",
            "Software update download",
            "User file upload",
            "API authentication request",
            "Health check ping",
        ]

        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": self._random.randint(32768, 65535),
            "dst_port": self._random.choice([80, 443, 25, 110, 143, 22]),
            "protocol": "TCP",
            "bytes_sent": self._random.randint(512, 2048),
            "bytes_received": self._random.randint(256, 4096),
            "text": self._random.choice(benign_activities),
            "label": "BENIGN",
            "attack_type": None,
            "severity": "LOW",
            "confidence": self._random.uniform(0.9, 0.99),
        }

    def generate_email_sample(
        self, is_phishing: bool = False, phishing_type: str | None = None
    ) -> dict[str, Any]:
        """
        Generate realistic email samples for phishing detection.

        Args:
            is_phishing: Whether to generate a phishing email
            phishing_type: Specific phishing type (spear_phishing, whaling, etc.)

        Returns:
            Dictionary containing email data
        """
        if is_phishing:
            if not phishing_type:
                phishing_type = self._random.choice(self.ATTACK_TYPES["phishing"])
            return self._generate_phishing_email(phishing_type)
        else:
            return self._generate_benign_email()

    def _generate_phishing_email(self, phishing_type: str) -> dict[str, Any]:
        """Generate phishing email sample."""
        suspicious_domains = [
            "secure-bank-update.com",
            "paypal-security.net",
            "microsoft-support.org",
            "amazon-verification.info",
            "google-security.biz",
        ]

        phishing_subjects = {
            "spear_phishing": [
                "Urgent: Your account will be suspended",
                "Security Alert: Unusual activity detected",
                "Action Required: Verify your identity",
            ],
            "whaling": [
                "CEO Approval Required: Urgent Wire Transfer",
                "Board Meeting: Confidential Financial Documents",
                "Executive Compensation Review - Immediate Response",
            ],
            "smishing": [
                "Your package delivery failed - reschedule now",
                "Bank Alert: Suspicious activity on your account",
                "Win $1000 - Click here to claim your prize",
            ],
        }

        sender_domain = self._random.choice(suspicious_domains)
        subject = self._random.choice(
            phishing_subjects.get(phishing_type, phishing_subjects["spear_phishing"])
        )

        return {
            "message_id": f"<{uuid.uuid4()}@{sender_domain}>",
            "timestamp": self.generate_timestamp(7),
            "sender": f"security@{sender_domain}",
            "recipient": "user@company.com",
            "subject": subject,
            "body": self._generate_phishing_body(phishing_type),
            "text": f"Email from {sender_domain}: {subject}",
            "label": "ATTACK",
            "attack_type": "phishing",
            "attack_subtype": phishing_type,
            "confidence": self._random.uniform(0.75, 0.95),
            "additional_data": {
                "sender_domain": sender_domain,
                "suspicious_urls": [
                    f"http://{sender_domain}/verify",
                    f"https://{sender_domain}/login",
                ],
                "attachment_count": self._random.randint(0, 2),
                "urgency_keywords": ["urgent", "immediate", "suspend", "verify", "action required"],
            },
        }

    def _generate_phishing_body(self, phishing_type: str) -> str:
        """Generate phishing email body content."""
        templates = {
            "spear_phishing": [
                "Dear Customer,\n\nWe've detected unusual activity on your account. Please verify your identity immediately to prevent suspension.\n\nClick here to verify: {url}\n\nSecurity Team",
                "Your account security has been compromised. Immediate action required to secure your account.\n\nVerify now: {url}\n\nThank you.",
            ],
            "whaling": [
                "I need you to process an urgent wire transfer of $50,000 to our overseas partner. Please handle this confidentially.\n\nDetails: {url}\n\nRegards,\nCEO"
            ],
        }

        template = self._random.choice(templates.get(phishing_type, templates["spear_phishing"]))
        return template.format(url="https://secure-verify.malicious-site.com/login")

    def _generate_benign_email(self) -> dict[str, Any]:
        """Generate benign email sample."""
        legitimate_domains = ["company.com", "newsletter.service.com", "notification.bank.com"]
        benign_subjects = [
            "Monthly Newsletter - February 2024",
            "Meeting reminder: Team standup tomorrow",
            "Account statement ready for download",
            "Software update notification",
            "Welcome to our service",
        ]

        sender_domain = self._random.choice(legitimate_domains)
        subject = self._random.choice(benign_subjects)

        return {
            "message_id": f"<{uuid.uuid4()}@{sender_domain}>",
            "timestamp": self.generate_timestamp(30),
            "sender": f"noreply@{sender_domain}",
            "recipient": "user@company.com",
            "subject": subject,
            "body": f"This is a legitimate email regarding: {subject}",
            "text": f"Legitimate email from {sender_domain}: {subject}",
            "label": "BENIGN",
            "attack_type": None,
            "confidence": self._random.uniform(0.85, 0.99),
            "additional_data": {
                "sender_domain": sender_domain,
                "legitimate_indicators": ["proper_spf", "dkim_valid", "known_sender"],
                "attachment_count": self._random.randint(0, 1),
            },
        }

    def generate_model_prediction(
        self, ground_truth_label: str, accuracy: float = 0.85
    ) -> dict[str, Any]:
        """
        Generate model prediction with realistic confidence scores.

        Args:
            ground_truth_label: The actual label (ATTACK or BENIGN)
            accuracy: Model accuracy rate (0.0 to 1.0)

        Returns:
            Dictionary containing prediction data
        """
        # Determine if prediction should be correct based on accuracy
        is_correct = self._random.random() < accuracy

        if is_correct:
            prediction = ground_truth_label
            confidence = self._random.uniform(0.7, 0.98)
        else:
            prediction = "BENIGN" if ground_truth_label == "ATTACK" else "ATTACK"
            confidence = self._random.uniform(0.5, 0.8)

        return {
            "sample_id": str(uuid.uuid4()),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "ground_truth": ground_truth_label,
            "is_correct": is_correct,
            "class_probabilities": {
                "ATTACK": confidence if prediction == "ATTACK" else 1.0 - confidence,
                "BENIGN": confidence if prediction == "BENIGN" else 1.0 - confidence,
            },
            "inference_time_ms": round(self._random.uniform(50, 500), 2),
            "model_version": f"v{self._random.randint(1, 5)}.{self._random.randint(0, 9)}.{self._random.randint(0, 9)}",
            "timestamp": self.generate_timestamp(1),
        }

    def generate_explanation(self, prediction: str, attack_type: str | None = None) -> str:
        """
        Generate explanation for model predictions.

        Args:
            prediction: The predicted class (ATTACK or BENIGN)
            attack_type: Type of attack if prediction is ATTACK

        Returns:
            Human-readable explanation string
        """
        if prediction == "ATTACK":
            if attack_type == "malware":
                explanations = [
                    "Detected suspicious file hash matching known malware signatures",
                    "Communication pattern consistent with C&C server interaction",
                    "Unusual network behavior indicating potential payload delivery",
                ]
            elif attack_type == "phishing":
                explanations = [
                    "Email contains multiple urgency keywords and suspicious URLs",
                    "Sender domain does not match claimed organization",
                    "Message structure follows known phishing templates",
                ]
            elif attack_type == "dos":
                explanations = [
                    "High volume of requests from single source IP address",
                    "Request pattern consistent with flooding attack",
                    "Abnormal packet sizes detected in traffic analysis",
                ]
            elif attack_type == "reconnaissance":
                explanations = [
                    "Sequential port scanning detected across multiple targets",
                    "Service enumeration attempts on common ports",
                    "Pattern consistent with network discovery activities",
                ]
            else:
                explanations = [
                    "Multiple security indicators suggest malicious activity",
                    "Behavioral analysis indicates anomalous network patterns",
                    "Risk score exceeds threshold based on feature analysis",
                ]
        else:  # BENIGN
            explanations = [
                "Normal traffic patterns consistent with legitimate user behavior",
                "All security checks passed with no suspicious indicators",
                "Communication follows standard protocols with expected parameters",
                "Sender authentication and reputation checks successful",
                "Traffic volume and timing within normal operational parameters",
            ]

        base_explanation = self._random.choice(explanations)

        # Add technical details
        technical_details = [
            f"Feature importance: network_pattern (0.{self._random.randint(10, 40)}), content_analysis (0.{self._random.randint(15, 35)})",
            f"Risk score: {self._random.uniform(0.1, 0.9):.3f} based on {self._random.randint(15, 45)} features",
            f"Similar patterns detected in {self._random.randint(0, 5)} recent samples",
        ]

        return f"{base_explanation}. {self._random.choice(technical_details)}"

    def generate_performance_data(self, num_samples: int = 100) -> list[dict[str, Any]]:
        """
        Generate performance timing data for model evaluation.

        Args:
            num_samples: Number of performance samples to generate

        Returns:
            List of performance measurement dictionaries
        """
        performance_data = []

        for i in range(num_samples):
            # Simulate different model sizes and complexities
            model_size = self._random.choice(["small", "medium", "large", "xlarge"])

            # Base inference times by model size
            base_times = {
                "small": (20, 80),  # 20-80ms
                "medium": (50, 150),  # 50-150ms
                "large": (100, 300),  # 100-300ms
                "xlarge": (200, 600),  # 200-600ms
            }

            min_time, max_time = base_times[model_size]
            inference_time = self._random.uniform(min_time, max_time)

            # Add occasional outliers (network delays, cold starts, etc.)
            if self._random.random() < 0.05:  # 5% chance of outlier
                inference_time *= self._random.uniform(2.0, 5.0)

            sample = {
                "sample_id": str(i + 1),
                "model_size": model_size,
                "inference_time_ms": round(inference_time, 2),
                "preprocessing_time_ms": round(self._random.uniform(5, 20), 2),
                "postprocessing_time_ms": round(self._random.uniform(2, 10), 2),
                "total_time_ms": round(inference_time + self._random.uniform(7, 30), 2),
                "memory_usage_mb": round(self._random.uniform(50, 500), 1),
                "cpu_usage_percent": round(self._random.uniform(10, 90), 1),
                "gpu_usage_percent": round(self._random.uniform(0, 95), 1)
                if model_size in ["large", "xlarge"]
                else 0.0,
                "batch_size": self._random.choice([1, 4, 8, 16, 32]),
                "timestamp": self.generate_timestamp(7),
            }

            # Calculate throughput
            sample["throughput_samples_per_second"] = round(
                sample["batch_size"] / (sample["total_time_ms"] / 1000), 2
            )

            performance_data.append(sample)

        return performance_data

    def _generate_file_hash(self) -> str:
        """Generate realistic looking file hash."""
        return "".join(self._random.choices("0123456789abcdef", k=64))  # SHA-256 style hash

    def generate_batch_samples(
        self,
        num_samples: int = 100,
        attack_ratio: float = 0.3,
        attack_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a batch of mixed attack and benign samples.

        Args:
            num_samples: Total number of samples to generate
            attack_ratio: Ratio of attack samples (0.0 to 1.0)
            attack_types: List of attack types to include

        Returns:
            List of generated samples
        """
        if attack_types is None:
            attack_types = list(self.ATTACK_TYPES.keys())

        samples = []
        num_attacks = int(num_samples * attack_ratio)

        # Generate attack samples
        for _ in range(num_attacks):
            attack_type = self._random.choice(attack_types)

            # Mix network logs and emails
            if self._random.choice([True, False]):
                if attack_type == "phishing":
                    sample = self.generate_email_sample(
                        is_phishing=True, phishing_type="spear_phishing"
                    )
                else:
                    sample = self.generate_network_log(is_attack=True, attack_type=attack_type)
            else:
                sample = self.generate_network_log(is_attack=True, attack_type=attack_type)

            samples.append(sample)

        # Generate benign samples
        for _ in range(num_samples - num_attacks):
            if self._random.choice([True, False]):
                sample = self.generate_email_sample(is_phishing=False)
            else:
                sample = self.generate_network_log(is_attack=False)

            samples.append(sample)

        # Shuffle the samples
        self._random.shuffle(samples)

        # Add sample IDs
        for i, sample in enumerate(samples):
            sample["sample_id"] = str(i + 1)

        return samples

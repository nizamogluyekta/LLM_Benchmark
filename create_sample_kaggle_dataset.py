#!/usr/bin/env python3
"""
Create Sample Kaggle-style Cybersecurity Dataset

This script creates a sample CSV dataset in the format typical of Kaggle cybersecurity datasets.
Use this to test the Kaggle benchmark functionality before downloading real datasets.
"""

import csv
import random
from datetime import datetime, timedelta


def generate_sample_dataset(
    filename: str = "sample_cybersecurity_dataset.csv", num_samples: int = 300
) -> None:
    """Generate a sample cybersecurity dataset in CSV format."""

    # Attack patterns
    attack_patterns = [
        {
            "description": "SQL injection attempt: ' OR '1'='1' -- detected in login parameter",
            "event_type": "web_attack",
            "severity": "high",
            "source_ip": "192.168.1.100",
            "target_port": "80",
            "classification": "attack",
        },
        {
            "description": "Multiple failed SSH login attempts detected from external IP",
            "event_type": "brute_force",
            "severity": "medium",
            "source_ip": "203.0.113.42",
            "target_port": "22",
            "classification": "malicious",
        },
        {
            "description": "Suspicious PowerShell execution attempting to download payload",
            "event_type": "malware",
            "severity": "high",
            "source_ip": "10.0.0.25",
            "target_port": "443",
            "classification": "threat",
        },
        {
            "description": "DDoS attack detected: abnormal traffic volume from botnet",
            "event_type": "ddos",
            "severity": "critical",
            "source_ip": "198.51.100.15",
            "target_port": "80",
            "classification": "attack",
        },
        {
            "description": "Phishing email with malicious attachment detected",
            "event_type": "phishing",
            "severity": "medium",
            "source_ip": "172.16.0.50",
            "target_port": "25",
            "classification": "malicious",
        },
        {
            "description": "Port scan activity detected from external network",
            "event_type": "reconnaissance",
            "severity": "low",
            "source_ip": "203.0.113.100",
            "target_port": "various",
            "classification": "suspicious",
        },
        {
            "description": "Ransomware signature detected in file upload",
            "event_type": "malware",
            "severity": "critical",
            "source_ip": "192.168.1.200",
            "target_port": "443",
            "classification": "threat",
        },
        {
            "description": "Buffer overflow attempt in web application parameter",
            "event_type": "exploit",
            "severity": "high",
            "source_ip": "10.0.0.100",
            "target_port": "8080",
            "classification": "attack",
        },
    ]

    # Normal patterns
    normal_patterns = [
        {
            "description": "User authentication successful via corporate VPN",
            "event_type": "authentication",
            "severity": "info",
            "source_ip": "192.168.1.55",
            "target_port": "443",
            "classification": "normal",
        },
        {
            "description": "Scheduled database backup completed successfully",
            "event_type": "maintenance",
            "severity": "info",
            "source_ip": "192.168.1.10",
            "target_port": "3306",
            "classification": "benign",
        },
        {
            "description": "Software update installation completed on workstation",
            "event_type": "update",
            "severity": "info",
            "source_ip": "192.168.1.75",
            "target_port": "80",
            "classification": "legitimate",
        },
        {
            "description": "Regular web traffic from corporate network to business site",
            "event_type": "web_browsing",
            "severity": "info",
            "source_ip": "192.168.1.120",
            "target_port": "443",
            "classification": "normal",
        },
        {
            "description": "Email sent successfully through corporate mail server",
            "event_type": "email",
            "severity": "info",
            "source_ip": "192.168.1.5",
            "target_port": "25",
            "classification": "legitimate",
        },
        {
            "description": "File upload to secure document management system",
            "event_type": "file_transfer",
            "severity": "info",
            "source_ip": "192.168.1.85",
            "target_port": "443",
            "classification": "benign",
        },
        {
            "description": "System health monitoring check completed normally",
            "event_type": "monitoring",
            "severity": "info",
            "source_ip": "192.168.1.1",
            "target_port": "161",
            "classification": "normal",
        },
        {
            "description": "Certificate renewal process completed for web server",
            "event_type": "certificate",
            "severity": "info",
            "source_ip": "192.168.1.50",
            "target_port": "443",
            "classification": "legitimate",
        },
    ]

    # Generate dataset
    samples = []
    start_time = datetime.now() - timedelta(days=30)

    # Generate samples (balanced between attack and normal)
    attack_count = num_samples // 2
    normal_count = num_samples - attack_count

    # Add attack samples
    for i in range(attack_count):
        pattern = random.choice(attack_patterns)
        timestamp = start_time + timedelta(seconds=random.randint(0, 30 * 24 * 60 * 60))

        # Add some variation to the patterns
        variation_words = ["attempted", "detected", "identified", "observed", "reported"]
        description = pattern["description"]
        if random.random() < 0.3:
            description = description.replace("detected", random.choice(variation_words))

        sample = {
            "id": f"event_{i + 1:04d}",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "event_type": pattern["event_type"],
            "severity": pattern["severity"],
            "source_ip": pattern["source_ip"],
            "target_port": pattern["target_port"],
            "classification": pattern["classification"],
            "is_attack": 1,
            "risk_score": random.randint(7, 10),
        }
        samples.append(sample)

    # Add normal samples
    for i in range(normal_count):
        pattern = random.choice(normal_patterns)
        timestamp = start_time + timedelta(seconds=random.randint(0, 30 * 24 * 60 * 60))

        # Add some variation to the patterns
        variation_words = ["completed", "processed", "executed", "performed", "finished"]
        description = pattern["description"]
        if random.random() < 0.3:
            description = description.replace("completed", random.choice(variation_words))

        sample = {
            "id": f"event_{attack_count + i + 1:04d}",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "event_type": pattern["event_type"],
            "severity": pattern["severity"],
            "source_ip": pattern["source_ip"],
            "target_port": pattern["target_port"],
            "classification": pattern["classification"],
            "is_attack": 0,
            "risk_score": random.randint(1, 3),
        }
        samples.append(sample)

    # Shuffle samples to randomize order
    random.shuffle(samples)

    # Write to CSV
    fieldnames = [
        "id",
        "timestamp",
        "description",
        "event_type",
        "severity",
        "source_ip",
        "target_port",
        "classification",
        "is_attack",
        "risk_score",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f"✅ Created sample dataset: {filename}")
    print(f"📊 Total samples: {len(samples)}")
    print(f"🔴 Attack samples: {attack_count}")
    print(f"🟢 Normal samples: {normal_count}")
    print("\n📋 Column Information:")
    print("  - description: Main text for analysis")
    print("  - classification: Label (attack/malicious/threat vs normal/benign/legitimate)")
    print("  - is_attack: Binary label (1=attack, 0=normal)")
    print("\n🧪 Test with:")
    print("python3 lm_studio_kaggle_benchmark.py \\")
    print("    --model-name 'your-model' \\")
    print(f"    --kaggle-dataset '{filename}' \\")
    print("    --text-column 'description' \\")
    print("    --label-column 'classification'")


def create_different_format_examples() -> None:
    """Create additional sample datasets in different formats."""

    # Format 1: Network intrusion style
    network_data = []
    for i in range(100):
        if i % 2 == 0:
            # Attack
            network_data.append(
                {
                    "flow_id": f"flow_{i:04d}",
                    "protocol": random.choice(["TCP", "UDP", "ICMP"]),
                    "service": random.choice(["http", "ftp", "ssh", "telnet"]),
                    "flag": random.choice(["SF", "S0", "REJ", "RSTR"]),
                    "packet_info": f"packets:{random.randint(1, 1000)} bytes:{random.randint(100, 10000)}",
                    "label": random.choice(["dos", "probe", "r2l", "u2r"]),
                }
            )
        else:
            # Normal
            network_data.append(
                {
                    "flow_id": f"flow_{i:04d}",
                    "protocol": random.choice(["TCP", "UDP"]),
                    "service": random.choice(["http", "smtp", "pop3", "dns"]),
                    "flag": "SF",
                    "packet_info": f"packets:{random.randint(1, 100)} bytes:{random.randint(50, 1000)}",
                    "label": "normal",
                }
            )

    with open("sample_network_intrusion.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["flow_id", "protocol", "service", "flag", "packet_info", "label"]
        )
        writer.writeheader()
        writer.writerows(network_data)

    # Format 2: Malware detection style
    malware_data = []
    for i in range(100):
        if i % 2 == 0:
            # Malware
            malware_data.append(
                {
                    "filename": f"file_{i:04d}.exe",
                    "file_info": f"size:{random.randint(1000, 100000)} entropy:{random.uniform(6.0, 8.0):.2f}",
                    "behavior": random.choice(
                        [
                            "attempts network connection to suspicious domain",
                            "modifies system registry keys",
                            "creates multiple processes",
                            "accesses sensitive files",
                        ]
                    ),
                    "is_malware": 1,
                }
            )
        else:
            # Benign
            malware_data.append(
                {
                    "filename": f"file_{i:04d}.exe",
                    "file_info": f"size:{random.randint(100, 10000)} entropy:{random.uniform(3.0, 6.0):.2f}",
                    "behavior": random.choice(
                        [
                            "normal application startup",
                            "reads configuration files",
                            "displays user interface",
                            "processes user input",
                        ]
                    ),
                    "is_malware": 0,
                }
            )

    with open("sample_malware_detection.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "file_info", "behavior", "is_malware"])
        writer.writeheader()
        writer.writerows(malware_data)

    print("\n✅ Created additional sample datasets:")
    print("📁 sample_network_intrusion.csv - Network flow analysis")
    print("📁 sample_malware_detection.csv - Malware behavior analysis")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create sample Kaggle-style cybersecurity datasets"
    )
    parser.add_argument("--samples", type=int, default=300, help="Number of samples to generate")
    parser.add_argument(
        "--filename", type=str, default="sample_cybersecurity_dataset.csv", help="Output filename"
    )
    parser.add_argument(
        "--create-variants", action="store_true", help="Create additional format variants"
    )

    args = parser.parse_args()

    # Set random seed for reproducible datasets
    random.seed(42)

    # Generate main dataset
    generate_sample_dataset(args.filename, args.samples)

    # Generate additional format examples
    if args.create_variants:
        create_different_format_examples()

    print("\n🎯 Next steps:")
    print("1. Test the sample dataset with the benchmark")
    print("2. Download real Kaggle datasets for production use")
    print("3. Adjust column names based on your dataset structure")

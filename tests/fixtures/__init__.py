"""
Test fixtures and utilities for the LLM Cybersecurity Benchmark system.

This module provides reusable test fixtures and utilities to support
comprehensive testing of all system components.
"""

from pathlib import Path
from typing import Any


def load_fixture_data(fixture_name: str) -> dict[str, Any]:
    """
    Load fixture data from YAML files in the fixtures directory.

    Args:
        fixture_name: Name of the fixture file (without .yaml extension)

    Returns:
        Dictionary containing the fixture data

    Raises:
        FileNotFoundError: If the fixture file doesn't exist
    """
    import yaml

    fixture_path = Path(__file__).parent / f"{fixture_name}.yaml"

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path) as f:
        return yaml.safe_load(f)


def create_sample_dataset_file(output_path: Path, num_samples: int = 10) -> Path:
    """
    Create a sample dataset file with cybersecurity data.

    Args:
        output_path: Path where to create the dataset file
        num_samples: Number of samples to generate

    Returns:
        Path to the created dataset file
    """
    import json

    # Sample cybersecurity data templates
    attack_samples = [
        {"text": "Port scan detected on {ip}", "label": "ATTACK", "attack_type": "reconnaissance"},
        {
            "text": "SQL injection attempt: ' OR 1=1 --",
            "label": "ATTACK",
            "attack_type": "injection",
        },
        {"text": "Malware detected: trojan.exe", "label": "ATTACK", "attack_type": "malware"},
        {
            "text": "Brute force login attempt detected",
            "label": "ATTACK",
            "attack_type": "brute_force",
        },
        {
            "text": "Cross-site scripting attempt: <script>alert('xss')</script>",
            "label": "ATTACK",
            "attack_type": "xss",
        },
    ]

    benign_samples = [
        {"text": "User logged in successfully", "label": "BENIGN", "attack_type": None},
        {"text": "Normal HTTP GET request to /api/users", "label": "BENIGN", "attack_type": None},
        {"text": "File upload completed successfully", "label": "BENIGN", "attack_type": None},
        {"text": "Database backup completed", "label": "BENIGN", "attack_type": None},
        {"text": "System health check passed", "label": "BENIGN", "attack_type": None},
    ]

    # Generate samples
    samples = []
    for i in range(num_samples):
        if i % 2 == 0:  # Alternate between attack and benign
            sample = attack_samples[i % len(attack_samples)].copy()
            sample["text"] = sample["text"].format(ip=f"192.168.1.{10 + i}")
        else:
            sample = benign_samples[i % len(benign_samples)].copy()

        sample["sample_id"] = str(i + 1)
        samples.append(sample)

    # Write to JSONL format
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return output_path


def create_sample_predictions_file(output_path: Path, num_predictions: int = 10) -> Path:
    """
    Create a sample predictions file.

    Args:
        output_path: Path where to create the predictions file
        num_predictions: Number of predictions to generate

    Returns:
        Path to the created predictions file
    """
    import json
    import random

    predictions = []
    for i in range(num_predictions):
        predictions.append(
            {
                "sample_id": str(i + 1),
                "input_text": f"Sample input text {i + 1}",
                "prediction": random.choice(["ATTACK", "BENIGN"]),
                "confidence": round(random.uniform(0.5, 1.0), 3),
                "explanation": f"Prediction explanation for sample {i + 1}",
                "inference_time_ms": round(random.uniform(50, 200), 2),
            }
        )

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    return output_path


__all__ = [
    "load_fixture_data",
    "create_sample_dataset_file",
    "create_sample_predictions_file",
]

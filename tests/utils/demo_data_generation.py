#!/usr/bin/env python3
"""
Demo script for CybersecurityDataGenerator.

This script demonstrates how to use the data generator to create realistic
cybersecurity test data for various attack types and scenarios.
"""

import json
from pathlib import Path

from data_generators import CybersecurityDataGenerator


def demo_network_logs():
    """Demonstrate network log generation."""
    print("🌐 Network Log Generation Demo")
    print("=" * 50)

    generator = CybersecurityDataGenerator(seed=42)

    # Generate different types of attack logs
    attack_types = ["malware", "intrusion", "dos", "reconnaissance"]

    for attack_type in attack_types:
        log = generator.generate_network_log(is_attack=True, attack_type=attack_type)
        print(f"\n📊 {attack_type.upper()} Attack Log:")
        print(f"  Source: {log['src_ip']}:{log['src_port']}")
        print(f"  Target: {log['dst_ip']}:{log['dst_port']}")
        print(f"  Description: {log['text']}")
        print(f"  Severity: {log['severity']} (Confidence: {log['confidence']:.3f})")

    # Generate benign log
    benign_log = generator.generate_network_log(is_attack=False)
    print("\n✅ Benign Network Log:")
    print(f"  Description: {benign_log['text']}")
    print(f"  Confidence: {benign_log['confidence']:.3f}")


def demo_email_generation():
    """Demonstrate email sample generation."""
    print("\n\n📧 Email Generation Demo")
    print("=" * 50)

    generator = CybersecurityDataGenerator(seed=123)

    # Generate phishing emails
    phishing_types = ["spear_phishing", "whaling", "smishing"]

    for phishing_type in phishing_types:
        email = generator.generate_email_sample(is_phishing=True, phishing_type=phishing_type)
        print(f"\n🎣 {phishing_type.replace('_', ' ').title()} Email:")
        print(f"  From: {email['sender']}")
        print(f"  Subject: {email['subject']}")
        print(f"  Suspicious URLs: {len(email['additional_data']['suspicious_urls'])}")
        print(f"  Confidence: {email['confidence']:.3f}")

    # Generate legitimate email
    legitimate = generator.generate_email_sample(is_phishing=False)
    print("\n✉️  Legitimate Email:")
    print(f"  From: {legitimate['sender']}")
    print(f"  Subject: {legitimate['subject']}")
    print(f"  Confidence: {legitimate['confidence']:.3f}")


def demo_model_predictions():
    """Demonstrate model prediction generation."""
    print("\n\n🤖 Model Prediction Demo")
    print("=" * 50)

    generator = CybersecurityDataGenerator(seed=456)

    # Generate predictions with different accuracy levels
    accuracies = [0.95, 0.85, 0.70]

    for accuracy in accuracies:
        print(f"\n📈 Model with {accuracy * 100}% accuracy:")

        # Generate multiple predictions to show distribution
        correct_count = 0
        total_predictions = 10

        for _ in range(total_predictions):
            pred = generator.generate_model_prediction("ATTACK", accuracy=accuracy)
            if pred["is_correct"]:
                correct_count += 1

        actual_accuracy = correct_count / total_predictions
        print(
            f"  Generated accuracy: {actual_accuracy * 100}% ({correct_count}/{total_predictions})"
        )

        # Show sample prediction
        sample_pred = generator.generate_model_prediction("ATTACK", accuracy=accuracy)
        print(
            f"  Sample prediction: {sample_pred['prediction']} (confidence: {sample_pred['confidence']:.3f})"
        )
        print(
            f"  Ground truth: {sample_pred['ground_truth']} - {'✓' if sample_pred['is_correct'] else '✗'}"
        )


def demo_explanations():
    """Demonstrate explanation generation."""
    print("\n\n💬 Explanation Generation Demo")
    print("=" * 50)

    generator = CybersecurityDataGenerator(seed=789)

    # Generate explanations for different attack types
    attack_types = ["malware", "phishing", "dos", "reconnaissance"]

    for attack_type in attack_types:
        explanation = generator.generate_explanation("ATTACK", attack_type)
        print(f"\n🔍 {attack_type.upper()} Explanation:")
        print(f"  {explanation}")

    # Generate benign explanation
    benign_explanation = generator.generate_explanation("BENIGN")
    print("\n✅ Benign Explanation:")
    print(f"  {benign_explanation}")


def demo_performance_data():
    """Demonstrate performance data generation."""
    print("\n\n⚡ Performance Data Demo")
    print("=" * 50)

    generator = CybersecurityDataGenerator(seed=101112)

    performance_data = generator.generate_performance_data(num_samples=5)

    print("\n📊 Sample Performance Metrics:")
    for i, sample in enumerate(performance_data, 1):
        print(f"\n  Sample {i} ({sample['model_size']} model):")
        print(f"    Inference time: {sample['inference_time_ms']:.1f}ms")
        print(f"    Memory usage: {sample['memory_usage_mb']:.1f}MB")
        print(f"    Throughput: {sample['throughput_samples_per_second']:.1f} samples/sec")
        print(f"    Batch size: {sample['batch_size']}")


def demo_batch_generation():
    """Demonstrate batch sample generation."""
    print("\n\n📦 Batch Generation Demo")
    print("=" * 50)

    generator = CybersecurityDataGenerator(seed=131415)

    # Generate mixed batch
    samples = generator.generate_batch_samples(
        num_samples=20, attack_ratio=0.4, attack_types=["malware", "phishing", "dos"]
    )

    # Analyze the batch
    attack_count = sum(1 for s in samples if s["label"] == "ATTACK")
    benign_count = sum(1 for s in samples if s["label"] == "BENIGN")

    attack_type_counts = {}
    for sample in samples:
        if sample["label"] == "ATTACK":
            attack_type = sample["attack_type"]
            attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1

    print("\n📈 Batch Statistics (20 samples):")
    print(f"  Attack samples: {attack_count}")
    print(f"  Benign samples: {benign_count}")
    print(f"  Attack ratio: {attack_count / len(samples) * 100:.1f}%")

    print("\n🎯 Attack Type Distribution:")
    for attack_type, count in attack_type_counts.items():
        print(f"  {attack_type}: {count} samples")

    # Show sample entries
    print("\n📝 Sample Entries:")
    for i, sample in enumerate(samples[:3], 1):
        sample_type = "Email" if "sender" in sample else "Network Log"
        print(f"  {i}. {sample_type} - {sample['label']} ({sample.get('attack_type', 'N/A')})")
        print(f"     {sample['text'][:80]}...")


def save_sample_data():
    """Save sample generated data to files."""
    print("\n\n💾 Saving Sample Data")
    print("=" * 50)

    output_dir = Path(__file__).parent / "sample_output"
    output_dir.mkdir(exist_ok=True)

    generator = CybersecurityDataGenerator(seed=161718)

    # Generate and save different types of data
    data_sets = {
        "network_logs.json": generator.generate_batch_samples(50, 0.3, ["malware", "intrusion"]),
        "email_samples.json": [generator.generate_email_sample(is_phishing=True) for _ in range(10)]
        + [generator.generate_email_sample(is_phishing=False) for _ in range(15)],
        "model_predictions.json": [
            generator.generate_model_prediction("ATTACK" if i % 3 != 0 else "BENIGN")
            for i in range(30)
        ],
        "performance_data.json": generator.generate_performance_data(25),
    }

    for filename, data in data_sets.items():
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved {len(data)} records to {filepath}")

    print(f"\n📁 All sample data saved to: {output_dir}")


def main():
    """Run all demonstrations."""
    print("🔒 Cybersecurity Data Generator Demo")
    print("=" * 60)

    demo_network_logs()
    demo_email_generation()
    demo_model_predictions()
    demo_explanations()
    demo_performance_data()
    demo_batch_generation()
    save_sample_data()

    print("\n\n🎉 Demo completed successfully!")
    print("Check the generated sample data files for more examples.")


if __name__ == "__main__":
    main()

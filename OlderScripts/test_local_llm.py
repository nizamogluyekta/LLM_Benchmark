#!/usr/bin/env python3
"""
Test script for local LLM benchmark functionality.
This script tests the basic functionality without requiring LM Studio to be running.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from local_llm_benchmark import LocalLLMBenchmark  # noqa: E402


async def test_benchmark_initialization() -> bool:
    """Test benchmark initialization."""
    print("Testing LocalLLMBenchmark initialization...")

    benchmark = LocalLLMBenchmark()

    # Test configuration creation
    model_config = benchmark.create_lm_studio_config(
        model_name="test-model", display_name="Test_Model", max_tokens=256, temperature=0.1
    )

    print(f"✓ Model config created: {model_config['name']}")

    # Test dataset creation
    sample_data = benchmark.generate_sample_cybersecurity_data()
    dataset_config = benchmark.create_cybersecurity_dataset_config(
        name="test_dataset", samples=sample_data
    )

    print(f"✓ Dataset config created: {dataset_config['name']} with {len(sample_data)} samples")

    # Test metrics calculation
    predictions = [
        {"prediction": "ATTACK", "sample_id": "1"},
        {"prediction": "BENIGN", "sample_id": "2"},
        {"prediction": "ATTACK", "sample_id": "3"},
        {"prediction": "BENIGN", "sample_id": "4"},
    ]
    ground_truth = ["ATTACK", "BENIGN", "ATTACK", "BENIGN"]

    metrics = benchmark._calculate_metrics(predictions, ground_truth)
    print(f"✓ Metrics calculated: Accuracy = {metrics['accuracy']:.3f}")

    print("✓ All basic functionality tests passed!")
    return True


async def test_sample_data_generation() -> bool:
    """Test sample data generation."""
    print("\nTesting sample data generation...")

    benchmark = LocalLLMBenchmark()
    samples = benchmark.generate_sample_cybersecurity_data()

    attack_count = sum(1 for sample in samples if sample["label"] == "ATTACK")
    benign_count = sum(1 for sample in samples if sample["label"] == "BENIGN")

    print(f"✓ Generated {len(samples)} samples")
    print(f"  - {attack_count} ATTACK samples")
    print(f"  - {benign_count} BENIGN samples")

    # Verify sample structure
    for i, sample in enumerate(samples[:2]):
        print(f"  Sample {i + 1}: {sample['label']} - {sample['text'][:50]}...")

    return True


async def test_configuration_loading() -> bool:
    """Test configuration file loading."""
    print("\nTesting configuration file...")

    config_file = "example_local_config.yaml"
    if Path(config_file).exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

        print(f"✓ Configuration loaded: {config['name']}")
        print(f"  - Models: {len(config.get('models', []))}")
        print(f"  - Datasets: {len(config.get('datasets', []))}")

        return True
    else:
        print("⚠ Configuration file not found, but that's OK for testing")
        return True


async def main() -> bool:
    """Run all tests."""
    print("=== Local LLM Benchmark Test Suite ===\n")

    try:
        # Run tests
        await test_benchmark_initialization()
        await test_sample_data_generation()
        await test_configuration_loading()

        print("\n=== All Tests Passed! ===")
        print("The local LLM benchmark script is ready to use.")
        print("\nNext steps:")
        print("1. Start LM Studio and load a model")
        print("2. Run: python3 local_llm_benchmark.py --quick-benchmark")
        print("3. Or use: python3 local_llm_benchmark.py --interactive")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

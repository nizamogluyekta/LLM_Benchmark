#!/usr/bin/env python3
"""
Test script for Kaggle dataset benchmark functionality.

This script demonstrates how to:
1. Create a sample cybersecurity dataset
2. Test the Kaggle benchmark functionality
3. Show different usage patterns

Usage:
    python3 test_kaggle_benchmark.py --create-sample
    python3 test_kaggle_benchmark.py --test-sample-dataset
"""

import argparse
import asyncio
from pathlib import Path

# Import our modules
from create_sample_kaggle_dataset import generate_sample_dataset
from lm_studio_kaggle_benchmark import LMStudioKaggleBenchmark


async def test_sample_dataset_creation() -> str:
    """Test creating and analyzing a sample dataset."""
    print("🔧 Creating sample cybersecurity dataset...")

    # Create a small test dataset
    dataset_file = "test_cybersecurity_dataset.csv"
    generate_sample_dataset(dataset_file, num_samples=50)

    print(f"\n✅ Created test dataset: {dataset_file}")

    # Show first few lines
    with open(dataset_file) as f:
        lines = f.readlines()[:6]
        print("\n📋 Dataset preview:")
        for line in lines:
            print(f"  {line.strip()}")

    return dataset_file


async def test_dataset_loader() -> None:
    """Test the dataset loading functionality."""
    print("\n🔍 Testing dataset loader...")

    # Create benchmark instance (no need for actual LM Studio connection for this test)
    benchmark = LMStudioKaggleBenchmark()

    # Create a test dataset
    dataset_file = await test_sample_dataset_creation()

    try:
        # Load the dataset
        dataset = benchmark.dataset_loader.load_dataset(
            dataset_file, text_column="description", label_column="classification", max_samples=20
        )

        print(f"✅ Successfully loaded {len(dataset)} samples")
        print("📊 Sample data structure:")
        if dataset:
            sample = dataset[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")

        # Test train/test split
        train_set, test_set = benchmark.split_dataset(dataset)
        print(f"📈 Train/test split: {len(train_set)} training, {len(test_set)} testing")

        # Test few-shot prompt creation
        if len(train_set) >= 10 and test_set:
            prompt, examples = benchmark.create_few_shot_prompt(
                train_set, test_set[0]["text"], num_examples=6
            )
            print(f"✅ Created few-shot prompt with {len(examples)} examples")
            print(f"📝 Prompt length: {len(prompt)} characters")

        print("✅ All dataset tests passed!")

    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        raise

    # Clean up
    Path(dataset_file).unlink(missing_ok=True)


async def simulate_benchmark_run() -> None:
    """Simulate a benchmark run without requiring LM Studio."""
    print("\n🎯 Simulating benchmark workflow...")

    # Create benchmark instance
    benchmark = LMStudioKaggleBenchmark()

    # Generate synthetic data
    print("📊 Generating synthetic cybersecurity data...")
    dataset = benchmark.generate_synthetic_cybersecurity_data(total_samples=30)

    print(f"✅ Generated {len(dataset)} synthetic samples")

    # Show distribution
    attack_count = sum(1 for s in dataset if s["label"] == "ATTACK")
    benign_count = len(dataset) - attack_count
    print(f"📈 Distribution: {attack_count} ATTACK, {benign_count} BENIGN")

    # Test split
    train_set, test_set = benchmark.split_dataset(dataset, train_ratio=0.7)
    print(f"📚 Split: {len(train_set)} training, {len(test_set)} testing")

    # Show sample training examples
    print("\n📋 Sample training examples:")
    for i, example in enumerate(train_set[:3], 1):
        print(f"  {i}. [{example['label']}] {example['text'][:60]}...")

    print("✅ Synthetic data generation and processing successful!")


async def show_usage_examples() -> None:
    """Show different usage examples."""
    print("\n📚 USAGE EXAMPLES:")
    print("=" * 50)

    print("\n1. 🎯 Use with sample dataset:")
    print("   python3 create_sample_kaggle_dataset.py --samples 200")
    print("   python3 lm_studio_kaggle_benchmark.py \\")
    print("       --model-name 'meta-llama-3.1-8b-instruct' \\")
    print("       --kaggle-dataset 'sample_cybersecurity_dataset.csv' \\")
    print("       --text-column 'description' \\")
    print("       --label-column 'classification' \\")
    print("       --max-samples 100")

    print("\n2. 🔄 Use with synthetic data:")
    print("   python3 lm_studio_kaggle_benchmark.py \\")
    print("       --model-name 'meta-llama-3.1-8b-instruct' \\")
    print("       --use-synthetic \\")
    print("       --max-samples 150")

    print("\n3. 🌐 Use with real Kaggle dataset:")
    print("   # Download from Kaggle: NSL-KDD, CICIDS2017, etc.")
    print("   python3 lm_studio_kaggle_benchmark.py \\")
    print("       --model-name 'your-model' \\")
    print("       --kaggle-dataset 'real_dataset.csv' \\")
    print("       --text-column 'description' \\")
    print("       --label-column 'label' \\")
    print("       --few-shot-examples 15")

    print("\n4. 🧬 Custom configuration:")
    print("   python3 lm_studio_kaggle_benchmark.py \\")
    print("       --model-name 'custom-model' \\")
    print("       --kaggle-dataset 'data.csv' \\")
    print("       --text-column 'event_description' \\")
    print("       --label-column 'is_attack' \\")
    print("       --max-samples 500 \\")
    print("       --few-shot-examples 20 \\")
    print("       --base-url 'http://localhost:8080/v1'")


async def main() -> bool:
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Kaggle benchmark functionality")
    parser.add_argument("--create-sample", action="store_true", help="Create sample dataset")
    parser.add_argument("--test-loader", action="store_true", help="Test dataset loader")
    parser.add_argument("--simulate", action="store_true", help="Simulate benchmark run")
    parser.add_argument("--show-examples", action="store_true", help="Show usage examples")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if args.all or not any(
        [args.create_sample, args.test_loader, args.simulate, args.show_examples]
    ):
        # Run all tests by default
        args.create_sample = True
        args.test_loader = True
        args.simulate = True
        args.show_examples = True

    print("🧪 KAGGLE DATASET BENCHMARK TESTING")
    print("=" * 40)

    try:
        if args.create_sample:
            await test_sample_dataset_creation()

        if args.test_loader:
            await test_dataset_loader()

        if args.simulate:
            await simulate_benchmark_run()

        if args.show_examples:
            await show_usage_examples()

        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n✅ Your Kaggle dataset benchmark is ready to use!")
        print("\n📋 Next steps:")
        print("1. Start LM Studio and load your model")
        print("2. Download a real cybersecurity dataset from Kaggle")
        print("3. Run the benchmark with your data")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

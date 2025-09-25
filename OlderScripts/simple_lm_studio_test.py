#!/usr/bin/env python3
"""
Simple LM Studio Test Script

This script directly tests LM Studio connection without the complex model service architecture.
Use this to verify your LM Studio setup works before running the full benchmark.
"""

import asyncio
import time

import openai


async def test_lm_studio_connection(
    model_name: str = "meta-llama-3.1-8b-instruct", base_url: str = "http://localhost:1234/v1"
) -> None:
    """Test LM Studio connection and run simple predictions."""

    print("🔍 Testing LM Studio connection...")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    print("-" * 50)

    # Initialize OpenAI client for LM Studio
    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key="lm-studio",  # Dummy key for local
    )

    # Test samples
    test_samples = [
        "192.168.1.100 attempted SQL injection: ' OR '1'='1' -- on login form",
        "User john.doe successfully logged in from 192.168.1.55 at 09:15 AM",
        "DDoS attack detected: 10,000 requests/second from botnet IPs",
        "Normal web traffic: GET /index.html HTTP/1.1 from legitimate browser",
    ]

    ground_truth = ["ATTACK", "BENIGN", "ATTACK", "BENIGN"]

    print("📡 Testing API connection...")

    try:
        # Test 1: List models
        print("1. Checking available models...")
        # Note: We'll make a simple chat request since LM Studio might not implement /models endpoint properly

        # Test 2: Simple prediction
        print("2. Testing simple prediction...")

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert. Classify the following as ATTACK or BENIGN.",
                },
                {
                    "role": "user",
                    "content": "Test: User successfully logged in. Classify this as ATTACK or BENIGN.",
                },
            ],
            max_tokens=10,
            temperature=0.1,
        )

        print("✅ API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        print("-" * 50)

        # Test 3: Run on all samples
        print("3. Running cybersecurity analysis on test samples...")

        predictions = []
        start_time = time.time()

        for i, sample in enumerate(test_samples):
            print(f"   Processing sample {i + 1}/4...")

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert analyzing network logs. Respond with only 'ATTACK' or 'BENIGN' followed by a brief explanation.",
                    },
                    {"role": "user", "content": f"Analyze this security event: {sample}"},
                ],
                max_tokens=100,
                temperature=0.1,
            )

            content = response.choices[0].message.content
            prediction = (content or "").strip()
            predictions.append(prediction)

            # Simple classification extraction
            classification = "ATTACK" if "ATTACK" in prediction.upper() else "BENIGN"
            correct = "✅" if classification == ground_truth[i] else "❌"

            print(f"   {correct} Expected: {ground_truth[i]}, Got: {classification}")
            print(f"   Full response: {prediction[:80]}...")

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate simple accuracy
        correct_predictions = 0
        for i, pred in enumerate(predictions):
            classification = "ATTACK" if "ATTACK" in pred.upper() else "BENIGN"
            if classification == ground_truth[i]:
                correct_predictions += 1

        accuracy = correct_predictions / len(predictions)

        print("-" * 50)
        print("📊 Test Results Summary:")
        print(f"✅ Total samples: {len(test_samples)}")
        print(f"✅ Correct predictions: {correct_predictions}")
        print(f"✅ Accuracy: {accuracy:.1%}")
        print(f"✅ Processing time: {total_time:.2f} seconds")
        print(f"✅ Speed: {len(test_samples) / total_time:.1f} samples/second")
        print("-" * 50)
        print("🎉 LM Studio connection test completed successfully!")
        print("\nYour setup is working! You can now run the full benchmark:")
        print(f'python3 local_llm_benchmark.py --quick-benchmark --model-name "{model_name}"')

    except Exception as e:
        print(f"❌ Error testing LM Studio connection: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Make sure LM Studio server is running (green status)")
        print("2. Check that the model is loaded in LM Studio")
        print("3. Verify the model name matches exactly")
        print("4. Try a different port if not using 1234")
        print(f"5. Test manually: curl {base_url}/models")


async def main() -> None:
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test LM Studio Connection")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama-3.1-8b-instruct",
        help="Model name in LM Studio",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:1234/v1", help="LM Studio base URL"
    )

    args = parser.parse_args()

    await test_lm_studio_connection(args.model_name, args.base_url)


if __name__ == "__main__":
    asyncio.run(main())

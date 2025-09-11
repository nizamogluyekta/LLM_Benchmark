# 🛡️ LLM Cybersecurity Testing Instructions

This guide provides complete step-by-step instructions for testing LLM models with cybersecurity datasets using the LLM Cybersecurity Benchmark system.

## 📋 Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Quick Start: Testing Your First Model](#quick-start-testing-your-first-model)
3. [Creating Cybersecurity Datasets](#creating-cybersecurity-datasets)
4. [Model Configuration & Testing](#model-configuration--testing)
5. [Running Comprehensive Evaluations](#running-comprehensive-evaluations)
6. [Performance Monitoring & Analysis](#performance-monitoring--analysis)
7. [Multi-Model Comparisons](#multi-model-comparisons)
8. [Advanced Workflows](#advanced-workflows)
9. [Troubleshooting](#troubleshooting)

---

## 🚀 Prerequisites & Setup

### 1. System Requirements

- **Operating System**: macOS (optimized for Apple Silicon M4 Pro) or Linux
- **Python**: 3.11 or higher
- **Poetry**: For dependency management
- **Memory**: 8GB+ RAM (16GB recommended for multiple models)
- **Storage**: 5GB+ free space

### 2. Installation

```bash
# Navigate to the project directory
cd /Users/nizamogluyekta/Desktop/Playground/LLM_Benchmark

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Verify installation
poetry run python --version
```

### 3. Validate Installation

```bash
# Run core system tests
poetry run pytest tests/unit/ -v --tb=short

# Check service imports
PYTHONPATH=src python -c "
from benchmark.services.data_service import DataService
from benchmark.services.model_service import ModelService
print('✅ All services imported successfully')
"
```

### 4. Environment Configuration (Optional)

If you want to test with real API models, set up environment variables:

```bash
# Create environment file
cp .env.example .env

# Edit .env and add your API keys
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Load environment variables
source .env
```

**Note**: The system works without API keys using mock implementations for testing.

---

## 🎯 Quick Start: Testing Your First Model

Let's start with a simple test to get familiar with the system.

### Step 1: Create Your First Test Script

```bash
# Create a simple test script
cat > my_first_test.py << 'EOF'
import asyncio
import json
import tempfile
from benchmark.services.data_service import DataService
from benchmark.services.model_service import ModelService
from benchmark.core.config import DatasetConfig

async def test_llm_cybersecurity():
    """Your first LLM cybersecurity test."""

    print("🚀 Starting LLM Cybersecurity Test")

    # 1. Initialize services
    print("📊 Initializing Data Service...")
    data_service = DataService(enable_hardware_optimization=True)
    await data_service.initialize()

    print("🤖 Initializing Model Service...")
    model_service = ModelService(enable_performance_monitoring=True)
    await model_service.initialize()

    # 2. Create test cybersecurity dataset
    print("🎲 Creating cybersecurity test dataset...")
    cybersecurity_samples = [
        {
            "content": "SELECT * FROM users WHERE id = '1' OR '1'='1'",
            "label": "ATTACK",
            "attack_type": "sql_injection",
            "severity": "high"
        },
        {
            "content": "<script>alert('XSS Attack')</script>",
            "label": "ATTACK",
            "attack_type": "xss",
            "severity": "medium"
        },
        {
            "content": "; rm -rf / --no-preserve-root",
            "label": "ATTACK",
            "attack_type": "command_injection",
            "severity": "critical"
        },
        {
            "content": "Welcome to our secure banking platform",
            "label": "BENIGN",
            "attack_type": None,
            "severity": "none"
        },
        {
            "content": "User login successful for user@company.com",
            "label": "BENIGN",
            "attack_type": None,
            "severity": "none"
        }
    ]

    # Save dataset to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cybersecurity_samples, f)
        dataset_file = f.name

    print(f"💾 Created dataset with {len(cybersecurity_samples)} samples")

    # 3. Load dataset through data service
    dataset_config = DatasetConfig(
        name="first_test_dataset",
        path=dataset_file,
        source="local",
        format="json"
    )

    dataset = await data_service.load_dataset(dataset_config)
    print(f"✅ Dataset loaded: {dataset.size} samples")

    # 4. Load and test model
    print("🤖 Loading model...")
    model_config = {
        "type": "openai_api",  # This will use mock implementation
        "model_name": "gpt-4o-mini",
        "name": "test-model",
        "api_key": "test-key-123"
    }

    model_id = await model_service.load_model(model_config)
    print(f"✅ Model loaded: {model_id}")

    # 5. Run inference on cybersecurity samples
    print("🧪 Running cybersecurity evaluation...")
    test_samples = [sample["content"] for sample in cybersecurity_samples]

    response = await model_service.predict_batch(
        model_id,
        test_samples,
        batch_size=3
    )

    print(f"📊 Inference Results:")
    print(f"   Processed: {response.successful_predictions}/{response.total_samples}")
    print(f"   Time: {response.total_inference_time_ms}ms")

    # 6. Analyze results
    attack_detected = 0
    for i, prediction in enumerate(response.predictions):
        actual_label = cybersecurity_samples[i]["label"]
        predicted_label = prediction.prediction

        print(f"   Sample {i+1}: '{prediction.input_text[:40]}...'")
        print(f"      Actual: {actual_label} | Predicted: {predicted_label} | Confidence: {prediction.confidence:.2f}")

        if predicted_label == "ATTACK":
            attack_detected += 1

    print(f"\n📈 Summary:")
    print(f"   Attack samples detected: {attack_detected}")
    print(f"   Detection rate: {attack_detected/3*100:.1f}% (3 actual attacks)")

    # 7. Get performance metrics
    performance = await model_service.get_model_performance(model_id)
    print(f"⚡ Performance: {performance['basic_metrics']['predictions_per_second']:.2f} samples/sec")

    # 8. Cleanup
    await model_service.cleanup_model(model_id)
    await data_service.shutdown()
    await model_service.shutdown()

    print("✅ First cybersecurity test completed!")

# Run the test
asyncio.run(test_llm_cybersecurity())
EOF

# Run your first test
PYTHONPATH=src python my_first_test.py
```

### Expected Output

You should see output like:
```
🚀 Starting LLM Cybersecurity Test
📊 Initializing Data Service...
🤖 Initializing Model Service...
🎲 Creating cybersecurity test dataset...
💾 Created dataset with 5 samples
✅ Dataset loaded: 5 samples
🤖 Loading model...
✅ Model loaded: test-model-12345
🧪 Running cybersecurity evaluation...
📊 Inference Results:
   Processed: 5/5
   Time: 150ms
   Sample 1: 'SELECT * FROM users WHERE id = '1' OR '1'='1'...'
      Actual: ATTACK | Predicted: ATTACK | Confidence: 0.85
📈 Summary:
   Attack samples detected: 3
   Detection rate: 100.0% (3 actual attacks)
⚡ Performance: 33.33 samples/sec
✅ First cybersecurity test completed!
```

---

## 🎲 Creating Cybersecurity Datasets

### Option 1: Use Built-in Data Generators

Our system includes realistic cybersecurity data generators:

```bash
# Generate realistic cybersecurity data
cat > generate_dataset.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tests')))

from utils.data_generators import CybersecurityDataGenerator
import json

def generate_cybersecurity_dataset():
    """Generate a comprehensive cybersecurity dataset."""

    generator = CybersecurityDataGenerator(seed=42)

    print("🎲 Generating comprehensive cybersecurity dataset...")

    # Generate different types of samples
    samples = []

    # 1. Network attack logs (500 samples)
    print("🌐 Generating network attack logs...")
    for i in range(500):
        is_attack = (i % 3 == 0)  # 33% attacks
        attack_type = ["malware", "intrusion", "dos"][i % 3] if is_attack else None

        sample = generator.generate_network_log(
            is_attack=is_attack,
            attack_type=attack_type
        )
        samples.append(sample)

    # 2. Email samples (300 samples)
    print("📧 Generating email samples...")
    for i in range(300):
        is_phishing = (i % 4 == 0)  # 25% phishing
        phishing_type = ["spear_phishing", "credential_harvesting", "malware_delivery"][i % 3] if is_phishing else None

        sample = generator.generate_email_sample(
            is_phishing=is_phishing,
            phishing_type=phishing_type
        )
        samples.append(sample)

    # 3. SQL injection samples (200 samples)
    print("💾 Generating SQL injection samples...")
    sql_attacks = [
        "SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "admin'; DROP TABLE users; --",
        "' UNION SELECT username, password FROM accounts --",
        "1; DELETE FROM products; --",
        "' OR 'x'='x",
        "admin'/**/UNION/**/SELECT/**/password/**/FROM/**/users--"
    ]

    sql_benign = [
        "SELECT name FROM products WHERE category = 'electronics'",
        "UPDATE users SET last_login = NOW() WHERE id = 123",
        "INSERT INTO orders (user_id, product_id) VALUES (1, 45)",
        "SELECT COUNT(*) FROM active_sessions",
        "DELETE FROM temp_data WHERE created < '2024-01-01'"
    ]

    for i in range(200):
        if i % 3 == 0:  # 33% attacks
            content = sql_attacks[i % len(sql_attacks)]
            label = "ATTACK"
            attack_type = "sql_injection"
        else:
            content = sql_benign[i % len(sql_benign)]
            label = "BENIGN"
            attack_type = None

        samples.append({
            "content": content,
            "label": label,
            "attack_type": attack_type,
            "category": "database_query",
            "timestamp": f"2024-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00Z"
        })

    print(f"✅ Generated {len(samples)} total samples")

    # Calculate statistics
    attack_count = sum(1 for s in samples if s["label"] == "ATTACK")
    attack_ratio = attack_count / len(samples)

    print(f"📊 Dataset Statistics:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Attack samples: {attack_count} ({attack_ratio:.1%})")
    print(f"   Benign samples: {len(samples) - attack_count} ({1-attack_ratio:.1%})")

    # Save dataset
    output_file = "cybersecurity_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"💾 Dataset saved to: {output_file}")
    return output_file

# Generate the dataset
dataset_file = generate_cybersecurity_dataset()
print(f"\n🎯 Ready to use dataset: {dataset_file}")
EOF

# Generate your dataset
PYTHONPATH=src python generate_dataset.py
```

### Option 2: Use Real Cybersecurity Datasets

```bash
# Create a script to load UNSW-NB15 style data
cat > load_unsw_dataset.py << 'EOF'
import pandas as pd
import json
import numpy as np

def create_unsw_style_dataset():
    """Create UNSW-NB15 style network traffic dataset."""

    print("🌐 Creating UNSW-NB15 style dataset...")

    # Generate realistic network traffic data
    samples = []

    for i in range(1000):
        # Generate IP addresses
        src_ip = f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        dst_ip = f"10.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"

        # Generate ports
        src_port = np.random.randint(1024, 65535)
        dst_port = np.random.choice([80, 443, 22, 21, 25, 53, 3389])

        # Determine if attack (30% attack rate)
        is_attack = np.random.random() < 0.3

        if is_attack:
            # Attack patterns
            attack_types = ["DoS", "Probe", "R2L", "U2R"]
            attack_type = np.random.choice(attack_types)

            # Abnormal patterns for attacks
            duration = np.random.exponential(10)  # Longer durations
            src_bytes = np.random.randint(0, 1000000)  # Large byte transfers
            dst_bytes = np.random.randint(0, 500000)

            content = f"Network traffic: {src_ip}:{src_port} -> {dst_ip}:{dst_port}, suspicious {attack_type.lower()} activity detected"
            label = "ATTACK"
        else:
            # Normal traffic
            attack_type = None
            duration = np.random.exponential(1)  # Shorter durations
            src_bytes = np.random.randint(0, 10000)  # Normal byte transfers
            dst_bytes = np.random.randint(0, 5000)

            content = f"Network traffic: {src_ip}:{src_port} -> {dst_ip}:{dst_port}, normal web browsing"
            label = "BENIGN"

        sample = {
            "content": content,
            "label": label,
            "attack_type": attack_type,
            "features": {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "duration": round(duration, 2),
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "protocol": "TCP"
            },
            "timestamp": f"2024-01-{(i % 30) + 1:02d}T{(i % 24):02d}:{(i % 60):02d}:00Z"
        }

        samples.append(sample)

    # Save dataset
    output_file = "unsw_style_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    attack_count = sum(1 for s in samples if s["label"] == "ATTACK")
    print(f"✅ Created UNSW-style dataset: {len(samples)} samples ({attack_count} attacks)")
    print(f"💾 Saved to: {output_file}")

    return output_file

# Create the dataset
dataset_file = create_unsw_style_dataset()
EOF

# Create UNSW-style dataset
python load_unsw_dataset.py
```

---

## 🤖 Model Configuration & Testing

### Testing Different Model Types

#### 1. OpenAI API Models

```bash
# Test with OpenAI models (requires API key)
cat > test_openai_model.py << 'EOF'
import asyncio
from benchmark.services.model_service import ModelService

async def test_openai_cybersecurity():
    """Test OpenAI model on cybersecurity tasks."""

    service = ModelService()
    await service.initialize()

    # Configure OpenAI model
    model_config = {
        "type": "openai_api",
        "model_name": "gpt-4o-mini",  # Cost-effective option
        "name": "openai-cybersec",
        "api_key": "${OPENAI_API_KEY}",  # Uses environment variable
        "temperature": 0.1,  # Low temperature for consistent results
        "max_tokens": 100
    }

    try:
        model_id = await service.load_model(model_config)
        print(f"✅ OpenAI model loaded: {model_id}")

        # Test cybersecurity samples
        test_samples = [
            "SELECT * FROM users WHERE password = '' OR '1'='1'",
            "<img src=x onerror=alert('XSS')>",
            "Welcome to our customer support portal"
        ]

        response = await service.predict_batch(model_id, test_samples)

        for pred in response.predictions:
            print(f"Input: {pred.input_text[:50]}...")
            print(f"Prediction: {pred.prediction} (confidence: {pred.confidence:.2f})")
            print(f"Explanation: {pred.explanation}")
            print("---")

        await service.cleanup_model(model_id)

    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        print("💡 Make sure OPENAI_API_KEY environment variable is set")

    await service.shutdown()

# Run OpenAI test
asyncio.run(test_openai_cybersecurity())
EOF

# Run the test (only if you have OpenAI API key)
PYTHONPATH=src python test_openai_model.py
```

#### 2. Anthropic Claude Models

```bash
# Test with Anthropic models
cat > test_anthropic_model.py << 'EOF'
import asyncio
from benchmark.services.model_service import ModelService

async def test_anthropic_cybersecurity():
    """Test Anthropic Claude on cybersecurity tasks."""

    service = ModelService()
    await service.initialize()

    model_config = {
        "type": "anthropic_api",
        "model_name": "claude-3-haiku-20240307",
        "name": "claude-cybersec",
        "api_key": "${ANTHROPIC_API_KEY}",
        "max_tokens": 150
    }

    try:
        model_id = await service.load_model(model_config)
        print(f"✅ Anthropic model loaded: {model_id}")

        # Test with complex cybersecurity scenarios
        test_samples = [
            "#!/bin/bash\nrm -rf /tmp/*\necho 'System cleaned'",
            "import subprocess; subprocess.call(['rm', '-rf', '/'])",
            "Dear customer, please update your account information at secure-bank-login.com"
        ]

        response = await service.predict_batch(model_id, test_samples)

        for pred in response.predictions:
            print(f"Input: {pred.input_text[:50]}...")
            print(f"Prediction: {pred.prediction}")
            print(f"Attack Type: {pred.attack_type}")
            print("---")

        await service.cleanup_model(model_id)

    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        print("💡 Make sure ANTHROPIC_API_KEY environment variable is set")

    await service.shutdown()

asyncio.run(test_anthropic_cybersecurity())
EOF

# Run Anthropic test
PYTHONPATH=src python test_anthropic_model.py
```

#### 3. Local MLX Models (Apple Silicon)

```bash
# Test with MLX local models
cat > test_mlx_model.py << 'EOF'
import asyncio
from benchmark.services.model_service import ModelService

async def test_mlx_cybersecurity():
    """Test MLX local model on cybersecurity tasks."""

    service = ModelService()
    await service.initialize()

    model_config = {
        "type": "mlx_local",
        "model_name": "llama2-7b",
        "name": "mlx-cybersec",
        "model_path": "/models/llama2-7b",  # Update with actual path
        "temperature": 0.0
    }

    try:
        model_id = await service.load_model(model_config)
        print(f"✅ MLX model loaded: {model_id}")

        # Test local processing capabilities
        test_samples = [
            "ping -c 1000 192.168.1.1 & ping -c 1000 8.8.8.8",
            "Normal user registration for john@company.com",
            "<?php system($_GET['cmd']); ?>",
            "User session expired, please log in again"
        ]

        response = await service.predict_batch(model_id, test_samples, batch_size=2)

        print(f"MLX Performance: {response.total_inference_time_ms}ms for {len(test_samples)} samples")

        for pred in response.predictions:
            print(f"Input: {pred.input_text[:40]}...")
            print(f"Result: {pred.prediction}")
            print("---")

        await service.cleanup_model(model_id)

    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        print("💡 MLX models require Apple Silicon and model files")

    await service.shutdown()

asyncio.run(test_mlx_cybersecurity())
EOF

# Run MLX test
PYTHONPATH=src python test_mlx_model.py
```

---

## 🔬 Running Comprehensive Evaluations

### Full Evaluation Pipeline

```bash
# Create comprehensive evaluation script
cat > comprehensive_evaluation.py << 'EOF'
import asyncio
import json
import time
from benchmark.services.data_service import DataService
from benchmark.services.model_service import ModelService
from benchmark.core.config import DatasetConfig

async def comprehensive_cybersecurity_evaluation():
    """Run comprehensive cybersecurity evaluation."""

    print("🔬 Starting Comprehensive Cybersecurity Evaluation")

    # Initialize services
    data_service = DataService(enable_hardware_optimization=True)
    model_service = ModelService(enable_performance_monitoring=True, max_models=3)

    await data_service.initialize()
    await model_service.initialize()

    # Load cybersecurity dataset (use generated dataset)
    dataset_config = DatasetConfig(
        name="comprehensive_test",
        path="cybersecurity_dataset.json",  # From previous step
        source="local",
        format="json"
    )

    print("📊 Loading cybersecurity dataset...")
    dataset = await data_service.load_dataset(dataset_config)
    print(f"✅ Loaded dataset: {dataset.size} samples")

    # Configure multiple models for comparison
    model_configs = [
        {
            "type": "openai_api",
            "model_name": "gpt-4o-mini",
            "name": "gpt4o-mini",
            "api_key": "mock-key-123"  # Uses mock for demo
        },
        {
            "type": "anthropic_api",
            "model_name": "claude-3-haiku-20240307",
            "name": "claude-haiku",
            "api_key": "mock-key-456"
        },
        {
            "type": "mlx_local",
            "model_name": "llama2-7b",
            "name": "llama2-7b-local"
        }
    ]

    # Load models
    print("🤖 Loading models...")
    model_ids = []
    for config in model_configs:
        try:
            model_id = await model_service.load_model(config)
            model_ids.append(model_id)
            print(f"✅ Loaded {config['name']}: {model_id}")
        except Exception as e:
            print(f"⚠️  Failed to load {config['name']}: {e}")

    if not model_ids:
        print("❌ No models loaded successfully")
        return

    # Prepare evaluation samples (limit for demo)
    eval_samples = []
    sample_count = 0
    for sample in dataset.samples:
        if sample_count >= 50:  # Limit for demo
            break
        eval_samples.append(sample)
        sample_count += 1

    print(f"🧪 Running evaluation on {len(eval_samples)} samples...")

    # Evaluate each model
    results = {}
    for model_id in model_ids:
        print(f"\n📊 Testing model: {model_id}")

        # Extract content for prediction
        contents = [sample.content for sample in eval_samples]

        start_time = time.time()
        response = await model_service.predict_batch(
            model_id,
            contents,
            batch_size=8
        )
        end_time = time.time()

        # Calculate metrics
        correct_predictions = 0
        attack_detected = 0
        attack_actual = 0

        for i, prediction in enumerate(response.predictions):
            actual_label = eval_samples[i].label
            predicted_label = prediction.prediction

            if actual_label == "ATTACK":
                attack_actual += 1

            if predicted_label == "ATTACK":
                attack_detected += 1

            if actual_label == predicted_label:
                correct_predictions += 1

        # Store results
        results[model_id] = {
            "accuracy": correct_predictions / len(eval_samples),
            "precision": attack_detected / max(attack_detected, 1),  # Avoid division by zero
            "recall": attack_detected / max(attack_actual, 1),
            "processing_time": end_time - start_time,
            "samples_per_second": len(eval_samples) / (end_time - start_time),
            "total_samples": len(eval_samples),
            "successful_predictions": response.successful_predictions
        }

        print(f"   ✅ Completed: {response.successful_predictions}/{len(eval_samples)} predictions")
        print(f"   ⚡ Speed: {results[model_id]['samples_per_second']:.2f} samples/sec")
        print(f"   🎯 Accuracy: {results[model_id]['accuracy']:.1%}")

    # Model comparison
    if len(model_ids) > 1:
        print("\n🏆 Comparing models...")
        comparison = await model_service.compare_model_performance(model_ids)
        print(f"   Best performer: {comparison.summary.get('best_performer', 'N/A')}")

    # Generate evaluation report
    print("\n📋 EVALUATION REPORT")
    print("=" * 50)

    for model_id, metrics in results.items():
        print(f"\n🤖 Model: {model_id}")
        print(f"   Accuracy: {metrics['accuracy']:.1%}")
        print(f"   Processing Speed: {metrics['samples_per_second']:.2f} samples/sec")
        print(f"   Total Time: {metrics['processing_time']:.2f}s")
        print(f"   Success Rate: {metrics['successful_predictions']}/{metrics['total_samples']}")

    # Save detailed results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: evaluation_results.json")

    # Cleanup
    for model_id in model_ids:
        await model_service.cleanup_model(model_id)

    await data_service.shutdown()
    await model_service.shutdown()

    print("\n✅ Comprehensive evaluation completed!")

# Run comprehensive evaluation
asyncio.run(comprehensive_cybersecurity_evaluation())
EOF

# Run the comprehensive evaluation
PYTHONPATH=src python comprehensive_evaluation.py
```

---

## 📊 Performance Monitoring & Analysis

### Real-time Performance Monitoring

```bash
# Create performance monitoring script
cat > monitor_performance.py << 'EOF'
import asyncio
import time
from benchmark.services.model_service import ModelService

async def monitor_model_performance():
    """Monitor model performance in real-time."""

    print("📊 Starting Performance Monitoring")

    service = ModelService(enable_performance_monitoring=True)
    await service.initialize()

    # Load model for monitoring
    model_config = {
        "type": "openai_api",
        "model_name": "gpt-4o-mini",
        "name": "performance-monitor",
        "api_key": "test-key"
    }

    model_id = await service.load_model(model_config)
    print(f"✅ Model loaded for monitoring: {model_id}")

    # Test samples for continuous monitoring
    test_samples = [
        "Analyze this SQL query: SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "Is this email suspicious: 'Update your password immediately'",
        "Check this code: <script>document.location='http://attacker.com'</script>",
        "Normal login attempt for user@company.com"
    ]

    print("\n🔄 Running performance monitoring rounds...")

    for round_num in range(5):
        print(f"\n--- Round {round_num + 1} ---")

        # Run inference
        start_time = time.time()
        response = await service.predict_batch(
            model_id,
            test_samples,
            batch_size=len(test_samples)
        )
        end_time = time.time()

        # Get real-time performance metrics
        performance = await service.get_model_performance(model_id)

        if 'basic_metrics' in performance:
            metrics = performance['basic_metrics']

            print(f"⚡ Performance Metrics:")
            print(f"   Predictions/sec: {metrics['predictions_per_second']:.2f}")
            print(f"   Success rate: {metrics['success_rate']:.1%}")
            print(f"   Avg inference time: {metrics['average_inference_time_ms']:.2f}ms")
            print(f"   Total predictions: {metrics['total_predictions']}")

        # Service-level statistics
        stats = await service.get_service_stats()
        print(f"📈 Service Stats:")
        print(f"   Loaded models: {stats['loaded_models']}")
        print(f"   Service uptime: {stats.get('uptime_seconds', 0):.1f}s")

        # Wait before next round
        await asyncio.sleep(2)

    await service.cleanup_model(model_id)
    await service.shutdown()

    print("\n✅ Performance monitoring completed!")

asyncio.run(monitor_model_performance())
EOF

# Run performance monitoring
PYTHONPATH=src python monitor_performance.py
```

### Cost Analysis

```bash
# Create cost analysis script
cat > analyze_costs.py << 'EOF'
import asyncio
from benchmark.services.model_service import ModelService

async def analyze_model_costs():
    """Analyze costs for different models."""

    print("💰 Starting Cost Analysis")

    service = ModelService()
    await service.initialize()

    # Different model configurations
    model_configs = [
        {
            "type": "openai_api",
            "model_name": "gpt-4o-mini",
            "name": "openai-cost-test",
            "api_key": "test-key"
        },
        {
            "type": "anthropic_api",
            "model_name": "claude-3-haiku-20240307",
            "name": "anthropic-cost-test",
            "api_key": "test-key"
        },
        {
            "type": "mlx_local",
            "model_name": "llama2-7b",
            "name": "mlx-cost-test"
        }
    ]

    # Sample sizes for cost estimation
    sample_sizes = [100, 1000, 10000, 50000]

    print("📊 Cost Estimation Results:")
    print("=" * 60)

    for sample_size in sample_sizes:
        print(f"\n📈 Sample Size: {sample_size:,} cybersecurity samples")

        try:
            cost_estimate = await service.get_cost_estimates(model_configs, sample_size)

            print(f"   Total Estimated Cost: ${cost_estimate.total_estimated_cost_usd:.4f}")

            for model_name, cost in cost_estimate.cost_by_model.items():
                print(f"   {model_name}: ${cost:.4f}")

            # Calculate monthly projections
            daily_samples = sample_size
            monthly_cost = cost_estimate.total_estimated_cost_usd * 30

            print(f"   Monthly projection (if run daily): ${monthly_cost:.2f}")

        except Exception as e:
            print(f"   ❌ Cost estimation failed: {e}")

    await service.shutdown()
    print("\n✅ Cost analysis completed!")

asyncio.run(analyze_model_costs())
EOF

# Run cost analysis
PYTHONPATH=src python analyze_costs.py
```

---

## 🏆 Multi-Model Comparisons

### Comprehensive Model Comparison

```bash
# Create model comparison script
cat > compare_models.py << 'EOF'
import asyncio
import json
from benchmark.services.model_service import ModelService

async def compare_cybersecurity_models():
    """Compare multiple models on cybersecurity tasks."""

    print("🏆 Starting Multi-Model Comparison")

    service = ModelService(max_models=4)
    await service.initialize()

    # Configure models for comparison
    model_configs = [
        {
            "type": "openai_api",
            "model_name": "gpt-4o-mini",
            "name": "OpenAI-GPT4o-mini",
            "api_key": "test-key-1"
        },
        {
            "type": "anthropic_api",
            "model_name": "claude-3-haiku-20240307",
            "name": "Anthropic-Claude-Haiku",
            "api_key": "test-key-2"
        },
        {
            "type": "mlx_local",
            "model_name": "llama2-7b",
            "name": "MLX-Llama2-7B"
        },
        {
            "type": "ollama_local",
            "model_name": "llama2:7b",
            "name": "Ollama-Llama2"
        }
    ]

    # Load models
    print("🤖 Loading models for comparison...")
    model_ids = []
    model_names = {}

    for config in model_configs:
        try:
            model_id = await service.load_model(config)
            model_ids.append(model_id)
            model_names[model_id] = config["name"]
            print(f"✅ Loaded: {config['name']}")
        except Exception as e:
            print(f"⚠️  Failed to load {config['name']}: {e}")

    if len(model_ids) < 2:
        print("❌ Need at least 2 models for comparison")
        return

    # Cybersecurity test scenarios
    test_scenarios = [
        {
            "name": "SQL Injection Detection",
            "samples": [
                "SELECT * FROM users WHERE id = '1' OR '1'='1'",
                "admin'; DROP TABLE users; --",
                "' UNION SELECT password FROM accounts --",
                "SELECT name FROM products WHERE category = 'electronics'"  # Benign
            ]
        },
        {
            "name": "XSS Detection",
            "samples": [
                "<script>alert('XSS Attack')</script>",
                "<img src=x onerror=alert('Malicious')>",
                "javascript:document.location='http://attacker.com'",
                "<p>Welcome to our website</p>"  # Benign
            ]
        },
        {
            "name": "Command Injection Detection",
            "samples": [
                "; rm -rf / --no-preserve-root",
                "| nc -l -p 1234 -e /bin/sh",
                "&& cat /etc/passwd > /tmp/stolen.txt",
                "ls -la /home/user/documents"  # Benign
            ]
        },
        {
            "name": "Phishing Detection",
            "samples": [
                "URGENT: Your account will be suspended. Click here immediately.",
                "Congratulations! You've won $1,000,000. Claim now!",
                "Update your banking details to avoid account closure.",
                "Your order #12345 has been shipped successfully."  # Benign
            ]
        }
    ]

    # Run comparison tests
    print(f"\n🧪 Running comparison across {len(test_scenarios)} scenarios...")

    scenario_results = {}

    for scenario in test_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        scenario_results[scenario['name']] = {}

        for model_id in model_ids:
            model_name = model_names[model_id]
            print(f"   Testing {model_name}...")

            try:
                response = await service.predict_batch(
                    model_id,
                    scenario['samples'],
                    batch_size=len(scenario['samples'])
                )

                # Analyze results
                attack_detected = sum(
                    1 for pred in response.predictions
                    if pred.prediction == "ATTACK"
                )

                avg_confidence = sum(
                    pred.confidence for pred in response.predictions
                ) / len(response.predictions)

                scenario_results[scenario['name']][model_name] = {
                    "attacks_detected": attack_detected,
                    "total_samples": len(scenario['samples']),
                    "avg_confidence": avg_confidence,
                    "processing_time_ms": response.total_inference_time_ms
                }

                print(f"      Attacks detected: {attack_detected}/{len(scenario['samples'])}")
                print(f"      Avg confidence: {avg_confidence:.2f}")

            except Exception as e:
                print(f"      ❌ Failed: {e}")
                scenario_results[scenario['name']][model_name] = {"error": str(e)}

    # Use built-in model comparison
    print(f"\n🏆 Running built-in model comparison...")
    try:
        comparison = await service.compare_model_performance(model_ids)

        print(f"Best overall performer: {comparison.summary.get('best_performer', 'N/A')}")
        print(f"Models compared: {len(comparison.model_ids)}")

        # Detailed metrics
        for model_id in model_ids:
            if model_id in comparison.metrics:
                model_name = model_names[model_id]
                metrics = comparison.metrics[model_id].get('basic_metrics', {})

                print(f"\n📈 {model_name} Metrics:")
                print(f"   Success Rate: {metrics.get('success_rate', 0):.1%}")
                print(f"   Predictions/sec: {metrics.get('predictions_per_second', 0):.2f}")
                print(f"   Avg Inference Time: {metrics.get('average_inference_time_ms', 0):.2f}ms")

    except Exception as e:
        print(f"⚠️  Built-in comparison failed: {e}")

    # Generate comparison report
    print("\n📋 COMPARISON REPORT")
    print("=" * 60)

    for scenario_name, results in scenario_results.items():
        print(f"\n🎯 {scenario_name}:")

        for model_name, metrics in results.items():
            if "error" not in metrics:
                print(f"   {model_name}:")
                print(f"      Detection: {metrics['attacks_detected']}/{metrics['total_samples']}")
                print(f"      Confidence: {metrics['avg_confidence']:.2f}")
                print(f"      Speed: {metrics['processing_time_ms']}ms")

    # Save detailed results
    with open("model_comparison_results.json", "w") as f:
        json.dump(scenario_results, f, indent=2)

    print(f"\n💾 Detailed comparison saved to: model_comparison_results.json")

    # Cleanup
    for model_id in model_ids:
        await service.cleanup_model(model_id)

    await service.shutdown()
    print("\n✅ Multi-model comparison completed!")

asyncio.run(compare_cybersecurity_models())
EOF

# Run model comparison
PYTHONPATH=src python compare_models.py
```

---

## 🚀 Advanced Workflows

### End-to-End Testing

```bash
# Run the built-in E2E tests
echo "🧪 Running End-to-End Tests..."

# Run all E2E tests (16 scenarios)
PYTHONPATH=src poetry run pytest tests/e2e/ -v

# Run specific E2E test scenarios
PYTHONPATH=src poetry run pytest tests/e2e/test_model_service_e2e.py::TestModelServiceE2E::test_complete_model_lifecycle -v

# Run performance tests
PYTHONPATH=src poetry run pytest tests/performance/ -v
```

### Custom Evaluation Metrics

```bash
# Create custom metrics evaluation
cat > custom_metrics.py << 'EOF'
import asyncio
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from benchmark.services.model_service import ModelService

async def evaluate_with_custom_metrics():
    """Evaluate models with custom cybersecurity metrics."""

    print("📊 Custom Cybersecurity Metrics Evaluation")

    service = ModelService()
    await service.initialize()

    # Load model
    model_config = {
        "type": "openai_api",
        "model_name": "gpt-4o-mini",
        "name": "metrics-test",
        "api_key": "test-key"
    }

    model_id = await service.load_model(model_config)

    # Test samples with known labels
    test_data = [
        {"content": "SELECT * FROM users WHERE '1'='1'", "label": 1},  # Attack
        {"content": "<script>alert('xss')</script>", "label": 1},    # Attack
        {"content": "; rm -rf /", "label": 1},                       # Attack
        {"content": "Welcome to our site", "label": 0},              # Benign
        {"content": "User login successful", "label": 0},            # Benign
        {"content": "Thank you for visiting", "label": 0},           # Benign
    ]

    # Get predictions
    contents = [item["content"] for item in test_data]
    response = await service.predict_batch(model_id, contents)

    # Convert predictions to binary (1=ATTACK, 0=BENIGN)
    y_true = [item["label"] for item in test_data]
    y_pred = [1 if pred.prediction == "ATTACK" else 0 for pred in response.predictions]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Custom cybersecurity metrics
    true_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    false_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    false_negatives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

    print(f"\n📈 Evaluation Results:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    print(f"\n🛡️  Cybersecurity Metrics:")
    print(f"   True Positives (Attacks Caught): {true_positives}")
    print(f"   False Positives (False Alarms): {false_positives}")
    print(f"   False Negatives (Missed Attacks): {false_negatives}")
    print(f"   Attack Detection Rate: {true_positives/(true_positives+false_negatives)*100:.1f}%")

    await service.cleanup_model(model_id)
    await service.shutdown()

asyncio.run(evaluate_with_custom_metrics())
EOF

# Run custom metrics evaluation
PYTHONPATH=src python custom_metrics.py
```

---

## ❗ Troubleshooting

### Common Issues and Solutions

#### 1. Installation Issues

```bash
# Fix Poetry installation issues
poetry cache clear --all pypi
poetry install --verbose

# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
echo 'export PYTHONPATH="${PYTHONPATH}:'$(pwd)'/src"' >> ~/.bashrc
```

#### 2. Model Loading Issues

```bash
# Debug model loading
cat > debug_model_loading.py << 'EOF'
import asyncio
from benchmark.services.model_service import ModelService

async def debug_models():
    service = ModelService()
    await service.initialize()

    # Test with different configurations
    configs = [
        {"type": "openai_api", "model_name": "gpt-4o-mini", "api_key": "test"},
        {"type": "anthropic_api", "model_name": "claude-3-haiku-20240307", "api_key": "test"},
        {"type": "mlx_local", "model_name": "llama2-7b"}
    ]

    for config in configs:
        try:
            print(f"Testing {config['type']}...")
            model_id = await service.load_model(config)
            print(f"✅ Success: {model_id}")
            await service.cleanup_model(model_id)
        except Exception as e:
            print(f"❌ Failed: {e}")

    await service.shutdown()

asyncio.run(debug_models())
EOF

PYTHONPATH=src python debug_model_loading.py
```

#### 3. Performance Issues

```bash
# Check system performance
cat > check_performance.py << 'EOF'
import psutil
import platform

print("🖥️  System Information:")
print(f"   Platform: {platform.platform()}")
print(f"   Processor: {platform.processor()}")
print(f"   CPU Cores: {psutil.cpu_count()}")
print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"   Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")

# Test data service performance
import asyncio
from benchmark.services.data_service import DataService

async def test_perf():
    service = DataService(enable_hardware_optimization=True)
    await service.initialize()

    stats = await service.get_performance_stats()
    print(f"\n⚡ Data Service Performance:")
    print(f"   Hardware optimization: {stats.get('hardware_optimization_enabled', False)}")

    await service.shutdown()

asyncio.run(test_perf())
EOF

PYTHONPATH=src python check_performance.py
```

#### 4. Dataset Issues

```bash
# Validate dataset format
cat > validate_dataset.py << 'EOF'
import json

def validate_dataset(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        print(f"✅ Dataset loaded: {len(data)} samples")

        # Check required fields
        required_fields = ['content', 'label']
        sample = data[0] if data else {}

        for field in required_fields:
            if field in sample:
                print(f"✅ Field '{field}' present")
            else:
                print(f"❌ Missing field '{field}'")

        # Check label distribution
        labels = [item.get('label') for item in data]
        attack_count = labels.count('ATTACK')
        benign_count = labels.count('BENIGN')

        print(f"📊 Label distribution:")
        print(f"   ATTACK: {attack_count} ({attack_count/len(data)*100:.1f}%)")
        print(f"   BENIGN: {benign_count} ({benign_count/len(data)*100:.1f}%)")

    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")

# Validate generated dataset
validate_dataset("cybersecurity_dataset.json")
EOF

python validate_dataset.py
```

### Getting Help

```bash
# Run system health check
PYTHONPATH=src poetry run pytest tests/unit/test_base_service.py -v

# Check service status
cat > health_check.py << 'EOF'
import asyncio
from benchmark.services.data_service import DataService
from benchmark.services.model_service import ModelService

async def system_health_check():
    print("🏥 System Health Check")

    # Test data service
    try:
        data_service = DataService()
        await data_service.initialize()
        health = await data_service.health_check()
        print(f"📊 Data Service: {health.status}")
        await data_service.shutdown()
    except Exception as e:
        print(f"❌ Data Service: {e}")

    # Test model service
    try:
        model_service = ModelService()
        await model_service.initialize()
        health = await model_service.health_check()
        print(f"🤖 Model Service: {health.status}")
        await model_service.shutdown()
    except Exception as e:
        print(f"❌ Model Service: {e}")

asyncio.run(system_health_check())
EOF

PYTHONPATH=src python health_check.py
```

---

## 🎯 Next Steps

After completing this guide, you can:

1. **Scale Up**: Test with larger datasets (10K+ samples)
2. **Real APIs**: Use actual OpenAI/Anthropic API keys for production testing
3. **Custom Models**: Integrate your own fine-tuned cybersecurity models
4. **Automation**: Set up scheduled evaluations with CI/CD
5. **Advanced Metrics**: Implement domain-specific cybersecurity metrics

### Running the Complete Test Suite

```bash
# Run everything to validate your setup
echo "🚀 Running Complete Validation..."

# 1. Generate dataset
PYTHONPATH=src python generate_dataset.py

# 2. Run first test
PYTHONPATH=src python my_first_test.py

# 3. Run comprehensive evaluation
PYTHONPATH=src python comprehensive_evaluation.py

# 4. Run E2E tests
PYTHONPATH=src poetry run pytest tests/e2e/ -v

echo "✅ Complete validation finished!"
```

You now have a fully functional LLM cybersecurity testing system! 🎉

For additional help, check the documentation files:
- `README.md` - Project overview
- `USER_GUIDE.md` - Detailed usage guide
- `CLI_README.md` - Command-line interface
- `IMPLEMENTATION_DONE.md` - Technical implementation details

# LLM Cybersecurity Benchmark - User Guide

Welcome to the LLM Cybersecurity Benchmark system! This guide will walk you through everything you need to know to use this powerful benchmarking framework, from basic setup to advanced features.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Installation & Setup](#installation--setup)
4. [Configuration Management](#configuration-management)
5. [Database Operations](#database-operations)
6. [Data Generation](#data-generation)
7. [Testing & Validation](#testing--validation)
8. [CI/CD & Automation](#cicd--automation)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11 or 3.12** (required)
- **Poetry** for dependency management
- **Git** for version control
- **macOS** (recommended for MLX support) or Linux

### 1. Clone & Setup
```bash
# Clone the repository
git clone <repository-url>
cd LLM_Benchmark

# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Verify Installation
```bash
# Run a quick test to make sure everything works
poetry run pytest tests/unit/test_data_generators.py -v

# Check if core modules import correctly
poetry run python -c "from benchmark.core.config import ExperimentConfig; print('✅ Setup successful!')"
```

### 3. Generate Your First Test Data
```python
# Create a simple script: test_generation.py
import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path('tests')))

from utils.data_generators import CybersecurityDataGenerator

# Create a data generator
generator = CybersecurityDataGenerator(seed=42)

# Generate some sample data
attack_log = generator.generate_network_log(is_attack=True, attack_type="malware")
benign_log = generator.generate_network_log(is_attack=False)

print("🦠 Attack Log:", attack_log['text'])
print("✅ Benign Log:", benign_log['text'])
print("\n🎉 Congratulations! You're ready to use the system.")
```

---

## 🏗️ System Overview

The LLM Cybersecurity Benchmark consists of several key components:

### Core Components
- **🔧 Configuration System**: Manages experiment settings and validation with advanced performance optimization
- **⚡ Performance Cache**: Advanced LRU caching with memory management and lazy loading
- **🗄️ Database Management**: Handles data storage and retrieval
- **🎲 Data Generators**: Creates realistic cybersecurity test data
- **🧪 Testing Framework**: Comprehensive testing and validation tools
- **🚀 CI/CD Pipeline**: Automated testing and deployment workflows

### Performance Features (NEW!)
- **🚀 Advanced Caching**: LRU cache with automatic eviction and memory management
- **📋 Lazy Loading**: Load only needed configuration sections for faster access
- **🔍 Diff Tracking**: Intelligent change detection to avoid reprocessing
- **📊 Performance Monitoring**: Real-time cache statistics and performance metrics

### What This System Does
This benchmark helps you:
1. **Generate realistic cybersecurity data** for testing LLMs
2. **Configure complex experiments** with multiple models and datasets
3. **Store and manage results** in a robust database
4. **Validate and test** all components thoroughly
5. **Automate workflows** with professional-grade CI/CD

---

## 💾 Installation & Setup

### Step 1: System Requirements
```bash
# Check your Python version (must be 3.11+)
python --version

# Check if Poetry is installed
poetry --version

# If Poetry is not installed:
curl -sSL https://install.python-poetry.org | python3 -
```

### Step 2: Project Setup
```bash
# Clone and enter the project
git clone <your-repo-url>
cd LLM_Benchmark

# Install all dependencies (this may take a few minutes)
poetry install

# Install pre-commit hooks for code quality
poetry run pre-commit install
```

### Step 3: Environment Configuration
```bash
# Create a .env file for API keys (optional)
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
EOF

# Note: These are optional - the system works without them for testing
```

### Step 4: Verify Everything Works
```bash
# Run the test suite to make sure everything is working
poetry run pytest tests/unit/ -v

# This should show all tests passing ✅
```

---

## ⚡ Performance Features

The system now includes advanced performance optimizations for enterprise-scale operations:

### Advanced Configuration Caching

The configuration system includes enterprise-grade caching with LRU eviction and memory management:

```python
from benchmark.services.configuration_service import ConfigurationService
import asyncio

async def use_performance_features():
    """Demonstrate advanced performance features."""

    # Create service with performance optimizations
    service = ConfigurationService(
        cache_ttl=3600,           # Cache entries for 1 hour
        max_cache_size=100,       # Maximum 100 configurations in cache
        max_cache_memory_mb=256,  # Limit cache memory usage to 256MB
        enable_lazy_loading=True  # Enable section-based loading
    )

    await service.initialize()

    # Load configurations (uses advanced caching)
    config = await service.load_experiment_config("my_experiment.yaml")

    # Get performance statistics
    stats = await service.get_cache_performance_stats()
    print(f"📈 Cache hit rate: {stats['advanced_cache']['hit_rate_percent']:.1f}%")
    print(f"💾 Memory usage: {stats['advanced_cache']['memory_usage_mb']:.2f}MB")
    print(f"🔄 Cache entries: {stats['advanced_cache']['current_size']}")

    # Get lightweight configuration outline (very fast)
    outline = await service.get_config_outline("large_config.yaml")
    print(f"📋 Configuration: {outline['name']}")
    print(f"🤖 Models: {outline['_models_count']}")
    print(f"📁 Datasets: {outline['_datasets_count']}")

    # Preload multiple configurations for better performance
    config_paths = ["config1.yaml", "config2.yaml", "config3.yaml"]
    result = await service.preload_configurations_bulk(config_paths)
    print(f"🚀 Preloaded {result.data['success_count']} configurations")

    await service.shutdown()

# Run the example
asyncio.run(use_performance_features())
```

### Performance Monitoring

Monitor cache performance and optimization effectiveness:

```python
import asyncio
from benchmark.services.configuration_service import ConfigurationService

async def monitor_performance():
    """Monitor configuration service performance."""

    service = ConfigurationService(enable_lazy_loading=True)
    await service.initialize()

    # Load several configurations
    for i in range(5):
        await service.load_experiment_config(f"config_{i}.yaml")

    # Get comprehensive performance statistics
    stats = await service.get_cache_performance_stats()

    print("📊 Performance Report:")
    print("=" * 40)

    # Advanced cache statistics
    cache_stats = stats['advanced_cache']
    print(f"🚀 Advanced Cache:")
    print(f"   Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"   Hits: {cache_stats['hits']}")
    print(f"   Misses: {cache_stats['misses']}")
    print(f"   Memory: {cache_stats['memory_usage_mb']:.2f}MB")
    print(f"   Evictions: {cache_stats['evictions']}")

    # Lazy loader statistics (if enabled)
    if 'lazy_loader' in stats:
        lazy_stats = stats['lazy_loader']
        print(f"📋 Lazy Loader:")
        print(f"   Cached Files: {lazy_stats['cached_files']}")
        print(f"   Total Sections: {lazy_stats['total_sections']}")

    await service.shutdown()

asyncio.run(monitor_performance())
```

### Performance Testing

Run comprehensive performance tests:

```bash
# Run all performance tests
poetry run pytest tests/performance/ -v

# Run specific performance test categories
poetry run pytest tests/performance/test_config_performance.py::TestConfigurationLoadingPerformance -v
poetry run pytest tests/performance/test_config_performance.py::TestCachePerformance -v
poetry run pytest tests/performance/test_config_performance.py::TestMemoryEfficiency -v

# Run cache-specific unit tests
poetry run pytest tests/unit/test_config_caching.py -v

# Run the interactive performance demonstration
poetry run python demo_performance.py
```

### Performance Best Practices

To get the best performance from the system:

1. **Enable Lazy Loading**: Always enable lazy loading for large configurations
2. **Configure Memory Limits**: Set appropriate memory limits based on your system
3. **Use Bulk Preloading**: Preload multiple configurations when possible
4. **Monitor Cache Performance**: Regularly check cache hit rates and memory usage
5. **Optimize Cache Size**: Tune cache size based on your usage patterns

```python
# Optimal configuration for large-scale operations
service = ConfigurationService(
    cache_ttl=7200,              # 2 hours cache
    max_cache_size=200,          # Large cache for many configurations
    max_cache_memory_mb=512,     # 512MB memory limit
    enable_lazy_loading=True     # Always enable for performance
)
```

---

## ⚙️ Configuration Management

The configuration system uses YAML files to define experiments. Here's how to create and use configurations:

### Basic Configuration Structure

Create a file called `my_experiment.yaml`:

```yaml
experiment:
  name: "My First Cybersecurity Benchmark"
  description: "Testing LLMs on cybersecurity tasks"
  output_dir: "./results"

datasets:
  - name: "sample_dataset"
    source: "local"
    path: "./data/samples.jsonl"
    max_samples: 100
    test_split: 0.2
    validation_split: 0.1

models:
  - name: "gpt-3.5-turbo"
    type: "openai_api"
    path: "gpt-3.5-turbo"
    config:
      api_key: "${OPENAI_API_KEY:test_key}"
    max_tokens: 512
    temperature: 0.1

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score"]
  parallel_jobs: 2
  timeout_minutes: 30
  batch_size: 16
```

### Loading and Validating Configurations

```python
from benchmark.core.config import ExperimentConfig
import yaml

def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate an experiment configuration."""

    # Load the YAML file
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Create and validate the configuration
    try:
        config = ExperimentConfig(**config_data)
        print(f"✅ Configuration loaded successfully!")
        print(f"📊 Experiment: {config.experiment.name}")
        print(f"🤖 Models: {len(config.models)}")
        print(f"📁 Datasets: {len(config.datasets)}")
        return config
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        raise

# Usage
config = load_config("my_experiment.yaml")
```

### Configuration Components Explained

#### Dataset Configuration
```yaml
datasets:
  - name: "my_dataset"           # Unique identifier
    source: "local"              # "local" or "remote"
    path: "/path/to/data.jsonl"  # File path
    max_samples: 1000            # Limit number of samples
    test_split: 0.2              # 20% for testing
    validation_split: 0.1        # 10% for validation
    preprocessing:               # Optional preprocessing steps
      - "tokenize"
      - "normalize"
```

#### Model Configuration
```yaml
models:
  # OpenAI API Model
  - name: "gpt-4"
    type: "openai_api"
    path: "gpt-4"
    config:
      api_key: "${OPENAI_API_KEY}"
    max_tokens: 1024
    temperature: 0.0

  # Anthropic API Model
  - name: "claude-3"
    type: "anthropic_api"
    path: "claude-3-haiku-20240307"
    config:
      api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 512

  # Local Model (future support)
  - name: "local-llama"
    type: "local"
    path: "/path/to/model"
```

#### Evaluation Configuration
```yaml
evaluation:
  metrics:                    # Which metrics to calculate
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
  parallel_jobs: 4           # Number of parallel processes
  timeout_minutes: 60        # Max time per evaluation
  batch_size: 32            # Samples per batch
```

---

## 🗄️ Database Operations

The system includes a powerful async database system for storing experiment results.

### Basic Database Usage

```python
import asyncio
from benchmark.core.database_manager import DatabaseManager

async def database_example():
    """Basic database operations example."""

    # Create database manager
    db_manager = DatabaseManager("sqlite+aiosqlite:///my_benchmark.db")

    try:
        # Initialize the database
        await db_manager.initialize()
        print("✅ Database connected!")

        # Create all tables
        await db_manager.create_tables()
        print("✅ Database tables created!")

        # Use database session
        async with db_manager.session_scope() as session:
            # Your database operations go here
            print("✅ Database session active!")

        print("✅ Database operations completed!")

    finally:
        # Always close the database connection
        await db_manager.close()
        print("✅ Database connection closed!")

# Run the example
asyncio.run(database_example())
```

### Advanced Database Operations

```python
from benchmark.core.database import Experiment, Dataset, Model
from sqlalchemy import select

async def advanced_database_example():
    """More advanced database operations."""

    db_manager = DatabaseManager("sqlite+aiosqlite:///benchmark.db")
    await db_manager.initialize()
    await db_manager.create_tables()

    try:
        async with db_manager.session_scope() as session:
            # Create a new experiment record
            experiment = Experiment(
                name="Test Experiment",
                description="My first benchmark experiment",
                config_data={"test": True}
            )
            session.add(experiment)
            await session.commit()

            print(f"✅ Created experiment: {experiment.name}")

            # Query experiments
            result = await session.execute(
                select(Experiment).where(Experiment.name == "Test Experiment")
            )
            found_experiment = result.scalar_one_or_none()

            if found_experiment:
                print(f"📊 Found experiment: {found_experiment.name}")
                print(f"📅 Created at: {found_experiment.created_at}")

    finally:
        await db_manager.close()

# Run the advanced example
asyncio.run(advanced_database_example())
```

### Database Configuration Options

```python
# SQLite (for development and testing)
db_manager = DatabaseManager("sqlite+aiosqlite:///benchmark.db")

# PostgreSQL (for production)
db_manager = DatabaseManager("postgresql+asyncpg://user:password@localhost/benchmark")

# MySQL (alternative option)
db_manager = DatabaseManager("mysql+aiomysql://user:password@localhost/benchmark")
```

---

## 🎲 Data Generation

One of the most powerful features is the realistic cybersecurity data generator.

### Getting Started with Data Generation

```python
import sys
from pathlib import Path

# Add tests directory to path for imports
sys.path.insert(0, str(Path('tests')))

from utils.data_generators import CybersecurityDataGenerator

def basic_data_generation():
    """Generate basic cybersecurity test data."""

    # Create generator with fixed seed for reproducible results
    generator = CybersecurityDataGenerator(seed=42)

    print("🎲 Generating cybersecurity test data...\n")

    # Generate a malware attack log
    malware_log = generator.generate_network_log(
        is_attack=True,
        attack_type="malware"
    )

    print("🦠 MALWARE ATTACK LOG:")
    print(f"   Text: {malware_log['text']}")
    print(f"   Severity: {malware_log['severity']}")
    print(f"   Confidence: {malware_log['confidence']:.2f}")
    print()

    # Generate a normal network log
    benign_log = generator.generate_network_log(is_attack=False)

    print("✅ BENIGN NETWORK LOG:")
    print(f"   Text: {benign_log['text']}")
    print(f"   Confidence: {benign_log['confidence']:.2f}")
    print()

    # Generate a phishing email
    phishing_email = generator.generate_email_sample(
        is_phishing=True,
        phishing_type="spear_phishing"
    )

    print("🎣 PHISHING EMAIL:")
    print(f"   Subject: {phishing_email['subject']}")
    print(f"   Sender: {phishing_email['sender']}")
    print(f"   Attack Type: {phishing_email['attack_subtype']}")
    print()

# Run the example
basic_data_generation()
```

### Attack Types Supported

The system supports many types of cybersecurity attacks:

```python
def explore_attack_types():
    """Explore all supported attack types."""

    generator = CybersecurityDataGenerator(seed=123)

    # All supported attack types
    attack_types = {
        "malware": ["trojan", "virus", "worm", "ransomware", "backdoor", "rootkit"],
        "intrusion": ["unauthorized_access", "privilege_escalation", "lateral_movement"],
        "dos": ["flooding", "amplification", "resource_exhaustion", "slowloris"],
        "phishing": ["spear_phishing", "whaling", "smishing", "vishing"],
        "injection": ["sql_injection", "xss", "command_injection", "ldap_injection"],
        "reconnaissance": ["port_scan", "network_scan", "vulnerability_scan", "enumeration"]
    }

    print("🎯 Generating examples of each attack type:\n")

    for attack_type, subtypes in attack_types.items():
        print(f"📍 {attack_type.upper()} ATTACKS:")

        # Generate an example for this attack type
        log = generator.generate_network_log(is_attack=True, attack_type=attack_type)

        print(f"   Subtype: {log['attack_subtype']}")
        print(f"   Text: {log['text'][:80]}...")
        print(f"   Severity: {log['severity']}")
        print()

explore_attack_types()
```

### Batch Data Generation

Generate large datasets efficiently:

```python
def generate_dataset():
    """Generate a complete dataset for training/testing."""

    generator = CybersecurityDataGenerator(seed=456)

    print("📦 Generating a large batch of samples...\n")

    # Generate 1000 samples with 30% attacks
    samples = generator.generate_batch_samples(
        num_samples=1000,
        attack_ratio=0.3,  # 30% attacks, 70% benign
        attack_types=["malware", "phishing", "dos"]  # Only these attack types
    )

    # Analyze the generated data
    attack_count = sum(1 for s in samples if s['label'] == 'ATTACK')
    benign_count = len(samples) - attack_count

    print(f"📊 Generated {len(samples)} total samples:")
    print(f"   🦠 Attack samples: {attack_count} ({attack_count/len(samples)*100:.1f}%)")
    print(f"   ✅ Benign samples: {benign_count} ({benign_count/len(samples)*100:.1f}%)")

    # Show attack type distribution
    attack_samples = [s for s in samples if s['label'] == 'ATTACK']
    attack_type_counts = {}
    for sample in attack_samples:
        attack_type = sample.get('attack_type', 'unknown')
        attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1

    print(f"\n🎯 Attack type distribution:")
    for attack_type, count in attack_type_counts.items():
        print(f"   {attack_type}: {count} samples")

    return samples

# Generate and save dataset
dataset = generate_dataset()

# Save to file (optional)
import json
with open('generated_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)
print("\n💾 Dataset saved to 'generated_dataset.json'")
```

### Model Prediction Simulation

Generate realistic model predictions for testing:

```python
def simulate_model_predictions():
    """Generate realistic model predictions."""

    generator = CybersecurityDataGenerator(seed=789)

    print("🤖 Simulating model predictions...\n")

    # Generate predictions with different accuracy levels
    accuracy_levels = [0.95, 0.85, 0.75, 0.60]

    for accuracy in accuracy_levels:
        print(f"📊 Model with {accuracy*100:.0f}% accuracy:")

        # Generate 20 predictions
        predictions = []
        for i in range(20):
            ground_truth = "ATTACK" if i % 2 == 0 else "BENIGN"
            pred = generator.generate_model_prediction(ground_truth, accuracy=accuracy)
            predictions.append(pred)

        # Calculate actual accuracy
        correct = sum(1 for p in predictions if p['is_correct'])
        actual_accuracy = correct / len(predictions)

        print(f"   Target accuracy: {accuracy*100:.0f}%")
        print(f"   Actual accuracy: {actual_accuracy*100:.0f}%")

        # Show some example predictions
        print("   Example predictions:")
        for i, pred in enumerate(predictions[:3]):
            print(f"     {i+1}. Predicted: {pred['prediction']}, "
                  f"Actual: {pred['ground_truth']}, "
                  f"Confidence: {pred['confidence']:.2f}, "
                  f"Correct: {'✅' if pred['is_correct'] else '❌'}")
        print()

simulate_model_predictions()
```

---

## 🧪 Testing & Validation

The system includes comprehensive testing tools to validate everything works correctly.

### Running Tests

```bash
# Run all unit tests
poetry run pytest tests/unit/ -v

# Run tests with coverage report
poetry run pytest tests/unit/ --cov=src/benchmark --cov-report=html

# Run specific test files
poetry run pytest tests/unit/test_data_generators.py -v
poetry run pytest tests/unit/test_config.py -v

# Run integration tests (if available)
poetry run pytest tests/integration/ -v

# Run performance tests
poetry run pytest tests/unit/test_data_generators.py::TestPerformanceDataGeneration -v
```

### Test Categories

#### Unit Tests
Test individual components in isolation:
```bash
# Test configuration system
poetry run pytest tests/unit/test_config.py -v

# Test database operations
poetry run pytest tests/unit/test_database_manager.py -v

# Test data generators
poetry run pytest tests/unit/test_data_generators.py -v
```

#### Integration Tests
Test how components work together:
```bash
# Test database integration with configuration
poetry run pytest tests/integration/test_database_integration.py -v
```

#### Performance Tests
Validate system performance:
```bash
# Test data generation speed
poetry run pytest tests/unit/test_data_generators.py::TestPerformanceDataGeneration::test_performance_data_generation -v
```

### Writing Your Own Tests

Create a test file `tests/unit/test_my_feature.py`:

```python
import pytest
from benchmark.core.config import ExperimentConfig

class TestMyFeature:
    """Test my custom feature."""

    def test_basic_functionality(self):
        """Test that basic functionality works."""
        # Your test code here
        assert True  # Replace with real test

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        # Your async test code here
        assert True  # Replace with real test

    def test_with_fixture(self, sample_experiment_config):
        """Test using a fixture from conftest.py."""
        assert sample_experiment_config.experiment.name is not None
```

### Using Test Fixtures

The system provides many useful fixtures:

```python
def test_database_operations(db_session):
    """Test using database fixture."""
    # db_session is automatically provided and cleaned up
    pass

def test_with_temp_directory(temp_dir):
    """Test using temporary directory."""
    # temp_dir is automatically created and cleaned up
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello, World!")
    assert test_file.exists()

def test_data_generation(cybersec_data_generator):
    """Test using data generator fixture."""
    # Generator is automatically created with seed=42
    log = cybersec_data_generator.generate_network_log(is_attack=True)
    assert log['label'] == 'ATTACK'
```

---

## 🚀 CI/CD & Automation

The system includes professional-grade GitHub Actions workflows for automated testing and deployment.

### Understanding the Workflows

#### 1. CI Workflow (`.github/workflows/ci.yml`)
Runs on every push and pull request:
```bash
# What it does:
# ✅ Code quality checks (linting, formatting, type checking)
# ✅ Unit tests with coverage reporting
# ✅ Data generator validation
# ✅ Security scanning

# Trigger it manually:
gh workflow run ci.yml
```

#### 2. Integration Tests (`.github/workflows/tests.yml`)
Comprehensive testing:
```bash
# What it does:
# ✅ Integration tests (database, config, components)
# ✅ End-to-end tests (full system simulation)
# ✅ Performance tests (speed benchmarks)
# ✅ MLX compatibility tests (Apple Silicon)

# Trigger it manually with options:
gh workflow run tests.yml -f test_type=integration
gh workflow run tests.yml -f test_type=performance
gh workflow run tests.yml -f test_type=all
```

#### 3. Security Scanning (`.github/workflows/security.yml`)
Weekly security audits:
```bash
# What it does:
# 🔒 Vulnerability scanning (Safety, Bandit, Semgrep)
# 🔒 Secret detection
# 🔒 License compliance checking
# 🔒 Dependency security audit

# Trigger it manually:
gh workflow run security.yml
```

#### 4. Dependency Management (`.github/workflows/dependencies.yml`)
Automated dependency updates:
```bash
# What it does:
# 📦 Weekly dependency health checks
# 📦 Automated security vulnerability reporting
# 📦 Creates PRs for dependency updates

# Trigger manual dependency updates:
gh workflow run dependencies.yml -f update_type=minor
gh workflow run dependencies.yml -f update_type=major
```

#### 5. Release & Documentation (`.github/workflows/release.yml`)
Automated releases:
```bash
# What it does:
# 🚀 Full test suite validation
# 🚀 Documentation generation
# 🚀 GitHub release creation
# 🚀 PyPI publishing (if configured)

# Create a new release:
gh workflow run release.yml -f release_type=patch
gh workflow run release.yml -f release_type=minor
```

### Local Development Workflow

Before pushing code, run these checks locally:

```bash
# 1. Code formatting and linting
poetry run ruff format src/ tests/
poetry run ruff check src/ tests/

# 2. Type checking
poetry run mypy src/ tests/

# 3. Run tests
poetry run pytest tests/unit/ -v

# 4. Security scanning
poetry run bandit -r src/
poetry run safety check

# 5. Pre-commit hooks (runs automatically on commit)
poetry run pre-commit run --all-files
```

### Monitoring Workflows

Check workflow status:
```bash
# List recent workflow runs
gh run list

# Watch a running workflow
gh run watch

# View workflow logs
gh run view [run-id] --log
```

---

## 🔧 Troubleshooting

Common issues and solutions:

### Installation Issues

#### Poetry Installation Problems
```bash
# If Poetry installation fails:
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (add to ~/.bashrc or ~/.zshrc):
export PATH="$HOME/.local/bin:$PATH"

# Verify Poetry installation:
poetry --version
```

#### Python Version Issues
```bash
# Check Python version (must be 3.11+):
python --version

# If you have multiple Python versions, specify the correct one:
poetry env use python3.11

# Or use pyenv to manage Python versions:
pyenv install 3.11.0
pyenv local 3.11.0
```

#### Dependency Installation Issues
```bash
# Clear Poetry cache and reinstall:
poetry cache clear --all pypi
poetry install

# If you get permission errors:
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

### Runtime Issues

#### Import Errors
```bash
# If you get import errors, make sure you're in the Poetry shell:
poetry shell

# Or run commands with poetry run:
poetry run python your_script.py

# Check if the package is installed correctly:
poetry run python -c "import benchmark.core.config; print('✅ Imports work!')"
```

#### Database Connection Issues
```python
# For SQLite issues, make sure the directory exists:
from pathlib import Path
Path("./data").mkdir(exist_ok=True)

# Test database connection:
import asyncio
from benchmark.core.database_manager import DatabaseManager

async def test_db():
    db = DatabaseManager("sqlite+aiosqlite:///./data/test.db")
    try:
        await db.initialize()
        print("✅ Database connection works!")
    finally:
        await db.close()

asyncio.run(test_db())
```

#### Data Generation Issues
```python
# If data generation fails, check the path setup:
import sys
from pathlib import Path

# Make sure tests directory is in path:
tests_dir = Path(__file__).parent / "tests"
if tests_dir.exists():
    sys.path.insert(0, str(tests_dir))
else:
    print(f"❌ Tests directory not found: {tests_dir}")

# Test data generation:
try:
    from utils.data_generators import CybersecurityDataGenerator
    generator = CybersecurityDataGenerator(seed=42)
    log = generator.generate_network_log()
    print("✅ Data generation works!")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

### Testing Issues

#### Test Failures
```bash
# Run tests in verbose mode to see detailed output:
poetry run pytest tests/unit/ -v -s

# Run a specific failing test:
poetry run pytest tests/unit/test_config.py::TestExperimentConfig::test_valid_config -v -s

# Run tests with Python debugging:
poetry run pytest tests/unit/ --pdb
```

#### Coverage Issues
```bash
# Generate detailed coverage report:
poetry run pytest tests/unit/ --cov=src/benchmark --cov-report=html --cov-report=term-missing

# View coverage report:
open htmlcov/index.html
```

### CI/CD Issues

#### Workflow Failures
```bash
# Check workflow status:
gh run list

# View detailed logs:
gh run view [run-id] --log

# Re-run failed workflows:
gh run rerun [run-id]
```

#### Apple Silicon / MLX Issues
```bash
# MLX is only available on Apple Silicon Macs
# If you're not on Apple Silicon, MLX imports may fail (this is expected)

# Test if MLX is available:
poetry run python -c "
try:
    import mlx.core as mx
    print('✅ MLX is available')
except ImportError:
    print('ℹ️ MLX not available (expected on non-Apple Silicon)')
"
```

### Getting Help

If you're still having issues:

1. **Check the logs**: Always look at the detailed error messages
2. **Search the issues**: Check if someone else has had the same problem
3. **Create a minimal example**: Try to reproduce the issue with the smallest possible code
4. **Check dependencies**: Make sure all dependencies are installed correctly

```bash
# Generate a system info report:
poetry run python -c "
import sys
import platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')
"

poetry show  # Show all installed packages
```

---

## 🚀 Advanced Usage

### Custom Attack Types

Add your own attack types to the data generator:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tests')))

from utils.data_generators import CybersecurityDataGenerator

class CustomDataGenerator(CybersecurityDataGenerator):
    """Extended data generator with custom attack types."""

    def __init__(self, seed=None):
        super().__init__(seed)

        # Add custom attack types
        self.ATTACK_TYPES.update({
            "cryptojacking": ["browser_mining", "malware_mining", "cloud_mining"],
            "supply_chain": ["dependency_confusion", "package_substitution", "build_compromise"]
        })

    def generate_custom_attack_log(self, attack_type: str, attack_subtype: str):
        """Generate custom attack log."""

        if attack_type == "cryptojacking":
            return {
                "timestamp": self.generate_timestamp(),
                "src_ip": self.generate_ip_address(private=False),
                "dst_ip": self.generate_ip_address(private=True),
                "text": f"Cryptocurrency mining detected: {attack_subtype}",
                "label": "ATTACK",
                "attack_type": attack_type,
                "attack_subtype": attack_subtype,
                "severity": "MEDIUM",
                "confidence": self._random.uniform(0.7, 0.9),
                "additional_data": {
                    "mining_pool": self._random.choice(["pool1.mining.com", "cryptopool.net"]),
                    "cpu_usage": self._random.uniform(80, 100),
                    "network_usage": self._random.uniform(10, 50)
                }
            }

        # Fall back to parent class for other types
        return super().generate_network_log(is_attack=True, attack_type=attack_type)

# Usage
custom_generator = CustomDataGenerator(seed=42)
cryptojacking_log = custom_generator.generate_custom_attack_log("cryptojacking", "browser_mining")
print(cryptojacking_log)
```

### Advanced Configuration

Create complex experiment configurations:

```python
from benchmark.core.config import ExperimentConfig, DatasetConfig, ModelConfig, EvaluationConfig
import yaml

def create_advanced_config():
    """Create a comprehensive experiment configuration."""

    config = {
        "experiment": {
            "name": "Large Scale Cybersecurity Evaluation",
            "description": "Comprehensive evaluation across multiple models and attack types",
            "output_dir": "./experiments/large_scale",
            "tags": ["cybersecurity", "comprehensive", "multi-model"]
        },

        "datasets": [
            {
                "name": "network_attacks",
                "source": "local",
                "path": "./data/network_logs.jsonl",
                "max_samples": 10000,
                "test_split": 0.2,
                "validation_split": 0.1,
                "preprocessing": ["normalize", "tokenize"],
                "filters": {
                    "attack_types": ["malware", "intrusion", "dos"],
                    "severity": ["HIGH", "CRITICAL"]
                }
            },
            {
                "name": "email_phishing",
                "source": "local",
                "path": "./data/emails.jsonl",
                "max_samples": 5000,
                "test_split": 0.15,
                "validation_split": 0.15,
                "preprocessing": ["clean_html", "extract_urls"]
            }
        ],

        "models": [
            {
                "name": "gpt-4-turbo",
                "type": "openai_api",
                "path": "gpt-4-1106-preview",
                "config": {
                    "api_key": "${OPENAI_API_KEY}",
                    "organization": "${OPENAI_ORG_ID:}"
                },
                "max_tokens": 2048,
                "temperature": 0.0,
                "system_prompt": "You are a cybersecurity expert. Analyze the given data and classify it as either ATTACK or BENIGN."
            },
            {
                "name": "claude-3-opus",
                "type": "anthropic_api",
                "path": "claude-3-opus-20240229",
                "config": {
                    "api_key": "${ANTHROPIC_API_KEY}"
                },
                "max_tokens": 1024,
                "temperature": 0.1
            },
            {
                "name": "llama-2-70b",
                "type": "local",
                "path": "./models/llama-2-70b-chat",
                "config": {
                    "device": "cuda",
                    "precision": "fp16"
                },
                "max_tokens": 512,
                "temperature": 0.2
            }
        ],

        "evaluation": {
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "confusion_matrix"
            ],
            "parallel_jobs": 8,
            "timeout_minutes": 120,
            "batch_size": 64,
            "cross_validation": {
                "folds": 5,
                "strategy": "stratified"
            },
            "statistical_tests": {
                "significance_level": 0.05,
                "correction": "bonferroni"
            }
        }
    }

    # Save configuration
    with open("advanced_experiment.yaml", "w") as f:
        yaml.dump(config, f, indent=2, default_flow_style=False)

    # Validate configuration
    validated_config = ExperimentConfig(**config)
    print("✅ Advanced configuration created and validated!")

    return validated_config

# Create the configuration
advanced_config = create_advanced_config()
```

### Performance Optimization

Optimize data generation for large-scale experiments:

```python
import sys
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path('tests')))
from utils.data_generators import CybersecurityDataGenerator

def generate_batch_worker(args):
    """Worker function for parallel data generation."""
    seed, batch_size, attack_ratio, worker_id = args

    # Create generator with unique seed per worker
    generator = CybersecurityDataGenerator(seed=seed + worker_id)

    # Generate batch
    samples = generator.generate_batch_samples(
        num_samples=batch_size,
        attack_ratio=attack_ratio
    )

    return samples

def generate_large_dataset_parallel(total_samples=100000, attack_ratio=0.3, num_workers=None):
    """Generate a large dataset using parallel processing."""

    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"🚀 Generating {total_samples:,} samples using {num_workers} workers...")

    # Calculate batch size per worker
    batch_size = total_samples // num_workers

    # Prepare worker arguments
    worker_args = [
        (42, batch_size, attack_ratio, worker_id)
        for worker_id in range(num_workers)
    ]

    # Handle remainder samples
    remainder = total_samples % num_workers
    if remainder > 0:
        worker_args.append((42, remainder, attack_ratio, num_workers))

    start_time = time.time()

    # Generate data in parallel
    all_samples = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(generate_batch_worker, worker_args)

        for batch in results:
            all_samples.extend(batch)

    generation_time = time.time() - start_time

    print(f"✅ Generated {len(all_samples):,} samples in {generation_time:.2f} seconds")
    print(f"📊 Generation rate: {len(all_samples)/generation_time:.0f} samples/second")

    # Analyze results
    attack_count = sum(1 for s in all_samples if s['label'] == 'ATTACK')
    print(f"🎯 Attack ratio: {attack_count/len(all_samples)*100:.1f}%")

    return all_samples

# Generate large dataset
large_dataset = generate_large_dataset_parallel(
    total_samples=50000,
    attack_ratio=0.4,
    num_workers=4
)

# Save dataset
import json
print("💾 Saving dataset...")
with open('large_dataset.json', 'w') as f:
    json.dump(large_dataset, f)
print("✅ Dataset saved!")
```

### Integration with External Tools

Integrate with popular ML frameworks:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
from pathlib import Path

sys.path.insert(0, str(Path('tests')))
from utils.data_generators import CybersecurityDataGenerator

def create_ml_dataset():
    """Create a dataset ready for machine learning."""

    generator = CybersecurityDataGenerator(seed=42)

    # Generate samples
    samples = generator.generate_batch_samples(
        num_samples=5000,
        attack_ratio=0.4
    )

    # Convert to pandas DataFrame
    df = pd.DataFrame(samples)

    # Feature engineering
    df['text_length'] = df['text'].str.len()
    df['has_ip'] = df['text'].str.contains(r'\d+\.\d+\.\d+\.\d+', na=False)
    df['has_port'] = df['text'].str.contains(r':\d+', na=False)
    df['severity_numeric'] = df['severity'].map({
        'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4
    })

    # Create binary target
    df['is_attack'] = (df['label'] == 'ATTACK').astype(int)

    print("📊 Dataset Statistics:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Attack samples: {df['is_attack'].sum():,}")
    print(f"   Features: {df.shape[1]}")
    print("\n📈 Label distribution:")
    print(df['label'].value_counts())

    # Split dataset
    X = df[['text_length', 'has_ip', 'has_port', 'severity_numeric', 'confidence']]
    y = df['is_attack']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\n🔄 Data split:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")

    return df, X_train, X_test, y_train, y_test

# Create ML-ready dataset
df, X_train, X_test, y_train, y_test = create_ml_dataset()

# Train a simple classifier (example)
try:
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n🤖 Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))

except ImportError:
    print("📦 Install scikit-learn to run the ML example: pip install scikit-learn")
```

This comprehensive user guide should help anyone, from beginners to advanced users, understand and effectively use all the features of the LLM Cybersecurity Benchmark system. Each section includes practical examples and detailed explanations to ensure users can successfully implement and use the system for their cybersecurity benchmarking needs.

---

## 🎓 Conclusion

You now have a comprehensive understanding of the LLM Cybersecurity Benchmark system! This guide covered:

- ✅ **Quick setup and installation**
- ✅ **Advanced performance features with caching and lazy loading**
- ✅ **Configuration management with YAML files**
- ✅ **Database operations for storing results**
- ✅ **Realistic cybersecurity data generation**
- ✅ **Comprehensive testing and validation**
- ✅ **Professional CI/CD automation**
- ✅ **Performance monitoring and optimization**
- ✅ **Troubleshooting common issues**
- ✅ **Advanced customization and optimization**

The system is production-ready with enterprise-grade performance optimizations and designed to scale with your needs. Whether you're a researcher, security professional, or developer, you now have the tools to benchmark LLMs on cybersecurity tasks effectively and efficiently.

Happy benchmarking! 🚀

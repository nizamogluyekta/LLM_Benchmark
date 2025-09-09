# 🚀 LLM Cybersecurity Benchmark CLI

A comprehensive command-line interface for managing benchmark configurations with beautiful output formatting and interactive features.

## 🛠️ Installation

```bash
# Install dependencies
pip install click rich pyyaml

# Add to Python path or install package
export PYTHONPATH="${PYTHONPATH}:/path/to/LLM_Benchmark/src"
```

## 📋 Commands Overview

### `benchmark config validate <config_file>`

Validates configuration files with comprehensive analysis including:
- Configuration structure and syntax validation
- Model compatibility and API key format checking
- Dataset availability and format validation
- Resource requirement analysis
- Performance optimization recommendations
- Cross-field consistency checks

**Options:**
- `--quiet, -q`: Show only errors and critical warnings
- `--json-output`: Output validation results as JSON

**Examples:**
```bash
# Validate with full output
benchmark config validate config.yaml

# Quiet mode (only errors)
benchmark config validate config.yaml --quiet

# JSON output for automation
benchmark config validate config.yaml --json-output
```

### `benchmark config generate`

Generates sample configuration files with customization options.

**Options:**
- `--output, -o`: Output file name (default: config.yaml)
- `--interactive, -i`: Interactive configuration generation
- `--format`: Output format (yaml/json, default: yaml)

**Examples:**
```bash
# Generate basic configuration
benchmark config generate

# Interactive generation
benchmark config generate --interactive

# Generate JSON format
benchmark config generate --format json --output config.json

# Custom output file
benchmark config generate --output my-experiment.yaml
```

### `benchmark config show <config_file>`

Displays parsed configurations with syntax highlighting and formatting.

**Options:**
- `--format`: Output format (yaml/json, auto-detect if not specified)
- `--pretty`: Pretty print with syntax highlighting (default: true)

**Examples:**
```bash
# Show with syntax highlighting
benchmark config show config.yaml

# Force JSON format
benchmark config show config.yaml --format json

# Plain output (no colors/formatting)
benchmark config show config.yaml --no-pretty
```

### `benchmark config check-env <config_file>`

Checks environment variable requirements and helps set missing variables.

**Options:**
- `--set-missing`: Interactively prompt to set missing environment variables

**Examples:**
```bash
# Check environment variables
benchmark config check-env config.yaml

# Interactive setup of missing variables
benchmark config check-env config.yaml --set-missing
```

## 🎨 Output Features

### Rich Formatting
- **Color-coded output**: Errors (red), warnings (yellow), info (blue), success (green)
- **Syntax highlighting**: YAML and JSON with proper formatting
- **Progress indicators**: Clear status symbols (✓, ✗, ▲, i)
- **Tables and panels**: Organized information display

### Validation Categories
- **🚨 Errors**: Critical issues that must be fixed
- **⚠️ Critical**: System-breaking problems
- **⚡ Warnings**: Issues that may affect performance
- **💡 Recommendations**: Optimization suggestions

### Interactive Features
- **Smart prompts**: Context-aware input validation
- **Confirmation dialogs**: Safe operations with user confirmation
- **Progress feedback**: Real-time status updates
- **Error suggestions**: Actionable guidance for fixing issues

## 📝 Configuration Examples

### Basic Configuration
```yaml
name: "My Cybersecurity Experiment"
description: "Testing LLM security capabilities"
output_dir: "./results"

datasets:
  - name: "phishing_detection"
    source: "local"
    path: "./data/phishing_samples.jsonl"
    max_samples: 1000
    test_split: 0.2
    validation_split: 0.1

models:
  - name: "gpt-3.5-turbo"
    type: "openai_api"
    path: "gpt-3.5-turbo"
    config:
      api_key: "${OPENAI_API_KEY}"
    max_tokens: 1024
    temperature: 0.1

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score"]
  parallel_jobs: 4
  timeout_minutes: 30
  batch_size: 32
```

### Environment Variables
```bash
# Required environment variables
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"

# Optional configuration variables
export EVAL_BATCH_SIZE="64"
export EVAL_TIMEOUT="60"
export MAX_SAMPLES="500"
```

## 🔧 Troubleshooting

### Common Issues

**Configuration Not Found**
```bash
benchmark config validate /path/to/config.yaml
# Error: Path '/path/to/config.yaml' does not exist.
```
*Solution*: Check file path and ensure file exists.

**Invalid YAML Syntax**
```bash
benchmark config validate invalid.yaml
# Error: Invalid YAML in configuration file
```
*Solution*: Use `benchmark config show` to identify syntax issues.

**Missing Environment Variables**
```bash
benchmark config check-env config.yaml
# Missing Variables (2):
#   ● OPENAI_API_KEY
#   ● ANTHROPIC_API_KEY
```
*Solution*: Set required variables or use `--set-missing` flag.

**API Key Format Issues**
```bash
benchmark config validate config.yaml
# [WARNING] Model gpt-3.5 API key format appears invalid
```
*Solution*: Ensure OpenAI keys start with `sk-` and Anthropic keys with `sk-ant-`.

### Debugging Tips

1. **Use quiet mode** to focus on critical issues:
   ```bash
   benchmark config validate config.yaml --quiet
   ```

2. **Check environment setup** first:
   ```bash
   benchmark config check-env config.yaml
   ```

3. **Generate working examples**:
   ```bash
   benchmark config generate --interactive
   ```

4. **Use JSON output** for automation:
   ```bash
   benchmark config validate config.yaml --json-output | jq .
   ```

## 🚀 Quick Start Workflow

1. **Generate a configuration**:
   ```bash
   benchmark config generate --interactive
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   ```

3. **Validate configuration**:
   ```bash
   benchmark config validate config.yaml
   ```

4. **Check environment setup**:
   ```bash
   benchmark config check-env config.yaml
   ```

5. **Review final configuration**:
   ```bash
   benchmark config show config.yaml
   ```

## 🎯 Best Practices

- **Always validate** configurations before running experiments
- **Use environment variables** for sensitive data like API keys
- **Start with interactive generation** for new configurations
- **Check environment setup** in new environments
- **Use quiet mode** in CI/CD pipelines
- **Save configurations** in version control (without secrets)

## ⚡ Performance Features

### Advanced Configuration Caching
The CLI now leverages advanced performance optimizations:
- **LRU Cache**: Memory-efficient configuration caching with automatic eviction
- **Lazy Loading**: Section-based loading for faster access to large configurations
- **Diff Tracking**: Intelligent change detection to avoid unnecessary reprocessing
- **Memory Management**: Configurable memory limits with real-time monitoring

### Performance Commands

#### `benchmark config cache-stats <config_file>`
View configuration cache performance statistics:

```bash
# View cache performance metrics
benchmark config cache-stats config.yaml

# Example output:
# 📊 Cache Performance Statistics:
#   ✅ Advanced cache enabled: True
#   📈 Hit rate: 85.2%
#   💾 Memory usage: 12.3MB / 256MB (4.8%)
#   🔄 Cache entries: 15 / 100
#   ⚡ Avg access time: 2.1ms
```

#### `benchmark config preload <config_files...>`
Preload multiple configurations for better performance:

```bash
# Preload configurations for faster access
benchmark config preload config1.yaml config2.yaml config3.yaml

# Bulk preload from directory
benchmark config preload configs/*.yaml
```

#### `benchmark config outline <config_file>`
Get lightweight configuration outline without full loading:

```bash
# Fast configuration overview
benchmark config outline large_config.yaml

# Example output:
# 📋 Configuration Outline:
#   Name: Large Scale Security Benchmark
#   Models: 12 configured
#   Datasets: 8 configured
#   Available sections: models, datasets, evaluation
```

## 🔗 Integration

The CLI integrates seamlessly with:
- **Configuration Service**: Automatic validation and parsing with advanced caching
- **Performance Optimizations**: LRU caching, lazy loading, and diff tracking
- **Data Service**: Complete E2E data loading and processing pipeline
- **Environment Resolution**: Dynamic variable substitution
- **Rich Output**: Beautiful terminal formatting
- **Error Handling**: Comprehensive error reporting with suggestions
- **Memory Management**: Intelligent memory usage optimization

## 📊 Data Service Integration

### Data Loading Commands

#### `benchmark data load <dataset_config>`
Load and process datasets with comprehensive validation:

```bash
# Load dataset from configuration
benchmark data load configs/datasets/unsw_nb15.yaml

# Load with custom preprocessing
benchmark data load configs/datasets/phishing_emails.yaml --preprocess tokenize,normalize

# Batch load multiple datasets
benchmark data load configs/datasets/*.yaml --batch-size 1000

# Example output:
# 🚀 Loading dataset: UNSW-NB15 Network Traffic
# 📊 Loaded 10,000 samples (3,000 attacks, 7,000 benign)
# ⚡ Processing speed: 91,234 samples/second
# ✅ Dataset validation: 98.5% quality score
# 💾 Memory usage: 45.2MB compressed
```

#### `benchmark data validate <dataset_path>`
Comprehensive dataset quality validation:

```bash
# Validate dataset quality
benchmark data validate data/cybersecurity_logs.json

# Validate with detailed statistics
benchmark data validate data/network_traffic.csv --detailed-stats

# Batch validation with quality thresholds
benchmark data validate data/*.jsonl --min-quality 0.8

# Example output:
# 🔍 Dataset Quality Report:
#   ✅ Total samples: 50,000
#   📈 Quality score: 0.94/1.0
#   🎯 Attack ratio: 32.1%
#   ⚠️ Issues found: 127 duplicate samples
#   💡 Recommendation: Apply deduplication preprocessing
```

#### `benchmark data stream <dataset_config>`
Stream large datasets with real-time processing:

```bash
# Stream dataset in batches
benchmark data stream large_dataset.yaml --batch-size 500

# Stream with concurrent processing
benchmark data stream huge_dataset.yaml --concurrent-batches 4

# Stream with progress monitoring
benchmark data stream dataset.yaml --show-progress --update-interval 1000

# Example output:
# 🌊 Streaming dataset: Large Scale Security Logs
# 📦 Processing batch 1/200 (500 samples)
# ⚡ Current speed: 1,234,567 samples/second validation
# 📊 Progress: [████████████████████] 100% (100,000/100,000)
# ✅ Stream completed: 2.3 minutes total
```

### Performance Monitoring Commands

#### `benchmark data performance <dataset_config>`
Monitor data service performance and optimization:

```bash
# Performance benchmark
benchmark data performance dataset.yaml

# Memory usage analysis
benchmark data performance dataset.yaml --memory-profile

# Concurrent load testing
benchmark data performance dataset.yaml --concurrent-streams 8

# Example output:
# 📊 Data Service Performance Report:
#   🚀 Loading speed: 91,234 samples/second
#   ✅ Validation speed: 1,234,567 samples/second
#   💾 Memory usage: 128MB (optimized compression)
#   🔄 Cache hit rate: 87.3%
#   ⚡ Hardware optimization: Active (Apple M4 Pro)
```

#### `benchmark data health-check`
Comprehensive data service health monitoring:

```bash
# Basic health check
benchmark data health-check

# Detailed system diagnostics
benchmark data health-check --detailed

# Export health metrics
benchmark data health-check --export-json health_metrics.json

# Example output:
# 🏥 Data Service Health Check:
#   ✅ Service status: HEALTHY
#   📊 Active datasets: 12 cached, 3 streaming
#   💾 Memory status: 245MB / 512MB (48% usage)
#   🔄 Cache performance: 87.3% hit rate
#   ⚡ Hardware optimization: Apple M4 Pro detected
#   🌊 Stream throughput: 45,234 samples/second
```

## 🧪 End-to-End Testing Integration

### E2E Test Execution Commands

#### `benchmark test e2e <test_suite>`
Execute comprehensive end-to-end test scenarios:

```bash
# Run complete E2E test suite
benchmark test e2e --suite complete

# Run specific E2E scenarios
benchmark test e2e --scenarios realistic_workflows,performance_benchmarks

# Run with custom dataset sizes
benchmark test e2e --dataset-size large --concurrent-streams 6

# Example output:
# 🧪 E2E Testing: Complete Data Service Pipeline
# ✅ Test 1/9: Complete dataset pipeline (PASSED)
# ✅ Test 2/9: Multi-source loading (PASSED)
# ✅ Test 3/9: Large dataset handling (PASSED)
# ✅ Test 4/9: Error recovery scenarios (PASSED)
# ✅ Test 5/9: Concurrent load testing (PASSED)
# ✅ Test 6/9: Realistic cybersecurity workflows (PASSED)
# ✅ Test 7/9: Integration with preprocessing (PASSED)
# ✅ Test 8/9: Performance benchmarks (PASSED)
# ✅ Test 9/9: Service resilience testing (PASSED)
#
# 📊 E2E Test Results Summary:
#   ✅ All 9 scenarios passed
#   ⚡ Average loading speed: 91,234+ samples/second
#   📈 Average validation speed: 1,234,567+ samples/second
#   💾 Memory efficiency: Outstanding (compression active)
#   🏆 Overall grade: EXCELLENT
```

#### `benchmark test performance <performance_suite>`
Run performance-specific testing scenarios:

```bash
# Performance test suite
benchmark test performance --suite data_service

# Hardware optimization tests
benchmark test performance --hardware-specific --platform apple_m4_pro

# Concurrent processing tests
benchmark test performance --concurrent-load --max-streams 10

# Example output:
# ⚡ Performance Testing: Data Service Optimization
# 🚀 Hardware optimization: ACTIVE (Apple M4 Pro detected)
# 📊 Loading performance: 91,234 samples/second (✅ EXCELLENT)
# ✅ Validation performance: 1,234,567 samples/second (✅ OUTSTANDING)
# 💾 Memory efficiency: 60% reduction through compression
# 🔄 Cache performance: 87.3% hit rate (✅ TARGET MET)
# 🌊 Concurrent streams: 8 streams processed simultaneously
# 🏆 Performance grade: OUTSTANDING
```

### Data Generation Testing Commands

#### `benchmark test datagen <generation_suite>`
Test realistic cybersecurity data generation capabilities:

```bash
# Test data generation pipeline
benchmark test datagen --suite cybersecurity

# Test specific attack types
benchmark test datagen --attack-types malware,phishing,dos

# Test large-scale generation
benchmark test datagen --scale large --samples 100000

# Example output:
# 🎲 Data Generation Testing: Cybersecurity Scenarios
# ✅ Network logs: 10,000 samples generated
# ✅ Phishing emails: 5,000 samples generated
# ✅ Web attack logs: 7,500 samples generated
# ✅ Malware samples: 2,500 samples generated
#
# 📊 Generation Performance:
#   ⚡ Speed: 15,234 samples/second
#   🎯 Attack distribution: 32.1% (target: 30%)
#   📈 Quality metrics: 94.2% realistic samples
#   🔍 Schema compliance: 100% valid
#   ✅ All generation tests passed
```

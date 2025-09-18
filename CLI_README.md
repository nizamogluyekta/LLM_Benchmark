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

6. **Test complete E2E pipeline**:
   ```bash
   # Test data service (9 E2E scenarios)
   PYTHONPATH=src pytest tests/e2e/test_data_service_e2e.py -v

   # Test model service (7 E2E scenarios)
   PYTHONPATH=src pytest tests/e2e/test_model_service_e2e.py -v
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

## 🤖 Model Service Integration

### Model Management Commands

#### `benchmark model load <model_config>`
Load and initialize models with performance monitoring:

```bash
# Load OpenAI API model
benchmark model load configs/models/openai_gpt4o.yaml

# Load MLX local model with performance tracking
benchmark model load configs/models/mlx_llama2_7b.yaml --enable-monitoring

# Load multiple models for comparison
benchmark model load configs/models/*.yaml --batch-load

# Example output:
# 🤖 Loading model: GPT-4o-mini
# ✅ Model loaded successfully (ID: openai-gpt4o-12345)
# 📊 Memory usage: 2.1GB
# ⚡ Initialization time: 3.2s
# 🏥 Health status: HEALTHY
```

#### `benchmark model predict <model_id> <samples_file>`
Run batch inference with cybersecurity samples:

```bash
# Single model inference
benchmark model predict openai-gpt4o-12345 data/cybersecurity_samples.jsonl

# Batch processing with custom batch size
benchmark model predict mlx-llama-67890 data/phishing_emails.json --batch-size 8

# Multi-model comparison
benchmark model predict --models openai-gpt4o-12345,mlx-llama-67890 data/attack_samples.json

# Example output:
# 🧪 Processing 100 cybersecurity samples
# 📊 Batch 1/13 (8 samples): 7 ATTACK, 1 BENIGN predictions
# ⚡ Processing speed: 45 samples/minute
# 🎯 Attack detection accuracy: 94.2%
# ✅ Batch inference completed: 2.3 minutes total
```

#### `benchmark model compare <model_ids>`
Compare multiple models on the same cybersecurity tasks:

```bash
# Compare two models
benchmark model compare openai-gpt4o-12345,mlx-llama-67890

# Compare with detailed performance metrics
benchmark model compare openai-gpt4o-12345,mlx-llama-67890,anthropic-claude-54321 --detailed

# Export comparison results
benchmark model compare model1,model2,model3 --export-json comparison_results.json

# Example output:
# 🏆 Model Performance Comparison:
#   Best performer: mlx-llama-67890 (95.1% accuracy)
#   Fastest inference: openai-gpt4o-12345 (0.8s avg)
#   Most cost-effective: mlx-llama-67890 ($0.00/1k tokens)
#
# 📈 Detailed Metrics:
#   openai-gpt4o-12345: 92.3% accuracy, $0.045/1k tokens
#   mlx-llama-67890: 95.1% accuracy, $0.00/1k tokens
#   anthropic-claude-54321: 93.7% accuracy, $0.032/1k tokens
```

#### `benchmark model health-check <model_id>`
Monitor model service health and performance:

```bash
# Basic health check
benchmark model health-check openai-gpt4o-12345

# Detailed health diagnostics
benchmark model health-check mlx-llama-67890 --detailed

# Health check all loaded models
benchmark model health-check --all

# Example output:
# 🏥 Model Health Report:
#   Model ID: mlx-llama-67890
#   Status: HEALTHY
#   Memory usage: 6.8GB / 16GB (42.5%)
#   Average response time: 0.95s
#   Success rate: 99.2%
#   Uptime: 2h 34m
#   Total predictions: 1,847
```

### Performance Monitoring Commands

#### `benchmark model performance <model_id>`
Monitor model performance and optimization:

```bash
# Performance report
benchmark model performance mlx-llama-67890

# Real-time performance monitoring
benchmark model performance openai-gpt4o-12345 --real-time --interval 30

# Performance comparison across models
benchmark model performance --compare-all --export-csv performance_report.csv

# Example output:
# 📊 Model Performance Report:
#   🚀 Inference speed: 8.2 tokens/second (✅ EXCELLENT)
#   ✅ Success rate: 99.2%
#   💾 Memory efficiency: 42.5% usage (6.8GB / 16GB)
#   ⚡ Hardware optimization: Active (Apple M4 Pro)
#   🎯 Accuracy on cybersecurity tasks: 95.1%
#   💰 Cost efficiency: $0.00/1k tokens (local model)
```

#### `benchmark model cost-estimate <model_configs>`
Estimate costs for model usage scenarios:

```bash
# Cost estimate for specific model
benchmark model cost-estimate configs/models/openai_gpt4o.yaml --samples 10000

# Compare costs across models
benchmark model cost-estimate configs/models/*.yaml --samples 50000 --compare

# Monthly cost projection
benchmark model cost-estimate configs/models/api_models.yaml --monthly-projection --daily-samples 1000

# Example output:
# 💰 Model Cost Estimation:
#   Sample size: 10,000 cybersecurity samples
#
#   OpenAI GPT-4o-mini: $0.45 ($0.045/1k tokens)
#   Anthropic Claude-3-Haiku: $0.32 ($0.032/1k tokens)
#   MLX Llama2-7B: $0.00 (local model)
#
# 📊 Monthly projection (30k samples):
#   Most cost-effective: MLX Llama2-7B ($0.00/month)
#   Best accuracy/cost ratio: Claude-3-Haiku (93.7%/$0.96)
```

### Model Service Testing Commands

#### `benchmark test model-service <test_suite>`
Execute comprehensive model service test scenarios:

```bash
# Run complete model service E2E tests
benchmark test model-service --suite e2e

# Run specific test scenarios
benchmark test model-service --scenarios lifecycle,comparison,resilience

# Run with realistic cybersecurity datasets
benchmark test model-service --dataset-size large --realistic-scenarios

# Example output:
# 🧪 E2E Testing: Complete Model Service Pipeline
# ✅ Test 1/7: Complete model lifecycle (PASSED)
# ✅ Test 2/7: Multi-model comparison workflow (PASSED)
# ✅ Test 3/7: Model service resilience (PASSED)
# ✅ Test 4/7: Realistic cybersecurity evaluation (PASSED)
# ✅ Test 5/7: Cost tracking accuracy (PASSED)
# ✅ Test 6/7: Performance monitoring integration (PASSED)
# ✅ Test 7/7: Configuration service integration (PASSED)
#
# 📊 E2E Test Results Summary:
#   ✅ All 7 scenarios passed
#   ⚡ Local MLX models: >8 tokens/sec achieved
#   📈 API models: <5s average response time
#   💾 Memory usage: <16GB for realistic combinations
#   🏆 Overall grade: EXCELLENT
```

#### `benchmark test performance <performance_suite>`
Run performance-specific testing scenarios:

```bash
# Performance test suite
benchmark test performance --suite model-service

# Hardware optimization tests
benchmark test performance --hardware-specific --platform apple_m4_pro

# Concurrent processing tests
benchmark test performance --concurrent-models --max-models 3

# Example output:
# ⚡ Performance Testing: Model Service Optimization
# 🚀 Hardware optimization: ACTIVE (Apple M4 Pro detected)
# 📊 Local MLX performance: 8.2 tokens/sec (✅ EXCELLENT)
# ✅ API model performance: 2.1s avg response (✅ TARGET MET)
# 💾 Memory efficiency: <16GB with 3 models loaded
# 🔄 Concurrent processing: 2-3 models simultaneously
# 🏆 Performance grade: OUTSTANDING
```

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

## 🔍 Advanced Explainability Analysis Integration

### Explainability Analysis Commands

#### `benchmark explain analyze <predictions_file>`
Run comprehensive explainability analysis on model predictions:

```bash
# Basic explainability analysis
benchmark explain analyze predictions.json --ground-truth ground_truth.json

# Advanced pattern analysis
benchmark explain analyze predictions.json --analysis pattern,clustering,quality

# Template-based evaluation
benchmark explain analyze predictions.json --templates cybersecurity --strict-mode

# Example output:
# 🔍 Advanced Explainability Analysis Results:
#   📊 Pattern Analysis: 4 attack types analyzed
#   🎯 Clustering: 3 explanation clusters identified
#   📈 Quality Distribution: 0.856 average quality score
#   ⚠️ Common Issues: 2 quality issues detected
#   💡 Improvement Suggestions: 5 recommendations generated
```

#### `benchmark explain compare <model_predictions>`
Compare explanation quality across multiple models:

```bash
# Compare two models
benchmark explain compare model_a.json model_b.json --ground-truth truth.json

# Multi-model comparison with detailed metrics
benchmark explain compare model_*.json --detailed --export-report comparison.json

# Model ranking with statistical significance
benchmark explain compare predictions_*.json --rank-models --significance-test

# Example output:
# 🏆 Model Explanation Comparison:
#   Best model: Advanced-Model-B (0.847 weighted score)
#   Quality difference: +0.123 (significant)
#   Consistency difference: +0.089 (significant)
#   Technical accuracy: +0.156 (highly significant)
#
# 📊 Detailed Rankings:
#   1. Advanced-Model-B: 0.847 overall score
#   2. GPT-4o-Mini: 0.724 overall score
#   3. Basic-Model-A: 0.691 overall score
```

#### `benchmark explain templates <action>`
Manage and evaluate cybersecurity explanation templates:

```bash
# List available templates
benchmark explain templates list

# Evaluate explanations against templates
benchmark explain templates evaluate predictions.json --attack-types malware,dos,phishing

# Generate template statistics
benchmark explain templates stats --export-csv template_stats.csv

# Add custom template
benchmark explain templates add custom_template.yaml

# Example output:
# 📋 Cybersecurity Explanation Templates:
#   ✅ Malware Analysis (4 required, 3 optional elements)
#   ✅ SQL Injection (4 required, 3 optional elements)
#   ✅ DoS Attack (3 required, 3 optional elements)
#   ✅ Phishing Detection (4 required, 3 optional elements)
#   ✅ Intrusion Detection (3 required, 3 optional elements)
#   ✅ Data Exfiltration (4 required, 3 optional elements)
#
# 📊 Template Coverage: 10 attack types, 95% cybersecurity domain coverage
```

#### `benchmark explain quality <predictions_file>`
Comprehensive explanation quality assessment:

```bash
# Quality distribution analysis
benchmark explain quality predictions.json --distribution-stats

# Identify quality issues
benchmark explain quality predictions.json --issues-analysis --fix-suggestions

# Export quality metrics
benchmark explain quality predictions.json --export-metrics quality_report.json

# Example output:
# 📈 Explanation Quality Assessment:
#   🎯 Overall Quality Score: 0.742/1.0
#   📊 Length Distribution: μ=18.5 words, σ=8.2
#   ✅ Completeness Ratio: 67.3% (causal reasoning present)
#   🔬 Technical Term Usage: 78.9% contain domain terminology
#   📝 Vocabulary Richness: 0.68 (good diversity)
#
# ⚠️ Quality Issues Identified:
#   • 15% explanations lack technical terminology
#   • 23% explanations are too vague without specifics
#   • 8% explanations have low completeness indicators
#
# 💡 Improvement Recommendations:
#   • Include more cybersecurity-specific technical terms
#   • Replace vague terms with specific details (IPs, ports, protocols)
#   • Improve causal reasoning with 'because', 'due to', etc.
```

#### `benchmark explain patterns <predictions_file>`
Advanced pattern recognition and clustering analysis:

```bash
# Pattern analysis by attack type
benchmark explain patterns predictions.json --attack-types --keyword-coverage

# Explanation clustering
benchmark explain patterns predictions.json --clustering --similarity-threshold 0.6

# Statistical pattern analysis
benchmark explain patterns predictions.json --statistics --distribution-analysis

# Example output:
# 🔍 Advanced Pattern Analysis Results:
#   🎯 Attack Type Patterns:
#     • Malware: 15 samples, 78% keyword coverage, 0.82 diversity
#     • SQL Injection: 12 samples, 91% keyword coverage, 0.67 diversity
#     • DoS: 8 samples, 85% keyword coverage, 0.74 diversity
#
#   🔗 Explanation Clusters:
#     • Cluster 1: "Pattern with phrases: malware detected, file hash" (8 explanations)
#     • Cluster 2: "Pattern with phrases: sql injection, union technique" (6 explanations)
#     • Cluster 3: "Pattern: attack-related, network-focused" (4 explanations)
#
#   📊 Statistical Analysis:
#     • Vocabulary Richness: 0.68
#     • Average Word Count: 18.5
#     • Consistency Ratio: 0.87
#     • Skewness: 0.23, Kurtosis: -0.15
```

### Integration Testing Commands

#### `benchmark test explainability <test_suite>`
Test advanced explainability analysis functionality:

```bash
# Run complete explainability test suite
benchmark test explainability --suite advanced_analysis

# Test specific components
benchmark test explainability --components analyzer,templates,comparison

# Test with realistic cybersecurity data
benchmark test explainability --realistic-data --attack-types all

# Example output:
# 🧪 Advanced Explainability Testing Suite:
# ✅ Test 1/24: AdvancedExplainabilityAnalyzer initialization (PASSED)
# ✅ Test 2/24: Pattern analysis functionality (PASSED)
# ✅ Test 3/24: Explanation clustering (PASSED)
# ✅ Test 4/24: Quality distribution analysis (PASSED)
# ✅ Test 5/24: Model comparison framework (PASSED)
# ✅ Test 6/24: Template-based evaluation (PASSED)
# ✅ Test 7/24: Batch processing capabilities (PASSED)
# ✅ Test 8/24: Statistical analysis metrics (PASSED)
# ...
# ✅ Test 24/24: Edge case handling (PASSED)
#
# 📊 Explainability Test Results:
#   ✅ All 24 tests passed
#   🔍 Pattern analysis: Identifies 6+ attack types accurately
#   🎯 Template evaluation: 10+ cybersecurity templates validated
#   📈 Statistical metrics: Complete analysis including skewness/kurtosis
#   🏆 Overall grade: EXCELLENT
```

### Evaluation Service Integration

#### `benchmark evaluate explainability <evaluation_config>`
Run explainability evaluation through the evaluation service:

```bash
# Standard explainability evaluation
benchmark evaluate explainability eval_config.yaml --predictions predictions.json

# Advanced analysis integration
benchmark evaluate explainability eval_config.yaml --advanced-analysis --all-features

# Custom configuration override
benchmark evaluate explainability eval_config.yaml --judge-model gpt-4 --batch-size 5

# Example output:
# 🔬 Explainability Evaluation Pipeline:
#   🤖 Judge Model: GPT-4o-mini
#   📊 Batch Size: 10 predictions per batch
#   🎯 Advanced Analysis: ENABLED
#
# 📈 Evaluation Results:
#   ✅ LLM Judge Scores: 0.758 average quality
#   📊 Automated Metrics: BLEU=0.45, ROUGE-L=0.52
#   🔍 Pattern Analysis: 4 attack types, 3 clusters identified
#   🎯 Template Evaluation: 0.736 average template score
#   💡 Improvement Suggestions: 6 actionable recommendations
#
# 🏆 Overall Explainability Score: 0.747/1.0 (Good)
```

### Performance and Monitoring

#### `benchmark explain performance <analysis_config>`
Monitor explainability analysis performance:

```bash
# Performance benchmarking
benchmark explain performance --samples 1000 --iterations 5

# Memory usage analysis
benchmark explain performance --memory-profile --large-dataset

# Concurrent analysis testing
benchmark explain performance --concurrent-batches 4 --batch-size 50

# Example output:
# ⚡ Explainability Analysis Performance:
#   🚀 Pattern Analysis: 1,234 explanations/second
#   🎯 Template Evaluation: 2,456 evaluations/second
#   📊 Statistical Analysis: 5,678 calculations/second
#   🔍 Clustering: 890 similarity computations/second
#   💾 Memory Usage: 45.2MB for 1,000 explanations
#   ⏱️ Total Analysis Time: 0.81 seconds
#   🏆 Performance Grade: EXCELLENT
```

### Integration with Existing Workflows

The advanced explainability features integrate seamlessly with existing CLI workflows:

```bash
# Complete evaluation pipeline with explainability
benchmark config validate config.yaml && \
benchmark data load dataset.yaml && \
benchmark model load model.yaml && \
benchmark model predict model-id data/samples.json --output predictions.json && \
benchmark explain analyze predictions.json --ground-truth ground_truth.json && \
benchmark explain compare predictions_*.json --rank-models

# Automated quality assurance
benchmark explain quality predictions.json --min-score 0.7 --auto-report

# Template-based validation
benchmark explain templates evaluate predictions.json --strict-mode --min-coverage 0.8
```

This advanced explainability analysis provides comprehensive insights into model explanation quality, helping improve cybersecurity model interpretability and trustworthiness.

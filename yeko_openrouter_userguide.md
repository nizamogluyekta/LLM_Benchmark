# Yeko's OpenRouter Cybersecurity Benchmark - User Guide

A comprehensive guide for using `yeko_openrouter_benchmark.py` - an advanced NSL-KDD cybersecurity benchmarking system with balanced attack type sampling and detailed reporting.

## 🚀 Overview

`yeko_openrouter_benchmark.py` is an advanced cybersecurity benchmarking tool that:
- Tests LLMs on network intrusion detection using NSL-KDD dataset
- Provides balanced sampling across all attack types (23 different attack categories)
- Offers detailed progress tracking and comprehensive reporting
- Uses OpenRouter API for access to multiple state-of-the-art models
- Generates academic-quality evaluation metrics

## 📋 Quick Setup

### Step 1: Get OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up/login and obtain your API key
3. Note down your API key for the next step

### Step 2: Configure API Key

Edit `yeko_openrouter_benchmark.py` and replace the hardcoded API key:

```python
# Find this line (around line 814):
HARDCODED_API_KEY = "YOUR_API_KEY_HERE"

# Replace with your actual key:
HARDCODED_API_KEY = "sk-or-v1-your-actual-api-key-here"
```

### Step 3: Verify Prerequisites

Ensure you have:
- ✅ NSL-KDD dataset in `NSL-KDD/` folder with `KDDTrain+.txt` file
- ✅ Python dependencies: `openai`, `numpy`
- ✅ Internet connection for OpenRouter API

## 🎯 Basic Usage

### Simple Test Command

```bash
# Most basic usage - 1000 balanced samples with Llama 3.2 3B
python3 yeko_openrouter_benchmark.py --model "meta-llama/llama-3.2-3b-instruct:free"
```

### Recommended Models

#### Free Models (No Cost)
```bash
# Llama 3.2 3B - Fast and efficient
python3 yeko_openrouter_benchmark.py --model "meta-llama/llama-3.2-3b-instruct:free"

# X.AI Grok - Very capable for cybersecurity
python3 yeko_openrouter_benchmark.py --model "x-ai/grok-4-fast:free"

# OpenAI GPT models
python3 yeko_openrouter_benchmark.py --model "openai/gpt-oss-120b:free"
python3 yeko_openrouter_benchmark.py --model "openai/gpt-oss-20b:free"

# Other excellent options
python3 yeko_openrouter_benchmark.py --model "z-ai/glm-4.5-air:free"
python3 yeko_openrouter_benchmark.py --model "qwen/qwen3-coder:free"
python3 yeko_openrouter_benchmark.py --model "nvidia/nemotron-nano-9b-v2:free"
```

## ⚙️ Advanced Usage

### Custom Dataset Files

```bash
# Use test dataset instead of training dataset
python3 yeko_openrouter_benchmark.py --model "meta-llama/llama-3.2-3b-instruct:free" --dataset "KDDTest+.txt"

# Use smaller subset files
python3 yeko_openrouter_benchmark.py --model "x-ai/grok-4-fast:free" --dataset "KDDTrain+_20Percent.txt"
```

### Custom Sample Sizes

```bash
# Quick test with 100 samples
python3 yeko_openrouter_benchmark.py --model "meta-llama/llama-3.2-3b-instruct:free" --samples 100

# Medium test with 500 samples
python3 yeko_openrouter_benchmark.py --model "x-ai/grok-4-fast:free" --samples 500

# Large-scale test with 2000 samples
python3 yeko_openrouter_benchmark.py --model "openai/gpt-oss-120b:free" --samples 2000
```

### Custom Output Files

```bash
# Save results to specific filename
python3 yeko_openrouter_benchmark.py --model "meta-llama/llama-3.2-3b-instruct:free" --output llama_cybersec_test.json

# Academic naming convention
python3 yeko_openrouter_benchmark.py --model "x-ai/grok-4-fast:free" --output experiment_1_grok_1000_samples.json
```

### Combined Advanced Options

```bash
# Comprehensive test setup
python3 yeko_openrouter_benchmark.py \
  --model "x-ai/grok-4-fast:free" \
  --dataset "KDDTest+.txt" \
  --samples 1500 \
  --output grok_comprehensive_test_20250925.json
```

## 📊 Understanding the Output

### Console Progress Tracking

During execution, you'll see detailed progress:

```
🚀 YEKO'S ADVANCED NSL-KDD CYBERSECURITY BENCHMARK
============================================================
Model: meta-llama/llama-3.2-3b-instruct:free
Dataset: KDDTrain+.txt
Target Samples: 1000
Balanced Attack Types: Yes
============================================================

📊 Loading NSL-KDD data from NSL-KDD/KDDTrain+.txt
📈 Dataset Analysis:
   Total rows processed: 125973
   Total valid connections: 125973
   Attack types found: 23
   - normal: 67343 samples
   - neptune: 41214 samples
   - back: 956 samples
   - satan: 691 samples
   - ipsweep: 690 samples
   ...

⚖️ Balanced Sampling Strategy:
   Base samples per type: 43
   Extra samples for first 11 types: 1 each
   ✅ normal: selected 44/67343 samples
   ✅ neptune: selected 44/41214 samples
   ✅ back: selected 44/956 samples
   ...

🚀 Starting Comprehensive Cybersecurity Benchmark
📊 Progress: 50/1000 (5.0%) | Rate: 25.5 samples/min | ETA: 37.3 min | Accuracy so far: 84.0%
📊 Progress: 100/1000 (10.0%) | Rate: 24.8 samples/min | ETA: 36.2 min | Accuracy so far: 86.0%
...
```

### Final Results Summary

```
🏆 YEKO'S ADVANCED NSL-KDD CYBERSECURITY BENCHMARK RESULTS
================================================================================

🔬 BENCHMARK INFORMATION
   Model: meta-llama/llama-3.2-3b-instruct:free
   Timestamp: 2025-01-25T14:30:22.123456
   Total Samples: 1000
   Processing Time: 2847.32 seconds
   Samples per Second: 0.35

📊 DATASET DISTRIBUTION
   - normal: 44 samples (4.4%)
   - neptune: 44 samples (4.4%)
   - back: 44 samples (4.4%)
   - satan: 44 samples (4.4%)
   - ipsweep: 44 samples (4.4%)
   - portsweep: 44 samples (4.4%)
   - warezclient: 43 samples (4.3%)
   ...

🎯 CLASSIFICATION PERFORMANCE
   Overall Accuracy:        0.847 (84.7%)
   Balanced Accuracy:       0.823
   Precision:               0.891
   Recall (Sensitivity):    0.756
   Specificity:             0.890
   F1-Score:                0.818
   Matthews Correlation:    0.652

🔍 CONFUSION MATRIX
   True Positives:   723   False Positives:   91
   False Negatives:  162   True Negatives:    24
   Total Valid Predictions: 1000

⚡ PERFORMANCE METRICS
   Average Response Time:   2.847s
   Valid Predictions:       976/1000
   Error Rate:              24/1000 (2.4%)

🏹 ATTACK TYPE PERFORMANCE
   normal         :  42/ 44 (95.5%)
   back           :  41/ 44 (93.2%)
   neptune        :  40/ 44 (90.9%)
   satan          :  39/ 44 (88.6%)
   ipsweep        :  38/ 44 (86.4%)
   portsweep      :  37/ 44 (84.1%)
   ...
```

## 📁 Generated Files

### 1. JSON Results File
**Format**: `yeko_benchmark_{model_name}_{timestamp}.json`
**Example**: `yeko_benchmark_meta_llama_llama_3_2_3b_instruct_free_20250925_143022.json`

**Contains**:
- Complete benchmark configuration
- All individual predictions with explanations
- Comprehensive metrics and statistics
- Model performance data
- Attack type breakdown

### 2. Log File
**Format**: `yeko_benchmark_{timestamp}.log`
**Example**: `yeko_benchmark_20250925_143022.log`

**Contains**:
- Detailed execution log
- Debug information
- API call tracking
- Error messages and warnings

## ⏱️ Expected Runtime

| Sample Size | Estimated Duration | Use Case |
|-------------|-------------------|----------|
| 100 samples | 5-8 minutes | Quick testing |
| 500 samples | 25-30 minutes | Medium evaluation |
| 1000 samples | 45-60 minutes | Standard benchmark |
| 2000 samples | 90-120 minutes | Comprehensive study |

*Note: Duration varies by model and API response times*

## 🎯 Key Features Explained

### Balanced Attack Type Sampling

Unlike random sampling, the script ensures:
- **Equal representation** of all 23 attack types in NSL-KDD
- **Realistic distribution** for comprehensive evaluation
- **No bias** toward common attack types like 'normal' or 'neptune'
- **Statistical validity** for academic research

### Advanced Progress Tracking

- **Real-time updates** every 50 samples
- **ETA calculation** based on current processing rate
- **Running accuracy** to monitor model performance
- **Rate limiting** to respect API constraints

### Comprehensive Metrics

#### Classification Metrics
- **Overall Accuracy**: Basic correctness measure
- **Balanced Accuracy**: Accounts for class imbalance
- **Precision/Recall**: Attack detection effectiveness
- **F1-Score**: Harmonic mean of precision/recall
- **Matthews Correlation**: Robust performance measure

#### Attack-Specific Analysis
- **Per-attack-type accuracy** for 23 different attack categories
- **Confusion matrix** with detailed TP/FP/TN/FN breakdown
- **Error analysis** with failure pattern identification

## 🔧 Troubleshooting

### Common Issues

#### ❌ "API key error"
**Problem**: Invalid or missing API key
**Solution**:
1. Verify your OpenRouter API key is correct
2. Check that you edited line 814 in the script properly
3. Ensure no extra spaces or quotes around the key

#### ❌ "Dataset file not found: NSL-KDD/KDDTrain+.txt"
**Problem**: Missing NSL-KDD dataset
**Solution**:
1. Download NSL-KDD dataset from [UNB.ca](https://www.unb.ca/cic/datasets/nsl.html)
2. Extract files to `NSL-KDD/` folder in your project directory
3. Verify file permissions are readable

#### ❌ "Rate limit exceeded"
**Problem**: Too many API requests
**Solution**:
- Script automatically handles this with retries
- If persistent, try smaller sample sizes first
- Check your OpenRouter account usage limits

#### ❌ Import errors (openai, numpy, etc.)
**Problem**: Missing dependencies
**Solution**:
```bash
pip install openai numpy
```

### Performance Tips

1. **Start Small**: Use `--samples 100` for initial testing
2. **Monitor Progress**: Watch console output for issues
3. **Stable Network**: Ensure reliable internet connection
4. **Free vs Paid**: Free models have rate limits, paid models are faster

## 📚 Command Reference

### Required Parameters
- `--model`: OpenRouter model name (required)

### Optional Parameters
- `--dataset`: NSL-KDD file to use (default: "KDDTrain+.txt")
- `--samples`: Number of samples to test (default: 1000)
- `--output`: Output filename (auto-generated if not specified)
- `--api-key`: API key override (uses hardcoded if not provided)

### Help Command
```bash
python3 yeko_openrouter_benchmark.py --help
```

## 🎓 Academic Usage

### For Research Papers
- Use consistent sample sizes (1000+ recommended)
- Document model versions and configurations
- Report all metrics (accuracy, F1, MCC)
- Include attack-type specific performance
- Save JSON results for reproducibility

### Citation Example
When using this tool in academic work:

```
Cybersecurity evaluation conducted using Yeko's OpenRouter Benchmark
with balanced NSL-KDD sampling (N=1000 samples across 23 attack types).
Model: {model_name}, Overall Accuracy: {accuracy}%, F1-Score: {f1_score}.
```

## 🚀 Best Practices

1. **Model Selection**: Test multiple models for comparison
2. **Sample Size**: Use 1000+ samples for statistical significance
3. **Balanced Testing**: The script automatically ensures balanced sampling
4. **Documentation**: Save all JSON results for later analysis
5. **Reproducibility**: Use consistent random seeds and configurations

This comprehensive tool provides everything needed for rigorous cybersecurity LLM evaluation with minimal setup and maximum insight!

# NSL-KDD Network Intrusion Detection Benchmark

This specialized benchmark evaluates Large Language Models (LLMs) on network intrusion detection using the NSL-KDD dataset and LM Studio for local model inference.

## Overview

The NSL-KDD benchmark converts network connection data into human-readable text and tasks LLMs with identifying malicious network activity. This provides insights into LLM performance on cybersecurity tasks involving network traffic analysis.

## NSL-KDD Dataset

The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset, containing:

- **125,973 training samples** in KDDTrain+.txt
- **22,544 test samples** in KDDTest+.txt
- **41 network features** (duration, protocol, service, bytes transferred, etc.)
- **23 attack types** including neptune, satan, ipsweep, portsweep, smurf, nmap, back, teardrop, etc.
- **Binary classification**: Normal vs Attack traffic

### Attack Types in NSL-KDD

- **DoS Attacks**: neptune, smurf, back, teardrop, pod, land
- **Probe Attacks**: satan, ipsweep, nmap, portsweep
- **R2L Attacks**: warezclient, warezmaster, ftp_write, guess_passwd, imap, multihop, phf, spy
- **U2R Attacks**: buffer_overflow, loadmodule, perl, rootkit

## Prerequisites

1. **LM Studio** running locally (default: http://localhost:1234)
2. **Model loaded** in LM Studio (any chat model)
3. **NSL-KDD dataset** in the `NSL-KDD/` folder
4. **Python dependencies**: `openai`, `numpy`

## Quick Start

### 1. Install Dependencies

```bash
pip install openai numpy
```

### 2. Start LM Studio

1. Download and start LM Studio
2. Load a model (e.g., Llama, Mistral, Qwen)
3. Ensure it's running on http://localhost:1234 (default)

### 3. Run Quick Test

```bash
# Test with 20 samples (recommended for first run)
python3 nsl_kdd_lm_studio_benchmark.py --quick-test

# Test with custom parameters
python3 nsl_kdd_lm_studio_benchmark.py --max-samples 50 --attack-ratio 0.4

# Full benchmark with 1000 samples
python3 nsl_kdd_lm_studio_benchmark.py --full-benchmark
```

## Usage Examples

### Basic Usage

```bash
# Quick test (20 samples)
python3 nsl_kdd_lm_studio_benchmark.py --quick-test

# Custom sample size and attack ratio
python3 nsl_kdd_lm_studio_benchmark.py --max-samples 100 --attack-ratio 0.3

# Different LM Studio URL
python3 nsl_kdd_lm_studio_benchmark.py --lm-studio-url http://localhost:8080 --quick-test
```

### Advanced Usage

```bash
# Test specific dataset file
python3 nsl_kdd_lm_studio_benchmark.py --dataset-file KDDTest+.txt --max-samples 200

# Save results to specific file
python3 nsl_kdd_lm_studio_benchmark.py --quick-test --output-file my_results.json

# Full benchmark with custom settings
python3 nsl_kdd_lm_studio_benchmark.py \
  --dataset-file KDDTrain+.txt \
  --max-samples 1000 \
  --attack-ratio 0.35 \
  --output-file full_benchmark_results.json
```

## How It Works

### 1. Data Processing

The benchmark converts raw NSL-KDD network features into human-readable descriptions:

**Raw data:**
```
0,tcp,http,SF,232,8153,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.20,0.20,0.00,0.00,1.00,0.00,0.00,30,255,1.00,0.00,0.03,0.04,0.03,0.01,0.00,0.01,normal,21
```

**Readable text:**
```
Network connection: TCP protocol. Service: http. Connection state: SF.
Data transfer: 232 bytes sent, 8153 bytes received. Connection count in time window: 5.
Same service connections: 5. Destination host connections: 30.
```

### 2. LLM Prompt

The LLM receives a cybersecurity analysis prompt:

```
Analyze the following network connection for potential security threats:

[Readable network connection description]

Task: Determine if this network connection represents:
- NORMAL: Legitimate network activity
- ATTACK: Malicious or suspicious activity

Please respond with your classification (NORMAL or ATTACK) followed by detailed reasoning...
```

### 3. Evaluation

The benchmark evaluates:
- **Classification accuracy** (Normal vs Attack)
- **Attack type detection** (per attack category)
- **Performance metrics** (response time, throughput)
- **Confusion matrix** (TP, FP, TN, FN)
- **Explainability quality** (reasoning analysis)

## Output

### Console Summary
```
NSL-KDD INTRUSION DETECTION BENCHMARK SUMMARY
============================================================

Dataset: KDDTrain+.txt
Sample size: 100 connections
Normal connections: 70
Attack connections: 30
Attack types: neptune, satan, ipsweep, portsweep, smurf

ACCURACY METRICS:
Overall Accuracy: 0.850
Precision: 0.833
Recall: 0.800
F1-Score: 0.816
False Positive Rate: 0.071

CONFUSION MATRIX:
True Positives: 24
False Positives: 5
False Negatives: 6
True Negatives: 65

ATTACK TYPE DETECTION:
  neptune: 12/15 (0.800)
  satan: 8/10 (0.800)
  ipsweep: 4/5 (0.800)

PERFORMANCE METRICS:
Average response time: 1.250s
Predictions per second: 0.80
Total processing time: 125.00s
```

### JSON Results File

Detailed results saved to timestamped JSON file containing:
- Complete predictions and explanations
- Detailed metrics breakdown
- Performance statistics
- Model and dataset configuration

## Troubleshooting

### Common Issues

1. **"Connection refused" error**
   - Ensure LM Studio is running
   - Check the URL (default: http://localhost:1234)
   - Verify a model is loaded in LM Studio

2. **"Dataset file not found"**
   - Ensure NSL-KDD folder exists
   - Download NSL-KDD dataset files
   - Check file permissions

3. **Poor model performance**
   - Try different models in LM Studio
   - Adjust temperature (lower = more consistent)
   - Increase max_tokens for longer explanations

### Dataset Download

If you need to download the NSL-KDD dataset:

1. Visit: https://www.unb.ca/cic/datasets/nsl.html
2. Download the dataset files
3. Extract to `NSL-KDD/` folder in your project directory

## Configuration

See `nsl_kdd_config_example.yaml` for advanced configuration options including:
- Custom model parameters
- Evaluation settings
- Output formats
- Performance tuning

## Performance Tips

1. **Start small**: Use `--quick-test` for initial testing
2. **Model selection**: Larger models generally perform better but slower
3. **Balanced sampling**: Default 30% attack ratio provides good evaluation
4. **Local inference**: LM Studio provides faster inference than API calls

## Research Applications

This benchmark is designed for academic research on:
- LLM performance in cybersecurity domains
- Network intrusion detection with natural language processing
- Explainable AI for security analysis
- Zero-shot learning on cybersecurity tasks

## Citation

If you use this benchmark in research, please cite the original NSL-KDD dataset:

```bibtex
@article{tavallaee2009detailed,
  title={A detailed analysis of the KDD CUP 99 data set},
  author={Tavallaee, Mahbod and Bagheri, Ebrahim and Lu, Wei and Ghorbani, Ali A},
  journal={IEEE Symposium on Computational Intelligence for Security and Defense Applications},
  year={2009}
}
```

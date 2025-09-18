# Local LLM Benchmarking with LM Studio

This guide shows how to use the LLM Cybersecurity Benchmark system with local models running on LM Studio instead of external API services.

## 🚀 Quick Start

### 1. Setup LM Studio
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download your preferred models (e.g., Llama 3, Mistral, CodeLlama)
3. Start the local server in LM Studio:
   - Go to the "Local Server" tab
   - Click "Start Server"
   - Note the server URL (usually `http://localhost:1234`)

### 2. Run Quick Benchmark
```bash
# Quick benchmark with default settings
python local_llm_benchmark.py --quick-benchmark

# Specify your model name
python local_llm_benchmark.py --quick-benchmark --model-name "llama-3-8b-instruct"

# Use custom LM Studio URL
python local_llm_benchmark.py --quick-benchmark --lm-studio-url "http://localhost:1234"
```

### 3. Interactive Mode
```bash
python local_llm_benchmark.py --interactive
```

### 4. Full Configuration-Based Benchmark
```bash
python local_llm_benchmark.py --config example_local_config.yaml
```

## 📋 Features

### ✅ Complete Local Benchmarking
- **No API Keys Required**: Works entirely with local models
- **Privacy-First**: All data stays on your machine
- **Cost-Free**: No token costs or rate limits
- **High Performance**: Optimized for local model inference

### 🔍 Comprehensive Analysis
- **Cybersecurity Focus**: Specialized prompts for security analysis
- **Advanced Explainability**: Pattern analysis and template evaluation
- **Multiple Metrics**: Accuracy, precision, recall, F1-score
- **Performance Tracking**: Inference speed and resource usage

### 🎯 Model Support
Works with any model compatible with LM Studio:
- **Llama Models**: Llama 3, Llama 2, Code Llama
- **Mistral Models**: Mistral 7B, Mixtral 8x7B
- **Other Models**: Any model with OpenAI-compatible API

## 🛠️ Configuration

### Model Configuration
The system automatically converts your LM Studio models to the benchmark format:

```python
# Automatic configuration for LM Studio models
model_config = {
    "name": "your-model-name",
    "type": "openai_api",  # LM Studio uses OpenAI-compatible API
    "base_url": "http://localhost:1234/v1",  # LM Studio endpoint
    "api_key": "lm-studio",  # Dummy key (not needed for local)
    "model_name": "your-actual-model-name-in-lm-studio",
    "max_tokens": 512,
    "temperature": 0.1
}
```

### Sample Configuration File
See `example_local_config.yaml` for a complete configuration example:

```yaml
name: "Local LLM Cybersecurity Benchmark"
description: "Benchmarking local LLMs running on LM Studio"

models:
  - name: "llama3-8b-instruct"
    model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
    max_tokens: 512
    temperature: 0.1

datasets:
  - name: "cybersecurity_samples"
    source: "memory"
    samples:
      - text: "SQL injection attempt detected"
        label: "ATTACK"
      - text: "Normal user login successful"
        label: "BENIGN"

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score"]
  explainability_analysis: true
```

## 📊 Sample Data

The system includes comprehensive cybersecurity samples:

### Attack Samples
- SQL injection attempts
- Brute force login attacks
- Malware detections
- DDoS attacks
- Phishing attempts
- Port scanning
- Data exfiltration
- Buffer overflow exploits

### Benign Samples
- Normal user logins
- Scheduled backups
- Software updates
- Regular web traffic
- Database maintenance
- Email communications
- System health checks
- File transfers

## 🔍 Explainability Analysis

### Pattern Analysis
- **Attack Type Classification**: Analyzes explanations by attack type
- **Keyword Coverage**: Measures domain-specific terminology usage
- **Explanation Clustering**: Groups similar explanations using Jaccard similarity

### Template Evaluation
- **Cybersecurity Templates**: 10+ specialized templates for different attack types
- **Element Coverage**: Required vs optional explanation elements
- **Quality Scoring**: Automated assessment of explanation completeness

### Statistical Metrics
- **Vocabulary Richness**: Diversity of explanation vocabulary
- **Length Distribution**: Statistical analysis of explanation lengths
- **Consistency Measures**: Coherence across similar samples

## 📈 Performance Metrics

### Speed Benchmarks
- **Inference Time**: Per-sample processing time
- **Throughput**: Samples processed per second
- **Total Processing**: End-to-end benchmark duration

### Quality Metrics
- **Classification Accuracy**: Overall prediction correctness
- **Attack Detection**: Precision/recall for cybersecurity threats
- **Explanation Quality**: Template compliance and pattern analysis

### Resource Usage
- **Memory Efficiency**: Local model memory usage
- **CPU Utilization**: Processing resource consumption
- **No Network Costs**: Eliminated API costs and rate limits

## 🎯 Use Cases

### Security Research
```bash
# Evaluate model performance on cybersecurity tasks
python local_llm_benchmark.py --config security_research_config.yaml
```

### Model Comparison
```bash
# Compare multiple local models
# Configure multiple models in your YAML file
python local_llm_benchmark.py --config multi_model_comparison.yaml
```

### Custom Dataset Testing
```bash
# Test on your own cybersecurity datasets
# Add your data to the configuration file
python local_llm_benchmark.py --config custom_dataset_config.yaml
```

### Development and Testing
```bash
# Quick testing during model development
python local_llm_benchmark.py --quick-benchmark --model-name "your-test-model"
```

## 🔧 Troubleshooting

### LM Studio Connection Issues
```bash
# Check if LM Studio server is running
curl http://localhost:1234/v1/models

# Test with different port if needed
python local_llm_benchmark.py --lm-studio-url "http://localhost:8080"
```

### Model Not Found
1. Ensure your model is loaded in LM Studio
2. Check the exact model name in LM Studio's interface
3. Update the `model_name` in your configuration

### Performance Issues
- **Reduce batch size**: Lower `batch_size` in evaluation config
- **Adjust max_tokens**: Reduce `max_tokens` for faster inference
- **Temperature settings**: Lower temperature for more consistent results

## 📋 Requirements

### Software Requirements
- Python 3.8+
- LM Studio installed and running
- Required Python packages (installed automatically):
  - openai
  - asyncio
  - pyyaml
  - pandas (optional, for data analysis)

### Hardware Requirements
- **RAM**: 8GB minimum (16GB+ recommended for larger models)
- **Storage**: Sufficient space for your chosen models
- **CPU/GPU**: Compatible with your LM Studio setup

## 🚀 Advanced Usage

### Custom Prompting
Modify the cybersecurity prompts in the script for specialized analysis:

```python
def _format_cybersecurity_prompt(self, sample: str) -> str:
    return f"""Your custom prompt here: {sample}"""
```

### Extended Analysis
Add custom evaluation metrics or analysis methods:

```python
async def run_custom_analysis(self, predictions):
    # Your custom analysis logic
    pass
```

### Integration with Existing Workflows
```python
# Use as a library in your existing code
from local_llm_benchmark import LocalLLMBenchmark

benchmark = LocalLLMBenchmark()
await benchmark.initialize_services()
results = await benchmark.run_quick_benchmark("your-model")
```

## 📄 Output Files

The benchmark generates comprehensive reports:

- **`benchmark_report_YYYYMMDD_HHMMSS.json`**: Detailed results with all data
- **`benchmark_summary_YYYYMMDD_HHMMSS.txt`**: Human-readable summary
- **Console Output**: Real-time progress and key metrics

## 🤝 Support

For issues specific to local LLM benchmarking:
1. Check LM Studio server status
2. Verify model names and configurations
3. Review logs for detailed error information
4. Test with the quick benchmark first

This local benchmarking solution provides the same comprehensive analysis capabilities as the cloud-based version while keeping everything private and cost-free on your local machine.

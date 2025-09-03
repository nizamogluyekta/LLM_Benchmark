# LLM Cybersecurity Attack Detection Benchmarking System

A comprehensive benchmarking framework for evaluating Large Language Models (LLMs) in cybersecurity attack detection scenarios using public datasets and automated evaluation methods.

## Overview

This project implements a **practical, feasible** benchmarking system designed for individual academic researchers using **MacBook Pro M4 hardware**. The system evaluates LLMs across four core dimensions: detection accuracy, explainability quality, response time, and false positive rates in realistic cybersecurity scenarios.

## Key Features

### 🎯 **Multi-Dimensional Evaluation**
- **Detection Accuracy**: Precision, recall, F1-score, ROC-AUC, Matthews correlation
- **Explainability Quality**: LLM-as-judge assessment, semantic similarity metrics
- **Response Time**: Latency analysis, throughput measurement, performance consistency
- **False Positive Rate**: Operational impact assessment, alert fatigue modeling

### 🔧 **Modular Architecture**
- **Plugin-based Design**: Easy integration of new models, datasets, and evaluation metrics
- **Service-oriented Structure**: Independent, scalable components with clear interfaces
- **Apple Silicon Optimized**: MLX framework integration for efficient local inference
- **API Integration**: Support for OpenAI, Anthropic, and other cloud-based models

### 📊 **Comprehensive Dataset Support**
- **Network Security**: UNSW-NB15, CIC-IDS datasets, SecRepo collections
- **Log Analysis**: Apache/Nginx logs, system logs, security event logs
- **Specialized Security**: Phishing detection, malware analysis, threat intelligence
- **Public Datasets**: 200,000+ labeled samples from Kaggle and HuggingFace

### 🖥️ **Hardware Requirements**
- **Target Platform**: MacBook Pro M4 Pro (12-14 core CPU, 16-20 core GPU)
- **Memory**: 24GB+ unified memory (48GB+ recommended)
- **Storage**: 1TB+ for dataset storage and model caching
- **Performance**: 12-15 tokens/sec for 7B models, ~30 minutes LoRA training

## Project Structure

```
├── src/benchmark/              # Core framework
│   ├── services/              # Service implementations
│   ├── models/                # Model management and plugins
│   ├── data/                  # Dataset handling and preprocessing
│   ├── evaluation/            # Evaluation metrics and engines
│   └── core/                  # Base classes and utilities
├── configs/                   # Configuration files
│   ├── experiments/           # Experiment configurations
│   ├── models/                # Model configurations
│   └── datasets/              # Dataset configurations
├── data/                      # Dataset storage
├── results/                   # Evaluation results
├── tests/                     # Comprehensive test suite
└── docs/                      # Documentation
```

## Academic Contributions

### Research Questions Addressed
1. How do state-of-the-art LLMs perform in real-world cybersecurity scenarios?
2. What is the trade-off between detection accuracy and false positive rates?
3. How robust are LLMs to adversarial attacks in cybersecurity contexts?
4. What factors most significantly impact LLM explainability in security analysis?

### Novel Methodological Contributions
- **Multi-dimensional Evaluation Framework**: Comprehensive cybersecurity-specific metrics
- **Automated Explainability Assessment**: LLM-as-judge evaluation for domain expertise
- **Hardware-Optimized Implementation**: Consumer-grade hardware compatibility
- **Public Dataset Integration**: Reproducible evaluation without data collection requirements

### Target Publication Venues
- **Tier 1 Security**: USENIX Security, CCS, NDSS, S&P
- **AI/ML Conferences**: NeurIPS, ICML (AI Security tracks)
- **Specialized Venues**: ACSAC, EuroS&P, RAID
- **Journals**: ACM TOPS, IEEE TDSC, Computers & Security

## Development Roadmap

The project follows a **12-phase development plan** spanning 16-20 weeks:

### Phase 1-2: Foundation (Weeks 1-3)
- Project structure and configuration management
- Base service interfaces and database models
- Development environment and CI/CD setup

### Phase 3: Data Service (Weeks 3-5)
- Dataset loading from multiple sources (Kaggle, HuggingFace, local)
- Data preprocessing and cybersecurity-specific feature extraction
- Caching and optimization for large datasets

### Phase 4: Model Service (Weeks 5-7)
- MLX local model integration for Apple Silicon
- API model plugins (OpenAI, Anthropic, Ollama)
- Performance monitoring and resource management

### Phase 5: Basic Evaluation (Weeks 7-8)
- Accuracy metrics implementation
- Performance evaluation framework
- Results storage and management

### Phase 6-12: Advanced Features (Weeks 9-20)
- Orchestration service and workflow management
- Advanced evaluation metrics (explainability, adversarial robustness)
- Reporting system and visualization
- API gateway and CLI interface
- Full system integration and optimization

## Evaluation Framework

### Supported Model Types
- **Local Models**: Llama 3.2 3B/7B, Qwen 2.5 7B, Mistral 7B (MLX-optimized)
- **Cybersecurity-Specific**: CySecBERT, SecBERT, SecureBERT
- **API Models**: GPT-4o-mini, Claude-3.5-Sonnet, Llama-3.1-70B

### Evaluation Pipeline
1. **Dataset Loading**: Automated download and preprocessing
2. **Model Inference**: Batch processing with cybersecurity-focused prompting
3. **Multi-Metric Assessment**: Parallel evaluation across all dimensions
4. **Results Aggregation**: Statistical analysis and significance testing
5. **Report Generation**: Comprehensive reports for academic publication

### Automated Evaluation Features
- **Reference-Based**: BLEU, ROUGE-L, BERTScore for explainability
- **LLM-as-Judge**: GPT-4 automated evaluation for explanation quality
- **Statistical Rigor**: 95% confidence intervals, cross-validation, significance testing
- **Reproducibility**: Version control for datasets, models, and configurations

## Getting Started

### Prerequisites
- macOS with Apple Silicon M4 Pro
- Python 3.11+
- Poetry for dependency management
- Git for version control

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd llm_cybersec_benchmark

# Install dependencies
poetry install

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run sample evaluation
poetry run benchmark run --config configs/experiments/basic_evaluation.yaml
```

## Configuration

The system uses YAML-based configuration with Pydantic validation:

```yaml
experiment:
  name: "Basic LLM Cybersecurity Evaluation"
  description: "Evaluate multiple models on public datasets"

datasets:
  - name: "unsw_nb15"
    source: "kaggle"
    path: "mrwellsdavid/unsw-nb15"
    max_samples: 10000

models:
  - name: "llama_3_2_3b"
    type: "mlx_local"
    path: "mlx-community/Llama-3.2-3B-Instruct-4bit"
  - name: "gpt_4o_mini"
    type: "openai_api"
    model: "gpt-4o-mini"

evaluation:
  metrics: ["accuracy", "explainability", "performance", "false_positive_rate"]
  parallel_jobs: 4
```

## Key Advantages for Academic Research

- **No Data Collection Required**: Leverages high-quality public datasets
- **No Expert Annotation**: Uses automated and LLM-as-judge evaluation
- **Consumer Hardware Compatible**: Optimized for Apple Silicon M4 Pro
- **Cost-Effective**: Minimal API costs through strategic model selection
- **Reproducible**: Detailed specifications and public dataset usage
- **Publication Ready**: Generates academic-quality reports and visualizations

## Contributing

This project follows academic research standards with comprehensive testing, documentation, and reproducibility requirements. See the development documentation for detailed implementation guidelines.

## License

Academic use license - suitable for research and educational purposes.

## Citation

```bibtex
@misc{llm_cybersec_benchmark_2025,
  title={LLM Cybersecurity Attack Detection Benchmarking System},
  author={[Your Names]},
  year={2025},
  note={Academic benchmarking framework for LLM cybersecurity evaluation}
}
```

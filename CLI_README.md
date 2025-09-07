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
- **Environment Resolution**: Dynamic variable substitution
- **Rich Output**: Beautiful terminal formatting
- **Error Handling**: Comprehensive error reporting with suggestions
- **Memory Management**: Intelligent memory usage optimization

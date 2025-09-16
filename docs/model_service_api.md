# Model Service API Documentation

The Model Service provides a comprehensive plugin-based architecture for managing and interacting with different types of Large Language Models (LLMs) in cybersecurity benchmarking contexts. It supports local models, API-based models, and provides advanced features like resource monitoring, performance optimization, and batch processing.

## Table of Contents

- [Overview](#overview)
- [Service Architecture](#service-architecture)
- [API Reference](#api-reference)
- [Model Plugins](#model-plugins)
- [Performance Optimization](#performance-optimization)
- [Resource Management](#resource-management)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Overview

The Model Service (`ModelService`) is a core component that enables:

- **Multi-Plugin Architecture**: Support for MLX, Ollama, OpenAI, Anthropic, and custom model providers
- **Resource Monitoring**: Real-time tracking of memory, CPU, and GPU usage
- **Performance Optimization**: Apple Silicon optimization, inference queuing, and request batching
- **Model Discovery**: Automatic discovery and validation of available models
- **Lifecycle Management**: Automated model loading, unloading, and cleanup

### Key Features

- ✅ **Universal Model Interface**: Consistent API across all model types
- ✅ **Performance Monitoring**: Real-time metrics collection and reporting
- ✅ **Resource Management**: Intelligent memory management and cleanup
- ✅ **Batch Processing**: Efficient batch inference capabilities
- ✅ **Cost Estimation**: API usage cost tracking and estimation
- ✅ **Health Monitoring**: Comprehensive service health checks

## Service Architecture

```
┌─────────────────────────────────────────────────┐
│                  ModelService                   │
├─────────────────────────────────────────────────┤
│  Plugin Registry    │  Resource Monitor         │
│  ┌─────────────────┐│ ┌───────────────────────┐  │
│  │ MLX Plugin      ││ │ Memory Tracker        │  │
│  │ Ollama Plugin   ││ │ CPU Monitor           │  │
│  │ OpenAI Plugin   ││ │ GPU Utilization       │  │
│  │ Anthropic Plugin││ │ Performance Metrics   │  │
│  │ Custom Plugins  ││ │ Cost Tracking         │  │
│  └─────────────────┘│ └───────────────────────┘  │
├─────────────────────────────────────────────────┤
│  Inference Engine   │  Optimization Layer       │
│  ┌─────────────────┐│ ┌───────────────────────┐  │
│  │ Request Queue   ││ │ Apple Silicon Opt     │  │
│  │ Batch Processor ││ │ Memory Management     │  │
│  │ Load Balancer   ││ │ Request Prioritization│  │
│  │ Result Cache    ││ │ Inference Queuing     │  │
│  └─────────────────┘│ └───────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## API Reference

### Service Lifecycle

#### `initialize(config: dict[str, Any]) -> ServiceResponse[dict[str, Any]]`

Initialize the model service with configuration parameters.

**Parameters:**
- `config`: Configuration dictionary with optional keys:
  - `max_concurrent_models` (int): Maximum concurrent loaded models (default: 3)
  - `default_timeout_seconds` (int): Default request timeout (default: 30)
  - `enable_performance_monitoring` (bool): Enable performance tracking (default: True)
  - `apple_silicon_optimization` (bool): Enable Apple Silicon optimizations (default: auto-detect)
  - `batch_size` (int): Default batch size for batch inference (default: 10)
  - `memory_threshold_mb` (int): Memory usage threshold for cleanup (default: 4096)

**Returns:**
- `ServiceResponse` with initialization status and configuration summary

**Example:**
```python
service = ModelService()
response = await service.initialize({
    "max_concurrent_models": 5,
    "enable_performance_monitoring": True,
    "apple_silicon_optimization": True,
    "batch_size": 32,
    "memory_threshold_mb": 8192
})
assert response.success
```

#### `shutdown() -> ServiceResponse[None]`

Gracefully shutdown the model service, unload all models, and cleanup resources.

**Returns:**
- `ServiceResponse` with shutdown status

### Model Discovery and Loading

#### `discover_models() -> ServiceResponse[ModelDiscoveryResult]`

Discover all available models across registered plugins.

**Returns:**
- `ServiceResponse` containing `ModelDiscoveryResult` with:
  - `available_models`: Dictionary mapping plugin type to available models
  - `total_models`: Total number of discovered models
  - `plugin_status`: Status of each model plugin

**Example:**
```python
discovery = await service.discover_models()
if discovery.success:
    result = discovery.data
    print(f"Found {result.total_models} total models")
    for plugin_type, models in result.available_models.items():
        print(f"{plugin_type}: {len(models)} models")
```

#### `load_model(model_id: str, config: dict[str, Any] | None = None, strategy: LoadingStrategy = LoadingStrategy.EAGER) -> ServiceResponse[ModelInfo]`

Load a model with the specified configuration and loading strategy.

**Parameters:**
- `model_id`: Unique identifier for the model to load
- `config`: Optional model-specific configuration
- `strategy`: Loading strategy (EAGER, LAZY, ON_DEMAND)

**Returns:**
- `ServiceResponse` containing `ModelInfo` with model details

**Example:**
```python
# Load an MLX model with custom configuration
config = {
    "max_tokens": 2048,
    "temperature": 0.1,
    "top_p": 0.95
}

response = await service.load_model(
    model_id="mlx://microsoft/DialoGPT-medium",
    config=config,
    strategy=LoadingStrategy.EAGER
)

if response.success:
    model_info = response.data
    print(f"Loaded: {model_info.name} ({model_info.type})")
```

#### `unload_model(model_id: str) -> ServiceResponse[dict[str, Any]]`

Unload a specific model and free its resources.

**Parameters:**
- `model_id`: ID of the model to unload

**Returns:**
- `ServiceResponse` with unloading status and resource cleanup summary

#### `get_loaded_models() -> ServiceResponse[list[EnhancedModelInfo]]`

Get information about all currently loaded models.

**Returns:**
- `ServiceResponse` containing list of `EnhancedModelInfo` objects with detailed model information

### Inference Operations

#### `predict(model_id: str, inputs: list[str], **kwargs) -> ServiceResponse[list[Prediction]]`

Perform inference on a list of inputs using the specified model.

**Parameters:**
- `model_id`: ID of the loaded model to use
- `inputs`: List of input texts for inference
- `**kwargs`: Additional inference parameters (temperature, max_tokens, etc.)

**Returns:**
- `ServiceResponse` containing list of `Prediction` objects

**Example:**
```python
inputs = [
    "Analyze this network log: 192.168.1.1 attempted connection to 10.0.0.1",
    "Email subject: 'Urgent: Update your banking information immediately'"
]

response = await service.predict(
    model_id="SecurityBERT_v2",
    inputs=inputs,
    temperature=0.1,
    max_tokens=512
)

if response.success:
    predictions = response.data
    for pred in predictions:
        print(f"Input: {pred.input_text[:50]}...")
        print(f"Prediction: {pred.prediction} (confidence: {pred.confidence:.3f})")
        if pred.explanation:
            print(f"Explanation: {pred.explanation}")
        print()
```

#### `batch_predict(requests: list[dict[str, Any]]) -> ServiceResponse[BatchInferenceResponse]`

Perform batch inference across multiple models and inputs efficiently.

**Parameters:**
- `requests`: List of inference requests, each containing:
  - `model_id`: Model to use for inference
  - `inputs`: List of input texts
  - `config`: Optional inference configuration
  - `metadata`: Optional metadata for tracking

**Returns:**
- `ServiceResponse` containing `BatchInferenceResponse` with:
  - `results`: List of prediction results
  - `processing_stats`: Batch processing statistics
  - `failed_requests`: Any failed requests with error details

**Example:**
```python
batch_requests = [
    {
        "model_id": "SecurityBERT",
        "inputs": ["Network traffic analysis request"],
        "config": {"temperature": 0.0}
    },
    {
        "model_id": "CyberLLM",
        "inputs": ["Malware signature detection task"],
        "config": {"temperature": 0.2}
    }
]

response = await service.batch_predict(batch_requests)
if response.success:
    batch_result = response.data
    print(f"Processed {len(batch_result.results)} predictions")
    print(f"Success rate: {batch_result.processing_stats.success_rate:.1%}")
```

### Performance and Monitoring

#### `get_performance_metrics(model_id: str | None = None) -> ServiceResponse[PerformanceMetrics | dict[str, PerformanceMetrics]]`

Retrieve performance metrics for a specific model or all loaded models.

**Parameters:**
- `model_id`: Optional model ID to get metrics for specific model

**Returns:**
- `ServiceResponse` containing performance metrics

#### `get_resource_usage() -> ServiceResponse[dict[str, Any]]`

Get current resource usage statistics for the model service.

**Returns:**
- `ServiceResponse` containing:
  - `memory_usage_mb`: Current memory usage
  - `cpu_usage_percent`: CPU utilization
  - `gpu_usage_percent`: GPU utilization (if available)
  - `active_models`: Number of loaded models
  - `inference_queue_size`: Current queue size

#### `compare_model_performance(model_ids: list[str], inputs: list[str]) -> ServiceResponse[PerformanceComparison]`

Compare performance across multiple models using the same input set.

**Parameters:**
- `model_ids`: List of model IDs to compare
- `inputs`: Common input set for comparison

**Returns:**
- `ServiceResponse` containing `PerformanceComparison` with detailed metrics

### Cost Management

#### `estimate_cost(model_id: str, inputs: list[str]) -> ServiceResponse[CostEstimate]`

Estimate the cost for inference requests (primarily for API-based models).

**Parameters:**
- `model_id`: Model to estimate costs for
- `inputs`: List of inputs to process

**Returns:**
- `ServiceResponse` containing `CostEstimate` with:
  - `estimated_cost_usd`: Estimated cost in USD
  - `token_usage`: Estimated token usage
  - `cost_breakdown`: Detailed cost breakdown

### Health and Diagnostics

#### `health_check() -> HealthCheck`

Perform comprehensive health check of the model service.

**Returns:**
- `HealthCheck` object with service status and detailed diagnostics

#### `validate_model(model_id: str) -> ServiceResponse[dict[str, Any]]`

Validate that a model is functioning correctly with test inputs.

**Parameters:**
- `model_id`: Model ID to validate

**Returns:**
- `ServiceResponse` with validation results

## Model Plugins

The Model Service supports multiple model providers through a plugin architecture:

### Supported Plugins

#### MLX Plugin (`mlx://model-name`)
- **Purpose**: Local Apple Silicon optimized models
- **Features**: Hardware acceleration, memory efficiency
- **Configuration**: Model path, quantization options, context length
- **Example**: `mlx://microsoft/DialoGPT-medium`

#### Ollama Plugin (`ollama://model-name`)
- **Purpose**: Local Ollama-managed models
- **Features**: Model versioning, easy deployment
- **Configuration**: Model name, parameters, system prompts
- **Example**: `ollama://llama2:7b`

#### OpenAI Plugin (`openai://model-name`)
- **Purpose**: OpenAI API models
- **Features**: Latest models, high reliability
- **Configuration**: API key, organization, model parameters
- **Example**: `openai://gpt-4`

#### Anthropic Plugin (`anthropic://model-name`)
- **Purpose**: Claude models via Anthropic API
- **Features**: Advanced reasoning, large context
- **Configuration**: API key, model version, safety settings
- **Example**: `anthropic://claude-3-sonnet`

### Plugin Configuration Examples

```python
# MLX Plugin Configuration
mlx_config = {
    "model_path": "/models/SecurityBERT_mlx",
    "quantization": "int4",
    "max_tokens": 2048,
    "context_length": 4096
}

# Ollama Plugin Configuration
ollama_config = {
    "model_name": "cybersecurity-llama",
    "system_prompt": "You are a cybersecurity expert. Analyze the following...",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.9
    }
}

# API Plugin Configuration
api_config = {
    "api_key": "${OPENAI_API_KEY}",  # Environment variable
    "organization": "cybersec-org",
    "model": "gpt-4",
    "max_tokens": 1024,
    "temperature": 0.0
}
```

### Creating Custom Plugins

To create a custom model plugin, implement the `ModelPlugin` interface:

```python
from benchmark.interfaces.model_interfaces import ModelPlugin, ModelInfo, Prediction
from typing import Any

class CustomSecurityModelPlugin(ModelPlugin):
    """Custom plugin for proprietary security models."""

    async def load_model(self, model_id: str, config: dict[str, Any]) -> ModelInfo:
        """Load the custom security model."""
        # Implementation for model loading
        return ModelInfo(
            model_id=model_id,
            name="Custom Security Model",
            type="custom_security",
            capabilities=["threat_detection", "malware_analysis"],
            parameters=config
        )

    async def predict(self, inputs: list[str], **kwargs) -> list[Prediction]:
        """Perform inference with the custom model."""
        predictions = []
        for input_text in inputs:
            # Custom inference logic
            result = await self._perform_custom_inference(input_text, **kwargs)

            prediction = Prediction(
                sample_id=f"custom_{len(predictions)}",
                input_text=input_text,
                prediction=result["prediction"],
                confidence=result["confidence"],
                attack_type=result.get("attack_type"),
                explanation=result.get("explanation"),
                inference_time_ms=result["inference_time_ms"]
            )
            predictions.append(prediction)

        return predictions

    async def unload_model(self) -> None:
        """Clean up model resources."""
        # Implementation for cleanup
        pass

    async def get_model_info(self) -> ModelInfo:
        """Get current model information."""
        # Implementation for model info retrieval
        pass

    async def validate_inputs(self, inputs: list[str]) -> bool:
        """Validate that inputs are compatible with this model."""
        # Implementation for input validation
        return True

    def get_supported_parameters(self) -> list[str]:
        """Get list of supported inference parameters."""
        return ["temperature", "max_tokens", "security_threshold"]

# Register the custom plugin
service = ModelService()
custom_plugin = CustomSecurityModelPlugin()
await service.register_plugin("custom_security", custom_plugin)
```

## Performance Optimization

### Apple Silicon Optimization

The service includes specialized optimizations for Apple Silicon Macs:

```python
# Enable Apple Silicon optimizations
config = {
    "apple_silicon_optimization": True,
    "memory_pressure_threshold": 0.8,
    "gpu_memory_fraction": 0.6
}

await service.initialize(config)

# Check if optimizations are active
resource_usage = await service.get_resource_usage()
print(f"Apple Silicon optimizations: {resource_usage.data.get('apple_silicon_enabled', False)}")
```

### Inference Queue Management

The service uses an intelligent inference queue for optimal performance:

```python
# Configure inference queue
queue_config = {
    "max_queue_size": 1000,
    "priority_levels": 3,
    "batch_timeout_ms": 100,
    "dynamic_batching": True
}

# Submit high-priority request
response = await service.predict(
    model_id="SecurityBERT",
    inputs=["Critical security alert analysis"],
    priority="high"  # Optional priority parameter
)
```

### Memory Management

Automatic memory management with configurable thresholds:

```python
# Memory management configuration
memory_config = {
    "memory_threshold_mb": 8192,
    "cleanup_interval_seconds": 300,
    "aggressive_cleanup": False,
    "model_cache_size": 3
}

# Manual memory cleanup
cleanup_response = await service.cleanup_unused_models()
print(f"Freed {cleanup_response.data['memory_freed_mb']} MB")
```

## Resource Management

### Resource Monitoring

The service provides comprehensive resource monitoring:

```python
async def monitor_resources():
    resource_stats = await service.get_resource_usage()
    stats = resource_stats.data

    print(f"Memory Usage: {stats['memory_usage_mb']} MB")
    print(f"CPU Usage: {stats['cpu_usage_percent']:.1f}%")
    print(f"Active Models: {stats['active_models']}")
    print(f"Queue Size: {stats['inference_queue_size']}")

    # Check for resource pressure
    if stats['memory_usage_mb'] > 8192:
        print("⚠️  High memory usage detected")
        cleanup_response = await service.cleanup_unused_models()
        print(f"Cleanup freed {cleanup_response.data['memory_freed_mb']} MB")

# Set up periodic monitoring
import asyncio
while True:
    await monitor_resources()
    await asyncio.sleep(60)  # Check every minute
```

### Model Lifecycle Management

Intelligent model lifecycle management:

```python
# Load models with different strategies
await service.load_model("fast-model", strategy=LoadingStrategy.EAGER)      # Load immediately
await service.load_model("heavy-model", strategy=LoadingStrategy.LAZY)       # Load on first use
await service.load_model("backup-model", strategy=LoadingStrategy.ON_DEMAND) # Load when needed

# Automatic cleanup based on usage patterns
cleanup_config = {
    "idle_timeout_minutes": 30,    # Unload after 30 minutes of inactivity
    "usage_threshold": 10,         # Keep models with >10 requests/hour
    "memory_pressure_cleanup": True # Aggressive cleanup when memory is low
}
```

## Usage Examples

### Basic Model Loading and Inference

```python
import asyncio
from benchmark.services.model_service import ModelService
from benchmark.interfaces.model_interfaces import LoadingStrategy

async def basic_inference_example():
    # Initialize service
    service = ModelService()
    await service.initialize({
        "max_concurrent_models": 3,
        "enable_performance_monitoring": True
    })

    # Discover available models
    discovery = await service.discover_models()
    print(f"Available models: {discovery.data.total_models}")

    # Load a security-focused model
    load_response = await service.load_model(
        model_id="mlx://security-bert-base",
        config={
            "temperature": 0.0,  # Deterministic for security analysis
            "max_tokens": 512
        },
        strategy=LoadingStrategy.EAGER
    )

    if load_response.success:
        print(f"Loaded model: {load_response.data.name}")

        # Perform cybersecurity analysis
        security_inputs = [
            "Network connection from 192.168.1.100 to 10.0.0.1 on port 22",
            "Email with attachment: invoice.pdf.exe",
            "Process spawned: powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden"
        ]

        predictions = await service.predict(
            model_id="mlx://security-bert-base",
            inputs=security_inputs
        )

        if predictions.success:
            for pred in predictions.data:
                print(f"Input: {pred.input_text[:50]}...")
                print(f"Threat Assessment: {pred.prediction}")
                print(f"Confidence: {pred.confidence:.3f}")
                if pred.attack_type:
                    print(f"Attack Type: {pred.attack_type}")
                print("---")

    # Cleanup
    await service.shutdown()

# Run the example
asyncio.run(basic_inference_example())
```

### Multi-Model Cybersecurity Pipeline

```python
async def cybersecurity_pipeline_example():
    service = ModelService()
    await service.initialize({"max_concurrent_models": 5})

    # Load specialized models for different security domains
    models = {
        "network_analyzer": "mlx://security-network-bert",
        "email_scanner": "ollama://cybersecurity-llama:7b",
        "malware_detector": "openai://gpt-4",
        "threat_classifier": "anthropic://claude-3-sonnet"
    }

    # Load all models
    for model_name, model_id in models.items():
        response = await service.load_model(model_id)
        print(f"Loaded {model_name}: {response.success}")

    # Process different types of security data
    security_data = {
        "network_logs": [
            "TCP connection attempt from external IP 203.0.113.1",
            "Multiple failed SSH login attempts from 198.51.100.2"
        ],
        "email_content": [
            "Urgent: Your account will be suspended unless you verify immediately",
            "Invoice attached - please process payment by end of day"
        ],
        "process_behavior": [
            "Unexpected network communication by notepad.exe",
            "Registry modification by unknown process"
        ]
    }

    # Analyze each data type with appropriate specialist model
    results = {}

    # Network analysis
    network_results = await service.predict(
        model_id=models["network_analyzer"],
        inputs=security_data["network_logs"]
    )
    results["network"] = network_results.data

    # Email analysis
    email_results = await service.predict(
        model_id=models["email_scanner"],
        inputs=security_data["email_content"]
    )
    results["email"] = email_results.data

    # Aggregate threat assessment
    all_threats = []
    for domain, predictions in results.items():
        for pred in predictions:
            if pred.prediction == "ATTACK":
                all_threats.append({
                    "domain": domain,
                    "threat": pred.attack_type,
                    "confidence": pred.confidence,
                    "details": pred.explanation
                })

    # Generate final security report
    print(f"\n🔐 Cybersecurity Analysis Report")
    print(f"{'='*50}")
    print(f"Total threats detected: {len(all_threats)}")

    for threat in sorted(all_threats, key=lambda x: x["confidence"], reverse=True):
        print(f"• {threat['domain'].upper()}: {threat['threat']} "
              f"(confidence: {threat['confidence']:.1%})")
        if threat['details']:
            print(f"  Details: {threat['details']}")

    await service.shutdown()
```

### Performance Comparison and Benchmarking

```python
async def model_performance_benchmark():
    service = ModelService()
    await service.initialize({
        "enable_performance_monitoring": True,
        "max_concurrent_models": 6
    })

    # Models to compare
    benchmark_models = [
        "mlx://security-bert-small",
        "mlx://security-bert-large",
        "ollama://cybersecurity-llama:7b",
        "openai://gpt-3.5-turbo"
    ]

    # Load all models
    for model_id in benchmark_models:
        await service.load_model(model_id)

    # Common test dataset
    test_inputs = [
        "Suspicious network activity detected on port 443",
        "Email attachment with double extension: document.pdf.exe",
        "Unusual process spawning pattern detected",
        "Multiple authentication failures from same IP",
        "Encrypted traffic to known malicious domain"
    ] * 20  # 100 total inputs

    # Run performance comparison
    comparison_result = await service.compare_model_performance(
        model_ids=benchmark_models,
        inputs=test_inputs
    )

    if comparison_result.success:
        comparison = comparison_result.data
        print("\n📊 Model Performance Comparison")
        print("="*60)

        for model_id, metrics in comparison.model_metrics.items():
            print(f"\n{model_id}:")
            print(f"  Average Inference Time: {metrics.average_inference_time_ms:.1f} ms")
            print(f"  Throughput: {metrics.predictions_per_second:.1f} pred/sec")
            print(f"  Memory Usage: {metrics.memory_usage_mb:.1f} MB")
            print(f"  Success Rate: {metrics.success_rate:.1%}")

        # Performance ranking
        print(f"\n🏆 Performance Rankings:")
        print(f"Fastest: {comparison.fastest_model}")
        print(f"Most Accurate: {comparison.most_accurate_model}")
        print(f"Most Efficient: {comparison.most_efficient_model}")
```

## Advanced Features

### Batch Processing with Smart Queuing

```python
async def advanced_batch_processing():
    service = ModelService()
    await service.initialize({
        "batch_size": 32,
        "dynamic_batching": True,
        "batch_timeout_ms": 50
    })

    await service.load_model("SecurityBERT")

    # Create varied batch requests
    batch_requests = []
    for i in range(100):
        request = {
            "model_id": "SecurityBERT",
            "inputs": [f"Security analysis request {i}"],
            "config": {"temperature": 0.0},
            "metadata": {
                "priority": "high" if i < 10 else "normal",
                "batch_id": i // 10
            }
        }
        batch_requests.append(request)

    # Process with intelligent batching
    batch_response = await service.batch_predict(batch_requests)

    if batch_response.success:
        stats = batch_response.data.processing_stats
        print(f"Processed {stats.total_requests} requests")
        print(f"Batch efficiency: {stats.batch_utilization:.1%}")
        print(f"Average latency: {stats.average_latency_ms:.1f} ms")
        print(f"Throughput: {stats.throughput_per_second:.1f} req/sec")
```

### Cost-Aware Model Selection

```python
async def cost_aware_inference():
    service = ModelService()
    await service.initialize()

    # Load models with different cost profiles
    models = [
        {"id": "local://fast-model", "cost_per_token": 0.0},      # Local model
        {"id": "openai://gpt-3.5-turbo", "cost_per_token": 0.002},
        {"id": "openai://gpt-4", "cost_per_token": 0.03}
    ]

    for model in models:
        await service.load_model(model["id"])

    inputs = ["Analyze this security event"] * 100

    # Estimate costs for each model
    cost_estimates = {}
    for model in models:
        estimate_response = await service.estimate_cost(model["id"], inputs)
        if estimate_response.success:
            cost_estimates[model["id"]] = estimate_response.data

    # Select most cost-effective model meeting performance requirements
    print("💰 Cost Analysis:")
    for model_id, estimate in cost_estimates.items():
        print(f"{model_id}: ${estimate.estimated_cost_usd:.4f} "
              f"({estimate.token_usage} tokens)")

    # Use cheapest model for batch processing
    cheapest_model = min(cost_estimates.items(),
                        key=lambda x: x[1].estimated_cost_usd)[0]

    print(f"Selected model: {cheapest_model}")
    batch_response = await service.predict(cheapest_model, inputs)
```

### Health Monitoring and Auto-Recovery

```python
async def health_monitoring_example():
    service = ModelService()
    await service.initialize()
    await service.load_model("SecurityBERT")

    async def health_monitor():
        while True:
            health = await service.health_check()

            if health.status != ServiceStatus.HEALTHY:
                print(f"⚠️  Service unhealthy: {health.status}")
                print(f"Issues: {health.details}")

                # Attempt recovery
                if "memory" in str(health.details).lower():
                    print("Attempting memory cleanup...")
                    await service.cleanup_unused_models()

                elif "model_load_failure" in str(health.details):
                    print("Attempting model reload...")
                    models = await service.get_loaded_models()
                    for model in models.data:
                        if not model.status == "healthy":
                            await service.unload_model(model.model_id)
                            await service.load_model(model.model_id)

            else:
                print(f"✅ Service healthy")

            await asyncio.sleep(30)  # Check every 30 seconds

    # Start monitoring task
    monitor_task = asyncio.create_task(health_monitor())

    # Run application workload
    try:
        while True:
            inputs = ["Security analysis task"]
            await service.predict("SecurityBERT", inputs)
            await asyncio.sleep(5)
    finally:
        monitor_task.cancel()
```

## Error Handling

### Common Error Types and Handling

```python
from benchmark.core.exceptions import (
    BenchmarkError,
    ModelLoadingError,
    ServiceUnavailableError,
    ValidationError
)

async def robust_inference_example():
    service = ModelService()

    try:
        # Initialize with error handling
        init_response = await service.initialize()
        if not init_response.success:
            raise BenchmarkError(f"Failed to initialize: {init_response.error}")

        # Load model with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                load_response = await service.load_model("SecurityBERT")
                if load_response.success:
                    break
            except ModelLoadingError as e:
                print(f"Load attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # Inference with graceful degradation
        inputs = ["Security analysis request"]

        try:
            predictions = await service.predict("SecurityBERT", inputs)
            if predictions.success:
                return predictions.data

        except ServiceUnavailableError:
            print("Primary model unavailable, trying fallback...")
            # Try fallback model
            fallback_response = await service.predict("BackupModel", inputs)
            return fallback_response.data

        except ValidationError as e:
            print(f"Input validation failed: {e}")
            # Clean up inputs and retry
            cleaned_inputs = [inp.strip()[:1000] for inp in inputs if inp.strip()]
            return await service.predict("SecurityBERT", cleaned_inputs)

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Perform cleanup
        await service.shutdown()
        raise

# Structured error handling with specific recovery strategies
async def handle_model_errors(service: ModelService, model_id: str, inputs: list[str]):
    try:
        return await service.predict(model_id, inputs)

    except ModelLoadingError:
        # Try to reload the model
        print("Reloading model...")
        await service.unload_model(model_id)
        await service.load_model(model_id)
        return await service.predict(model_id, inputs)

    except ServiceUnavailableError:
        # Wait and retry with backoff
        print("Service busy, retrying...")
        await asyncio.sleep(1)
        return await service.predict(model_id, inputs)

    except ValidationError as e:
        # Fix input format and retry
        print(f"Fixing input validation: {e}")
        fixed_inputs = [inp[:500] for inp in inputs if len(inp.strip()) > 0]
        return await service.predict(model_id, fixed_inputs)
```

## Best Practices

### Service Configuration for Production

```python
# Production configuration
production_config = {
    # Performance settings
    "max_concurrent_models": 8,  # Based on available memory
    "default_timeout_seconds": 120,
    "batch_size": 16,
    "enable_performance_monitoring": True,

    # Resource management
    "memory_threshold_mb": 12288,  # 12GB threshold
    "cleanup_interval_seconds": 300,
    "aggressive_cleanup": True,

    # Optimization
    "apple_silicon_optimization": True,
    "dynamic_batching": True,
    "batch_timeout_ms": 100,

    # Reliability
    "health_check_interval": 60,
    "auto_recovery_enabled": True,
    "model_validation_on_load": True
}

# Development configuration
dev_config = {
    "max_concurrent_models": 2,
    "default_timeout_seconds": 30,
    "enable_performance_monitoring": False,
    "memory_threshold_mb": 4096,
    "apple_silicon_optimization": False
}
```

### Model Selection Guidelines

1. **Local Models (MLX/Ollama)**: Best for privacy, low latency, cost efficiency
2. **API Models (OpenAI/Anthropic)**: Best for accuracy, latest capabilities, no hardware requirements
3. **Hybrid Approach**: Use local models for bulk processing, API models for complex analysis

### Performance Optimization Tips

```python
# 1. Batch similar requests
batch_requests = group_by_model_and_config(all_requests)
for batch in batch_requests:
    results = await service.batch_predict(batch)

# 2. Use appropriate model sizes
light_analysis = await service.predict("small-model", simple_inputs)
complex_analysis = await service.predict("large-model", complex_inputs)

# 3. Cache frequently used models
cache_models = ["SecurityBERT", "ThreatAnalyzer", "MalwareDetector"]
for model_id in cache_models:
    await service.load_model(model_id, strategy=LoadingStrategy.EAGER)

# 4. Monitor and tune performance
performance_metrics = await service.get_performance_metrics()
if performance_metrics.data["SecurityBERT"].average_inference_time_ms > 500:
    print("Consider using a smaller/faster model variant")
```

### Resource Management Best Practices

```python
# 1. Regular health monitoring
async def setup_monitoring(service: ModelService):
    async def monitor():
        while True:
            health = await service.health_check()
            resource_usage = await service.get_resource_usage()

            # Log metrics
            logger.info(f"Service health: {health.status}")
            logger.info(f"Memory usage: {resource_usage.data['memory_usage_mb']} MB")

            # Alert on issues
            if resource_usage.data['memory_usage_mb'] > 10240:  # 10GB
                logger.warning("High memory usage detected")

            await asyncio.sleep(60)

    return asyncio.create_task(monitor())

# 2. Graceful shutdown
async def graceful_shutdown(service: ModelService):
    # Stop accepting new requests
    service.stop_accepting_requests()

    # Wait for active requests to complete
    await service.wait_for_completion(timeout=30)

    # Unload models and cleanup
    await service.shutdown()

# 3. Error recovery
async def setup_error_recovery(service: ModelService):
    async def recovery_handler():
        health = await service.health_check()
        if health.status == ServiceStatus.DEGRADED:
            await service.cleanup_unused_models()
            await service.restart_failed_models()

    # Schedule periodic recovery checks
    while True:
        await recovery_handler()
        await asyncio.sleep(300)  # Every 5 minutes
```

---

## See Also

- [Evaluation Service API Documentation](evaluation_service_api.md)
- [Integration Guide](integration_guide.md)
- [Performance Benchmarks](../benchmarks/performance_results.md)
- [Model Interface Reference](../src/benchmark/interfaces/model_interfaces.py)
- [Plugin Examples](../examples/)

# Evaluation Service API Documentation

The Evaluation Service provides a comprehensive plugin-based architecture for evaluating model predictions against ground truth data using various metrics. This service supports concurrent evaluations, result tracking, and performance monitoring.

## Table of Contents

- [Overview](#overview)
- [Service Architecture](#service-architecture)
- [API Reference](#api-reference)
- [Metric Types](#metric-types)
- [Plugin System](#plugin-system)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Overview

The Evaluation Service (`EvaluationService`) is a core component of the LLM Cybersecurity Benchmark system that enables:

- **Multi-metric Evaluation**: Support for accuracy, precision, recall, F1-score, ROC-AUC, and custom metrics
- **Concurrent Processing**: Parallel evaluation of multiple requests with configurable limits
- **Result Tracking**: Comprehensive history and summary generation
- **Plugin Architecture**: Extensible metric evaluator system
- **Performance Monitoring**: Built-in tracking of evaluation performance and resource usage

### Key Features

- ✅ **Async/Await Support**: Full asynchronous operation for high-performance evaluations
- ✅ **Type Safety**: Comprehensive type hints and validation using Pydantic
- ✅ **Resource Management**: Automatic cleanup and resource monitoring
- ✅ **Error Recovery**: Robust error handling with detailed error reporting
- ✅ **Historical Analytics**: Complete evaluation history with filtering and aggregation

## Service Architecture

```
┌─────────────────────────────────────────────────┐
│                EvaluationService                │
├─────────────────────────────────────────────────┤
│  Plugin Registry    │  Evaluation Engine        │
│  ┌─────────────────┐│ ┌───────────────────────┐  │
│  │ Accuracy        ││ │ Request Queue         │  │
│  │ Precision       ││ │ Concurrent Executor   │  │
│  │ Performance     ││ │ Progress Tracking     │  │
│  │ Custom Metrics  ││ │ Result Aggregator     │  │
│  └─────────────────┘│ └───────────────────────┘  │
├─────────────────────────────────────────────────┤
│  Result Storage     │  Analytics Engine         │
│  ┌─────────────────┐│ ┌───────────────────────┐  │
│  │ History Buffer  ││ │ Summary Generator     │  │
│  │ Filtering       ││ │ Statistics Computer   │  │
│  │ Serialization   ││ │ Performance Monitor   │  │
│  └─────────────────┘│ └───────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## API Reference

### Service Lifecycle

#### `initialize(config: dict[str, Any]) -> ServiceResponse[dict[str, Any]]`

Initialize the evaluation service with configuration parameters.

**Parameters:**
- `config`: Configuration dictionary with the following optional keys:
  - `max_concurrent_evaluations` (int): Maximum concurrent evaluations (default: 5)
  - `evaluation_timeout_seconds` (float): Timeout for individual evaluations (default: 30.0)
  - `max_history_size` (int): Maximum number of results to keep in memory (default: 1000)

**Returns:**
- `ServiceResponse` with initialization status and configuration summary

**Example:**
```python
service = EvaluationService()
response = await service.initialize({
    "max_concurrent_evaluations": 10,
    "evaluation_timeout_seconds": 60.0,
    "max_history_size": 2000
})
assert response.success
```

#### `shutdown() -> ServiceResponse[dict[str, Any]]`

Gracefully shutdown the evaluation service and cleanup resources.

**Returns:**
- `ServiceResponse` with shutdown status and cleanup summary

### Metric Evaluator Management

#### `register_evaluator(metric_type: MetricType, evaluator: MetricEvaluator) -> ServiceResponse[dict[str, str]]`

Register a metric evaluator for the specified metric type.

**Parameters:**
- `metric_type`: The type of metric this evaluator handles
- `evaluator`: Implementation of the `MetricEvaluator` interface

**Returns:**
- `ServiceResponse` containing registration details

**Example:**
```python
from benchmark.evaluation.metrics.accuracy import AccuracyEvaluator

accuracy_evaluator = AccuracyEvaluator()
response = await service.register_evaluator(
    MetricType.ACCURACY,
    accuracy_evaluator
)
```

#### `get_available_metrics() -> ServiceResponse[dict[str, Any]]`

Retrieve information about all registered metric evaluators.

**Returns:**
- `ServiceResponse` containing:
  - `total_evaluators`: Number of registered evaluators
  - `metrics`: List of available metric types
  - `evaluator_info`: Detailed information about each evaluator

### Evaluation Operations

#### `evaluate_predictions(request: EvaluationRequest) -> EvaluationResult`

Perform evaluation of model predictions against ground truth data.

**Parameters:**
- `request`: `EvaluationRequest` object containing:
  - `experiment_id`: Unique identifier for the experiment
  - `model_id`: Identifier for the model being evaluated
  - `dataset_id`: Identifier for the dataset being used
  - `predictions`: List of model predictions
  - `ground_truth`: List of ground truth values
  - `metrics`: List of metrics to compute
  - `metadata`: Additional metadata for the evaluation

**Returns:**
- `EvaluationResult` object with computed metrics and metadata

**Example:**
```python
request = EvaluationRequest(
    experiment_id="cybersec_eval_001",
    model_id="SecurityBERT_v2",
    dataset_id="network_intrusion_detection",
    predictions=[
        {"predicted_class": "dos", "confidence": 0.95},
        {"predicted_class": "normal", "confidence": 0.88}
    ],
    ground_truth=[
        {"true_class": "dos"},
        {"true_class": "normal"}
    ],
    metrics=[MetricType.ACCURACY, MetricType.PRECISION],
    metadata={
        "experiment_type": "baseline_comparison",
        "dataset_version": "v2.1"
    }
)

result = await service.evaluate_predictions(request)
print(f"Accuracy: {result.get_metric_value('accuracy'):.3f}")
```

### Result Management

#### `get_evaluation_history(limit: int = 100, model_id: str | None = None, dataset_id: str | None = None) -> ServiceResponse[dict[str, Any]]`

Retrieve evaluation history with optional filtering.

**Parameters:**
- `limit`: Maximum number of results to return
- `model_id`: Filter by specific model ID
- `dataset_id`: Filter by specific dataset ID

**Returns:**
- `ServiceResponse` containing:
  - `results`: List of evaluation results
  - `total_results`: Total number of results available
  - `filters_applied`: Summary of applied filters

#### `get_evaluation_summary(days_back: int = 7) -> ServiceResponse[dict[str, Any]]`

Generate comprehensive summary of evaluation history.

**Parameters:**
- `days_back`: Number of days to include in summary

**Returns:**
- `ServiceResponse` containing:
  - `total_evaluations`: Total number of evaluations
  - `successful_evaluations`: Number of successful evaluations
  - `failed_evaluations`: Number of failed evaluations
  - `average_execution_time`: Average evaluation execution time
  - `metric_summaries`: Statistical summaries for each metric type
  - `models_evaluated`: List of unique models evaluated
  - `datasets_evaluated`: List of unique datasets evaluated
  - `time_range`: Time range covered by the summary

### Health Monitoring

#### `health_check() -> HealthCheck`

Perform comprehensive health check of the evaluation service.

**Returns:**
- `HealthCheck` object with service status and diagnostic information

## Metric Types

The evaluation service supports the following metric types through the `MetricType` enum:

### Classification Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `ACCURACY` | Overall classification accuracy | General performance assessment |
| `PRECISION` | Positive predictive value | When false positives are costly |
| `RECALL` | True positive rate | When false negatives are critical |
| `F1_SCORE` | Harmonic mean of precision and recall | Balanced precision/recall importance |
| `ROC_AUC` | Area under the ROC curve | Binary classification performance |

### Performance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `PERFORMANCE` | Inference speed and resource usage | Performance optimization |
| `FALSE_POSITIVE_RATE` | Rate of false positive predictions | Security-critical applications |

### Advanced Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `CONFUSION_MATRIX` | Complete classification breakdown | Detailed error analysis |
| `EXPLAINABILITY` | Model explanation quality | Interpretability assessment |

## Plugin System

### Creating Custom Metric Evaluators

To create a custom metric evaluator, implement the `MetricEvaluator` interface:

```python
from benchmark.interfaces.evaluation_interfaces import MetricEvaluator, MetricType
from typing import Any

class CustomSecurityMetricEvaluator(MetricEvaluator):
    """Custom evaluator for cybersecurity-specific metrics."""

    async def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute custom security metrics."""
        # Implementation of custom metric computation
        threat_detection_rate = self._compute_threat_detection_rate(
            predictions, ground_truth
        )
        false_alarm_rate = self._compute_false_alarm_rate(
            predictions, ground_truth
        )

        return {
            "threat_detection_rate": threat_detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "security_score": threat_detection_rate - false_alarm_rate
        }

    def get_metric_names(self) -> list[str]:
        return ["threat_detection_rate", "false_alarm_rate", "security_score"]

    def get_required_prediction_fields(self) -> list[str]:
        return ["predicted_class", "confidence", "threat_level"]

    def get_required_ground_truth_fields(self) -> list[str]:
        return ["true_class", "severity"]

    def get_metric_type(self) -> MetricType:
        return MetricType.ACCURACY  # Use existing enum or extend

    def _compute_threat_detection_rate(self, predictions, ground_truth) -> float:
        # Custom implementation
        pass

    def _compute_false_alarm_rate(self, predictions, ground_truth) -> float:
        # Custom implementation
        pass

# Register the custom evaluator
custom_evaluator = CustomSecurityMetricEvaluator()
await service.register_evaluator(MetricType.ACCURACY, custom_evaluator)
```

## Usage Examples

### Basic Evaluation Workflow

```python
import asyncio
from benchmark.services.evaluation_service import EvaluationService
from benchmark.interfaces.evaluation_interfaces import EvaluationRequest, MetricType
from benchmark.evaluation.metrics.accuracy import AccuracyEvaluator

async def basic_evaluation_example():
    # Initialize service
    service = EvaluationService()
    await service.initialize({
        "max_concurrent_evaluations": 5,
        "evaluation_timeout_seconds": 30.0
    })

    # Register evaluators
    await service.register_evaluator(MetricType.ACCURACY, AccuracyEvaluator())

    # Prepare evaluation data
    predictions = [
        {"predicted_class": "attack", "confidence": 0.95},
        {"predicted_class": "benign", "confidence": 0.88},
        {"predicted_class": "attack", "confidence": 0.92}
    ]

    ground_truth = [
        {"true_class": "attack"},
        {"true_class": "benign"},
        {"true_class": "benign"}
    ]

    # Create evaluation request
    request = EvaluationRequest(
        experiment_id="basic_eval_example",
        model_id="example_model",
        dataset_id="example_dataset",
        predictions=predictions,
        ground_truth=ground_truth,
        metrics=[MetricType.ACCURACY],
        metadata={"example": True}
    )

    # Perform evaluation
    result = await service.evaluate_predictions(request)

    print(f"Evaluation completed: {result.success}")
    print(f"Accuracy: {result.get_metric_value('accuracy'):.3f}")
    print(f"Execution time: {result.execution_time_seconds:.2f}s")

    # Cleanup
    await service.shutdown()

# Run the example
asyncio.run(basic_evaluation_example())
```

### Multi-Model Comparison

```python
async def multi_model_comparison():
    service = EvaluationService()
    await service.initialize({"max_concurrent_evaluations": 8})

    # Register multiple evaluators
    await service.register_evaluator(MetricType.ACCURACY, AccuracyEvaluator())
    await service.register_evaluator(MetricType.PRECISION, PrecisionEvaluator())

    models = ["SecurityBERT", "CyberLSTM", "ThreatCNN"]
    results = []

    for model_id in models:
        request = EvaluationRequest(
            experiment_id=f"comparison_{model_id}",
            model_id=model_id,
            dataset_id="cybersec_benchmark",
            predictions=load_predictions(model_id),
            ground_truth=load_ground_truth(),
            metrics=[MetricType.ACCURACY, MetricType.PRECISION],
            metadata={"comparison_study": True}
        )

        result = await service.evaluate_predictions(request)
        results.append(result)

    # Generate comparison report
    print("Model Comparison Results:")
    for result in results:
        print(f"{result.model_id:15} | "
              f"Accuracy: {result.get_metric_value('accuracy'):.3f} | "
              f"Precision: {result.get_metric_value('precision'):.3f}")
```

### Concurrent Evaluation with Progress Tracking

```python
async def concurrent_evaluation_example():
    service = EvaluationService()
    await service.initialize({"max_concurrent_evaluations": 10})

    # Register evaluators
    await service.register_evaluator(MetricType.ACCURACY, AccuracyEvaluator())

    # Create multiple evaluation requests
    requests = []
    for i in range(20):
        request = EvaluationRequest(
            experiment_id=f"concurrent_eval_{i:02d}",
            model_id=f"model_{i % 4}",  # 4 different models
            dataset_id=f"dataset_{i % 3}",  # 3 different datasets
            predictions=generate_sample_predictions(50),
            ground_truth=generate_sample_ground_truth(50),
            metrics=[MetricType.ACCURACY],
            metadata={"batch": i // 5}
        )
        requests.append(request)

    # Execute all evaluations concurrently
    tasks = [service.evaluate_predictions(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Analyze results
    successful = [r for r in results if hasattr(r, 'success') and r.success]
    failed = [r for r in results if isinstance(r, Exception)]

    print(f"Completed: {len(successful)}/{len(requests)}")
    print(f"Failed: {len(failed)}")
    print(f"Average accuracy: {sum(r.get_metric_value('accuracy') for r in successful) / len(successful):.3f}")
```

## Performance Considerations

### Concurrent Evaluation Limits

The service supports configurable concurrent evaluation limits to balance performance and resource usage:

```python
# High-throughput configuration
await service.initialize({
    "max_concurrent_evaluations": 20,  # Increase for more parallelism
    "evaluation_timeout_seconds": 120.0  # Allow longer evaluations
})
```

### Memory Management

- **History Buffer**: Configure `max_history_size` to control memory usage
- **Automatic Cleanup**: Service automatically cleans up completed evaluations
- **Resource Monitoring**: Built-in tracking of memory and CPU usage

### Performance Optimization Tips

1. **Batch Similar Evaluations**: Group evaluations by model or dataset for better cache efficiency
2. **Use Appropriate Concurrency**: Set `max_concurrent_evaluations` based on available CPU cores
3. **Monitor Resource Usage**: Use health checks to monitor service performance
4. **Optimize Evaluator Implementations**: Ensure custom evaluators are efficient and use appropriate data structures

### Benchmark Performance

Expected performance characteristics:

| Metric | Performance Target | Notes |
|--------|-------------------|--------|
| Throughput | 50-200 evaluations/second | Depends on metric complexity |
| Latency | < 100ms per evaluation | For simple accuracy metrics |
| Memory Usage | < 500MB for 1000 results | With default configuration |
| Concurrent Limit | Up to 50 concurrent evaluations | Hardware dependent |

## Error Handling

### Common Error Types

The evaluation service provides comprehensive error handling with specific error types:

#### `ValidationError`
- **Cause**: Invalid input data or configuration
- **Resolution**: Validate input data format and completeness

#### `ServiceUnavailableError`
- **Cause**: Service overloaded or shutting down
- **Resolution**: Retry with backoff or reduce concurrent requests

#### `EvaluationTimeoutError`
- **Cause**: Evaluation exceeded timeout limit
- **Resolution**: Increase timeout or optimize evaluator implementation

#### `MetricNotSupportedError`
- **Cause**: Requested metric type not registered
- **Resolution**: Register required metric evaluator

### Error Handling Examples

```python
from benchmark.core.exceptions import BenchmarkError, ErrorCode

try:
    result = await service.evaluate_predictions(request)
except BenchmarkError as e:
    if e.error_code == ErrorCode.VALIDATION_ERROR:
        print(f"Invalid input data: {e.message}")
        # Handle validation errors
    elif e.error_code == ErrorCode.SERVICE_UNAVAILABLE:
        print(f"Service overloaded: {e.message}")
        # Implement retry logic
    else:
        print(f"Unexpected error: {e.message}")
        # Handle other errors

# Check result status
if not result.success:
    print(f"Evaluation failed: {result.error_message}")
```

### Graceful Degradation

The service implements graceful degradation strategies:

1. **Partial Results**: Return available metrics even if some evaluators fail
2. **Timeout Handling**: Individual evaluator timeouts don't crash the entire evaluation
3. **Resource Limits**: Automatically throttle requests when resource limits are reached
4. **Health Monitoring**: Continuous health checks with automatic recovery

## Best Practices

### Service Configuration

```python
# Production configuration
production_config = {
    "max_concurrent_evaluations": 16,  # Match CPU cores
    "evaluation_timeout_seconds": 300.0,  # 5 minutes for complex metrics
    "max_history_size": 5000,  # Balance memory vs. history retention
}

# Development configuration
dev_config = {
    "max_concurrent_evaluations": 4,
    "evaluation_timeout_seconds": 30.0,
    "max_history_size": 100,
}
```

### Request Design

1. **Meaningful IDs**: Use descriptive experiment, model, and dataset IDs
2. **Structured Metadata**: Include relevant context in metadata for later analysis
3. **Appropriate Sample Sizes**: Balance evaluation accuracy with performance
4. **Consistent Data Formats**: Ensure predictions and ground truth use consistent schemas

### Evaluation Workflow

1. **Initialize Once**: Create and configure service instance at application startup
2. **Register All Metrics**: Register all needed evaluators before starting evaluations
3. **Batch Related Evaluations**: Group similar evaluations for better performance
4. **Monitor Health**: Regular health checks in production environments
5. **Graceful Shutdown**: Always call shutdown() for proper cleanup

### Monitoring and Debugging

```python
# Health monitoring
health = await service.health_check()
if health.status != ServiceStatus.HEALTHY:
    print(f"Service unhealthy: {health.details}")

# Performance monitoring
summary = await service.get_evaluation_summary(days_back=1)
print(f"Average execution time: {summary.data['average_execution_time']:.2f}s")
print(f"Success rate: {summary.data['successful_evaluations'] / summary.data['total_evaluations']:.1%}")

# Resource usage tracking
available_metrics = await service.get_available_metrics()
print(f"Registered evaluators: {available_metrics.data['total_evaluators']}")
```

---

## See Also

- [Model Service API Documentation](model_service_api.md)
- [Integration Guide](integration_guide.md)
- [Performance Benchmarks](../benchmarks/performance_results.md)
- [Evaluation Interface Reference](../src/benchmark/interfaces/evaluation_interfaces.py)
- [Example Implementations](../examples/)

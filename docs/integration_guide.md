# LLM Cybersecurity Benchmark Integration Guide

This comprehensive guide demonstrates how to integrate the LLM Cybersecurity Benchmark system into your applications, covering everything from basic setup to advanced production deployment patterns.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Service Integration](#service-integration)
- [Configuration Management](#configuration-management)
- [Data Pipeline Integration](#data-pipeline-integration)
- [Model Management](#model-management)
- [Evaluation Workflows](#evaluation-workflows)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation and Basic Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/llm-cybersec-benchmark.git
cd llm-cybersec-benchmark

# 2. Install dependencies
pip install -r requirements.txt
# OR using poetry
poetry install

# 3. Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# 4. Initialize configuration
python -m benchmark.cli.main init --config-path configs/default.yaml

# 5. Run a basic evaluation
python -m benchmark.cli.main evaluate --experiment basic_test --config configs/experiments/basic_evaluation.yaml
```

### Minimal Integration Example

```python
import asyncio
from benchmark.services.model_service import ModelService
from benchmark.services.evaluation_service import EvaluationService
from benchmark.interfaces.evaluation_interfaces import EvaluationRequest, MetricType

async def minimal_integration():
    # Initialize services
    model_service = ModelService()
    eval_service = EvaluationService()

    await model_service.initialize()
    await eval_service.initialize()

    # Load a model
    await model_service.load_model("ollama://llama2:7b")

    # Register evaluators
    from benchmark.evaluation.metrics.accuracy import AccuracyEvaluator
    await eval_service.register_evaluator(MetricType.ACCURACY, AccuracyEvaluator())

    # Perform inference
    inputs = ["Analyze this network log: Suspicious connection to 10.0.0.1"]
    predictions = await model_service.predict("ollama://llama2:7b", inputs)

    # Evaluate results
    request = EvaluationRequest(
        experiment_id="minimal_test",
        model_id="ollama://llama2:7b",
        dataset_id="test_data",
        predictions=[{"predicted_class": "attack", "confidence": 0.95}],
        ground_truth=[{"true_class": "attack"}],
        metrics=[MetricType.ACCURACY],
        metadata={}
    )

    result = await eval_service.evaluate_predictions(request)
    print(f"Accuracy: {result.get_metric_value('accuracy'):.3f}")

    # Cleanup
    await model_service.shutdown()
    await eval_service.shutdown()

# Run minimal example
asyncio.run(minimal_integration())
```

## Architecture Overview

The LLM Cybersecurity Benchmark system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface  │  API Server   │  Jupyter Notebooks │ Custom Apps │
├─────────────────────────────────────────────────────────────────┤
│                         Service Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Model Service  │ Evaluation    │ Data Service │ Configuration  │
│                │ Service       │              │ Service        │
├─────────────────────────────────────────────────────────────────┤
│                        Plugin Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  MLX Plugin    │ Ollama Plugin │ OpenAI Plugin │ Custom Plugins │
│  Accuracy      │ Precision     │ Performance   │ Custom Metrics │
│  Evaluator     │ Evaluator     │ Evaluator     │                │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure Layer                       │
├─────────────────────────────────────────────────────────────────┤
│  Database      │ File Storage  │ Caching       │ Resource       │
│  Management    │               │ Layer         │ Monitoring     │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

- **Service Layer**: High-level business logic and orchestration
- **Plugin Layer**: Extensible components for models and metrics
- **Infrastructure Layer**: Storage, caching, and monitoring
- **Configuration System**: Centralized configuration management

## Service Integration

### Dependency Injection Pattern

Use dependency injection for clean service integration:

```python
from typing import Protocol
from benchmark.services import ModelService, EvaluationService, DataService

class BenchmarkServices(Protocol):
    """Protocol defining the benchmark service interface."""
    model_service: ModelService
    evaluation_service: EvaluationService
    data_service: DataService

class BenchmarkContainer:
    """Dependency injection container for benchmark services."""

    def __init__(self):
        self._model_service = None
        self._evaluation_service = None
        self._data_service = None

    @property
    async def model_service(self) -> ModelService:
        if self._model_service is None:
            self._model_service = ModelService()
            await self._model_service.initialize(self._get_model_config())
        return self._model_service

    @property
    async def evaluation_service(self) -> EvaluationService:
        if self._evaluation_service is None:
            self._evaluation_service = EvaluationService()
            await self._evaluation_service.initialize(self._get_eval_config())
            await self._register_default_evaluators()
        return self._evaluation_service

    @property
    async def data_service(self) -> DataService:
        if self._data_service is None:
            self._data_service = DataService()
            await self._data_service.initialize(self._get_data_config())
        return self._data_service

    def _get_model_config(self) -> dict:
        return {
            "max_concurrent_models": 5,
            "enable_performance_monitoring": True,
            "apple_silicon_optimization": True
        }

    def _get_eval_config(self) -> dict:
        return {
            "max_concurrent_evaluations": 10,
            "evaluation_timeout_seconds": 120.0
        }

    def _get_data_config(self) -> dict:
        return {
            "cache_enabled": True,
            "max_cache_size_mb": 1024
        }

    async def _register_default_evaluators(self):
        """Register commonly used evaluators."""
        eval_service = await self.evaluation_service

        from benchmark.evaluation.metrics import (
            AccuracyEvaluator, PrecisionRecallEvaluator, PerformanceEvaluator
        )

        await eval_service.register_evaluator(MetricType.ACCURACY, AccuracyEvaluator())
        await eval_service.register_evaluator(MetricType.PRECISION, PrecisionRecallEvaluator())
        await eval_service.register_evaluator(MetricType.PERFORMANCE, PerformanceEvaluator())

    async def shutdown(self):
        """Clean shutdown of all services."""
        if self._model_service:
            await self._model_service.shutdown()
        if self._evaluation_service:
            await self._evaluation_service.shutdown()
        if self._data_service:
            await self._data_service.shutdown()

# Usage example
async def main():
    container = BenchmarkContainer()

    try:
        # Use services through the container
        model_service = await container.model_service
        eval_service = await container.evaluation_service

        # Your application logic here
        await run_benchmarks(model_service, eval_service)

    finally:
        await container.shutdown()
```

### Service Factory Pattern

For more complex integration scenarios:

```python
from abc import ABC, abstractmethod
from enum import Enum

class ServiceMode(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ServiceFactory(ABC):
    """Abstract factory for creating benchmark services."""

    @abstractmethod
    async def create_model_service(self) -> ModelService:
        pass

    @abstractmethod
    async def create_evaluation_service(self) -> EvaluationService:
        pass

    @abstractmethod
    async def create_data_service(self) -> DataService:
        pass

class ProductionServiceFactory(ServiceFactory):
    """Production-optimized service factory."""

    async def create_model_service(self) -> ModelService:
        service = ModelService()
        await service.initialize({
            "max_concurrent_models": 16,
            "memory_threshold_mb": 16384,
            "enable_performance_monitoring": True,
            "apple_silicon_optimization": True,
            "health_check_interval": 30
        })
        return service

    async def create_evaluation_service(self) -> EvaluationService:
        service = EvaluationService()
        await service.initialize({
            "max_concurrent_evaluations": 32,
            "evaluation_timeout_seconds": 300.0,
            "max_history_size": 10000
        })

        # Register production evaluators
        await self._register_production_evaluators(service)
        return service

    async def create_data_service(self) -> DataService:
        service = DataService()
        await service.initialize({
            "cache_enabled": True,
            "cache_backend": "redis",
            "max_cache_size_mb": 4096,
            "database_pool_size": 20
        })
        return service

    async def _register_production_evaluators(self, service: EvaluationService):
        """Register all evaluators needed for production."""
        evaluators = [
            (MetricType.ACCURACY, AccuracyEvaluator()),
            (MetricType.PRECISION, PrecisionRecallEvaluator()),
            (MetricType.PERFORMANCE, PerformanceEvaluator()),
            (MetricType.ROC_AUC, ROCEvaluator()),
            (MetricType.CONFUSION_MATRIX, ConfusionMatrixEvaluator())
        ]

        for metric_type, evaluator in evaluators:
            await service.register_evaluator(metric_type, evaluator)

class DevelopmentServiceFactory(ServiceFactory):
    """Development-optimized service factory."""

    async def create_model_service(self) -> ModelService:
        service = ModelService()
        await service.initialize({
            "max_concurrent_models": 2,
            "memory_threshold_mb": 4096,
            "enable_performance_monitoring": False,
            "apple_silicon_optimization": False
        })
        return service

    async def create_evaluation_service(self) -> EvaluationService:
        service = EvaluationService()
        await service.initialize({
            "max_concurrent_evaluations": 4,
            "evaluation_timeout_seconds": 60.0,
            "max_history_size": 100
        })

        # Register basic evaluators for development
        await service.register_evaluator(MetricType.ACCURACY, AccuracyEvaluator())
        return service

    async def create_data_service(self) -> DataService:
        service = DataService()
        await service.initialize({
            "cache_enabled": False,
            "database_pool_size": 2
        })
        return service

# Factory selection based on environment
def get_service_factory(mode: ServiceMode) -> ServiceFactory:
    factories = {
        ServiceMode.DEVELOPMENT: DevelopmentServiceFactory,
        ServiceMode.STAGING: ProductionServiceFactory,  # Same as production
        ServiceMode.PRODUCTION: ProductionServiceFactory
    }
    return factories[mode]()

# Usage
async def create_services(mode: ServiceMode):
    factory = get_service_factory(mode)

    model_service = await factory.create_model_service()
    eval_service = await factory.create_evaluation_service()
    data_service = await factory.create_data_service()

    return model_service, eval_service, data_service
```

## Configuration Management

### Hierarchical Configuration System

The benchmark system uses a hierarchical configuration system:

```yaml
# configs/production.yaml
benchmark:
  name: "Production Cybersecurity Benchmark"
  version: "2.0"

services:
  model_service:
    max_concurrent_models: 16
    memory_threshold_mb: 16384
    plugins:
      - mlx
      - ollama
      - openai
      - anthropic

    plugin_configs:
      mlx:
        model_cache_dir: "/models/mlx"
        quantization_enabled: true

      openai:
        api_key: "${OPENAI_API_KEY}"
        organization: "${OPENAI_ORG}"
        rate_limit_rpm: 3000

      anthropic:
        api_key: "${ANTHROPIC_API_KEY}"
        rate_limit_rpm: 1000

  evaluation_service:
    max_concurrent_evaluations: 32
    timeout_seconds: 300
    metrics:
      - accuracy
      - precision
      - recall
      - f1_score
      - roc_auc

  data_service:
    cache_enabled: true
    cache_backend: "redis"
    cache_ttl_seconds: 3600
    database:
      url: "${DATABASE_URL}"
      pool_size: 20

models:
  cybersecurity_models:
    - id: "SecurityBERT"
      type: "mlx"
      path: "/models/security-bert-mlx"
      config:
        max_tokens: 2048
        temperature: 0.0

    - id: "CyberLLaMA"
      type: "ollama"
      model: "cybersecurity-llama:7b"
      config:
        temperature: 0.1
        system_prompt: "You are a cybersecurity expert..."

datasets:
  benchmark_datasets:
    - id: "network_intrusion"
      type: "csv"
      path: "data/network_intrusion.csv"
      preprocessing:
        - normalize_ip_addresses
        - extract_features

    - id: "malware_samples"
      type: "json"
      path: "data/malware_samples.json"
      preprocessing:
        - hash_normalization
        - feature_extraction

experiments:
  default_metrics: [accuracy, precision, recall, f1_score]
  output_format: "json"
  results_dir: "results/"

monitoring:
  enabled: true
  metrics_port: 9090
  log_level: "INFO"

  alerts:
    memory_threshold: 80  # Percentage
    error_rate_threshold: 5  # Percentage
    latency_threshold_ms: 1000
```

### Configuration Loading and Validation

```python
from benchmark.core.config_loader import ConfigLoader
from benchmark.core.config_validators import validate_config
from pathlib import Path

class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.loader = ConfigLoader()
        self._config = None

    async def load_config(self) -> dict:
        """Load and validate configuration."""
        if self._config is None:
            # Load base configuration
            self._config = await self.loader.load_config(self.config_path)

            # Validate configuration
            validation_result = validate_config(self._config)
            if not validation_result.is_valid:
                raise ValueError(f"Configuration validation failed: {validation_result.errors}")

            # Resolve environment variables
            self._config = self._resolve_environment_variables(self._config)

        return self._config

    def _resolve_environment_variables(self, config: dict) -> dict:
        """Resolve environment variables in configuration."""
        import os
        import re

        def resolve_env_vars(obj):
            if isinstance(obj, dict):
                return {k: resolve_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # Replace ${VAR_NAME} with environment variable value
                pattern = r'\$\{([^}]+)\}'

                def replacer(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, match.group(0))

                return re.sub(pattern, replacer, obj)
            else:
                return obj

        return resolve_env_vars(config)

    def get_service_config(self, service_name: str) -> dict:
        """Get configuration for a specific service."""
        config = self._config or {}
        return config.get("services", {}).get(service_name, {})

    def get_model_config(self, model_id: str) -> dict:
        """Get configuration for a specific model."""
        config = self._config or {}
        models = config.get("models", {}).get("cybersecurity_models", [])

        for model in models:
            if model.get("id") == model_id:
                return model.get("config", {})

        return {}

    def get_dataset_config(self, dataset_id: str) -> dict:
        """Get configuration for a specific dataset."""
        config = self._config or {}
        datasets = config.get("datasets", {}).get("benchmark_datasets", [])

        for dataset in datasets:
            if dataset.get("id") == dataset_id:
                return dataset

        return {}

# Usage example
async def setup_with_config():
    config_manager = ConfigManager(Path("configs/production.yaml"))
    config = await config_manager.load_config()

    # Create services with configuration
    model_service = ModelService()
    model_config = config_manager.get_service_config("model_service")
    await model_service.initialize(model_config)

    eval_service = EvaluationService()
    eval_config = config_manager.get_service_config("evaluation_service")
    await eval_service.initialize(eval_config)

    return model_service, eval_service, config_manager
```

## Data Pipeline Integration

### Data Loading and Preprocessing

```python
from benchmark.services.data_service import DataService
from benchmark.data.models import DatasetInfo, ProcessedDataset
from benchmark.data.preprocessors import NetworkLogPreprocessor, EmailContentPreprocessor

class DataPipeline:
    """Integrated data processing pipeline."""

    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.preprocessors = {
            "network_logs": NetworkLogPreprocessor(),
            "email_content": EmailContentPreprocessor()
        }

    async def load_dataset(self, dataset_id: str, config: dict) -> ProcessedDataset:
        """Load and preprocess a dataset."""

        # Load raw data
        dataset_info = DatasetInfo(
            dataset_id=dataset_id,
            source_path=config["path"],
            format=config["type"]
        )

        raw_data = await self.data_service.load_dataset(dataset_info)

        # Apply preprocessing
        preprocessor_names = config.get("preprocessing", [])
        processed_data = raw_data

        for preprocessor_name in preprocessor_names:
            if preprocessor_name in self.preprocessors:
                preprocessor = self.preprocessors[preprocessor_name]
                processed_data = await preprocessor.process(processed_data)

        return ProcessedDataset(
            dataset_id=dataset_id,
            data=processed_data,
            metadata={
                "preprocessing_steps": preprocessor_names,
                "original_size": len(raw_data),
                "processed_size": len(processed_data)
            }
        )

    async def create_train_test_split(
        self,
        dataset: ProcessedDataset,
        test_size: float = 0.2
    ) -> tuple[ProcessedDataset, ProcessedDataset]:
        """Create train/test split."""

        split_result = await self.data_service.create_split(
            dataset.data,
            test_size=test_size,
            stratify=True
        )

        train_dataset = ProcessedDataset(
            dataset_id=f"{dataset.dataset_id}_train",
            data=split_result.train_data,
            metadata={**dataset.metadata, "split": "train"}
        )

        test_dataset = ProcessedDataset(
            dataset_id=f"{dataset.dataset_id}_test",
            data=split_result.test_data,
            metadata={**dataset.metadata, "split": "test"}
        )

        return train_dataset, test_dataset

    async def generate_synthetic_data(
        self,
        base_dataset: ProcessedDataset,
        augmentation_factor: int = 2
    ) -> ProcessedDataset:
        """Generate synthetic data for training augmentation."""

        synthetic_data = await self.data_service.generate_synthetic_samples(
            base_dataset.data,
            count=len(base_dataset.data) * augmentation_factor
        )

        return ProcessedDataset(
            dataset_id=f"{base_dataset.dataset_id}_synthetic",
            data=synthetic_data,
            metadata={
                **base_dataset.metadata,
                "synthetic": True,
                "augmentation_factor": augmentation_factor
            }
        )

# Integration example
async def integrated_data_pipeline():
    data_service = DataService()
    await data_service.initialize()

    pipeline = DataPipeline(data_service)

    # Load and preprocess multiple datasets
    datasets = {}

    dataset_configs = [
        {
            "id": "network_intrusion",
            "path": "data/network_logs.csv",
            "type": "csv",
            "preprocessing": ["normalize_ip_addresses", "extract_features"]
        },
        {
            "id": "malware_samples",
            "path": "data/malware.json",
            "type": "json",
            "preprocessing": ["hash_normalization", "feature_extraction"]
        }
    ]

    for config in dataset_configs:
        dataset = await pipeline.load_dataset(config["id"], config)
        train_data, test_data = await pipeline.create_train_test_split(dataset)

        datasets[config["id"]] = {
            "train": train_data,
            "test": test_data,
            "full": dataset
        }

    return datasets
```

### Real-time Data Streaming

```python
import asyncio
from typing import AsyncGenerator
from benchmark.data.streaming import StreamingDataProvider

class RealTimeIntegration:
    """Integration for real-time cybersecurity data processing."""

    def __init__(self, model_service: ModelService, eval_service: EvaluationService):
        self.model_service = model_service
        self.eval_service = eval_service
        self.stream_providers = {}

    def register_stream(self, stream_id: str, provider: StreamingDataProvider):
        """Register a data stream provider."""
        self.stream_providers[stream_id] = provider

    async def process_stream(
        self,
        stream_id: str,
        model_id: str,
        batch_size: int = 10
    ) -> AsyncGenerator[dict, None]:
        """Process data from a stream in real-time."""

        if stream_id not in self.stream_providers:
            raise ValueError(f"Stream {stream_id} not registered")

        provider = self.stream_providers[stream_id]
        batch = []

        async for data_point in provider.stream():
            batch.append(data_point)

            if len(batch) >= batch_size:
                # Process batch
                results = await self._process_batch(batch, model_id)

                yield {
                    "stream_id": stream_id,
                    "batch_size": len(batch),
                    "predictions": results,
                    "timestamp": datetime.now().isoformat()
                }

                batch = []

        # Process remaining items
        if batch:
            results = await self._process_batch(batch, model_id)
            yield {
                "stream_id": stream_id,
                "batch_size": len(batch),
                "predictions": results,
                "timestamp": datetime.now().isoformat()
            }

    async def _process_batch(self, batch: list, model_id: str) -> list:
        """Process a batch of data points."""
        inputs = [item["text"] for item in batch]

        response = await self.model_service.predict(model_id, inputs)
        if response.success:
            return response.data
        else:
            return [{"error": response.error} for _ in inputs]

# Stream processing example
async def real_time_processing_example():
    # Setup services
    model_service = await container.model_service
    eval_service = await container.evaluation_service

    # Load real-time processing model
    await model_service.load_model("SecurityBERT", strategy=LoadingStrategy.EAGER)

    # Create real-time integration
    rt_integration = RealTimeIntegration(model_service, eval_service)

    # Register network log stream
    from benchmark.data.streaming import NetworkLogStream
    network_stream = NetworkLogStream(source="syslog://localhost:514")
    rt_integration.register_stream("network_logs", network_stream)

    # Process stream
    async for batch_result in rt_integration.process_stream(
        "network_logs",
        "SecurityBERT",
        batch_size=20
    ):
        # Handle real-time results
        print(f"Processed {batch_result['batch_size']} network events")

        # Check for threats
        threats = [
            pred for pred in batch_result["predictions"]
            if pred.prediction == "ATTACK" and pred.confidence > 0.8
        ]

        if threats:
            await handle_security_alerts(threats)
```

## Model Management

### Model Registry and Versioning

```python
from benchmark.models.registry import ModelRegistry
from benchmark.models.model_cache import ModelCache

class ModelManager:
    """Centralized model management."""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.registry = ModelRegistry()
        self.cache = ModelCache()

    async def register_model(
        self,
        model_id: str,
        model_config: dict,
        metadata: dict | None = None
    ) -> bool:
        """Register a new model in the registry."""

        # Validate model before registration
        validation_result = await self.model_service.validate_model(model_id)
        if not validation_result.success:
            return False

        # Register in model registry
        registry_entry = {
            "model_id": model_id,
            "config": model_config,
            "metadata": metadata or {},
            "validation_status": "passed",
            "registered_at": datetime.now().isoformat()
        }

        await self.registry.register_model(model_id, registry_entry)
        return True

    async def deploy_model(self, model_id: str, deployment_config: dict) -> bool:
        """Deploy a registered model."""

        # Check if model is registered
        if not await self.registry.is_registered(model_id):
            raise ValueError(f"Model {model_id} not registered")

        # Get model configuration
        model_config = await self.registry.get_config(model_id)

        # Load model with deployment configuration
        combined_config = {**model_config, **deployment_config}
        response = await self.model_service.load_model(model_id, combined_config)

        if response.success:
            # Cache model for quick access
            await self.cache.cache_model(model_id, response.data)
            return True

        return False

    async def update_model(
        self,
        model_id: str,
        new_config: dict,
        version: str
    ) -> bool:
        """Update a model configuration."""

        # Create versioned entry
        versioned_id = f"{model_id}:v{version}"

        # Register new version
        success = await self.register_model(versioned_id, new_config, {
            "base_model": model_id,
            "version": version,
            "update_type": "configuration"
        })

        if success:
            # Update default model reference
            await self.registry.set_default_version(model_id, version)

        return success

    async def rollback_model(self, model_id: str, target_version: str) -> bool:
        """Rollback a model to a previous version."""

        versioned_id = f"{model_id}:v{target_version}"

        # Check if target version exists
        if not await self.registry.is_registered(versioned_id):
            return False

        # Unload current version
        await self.model_service.unload_model(model_id)

        # Load target version
        target_config = await self.registry.get_config(versioned_id)
        response = await self.model_service.load_model(model_id, target_config)

        if response.success:
            await self.registry.set_default_version(model_id, target_version)
            return True

        return False

    async def list_available_models(self) -> dict:
        """List all available models with their status."""

        models = await self.registry.list_models()
        loaded_models = await self.model_service.get_loaded_models()

        result = {}
        for model_id, model_info in models.items():
            is_loaded = any(m.model_id == model_id for m in loaded_models.data)

            result[model_id] = {
                "info": model_info,
                "loaded": is_loaded,
                "versions": await self.registry.list_versions(model_id)
            }

        return result

# Usage example
async def model_management_example():
    model_service = await container.model_service
    model_manager = ModelManager(model_service)

    # Register a new cybersecurity model
    await model_manager.register_model(
        "ThreatDetector_v1",
        {
            "type": "mlx",
            "path": "/models/threat-detector-v1",
            "max_tokens": 1024,
            "temperature": 0.0
        },
        {
            "domain": "cybersecurity",
            "specialization": "threat_detection",
            "training_data": "cybersec_dataset_v2.1"
        }
    )

    # Deploy the model
    await model_manager.deploy_model("ThreatDetector_v1", {
        "memory_limit_mb": 4096,
        "priority": "high"
    })

    # Update model with new configuration
    await model_manager.update_model(
        "ThreatDetector_v1",
        {
            "type": "mlx",
            "path": "/models/threat-detector-v1.1",
            "max_tokens": 2048,  # Increased context
            "temperature": 0.1   # Slightly more creative
        },
        "1.1"
    )

    # List all available models
    models = await model_manager.list_available_models()
    for model_id, model_data in models.items():
        print(f"{model_id}: {'loaded' if model_data['loaded'] else 'not loaded'}")
        print(f"  Versions: {model_data['versions']}")
```

### A/B Testing Framework

```python
class ModelABTestFramework:
    """Framework for A/B testing model performance."""

    def __init__(
        self,
        model_service: ModelService,
        eval_service: EvaluationService
    ):
        self.model_service = model_service
        self.eval_service = eval_service
        self.active_tests = {}

    async def start_ab_test(
        self,
        test_id: str,
        model_a: str,
        model_b: str,
        test_config: dict
    ) -> bool:
        """Start an A/B test between two models."""

        # Validate both models are loaded
        loaded_models = await self.model_service.get_loaded_models()
        loaded_ids = [m.model_id for m in loaded_models.data]

        if model_a not in loaded_ids or model_b not in loaded_ids:
            return False

        # Create test configuration
        self.active_tests[test_id] = {
            "model_a": model_a,
            "model_b": model_b,
            "config": test_config,
            "results": {"a": [], "b": []},
            "start_time": datetime.now(),
            "status": "active"
        }

        return True

    async def process_test_sample(
        self,
        test_id: str,
        sample: dict,
        model_selection: str = "random"
    ) -> dict:
        """Process a sample through the A/B test."""

        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_config = self.active_tests[test_id]

        # Select model based on strategy
        if model_selection == "random":
            selected_model = random.choice([test_config["model_a"], test_config["model_b"]])
        elif model_selection == "a":
            selected_model = test_config["model_a"]
        else:
            selected_model = test_config["model_b"]

        # Process with selected model
        response = await self.model_service.predict(
            selected_model,
            [sample["input"]]
        )

        if response.success:
            prediction = response.data[0]

            # Store result
            variant = "a" if selected_model == test_config["model_a"] else "b"
            test_config["results"][variant].append({
                "input": sample["input"],
                "prediction": prediction,
                "ground_truth": sample.get("ground_truth"),
                "timestamp": datetime.now()
            })

            return {
                "test_id": test_id,
                "variant": variant,
                "model": selected_model,
                "prediction": prediction
            }

        return {"error": "Prediction failed"}

    async def evaluate_test_results(self, test_id: str) -> dict:
        """Evaluate A/B test results."""

        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_data = self.active_tests[test_id]
        results = {"variant_a": {}, "variant_b": {}}

        # Evaluate each variant
        for variant in ["a", "b"]:
            variant_results = test_data["results"][variant]

            if not variant_results:
                continue

            # Create evaluation request
            predictions = []
            ground_truth = []

            for result in variant_results:
                if result.get("ground_truth"):
                    predictions.append({
                        "predicted_class": result["prediction"].prediction,
                        "confidence": result["prediction"].confidence
                    })
                    ground_truth.append({
                        "true_class": result["ground_truth"]
                    })

            if predictions and ground_truth:
                eval_request = EvaluationRequest(
                    experiment_id=f"{test_id}_variant_{variant}",
                    model_id=test_data[f"model_{variant}"],
                    dataset_id="ab_test_data",
                    predictions=predictions,
                    ground_truth=ground_truth,
                    metrics=[MetricType.ACCURACY, MetricType.PRECISION],
                    metadata={"ab_test": test_id, "variant": variant}
                )

                eval_result = await self.eval_service.evaluate_predictions(eval_request)
                results[f"variant_{variant}"] = {
                    "model": test_data[f"model_{variant}"],
                    "sample_count": len(predictions),
                    "accuracy": eval_result.get_metric_value("accuracy"),
                    "precision": eval_result.get_metric_value("precision"),
                    "avg_confidence": sum(p["confidence"] for p in predictions) / len(predictions)
                }

        # Statistical significance testing
        significance_test = self._calculate_statistical_significance(results)

        return {
            "test_id": test_id,
            "results": results,
            "statistical_significance": significance_test,
            "recommendation": self._generate_recommendation(results, significance_test)
        }

    def _calculate_statistical_significance(self, results: dict) -> dict:
        """Calculate statistical significance of A/B test results."""
        # Simplified significance testing
        # In production, use proper statistical tests (t-test, chi-square, etc.)

        variant_a = results.get("variant_a", {})
        variant_b = results.get("variant_b", {})

        if not variant_a or not variant_b:
            return {"significant": False, "reason": "Insufficient data"}

        accuracy_diff = abs(variant_a["accuracy"] - variant_b["accuracy"])
        sample_sizes = [variant_a["sample_count"], variant_b["sample_count"]]

        # Simple heuristic for significance
        min_sample_size = 100
        min_effect_size = 0.05

        significant = (
            all(size >= min_sample_size for size in sample_sizes) and
            accuracy_diff >= min_effect_size
        )

        return {
            "significant": significant,
            "effect_size": accuracy_diff,
            "sample_sizes": sample_sizes,
            "confidence_level": 0.95 if significant else None
        }

    def _generate_recommendation(self, results: dict, significance: dict) -> str:
        """Generate recommendation based on A/B test results."""

        if not significance["significant"]:
            return "No statistically significant difference detected. Continue testing or investigate other factors."

        variant_a = results.get("variant_a", {})
        variant_b = results.get("variant_b", {})

        if variant_a["accuracy"] > variant_b["accuracy"]:
            return f"Model A ({variant_a['model']}) shows superior performance. Recommend deployment."
        else:
            return f"Model B ({variant_b['model']}) shows superior performance. Recommend deployment."

# A/B testing example
async def ab_testing_example():
    model_service = await container.model_service
    eval_service = await container.evaluation_service

    # Load competing models
    await model_service.load_model("SecurityBERT_v1")
    await model_service.load_model("SecurityBERT_v2")

    # Start A/B test
    ab_framework = ModelABTestFramework(model_service, eval_service)

    await ab_framework.start_ab_test(
        "security_bert_comparison",
        "SecurityBERT_v1",
        "SecurityBERT_v2",
        {
            "test_duration_days": 7,
            "min_samples_per_variant": 1000,
            "metrics": ["accuracy", "precision", "latency"]
        }
    )

    # Simulate processing test samples
    test_samples = [
        {"input": "Network connection analysis...", "ground_truth": "benign"},
        {"input": "Suspicious email detected...", "ground_truth": "attack"},
        # ... more samples
    ]

    for sample in test_samples:
        result = await ab_framework.process_test_sample(
            "security_bert_comparison",
            sample
        )
        print(f"Processed with variant {result['variant']}")

    # Evaluate results
    evaluation = await ab_framework.evaluate_test_results("security_bert_comparison")
    print(f"Test recommendation: {evaluation['recommendation']}")
```

## Evaluation Workflows

### Custom Evaluation Pipeline

```python
from benchmark.evaluation.evaluation_workflow import EvaluationWorkflow
from benchmark.evaluation.result_models import EvaluationReport

class CybersecurityEvaluationPipeline:
    """Specialized evaluation pipeline for cybersecurity models."""

    def __init__(
        self,
        model_service: ModelService,
        eval_service: EvaluationService,
        data_service: DataService
    ):
        self.model_service = model_service
        self.eval_service = eval_service
        self.data_service = data_service

        self.workflow = EvaluationWorkflow()
        self.threat_categories = [
            "malware", "phishing", "network_intrusion",
            "data_breach", "insider_threat"
        ]

    async def run_comprehensive_evaluation(
        self,
        model_ids: list[str],
        dataset_configs: list[dict],
        evaluation_config: dict
    ) -> EvaluationReport:
        """Run comprehensive cybersecurity evaluation."""

        all_results = []
        evaluation_metadata = {
            "evaluation_type": "comprehensive_cybersecurity",
            "start_time": datetime.now().isoformat(),
            "models_evaluated": model_ids,
            "datasets_used": [d["id"] for d in dataset_configs]
        }

        # Load and prepare datasets
        datasets = {}
        for dataset_config in dataset_configs:
            dataset = await self.data_service.load_dataset(
                DatasetInfo(
                    dataset_id=dataset_config["id"],
                    source_path=dataset_config["path"],
                    format=dataset_config["format"]
                )
            )
            datasets[dataset_config["id"]] = dataset

        # Evaluate each model on each dataset
        for model_id in model_ids:
            for dataset_id, dataset in datasets.items():

                # Generate predictions
                inputs = [item["text"] for item in dataset]
                prediction_response = await self.model_service.predict(model_id, inputs)

                if not prediction_response.success:
                    continue

                predictions = prediction_response.data

                # Prepare evaluation request
                eval_request = EvaluationRequest(
                    experiment_id=f"{model_id}_{dataset_id}_comprehensive",
                    model_id=model_id,
                    dataset_id=dataset_id,
                    predictions=[{
                        "predicted_class": pred.prediction,
                        "confidence": pred.confidence,
                        "attack_type": pred.attack_type
                    } for pred in predictions],
                    ground_truth=[{
                        "true_class": item["label"],
                        "threat_category": item.get("category")
                    } for item in dataset],
                    metrics=[
                        MetricType.ACCURACY,
                        MetricType.PRECISION,
                        MetricType.RECALL,
                        MetricType.F1_SCORE,
                        MetricType.ROC_AUC
                    ],
                    metadata={
                        **evaluation_metadata,
                        "model_id": model_id,
                        "dataset_id": dataset_id
                    }
                )

                # Perform evaluation
                eval_result = await self.eval_service.evaluate_predictions(eval_request)
                all_results.append(eval_result)

        # Generate comprehensive report
        report = await self._generate_comprehensive_report(
            all_results,
            evaluation_metadata,
            evaluation_config
        )

        return report

    async def run_adversarial_evaluation(
        self,
        model_id: str,
        base_dataset: dict,
        adversarial_config: dict
    ) -> dict:
        """Run adversarial robustness evaluation."""

        # Generate adversarial examples
        adversarial_dataset = await self._generate_adversarial_examples(
            base_dataset,
            adversarial_config
        )

        # Test model on adversarial examples
        adversarial_inputs = [item["text"] for item in adversarial_dataset]
        adv_predictions = await self.model_service.predict(model_id, adversarial_inputs)

        # Test model on clean examples for comparison
        clean_inputs = [item["original_text"] for item in adversarial_dataset]
        clean_predictions = await self.model_service.predict(model_id, clean_inputs)

        # Analyze robustness
        robustness_analysis = await self._analyze_adversarial_robustness(
            clean_predictions.data,
            adv_predictions.data,
            adversarial_dataset
        )

        return robustness_analysis

    async def run_bias_evaluation(
        self,
        model_id: str,
        demographic_datasets: dict
    ) -> dict:
        """Evaluate model for demographic bias."""

        bias_results = {}

        for demographic, dataset in demographic_datasets.items():
            inputs = [item["text"] for item in dataset]
            predictions = await self.model_service.predict(model_id, inputs)

            if predictions.success:
                # Analyze prediction patterns by demographic
                bias_metrics = await self._calculate_bias_metrics(
                    predictions.data,
                    dataset,
                    demographic
                )
                bias_results[demographic] = bias_metrics

        # Generate bias report
        bias_report = await self._generate_bias_report(bias_results)
        return bias_report

    async def _generate_comprehensive_report(
        self,
        results: list,
        metadata: dict,
        config: dict
    ) -> EvaluationReport:
        """Generate comprehensive evaluation report."""

        # Aggregate results by model
        model_results = {}
        for result in results:
            model_id = result.model_id
            if model_id not in model_results:
                model_results[model_id] = []
            model_results[model_id].append(result)

        # Calculate aggregate metrics
        aggregate_metrics = {}
        for model_id, model_evals in model_results.items():
            metrics = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1_score": []
            }

            for eval_result in model_evals:
                for metric_name in metrics.keys():
                    value = eval_result.get_metric_value(metric_name)
                    if value is not None:
                        metrics[metric_name].append(value)

            # Calculate averages
            aggregate_metrics[model_id] = {
                metric_name: sum(values) / len(values) if values else 0.0
                for metric_name, values in metrics.items()
            }

            # Add additional statistics
            aggregate_metrics[model_id].update({
                "evaluations_count": len(model_evals),
                "avg_execution_time": sum(r.execution_time_seconds for r in model_evals) / len(model_evals)
            })

        # Generate ranking
        ranking = sorted(
            aggregate_metrics.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )

        return EvaluationReport(
            experiment_id=f"comprehensive_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            metadata=metadata,
            results=results,
            aggregate_metrics=aggregate_metrics,
            model_ranking=ranking,
            generated_at=datetime.now().isoformat()
        )

    async def _generate_adversarial_examples(self, dataset: dict, config: dict) -> list:
        """Generate adversarial examples for robustness testing."""
        # Implementation would depend on specific adversarial attack methods
        # This is a simplified version

        adversarial_examples = []

        for item in dataset:
            original_text = item["text"]

            # Apply various adversarial transformations
            transformations = [
                self._typo_injection,
                self._synonym_replacement,
                self._word_reordering,
                self._character_insertion
            ]

            for transform in transformations:
                adversarial_text = await transform(original_text, config)
                adversarial_examples.append({
                    "original_text": original_text,
                    "text": adversarial_text,
                    "transformation": transform.__name__,
                    "label": item["label"]
                })

        return adversarial_examples

    async def _typo_injection(self, text: str, config: dict) -> str:
        """Inject typos into text."""
        # Simple typo injection implementation
        words = text.split()
        typo_rate = config.get("typo_rate", 0.1)

        for i, word in enumerate(words):
            if random.random() < typo_rate and len(word) > 3:
                # Random character replacement
                pos = random.randint(1, len(word) - 2)
                word_list = list(word)
                word_list[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
                words[i] = ''.join(word_list)

        return ' '.join(words)

    async def _synonym_replacement(self, text: str, config: dict) -> str:
        """Replace words with synonyms."""
        # Simplified synonym replacement
        # In practice, would use wordnet or similar

        simple_synonyms = {
            "attack": "assault",
            "malicious": "harmful",
            "suspicious": "questionable",
            "threat": "danger",
            "security": "safety"
        }

        words = text.split()
        replacement_rate = config.get("synonym_rate", 0.2)

        for i, word in enumerate(words):
            if random.random() < replacement_rate and word.lower() in simple_synonyms:
                words[i] = simple_synonyms[word.lower()]

        return ' '.join(words)

    async def _analyze_adversarial_robustness(
        self,
        clean_predictions: list,
        adv_predictions: list,
        adversarial_dataset: list
    ) -> dict:
        """Analyze model robustness to adversarial examples."""

        robustness_metrics = {
            "total_examples": len(clean_predictions),
            "successful_attacks": 0,
            "robustness_score": 0.0,
            "attack_success_by_type": {}
        }

        for clean_pred, adv_pred, adv_example in zip(
            clean_predictions, adv_predictions, adversarial_dataset
        ):
            # Check if adversarial example caused misclassification
            clean_correct = clean_pred.prediction == adv_example["label"]
            adv_correct = adv_pred.prediction == adv_example["label"]

            if clean_correct and not adv_correct:
                robustness_metrics["successful_attacks"] += 1

                attack_type = adv_example["transformation"]
                if attack_type not in robustness_metrics["attack_success_by_type"]:
                    robustness_metrics["attack_success_by_type"][attack_type] = 0
                robustness_metrics["attack_success_by_type"][attack_type] += 1

        # Calculate robustness score (higher is better)
        robustness_metrics["robustness_score"] = 1.0 - (
            robustness_metrics["successful_attacks"] / robustness_metrics["total_examples"]
        )

        return robustness_metrics

# Usage example
async def comprehensive_evaluation_example():
    # Initialize services
    container = BenchmarkContainer()
    model_service = await container.model_service
    eval_service = await container.evaluation_service
    data_service = await container.data_service

    # Create evaluation pipeline
    pipeline = CybersecurityEvaluationPipeline(
        model_service, eval_service, data_service
    )

    # Define models to evaluate
    models = ["SecurityBERT_v1", "SecurityBERT_v2", "CyberLLaMA"]

    # Define datasets
    datasets = [
        {
            "id": "network_intrusion",
            "path": "data/network_intrusion.json",
            "format": "json"
        },
        {
            "id": "malware_detection",
            "path": "data/malware_samples.json",
            "format": "json"
        },
        {
            "id": "phishing_emails",
            "path": "data/phishing_emails.json",
            "format": "json"
        }
    ]

    # Run comprehensive evaluation
    evaluation_report = await pipeline.run_comprehensive_evaluation(
        models,
        datasets,
        {
            "include_confidence_analysis": True,
            "generate_confusion_matrices": True,
            "export_format": "json"
        }
    )

    print("📊 Comprehensive Evaluation Results")
    print("="*50)

    # Display ranking
    print("\n🏆 Model Rankings:")
    for rank, (model_id, metrics) in enumerate(evaluation_report.model_ranking, 1):
        print(f"{rank}. {model_id}")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   Avg. Execution Time: {metrics['avg_execution_time']:.2f}s")
        print()

    # Run adversarial evaluation on top model
    top_model = evaluation_report.model_ranking[0][0]
    print(f"\n🛡️  Adversarial Robustness Test: {top_model}")

    base_dataset = await data_service.load_dataset(
        DatasetInfo(dataset_id="network_intrusion", source_path="data/network_intrusion.json", format="json")
    )

    robustness_results = await pipeline.run_adversarial_evaluation(
        top_model,
        base_dataset,
        {"typo_rate": 0.1, "synonym_rate": 0.2}
    )

    print(f"Robustness Score: {robustness_results['robustness_score']:.3f}")
    print(f"Successful Attacks: {robustness_results['successful_attacks']}/{robustness_results['total_examples']}")

    await container.shutdown()
```

## Production Deployment

### Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV BENCHMARK_CONFIG_PATH=/app/configs/production.yaml

# Expose ports for API and monitoring
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "benchmark.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  benchmark-api:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - PYTHONPATH=/app/src
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://user:password@postgres:5432/benchmark
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./results:/app/results
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=benchmark
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - benchmark-api
    restart: unless-stopped

volumes:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: benchmark-api
  labels:
    app: benchmark-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: benchmark-api
  template:
    metadata:
      labels:
        app: benchmark-api
    spec:
      containers:
      - name: benchmark-api
        image: benchmark:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: anthropic-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: data-storage
          mountPath: /app/data
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: benchmark-api-service
spec:
  selector:
    app: benchmark-api
  ports:
  - name: api
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  openai-key: <base64-encoded-key>
  anthropic-key: <base64-encoded-key>

---
apiVersion: v1
kind: Secret
metadata:
  name: database-secrets
type: Opaque
data:
  url: <base64-encoded-database-url>
```

### Monitoring and Observability

```python
# monitoring.py
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any

# Prometheus metrics
REQUEST_COUNT = Counter('benchmark_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('benchmark_request_duration_seconds', 'Request duration')
ACTIVE_MODELS = Gauge('benchmark_active_models', 'Number of active models')
EVALUATION_DURATION = Histogram('benchmark_evaluation_duration_seconds', 'Evaluation duration')
ERROR_COUNT = Counter('benchmark_errors_total', 'Total errors', ['error_type'])

class MonitoringService:
    """Service for monitoring and observability."""

    def __init__(self, port: int = 9090):
        self.port = port
        self.logger = logging.getLogger("monitoring")

    async def start_metrics_server(self):
        """Start Prometheus metrics server."""
        start_http_server(self.port)
        self.logger.info(f"Metrics server started on port {self.port}")

    def record_request(self, method: str, endpoint: str, duration: float):
        """Record API request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_DURATION.observe(duration)

    def record_evaluation(self, duration: float):
        """Record evaluation metrics."""
        EVALUATION_DURATION.observe(duration)

    def update_active_models(self, count: int):
        """Update active models gauge."""
        ACTIVE_MODELS.set(count)

    def record_error(self, error_type: str):
        """Record error metrics."""
        ERROR_COUNT.labels(error_type=error_type).inc()

# Structured logging setup
def setup_logging():
    """Setup structured logging."""
    import json
    import sys

    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }

            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)

            return json.dumps(log_entry)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

# Health check endpoints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]

async def create_health_endpoints(
    app: FastAPI,
    model_service: ModelService,
    eval_service: EvaluationService
):
    """Add health check endpoints to FastAPI app."""

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Comprehensive health check."""

        services = {}
        overall_status = "healthy"

        # Check model service
        try:
            model_health = await model_service.health_check()
            services["model_service"] = {
                "status": model_health.status.value,
                "details": model_health.details
            }
            if model_health.status != ServiceStatus.HEALTHY:
                overall_status = "degraded"
        except Exception as e:
            services["model_service"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"

        # Check evaluation service
        try:
            eval_health = await eval_service.health_check()
            services["evaluation_service"] = {
                "status": eval_health.status.value,
                "details": eval_health.details
            }
            if eval_health.status != ServiceStatus.HEALTHY:
                overall_status = "degraded"
        except Exception as e:
            services["evaluation_service"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            services=services
        )

    @app.get("/ready")
    async def readiness_check():
        """Readiness check for Kubernetes."""

        # Check if services are ready to accept requests
        try:
            loaded_models = await model_service.get_loaded_models()
            if not loaded_models.success or not loaded_models.data:
                raise HTTPException(status_code=503, detail="No models loaded")

            available_metrics = await eval_service.get_available_metrics()
            if not available_metrics.success or available_metrics.data["total_evaluators"] == 0:
                raise HTTPException(status_code=503, detail="No evaluators registered")

            return {"status": "ready"}

        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Service not ready: {e}")

# Usage example
async def setup_production_monitoring():
    """Setup production monitoring."""

    # Setup structured logging
    setup_logging()

    # Start monitoring service
    monitoring = MonitoringService(port=9090)
    await monitoring.start_metrics_server()

    # Create FastAPI app with health endpoints
    from fastapi import FastAPI
    app = FastAPI(title="LLM Cybersecurity Benchmark API")

    # Initialize services (would be done in main application)
    container = BenchmarkContainer()
    model_service = await container.model_service
    eval_service = await container.evaluation_service

    # Add health endpoints
    await create_health_endpoints(app, model_service, eval_service)

    return app, monitoring
```

## Security Considerations

### API Security

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import secrets

class SecurityManager:
    """Centralized security management."""

    def __init__(self):
        self.secret_key = secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.security = HTTPBearer()

    def create_access_token(self, data: dict, expires_delta: int = 3600) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(seconds=expires_delta)
        to_encode.update({"exp": expire})

        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Verify JWT token."""
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            return username
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

    def rate_limit_check(self, client_id: str, max_requests: int = 100, window_minutes: int = 60):
        """Check rate limits (simplified implementation)."""
        # In production, use Redis or similar for distributed rate limiting
        pass

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        import html
        import re

        # HTML escape
        text = html.escape(text)

        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror='
        ]

        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text

# Secure API implementation
def create_secure_api(
    model_service: ModelService,
    eval_service: EvaluationService
) -> FastAPI:
    """Create secure API with authentication and rate limiting."""

    app = FastAPI(title="Secure LLM Cybersecurity Benchmark API")
    security_manager = SecurityManager()

    @app.post("/auth/token")
    async def login(username: str, password: str):
        """Authenticate and get access token."""
        # In production, verify against user database
        if username == "admin" and password == "secure_password":
            access_token = security_manager.create_access_token({"sub": username})
            return {"access_token": access_token, "token_type": "bearer"}

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    @app.post("/predict")
    async def secure_predict(
        model_id: str,
        inputs: list[str],
        current_user: str = Depends(security_manager.verify_token)
    ):
        """Secure prediction endpoint."""

        # Rate limiting
        security_manager.rate_limit_check(current_user)

        # Input sanitization
        sanitized_inputs = [security_manager.sanitize_input(inp) for inp in inputs]

        # Input validation
        if not model_id or not sanitized_inputs:
            raise HTTPException(status_code=400, detail="Invalid input")

        if len(sanitized_inputs) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large")

        # Perform prediction
        try:
            response = await model_service.predict(model_id, sanitized_inputs)
            if response.success:
                return {"predictions": response.data, "user": current_user}
            else:
                raise HTTPException(status_code=500, detail=response.error)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return app
```

### Data Privacy and Compliance

```python
class DataPrivacyManager:
    """Manage data privacy and compliance requirements."""

    def __init__(self):
        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),           # Social Security Numbers
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'CREDIT_CARD'),  # Credit cards
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),  # Email addresses
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDRESS'),      # IP addresses
        ]

    def detect_pii(self, text: str) -> list[dict]:
        """Detect personally identifiable information in text."""
        import re

        detected_pii = []
        for pattern, pii_type in self.pii_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                detected_pii.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        return detected_pii

    def anonymize_text(self, text: str) -> tuple[str, list[dict]]:
        """Anonymize text by removing/masking PII."""
        import re

        anonymized_text = text
        anonymization_log = []

        for pattern, pii_type in self.pii_patterns:
            matches = list(re.finditer(pattern, anonymized_text))
            for match in reversed(matches):  # Reverse to maintain positions
                original_value = match.group()
                masked_value = self._mask_value(original_value, pii_type)

                anonymized_text = (
                    anonymized_text[:match.start()] +
                    masked_value +
                    anonymized_text[match.end():]
                )

                anonymization_log.append({
                    "type": pii_type,
                    "original_length": len(original_value),
                    "masked_value": masked_value,
                    "position": match.start()
                })

        return anonymized_text, anonymization_log

    def _mask_value(self, value: str, pii_type: str) -> str:
        """Mask PII value based on type."""
        if pii_type == "SSN":
            return "***-**-" + value[-4:]
        elif pii_type == "CREDIT_CARD":
            return "**** **** **** " + value.replace(" ", "").replace("-", "")[-4:]
        elif pii_type == "EMAIL":
            parts = value.split("@")
            return parts[0][:2] + "***@" + parts[1]
        elif pii_type == "IP_ADDRESS":
            parts = value.split(".")
            return f"{parts[0]}.{parts[1]}.***.***.***"
        else:
            return "***REDACTED***"

    def audit_log_access(self, user_id: str, action: str, data_type: str, details: dict):
        """Log data access for compliance auditing."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "data_type": data_type,
            "details": details,
            "ip_address": details.get("ip_address"),
            "user_agent": details.get("user_agent")
        }

        # In production, log to secure audit system
        logger.info(f"AUDIT: {audit_entry}")

# GDPR compliance wrapper
class GDPRCompliantService:
    """Wrapper service ensuring GDPR compliance."""

    def __init__(self, base_service: ModelService):
        self.base_service = base_service
        self.privacy_manager = DataPrivacyManager()
        self.user_consents = {}  # In production, use database

    async def predict_with_compliance(
        self,
        model_id: str,
        inputs: list[str],
        user_id: str,
        processing_purpose: str = "cybersecurity_analysis"
    ) -> dict:
        """Perform prediction with GDPR compliance checks."""

        # Check user consent
        if not self._has_valid_consent(user_id, processing_purpose):
            raise HTTPException(
                status_code=403,
                detail="No valid consent for data processing"
            )

        # Detect and handle PII
        processed_inputs = []
        pii_detections = []

        for input_text in inputs:
            pii_detected = self.privacy_manager.detect_pii(input_text)

            if pii_detected:
                # Anonymize if PII is detected
                anonymized_text, anonymization_log = self.privacy_manager.anonymize_text(input_text)
                processed_inputs.append(anonymized_text)
                pii_detections.extend(anonymization_log)
            else:
                processed_inputs.append(input_text)

        # Log data processing
        self.privacy_manager.audit_log_access(
            user_id=user_id,
            action="model_prediction",
            data_type="text_input",
            details={
                "model_id": model_id,
                "input_count": len(inputs),
                "pii_detected": len(pii_detections) > 0,
                "processing_purpose": processing_purpose
            }
        )

        # Perform prediction
        response = await self.base_service.predict(model_id, processed_inputs)

        return {
            "predictions": response.data if response.success else [],
            "privacy_summary": {
                "pii_detected": len(pii_detections) > 0,
                "anonymization_applied": len(pii_detections) > 0,
                "processed_inputs": len(processed_inputs)
            },
            "compliance_status": "gdpr_compliant"
        }

    def _has_valid_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has valid consent for processing."""
        consent = self.user_consents.get(user_id)
        if not consent:
            return False

        # Check if consent covers the purpose and is still valid
        return (
            purpose in consent.get("purposes", []) and
            consent.get("expires_at", datetime.min) > datetime.now()
        )

    async def handle_data_subject_request(self, user_id: str, request_type: str) -> dict:
        """Handle GDPR data subject requests (access, deletion, etc.)."""

        if request_type == "access":
            # Provide all data associated with user
            return await self._get_user_data(user_id)

        elif request_type == "deletion":
            # Delete all user data
            return await self._delete_user_data(user_id)

        elif request_type == "portability":
            # Export user data in structured format
            return await self._export_user_data(user_id)

        else:
            raise ValueError(f"Unsupported request type: {request_type}")
```

## Troubleshooting

### Common Issues and Solutions

```python
class TroubleshootingGuide:
    """Automated troubleshooting and diagnostics."""

    @staticmethod
    async def diagnose_service_issues(
        model_service: ModelService,
        eval_service: EvaluationService
    ) -> dict:
        """Run comprehensive service diagnostics."""

        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "model_service": {},
            "evaluation_service": {},
            "system_info": {},
            "recommendations": []
        }

        # Model service diagnostics
        try:
            model_health = await model_service.health_check()
            loaded_models = await model_service.get_loaded_models()
            resource_usage = await model_service.get_resource_usage()

            diagnostics["model_service"] = {
                "health_status": model_health.status.value,
                "loaded_models_count": len(loaded_models.data) if loaded_models.success else 0,
                "memory_usage_mb": resource_usage.data.get("memory_usage_mb", 0),
                "cpu_usage_percent": resource_usage.data.get("cpu_usage_percent", 0)
            }

            # Check for common issues
            if resource_usage.data.get("memory_usage_mb", 0) > 8192:
                diagnostics["recommendations"].append(
                    "High memory usage detected. Consider unloading unused models."
                )

            if len(loaded_models.data) == 0:
                diagnostics["recommendations"].append(
                    "No models loaded. Load at least one model to enable predictions."
                )

        except Exception as e:
            diagnostics["model_service"]["error"] = str(e)
            diagnostics["recommendations"].append(
                "Model service error detected. Check service initialization and configuration."
            )

        # Evaluation service diagnostics
        try:
            eval_health = await eval_service.health_check()
            available_metrics = await eval_service.get_available_metrics()

            diagnostics["evaluation_service"] = {
                "health_status": eval_health.status.value,
                "registered_evaluators": available_metrics.data.get("total_evaluators", 0)
            }

            if available_metrics.data.get("total_evaluators", 0) == 0:
                diagnostics["recommendations"].append(
                    "No metric evaluators registered. Register evaluators before running evaluations."
                )

        except Exception as e:
            diagnostics["evaluation_service"]["error"] = str(e)
            diagnostics["recommendations"].append(
                "Evaluation service error detected. Check service initialization."
            )

        # System diagnostics
        import psutil
        diagnostics["system_info"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }

        # System-level recommendations
        if psutil.virtual_memory().percent > 85:
            diagnostics["recommendations"].append(
                "System memory usage is high. Consider reducing concurrent operations."
            )

        if psutil.disk_usage('/').percent > 90:
            diagnostics["recommendations"].append(
                "Disk space is low. Clean up temporary files and old results."
            )

        return diagnostics

    @staticmethod
    def get_error_solution(error_type: str, error_message: str) -> dict:
        """Get solutions for common errors."""

        solutions = {
            "ModelLoadingError": {
                "description": "Failed to load model",
                "common_causes": [
                    "Model file not found or corrupted",
                    "Insufficient memory",
                    "Incorrect model configuration",
                    "Plugin not available"
                ],
                "solutions": [
                    "Verify model file path and permissions",
                    "Check available memory and unload unused models",
                    "Validate model configuration parameters",
                    "Ensure required plugin is installed and configured"
                ]
            },

            "ValidationError": {
                "description": "Input validation failed",
                "common_causes": [
                    "Empty or malformed input data",
                    "Incorrect data format",
                    "Missing required fields",
                    "Data size exceeds limits"
                ],
                "solutions": [
                    "Check input data format and completeness",
                    "Ensure predictions and ground truth have same length",
                    "Verify all required fields are present",
                    "Reduce batch size if data is too large"
                ]
            },

            "ServiceUnavailableError": {
                "description": "Service temporarily unavailable",
                "common_causes": [
                    "Service overloaded",
                    "Resource exhaustion",
                    "Service shutting down",
                    "Network connectivity issues"
                ],
                "solutions": [
                    "Reduce concurrent request load",
                    "Check system resources and clean up",
                    "Wait for service to restart",
                    "Verify network connectivity and firewall settings"
                ]
            },

            "ConfigurationError": {
                "description": "Configuration error",
                "common_causes": [
                    "Invalid configuration file",
                    "Missing environment variables",
                    "Incorrect parameter values",
                    "Configuration file not found"
                ],
                "solutions": [
                    "Validate configuration file syntax",
                    "Set required environment variables",
                    "Check parameter value ranges and types",
                    "Ensure configuration file exists and is readable"
                ]
            }
        }

        default_solution = {
            "description": "Unknown error",
            "common_causes": ["Unexpected error occurred"],
            "solutions": [
                "Check logs for detailed error information",
                "Restart services if problem persists",
                "Contact support with error details"
            ]
        }

        return solutions.get(error_type, default_solution)

# Self-healing service wrapper
class SelfHealingService:
    """Wrapper that provides self-healing capabilities."""

    def __init__(self, base_service: BaseService):
        self.base_service = base_service
        self.healing_attempts = {}
        self.max_healing_attempts = 3
        self.healing_cooldown = 300  # 5 minutes

    async def execute_with_healing(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute operation with automatic healing on failure."""

        try:
            return await operation_func(*args, **kwargs)

        except Exception as e:
            error_type = type(e).__name__

            # Check if we should attempt healing
            if self._should_attempt_healing(operation_name, error_type):
                print(f"Attempting self-healing for {error_type} in {operation_name}")

                # Attempt healing
                healed = await self._attempt_healing(error_type, str(e))

                if healed:
                    # Retry operation after healing
                    try:
                        return await operation_func(*args, **kwargs)
                    except Exception as retry_error:
                        print(f"Operation failed after healing: {retry_error}")
                        raise
                else:
                    print(f"Healing unsuccessful for {error_type}")
                    raise
            else:
                # Max attempts reached or no healing available
                raise

    def _should_attempt_healing(self, operation_name: str, error_type: str) -> bool:
        """Check if healing should be attempted."""

        key = f"{operation_name}:{error_type}"
        current_time = time.time()

        if key not in self.healing_attempts:
            self.healing_attempts[key] = {"count": 0, "last_attempt": 0}

        attempt_info = self.healing_attempts[key]

        # Check if within cooldown period
        if current_time - attempt_info["last_attempt"] < self.healing_cooldown:
            return False

        # Check if max attempts reached
        if attempt_info["count"] >= self.max_healing_attempts:
            return False

        return True

    async def _attempt_healing(self, error_type: str, error_message: str) -> bool:
        """Attempt to heal the service based on error type."""

        try:
            if error_type == "ModelLoadingError":
                # Attempt to free memory and reload models
                if hasattr(self.base_service, 'cleanup_unused_models'):
                    await self.base_service.cleanup_unused_models()
                return True

            elif error_type == "ServiceUnavailableError":
                # Wait for service recovery
                await asyncio.sleep(5)
                return True

            elif error_type == "ValidationError":
                # Not much we can do for validation errors
                return False

            else:
                # Generic healing attempt
                if hasattr(self.base_service, 'health_check'):
                    health = await self.base_service.health_check()
                    return health.status != ServiceStatus.ERROR

                return False

        except Exception as healing_error:
            print(f"Healing attempt failed: {healing_error}")
            return False

# Usage example
async def troubleshooting_example():
    # Initialize services
    container = BenchmarkContainer()
    model_service = await container.model_service
    eval_service = await container.evaluation_service

    # Run diagnostics
    diagnostics = await TroubleshootingGuide.diagnose_service_issues(
        model_service, eval_service
    )

    print("🔧 System Diagnostics")
    print("="*30)
    print(f"Model Service Health: {diagnostics['model_service'].get('health_status', 'Unknown')}")
    print(f"Evaluation Service Health: {diagnostics['evaluation_service'].get('health_status', 'Unknown')}")
    print(f"System Memory Usage: {diagnostics['system_info']['memory_total_gb'] - diagnostics['system_info']['memory_available_gb']:.1f}GB")

    if diagnostics["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in diagnostics["recommendations"]:
            print(f"  • {rec}")

    # Create self-healing wrapper
    healing_model_service = SelfHealingService(model_service)

    # Use with automatic healing
    try:
        result = await healing_model_service.execute_with_healing(
            "model_prediction",
            model_service.predict,
            "SecurityBERT",
            ["Test input"]
        )
        print("Prediction successful:", result.success)

    except Exception as e:
        # Get error solution
        solution = TroubleshootingGuide.get_error_solution(type(e).__name__, str(e))
        print(f"\n🆘 Error Solution for {type(e).__name__}:")
        print(f"Description: {solution['description']}")
        print("Possible solutions:")
        for sol in solution['solutions']:
            print(f"  • {sol}")
```

---

## Conclusion

This integration guide provides comprehensive examples for integrating the LLM Cybersecurity Benchmark system into your applications. The modular architecture supports everything from simple script integration to complex production deployments with monitoring, security, and compliance features.

For additional examples and advanced use cases, refer to the [examples/](../examples/) directory and the API documentation for specific services.

## See Also

- [Model Service API Documentation](model_service_api.md)
- [Evaluation Service API Documentation](evaluation_service_api.md)
- [Performance Benchmarks](../benchmarks/performance_results.md)
- [Configuration Reference](../configs/default.yaml)
- [CLI Reference](../CLI_README.md)

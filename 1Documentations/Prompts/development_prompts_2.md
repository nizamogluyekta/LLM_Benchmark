# Development Prompts 2: LLM Cybersecurity Benchmark (Weeks 5-8)

## How to Use This Guide

This is the continuation of development_prompts.md, covering weeks 5-8 of the project. Each prompt builds on the work completed in the first 4 weeks and follows the same format:

- **Sequential**: Follow them in order, each builds on the previous
- **Granular**: Small, focused tasks that can be completed in one session
- **Self-contained**: Each prompt includes context, requirements, and validation
- **Testing-focused**: Includes appropriate tests for each component

### Prompt Format
- 🎯 **Goal**: What we're building
- 📁 **Files**: Files to create/modify
- 🔧 **Task**: Specific implementation requirements
- ✅ **Tests**: Testing requirements
- 🔍 **Validation**: How to verify it works

---

## Phase 4 Continuation: Model Service Completion (Weeks 5-7)

*Building on the Model Service foundation from development_prompts.md*

### 4.4: Model Service Integration and Optimization

#### Prompt 4.4.1: Create Model Resource Management
🎯 **Goal**: Implement intelligent resource management for multiple models on M4 Pro hardware

📁 **Files**:
- `src/benchmark/models/resource_manager.py`
- `src/benchmark/models/model_cache.py`

🔧 **Task**:
Create a resource management system that efficiently handles multiple models within the memory constraints of MacBook Pro M4 Pro.

Requirements:
- Monitor memory usage across loaded models
- Implement model unloading/reloading strategies
- Optimize for Apple Silicon unified memory architecture
- Support concurrent model inference with resource limits
- Provide recommendations for optimal model combinations

```python
class ModelResourceManager:
    def __init__(self, max_memory_gb: float = 32.0):  # Conservative for M4 Pro
        self.max_memory_gb = max_memory_gb
        self.loaded_models: Dict[str, LoadedModelInfo] = {}
        self.memory_monitor = MemoryMonitor()

    async def can_load_model(self, model_config: ModelConfig) -> ResourceCheckResult:
        """Check if model can be loaded within resource constraints"""
        estimated_memory = await self._estimate_model_memory(model_config)
        current_usage = await self.memory_monitor.get_current_usage()

        return ResourceCheckResult(
            can_load=current_usage + estimated_memory <= self.max_memory_gb,
            estimated_memory_gb=estimated_memory,
            current_usage_gb=current_usage,
            recommendations=self._get_optimization_recommendations(model_config)
        )

    async def optimize_model_loading_order(self, model_configs: List[ModelConfig]) -> List[ModelConfig]:
        """Optimize the order of model loading for best resource utilization"""

    async def suggest_model_unload_candidates(self, required_memory_gb: float) -> List[str]:
        """Suggest which models to unload to free up required memory"""

class ModelCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.lru_cache = {}

    async def cache_model_state(self, model_id: str, model_state: Any) -> None:
        """Cache model state to disk for fast reloading"""

    async def load_cached_model(self, model_id: str) -> Optional[Any]:
        """Load model from cache if available"""

    async def cleanup_old_cache(self, max_age_days: int = 7) -> None:
        """Clean up old cached models"""

class MemoryMonitor:
    def __init__(self):
        import psutil
        self.process = psutil.Process()

    async def get_current_usage(self) -> float:
        """Get current memory usage in GB"""
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024**3)  # Convert to GB

    async def get_system_memory_info(self) -> SystemMemoryInfo:
        """Get system memory information"""
        import psutil
        memory = psutil.virtual_memory()
        return SystemMemoryInfo(
            total_gb=memory.total / (1024**3),
            available_gb=memory.available / (1024**3),
            used_gb=memory.used / (1024**3),
            percentage=memory.percent
        )

@dataclass
class LoadedModelInfo:
    model_id: str
    config: ModelConfig
    memory_usage_gb: float
    last_used: datetime
    plugin_type: str

@dataclass
class ResourceCheckResult:
    can_load: bool
    estimated_memory_gb: float
    current_usage_gb: float
    recommendations: List[str]

@dataclass
class SystemMemoryInfo:
    total_gb: float
    available_gb: float
    used_gb: float
    percentage: float
```

✅ **Tests**:
Create `tests/unit/test_resource_manager.py`:
- Test memory usage estimation for different model sizes
- Test resource constraint checking
- Test model loading order optimization
- Test cache management functionality

Create `tests/integration/test_resource_integration.py`:
- Test with actual model loading on M4 Pro
- Test concurrent model loading within memory limits
- Test model unloading and reloading scenarios

🔍 **Validation**:
- Resource manager prevents memory overflow on M4 Pro
- Model caching reduces reload times significantly
- Memory monitoring provides accurate usage data
- Resource optimization recommendations are practical
- Tests pass on target hardware

#### Prompt 4.4.2: Create Model Service Performance Optimization
🎯 **Goal**: Optimize model service for maximum performance on Apple Silicon M4 Pro

📁 **Files**:
- Modify `src/benchmark/services/model_service.py`
- `src/benchmark/models/optimization.py`

🔧 **Task**:
Implement performance optimizations specifically for Apple Silicon M4 Pro architecture.

Requirements:
- Optimize batch processing for different model types
- Implement async inference with proper concurrency limits
- Add Apple Silicon specific optimizations (Metal, Neural Engine)
- Create adaptive batch sizing based on model performance
- Implement request queuing and priority management

```python
# Add to ModelService class:
class ModelService(BaseService):
    def __init__(self):
        self.plugins: Dict[str, ModelPlugin] = {}
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.resource_manager = ModelResourceManager()
        self.performance_optimizer = AppleSiliconOptimizer()
        self.inference_queue = InferenceQueue()

    async def optimize_for_hardware(self) -> None:
        """Apply Apple Silicon specific optimizations"""
        hardware_info = await self.performance_optimizer.detect_hardware()

        if hardware_info.has_neural_engine:
            await self._enable_neural_engine_acceleration()
        if hardware_info.has_metal_gpu:
            await self._enable_metal_acceleration()

        # Set optimal batch sizes for detected hardware
        self.optimal_batch_sizes = await self.performance_optimizer.calculate_optimal_batch_sizes(
            available_memory=hardware_info.memory_gb,
            gpu_cores=hardware_info.gpu_cores
        )

    async def predict_batch_optimized(self, model_id: str, samples: List[str],
                                    priority: Priority = Priority.NORMAL) -> List[Prediction]:
        """Optimized batch prediction with dynamic batching"""

        # Determine optimal batch size for this model
        optimal_batch_size = self.optimal_batch_sizes.get(model_id, 32)

        # Queue request with priority
        request = InferenceRequest(
            model_id=model_id,
            samples=samples,
            batch_size=optimal_batch_size,
            priority=priority,
            timestamp=datetime.now()
        )

        return await self.inference_queue.process_request(request)

class AppleSiliconOptimizer:
    async def detect_hardware(self) -> HardwareInfo:
        """Detect M4 Pro specific hardware capabilities"""
        import platform
        import subprocess

        # Detect Apple Silicon
        is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'

        if is_apple_silicon:
            # Get M4 Pro specific info
            gpu_info = await self._get_metal_gpu_info()
            memory_info = await self._get_unified_memory_info()
            neural_engine = await self._detect_neural_engine()

            return HardwareInfo(
                is_apple_silicon=True,
                chip_type="M4 Pro",
                memory_gb=memory_info['unified_memory'],
                gpu_cores=gpu_info['cores'],
                has_metal_gpu=True,
                has_neural_engine=neural_engine,
                performance_cores=10,  # M4 Pro typical
                efficiency_cores=4
            )

        return HardwareInfo(is_apple_silicon=False)

    async def calculate_optimal_batch_sizes(self, available_memory: float,
                                          gpu_cores: int) -> Dict[str, int]:
        """Calculate optimal batch sizes for different model types on M4 Pro"""

        # Conservative estimates for M4 Pro
        batch_sizes = {}

        if available_memory >= 48:  # 48GB+ M4 Pro
            batch_sizes = {
                'mlx_3b': 64,
                'mlx_7b': 32,
                'api_models': 16,  # Limited by API rate limits
                'default': 32
            }
        elif available_memory >= 24:  # 24GB M4 Pro
            batch_sizes = {
                'mlx_3b': 32,
                'mlx_7b': 16,
                'api_models': 16,
                'default': 16
            }
        else:  # Base configuration
            batch_sizes = {
                'mlx_3b': 16,
                'mlx_7b': 8,
                'api_models': 8,
                'default': 8
            }

        return batch_sizes

class InferenceQueue:
    def __init__(self, max_concurrent: int = 3):  # Conservative for M4 Pro
        self.max_concurrent = max_concurrent
        self.queue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_request(self, request: InferenceRequest) -> List[Prediction]:
        """Process inference request with concurrency control"""

        async with self.semaphore:  # Limit concurrent inference
            # Add to active requests for monitoring
            self.active_requests[request.request_id] = request

            try:
                # Process in batches with optimal size
                predictions = []
                for batch in self._create_batches(request.samples, request.batch_size):
                    batch_predictions = await self._process_batch(request.model_id, batch)
                    predictions.extend(batch_predictions)

                return predictions
            finally:
                # Remove from active requests
                self.active_requests.pop(request.request_id, None)

    def _create_batches(self, samples: List[str], batch_size: int) -> List[List[str]]:
        """Create batches of optimal size"""
        return [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

@dataclass
class HardwareInfo:
    is_apple_silicon: bool
    chip_type: Optional[str] = None
    memory_gb: Optional[float] = None
    gpu_cores: Optional[int] = None
    has_metal_gpu: bool = False
    has_neural_engine: bool = False
    performance_cores: Optional[int] = None
    efficiency_cores: Optional[int] = None

@dataclass
class InferenceRequest:
    model_id: str
    samples: List[str]
    batch_size: int
    priority: Priority
    timestamp: datetime
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class Priority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
```

✅ **Tests**:
Create `tests/unit/test_performance_optimization.py`:
- Test hardware detection on M4 Pro
- Test optimal batch size calculation
- Test inference queue priority management
- Mock Apple Silicon specific features

Create `tests/performance/test_apple_silicon_performance.py`:
- Benchmark inference performance before/after optimization
- Test concurrent inference performance
- Test memory usage under optimal settings
- Compare performance across different batch sizes

🔍 **Validation**:
- Hardware detection correctly identifies M4 Pro capabilities
- Batch size optimization improves throughput
- Concurrent inference works within memory limits
- Performance benchmarks show measurable improvement
- All optimization tests pass

#### Prompt 4.4.3: Complete Model Service Integration Testing
🎯 **Goal**: Create comprehensive integration tests for the complete Model Service

📁 **Files**: `tests/integration/test_complete_model_service.py`

🔧 **Task**:
Create thorough integration tests that validate the complete Model Service functionality with all plugins and optimizations.

Test scenarios to implement:
- Load and use each model plugin type (MLX, OpenAI, Anthropic, Ollama)
- Test resource management with multiple concurrent models
- Test performance optimization on M4 Pro hardware
- Test error recovery and resilience
- Test model service integration with Configuration Service
- Validate cybersecurity-specific model outputs

```python
import pytest
from typing import List, Dict
import asyncio
import time
from unittest.mock import patch, MagicMock

from benchmark.services.model_service import ModelService
from benchmark.services.configuration_service import ConfigurationService
from benchmark.core.config import ModelConfig, ModelType
from benchmark.models.performance_monitor import ModelPerformanceMonitor

class TestCompleteModelService:
    """Comprehensive integration tests for Model Service"""

    @pytest.fixture
    async def model_service(self, config_service):
        """Configured model service with all plugins"""
        service = ModelService()
        await service.initialize()
        return service

    @pytest.fixture
    def sample_cybersecurity_data(self):
        """Sample cybersecurity data for testing"""
        return [
            "192.168.1.100 -> 10.0.0.5 PORT_SCAN detected on ports 22,23,80,443",
            "2024-01-15 14:32:18 [INFO] User authentication successful: admin@company.com",
            "TCP connection established: 203.0.113.42:4444 -> 192.168.1.50:1337 SUSPICIOUS",
            "Email received with attachment: invoice.pdf.exe from unknown-sender@malicious.com",
            "Normal HTTP GET request to /api/users/profile from authenticated session"
        ]

    async def test_load_all_model_types(self, model_service, config_service):
        """Test loading models from all available plugin types"""

        # Test configurations for each model type
        model_configs = [
            ModelConfig(
                name="test_mlx_model",
                type=ModelType.MLX_LOCAL,
                path="mlx-community/Llama-3.2-3B-Instruct-4bit",
                max_tokens=256,
                temperature=0.1
            ),
            ModelConfig(
                name="test_openai_model",
                type=ModelType.OPENAI_API,
                path="gpt-4o-mini",
                max_tokens=256,
                temperature=0.1
            ),
            ModelConfig(
                name="test_anthropic_model",
                type=ModelType.ANTHROPIC_API,
                path="claude-3-haiku-20240307",
                max_tokens=256,
                temperature=0.1
            )
        ]

        loaded_model_ids = []

        for config in model_configs:
            try:
                model_id = await model_service.load_model(config)
                loaded_model_ids.append(model_id)

                # Verify model info can be retrieved
                model_info = await model_service.get_model_info(model_id)
                assert model_info.success
                assert model_info.data['model_name'] == config.name

            except Exception as e:
                # Skip if model not available (e.g., missing API keys)
                if "API key" in str(e) or "not found" in str(e):
                    pytest.skip(f"Skipping {config.name}: {e}")
                else:
                    raise

        # Ensure at least one model loaded successfully
        assert len(loaded_model_ids) >= 1, "At least one model should load successfully"

        # Cleanup
        for model_id in loaded_model_ids:
            await model_service.cleanup_model(model_id)

    async def test_cybersecurity_prediction_pipeline(self, model_service, sample_cybersecurity_data):
        """Test complete cybersecurity prediction pipeline"""

        # Load a test model (mock if necessary)
        model_config = ModelConfig(
            name="test_cyber_model",
            type=ModelType.MLX_LOCAL,
            path="test://mock-model",  # Use mock for testing
            max_tokens=512,
            temperature=0.1
        )

        with patch.object(model_service, '_load_model_plugin') as mock_load:
            # Mock the model plugin
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [
                {
                    'sample_id': str(i),
                    'input_text': sample,
                    'prediction': 'ATTACK' if 'SUSPICIOUS' in sample or 'PORT_SCAN' in sample else 'BENIGN',
                    'confidence': 0.95 if 'SUSPICIOUS' in sample else 0.75,
                    'attack_type': 'reconnaissance' if 'PORT_SCAN' in sample else None,
                    'explanation': f'Analysis of: {sample[:50]}...',
                    'inference_time_ms': 150.0
                }
                for i, sample in enumerate(sample_cybersecurity_data)
            ]
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Test batch prediction
            predictions = await model_service.predict_batch(model_id, sample_cybersecurity_data)

            # Validate predictions
            assert len(predictions) == len(sample_cybersecurity_data)

            for prediction in predictions:
                # Validate required fields
                assert 'prediction' in prediction
                assert prediction['prediction'] in ['ATTACK', 'BENIGN']
                assert 'confidence' in prediction
                assert 0.0 <= prediction['confidence'] <= 1.0
                assert 'inference_time_ms' in prediction
                assert prediction['inference_time_ms'] > 0

                # Validate cybersecurity-specific fields
                if prediction['prediction'] == 'ATTACK':
                    assert 'attack_type' in prediction
                    assert 'explanation' in prediction
                    assert len(prediction['explanation']) > 0

    async def test_resource_management_multiple_models(self, model_service):
        """Test resource management with multiple concurrent models"""

        # Create multiple model configs
        model_configs = [
            ModelConfig(name=f"model_{i}", type=ModelType.MLX_LOCAL,
                       path="test://mock-model", max_tokens=256)
            for i in range(3)
        ]

        # Mock resource manager to simulate M4 Pro constraints
        with patch.object(model_service.resource_manager, 'can_load_model') as mock_check:
            mock_check.return_value = ResourceCheckResult(
                can_load=True,
                estimated_memory_gb=8.0,
                current_usage_gb=16.0,
                recommendations=["Consider using quantized models"]
            )

            with patch.object(model_service, '_load_model_plugin') as mock_load:
                mock_plugin = MagicMock()
                mock_plugin.predict.return_value = [{'prediction': 'BENIGN', 'confidence': 0.8}]
                mock_load.return_value = mock_plugin

                # Load multiple models
                model_ids = []
                for config in model_configs:
                    model_id = await model_service.load_model(config)
                    model_ids.append(model_id)

                # Test concurrent predictions
                test_samples = ["Test network log entry"]

                tasks = [
                    model_service.predict_batch(model_id, test_samples)
                    for model_id in model_ids
                ]

                results = await asyncio.gather(*tasks)

                # Validate all predictions completed
                assert len(results) == len(model_ids)
                for result in results:
                    assert len(result) == len(test_samples)

                # Cleanup
                for model_id in model_ids:
                    await model_service.cleanup_model(model_id)

    async def test_performance_optimization_integration(self, model_service):
        """Test performance optimization features"""

        # Test hardware detection
        hardware_info = await model_service.performance_optimizer.detect_hardware()

        if hardware_info.is_apple_silicon:
            # Validate M4 Pro detection
            assert hardware_info.chip_type is not None
            assert hardware_info.memory_gb is not None
            assert hardware_info.memory_gb > 0

            # Test optimization application
            await model_service.optimize_for_hardware()

            # Verify optimal batch sizes were set
            assert hasattr(model_service, 'optimal_batch_sizes')
            assert len(model_service.optimal_batch_sizes) > 0

            # Test optimized batch prediction
            model_config = ModelConfig(
                name="optimized_model",
                type=ModelType.MLX_LOCAL,
                path="test://mock-model"
            )

            with patch.object(model_service, '_load_model_plugin') as mock_load:
                mock_plugin = MagicMock()
                mock_plugin.predict.return_value = [
                    {'prediction': 'BENIGN', 'confidence': 0.8, 'inference_time_ms': 100.0}
                    for _ in range(32)  # Batch size
                ]
                mock_load.return_value = mock_plugin

                model_id = await model_service.load_model(model_config)

                # Test optimized batch processing
                large_sample_set = ["Test sample"] * 100
                predictions = await model_service.predict_batch_optimized(model_id, large_sample_set)

                assert len(predictions) == 100

                await model_service.cleanup_model(model_id)
        else:
            pytest.skip("Apple Silicon specific tests - running on different hardware")

    async def test_error_recovery_and_resilience(self, model_service):
        """Test error handling and recovery scenarios"""

        # Test model loading failure
        invalid_config = ModelConfig(
            name="invalid_model",
            type=ModelType.MLX_LOCAL,
            path="nonexistent://invalid-path"
        )

        result = await model_service.load_model(invalid_config)
        assert not hasattr(model_service.loaded_models, result)  # Should not be loaded

        # Test prediction with invalid model_id
        with pytest.raises(Exception):  # Should raise appropriate exception
            await model_service.predict_batch("nonexistent_model", ["test"])

        # Test recovery from API failures
        api_config = ModelConfig(
            name="api_model",
            type=ModelType.OPENAI_API,
            path="gpt-4o-mini"
        )

        with patch.object(model_service, '_load_model_plugin') as mock_load:
            mock_plugin = MagicMock()
            # Simulate API failure then success
            mock_plugin.predict.side_effect = [
                Exception("API Error"),
                [{'prediction': 'BENIGN', 'confidence': 0.8}]
            ]
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(api_config)

            # First call should handle error gracefully
            with pytest.raises(Exception):
                await model_service.predict_batch(model_id, ["test"])

            # Second call should succeed
            result = await model_service.predict_batch(model_id, ["test"])
            assert len(result) == 1
            assert result[0]['prediction'] == 'BENIGN'

    async def test_performance_monitoring_integration(self, model_service):
        """Test integration with performance monitoring"""

        model_config = ModelConfig(
            name="monitored_model",
            type=ModelType.MLX_LOCAL,
            path="test://mock-model"
        )

        with patch.object(model_service, '_load_model_plugin') as mock_load:
            mock_plugin = MagicMock()
            mock_plugin.predict.return_value = [
                {'prediction': 'ATTACK', 'confidence': 0.9, 'inference_time_ms': 200.0}
            ]
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Make several predictions to generate performance data
            for _ in range(5):
                await model_service.predict_batch(model_id, ["test sample"])

            # Check performance metrics were collected
            performance_summary = await model_service.get_model_performance(model_id)

            assert performance_summary.success
            performance_data = performance_summary.data

            assert 'avg_inference_time_ms' in performance_data
            assert 'total_predictions' in performance_data
            assert performance_data['total_predictions'] == 5

            await model_service.cleanup_model(model_id)

    async def test_integration_with_configuration_service(self, model_service, config_service):
        """Test Model Service integration with Configuration Service"""

        # Create experiment config with multiple models
        experiment_config = {
            'name': 'Integration Test Experiment',
            'models': [
                {
                    'name': 'test_model_1',
                    'type': 'mlx_local',
                    'path': 'test://mock-model-1',
                    'max_tokens': 256
                },
                {
                    'name': 'test_model_2',
                    'type': 'mlx_local',
                    'path': 'test://mock-model-2',
                    'max_tokens': 512
                }
            ]
        }

        # Load experiment config through config service
        with patch.object(config_service, 'load_experiment_config') as mock_load_config:
            mock_load_config.return_value = experiment_config

            config = await config_service.load_experiment_config('test_config.yaml')

            # Load models based on configuration
            with patch.object(model_service, '_load_model_plugin') as mock_load:
                mock_plugin = MagicMock()
                mock_plugin.predict.return_value = [{'prediction': 'BENIGN'}]
                mock_load.return_value = mock_plugin

                model_ids = []
                for model_config_dict in config['models']:
                    model_config = ModelConfig(**model_config_dict)
                    model_id = await model_service.load_model(model_config)
                    model_ids.append(model_id)

                # Verify all models loaded successfully
                assert len(model_ids) == 2

                # Test that models work with configuration settings
                for model_id in model_ids:
                    predictions = await model_service.predict_batch(model_id, ["test"])
                    assert len(predictions) == 1

                # Cleanup
                for model_id in model_ids:
                    await model_service.cleanup_model(model_id)

    @pytest.mark.performance
    async def test_realistic_cybersecurity_workload(self, model_service, sample_cybersecurity_data):
        """Test Model Service with realistic cybersecurity evaluation workload"""

        # Simulate realistic workload: multiple models, larger dataset
        large_dataset = sample_cybersecurity_data * 20  # 100 samples total

        model_config = ModelConfig(
            name="cyber_workload_model",
            type=ModelType.MLX_LOCAL,
            path="test://mock-cyber-model"
        )

        with patch.object(model_service, '_load_model_plugin') as mock_load:
            mock_plugin = MagicMock()

            def mock_predict(samples):
                return [
                    {
                        'sample_id': str(i),
                        'input_text': sample,
                        'prediction': 'ATTACK' if any(indicator in sample for indicator in
                                                    ['SUSPICIOUS', 'PORT_SCAN', 'malicious']) else 'BENIGN',
                        'confidence': 0.85,
                        'attack_type': 'malware' if 'malicious' in sample else None,
                        'explanation': f'Cybersecurity analysis of log entry',
                        'inference_time_ms': 180.0
                    }
                    for i, sample in enumerate(samples)
                ]

            mock_plugin.predict.side_effect = mock_predict
            mock_load.return_value = mock_plugin

            model_id = await model_service.load_model(model_config)

            # Measure performance
            start_time = time.time()
            predictions = await model_service.predict_batch(model_id, large_dataset)
            total_time = time.time() - start_time

            # Validate results
            assert len(predictions) == len(large_dataset)

            # Calculate performance metrics
            throughput = len(predictions) / total_time

            # Validate performance meets expectations
            assert throughput > 10, f"Throughput too low: {throughput} samples/sec"

            # Validate prediction quality
            attack_predictions = [p for p in predictions if p['prediction'] == 'ATTACK']
            benign_predictions = [p for p in predictions if p['prediction'] == 'BENIGN']

            assert len(attack_predictions) > 0, "Should detect some attacks"
            assert len(benign_predictions) > 0, "Should detect some benign traffic"

            # All predictions should have required fields
            for prediction in predictions:
                assert all(field in prediction for field in
                         ['prediction', 'confidence', 'inference_time_ms'])
                assert prediction['prediction'] in ['ATTACK', 'BENIGN']
                assert 0.0 <= prediction['confidence'] <= 1.0

            await model_service.cleanup_model(model_id)

# Additional test fixtures and utilities
@pytest.fixture
def resource_check_result():
    """Sample resource check result for testing"""
    return ResourceCheckResult(
        can_load=True,
        estimated_memory_gb=8.0,
        current_usage_gb=16.0,
        recommendations=["Use quantized models for better memory efficiency"]
    )

@pytest.fixture
def mock_hardware_info():
    """Mock M4 Pro hardware info for testing"""
    return HardwareInfo(
        is_apple_silicon=True,
        chip_type="M4 Pro",
        memory_gb=48.0,
        gpu_cores=20,
        has_metal_gpu=True,
        has_neural_engine=True,
        performance_cores=10,
        efficiency_cores=4
    )

# Performance test utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking"""

    @staticmethod
    async def benchmark_inference_speed(model_service, model_id: str,
                                       samples: List[str], iterations: int = 3) -> Dict[str, float]:
        """Benchmark inference speed for a model"""
        times = []

        for _ in range(iterations):
            start_time = time.time()
            await model_service.predict_batch(model_id, samples)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        throughput = len(samples) / avg_time

        return {
            'avg_inference_time_sec': avg_time,
            'throughput_samples_per_sec': throughput,
            'samples_processed': len(samples),
            'iterations': iterations
        }

    @staticmethod
    async def benchmark_memory_usage(model_service, model_configs: List[ModelConfig]) -> Dict[str, float]:
        """Benchmark memory usage for different model configurations"""
        import psutil
        process = psutil.Process()

        initial_memory = process.memory_info().rss / (1024**3)  # GB

        model_ids = []
        memory_usage = {}

        try:
            for config in model_configs:
                model_id = await model_service.load_model(config)
                model_ids.append(model_id)

                current_memory = process.memory_info().rss / (1024**3)
                memory_usage[config.name] = current_memory - initial_memory

        finally:
            # Cleanup
            for model_id in model_ids:
                try:
                    await model_service.cleanup_model(model_id)
                except:
                    pass  # Ignore cleanup errors

        return memory_usage
```

✅ **Tests**:
The entire file is comprehensive integration testing. Additional supporting tests:

Create `tests/integration/test_model_service_edge_cases.py`:
- Test edge cases and error conditions
- Test with various model configurations
- Test resource exhaustion scenarios

🔍 **Validation**:
- All model plugin types can be loaded and used
- Cybersecurity prediction pipeline works end-to-end
- Resource management prevents memory overflow
- Performance optimization shows measurable improvements
- Error recovery handles failures gracefully
- Integration with Configuration Service works correctly
- Realistic workloads complete successfully within performance expectations

---

## Phase 5: Basic Evaluation Service (Weeks 7-8)

### 5.1: Evaluation Service Foundation

#### Prompt 5.1.1: Create Evaluation Service Base Structure
🎯 **Goal**: Create the foundation for the evaluation service with plugin architecture for different metrics

📁 **Files**:
- `src/benchmark/services/evaluation_service.py`
- `src/benchmark/interfaces/evaluation_interfaces.py`
- `src/benchmark/evaluation/base_evaluator.py`

🔧 **Task**:
Create the evaluation service foundation that manages different evaluation metrics through a plugin architecture.

Requirements:
- Implement BaseService interface
- Plugin registry for different evaluation metrics
- Standardized evaluation pipeline
- Results aggregation and storage
- Parallel evaluation support
- Progress tracking for long evaluations

```python
# interfaces/evaluation_interfaces.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    ACCURACY = "accuracy"
    EXPLAINABILITY = "explainability"
    PERFORMANCE = "performance"
    FALSE_POSITIVE_RATE = "false_positive_rate"

@dataclass
class EvaluationRequest:
    experiment_id: str
    model_id: str
    dataset_id: str
    predictions: List[Dict[str, Any]]
    ground_truth: List[Dict[str, Any]]
    metrics: List[MetricType]
    metadata: Dict[str, Any]

@dataclass
class EvaluationResult:
    experiment_id: str
    model_id: str
    dataset_id: str
    metrics: Dict[str, float]
    detailed_results: Dict[str, Any]
    execution_time_seconds: float
    timestamp: str
    metadata: Dict[str, Any]

class MetricEvaluator(ABC):
    """Base interface for all metric evaluators"""

    @abstractmethod
    async def evaluate(self, predictions: List[Dict[str, Any]],
                      ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate predictions against ground truth"""
        pass

    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get list of metrics this evaluator produces"""
        pass

    @abstractmethod
    def get_required_prediction_fields(self) -> List[str]:
        """Get required fields in prediction data"""
        pass

    @abstractmethod
    def get_required_ground_truth_fields(self) -> List[str]:
        """Get required fields in ground truth data"""
        pass

# services/evaluation_service.py
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import time

from benchmark.core.base import BaseService, ServiceResponse, HealthCheck, ServiceStatus
from benchmark.interfaces.evaluation_interfaces import (
    MetricEvaluator, EvaluationRequest, EvaluationResult, MetricType
)

class EvaluationService(BaseService):
    """Service for evaluating model predictions using various metrics"""

    def __init__(self):
        self.evaluators: Dict[MetricType, MetricEvaluator] = {}
        self.evaluation_history: List[EvaluationResult] = []
        self.active_evaluations: Dict[str, EvaluationRequest] = {}

    async def initialize(self) -> ServiceResponse:
        """Initialize evaluation service and register default evaluators"""
        try:
            await self._register_default_evaluators()
            return ServiceResponse(
                success=True,
                data={"evaluators_registered": len(self.evaluators)}
            )
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def _register_default_evaluators(self) -> None:
        """Register all available metric evaluators"""
        # Import evaluators here to avoid circular imports
        from benchmark.evaluation.metrics.accuracy import AccuracyEvaluator

        self.evaluators[MetricType.ACCURACY] = AccuracyEvaluator()

        # Additional evaluators will be added in subsequent prompts

    async def register_evaluator(self, metric_type: MetricType, evaluator: MetricEvaluator) -> ServiceResponse:
        """Register a new metric evaluator"""
        try:
            self.evaluators[metric_type] = evaluator
            return ServiceResponse(success=True, data={"metric_type": metric_type.value})
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def evaluate_predictions(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate predictions using requested metrics"""
        start_time = time.time()

        # Add to active evaluations for tracking
        self.active_evaluations[request.experiment_id] = request

        try:
            # Validate input data
            validation_result = await self._validate_evaluation_data(request)
            if not validation_result.success:
                raise ValueError(f"Data validation failed: {validation_result.error}")

            # Run evaluations in parallel for different metrics
            evaluation_tasks = []
            for metric_type in request.metrics:
                if metric_type in self.evaluators:
                    task = self._evaluate_metric(
                        metric_type,
                        request.predictions,
                        request.ground_truth
                    )
                    evaluation_tasks.append((metric_type, task))
                else:
                    raise ValueError(f"Evaluator not available for metric: {metric_type}")

            # Execute all evaluations in parallel
            metric_results = {}
            detailed_results = {}

            for metric_type, task in evaluation_tasks:
                try:
                    result = await task
                    metric_results.update(result)
                    detailed_results[metric_type.value] = result
                except Exception as e:
                    # Log error but continue with other metrics
                    detailed_results[metric_type.value] = {"error": str(e)}

            execution_time = time.time() - start_time

            # Create evaluation result
            evaluation_result = EvaluationResult(
                experiment_id=request.experiment_id,
                model_id=request.model_id,
                dataset_id=request.dataset_id,
                metrics=metric_results,
                detailed_results=detailed_results,
                execution_time_seconds=execution_time,
                timestamp=datetime.now().isoformat(),
                metadata=request.metadata
            )

            # Store result in history
            self.evaluation_history.append(evaluation_result)

            return evaluation_result

        finally:
            # Remove from active evaluations
            self.active_evaluations.pop(request.experiment_id, None)

    async def _validate_evaluation_data(self, request: EvaluationRequest) -> ServiceResponse:
        """Validate evaluation request data"""
        try:
            # Check that predictions and ground truth have same length
            if len(request.predictions) != len(request.ground_truth):
                return ServiceResponse(
                    success=False,
                    error="Predictions and ground truth must have same length"
                )

            # Validate that all required fields are present for requested metrics
            for metric_type in request.metrics:
                evaluator = self.evaluators.get(metric_type)
                if evaluator:
                    # Check prediction fields
                    required_pred_fields = evaluator.get_required_prediction_fields()
                    for i, pred in enumerate(request.predictions):
                        for field in required_pred_fields:
                            if field not in pred:
                                return ServiceResponse(
                                    success=False,
                                    error=f"Missing required prediction field '{field}' in sample {i}"
                                )

                    # Check ground truth fields
                    required_gt_fields = evaluator.get_required_ground_truth_fields()
                    for i, gt in enumerate(request.ground_truth):
                        for field in required_gt_fields:
                            if field not in gt:
                                return ServiceResponse(
                                    success=False,
                                    error=f"Missing required ground truth field '{field}' in sample {i}"
                                )

            return ServiceResponse(success=True)

        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def _evaluate_metric(self, metric_type: MetricType,
                              predictions: List[Dict[str, Any]],
                              ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate a specific metric"""
        evaluator = self.evaluators[metric_type]
        return await evaluator.evaluate(predictions, ground_truth)

    async def get_available_metrics(self) -> ServiceResponse:
        """Get list of available evaluation metrics"""
        try:
            metrics_info = {}
            for metric_type, evaluator in self.evaluators.items():
                metrics_info[metric_type.value] = {
                    "metric_names": evaluator.get_metric_names(),
                    "required_prediction_fields": evaluator.get_required_prediction_fields(),
                    "required_ground_truth_fields": evaluator.get_required_ground_truth_fields()
                }

            return ServiceResponse(success=True, data=metrics_info)
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def get_evaluation_history(self, experiment_id: Optional[str] = None,
                                   limit: int = 100) -> ServiceResponse:
        """Get evaluation history, optionally filtered by experiment"""
        try:
            if experiment_id:
                filtered_history = [
                    result for result in self.evaluation_history
                    if result.experiment_id == experiment_id
                ]
            else:
                filtered_history = self.evaluation_history

            # Limit results
            limited_history = filtered_history[-limit:] if limit else filtered_history

            return ServiceResponse(success=True, data=limited_history)
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def health_check(self) -> HealthCheck:
        """Check evaluation service health"""
        try:
            status = ServiceStatus.HEALTHY
            details = {
                "evaluators_count": len(self.evaluators),
                "active_evaluations": len(self.active_evaluations),
                "total_evaluations_completed": len(self.evaluation_history)
            }

            # Check if any evaluators are failing
            if len(self.evaluators) == 0:
                status = ServiceStatus.UNHEALTHY
                details["issue"] = "No evaluators registered"

            return HealthCheck(
                service_name="evaluation_service",
                status=status,
                timestamp=datetime.now().isoformat(),
                details=details
            )
        except Exception as e:
            return HealthCheck(
                service_name="evaluation_service",
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )

    async def shutdown(self) -> ServiceResponse:
        """Graceful shutdown of evaluation service"""
        try:
            # Wait for active evaluations to complete (with timeout)
            if self.active_evaluations:
                await asyncio.sleep(1.0)  # Give active evaluations time to complete

            # Clear evaluators and history
            self.evaluators.clear()
            self.evaluation_history.clear()
            self.active_evaluations.clear()

            return ServiceResponse(success=True, data={"message": "Evaluation service shut down"})
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

# evaluation/base_evaluator.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from benchmark.interfaces.evaluation_interfaces import MetricEvaluator

class BaseEvaluator(MetricEvaluator):
    """Base class with common evaluation utilities"""

    def validate_input_data(self, predictions: List[Dict[str, Any]],
                           ground_truth: List[Dict[str, Any]]) -> None:
        """Validate input data format"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        # Validate required fields
        for i, pred in enumerate(predictions):
            for field in self.get_required_prediction_fields():
                if field not in pred:
                    raise ValueError(f"Missing required prediction field '{field}' in sample {i}")

        for i, gt in enumerate(ground_truth):
            for field in self.get_required_ground_truth_fields():
                if field not in gt:
                    raise ValueError(f"Missing required ground truth field '{field}' in sample {i}")

    def extract_labels(self, predictions: List[Dict[str, Any]],
                      ground_truth: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
        """Extract prediction and ground truth labels"""
        pred_labels = [pred['prediction'] for pred in predictions]
        true_labels = [gt['label'] for gt in ground_truth]
        return pred_labels, true_labels

    def extract_confidences(self, predictions: List[Dict[str, Any]]) -> List[float]:
        """Extract confidence scores from predictions"""
        return [pred.get('confidence', 0.5) for pred in predictions]

    def calculate_basic_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values"""
        if not values:
            return {}

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
```

✅ **Tests**:
Create `tests/unit/test_evaluation_service.py`:
- Test service initialization and evaluator registration
- Test evaluation request validation
- Test parallel metric evaluation
- Test error handling for invalid data
- Test service health checks

Create `tests/unit/test_base_evaluator.py`:
- Test input data validation
- Test label and confidence extraction
- Test statistical calculations

🔍 **Validation**:
- Evaluation service initializes correctly
- Evaluator registration works properly
- Input validation catches data format errors
- Service can handle multiple concurrent evaluations
- Health checks return accurate status
- All tests pass

#### Prompt 5.1.2: Create Accuracy Evaluator
🎯 **Goal**: Implement comprehensive accuracy evaluation metrics for cybersecurity classification

📁 **Files**: `src/benchmark/evaluation/metrics/accuracy.py`

🔧 **Task**:
Create a comprehensive accuracy evaluator that calculates various classification metrics relevant to cybersecurity attack detection.

Requirements:
- Standard classification metrics (precision, recall, F1, accuracy)
- ROC-AUC and PR-AUC for probability-based evaluation
- Per-class metrics for different attack types
- Confusion matrix analysis
- Statistical significance testing
- Handle multi-class and binary classification scenarios

```python
# src/benchmark/evaluation/metrics/accuracy.py
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    matthews_corrcoef
)
from sklearn.preprocessing import LabelBinarizer
import warnings

from benchmark.evaluation.base_evaluator import BaseEvaluator

class AccuracyEvaluator(BaseEvaluator):
    """Comprehensive accuracy evaluation for cybersecurity classification"""

    def __init__(self):
        self.metric_names = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'pr_auc', 'matthews_corr',
            'precision_macro', 'recall_macro', 'f1_macro',
            'true_positive_rate', 'false_positive_rate', 'specificity'
        ]

    async def evaluate(self, predictions: List[Dict[str, Any]],
                      ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics"""

        # Validate input data
        self.validate_input_data(predictions, ground_truth)

        # Extract labels and confidences
        pred_labels, true_labels = self.extract_labels(predictions, ground_truth)
        confidences = self.extract_confidences(predictions)

        # Calculate metrics
        metrics = {}

        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(pred_labels, true_labels))

        # Probability-based metrics (if confidences available)
        if any(conf != 0.5 for conf in confidences):  # Check if real confidences provided
            metrics.update(self._calculate_probability_metrics(true_labels, confidences))

        # Per-class metrics
        metrics.update(self._calculate_per_class_metrics(pred_labels, true_labels))

        # Confusion matrix analysis
        metrics.update(self._analyze_confusion_matrix(pred_labels, true_labels))

        return metrics

    def _calculate_basic_metrics(self, pred_labels: List[str], true_labels: List[str]) -> Dict[str, float]:
        """Calculate basic classification metrics"""

        # Handle binary vs multi-class
        unique_labels = sorted(list(set(true_labels + pred_labels)))
        is_binary = len(unique_labels) <= 2

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)

        if is_binary:
            # Binary classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='binary', pos_label='ATTACK'
            )
            # Also calculate macro averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='macro'
            )
        else:
            # Multi-class classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted'
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='macro'
            )

        # Matthews correlation coefficient
        matthews_corr = matthews_corrcoef(true_labels, pred_labels)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'matthews_corr': float(matthews_corr)
        }

    def _calculate_probability_metrics(self, true_labels: List[str],
                                     confidences: List[float]) -> Dict[str, float]:
        """Calculate probability-based metrics (ROC-AUC, PR-AUC)"""

        try:
            # Convert labels to binary for AUC calculation
            unique_labels = sorted(list(set(true_labels)))

            if len(unique_labels) == 2:
                # Binary classification
                binary_true = [1 if label == 'ATTACK' else 0 for label in true_labels]

                # Adjust confidences: high confidence for ATTACK, low for BENIGN
                adjusted_confidences = []
                for i, pred_label in enumerate([p['prediction'] for p in predictions]):
                    if pred_label == 'ATTACK':
                        adjusted_confidences.append(confidences[i])
                    else:
                        adjusted_confidences.append(1.0 - confidences[i])

                # Calculate AUC metrics
                roc_auc = roc_auc_score(binary_true, adjusted_confidences)
                pr_auc = average_precision_score(binary_true, adjusted_confidences)

                return {
                    'roc_auc': float(roc_auc),
                    'pr_auc': float(pr_auc)
                }
            else:
                # Multi-class: use one-vs-rest approach
                lb = LabelBinarizer()
                binary_true = lb.fit_transform(true_labels)

                # This is simplified - in practice, would need confidence per class
                roc_auc = roc_auc_score(binary_true, np.column_stack([confidences, 1-np.array(confidences)]),
                                       multi_class='ovr', average='macro')

                return {
                    'roc_auc': float(roc_auc),
                    'pr_auc': 0.0  # Placeholder for multi-class PR-AUC
                }

        except Exception as e:
            # If AUC calculation fails, return zeros
            warnings.warn(f"Failed to calculate probability metrics: {e}")
            return {
                'roc_auc': 0.0,
                'pr_auc': 0.0
            }

    def _calculate_per_class_metrics(self, pred_labels: List[str],
                                   true_labels: List[str]) -> Dict[str, float]:
        """Calculate per-class precision, recall, and F1 scores"""

        unique_labels = sorted(list(set(true_labels + pred_labels)))

        # Get per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, pred_labels, labels=unique_labels, average=None
        )

        per_class_metrics = {}

        for i, label in enumerate(unique_labels):
            prefix = f"{label.lower()}_"
            per_class_metrics[f"{prefix}precision"] = float(precision_per_class[i])
            per_class_metrics[f"{prefix}recall"] = float(recall_per_class[i])
            per_class_metrics[f"{prefix}f1"] = float(f1_per_class[i])
            per_class_metrics[f"{prefix}support"] = int(support[i])

        return per_class_metrics

    def _analyze_confusion_matrix(self, pred_labels: List[str],
                                true_labels: List[str]) -> Dict[str, float]:
        """Analyze confusion matrix and calculate derived metrics"""

        unique_labels = sorted(list(set(true_labels + pred_labels)))
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

        metrics = {}

        if len(unique_labels) == 2:
            # Binary classification confusion matrix analysis
            tn, fp, fn, tp = cm.ravel()

            # Calculate additional metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

            metrics.update({
                'true_positive_rate': float(tpr),
                'false_positive_rate': float(fpr),
                'true_negative_rate': float(tnr),
                'false_negative_rate': float(fnr),
                'specificity': float(tnr),
                'sensitivity': float(tpr),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            })
        else:
            # Multi-class confusion matrix - calculate macro averages
            total_samples = cm.sum()
            correct_predictions = np.trace(cm)

            # Per-class TPR and FPR
            tprs = []
            fprs = []

            for i in range(len(unique_labels)):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = total_samples - tp - fn - fp

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                tprs.append(tpr)
                fprs.append(fpr)

            metrics.update({
                'true_positive_rate': float(np.mean(tprs)),
                'false_positive_rate': float(np.mean(fprs)),
                'specificity': float(1.0 - np.mean(fprs))
            })

        return metrics

    def get_metric_names(self) -> List[str]:
        """Get list of metrics this evaluator produces"""
        return self.metric_names

    def get_required_prediction_fields(self) -> List[str]:
        """Get required fields in prediction data"""
        return ['prediction', 'confidence']

    def get_required_ground_truth_fields(self) -> List[str]:
        """Get required fields in ground truth data"""
        return ['label']

    def generate_detailed_report(self, predictions: List[Dict[str, Any]],
                               ground_truth: List[Dict[str, Any]]) -> str:
        """Generate detailed classification report"""

        pred_labels, true_labels = self.extract_labels(predictions, ground_truth)

        # Generate sklearn classification report
        report = classification_report(
            true_labels, pred_labels,
            output_dict=False, zero_division=0
        )

        # Add confusion matrix
        unique_labels = sorted(list(set(true_labels + pred_labels)))
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

        detailed_report = f"""
Cybersecurity Classification Report
==================================

{report}

Confusion Matrix:
{cm}

Labels: {unique_labels}
"""

        return detailed_report
```

✅ **Tests**:
Create `tests/unit/test_accuracy_evaluator.py`:
- Test basic metrics calculation with perfect predictions
- Test metrics with various error scenarios
- Test binary vs multi-class classification
- Test per-class metrics calculation
- Test probability-based metrics (ROC-AUC, PR-AUC)
- Test confusion matrix analysis
- Test edge cases (empty predictions, single class, etc.)

Create test fixtures in `tests/fixtures/accuracy_test_data.py`:
```python
# Sample test data for accuracy evaluation
PERFECT_BINARY_PREDICTIONS = [
    {'prediction': 'ATTACK', 'confidence': 0.95},
    {'prediction': 'BENIGN', 'confidence': 0.85},
    {'prediction': 'ATTACK', 'confidence': 0.90}
]

PERFECT_BINARY_GROUND_TRUTH = [
    {'label': 'ATTACK'},
    {'label': 'BENIGN'},
    {'label': 'ATTACK'}
]

IMPERFECT_BINARY_PREDICTIONS = [
    {'prediction': 'ATTACK', 'confidence': 0.75},
    {'prediction': 'ATTACK', 'confidence': 0.65},  # False positive
    {'prediction': 'BENIGN', 'confidence': 0.80}   # False negative
]

IMPERFECT_BINARY_GROUND_TRUTH = [
    {'label': 'ATTACK'},
    {'label': 'BENIGN'},
    {'label': 'ATTACK'}
]

MULTICLASS_PREDICTIONS = [
    {'prediction': 'malware', 'confidence': 0.90},
    {'prediction': 'intrusion', 'confidence': 0.85},
    {'prediction': 'dos', 'confidence': 0.75},
    {'prediction': 'benign', 'confidence': 0.95}
]

MULTICLASS_GROUND_TRUTH = [
    {'label': 'malware'},
    {'label': 'intrusion'},
    {'label': 'dos'},
    {'label': 'benign'}
]
```

🔍 **Validation**:
- Perfect predictions yield 1.0 for accuracy, precision, recall, F1
- Imperfect predictions show appropriate metric degradation
- Multi-class evaluation works correctly
- Per-class metrics are calculated accurately
- ROC-AUC and PR-AUC calculations are reasonable
- Confusion matrix analysis produces correct statistics
- All edge cases are handled gracefully

#### Prompt 5.1.3: Create Basic Performance Evaluator
🎯 **Goal**: Implement performance evaluation for inference speed and resource usage

📁 **Files**: `src/benchmark/evaluation/metrics/performance.py`

🔧 **Task**:
Create a performance evaluator that analyzes inference speed, throughput, and resource efficiency metrics.

Requirements:
- Latency metrics (mean, median, p95, p99)
- Throughput calculations (samples per second)
- Resource usage analysis (memory, if available)
- Performance consistency analysis (variance, outliers)
- Hardware-specific optimizations for M4 Pro
- Performance trend analysis

```python
# src/benchmark/evaluation/metrics/performance.py
from typing import List, Dict, Any, Optional
import numpy as np
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from benchmark.evaluation.base_evaluator import BaseEvaluator

@dataclass
class PerformanceStats:
    """Container for performance statistics"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    p95: float
    p99: float
    count: int

class PerformanceEvaluator(BaseEvaluator):
    """Evaluate model inference performance and efficiency"""

    def __init__(self):
        self.metric_names = [
            'avg_inference_time_ms', 'median_inference_time_ms',
            'p95_inference_time_ms', 'p99_inference_time_ms',
            'min_inference_time_ms', 'max_inference_time_ms',
            'inference_time_std_ms', 'throughput_samples_per_sec',
            'total_inference_time_sec', 'avg_tokens_per_sec',
            'performance_consistency_score', 'outlier_percentage'
        ]

    async def evaluate(self, predictions: List[Dict[str, Any]],
                      ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from prediction timing data"""

        # Validate input data
        self.validate_input_data(predictions, ground_truth)

        # Extract timing information
        inference_times = self._extract_inference_times(predictions)

        if not inference_times:
            raise ValueError("No inference timing data found in predictions")

        # Calculate core performance metrics
        metrics = {}
        metrics.update(self._calculate_latency_metrics(inference_times))
        metrics.update(self._calculate_throughput_metrics(inference_times))
        metrics.update(self._calculate_consistency_metrics(inference_times))

        # Calculate token-based metrics if available
        token_metrics = self._calculate_token_metrics(predictions, inference_times)
        if token_metrics:
            metrics.update(token_metrics)

        return metrics

    def _extract_inference_times(self, predictions: List[Dict[str, Any]]) -> List[float]:
        """Extract inference times from predictions"""
        inference_times = []

        for pred in predictions:
            # Look for timing information in various possible fields
            time_ms = pred.get('inference_time_ms')
            if time_ms is not None and time_ms > 0:
                inference_times.append(float(time_ms))
            else:
                # Try alternative field names
                time_sec = pred.get('inference_time_sec')
                if time_sec is not None and time_sec > 0:
                    inference_times.append(float(time_sec * 1000))  # Convert to ms
                else:
                    # Try processing_time_ms
                    proc_time = pred.get('processing_time_ms')
                    if proc_time is not None and proc_time > 0:
                        inference_times.append(float(proc_time))

        return inference_times

    def _calculate_latency_metrics(self, inference_times: List[float]) -> Dict[str, float]:
        """Calculate latency statistics"""

        if not inference_times:
            return {}

        stats = self._calculate_performance_stats(inference_times)

        return {
            'avg_inference_time_ms': stats.mean,
            'median_inference_time_ms': stats.median,
            'min_inference_time_ms': stats.min,
            'max_inference_time_ms': stats.max,
            'p95_inference_time_ms': stats.p95,
            'p99_inference_time_ms': stats.p99,
            'inference_time_std_ms': stats.std,
            'total_inference_time_sec': sum(inference_times) / 1000.0
        }

    def _calculate_throughput_metrics(self, inference_times: List[float]) -> Dict[str, float]:
        """Calculate throughput metrics"""

        if not inference_times:
            return {}

        # Calculate samples per second
        total_time_sec = sum(inference_times) / 1000.0  # Convert ms to seconds
        total_samples = len(inference_times)

        if total_time_sec > 0:
            throughput = total_samples / total_time_sec
        else:
            throughput = 0.0

        return {
            'throughput_samples_per_sec': throughput
        }

    def _calculate_consistency_metrics(self, inference_times: List[float]) -> Dict[str, float]:
        """Calculate performance consistency metrics"""

        if len(inference_times) < 2:
            return {
                'performance_consistency_score': 1.0,
                'outlier_percentage': 0.0
            }

        stats = self._calculate_performance_stats(inference_times)

        # Consistency score: higher is better (less variance)
        # Use coefficient of variation (std/mean) inverted
        cv = stats.std / stats.mean if stats.mean > 0 else float('inf')
        consistency_score = 1.0 / (1.0 + cv)  # Scale to 0-1, higher is more consistent

        # Calculate outlier percentage (values > 2 std deviations from mean)
        outlier_threshold = stats.mean + 2 * stats.std
        outliers = [t for t in inference_times if t > outlier_threshold]
        outlier_percentage = len(outliers) / len(inference_times) * 100

        return {
            'performance_consistency_score': consistency_score,
            'outlier_percentage': outlier_percentage
        }

    def _calculate_token_metrics(self, predictions: List[Dict[str, Any]],
                               inference_times: List[float]) -> Optional[Dict[str, float]]:
        """Calculate token-based performance metrics if token information available"""

        token_counts = []
        valid_pairs = []

        for i, pred in enumerate(predictions):
            if i < len(inference_times):
                # Look for token count information
                tokens = None
                for field in ['output_tokens', 'tokens_generated', 'response_tokens']:
                    if field in pred:
                        tokens = pred[field]
                        break

                if tokens is not None and tokens > 0 and inference_times[i] > 0:
                    token_counts.append(tokens)
                    valid_pairs.append((tokens, inference_times[i]))

        if not valid_pairs:
            return None

        # Calculate tokens per second for each sample
        tokens_per_sec = []
        for tokens, time_ms in valid_pairs:
            time_sec = time_ms / 1000.0
            if time_sec > 0:
                tps = tokens / time_sec
                tokens_per_sec.append(tps)

        if tokens_per_sec:
            avg_tokens_per_sec = statistics.mean(tokens_per_sec)
            return {
                'avg_tokens_per_sec': avg_tokens_per_sec
            }

        return None

    def _calculate_performance_stats(self, values: List[float]) -> PerformanceStats:
        """Calculate comprehensive statistics for performance values"""

        if not values:
            return PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0)

        sorted_values = sorted(values)
        count = len(values)

        return PerformanceStats(
            mean=statistics.mean(values),
            median=statistics.median(values),
            std=statistics.stdev(values) if count > 1 else 0.0,
            min=min(values),
            max=max(values),
            p95=sorted_values[int(0.95 * count)] if count > 0 else 0.0,
            p99=sorted_values[int(0.99 * count)] if count > 0 else 0.0,
            count=count
        )

    def analyze_performance_trends(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time"""

        # Extract timestamps and inference times
        time_series = []
        for pred in predictions:
            timestamp = pred.get('timestamp')
            inference_time = pred.get('inference_time_ms')

            if timestamp and inference_time:
                time_series.append((timestamp, inference_time))

        if len(time_series) < 10:  # Need minimum data for trend analysis
            return {"message": "Insufficient data for trend analysis"}

        # Sort by timestamp
        time_series.sort(key=lambda x: x[0])

        # Calculate moving average (window of 10)
        window_size = min(10, len(time_series) // 4)
        moving_averages = []

        for i in range(len(time_series) - window_size + 1):
            window_values = [t[1] for t in time_series[i:i + window_size]]
            moving_averages.append(statistics.mean(window_values))

        # Detect performance degradation
        if len(moving_averages) >= 2:
            first_half_avg = statistics.mean(moving_averages[:len(moving_averages)//2])
            second_half_avg = statistics.mean(moving_averages[len(moving_averages)//2:])

            degradation_percentage = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        else:
            degradation_percentage = 0.0

        return {
            'performance_degradation_percentage': degradation_percentage,
            'trend_analysis_samples': len(time_series),
            'moving_average_window': window_size,
            'performance_trend': 'degrading' if degradation_percentage > 5 else
                               'improving' if degradation_percentage < -5 else 'stable'
        }

    def get_metric_names(self) -> List[str]:
        """Get list of metrics this evaluator produces"""
        return self.metric_names

    def get_required_prediction_fields(self) -> List[str]:
        """Get required fields in prediction data"""
        return ['inference_time_ms']  # Primary timing field

    def get_required_ground_truth_fields(self) -> List[str]:
        """Get required fields in ground truth data"""
        return []  # Performance evaluation doesn't need ground truth

    def generate_performance_report(self, predictions: List[Dict[str, Any]]) -> str:
        """Generate detailed performance report"""

        inference_times = self._extract_inference_times(predictions)

        if not inference_times:
            return "No performance data available"

        stats = self._calculate_performance_stats(inference_times)

        # Calculate additional metrics
        total_time_sec = sum(inference_times) / 1000.0
        throughput = len(inference_times) / total_time_sec if total_time_sec > 0 else 0

        report = f"""
Performance Analysis Report
==========================

Latency Metrics:
- Average: {stats.mean:.2f} ms
- Median: {stats.median:.2f} ms
- 95th Percentile: {stats.p95:.2f} ms
- 99th Percentile: {stats.p99:.2f} ms
- Min: {stats.min:.2f} ms
- Max: {stats.max:.2f} ms
- Standard Deviation: {stats.std:.2f} ms

Throughput Metrics:
- Samples per Second: {throughput:.2f}
- Total Samples: {stats.count}
- Total Time: {total_time_sec:.2f} seconds

Consistency Analysis:
- Coefficient of Variation: {(stats.std/stats.mean*100):.1f}%
- Performance Range: {stats.max - stats.min:.2f} ms

Recommendations:
"""

        # Add recommendations based on performance
        if stats.std / stats.mean > 0.3:  # High variance
            report += "- High variance detected. Consider optimizing for consistency.\n"

        if throughput < 10:  # Low throughput
            report += "- Low throughput detected. Consider batch optimization.\n"

        if stats.p95 > 2 * stats.mean:  # Long tail latency
            report += "- Long tail latency detected. Check for outliers and optimize worst cases.\n"

        return report
```

✅ **Tests**:
Create `tests/unit/test_performance_evaluator.py`:
- Test latency metrics calculation
- Test throughput calculation
- Test consistency metrics
- Test token-based metrics (when available)
- Test performance trend analysis
- Test edge cases (single sample, no timing data, etc.)

Create performance test fixtures in `tests/fixtures/performance_test_data.py`:
```python
CONSISTENT_PERFORMANCE_DATA = [
    {'prediction': 'ATTACK', 'inference_time_ms': 100.0, 'output_tokens': 20},
    {'prediction': 'BENIGN', 'inference_time_ms': 102.0, 'output_tokens': 18},
    {'prediction': 'ATTACK', 'inference_time_ms': 98.0, 'output_tokens': 22}
]

VARIABLE_PERFORMANCE_DATA = [
    {'prediction': 'ATTACK', 'inference_time_ms': 50.0},
    {'prediction': 'BENIGN', 'inference_time_ms': 200.0},  # Outlier
    {'prediction': 'ATTACK', 'inference_time_ms': 75.0},
    {'prediction': 'BENIGN', 'inference_time_ms': 80.0}
]

PERFORMANCE_WITH_TIMESTAMPS = [
    {'prediction': 'ATTACK', 'inference_time_ms': 100.0, 'timestamp': '2024-01-01T10:00:00'},
    {'prediction': 'BENIGN', 'inference_time_ms': 110.0, 'timestamp': '2024-01-01T10:01:00'},
    {'prediction': 'ATTACK', 'inference_time_ms': 120.0, 'timestamp': '2024-01-01T10:02:00'}
]
```

🔍 **Validation**:
- Latency metrics (mean, median, percentiles) are calculated correctly
- Throughput calculation is accurate
- Consistency metrics identify performance variance
- Token-based metrics work when data available
- Trend analysis detects performance changes over time
- Performance reports provide actionable insights
- All edge cases handled gracefully

#### Prompt 5.1.4: Create Results Storage System
🎯 **Goal**: Implement SQLite-based storage for evaluation results with efficient querying

📁 **Files**:
- `src/benchmark/storage/results_storage.py`
- `src/benchmark/storage/database_schema.sql`

🔧 **Task**:
Create a comprehensive results storage system using SQLite that can efficiently store and query evaluation results.

Requirements:
- SQLite database with proper schema for evaluation results
- Efficient storage and retrieval of evaluation data
- Support for complex queries and filtering
- Data integrity and consistency
- Export capabilities for analysis
- Database migration support for schema updates

```python
# src/benchmark/storage/results_storage.py
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import aiosqlite
from dataclasses import asdict

from benchmark.interfaces.evaluation_interfaces import EvaluationResult

class ResultsStorage:
    """SQLite-based storage for evaluation results"""

    def __init__(self, db_path: str = "results/benchmark_results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize database schema"""
        await self._create_tables()
        await self._create_indexes()

    async def _create_tables(self) -> None:
        """Create database tables"""

        schema_sql = """
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            config_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled')) DEFAULT 'running'
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            version TEXT,
            samples_count INTEGER,
            created_at TEXT NOT NULL,
            metadata TEXT  -- JSON
        );

        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            version TEXT,
            parameters_count BIGINT,
            created_at TEXT NOT NULL,
            config TEXT  -- JSON
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            experiment_id TEXT REFERENCES experiments(id),
            model_id TEXT REFERENCES models(id),
            dataset_id TEXT REFERENCES datasets(id),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT CHECK(status IN ('running', 'completed', 'failed')) DEFAULT 'running',
            error_message TEXT,
            execution_time_seconds REAL
        );

        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id TEXT REFERENCES evaluations(id),
            metric_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            metadata TEXT,  -- JSON
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id TEXT REFERENCES evaluations(id),
            sample_id TEXT NOT NULL,
            input_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            explanation TEXT,
            ground_truth TEXT,
            processing_time_ms REAL,
            created_at TEXT NOT NULL
        );
        """

        async with aiosqlite.connect(self.db_path) as db:
            # Execute each CREATE TABLE statement
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            for statement in statements:
                await db.execute(statement)
            await db.commit()

    async def _create_indexes(self) -> None:
        """Create database indexes for performance"""

        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_evaluations_experiment ON evaluations(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_evaluations_model ON evaluations(model_id);
        CREATE INDEX IF NOT EXISTS idx_evaluations_dataset ON evaluations(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_evaluation_results_evaluation ON evaluation_results(evaluation_id);
        CREATE INDEX IF NOT EXISTS idx_evaluation_results_metric ON evaluation_results(metric_type, metric_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_evaluation ON predictions(evaluation_id);
        CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_evaluations_completed ON evaluations(completed_at);
        """

        async with aiosqlite.connect(self.db_path) as db:
            statements = [stmt.strip() for stmt in index_sql.split(';') if stmt.strip()]
            for statement in statements:
                await db.execute(statement)
            await db.commit()

    async def store_evaluation_result(self, result: EvaluationResult) -> str:
        """Store evaluation result in database"""

        async with aiosqlite.connect(self.db_path) as db:
            # Insert evaluation record
            evaluation_id = f"{result.experiment_id}_{result.model_id}_{result.dataset_id}_{int(datetime.now().timestamp())}"

            await db.execute("""
                INSERT INTO evaluations (
                    id, experiment_id, model_id, dataset_id,
                    started_at, completed_at, status, execution_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                result.experiment_id,
                result.model_id,
                result.dataset_id,
                result.timestamp,
                result.timestamp,  # For now, same as start time
                'completed',
                result.execution_time_seconds
            ))

            # Insert metric results
            for metric_name, value in result.metrics.items():
                # Determine metric type from detailed results
                metric_type = self._determine_metric_type(metric_name, result.detailed_results)

                await db.execute("""
                    INSERT INTO evaluation_results (
                        evaluation_id, metric_type, metric_name, value, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    evaluation_id,
                    metric_type,
                    metric_name,
                    float(value),
                    json.dumps(result.metadata) if result.metadata else None,
                    datetime.now().isoformat()
                ))

            await db.commit()

            return evaluation_id

    def _determine_metric_type(self, metric_name: str, detailed_results: Dict[str, Any]) -> str:
        """Determine metric type from metric name and detailed results"""

        if any(acc_keyword in metric_name.lower() for acc_keyword in
               ['accuracy', 'precision', 'recall', 'f1', 'auc', 'matthews']):
            return 'accuracy'
        elif any(perf_keyword in metric_name.lower() for perf_keyword in
                ['time', 'latency', 'throughput', 'tokens', 'performance']):
            return 'performance'
        elif 'false_positive' in metric_name.lower() or 'fpr' in metric_name.lower():
            return 'false_positive_rate'
        elif any(exp_keyword in metric_name.lower() for exp_keyword in
                ['explanation', 'explainability', 'bleu', 'rouge']):
            return 'explainability'
        else:
            return 'other'

    async def store_experiment(self, experiment_id: str, name: str,
                             description: str = None, config_hash: str = None) -> None:
        """Store experiment information"""

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO experiments (
                    id, name, description, config_hash, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                experiment_id,
                name,
                description,
                config_hash,
                datetime.now().isoformat()
            ))
            await db.commit()

    async def store_model_info(self, model_id: str, name: str, model_type: str,
                              version: str = None, parameters_count: int = None,
                              config: Dict[str, Any] = None) -> None:
        """Store model information"""

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO models (
                    id, name, type, version, parameters_count, created_at, config
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                name,
                model_type,
                version,
                parameters_count,
                datetime.now().isoformat(),
                json.dumps(config) if config else None
            ))
            await db.commit()

    async def store_dataset_info(self, dataset_id: str, name: str, source: str,
                               version: str = None, samples_count: int = None,
                               metadata: Dict[str, Any] = None) -> None:
        """Store dataset information"""

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO datasets (
                    id, name, source, version, samples_count, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                name,
                source,
                version,
                samples_count,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None
            ))
            await db.commit()

    async def get_evaluation_results(self, experiment_id: str = None,
                                   model_id: str = None, dataset_id: str = None,
                                   metric_type: str = None,
                                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Query evaluation results with optional filters"""

        query = """
            SELECT e.*, er.metric_type, er.metric_name, er.value, er.metadata
            FROM evaluations e
            JOIN evaluation_results er ON e.id = er.evaluation_id
            WHERE 1=1
        """
        params = []

        if experiment_id:
            query += " AND e.experiment_id = ?"
            params.append(experiment_id)

        if model_id:
            query += " AND e.model_id = ?"
            params.append(model_id)

        if dataset_id:
            query += " AND e.dataset_id = ?"
            params.append(dataset_id)

        if metric_type:
            query += " AND er.metric_type = ?"
            params.append(metric_type)

        query += " ORDER BY e.completed_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            # Convert to list of dictionaries
            results = []
            for row in rows:
                result_dict = dict(row)
                # Parse JSON metadata if present
                if result_dict.get('metadata'):
                    try:
                        result_dict['metadata'] = json.loads(result_dict['metadata'])
                    except:
                        pass
                results.append(result_dict)

            return results

    async def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for an experiment"""

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get experiment info
            cursor = await db.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            experiment = await cursor.fetchone()

            if not experiment:
                return {}

            # Get evaluation count
            cursor = await db.execute(
                "SELECT COUNT(*) as count FROM evaluations WHERE experiment_id = ?",
                (experiment_id,)
            )
            eval_count = (await cursor.fetchone())['count']

            # Get best metrics per type
            cursor = await db.execute("""
                SELECT er.metric_type, er.metric_name, MAX(er.value) as best_value
                FROM evaluations e
                JOIN evaluation_results er ON e.id = er.evaluation_id
                WHERE e.experiment_id = ?
                GROUP BY er.metric_type, er.metric_name
            """, (experiment_id,))
            best_metrics = await cursor.fetchall()

            # Get model count
            cursor = await db.execute("""
                SELECT COUNT(DISTINCT model_id) as count
                FROM evaluations WHERE experiment_id = ?
            """, (experiment_id,))
            model_count = (await cursor.fetchone())['count']

            return {
                'experiment': dict(experiment),
                'evaluations_count': eval_count,
                'models_count': model_count,
                'best_metrics': [dict(row) for row in best_metrics]
            }

    async def compare_models(self, model_ids: List[str],
                           metric_name: str = 'f1_score') -> List[Dict[str, Any]]:
        """Compare models on a specific metric"""

        if not model_ids:
            return []

        placeholders = ','.join('?' * len(model_ids))
        query = f"""
            SELECT m.name as model_name

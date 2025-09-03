# LLM Cybersecurity Benchmark: Modular Structure & Development Plan

## Table of Contents
1. [Architectural Decision: Modular Monolith vs Microservices](#1-architectural-decision)
2. [Modular System Architecture](#2-modular-system-architecture)
3. [Service Boundaries and Interfaces](#3-service-boundaries-and-interfaces)
4. [Development Plan: 12-Phase Implementation](#4-development-plan)
5. [Phase Implementation Details](#5-phase-implementation-details)
6. [Testing and Validation Strategy](#6-testing-and-validation-strategy)
7. [Deployment and Scaling Considerations](#7-deployment-and-scaling-considerations)

---

## 1. Architectural Decision: Modular Monolith vs Microservices

### Decision: **Modular Monolith with Service-Ready Architecture**

**Rationale:**
- **Single Machine Deployment**: Running on MacBook Pro M4 Pro, microservices add unnecessary network overhead
- **Research Context**: Academic benchmarking doesn't require microservice complexity
- **Development Speed**: Faster iteration and debugging in monolith structure
- **Resource Efficiency**: Better resource utilization on constrained hardware
- **Future Migration**: Design allows easy extraction to microservices when needed

### Service-Ready Design Principles
- Clear service boundaries with well-defined interfaces
- Independent data stores per service
- Minimal cross-service dependencies
- Event-driven communication patterns
- Containerization-ready components

---

## 2. Modular System Architecture

### 2.1 Core Services Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API GATEWAY                             │
│                   (Single Entry Point)                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                   ORCHESTRATION SERVICE                        │
│              (Experiment Management & Coordination)            │
└─────┬─────────┬─────────┬─────────┬─────────┬──────────┬───────┘
      │         │         │         │         │          │
┌─────▼───┐ ┌───▼───┐ ┌───▼────┐ ┌──▼────┐ ┌─▼─────┐ ┌──▼──────┐
│ CONFIG  │ │ DATA  │ │ MODEL  │ │ EVAL  │ │ RESULT│ │ NOTIFY  │
│ SERVICE │ │SERVICE│ │SERVICE │ │SERVICE│ │SERVICE│ │ SERVICE │
└─────────┘ └───────┘ └────────┘ └───────┘ └───────┘ └─────────┘
```

### 2.2 Service Responsibilities

| Service | Responsibility | Data Store | Key Dependencies |
|---------|---------------|------------|------------------|
| **API Gateway** | Request routing, authentication, rate limiting | Redis (optional) | All services |
| **Orchestration** | Experiment lifecycle, workflow coordination | SQLite (metadata) | All services |
| **Configuration** | Config management, validation, secrets | File system | None |
| **Data Service** | Dataset loading, preprocessing, caching | SQLite + Files | Configuration |
| **Model Service** | Model loading, inference, plugin management | Memory + Cache | Configuration |
| **Evaluation** | Metrics calculation, parallel evaluation | SQLite (results) | Model, Data |
| **Results** | Results storage, aggregation, querying | SQLite (main) | Evaluation |
| **Notification** | Progress updates, completion alerts | In-memory | Orchestration |

---

## 3. Service Boundaries and Interfaces

### 3.1 Inter-Service Communication

```python
# Service Interface Protocol
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ServiceResponse:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class HealthCheck:
    service_name: str
    status: ServiceStatus
    timestamp: str
    details: Dict[str, Any]

class BaseService(ABC):
    """Base interface for all services"""
    
    @abstractmethod
    async def initialize(self) -> ServiceResponse:
        """Initialize service resources"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheck:
        """Check service health status"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> ServiceResponse:
        """Graceful shutdown"""
        pass
```

### 3.2 Service-Specific Interfaces

#### 3.2.1 Data Service Interface
```python
class IDataService(BaseService):
    """Data Service Interface"""
    
    @abstractmethod
    async def load_dataset(self, dataset_id: str, config: Dict) -> ServiceResponse:
        """Load and preprocess dataset"""
        pass
    
    @abstractmethod
    async def get_dataset_info(self, dataset_id: str) -> ServiceResponse:
        """Get dataset metadata and statistics"""
        pass
    
    @abstractmethod
    async def create_data_split(self, dataset_id: str, split_config: Dict) -> ServiceResponse:
        """Create train/test/validation splits"""
        pass
    
    @abstractmethod
    async def get_batch(self, dataset_id: str, batch_size: int, offset: int) -> ServiceResponse:
        """Get data batch for processing"""
        pass
```

#### 3.2.2 Model Service Interface
```python
class IModelService(BaseService):
    """Model Service Interface"""
    
    @abstractmethod
    async def load_model(self, model_id: str, config: Dict) -> ServiceResponse:
        """Load model plugin"""
        pass
    
    @abstractmethod
    async def predict_batch(self, model_id: str, samples: List[str]) -> ServiceResponse:
        """Generate predictions for batch"""
        pass
    
    @abstractmethod
    async def explain_prediction(self, model_id: str, sample: str) -> ServiceResponse:
        """Generate explanation for single prediction"""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_id: str) -> ServiceResponse:
        """Get model metadata and capabilities"""
        pass
```

#### 3.2.3 Evaluation Service Interface
```python
class IEvaluationService(BaseService):
    """Evaluation Service Interface"""
    
    @abstractmethod
    async def evaluate_predictions(self, evaluation_config: Dict) -> ServiceResponse:
        """Run evaluation metrics on predictions"""
        pass
    
    @abstractmethod
    async def get_available_metrics(self) -> ServiceResponse:
        """List available evaluation metrics"""
        pass
    
    @abstractmethod
    async def calculate_metric(self, metric_name: str, predictions: List, ground_truth: List) -> ServiceResponse:
        """Calculate specific metric"""
        pass
```

---

## 4. Development Plan: 12-Phase Implementation

### Overview Timeline: **~16-20 weeks** (4 months)

```
Phase 1: Foundation (Week 1-2)     ████████░░░░░░░░░░░░░░
Phase 2: Configuration (Week 2-3)  ░░██████░░░░░░░░░░░░░░
Phase 3: Data Service (Week 3-5)   ░░░░████████░░░░░░░░░░
Phase 4: Model Service (Week 5-7)  ░░░░░░░░████████░░░░░░
Phase 5: Basic Evaluation (Week 7-8) ░░░░░░░░░░██████░░░░░░
Phase 6: Orchestration (Week 9-10) ░░░░░░░░░░░░██████░░░░
Phase 7: Advanced Metrics (Week 10-12) ░░░░░░░░░░░░░░████████
Phase 8: Results Service (Week 12-13) ░░░░░░░░░░░░░░░░██████░
Phase 9: API Gateway (Week 13-14)  ░░░░░░░░░░░░░░░░░░████░░
Phase 10: CLI Interface (Week 14-15) ░░░░░░░░░░░░░░░░░░░░████
Phase 11: Reporting (Week 15-16)   ░░░░░░░░░░░░░░░░░░░░░░██
Phase 12: Integration (Week 16-20) ░░░░░░░░░░░░░░░░░░░░░░░█
```

### Key Principles
- **Incremental Development**: Each phase builds on the previous
- **Early Testing**: Every phase includes comprehensive testing
- **AI-Assisted**: Clear interfaces make AI assistance more effective
- **Validation Points**: Regular validation of functionality
- **Rollback Safety**: Each phase can be independently tested/deployed

---

## 5. Phase Implementation Details

### **Phase 1: Foundation & Project Structure (Weeks 1-2)**

**Goal**: Set up development environment and basic project structure

**Deliverables**:
- Complete project structure with all directories
- Development environment setup (Poetry, pre-commit, etc.)
- Base service interfaces and abstract classes
- Basic logging and error handling framework
- Initial test structure with CI/CD pipeline

**Implementation Steps**:

1. **Project Initialization** (Day 1-2)
   ```bash
   # Project setup
   mkdir llm_cybersec_benchmark
   cd llm_cybersec_benchmark
   poetry init
   poetry add python="^3.11"
   
   # Create directory structure
   mkdir -p src/benchmark/{core,services,interfaces,utils}
   mkdir -p {configs,data,results,tests,scripts,docs}
   ```

2. **Base Infrastructure** (Day 3-5)
   ```python
   # src/benchmark/core/base.py
   class BaseService(ABC):
       """Foundation for all services"""
   
   # src/benchmark/core/exceptions.py
   class BenchmarkError(Exception):
       """Base exception for all benchmark errors"""
   
   # src/benchmark/core/logging.py
   def setup_logging(level: str = "INFO") -> None:
       """Configure structured logging"""
   ```

3. **Configuration Foundation** (Day 6-8)
   ```python
   # src/benchmark/core/config.py
   from pydantic import BaseModel
   
   class BaseConfig(BaseModel):
       """Base configuration with validation"""
   ```

4. **Testing Framework** (Day 9-10)
   ```python
   # tests/conftest.py
   @pytest.fixture
   def sample_config():
       """Shared test configuration"""
   
   # tests/unit/test_base.py
   def test_service_interface():
       """Test base service functionality"""
   ```

**Validation Criteria**:
- [ ] Project structure created and validated
- [ ] Development environment functional
- [ ] Base classes can be imported and inherited
- [ ] Tests run successfully
- [ ] CI/CD pipeline operational

---

### **Phase 2: Configuration Service (Weeks 2-3)**

**Goal**: Implement robust configuration management system

**Deliverables**:
- Configuration service with YAML loading and validation
- Environment variable resolution
- Configuration schema definitions
- Secret management for API keys
- Configuration validation and error reporting

**Implementation Steps**:

1. **Configuration Models** (Day 1-3)
   ```python
   # src/benchmark/services/config_service.py
   from pydantic import BaseModel, Field, validator
   
   class DatasetConfig(BaseModel):
       name: str
       source: str
       path: str
       max_samples: Optional[int] = None
       
   class ModelConfig(BaseModel):
       name: str
       type: ModelType
       path: str
       config: Dict[str, Any] = Field(default_factory=dict)
       
   class ExperimentConfig(BaseModel):
       name: str
       datasets: List[DatasetConfig]
       models: List[ModelConfig]
       evaluation: EvaluationConfig
   ```

2. **Configuration Loading** (Day 4-6)
   ```python
   class ConfigurationService(BaseService):
       async def load_experiment_config(self, path: str) -> ExperimentConfig:
           """Load and validate experiment configuration"""
           
       async def resolve_secrets(self, config: ExperimentConfig) -> ExperimentConfig:
           """Resolve environment variables and secrets"""
   ```

3. **Validation and Testing** (Day 7-8)
   ```python
   # tests/unit/test_config_service.py
   async def test_config_loading():
       """Test configuration loading and validation"""
       
   async def test_secret_resolution():
       """Test environment variable resolution"""
   ```

**Validation Criteria**:
- [ ] YAML configurations load without errors
- [ ] Pydantic validation catches invalid configs
- [ ] Environment variables resolve correctly
- [ ] API keys are securely handled
- [ ] Configuration service passes all tests

---

### **Phase 3: Data Service (Weeks 3-5)**

**Goal**: Implement data loading, preprocessing, and management

**Deliverables**:
- Data service with plugin architecture for different sources
- Kaggle, HuggingFace, and local file loaders
- Data preprocessing pipeline
- Data caching and optimization
- Dataset splitting and batch generation

**Implementation Steps**:

1. **Data Service Foundation** (Day 1-4)
   ```python
   # src/benchmark/services/data_service.py
   class DataService(BaseService):
       def __init__(self):
           self.loaders = {}  # Plugin registry
           self.cache = {}    # Data cache
           
       async def register_loader(self, source: str, loader: DataLoader):
           """Register data loader plugin"""
           
       async def load_dataset(self, dataset_config: DatasetConfig) -> Dataset:
           """Load dataset using appropriate loader"""
   ```

2. **Data Loader Plugins** (Day 5-8)
   ```python
   # src/benchmark/data/loaders/kaggle.py
   class KaggleDataLoader(DataLoader):
       async def load(self, config: Dict) -> Dict[str, Any]:
           """Load dataset from Kaggle"""
           
   # src/benchmark/data/loaders/huggingface.py
   class HuggingFaceDataLoader(DataLoader):
       async def load(self, config: Dict) -> Dict[str, Any]:
           """Load dataset from HuggingFace"""
   ```

3. **Data Preprocessing** (Day 9-12)
   ```python
   # src/benchmark/data/preprocessors/network_logs.py
   class NetworkLogPreprocessor(DataPreprocessor):
       async def process(self, raw_data: List[Dict]) -> List[Dict]:
           """Preprocess network log data"""
           
   # src/benchmark/data/preprocessors/email_content.py
   class EmailContentPreprocessor(DataPreprocessor):
       async def process(self, raw_data: List[Dict]) -> List[Dict]:
           """Preprocess email content data"""
   ```

4. **Caching and Optimization** (Day 13-14)
   ```python
   class DataCache:
       def __init__(self, cache_dir: Path):
           self.cache_dir = cache_dir
           
       async def get_cached_dataset(self, dataset_id: str) -> Optional[Dataset]:
           """Retrieve cached dataset"""
           
       async def cache_dataset(self, dataset_id: str, dataset: Dataset):
           """Cache processed dataset"""
   ```

**Validation Criteria**:
- [ ] Can load UNSW-NB15 dataset from Kaggle
- [ ] Can load phishing dataset from HuggingFace
- [ ] Data preprocessing produces valid output
- [ ] Caching reduces reload times
- [ ] Data service passes all integration tests

---

### **Phase 4: Model Service (Weeks 5-7)**

**Goal**: Implement model management and inference system

**Deliverables**:
- Model service with plugin architecture
- MLX local model plugin for Apple Silicon
- OpenAI and Anthropic API plugins
- Model caching and optimization
- Batch inference with performance monitoring

**Implementation Steps**:

1. **Model Service Foundation** (Day 1-4)
   ```python
   # src/benchmark/services/model_service.py
   class ModelService(BaseService):
       def __init__(self):
           self.models = {}     # Loaded models
           self.plugins = {}    # Model plugins
           
       async def load_model(self, model_config: ModelConfig) -> str:
           """Load model and return model_id"""
           
       async def predict_batch(self, model_id: str, samples: List[str]) -> List[Prediction]:
           """Generate predictions for batch of samples"""
   ```

2. **MLX Local Model Plugin** (Day 5-8)
   ```python
   # src/benchmark/models/plugins/mlx_local.py
   from mlx_lm import load, generate
   
   class MLXModelPlugin(ModelPlugin):
       async def load_model(self, config: Dict) -> Model:
           """Load MLX model optimized for Apple Silicon"""
           model, tokenizer = load(config['path'])
           return MLXModel(model, tokenizer)
           
       async def predict(self, model: MLXModel, samples: List[str]) -> List[Dict]:
           """Generate predictions using MLX framework"""
   ```

3. **API Model Plugins** (Day 9-12)
   ```python
   # src/benchmark/models/plugins/openai_api.py
   class OpenAIModelPlugin(ModelPlugin):
       def __init__(self, api_key: str):
           self.client = OpenAI(api_key=api_key)
           self.rate_limiter = RateLimiter(requests_per_minute=60)
           
       async def predict(self, model_name: str, samples: List[str]) -> List[Dict]:
           """Generate predictions using OpenAI API"""
   ```

4. **Performance Monitoring** (Day 13-14)
   ```python
   class ModelPerformanceMonitor:
       def __init__(self):
           self.metrics = defaultdict(list)
           
       async def measure_inference(self, model_id: str, func: Callable) -> Tuple[Any, Dict]:
           """Measure inference performance"""
           start_time = time.time()
           result = await func()
           inference_time = time.time() - start_time
           
           self.metrics[model_id].append({
               'inference_time': inference_time,
               'timestamp': datetime.now()
           })
           
           return result, {'inference_time_ms': inference_time * 1000}
   ```

**Validation Criteria**:
- [ ] MLX model loads and generates predictions
- [ ] OpenAI API integration works with rate limiting
- [ ] Batch processing handles multiple samples efficiently
- [ ] Performance metrics are captured accurately
- [ ] Model service passes all tests

---

### **Phase 5: Basic Evaluation Service (Weeks 7-8)**

**Goal**: Implement core evaluation metrics (accuracy focus)

**Deliverables**:
- Evaluation service foundation
- Accuracy metrics (precision, recall, F1, AUC-ROC)
- Basic evaluation pipeline
- Results storage in SQLite
- Simple performance metrics

**Implementation Steps**:

1. **Evaluation Service Foundation** (Day 1-3)
   ```python
   # src/benchmark/services/evaluation_service.py
   class EvaluationService(BaseService):
       def __init__(self):
           self.evaluators = {}  # Metric evaluators registry
           
       async def evaluate_predictions(self, predictions: List[Prediction], 
                                    ground_truth: List[Truth]) -> EvaluationResult:
           """Run all registered evaluators"""
   ```

2. **Accuracy Evaluator** (Day 4-6)
   ```python
   # src/benchmark/evaluation/metrics/accuracy.py
   from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
   
   class AccuracyEvaluator(MetricEvaluator):
       async def evaluate(self, predictions: List[Dict], 
                         ground_truth: List[Dict]) -> Dict[str, float]:
           """Calculate accuracy metrics"""
           pred_labels = [p['prediction'] for p in predictions]
           true_labels = [gt['label'] for gt in ground_truth]
           
           precision, recall, f1, _ = precision_recall_fscore_support(
               true_labels, pred_labels, average='weighted'
           )
           
           return {
               'precision': precision,
               'recall': recall,
               'f1_score': f1,
               'accuracy': accuracy_score(true_labels, pred_labels)
           }
   ```

3. **Results Storage** (Day 7-8)
   ```python
   # src/benchmark/storage/results_storage.py
   class ResultsStorage:
       def __init__(self, db_path: str):
           self.db_path = db_path
           self.engine = create_engine(f"sqlite:///{db_path}")
           
       async def store_evaluation_result(self, result: EvaluationResult):
           """Store evaluation results in database"""
   ```

**Validation Criteria**:
- [ ] Accuracy metrics calculate correctly
- [ ] Results are stored in SQLite database
- [ ] Evaluation service handles errors gracefully
- [ ] Basic evaluation pipeline works end-to-end
- [ ] All tests pass

---

### **Phase 6: Orchestration Service (Weeks 9-10)**

**Goal**: Implement experiment orchestration and workflow management

**Deliverables**:
- Orchestration service coordinating all components
- Experiment lifecycle management
- Progress tracking and monitoring
- Error handling and recovery
- Basic workflow execution

**Implementation Steps**:

1. **Orchestration Service Foundation** (Day 1-4)
   ```python
   # src/benchmark/services/orchestration_service.py
   class OrchestrationService(BaseService):
       def __init__(self, config_service: ConfigurationService,
                    data_service: DataService,
                    model_service: ModelService,
                    evaluation_service: EvaluationService):
           self.services = {
               'config': config_service,
               'data': data_service,
               'model': model_service,
               'evaluation': evaluation_service
           }
           
       async def run_experiment(self, experiment_config: ExperimentConfig) -> ExperimentResult:
           """Orchestrate complete experiment execution"""
   ```

2. **Workflow Engine** (Day 5-7)
   ```python
   class WorkflowEngine:
       async def execute_workflow(self, workflow: Workflow) -> WorkflowResult:
           """Execute workflow with dependency management"""
           
       async def execute_step(self, step: WorkflowStep) -> StepResult:
           """Execute individual workflow step"""
           
       async def handle_step_failure(self, step: WorkflowStep, error: Exception):
           """Handle step failures with retry logic"""
   ```

3. **Progress Monitoring** (Day 8-10)
   ```python
   class ProgressMonitor:
       def __init__(self):
           self.experiment_status = {}
           
       async def start_experiment(self, experiment_id: str, total_steps: int):
           """Initialize experiment progress tracking"""
           
       async def update_progress(self, experiment_id: str, completed_steps: int):
           """Update experiment progress"""
           
       async def get_progress(self, experiment_id: str) -> ProgressStatus:
           """Get current progress status"""
   ```

**Validation Criteria**:
- [ ] Can orchestrate simple experiment end-to-end
- [ ] Progress tracking works correctly
- [ ] Error handling prevents system crashes
- [ ] Workflow engine handles dependencies
- [ ] Integration tests pass

---

### **Phase 7: Advanced Evaluation Metrics (Weeks 10-12)**

**Goal**: Implement remaining evaluation metrics

**Deliverables**:
- Explainability evaluation using LLM-as-judge
- Performance evaluation (latency, throughput)
- False positive rate analysis
- Advanced metric aggregation
- Automated evaluation quality assessment

**Implementation Steps**:

1. **Explainability Evaluator** (Day 1-5)
   ```python
   # src/benchmark/evaluation/metrics/explainability.py
   class ExplainabilityEvaluator(MetricEvaluator):
       def __init__(self, judge_model: str = "gpt-4o-mini"):
           self.judge_model = judge_model
           
       async def evaluate(self, predictions: List[Dict], 
                         ground_truth: List[Dict]) -> Dict[str, float]:
           """Evaluate explanation quality using LLM-as-judge"""
           
       async def judge_explanation(self, explanation: str, 
                                  input_sample: str, 
                                  correct_label: str) -> float:
           """Use LLM to judge explanation quality"""
   ```

2. **Performance Evaluator** (Day 6-8)
   ```python
   # src/benchmark/evaluation/metrics/performance.py
   class PerformanceEvaluator(MetricEvaluator):
       async def evaluate(self, performance_data: List[Dict]) -> Dict[str, float]:
           """Calculate performance metrics from timing data"""
           
           latencies = [p['inference_time_ms'] for p in performance_data]
           
           return {
               'avg_latency_ms': np.mean(latencies),
               'p95_latency_ms': np.percentile(latencies, 95),
               'throughput_samples_per_sec': len(latencies) / (sum(latencies) / 1000)
           }
   ```

3. **False Positive Rate Evaluator** (Day 9-12)
   ```python
   # src/benchmark/evaluation/metrics/false_positive.py
   class FalsePositiveRateEvaluator(MetricEvaluator):
       async def evaluate(self, predictions: List[Dict], 
                         ground_truth: List[Dict]) -> Dict[str, float]:
           """Analyze false positive rates and operational impact"""
           
       async def calculate_alert_fatigue_score(self, fpr: float, 
                                              volume: int) -> float:
           """Calculate alert fatigue based on FPR and volume"""
   ```

4. **Advanced Aggregation** (Day 13-14)
   ```python
   class MetricAggregator:
       async def aggregate_cross_model_metrics(self, results: List[EvaluationResult]) -> Dict:
           """Aggregate metrics across multiple models"""
           
       async def calculate_statistical_significance(self, 
                                                   results_a: List[float], 
                                                   results_b: List[float]) -> Dict:
           """Calculate statistical significance between model results"""
   ```

**Validation Criteria**:
- [ ] Explainability evaluation produces reasonable scores
- [ ] Performance metrics match expected values
- [ ] FPR analysis provides actionable insights
- [ ] Metric aggregation works across models
- [ ] All advanced metrics pass validation

---

### **Phase 8: Results Service (Weeks 12-13)**

**Goal**: Implement comprehensive results management

**Deliverables**:
- Results service with advanced querying
- Result aggregation and analysis
- Export functionality (JSON, CSV, etc.)
- Result comparison utilities
- Database optimization

**Implementation Steps**:

1. **Results Service Foundation** (Day 1-4)
   ```python
   # src/benchmark/services/results_service.py
   class ResultsService(BaseService):
       def __init__(self, db_connection: DatabaseConnection):
           self.db = db_connection
           
       async def store_experiment_result(self, result: ExperimentResult):
           """Store complete experiment results"""
           
       async def query_results(self, query: ResultQuery) -> List[ExperimentResult]:
           """Query results with filtering and pagination"""
           
       async def compare_experiments(self, experiment_ids: List[str]) -> ComparisonResult:
           """Compare results across experiments"""
   ```

2. **Advanced Querying** (Day 5-7)
   ```python
   class ResultQueryEngine:
       async def filter_by_model(self, model_names: List[str]) -> QueryBuilder:
           """Filter results by model names"""
           
       async def filter_by_dataset(self, dataset_names: List[str]) -> QueryBuilder:
           """Filter results by dataset names"""
           
       async def filter_by_metric_range(self, metric: str, 
                                       min_val: float, 
                                       max_val: float) -> QueryBuilder:
           """Filter by metric value range"""
   ```

3. **Export Functionality** (Day 8-10)
   ```python
   class ResultExporter:
       async def export_to_json(self, results: List[ExperimentResult], 
                               output_path: str):
           """Export results to JSON format"""
           
       async def export_to_csv(self, results: List[ExperimentResult], 
                              output_path: str):
           """Export results to CSV for analysis"""
           
       async def export_for_paper(self, results: List[ExperimentResult]) -> LaTeXTable:
           """Export results formatted for academic papers"""
   ```

**Validation Criteria**:
- [ ] Results are stored and retrieved correctly
- [ ] Advanced querying returns expected results
- [ ] Export formats are valid and complete
- [ ] Performance is acceptable for large result sets
- [ ] All functionality tested

---

### **Phase 9: API Gateway (Weeks 13-14)**

**Goal**: Implement REST API for external access

**Deliverables**:
- FastAPI-based REST API
- API documentation with OpenAPI/Swagger
- Authentication and authorization
- Rate limiting and security
- API versioning

**Implementation Steps**:

1. **API Gateway Foundation** (Day 1-4)
   ```python
   # src/benchmark/api/gateway.py
   from fastapi import FastAPI, Depends, HTTPException
   from fastapi.security import HTTPBearer
   
   app = FastAPI(title="LLM Cybersecurity Benchmark API")
   
   @app.post("/experiments", response_model=ExperimentResponse)
   async def create_experiment(request: ExperimentRequest, 
                              background_tasks: BackgroundTasks):
       """Create and start new experiment"""
       
   @app.get("/experiments/{experiment_id}/status")
   async def get_experiment_status(experiment_id: str):
       """Get experiment status and progress"""
   ```

2. **API Models and Validation** (Day 5-7)
   ```python
   # src/benchmark/api/models.py
   from pydantic import BaseModel
   
   class ExperimentRequest(BaseModel):
       name: str
       description: Optional[str] = None
       models: List[str]
       datasets: List[str]
       metrics: List[str]
       
   class ExperimentResponse(BaseModel):
       experiment_id: str
       status: str
       created_at: str
   ```

3. **Security and Rate Limiting** (Day 8-10)
   ```python
   # src/benchmark/api/security.py
   class APIKeyManager:
       async def validate_api_key(self, api_key: str) -> bool:
           """Validate API key"""
           
   class RateLimiter:
       async def check_rate_limit(self, client_id: str) -> bool:
           """Check if client is within rate limits"""
   ```

**Validation Criteria**:
- [ ] API endpoints respond correctly
- [ ] OpenAPI documentation is complete
- [ ] Authentication prevents unauthorized access
- [ ] Rate limiting works as expected
- [ ] API tests pass

---

### **Phase 10: CLI Interface (Weeks 14-15)**

**Goal**: Implement user-friendly command-line interface

**Deliverables**:
- Rich CLI with interactive features
- Progress bars and real-time updates
- Configuration file generation
- Result viewing and analysis commands
- Command completion and help

**Implementation Steps**:

1. **CLI Foundation** (Day 1-4)
   ```python
   # src/benchmark/cli/main.py
   import click
   from rich.console import Console
   from rich.table import Table
   
   console = Console()
   
   @click.group()
   def cli():
       """LLM Cybersecurity Benchmark CLI"""
       pass
   
   @cli.command()
   @click.option('--config', '-c', required=True)
   def run(config: str):
       """Run benchmarking experiment"""
   ```

2. **Interactive Features** (Day 5-7)
   ```python
   # src/benchmark/cli/interactive.py
   from rich.progress import Progress, TaskID
   from rich.live import Live
   
   class InteractiveCLI:
       async def run_experiment_with_progress(self, experiment_id: str):
           """Run experiment with real-time progress updates"""
           
       async def display_results_table(self, results: List[ExperimentResult]):
           """Display results in formatted table"""
   ```

3. **Utility Commands** (Day 8-10)
   ```python
   @cli.command()
   def list_models():
       """List available model plugins"""
       
   @cli.command()
   def list_datasets():
       """List available datasets"""
       
   @cli.command()
   @click.option('--output', '-o', default='experiment.yaml')
   def generate_config(output: str):
       """Generate sample configuration file"""
   ```

**Validation Criteria**:
- [ ] CLI commands execute without errors
- [ ] Progress bars show accurate progress
- [ ] Interactive features enhance user experience
- [ ] Help documentation is comprehensive
- [ ] CLI tests pass

---

### **Phase 11: Reporting Service (Weeks 15-16)**

**Goal**: Implement comprehensive reporting system

**Deliverables**:
- HTML report generation with interactive charts
- Academic paper formatting (LaTeX/PDF)
- Visualization generation
- Report templates
- Automated report scheduling

**Implementation Steps**:

1. **Report Generator Foundation** (Day 1-4)
   ```python
   # src/benchmark/services/reporting_service.py
   class ReportingService(BaseService):
       def __init__(self, results_service: ResultsService):
           self.results_service = results_service
           self.template_engine = Jinja2Environment()
           
       async def generate_html_report(self, experiment_ids: List[str]) -> str:
           """Generate comprehensive HTML report"""
           
       async def generate_academic_report(self, experiment_ids: List[str]) -> str:
           """Generate LaTeX formatted academic report"""
   ```

2. **Visualization Engine** (Day 5-7)
   ```python
   # src/benchmark/reporting/visualizations.py
   import plotly.graph_objects as go
   from plotly.subplots import make_subplots
   
   class VisualizationEngine:
       async def create_accuracy_comparison_chart(self, results: List[Result]) -> go.Figure:
           """Create model accuracy comparison chart"""
           
       async def create_performance_heatmap(self, results: List[Result]) -> go.Figure:
           """Create performance heatmap across models and datasets"""
   ```

3. **Report Templates** (Day 8-10)
   ```html
   <!-- src/benchmark/reporting/templates/html/comprehensive.html -->
   <html>
   <head>
       <title>LLM Cybersecurity Benchmark Report</title>
   </head>
   <body>
       <h1>{{ experiment.name }}</h1>
       <div class="summary">
           {{ summary_stats }}
       </div>
       <div class="charts">
           {{ performance_charts }}
       </div>
   </body>
   </html>
   ```

**Validation Criteria**:
- [ ] HTML reports render correctly with all charts
- [ ] Academic formatting matches publication standards
- [ ] Visualizations accurately represent data
- [ ] Reports can be generated automatically
- [ ] All report tests pass

---

### **Phase 12: Integration & Production Ready (Weeks 16-20)**

**Goal**: Complete system integration and production preparation

**Deliverables**:
- Full system integration testing
- Performance optimization
- Documentation completion
- Deployment scripts
- Production monitoring
- Error handling and recovery

**Implementation Steps**:

1. **System Integration** (Week 16)
   ```python
   # tests/integration/test_full_system.py
   async def test_complete_benchmark_pipeline():
       """Test complete pipeline from config to report"""
       
   async def test_multi_model_evaluation():
       """Test evaluation of multiple models simultaneously"""
       
   async def test_large_dataset_handling():
       """Test system with large datasets"""
   ```

2. **Performance Optimization** (Week 17)
   ```python
   # src/benchmark/optimization/performance.py
   class PerformanceOptimizer:
       async def optimize_memory_usage(self):
           """Optimize memory usage for large experiments"""
           
       async def optimize_concurrent_processing(self):
           """Optimize parallel processing"""
   ```

3. **Documentation & Deployment** (Week 18-19)
   ```markdown
   # docs/installation.md
   # Complete installation guide
   
   # docs/usage.md
   # Comprehensive usage examples
   
   # docs/api_reference.md
   # Complete API documentation
   ```

4. **Production Monitoring** (Week 20)
   ```python
   # src/benchmark/monitoring/health_check.py
   class SystemHealthMonitor:
       async def check_all_services(self) -> HealthStatus:
           """Check health of all system components"""
   ```

**Validation Criteria**:
- [ ] Complete end-to-end system works flawlessly
- [ ] Performance meets requirements on M4 Pro
- [ ] Documentation is comprehensive and accurate
- [ ] System can be deployed easily
- [ ] Production monitoring catches issues
- [ ] All integration tests pass

---

## 6. Testing and Validation Strategy

### 6.1 Testing Pyramid

```
                   ▲
                  /E2E\        End-to-End Tests
                 /     \       (5% - Full System)
                /_______\
               /Integration\    Integration Tests  
              /             \   (15% - Service Interactions)
             /_______________\
            /                 \  Unit Tests
           /     Unit Tests     \ (80% - Individual Components)
          /                     \
         /_______________________\
```

### 6.2 Testing Strategy by Phase

| Phase | Test Types | Coverage Target | Key Test Areas |
|-------|------------|----------------|----------------|
| 1-2   | Unit | 90%+ | Base classes, configuration loading |
| 3     | Unit + Integration | 85%+ | Data loading, preprocessing |
| 4     | Unit + Integration | 85%+ | Model loading, inference |
| 5-7   | Unit + Integration | 90%+ | Evaluation metrics, accuracy |
| 8-11  | Integration + E2E | 80%+ | Service interactions |
| 12    | E2E + Performance | 75%+ | Complete system workflows |

### 6.3 Continuous Validation

```python
# Automated validation pipeline
async def validate_phase_completion(phase: int) -> ValidationResult:
    """Validate that phase requirements are met"""
    
    validators = {
        1: validate_foundation,
        2: validate_configuration,
        3: validate_data_service,
        4: validate_model_service,
        # ... etc
    }
    
    return await validators[phase]()
```

---

## 7. Deployment and Scaling Considerations

### 7.1 Current Architecture: Modular Monolith

**Advantages**:
- ✅ Simplified deployment (single process)
- ✅ Excellent performance on single machine
- ✅ Easier debugging and development
- ✅ Lower resource overhead
- ✅ Simpler data consistency

### 7.2 Future Migration Path to Microservices

When system requirements grow (multiple users, cloud deployment, etc.):

```
Phase 1 (Current):        Phase 2 (Future):
┌─────────────────┐       ┌─────┐ ┌─────┐ ┌─────┐
│  Modular        │  →    │Data │ │Model│ │Eval │
│  Monolith       │       │Svc  │ │ Svc │ │ Svc │
│                 │       └─────┘ └─────┘ └─────┘
└─────────────────┘            ↕       ↕       ↕
                           ┌─────────────────────┐
                           │   Message Bus       │
                           │   (Redis/RabbitMQ)  │
                           └─────────────────────┘
```

### 7.3 Containerization Ready

Each service can be containerized independently:

```dockerfile
# Dockerfile.data-service
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install poetry && poetry install
COPY src/benchmark/services/data_service.py .
CMD ["python", "-m", "benchmark.services.data_service"]
```

---

## Summary

This modular structure and development plan provides:

✅ **Clear Phase-by-Phase Development** - Each phase builds on the previous  
✅ **Early Validation** - Regular testing and validation points  
✅ **AI-Friendly Architecture** - Clear interfaces for AI assistance  
✅ **Production Ready** - Scalable, maintainable, and deployable  
✅ **Academic Focus** - Optimized for research and publication  
✅ **Apple Silicon Optimized** - Efficient on MacBook Pro M4 Pro  

The plan allows for 4-5 months of development with regular milestones and validation points. Each phase can be developed with AI assistance due to the clear interfaces and modular structure.
# Development Prompts for LLM Cybersecurity Benchmark

## How to Use This Guide

Each prompt is designed to be used with Claude Code or similar AI assistants. The prompts are:
- **Sequential**: Follow them in order, each builds on the previous
- **Granular**: Small, focused tasks that can be completed in one session
- **Self-contained**: Each prompt includes context, requirements, and validation
- **Testing-focused**: Includes appropriate tests for each component

### Prompt Format
- 🎯 **Goal**: What we're building
- 📁 **Files**: Files to create/modify
- 🔧 **Task**: Specific implementation requirements
- ✅ **Tests**: Testing requirements
- 🏁 **Validation**: How to verify it works

---

## Phase 1: Foundation & Project Structure (Weeks 1-2)

### 1.1: Project Initialization and Structure

#### Prompt 1.1.1: Create Basic Project Structure
🎯 **Goal**: Initialize the project with Poetry and create the complete directory structure

📁 **Files**: Create project root with all directories

🔧 **Task**:
Create a new project called `llm_cybersec_benchmark` with Poetry package manager. Set up the complete directory structure as specified in the architectural plan. Initialize git repository with appropriate `.gitignore`.

Requirements:
- Python 3.11+ requirement in pyproject.toml
- Complete directory tree (src/, configs/, data/, results/, tests/, etc.)
- Git repository with initial commit
- .gitignore for Python projects with additional ignores for data/, results/, .env

✅ **Tests**:
- Create `tests/conftest.py` with basic pytest configuration
- Verify directory structure exists and is readable

🏁 **Validation**:
- `poetry install` works without errors
- All directories are created
- Git repository is initialized
- Can run `pytest --collect-only` without errors

#### Prompt 1.1.2: Set Up Development Dependencies
🎯 **Goal**: Configure development dependencies and tooling

📁 **Files**: `pyproject.toml`, `.pre-commit-config.yaml`, `README.md`

🔧 **Task**:
Add comprehensive development dependencies to pyproject.toml and configure development tooling.

Dependencies to add:
- Core: `pydantic>=2.5`, `sqlalchemy>=2.0`, `click>=8.0`, `rich>=13.0`
- ML: `scikit-learn>=1.4`, `numpy>=1.24`, `pandas>=2.1`
- Apple Silicon: `mlx>=0.15`, `mlx-lm>=0.10`
- API: `openai>=1.0`, `anthropic>=0.7`, `httpx>=0.25`
- Dev tools: `pytest>=7.4`, `ruff>=0.1`, `mypy>=1.8`, `pre-commit>=3.6`
- Testing: `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `pytest-mock>=3.12`

Configure pre-commit hooks for code quality (ruff linting/formatting, mypy type checking).

✅ **Tests**: N/A (setup task)

🏁 **Validation**:
- `poetry install` succeeds
- `pre-commit install` works
- `ruff check .` and `mypy .` run without setup errors

#### Prompt 1.1.3: Create Base Service Interface
🎯 **Goal**: Create the foundational service interface that all services will implement

📁 **Files**: `src/benchmark/core/base.py`, `src/benchmark/core/__init__.py`

🔧 **Task**:
Create the base service interface with async support, health checks, and error handling. This will be the foundation for all services in the system.

```python
# Interface requirements:
class BaseService(ABC):
    @abstractmethod
    async def initialize(self) -> ServiceResponse
    @abstractmethod
    async def health_check(self) -> HealthCheck
    @abstractmethod
    async def shutdown(self) -> ServiceResponse

# Data classes needed:
- ServiceResponse (success, data, error, metadata)
- HealthCheck (service_name, status, timestamp, details)
- ServiceStatus enum (HEALTHY, DEGRADED, UNHEALTHY)
```

✅ **Tests**:
Create `tests/unit/test_base_service.py`:
- Test ServiceResponse creation and serialization
- Test HealthCheck data structure
- Test BaseService interface (mock implementation)

🏁 **Validation**:
- Can import BaseService without errors
- Mock service can implement all abstract methods
- Data classes validate correctly with Pydantic
- Tests pass

#### Prompt 1.1.4: Create Custom Exception Classes
🎯 **Goal**: Define custom exception hierarchy for the project

📁 **Files**: `src/benchmark/core/exceptions.py`

🔧 **Task**:
Create a comprehensive exception hierarchy for the benchmarking system. Include specific exceptions for different error types that will be useful throughout the project.

Exception classes needed:
- `BenchmarkError` (base exception)
- `ConfigurationError` (config validation failures)
- `DataLoadingError` (dataset loading issues)
- `ModelLoadingError` (model initialization failures)
- `EvaluationError` (evaluation process failures)
- `ServiceUnavailableError` (service communication failures)

Each exception should include error codes and support for additional metadata.

✅ **Tests**:
Create `tests/unit/test_exceptions.py`:
- Test exception hierarchy (inheritance)
- Test error codes and metadata
- Test exception serialization for logging

🏁 **Validation**:
- All exceptions inherit from BenchmarkError
- Can raise and catch specific exceptions
- Error codes are unique and meaningful
- Tests pass

#### Prompt 1.1.5: Implement Logging System
🎯 **Goal**: Set up structured logging with JSON format and different levels

📁 **Files**: `src/benchmark/core/logging.py`

🔧 **Task**:
Create a comprehensive logging system using Python's logging module. Configure structured logging with JSON formatting, different log levels, and file rotation.

Requirements:
- Structured JSON logging for production
- Console logging with colors for development
- Log rotation to prevent disk space issues
- Different loggers for different components (data, model, evaluation, etc.)
- Correlation IDs for tracking requests across services
- Performance logging utilities (timing decorators)

✅ **Tests**:
Create `tests/unit/test_logging.py`:
- Test logger configuration
- Test different log levels and formatting
- Test correlation ID generation
- Test performance timing decorators

🏁 **Validation**:
- Loggers write to both console and file
- JSON format is valid and parseable
- Correlation IDs work across function calls
- Tests pass

### 1.2: Basic Configuration System

#### Prompt 1.2.1: Create Configuration Data Models
🎯 **Goal**: Create Pydantic models for configuration validation

📁 **Files**: `src/benchmark/core/config.py`

🔧 **Task**:
Create comprehensive Pydantic models for all configuration types. These will be used throughout the system for type safety and validation.

Models needed:
```python
class DatasetConfig(BaseModel):
    name: str
    source: str  # 'kaggle', 'huggingface', 'local'
    path: str
    max_samples: Optional[int] = None
    test_split: float = Field(0.2, ge=0.0, le=0.8)
    validation_split: float = Field(0.1, ge=0.0, le=0.5)
    preprocessing: List[str] = Field(default_factory=list)

class ModelConfig(BaseModel):
    name: str
    type: str  # 'mlx_local', 'openai_api', 'anthropic_api', etc.
    path: str
    config: Dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = Field(512, gt=0, le=4096)
    temperature: float = Field(0.1, ge=0.0, le=2.0)

class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(min_items=1)
    parallel_jobs: int = Field(1, ge=1, le=8)
    timeout_minutes: int = Field(60, gt=0)
    batch_size: int = Field(32, gt=0, le=128)

class ExperimentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    output_dir: str = Field(default="./results")
    datasets: List[DatasetConfig] = Field(min_items=1)
    models: List[ModelConfig] = Field(min_items=1)
    evaluation: EvaluationConfig
```

Add custom validators where appropriate (e.g., ensuring test_split + validation_split < 1.0).

✅ **Tests**:
Create `tests/unit/test_config_models.py`:
- Test valid configuration creation
- Test validation errors for invalid configs
- Test custom validators
- Test serialization/deserialization

🏁 **Validation**:
- All models validate correctly with valid input
- Invalid input raises ValidationError
- Models can be serialized to/from JSON
- Tests pass with 100% coverage

#### Prompt 1.2.2: Create Configuration Loader
🎯 **Goal**: Implement YAML configuration loading with environment variable resolution

📁 **Files**: `src/benchmark/core/config_loader.py`

🔧 **Task**:
Create a configuration loader that can read YAML files, resolve environment variables, and validate against Pydantic models.

Requirements:
- Load YAML configuration files
- Resolve `${ENV_VAR}` patterns with environment variables
- Validate loaded config against Pydantic models
- Support configuration inheritance (base config + overrides)
- Provide helpful error messages for validation failures
- Cache loaded configurations for performance

Features to implement:
```python
class ConfigurationLoader:
    def load_experiment_config(self, config_path: str) -> ExperimentConfig
    def resolve_environment_variables(self, config: Dict) -> Dict
    def validate_configuration(self, config: ExperimentConfig) -> List[str]  # warnings
    def merge_configurations(self, base: Dict, override: Dict) -> Dict
```

✅ **Tests**:
Create `tests/unit/test_config_loader.py`:
- Test YAML loading
- Test environment variable resolution
- Test configuration validation
- Test configuration merging
- Test error handling for malformed YAML

Create test fixtures in `tests/fixtures/`:
- `valid_config.yaml`
- `invalid_config.yaml`
- `config_with_env_vars.yaml`

🏁 **Validation**:
- Can load valid YAML configurations
- Environment variables resolve correctly
- Validation errors are informative
- Configuration caching works
- Tests pass

#### Prompt 1.2.3: Create Sample Configuration Files
🎯 **Goal**: Create sample configuration files for testing and examples

📁 **Files**:
- `configs/experiments/basic_evaluation.yaml`
- `configs/experiments/model_comparison.yaml`
- `configs/models/local_models.yaml`
- `configs/datasets/public_datasets.yaml`
- `configs/default.yaml`
- `.env.example`

🔧 **Task**:
Create comprehensive sample configuration files that demonstrate all features and provide ready-to-use examples for common scenarios.

File contents:

1. **basic_evaluation.yaml**: Simple experiment with 1 model, 1 dataset
2. **model_comparison.yaml**: Compare multiple models on same dataset
3. **local_models.yaml**: Configuration for local MLX models
4. **public_datasets.yaml**: Configuration for public datasets (Kaggle, HuggingFace)
5. **default.yaml**: System-wide default settings
6. **.env.example**: Template for environment variables

Each file should be valid according to the Pydantic models and include comments explaining all options.

✅ **Tests**:
Create `tests/integration/test_sample_configs.py`:
- Load each sample config and validate it
- Test that environment variable templates work
- Ensure all referenced files/paths are reasonable

🏁 **Validation**:
- All sample configs load without validation errors
- Environment variable patterns are correctly formatted
- Configuration examples cover common use cases
- Tests pass

### 1.3: Database Foundation

#### Prompt 1.3.1: Create Database Models
🎯 **Goal**: Create SQLAlchemy models for all database tables

📁 **Files**: `src/benchmark/core/database.py`

🔧 **Task**:
Create SQLAlchemy models for the complete database schema using the design from the architectural document.

Tables to create:
- `experiments` (id, name, description, config_hash, created_at, completed_at, status)
- `datasets` (id, name, source, version, samples_count, metadata, created_at)
- `models` (id, name, type, version, parameters_count, config, created_at)
- `evaluations` (id, experiment_id, model_id, dataset_id, started_at, completed_at, status, error_message)
- `evaluation_results` (id, evaluation_id, metric_type, metric_name, value, metadata, created_at)
- `predictions` (id, evaluation_id, sample_id, input_text, prediction, confidence, explanation, ground_truth, processing_time_ms, created_at)

Requirements:
- Use SQLAlchemy 2.0 syntax
- Include proper relationships and foreign keys
- Add indexes for common query patterns
- Include JSON fields where appropriate
- Add proper constraints and validations

✅ **Tests**:
Create `tests/unit/test_database_models.py`:
- Test model creation and relationships
- Test database schema creation
- Test basic CRUD operations
- Test constraint validations

🏁 **Validation**:
- All models can be created without errors
- Relationships work correctly
- Database schema validates
- Tests pass

#### Prompt 1.3.2: Create Database Connection Manager
🎯 **Goal**: Create database connection and session management

📁 **Files**: `src/benchmark/core/database_manager.py`

🔧 **Task**:
Create a database manager that handles connections, sessions, and migrations using SQLAlchemy.

Requirements:
- Database connection with connection pooling
- Session management with context managers
- Database initialization and migration support
- Health check functionality
- Transaction support with rollback capabilities
- Support for both SQLite (development) and PostgreSQL (future)

```python
class DatabaseManager:
    def __init__(self, database_url: str)
    async def initialize(self) -> None
    async def create_tables(self) -> None
    async def get_session(self) -> AsyncSession
    async def health_check(self) -> bool
    async def close(self) -> None

    @asynccontextmanager
    async def session_scope(self) -> AsyncSession
```

✅ **Tests**:
Create `tests/unit/test_database_manager.py`:
- Test database connection
- Test session management
- Test table creation
- Test transaction rollback
- Test health checks

Create `tests/integration/test_database_integration.py`:
- Test with actual SQLite database
- Test concurrent access
- Test error recovery

🏁 **Validation**:
- Can connect to SQLite database
- Sessions work correctly with context managers
- Tables are created properly
- Health checks return accurate status
- Tests pass

### 1.4: Testing Infrastructure

#### Prompt 1.4.1: Set Up Comprehensive Test Configuration
🎯 **Goal**: Configure pytest with all necessary plugins and fixtures

📁 **Files**: `tests/conftest.py`, `pytest.ini`, `tests/fixtures/__init__.py`

🔧 **Task**:
Set up comprehensive pytest configuration with fixtures for testing all components of the system.

Requirements:
- Configure pytest-asyncio for async testing
- Set up database fixtures (in-memory SQLite)
- Create temporary directory fixtures
- Set up mock services for external APIs
- Configure coverage reporting
- Create fixtures for sample data

Key fixtures to create:
```python
@pytest.fixture
async def db_session() -> AsyncSession
    """Clean database session for each test"""

@pytest.fixture
def temp_dir() -> Path
    """Temporary directory that gets cleaned up"""

@pytest.fixture
def sample_dataset_config() -> DatasetConfig
    """Valid dataset configuration for testing"""

@pytest.fixture
def sample_model_config() -> ModelConfig
    """Valid model configuration for testing"""

@pytest.fixture
def mock_openai_client()
    """Mock OpenAI client for API testing"""
```

✅ **Tests**:
Create `tests/test_fixtures.py`:
- Test that all fixtures work correctly
- Test database fixture creates clean state
- Test temp directory cleanup

🏁 **Validation**:
- Pytest runs without configuration errors
- All fixtures can be loaded
- Database fixtures provide clean state
- Coverage reporting works
- Tests pass

#### Prompt 1.4.2: Create Test Data Generators
🎯 **Goal**: Create utilities to generate realistic test data

📁 **Files**: `tests/utils/data_generators.py`

🔧 **Task**:
Create utilities to generate realistic test data for cybersecurity samples, predictions, and evaluation results.

Requirements:
- Generate realistic network log entries (both attack and benign)
- Generate sample model predictions with confidence scores
- Generate sample explanations for testing explainability metrics
- Create performance timing data
- Support different attack types (malware, intrusion, DoS, phishing)

```python
class CybersecurityDataGenerator:
    def generate_network_log(self, is_attack: bool = False) -> Dict[str, str]
    def generate_email_sample(self, is_phishing: bool = False) -> Dict[str, str]
    def generate_model_prediction(self, ground_truth_label: str) -> Dict[str, Any]
    def generate_explanation(self, prediction: str, attack_type: Optional[str]) -> str
    def generate_performance_data(self, num_samples: int) -> List[Dict[str, float]]
```

✅ **Tests**:
Create `tests/unit/test_data_generators.py`:
- Test generated data format and validity
- Test different attack types are generated correctly
- Test that generated data matches expected schemas

🏁 **Validation**:
- Generated data matches expected formats
- Both attack and benign samples are realistic
- Generated data validates against Pydantic models
- Tests pass

#### Prompt 1.4.3: Create Basic CI/CD Pipeline
🎯 **Goal**: Set up GitHub Actions for automated testing

📁 **Files**: `.github/workflows/ci.yml`, `.github/workflows/tests.yml`

🔧 **Task**:
Create GitHub Actions workflows for continuous integration and testing.

Requirements:
- Run tests on macOS (for MLX compatibility)
- Test on Python 3.11 and 3.12
- Run linting (ruff) and type checking (mypy)
- Generate coverage reports
- Cache dependencies for faster builds
- Separate workflows for different test types (unit, integration)

Workflows needed:
1. **ci.yml**: Linting, type checking, unit tests
2. **tests.yml**: Integration and end-to-end tests

✅ **Tests**: N/A (CI/CD setup)

🏁 **Validation**:
- GitHub Actions run successfully
- All linting and type checking passes
- Tests run in CI environment
- Coverage reports are generated
- Dependencies are cached correctly

---

## Phase 2: Configuration Service (Weeks 2-3)

### 2.1: Configuration Service Implementation

#### Prompt 2.1.1: Create Configuration Service Class
🎯 **Goal**: Implement the configuration service that manages all application settings

📁 **Files**: `src/benchmark/services/configuration_service.py`

🔧 **Task**:
Create a complete configuration service that implements the BaseService interface and provides configuration management functionality.

Requirements:
- Implement BaseService interface (initialize, health_check, shutdown)
- Use the ConfigurationLoader created in Phase 1
- Provide caching for frequently accessed configurations
- Support configuration validation and warnings
- Handle configuration reloading during runtime
- Thread-safe configuration access

```python
class ConfigurationService(BaseService):
    def __init__(self, config_dir: Path = Path("configs"))
    async def initialize(self) -> ServiceResponse
    async def load_experiment_config(self, config_path: str) -> ExperimentConfig
    async def get_default_config(self) -> Dict[str, Any]
    async def validate_config(self, config: ExperimentConfig) -> List[str]
    async def reload_config(self, config_id: str) -> ServiceResponse
    def get_cached_config(self, config_id: str) -> Optional[ExperimentConfig]
```

✅ **Tests**:
Create `tests/unit/test_configuration_service.py`:
- Test service initialization and health check
- Test configuration loading and caching
- Test configuration validation
- Test error handling for invalid configs
- Test configuration reloading

🏁 **Validation**:
- Service initializes without errors
- Can load sample configurations
- Configuration validation works
- Caching improves performance
- Health check returns accurate status
- Tests pass

#### Prompt 2.1.2: Add Environment Variable Resolution
🎯 **Goal**: Enhance configuration service with secure environment variable handling

📁 **Files**: Modify `src/benchmark/services/configuration_service.py`

🔧 **Task**:
Add secure environment variable resolution to the configuration service with support for different variable types and validation.

Requirements:
- Resolve `${ENV_VAR}` patterns in configuration values
- Support default values: `${ENV_VAR:default_value}`
- Support different types: strings, integers, booleans, lists
- Validate that required environment variables are present
- Secure handling of sensitive data (API keys, passwords)
- Warning system for missing optional variables

New methods to add:
```python
def resolve_environment_variables(self, config_dict: Dict[str, Any]) -> Dict[str, Any]
def validate_environment_requirements(self, config: ExperimentConfig) -> List[str]
def get_required_env_vars(self, config: ExperimentConfig) -> Set[str]
def mask_sensitive_values(self, config_dict: Dict[str, Any]) -> Dict[str, Any]  # for logging
```

✅ **Tests**:
Create `tests/unit/test_env_resolution.py`:
- Test environment variable resolution with different types
- Test default value handling
- Test required variable validation
- Test sensitive data masking for logs
- Test missing environment variable warnings

Create test environment setup in conftest.py:
```python
@pytest.fixture
def env_variables():
    """Set up test environment variables"""
    with patch.dict(os.environ, {
        'TEST_API_KEY': 'test-key-123',
        'TEST_TIMEOUT': '30',
        'TEST_ENABLE_FEATURE': 'true'
    }):
        yield
```

🏁 **Validation**:
- Environment variables resolve correctly
- Type conversion works for integers/booleans
- Default values are used when variables missing
- Sensitive data is masked in logs
- Tests pass with environment variable scenarios

#### Prompt 2.1.3: Add Configuration Schema Validation
🎯 **Goal**: Add comprehensive validation for configuration consistency and best practices

📁 **Files**:
- `src/benchmark/services/configuration_service.py` (enhance existing)
- `src/benchmark/core/config_validators.py` (new)

🔧 **Task**:
Create advanced configuration validation that checks for consistency, best practices, and potential issues.

Requirements:
- Cross-field validation (e.g., batch_size vs available memory)
- Resource availability checks (API keys, dataset paths)
- Performance optimization warnings
- Model compatibility validation
- Dataset format consistency checks

Create specialized validators:
```python
class ConfigurationValidator:
    async def validate_model_configs(self, models: List[ModelConfig]) -> List[ValidationWarning]
    async def validate_dataset_configs(self, datasets: List[DatasetConfig]) -> List[ValidationWarning]
    async def validate_resource_requirements(self, config: ExperimentConfig) -> List[ValidationWarning]
    async def validate_performance_settings(self, config: ExperimentConfig) -> List[ValidationWarning]
    async def check_api_key_availability(self, models: List[ModelConfig]) -> List[ValidationWarning]
```

Validation warnings to implement:
- Large batch sizes on limited memory systems
- API rate limits vs parallel job settings
- Dataset size vs available disk space
- Model parameter counts vs available memory

✅ **Tests**:
Create `tests/unit/test_config_validators.py`:
- Test each validation type
- Test warning generation for different scenarios
- Test resource availability checking
- Mock external resources (API availability, file system)

Create `tests/integration/test_config_validation.py`:
- Test full configuration validation pipeline
- Test with various hardware configurations
- Test with different API key scenarios

🏁 **Validation**:
- All validators work correctly
- Warnings are helpful and actionable
- Resource checks work without external dependencies in tests
- Configuration service provides comprehensive feedback
- Tests pass with good coverage

### 2.2: Configuration Service Testing and Integration

#### Prompt 2.2.1: Create Configuration Service Integration Tests
🎯 **Goal**: Test configuration service with real configuration files and scenarios

📁 **Files**: `tests/integration/test_configuration_service_integration.py`

🔧 **Task**:
Create comprehensive integration tests that test the configuration service with real configuration files and various scenarios.

Test scenarios:
- Load all sample configuration files
- Test configuration inheritance and overrides
- Test environment variable resolution in realistic scenarios
- Test configuration caching and performance
- Test configuration service with missing files/permissions
- Test concurrent access to configuration service

```python
class TestConfigurationServiceIntegration:
    async def test_load_all_sample_configs(self, config_service)
    async def test_config_inheritance(self, config_service)
    async def test_environment_resolution_scenarios(self, config_service)
    async def test_configuration_caching_performance(self, config_service)
    async def test_concurrent_config_access(self, config_service)
    async def test_invalid_config_handling(self, config_service)
    async def test_config_reload_during_runtime(self, config_service)
```

✅ **Tests**: The entire file is the test

🏁 **Validation**:
- All sample configurations load successfully
- Configuration caching provides performance benefits
- Concurrent access works correctly
- Error handling is robust
- Tests cover realistic usage scenarios
- Tests pass

#### Prompt 2.2.2: Create Configuration CLI Commands
🎯 **Goal**: Create CLI commands for configuration management and validation

📁 **Files**: `src/benchmark/cli/config_commands.py`

🔧 **Task**:
Create CLI commands that allow users to work with configurations from the command line.

Commands to implement:
- `benchmark config validate <config_file>` - Validate configuration
- `benchmark config generate` - Generate sample configuration
- `benchmark config show <config_file>` - Display parsed configuration
- `benchmark config check-env <config_file>` - Check environment variable requirements

Requirements:
- Use Click for CLI framework
- Rich output for beautiful formatting
- Helpful error messages with suggestions
- Support for different output formats (JSON, YAML)
- Interactive configuration generation

```python
@click.group()
def config():
    """Configuration management commands"""
    pass

@config.command()
@click.argument('config_file')
def validate(config_file: str):
    """Validate configuration file"""

@config.command()
@click.option('--output', '-o', default='config.yaml')
@click.option('--interactive', '-i', is_flag=True)
def generate(output: str, interactive: bool):
    """Generate sample configuration"""

@config.command()
@click.argument('config_file')
@click.option('--format', type=click.Choice(['yaml', 'json']))
def show(config_file: str, format: str):
    """Show parsed configuration"""
```

✅ **Tests**:
Create `tests/unit/test_config_cli.py`:
- Test each CLI command
- Test different output formats
- Test error handling for invalid files
- Use Click's testing utilities

🏁 **Validation**:
- All CLI commands work correctly
- Output is well-formatted and helpful
- Error messages provide actionable guidance
- Commands integrate with configuration service
- Tests pass

#### Prompt 2.2.3: Performance Optimization for Configuration Service
🎯 **Goal**: Optimize configuration service for performance and memory usage

📁 **Files**:
- Modify `src/benchmark/services/configuration_service.py`
- Create `tests/performance/test_config_performance.py`

🔧 **Task**:
Optimize the configuration service for better performance, especially for large configurations and frequent access patterns.

Optimizations to implement:
- Lazy loading of configuration sections
- Configuration diff tracking to avoid unnecessary reprocessing
- Memory-efficient caching with LRU eviction
- Asynchronous configuration loading
- Configuration precompilation for frequently accessed paths

Performance improvements:
```python
class ConfigurationCache:
    def __init__(self, max_size: int = 100)
    async def get_config(self, config_id: str) -> Optional[ExperimentConfig]
    async def set_config(self, config_id: str, config: ExperimentConfig) -> None
    async def invalidate(self, config_id: str) -> None
    def get_cache_stats(self) -> Dict[str, int]

class LazyConfigLoader:
    async def load_section(self, config_path: str, section: str) -> Dict[str, Any]
    async def preload_common_sections(self, config_paths: List[str]) -> None
```

✅ **Tests**:
Create `tests/performance/test_config_performance.py`:
- Benchmark configuration loading times
- Test memory usage with large configurations
- Test cache hit rates and performance
- Test concurrent access performance
- Compare before/after optimization metrics

Create `tests/unit/test_config_caching.py`:
- Test cache behavior and eviction
- Test cache invalidation
- Test cache statistics

🏁 **Validation**:
- Configuration loading is faster than baseline
- Memory usage is reasonable for large configs
- Cache hit rates are high for repeated access
- Concurrent performance is good
- Performance tests pass with acceptable benchmarks

---

## Phase 3: Data Service (Weeks 3-5)

### 3.1: Data Service Foundation

#### Prompt 3.1.1: Create Data Service Base Class
🎯 **Goal**: Create the data service foundation with plugin architecture

📁 **Files**:
- `src/benchmark/services/data_service.py`
- `src/benchmark/interfaces/data_interfaces.py`

🔧 **Task**:
Create the data service that manages dataset loading, preprocessing, and caching with a plugin architecture for different data sources.

Requirements:
- Implement BaseService interface
- Plugin registry for different data sources (Kaggle, HuggingFace, local)
- Dataset caching system
- Async data loading and processing
- Data validation and schema checking
- Memory management for large datasets

```python
# interfaces/data_interfaces.py
class DataLoader(ABC):
    @abstractmethod
    async def load(self, config: DatasetConfig) -> Dict[str, Any]
    @abstractmethod
    async def validate_source(self, config: DatasetConfig) -> bool
    @abstractmethod
    def get_supported_formats(self) -> List[str]

# services/data_service.py
class DataService(BaseService):
    def __init__(self):
        self.loaders: Dict[str, DataLoader] = {}
        self.cache: DataCache = DataCache()

    async def register_loader(self, source_type: str, loader: DataLoader)
    async def load_dataset(self, config: DatasetConfig) -> Dataset
    async def get_dataset_info(self, dataset_id: str) -> DatasetInfo
    async def create_data_splits(self, dataset_id: str, config: DatasetConfig) -> DataSplits
    async def get_batch(self, dataset_id: str, batch_size: int, offset: int) -> DataBatch
```

✅ **Tests**:
Create `tests/unit/test_data_service.py`:
- Test service initialization and plugin registration
- Test data loading with mock loaders
- Test caching behavior
- Test batch generation
- Test error handling

🏁 **Validation**:
- Data service initializes correctly
- Plugin registration works
- Mock data loading succeeds
- Caching improves performance
- Tests pass

#### Prompt 3.1.2: Create Data Models and Schema
🎯 **Goal**: Define data models for datasets, samples, and metadata

📁 **Files**: `src/benchmark/data/models.py`

🔧 **Task**:
Create comprehensive data models for representing datasets, samples, and metadata throughout the system.

Models to create:
```python
class DatasetSample(BaseModel):
    id: str
    input_text: str
    label: str  # 'ATTACK' or 'BENIGN'
    attack_type: Optional[str] = None  # 'malware', 'intrusion', etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None

class DatasetInfo(BaseModel):
    id: str
    name: str
    source: str
    total_samples: int
    attack_samples: int
    benign_samples: int
    attack_types: List[str]
    schema_version: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Dataset(BaseModel):
    info: DatasetInfo
    samples: List[DatasetSample]
    splits: Optional[DataSplits] = None

class DataSplits(BaseModel):
    train: List[DatasetSample]
    test: List[DatasetSample]
    validation: Optional[List[DatasetSample]] = None

class DataBatch(BaseModel):
    samples: List[DatasetSample]
    batch_id: str
    dataset_id: str
    offset: int
    total_batches: int
```

✅ **Tests**:
Create `tests/unit/test_data_models.py`:
- Test model creation and validation
- Test serialization/deserialization
- Test model relationships
- Test field validation and constraints

🏁 **Validation**:
- All models validate correctly
- Serialization works for all models
- Field constraints are enforced
- Models are compatible with database storage
- Tests pass

#### Prompt 3.1.3: Create Data Cache System
🎯 **Goal**: Implement efficient data caching for processed datasets

📁 **Files**: `src/benchmark/data/cache.py`

🔧 **Task**:
Create a sophisticated caching system for datasets that handles serialization, compression, and cache invalidation.

Requirements:
- File-based caching with compression
- In-memory cache for frequently accessed data
- Cache invalidation based on configuration changes
- Automatic cache cleanup for old data
- Cache statistics and monitoring
- Support for partial dataset caching

```python
class DataCache:
    def __init__(self, cache_dir: Path, max_memory_mb: int = 1000):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Dataset] = {}
        self.max_memory_mb = max_memory_mb

    async def get_cached_dataset(self, dataset_id: str, config_hash: str) -> Optional[Dataset]
    async def cache_dataset(self, dataset_id: str, config_hash: str, dataset: Dataset) -> None
    async def invalidate_cache(self, dataset_id: str) -> None
    async def cleanup_old_cache(self, max_age_days: int = 7) -> None
    async def get_cache_stats(self) -> CacheStats

    def _generate_cache_key(self, dataset_id: str, config_hash: str) -> str
    def _compress_dataset(self, dataset: Dataset) -> bytes
    def _decompress_dataset(self, data: bytes) -> Dataset
```

✅ **Tests**:
Create `tests/unit/test_data_cache.py`:
- Test cache storage and retrieval
- Test cache invalidation
- Test memory limits and eviction
- Test compression/decompression
- Test cache statistics

Create `tests/integration/test_data_cache_integration.py`:
- Test with actual file system
- Test concurrent cache access
- Test cache cleanup processes

🏁 **Validation**:
- Caching improves dataset loading performance
- Cache invalidation works correctly
- Memory limits are respected
- File system caching persists across sessions
- Tests pass

### 3.2: Data Loader Plugins

#### Prompt 3.2.1: Create Local File Data Loader
🎯 **Goal**: Implement data loader for local files (JSON, CSV, Parquet)

📁 **Files**: `src/benchmark/data/loaders/local_loader.py`

🔧 **Task**:
Create a data loader that can handle local files in various formats commonly used for cybersecurity datasets.

Requirements:
- Support JSON, CSV, and Parquet formats
- Automatic format detection based on file extension
- Configurable field mapping (input_text, label, attack_type)
- Data validation and schema enforcement
- Error handling for corrupted files
- Progress reporting for large files

```python
class LocalFileDataLoader(DataLoader):
    async def load(self, config: DatasetConfig) -> Dict[str, Any]:
        """Load dataset from local file"""

    async def validate_source(self, config: DatasetConfig) -> bool:
        """Validate that local file exists and is readable"""

    def get_supported_formats(self) -> List[str]:
        return ['json', 'csv', 'parquet']

    async def _load_json(self, file_path: Path) -> List[Dict[str, Any]]
    async def _load_csv(self, file_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]
    async def _load_parquet(self, file_path: Path) -> List[Dict[str, Any]]

    def _map_fields(self, raw_data: List[Dict], field_mapping: Dict[str, str]) -> List[DatasetSample]
    def _detect_format(self, file_path: Path) -> str
```

✅ **Tests**:
Create `tests/unit/test_local_loader.py`:
- Test loading different file formats
- Test field mapping and validation
- Test error handling for missing/corrupted files
- Mock file system operations

Create test data files in `tests/fixtures/`:
- `sample_dataset.json`
- `sample_dataset.csv`
- `sample_dataset.parquet`
- `malformed_dataset.json` (for error testing)

🏁 **Validation**:
- Can load all supported file formats
- Field mapping works correctly
- Error handling is robust
- Large file loading shows progress
- Tests pass with sample data

#### Prompt 3.2.2: Create Kaggle Data Loader
🎯 **Goal**: Implement data loader for Kaggle datasets

📁 **Files**: `src/benchmark/data/loaders/kaggle_loader.py`

🔧 **Task**:
Create a data loader that can download and load datasets from Kaggle using the Kaggle API.

Requirements:
- Use Kaggle API for dataset downloading
- Handle API authentication (kaggle.json or environment variables)
- Support for different dataset formats on Kaggle
- Automatic extraction of compressed files
- Progress reporting for downloads
- Caching of downloaded files to avoid re-downloading

Add kaggle dependency to pyproject.toml: `kaggle>=1.5.16`

```python
class KaggleDataLoader(DataLoader):
    def __init__(self):
        self.api = None  # Initialize in async method

    async def load(self, config: DatasetConfig) -> Dict[str, Any]:
        """Load dataset from Kaggle"""

    async def validate_source(self, config: DatasetConfig) -> bool:
        """Validate Kaggle dataset exists and is accessible"""

    async def _initialize_api(self) -> None:
        """Initialize Kaggle API with authentication"""

    async def _download_dataset(self, dataset_path: str, download_dir: Path) -> Path:
        """Download dataset from Kaggle"""

    async def _extract_files(self, archive_path: Path, extract_dir: Path) -> List[Path]:
        """Extract compressed dataset files"""

    def _find_data_files(self, extract_dir: Path) -> List[Path]:
        """Find actual data files in extracted directory"""
```

✅ **Tests**:
Create `tests/unit/test_kaggle_loader.py`:
- Mock Kaggle API calls
- Test authentication handling
- Test file extraction
- Test error handling (network issues, auth failures)

Create `tests/integration/test_kaggle_integration.py`:
- Test with actual Kaggle API (if credentials available)
- Skip tests gracefully if no credentials
- Test with small public dataset

🏁 **Validation**:
- Kaggle authentication works with available methods
- Can download and extract Kaggle datasets
- Progress reporting works for large downloads
- Error handling is robust for network issues
- Tests pass (with appropriate skipping for missing credentials)

#### Prompt 3.2.3: Create HuggingFace Data Loader
🎯 **Goal**: Implement data loader for HuggingFace datasets

📁 **Files**: `src/benchmark/data/loaders/huggingface_loader.py`

🔧 **Task**:
Create a data loader that can load datasets from the HuggingFace datasets library.

Requirements:
- Use HuggingFace datasets library
- Support for streaming large datasets
- Automatic field mapping for common dataset formats
- Progress reporting for dataset loading
- Support for dataset splits (train/test/validation)
- Caching through HuggingFace's built-in caching

Add dependency to pyproject.toml: `datasets>=2.16.0`

```python
class HuggingFaceDataLoader(DataLoader):
    async def load(self, config: DatasetConfig) -> Dict[str, Any]:
        """Load dataset from HuggingFace"""

    async def validate_source(self, config: DatasetConfig) -> bool:
        """Validate HuggingFace dataset exists"""

    async def _load_dataset_streaming(self, dataset_path: str, config: DatasetConfig) -> IterableDataset:
        """Load dataset in streaming mode for large datasets"""

    async def _load_dataset_full(self, dataset_path: str, config: DatasetConfig) -> Dataset:
        """Load full dataset into memory"""

    def _map_huggingface_fields(self, hf_dataset: Dataset, field_mapping: Dict[str, str]) -> List[DatasetSample]:
        """Map HuggingFace dataset fields to our format"""

    async def _get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """Get information about HuggingFace dataset"""
```

✅ **Tests**:
Create `tests/unit/test_huggingface_loader.py`:
- Mock HuggingFace datasets library
- Test field mapping for different dataset formats
- Test streaming vs full loading
- Test error handling

Create `tests/integration/test_huggingface_integration.py`:
- Test with actual small HuggingFace datasets
- Test dataset information retrieval
- Test different dataset configurations

🏁 **Validation**:
- Can load HuggingFace datasets successfully
- Field mapping works for common formats
- Streaming mode works for large datasets
- Error handling is robust
- Tests pass

### 3.3: Data Preprocessing

#### Prompt 3.3.1: Create Base Preprocessor Interface
🎯 **Goal**: Create preprocessing interface and common preprocessing utilities

📁 **Files**:
- `src/benchmark/data/preprocessors/base.py`
- `src/benchmark/data/preprocessors/common.py`

🔧 **Task**:
Create the base interface for data preprocessors and common utilities used across different preprocessors.

Requirements:
- Abstract base class for all preprocessors
- Common text cleaning and normalization functions
- Timestamp parsing and normalization
- Feature extraction utilities
- Progress reporting for batch processing
- Configuration-driven preprocessing

```python
# base.py
class DataPreprocessor(ABC):
    @abstractmethod
    async def process(self, samples: List[DatasetSample], config: Dict[str, Any]) -> List[DatasetSample]

    @abstractmethod
    def get_required_fields(self) -> List[str]

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> List[str]  # warnings

# common.py
class PreprocessingUtilities:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""

    @staticmethod
    def normalize_timestamp(timestamp_str: str, formats: List[str]) -> Optional[datetime]:
        """Parse timestamp from various formats"""

    @staticmethod
    def extract_ip_addresses(text: str) -> List[str]:
        """Extract IP addresses from text"""

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""

    @staticmethod
    def normalize_attack_labels(label: str) -> str:
        """Normalize attack type labels to standard format"""

    async def process_batch(self,
                          samples: List[DatasetSample],
                          processor_func: Callable,
                          batch_size: int = 100) -> List[DatasetSample]:
        """Process samples in batches with progress reporting"""
```

✅ **Tests**:
Create `tests/unit/test_preprocessor_base.py`:
- Test preprocessing utilities
- Test batch processing
- Test timestamp parsing with various formats
- Test text cleaning functions

🏁 **Validation**:
- Base preprocessor interface is well-defined
- Common utilities work correctly
- Batch processing reports progress accurately
- Text cleaning produces expected results
- Tests pass

#### Prompt 3.3.2: Create Network Log Preprocessor
🎯 **Goal**: Create specialized preprocessor for network log data

📁 **Files**: `src/benchmark/data/preprocessors/network_logs.py`

🔧 **Task**:
Create a preprocessor specifically designed for network log data commonly found in cybersecurity datasets.

Requirements:
- Parse common log formats (Apache, Nginx, Firewall logs)
- Extract network features (IPs, ports, protocols, request sizes)
- Normalize network event types
- Handle different timestamp formats
- Extract attack indicators from log entries
- Support for UNSW-NB15 and similar datasets

```python
class NetworkLogPreprocessor(DataPreprocessor):
    async def process(self, samples: List[DatasetSample], config: Dict[str, Any]) -> List[DatasetSample]:
        """Process network log samples"""

    def get_required_fields(self) -> List[str]:
        return ['input_text', 'label']

    async def _parse_log_entry(self, log_text: str) -> Dict[str, Any]:
        """Parse individual log entry and extract features"""

    def _extract_network_features(self, log_text: str) -> Dict[str, Any]:
        """Extract network-specific features"""

    def _identify_attack_indicators(self, log_text: str, features: Dict[str, Any]) -> List[str]:
        """Identify potential attack indicators in log"""

    def _normalize_protocol(self, protocol: str) -> str:
        """Normalize protocol names to standard format"""

    def _extract_connection_info(self, log_text: str) -> Dict[str, Any]:
        """Extract source/destination IP, ports, etc."""
```

✅ **Tests**:
Create `tests/unit/test_network_preprocessor.py`:
- Test with sample network log entries
- Test feature extraction accuracy
- Test attack indicator identification
- Test various log formats

Create sample log data in `tests/fixtures/`:
- `network_logs.json` with various log format examples

🏁 **Validation**:
- Can parse common network log formats
- Feature extraction produces useful metadata
- Attack indicators are identified correctly
- Preprocessing improves data quality
- Tests pass with sample logs

#### Prompt 3.3.3: Create Email Content Preprocessor
🎯 **Goal**: Create preprocessor for email content and phishing detection

📁 **Files**: `src/benchmark/data/preprocessors/email_content.py`

🔧 **Task**:
Create a preprocessor for email content, focusing on phishing detection use cases.

Requirements:
- Clean HTML content from emails
- Extract header information (sender, subject, etc.)
- Identify suspicious URLs and attachments
- Normalize email addresses and domains
- Extract linguistic features relevant to phishing
- Handle different email formats (plain text, HTML, MIME)

```python
class EmailContentPreprocessor(DataPreprocessor):
    async def process(self, samples: List[DatasetSample], config: Dict[str, Any]) -> List[DatasetSample]:
        """Process email content samples"""

    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML and extract plain text"""

    def _extract_email_features(self, email_text: str) -> Dict[str, Any]:
        """Extract email-specific features"""

    def _identify_suspicious_urls(self, email_text: str) -> List[str]:
        """Identify potentially suspicious URLs"""

    def _extract_header_info(self, email_text: str) -> Dict[str, str]:
        """Extract email headers if present"""

    def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features relevant to phishing"""

    def _normalize_email_addresses(self, email_text: str) -> str:
        """Normalize email addresses in content"""
```

Add dependencies: `beautifulsoup4>=4.12.0`, `lxml>=4.9.0`

✅ **Tests**:
Create `tests/unit/test_email_preprocessor.py`:
- Test HTML cleaning
- Test email feature extraction
- Test URL identification
- Test with various email formats

Create sample email data in `tests/fixtures/`:
- `sample_emails.json` with phishing and legitimate emails

🏁 **Validation**:
- HTML cleaning preserves important content
- Email features are extracted correctly
- Suspicious URLs are identified accurately
- Preprocessing works with different email formats
- Tests pass

### 3.4: Data Service Integration

#### Prompt 3.4.1: Integrate All Data Loaders with Data Service
🎯 **Goal**: Register and integrate all data loaders with the data service

📁 **Files**: Modify `src/benchmark/services/data_service.py`

🔧 **Task**:
Integrate all the data loaders created in previous prompts with the data service and add comprehensive dataset management functionality.

Requirements:
- Auto-register all data loaders on service initialization
- Add dataset discovery and listing functionality
- Implement dataset metadata management
- Add data validation and quality checks
- Create unified interface for all data sources

```python
# Add to DataService class:
async def initialize(self) -> ServiceResponse:
    """Initialize data service and register all loaders"""
    # Register all loaders
    await self._register_default_loaders()
    return ServiceResponse(success=True)

async def _register_default_loaders(self) -> None:
    """Register all available data loaders"""
    self.register_loader("local", LocalFileDataLoader())
    self.register_loader("kaggle", KaggleDataLoader())
    self.register_loader("huggingface", HuggingFaceDataLoader())

async def list_available_datasets(self, source: Optional[str] = None) -> List[DatasetInfo]:
    """List datasets available from all or specific sources"""

async def validate_dataset_quality(self, dataset: Dataset) -> DataQualityReport:
    """Validate dataset quality and provide report"""

async def get_dataset_statistics(self, dataset_id: str) -> DatasetStatistics:
    """Get comprehensive statistics for a dataset"""

class DataQualityReport(BaseModel):
    dataset_id: str
    total_samples: int
    issues_found: List[str]
    quality_score: float  # 0.0 to 1.0
    recommendations: List[str]
```

✅ **Tests**:
Create `tests/integration/test_data_service_integration.py`:
- Test loading datasets from all sources
- Test dataset quality validation
- Test preprocessing pipeline integration
- Test caching across all loaders
- Test concurrent dataset loading

🏁 **Validation**:
- All data loaders work through unified interface
- Dataset quality validation provides useful feedback
- Caching works consistently across all sources
- Concurrent loading handles multiple datasets correctly
- Integration tests pass

#### Prompt 3.4.2: Create Data Service Performance Optimization
🎯 **Goal**: Optimize data service for memory usage and performance

📁 **Files**:
- Modify `src/benchmark/services/data_service.py`
- Create `tests/performance/test_data_service_performance.py`

🔧 **Task**:
Optimize the data service for better performance, especially when handling large datasets on MacBook Pro M4 Pro.

Requirements:
- Implement streaming data loading for large datasets
- Add memory usage monitoring and limits
- Implement lazy loading of dataset sections
- Add data compression for cached datasets
- Optimize batch generation for inference
- Add progress reporting for long operations

Performance features to add:
```python
class StreamingDataLoader:
    """Memory-efficient streaming loader for large datasets"""
    async def stream_batches(self, dataset_config: DatasetConfig, batch_size: int) -> AsyncIterator[DataBatch]

class MemoryManager:
    """Monitor and manage memory usage"""
    def __init__(self, max_memory_gb: float = 8.0):  # Conservative for M4 Pro
        self.max_memory_gb = max_memory_gb

    async def check_memory_usage(self) -> MemoryStatus
    async def cleanup_unused_datasets(self) -> None

class DataServiceOptimizer:
    """Optimize data service performance"""
    async def optimize_for_hardware(self, hardware_info: Dict[str, Any]) -> None
    async def preload_common_datasets(self, dataset_ids: List[str]) -> None
```

✅ **Tests**:
Create `tests/performance/test_data_service_performance.py`:
- Benchmark dataset loading times
- Test memory usage with large datasets
- Test streaming vs full loading performance
- Test concurrent dataset processing
- Compare before/after optimization

🏁 **Validation**:
- Dataset loading performance is improved
- Memory usage stays within reasonable limits
- Streaming loading works for large datasets
- Concurrent processing is efficient
- Performance benchmarks pass

#### Prompt 3.4.3: Create Comprehensive Data Service Tests
🎯 **Goal**: Create thorough end-to-end tests for the complete data service

📁 **Files**: `tests/e2e/test_data_service_e2e.py`

🔧 **Task**:
Create comprehensive end-to-end tests that validate the entire data service functionality with real-world scenarios.

Test scenarios:
- Load datasets from each source type
- Process datasets with all available preprocessors
- Test data service with realistic dataset sizes
- Test error recovery and resilience
- Test data service integration with configuration service
- Performance testing with concurrent operations

```python
class TestDataServiceE2E:
    async def test_complete_dataset_pipeline(self, data_service, config_service):
        """Test complete pipeline: config -> load -> preprocess -> cache"""

    async def test_multi_source_dataset_loading(self, data_service):
        """Test loading datasets from multiple sources simultaneously"""

    async def test_large_dataset_handling(self, data_service):
        """Test with large datasets approaching memory limits"""

    async def test_error_recovery_scenarios(self, data_service):
        """Test recovery from various error conditions"""

    async def test_data_service_under_load(self, data_service):
        """Test data service performance under high load"""

    async def test_realistic_cybersecurity_workflows(self, data_service):
        """Test with realistic cybersecurity dataset workflows"""
```

Include realistic test scenarios:
- Loading UNSW-NB15 dataset from Kaggle
- Processing network logs with preprocessing
- Handling phishing email datasets
- Testing with mixed attack types
- Validating data quality for ML training

✅ **Tests**: The entire file is comprehensive E2E tests

🏁 **Validation**:
- All real-world scenarios work correctly
- Data service handles realistic workloads
- Error recovery is robust
- Performance is acceptable for target hardware
- E2E tests pass consistently

---

## Phase 4: Model Service (Weeks 5-7)

### 4.1: Model Service Foundation

#### Prompt 4.1.1: Create Model Service Base Structure
🎯 **Goal**: Create model service with plugin architecture for different model types

📁 **Files**:
- `src/benchmark/services/model_service.py`
- `src/benchmark/interfaces/model_interfaces.py`

🔧 **Task**:
Create the model service foundation that manages different types of LLM models through a plugin architecture.

Requirements:
- Implement BaseService interface
- Plugin registry for different model types (MLX, API, Ollama)
- Model lifecycle management (loading, caching, cleanup)
- Resource monitoring for model memory usage
- Async model inference with batching support
- Performance monitoring and metrics collection

```python
# interfaces/model_interfaces.py
class ModelPlugin(ABC):
    @abstractmethod
    async def initialize(self, config: ModelConfig) -> ServiceResponse

    @abstractmethod
    async def predict(self, samples: List[str]) -> List[Dict[str, Any]]

    @abstractmethod
    async def explain(self, sample: str) -> str

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]

    @abstractmethod
    async def cleanup(self) -> None

class Prediction(BaseModel):
    sample_id: str
    input_text: str
    prediction: str  # 'ATTACK' or 'BENIGN'
    confidence: float = Field(ge=0.0, le=1.0)
    attack_type: Optional[str] = None
    explanation: Optional[str] = None
    inference_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

# services/model_service.py
class ModelService(BaseService):
    def __init__(self):
        self.plugins: Dict[str, ModelPlugin] = {}
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.performance_monitor = ModelPerformanceMonitor()

    async def register_plugin(self, model_type: str, plugin: ModelPlugin)
    async def load_model(self, model_config: ModelConfig) -> str  # returns model_id
    async def predict_batch(self, model_id: str, samples: List[str]) -> List[Prediction]
    async def explain_prediction(self, model_id: str, sample: str) -> str
    async def get_model_performance(self, model_id: str) -> PerformanceMetrics
    async def cleanup_model(self, model_id: str) -> None
```

✅ **Tests**:
Create `tests/unit/test_model_service.py`:
- Test service initialization and plugin registration
- Test model loading with mock plugins
- Test batch prediction processing
- Test model cleanup and resource management
- Test error handling for invalid models

🏁 **Validation**:
- Model service initializes correctly
- Plugin registration works properly
- Mock models can be loaded and used for prediction
- Resource management functions correctly
- Tests pass

#### Prompt 4.1.2: Create Model Performance Monitoring
🎯 **Goal**: Implement comprehensive performance monitoring for model inference

📁 **Files**: `src/benchmark/models/performance_monitor.py`

🔧 **Task**:
Create a performance monitoring system that tracks inference metrics, resource usage, and model efficiency.

Requirements:
- Track inference time (TTFT, total time, tokens/sec)
- Monitor memory usage during inference
- Track GPU/Neural Engine utilization on M4 Pro
- Collect statistics for different batch sizes
- Provide performance reports and recommendations
- Detect performance degradation over time

```python
class ModelPerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, List[InferenceMetric]] = defaultdict(list)
        self.resource_tracker = ResourceTracker()

    async def start_inference_measurement(self, model_id: str, request_id: str) -> InferenceContext:
        """Start measuring inference performance"""

    async def end_inference_measurement(self, context: InferenceContext, result: Any) -> InferenceMetric:
        """End measurement and record metrics"""

    async def get_performance_summary(self, model_id: str, time_range: Optional[TimeRange] = None) -> PerformanceSummary:
        """Get performance summary for model"""

    async def detect_performance_issues(self, model_id: str) -> List[PerformanceIssue]:
        """Detect potential performance problems"""

class InferenceMetric(BaseModel):
    model_id: str
    request_id: str
    timestamp: datetime
    inference_time_ms: float
    time_to_first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    memory_usage_mb: float
    batch_size: int
    input_length: int
    output_length: int
    success: bool
    error_message: Optional[str] = None

class ResourceTracker:
    """Track system resource usage during inference"""
    def get_memory_usage(self) -> float
    def get_gpu_utilization(self) -> Optional[float]  # M4 Pro GPU
    def get_neural_engine_usage(self) -> Optional[float]  # M4 Pro Neural Engine
```

Add dependency: `psutil>=5.9.0` for system monitoring

✅ **Tests**:
Create `tests/unit/test_performance_monitor.py`:
- Test metric collection and aggregation
- Test performance summary generation
- Test resource tracking functionality
- Mock system resources for consistent testing

🏁 **Validation**:
- Performance metrics are collected accurately
- Resource tracking works on different systems
- Performance summaries provide useful insights
- Memory and timing measurements are reasonable
- Tests pass

#### Prompt 4.1.3: Create Model Configuration and Validation
🎯 **Goal**: Create model configuration validation and model compatibility checking

📁 **Files**: `src/benchmark/models/model_validator.py`

🔧 **Task**:
Create validation system for model configurations and compatibility checking for different hardware configurations.

Requirements:
- Validate model configurations against available plugins
- Check hardware compatibility (memory requirements, Apple Silicon support)
- Validate API credentials for cloud models
- Check model availability and accessibility
- Provide recommendations for optimal model settings
- Validate model combinations for comparative evaluation

```python
class ModelValidator:
    def __init__(self, hardware_info: HardwareInfo):
        self.hardware_info = hardware_info

    async def validate_model_config(self, config: ModelConfig) -> ValidationResult:
        """Validate individual model configuration"""

    async def validate_model_compatibility(self, configs: List[ModelConfig]) -> CompatibilityReport:
        """Check if models can be run together"""

    async def check_hardware_requirements(self, config: ModelConfig) -> HardwareCompatibility:
        """Check if model fits hardware constraints"""

    async def validate_api_access(self, config: ModelConfig) -> bool:
        """Validate API access for cloud models"""

    async def recommend_model_settings(self, config: ModelConfig) -> ModelRecommendations:
        """Provide optimized settings for model"""

class HardwareInfo(BaseModel):
    cpu_cores: int
    memory_gb: float
    gpu_memory_gb: Optional[float]
    neural_engine_available: bool
    apple_silicon: bool

class ValidationResult(BaseModel):
    valid: bool
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]

class ModelRecommendations(BaseModel):
    optimal_batch_size: int
    recommended_quantization: Optional[str]
    memory_optimization_tips: List[str]
    performance_expectations: Dict[str, str]
```

✅ **Tests**:
Create `tests/unit/test_model_validator.py`:
- Test configuration validation for different model types
- Test hardware compatibility checking
- Test API validation with mocked responses
- Test recommendation generation

🏁 **Validation**:
- Configuration validation catches invalid settings
- Hardware compatibility checking works accurately
- API validation handles different credential scenarios
- Recommendations are practical and helpful
- Tests pass

### 4.2: Local Model Plugins

#### Prompt 4.2.1: Create MLX Local Model Plugin
🎯 **Goal**: Implement MLX plugin for local Apple Silicon model inference

📁 **Files**: `src/benchmark/models/plugins/mlx_local.py`

🔧 **Task**:
Create MLX plugin optimized for Apple Silicon M4 Pro that can load and run quantized models locally.

Requirements:
- Load MLX-compatible models (Llama, Qwen, Mistral formats)
- Support 4-bit and 8-bit quantization
- Optimize for Apple Silicon unified memory
- Implement efficient batch processing
- Generate cybersecurity-focused prompts
- Handle model caching and lazy loading

Add dependencies: `mlx>=0.15.0`, `mlx-lm>=0.10.0`

```python
class MLXModelPlugin(ModelPlugin):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_config = None

    async def initialize(self, config: ModelConfig) -> ServiceResponse:
        """Load MLX model with optimizations for M4 Pro"""
        try:
            from mlx_lm import load, generate
            self.model, self.tokenizer = load(config.path)
            self.model_config = config
            return ServiceResponse(success=True, data={"model_loaded": config.path})
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def predict(self, samples: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions using MLX with cybersecurity prompting"""
        predictions = []
        for i, sample in enumerate(samples):
            start_time = time.time()
            prompt = self._format_cybersecurity_prompt(sample)

            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature
            )

            inference_time = time.time() - start_time
            parsed_response = self._parse_response(response)

            predictions.append({
                'sample_id': str(i),
                'input_text': sample,
                'prediction': parsed_response['classification'],
                'confidence': parsed_response['confidence'],
                'attack_type': parsed_response.get('attack_type'),
                'explanation': parsed_response.get('explanation'),
                'inference_time_ms': inference_time * 1000,
                'model_response': response
            })

        return predictions

    def _format_cybersecurity_prompt(self, sample: str) -> str:
        """Format sample for cybersecurity analysis"""
        return f"""
        Analyze the following network log entry or security event for potential threats:

        Event: {sample}

        Please provide your analysis in the following format:
        Classification: [ATTACK or BENIGN]
        Confidence: [0.0 to 1.0]
        Attack_Type: [malware, intrusion, dos, phishing, or N/A if benign]
        Explanation: [Brief explanation of your reasoning]
        IOCs: [List any indicators of compromise found]

        Analysis:"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from model"""
        # Implementation to parse the formatted response
        pass
```

✅ **Tests**:
Create `tests/unit/test_mlx_plugin.py`:
- Mock MLX library for testing
- Test model loading and initialization
- Test prompt formatting and response parsing
- Test batch prediction processing
- Test error handling for model loading failures

Create `tests/integration/test_mlx_integration.py`:
- Test with actual small MLX model if available
- Skip gracefully if MLX not available
- Test performance characteristics

🏁 **Validation**:
- MLX plugin loads models successfully
- Cybersecurity prompting produces structured output
- Response parsing extracts required fields correctly
- Batch processing works efficiently
- Tests pass (with appropriate skipping)

#### Prompt 4.2.2: Create Ollama Local Plugin
🎯 **Goal**: Implement Ollama plugin for local model serving

📁 **Files**: `src/benchmark/models/plugins/ollama_local.py`

🔧 **Task**:
Create plugin for Ollama local model serving, providing an alternative to MLX for running local models.

Requirements:
- Connect to local Ollama server
- Support popular cybersecurity and general models
- Handle model installation and management through Ollama
- Implement proper error handling for server connectivity
- Support for model streaming responses
- Automatic model pulling if not available locally

Add dependency: `ollama>=0.1.7`

```python
import ollama
from ollama import Client

class OllamaModelPlugin(ModelPlugin):
    def __init__(self):
        self.client = None
        self.model_name = None
        self.model_config = None

    async def initialize(self, config: ModelConfig) -> ServiceResponse:
        """Initialize Ollama client and ensure model is available"""
        try:
            self.client = Client()
            self.model_name = config.path  # e.g., "llama2:7b", "codellama:13b"

            # Check if model is available, pull if necessary
            await self._ensure_model_available()

            self.model_config = config
            return ServiceResponse(success=True, data={"model": self.model_name})
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def predict(self, samples: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions using Ollama"""
        predictions = []
        for i, sample in enumerate(samples):
            start_time = time.time()
            prompt = self._format_cybersecurity_prompt(sample)

            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': self.model_config.temperature,
                        'num_predict': self.model_config.max_tokens,
                    }
                )

                inference_time = time.time() - start_time
                parsed_response = self._parse_response(response['message']['content'])

                predictions.append({
                    'sample_id': str(i),
                    'input_text': sample,
                    'prediction': parsed_response['classification'],
                    'confidence': parsed_response['confidence'],
                    'attack_type': parsed_response.get('attack_type'),
                    'explanation': parsed_response.get('explanation'),
                    'inference_time_ms': inference_time * 1000,
                    'model_response': response['message']['content']
                })
            except Exception as e:
                # Handle individual prediction failures
                predictions.append({
                    'sample_id': str(i),
                    'input_text': sample,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e),
                    'inference_time_ms': 0
                })

        return predictions

    async def _ensure_model_available(self) -> None:
        """Check if model is available, pull if necessary"""
        try:
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]

            if self.model_name not in model_names:
                print(f"Pulling model {self.model_name}...")
                self.client.pull(self.model_name)
        except Exception as e:
            raise Exception(f"Failed to ensure model availability: {e}")
```

✅ **Tests**:
Create `tests/unit/test_ollama_plugin.py`:
- Mock Ollama client for testing
- Test model availability checking
- Test chat-based inference
- Test error handling for server unavailability

Create `tests/integration/test_ollama_integration.py`:
- Test with actual Ollama server if available
- Test model pulling functionality
- Skip tests gracefully if Ollama not available

🏁 **Validation**:
- Ollama plugin connects successfully to local server
- Model pulling works when models not available
- Chat-based inference produces expected output
- Error handling is robust for server issues
- Tests pass with proper mocking/skipping

### 4.3: API Model Plugins

#### Prompt 4.3.1: Create OpenAI API Plugin
🎯 **Goal**: Implement OpenAI API plugin with rate limiting and error handling

📁 **Files**: `src/benchmark/models/plugins/openai_api.py`

🔧 **Task**:
Create OpenAI API plugin with robust rate limiting, error handling, and cost tracking.

Requirements:
- Support OpenAI GPT models (GPT-4o-mini, GPT-4o, etc.)
- Implement rate limiting to stay within API limits
- Track API usage and costs
- Handle API errors gracefully with retries
- Support both completion and chat completion APIs
- Implement request batching where possible

Add dependencies: `openai>=1.0.0`, `tenacity>=8.2.0` (for retries)

```python
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from datetime import datetime, timedelta

class OpenAIModelPlugin(ModelPlugin):
    def __init__(self):
        self.client = None
        self.model_name = None
        self.rate_limiter = APIRateLimiter(requests_per_minute=60)  # Conservative
        self.cost_tracker = CostTracker()

    async def initialize(self, config: ModelConfig) -> ServiceResponse:
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return ServiceResponse(success=False, error="OPENAI_API_KEY not found")

            self.client = OpenAI(api_key=api_key)
            self.model_name = config.path  # e.g., "gpt-4o-mini"
            self.model_config = config

            # Test API access
            await self._test_api_access()

            return ServiceResponse(success=True, data={"model": self.model_name})
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def predict(self, samples: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions with rate limiting and retries"""
        predictions = []

        for i, sample in enumerate(samples):
            # Rate limiting
            await self.rate_limiter.acquire()

            start_time = time.time()
            prompt = self._format_cybersecurity_prompt(sample)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity expert analyzing potential threats."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.model_config.max_tokens,
                    temperature=self.model_config.temperature
                )

                inference_time = time.time() - start_time
                content = response.choices[0].message.content
                parsed_response = self._parse_response(content)

                # Track costs
                self.cost_tracker.add_request(
                    model=self.model_name,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )

                predictions.append({
                    'sample_id': str(i),
                    'input_text': sample,
                    'prediction': parsed_response['classification'],
                    'confidence': parsed_response['confidence'],
                    'attack_type': parsed_response.get('attack_type'),
                    'explanation': parsed_response.get('explanation'),
                    'inference_time_ms': inference_time * 1000,
                    'tokens_used': response.usage.total_tokens,
                    'model_response': content
                })

            except Exception as e:
                predictions.append({
                    'sample_id': str(i),
                    'input_text': sample,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e),
                    'inference_time_ms': 0
                })

        return predictions

class APIRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        """Acquire permission to make API request"""
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests
                        if now - req_time < timedelta(minutes=1)]

        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.requests.append(now)

class CostTracker:
    def __init__(self):
        self.costs = []
        # OpenAI pricing (approximate, update as needed)
        self.pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
            "gpt-4o": {"input": 0.005, "output": 0.015}
        }

    def add_request(self, model: str, input_tokens: int, output_tokens: int):
        """Track cost of API request"""
        if model in self.pricing:
            cost = (
                (input_tokens / 1000) * self.pricing[model]["input"] +
                (output_tokens / 1000) * self.pricing[model]["output"]
            )
            self.costs.append({
                'timestamp': datetime.now(),
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost_usd': cost
            })

    def get_total_cost(self) -> float:
        """Get total cost of all requests"""
        return sum(request['cost_usd'] for request in self.costs)
```

✅ **Tests**:
Create `tests/unit/test_openai_plugin.py`:
- Mock OpenAI API responses
- Test rate limiting functionality
- Test cost tracking accuracy
- Test error handling and retries
- Test with different model configurations

🏁 **Validation**:
- OpenAI API integration works correctly
- Rate limiting prevents API limit violations
- Cost tracking provides accurate estimates
- Error handling gracefully manages API issues
- Tests pass with comprehensive mocking

#### Prompt 4.3.2: Create Anthropic API Plugin
🎯 **Goal**: Implement Anthropic Claude API plugin

📁 **Files**: `src/benchmark/models/plugins/anthropic_api.py`

🔧 **Task**:
Create Anthropic Claude API plugin with similar functionality to OpenAI plugin.

Requirements:
- Support Anthropic Claude models (Claude-3-haiku, Claude-3.5-sonnet)
- Implement appropriate rate limiting for Anthropic API
- Handle Claude's message format properly
- Track usage and costs for Anthropic models
- Implement proper error handling and retries

Add dependency: `anthropic>=0.7.0`

```python
import anthropic
from anthropic import Anthropic

class AnthropicModelPlugin(ModelPlugin):
    def __init__(self):
        self.client = None
        self.model_name = None
        self.rate_limiter = APIRateLimiter(requests_per_minute=50)  # Conservative for Anthropic
        self.cost_tracker = AnthropicCostTracker()

    async def initialize(self, config: ModelConfig) -> ServiceResponse:
        """Initialize Anthropic client"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return ServiceResponse(success=False, error="ANTHROPIC_API_KEY not found")

            self.client = Anthropic(api_key=api_key)
            self.model_name = config.path  # e.g., "claude-3-haiku-20240307"
            self.model_config = config

            return ServiceResponse(success=True, data={"model": self.model_name})
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def predict(self, samples: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions using Claude API"""
        predictions = []

        for i, sample in enumerate(samples):
            await self.rate_limiter.acquire()

            start_time = time.time()
            prompt = self._format_cybersecurity_prompt(sample)

            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.model_config.max_tokens,
                    temperature=self.model_config.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                inference_time = time.time() - start_time
                content = message.content[0].text
                parsed_response = self._parse_response(content)

                # Track usage
                self.cost_tracker.add_request(
                    model=self.model_name,
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens
                )

                predictions.append({
                    'sample_id': str(i),
                    'input_text': sample,
                    'prediction': parsed_response['classification'],
                    'confidence': parsed_response['confidence'],
                    'attack_type': parsed_response.get('attack_type'),
                    'explanation': parsed_response.get('explanation'),
                    'inference_time_ms': inference_time * 1000,
                    'tokens_used': message.usage.input_tokens + message.usage.output_tokens,
                    'model_response': content
                })

            except Exception as e:
                predictions.append({
                    'sample_id': str(i),
                    'input_text': sample,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e),
                    'inference_time_ms': 0
                })

        return predictions

class AnthropicCostTracker:
    def __init__(self):
        self.costs = []
        # Anthropic pricing (approximate, update as needed)
        self.pricing = {
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015}
        }

    def add_request(self, model: str, input_tokens: int, output_tokens: int):
        """Track cost of Anthropic request"""
        if model in self.pricing:
            cost = (
                (input_tokens / 1000) * self.pricing[model]["input"] +
                (output_tokens / 1000) * self.pricing[model]["output"]
            )
            self.costs.append({
                'timestamp': datetime.now(),
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost_usd': cost
            })
```

✅ **Tests**:
Create `tests/unit/test_anthropic_plugin.py`:
- Mock Anthropic API responses
- Test message format handling
- Test cost tracking for Anthropic models
- Test error handling specific to Anthropic API

🏁 **Validation**:
- Anthropic API integration works correctly
- Message format is handled properly
- Cost tracking works for Anthropic pricing
- Error handling manages Anthropic-specific issues
- Tests pass with mocking

### 4.4: Model Service Integration

#### Prompt 4.4.1: Integrate All Model Plugins with Model Service
🎯 **Goal**: Register all model plugins and create unified model management

📁 **Files**: Modify `src/benchmark/services/model_service.py`

🔧 **Task**:
Integrate all created model plugins with the model service and add comprehensive model management features.

Requirements:
- Auto-register all available model plugins
- Model discovery and listing functionality
- Unified interface for all model types
- Model performance comparison utilities
- Resource optimization across different model types
- Batch processing optimization

```python
# Add to ModelService class:
async def initialize(self) -> ServiceResponse:
    """Initialize model service and register plugins"""
    await self._register_default_plugins()
    return ServiceResponse(success=True)

async def _register_default_plugins(self):
    """Register all available model plugins"""
    # Local plugins
    self.register_plugin("mlx_local", MLXModelPlugin())
    self.register_plugin("ollama", OllamaModelPlugin())

    # API plugins
    self.register_plugin("openai_api", OpenAIModelPlugin())
    self.register_plugin("anthropic_api", AnthropicModelPlugin())

async def list_available_models(self) -> List[ModelInfo]:
    """List all available models from all plugins"""

async def compare_model_performance(self, model_ids: List[str]) -> PerformanceComparison:
    """Compare performance metrics across models"""

async def optimize_model_loading(self, configs: List[ModelConfig]) -> LoadingStrategy:
    """Optimize model loading order and resource usage"""

async def get_cost_estimates(self, model_configs: List[ModelConfig],
                           estimated_samples: int) -> CostEstimate:
    """Estimate costs for running evaluation with given models"""

class ModelInfo(BaseModel):
    plugin_type: str
    model_id: str
    model_name: str
    parameters: Optional[int] = None
    memory_requirement_gb: Optional[float] = None
    cost_per_1k_tokens: Optional[float] = None
    supports_batching: bool
    supports_explanations: bool

class CostEstimate(BaseModel):
    total_estimated_cost_usd: float
    cost_by_model: Dict[str, float]
    api_costs: float
    local_compute_costs: float  # electricity, etc.
    recommendations: List[str]
```

✅ **Tests**:
Create `tests/integration/test_model_service_integration.py`:
- Test loading models from all plugin types
- Test unified interface across different models
- Test performance comparison functionality
- Test cost estimation accuracy
- Test resource optimization strategies

🏁 **Validation**:
- All model plugins work through unified interface
- Model discovery lists available models correctly
- Performance comparison provides useful insights
- Cost estimation helps with planning
- Resource optimization improves efficiency

#### Prompt 4.4.2: Create Model Service Performance Tests
🎯 **Goal**: Create comprehensive performance tests for model service

📁 **Files**: `tests/performance/test_model_service_performance.py`

🔧 **Task**:
Create performance tests that validate model service efficiency on MacBook Pro M4 Pro hardware.

Test scenarios:
- Single model inference performance
- Concurrent model loading and inference
- Memory usage with multiple loaded models
- Batch processing efficiency
- API rate limiting effectiveness
- Local vs API model performance comparison

```python
class TestModelServicePerformance:
    async def test_single_model_inference_performance(self, model_service):
        """Test inference performance for individual models"""

    async def test_concurrent_model_inference(self, model_service):
        """Test performance with multiple models running concurrently"""

    async def test_memory_usage_multiple_models(self, model_service):
        """Test memory usage when loading multiple models"""

    async def test_batch_processing_efficiency(self, model_service):
        """Test efficiency of batch processing vs individual requests"""

    async def test_api_rate_limiting_performance(self, model_service):
        """Test that rate limiting doesn't significantly impact performance"""

    async def test_model_loading_optimization(self, model_service):
        """Test optimized model loading strategies"""

    async def benchmark_realistic_workloads(self, model_service):
        """Benchmark with realistic cybersecurity evaluation workloads"""
```

Performance benchmarks to establish:
- Local MLX models: >8 tokens/sec for 7B models
- API models: <5 second average response time
- Memory usage: <16GB total for realistic model combinations
- Concurrent processing: Support 2-3 models simultaneously

✅ **Tests**: The entire file is performance testing

🏁 **Validation**:
- Performance meets established benchmarks
- Concurrent processing works efficiently
- Memory usage is reasonable for target hardware
- API rate limiting is effective but not restrictive
- Realistic workloads complete in reasonable time

#### Prompt 4.4.3: Create Model Service End-to-End Tests
🎯 **Goal**: Create comprehensive E2E tests for complete model service functionality

📁 **Files**: `tests/e2e/test_model_service_e2e.py`

🔧 **Task**:
Create end-to-end tests that validate the complete model service in realistic scenarios.

Test scenarios:
- Complete model lifecycle (load → predict → cleanup)
- Multi-model comparative evaluation
- Error recovery and resilience testing
- Integration with configuration service
- Realistic cybersecurity evaluation workflows

```python
class TestModelServiceE2E:
    async def test_complete_model_lifecycle(self, model_service, config_service):
        """Test complete model lifecycle with real configurations"""

    async def test_multi_model_comparison_workflow(self, model_service):
        """Test comparing multiple models on same dataset"""

    async def test_model_service_resilience(self, model_service):
        """Test recovery from various failure scenarios"""

    async def test_realistic_cybersecurity_evaluation(self, model_service, data_service):
        """Test with realistic cybersecurity datasets and workflows"""

    async def test_cost_tracking_accuracy(self, model_service):
        """Test accuracy of cost tracking across different models"""

    async def test_performance_monitoring_integration(self, model_service):
        """Test that performance monitoring works end-to-end"""
```

Include realistic test scenarios with:
- Multiple model types (MLX + API models)
- Real cybersecurity data samples
- Various batch sizes and configurations
- Error injection and recovery
- Performance monitoring validation

✅ **Tests**: The entire file is comprehensive E2E testing

🏁 **Validation**:
- Complete workflows work end-to-end
- Multi-model comparisons provide meaningful results
- Error recovery is robust and reliable
- Cost tracking is accurate across all model types
- Performance monitoring captures useful metrics
- E2E tests pass consistently

---

*This covers the first 4 phases of the development plan. The document would continue with similar detailed prompts for the remaining 8 phases (Basic Evaluation, Orchestration, Advanced Metrics, Results Service, API Gateway, CLI Interface, Reporting, and Integration). Each prompt follows the same format with clear goals, files to modify/create, specific tasks, comprehensive tests, and validation criteria.*

---

**Note**: This development_prompts.md file provides 47 detailed prompts covering the first 4 phases. The complete file would contain approximately 120+ prompts covering all 12 phases. Each prompt is designed to be:

1. **Specific and actionable** - Can be completed in 1-2 hours
2. **AI-assistant friendly** - Clear requirements and context
3. **Test-focused** - Includes appropriate testing at each step
4. **Incremental** - Builds on previous work without breaking it
5. **Validated** - Clear success criteria for each step

Would you like me to continue with the remaining phases (5-12) or would you prefer to start implementing with these first 47 prompts?

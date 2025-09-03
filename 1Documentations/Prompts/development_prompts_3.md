# Development Prompts 3: LLM Cybersecurity Benchmark (Weeks 9-12)

## How to Use This Guide

This is the continuation of development_prompts_2.md, covering weeks 9-12 of the project. Each prompt builds on the work completed in the first 8 weeks and follows the same format:

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

## Phase 6: Orchestration Service (Weeks 9-10)

*Building on the completed Configuration, Data, Model, and Evaluation services*

### 6.1: Orchestration Service Foundation

#### Prompt 6.1.1: Create Orchestration Service Base Structure
🎯 **Goal**: Create the orchestration service that coordinates all other services and manages experiment lifecycle

📁 **Files**:
- `src/benchmark/services/orchestration_service.py`
- `src/benchmark/interfaces/orchestration_interfaces.py`
- `src/benchmark/workflow/workflow_engine.py`

🔧 **Task**:
Create the orchestration service that manages complete experiment workflows, coordinating between Configuration, Data, Model, and Evaluation services.

Requirements:
- Implement BaseService interface
- Coordinate all other services (config, data, model, evaluation)
- Manage experiment lifecycle (create, start, monitor, complete)
- Handle service dependencies and initialization order
- Provide unified experiment execution interface
- Support experiment cancellation and cleanup

```python
# interfaces/orchestration_interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class ExperimentStatus(Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentProgress:
    experiment_id: str
    status: ExperimentStatus
    current_step: str
    total_steps: int
    completed_steps: int
    percentage: float
    elapsed_time_seconds: float
    estimated_remaining_seconds: Optional[float]
    error_message: Optional[str] = None

@dataclass
class ExperimentResult:
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    started_at: str
    completed_at: Optional[str]
    total_duration_seconds: float
    models_evaluated: List[str]
    datasets_used: List[str]
    evaluation_results: Dict[str, Any]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class WorkflowStep(ABC):
    """Abstract base class for workflow steps"""

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step"""
        pass

    @abstractmethod
    def get_step_name(self) -> str:
        """Get human-readable step name"""
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get list of required service dependencies"""
        pass

# services/orchestration_service.py
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from benchmark.core.base import BaseService, ServiceResponse, HealthCheck, ServiceStatus
from benchmark.interfaces.orchestration_interfaces import (
    ExperimentStatus, ExperimentProgress, ExperimentResult, WorkflowStep
)
from benchmark.services.configuration_service import ConfigurationService
from benchmark.services.data_service import DataService
from benchmark.services.model_service import ModelService
from benchmark.services.evaluation_service import EvaluationService

class OrchestrationService(BaseService):
    """Service for orchestrating complete experiment workflows"""

    def __init__(self):
        self.services: Dict[str, BaseService] = {}
        self.experiments: Dict[str, ExperimentContext] = {}
        self.workflow_engine = WorkflowEngine()

    async def initialize(self) -> ServiceResponse:
        """Initialize orchestration service and all dependent services"""
        try:
            # Initialize all services in dependency order
            self.services['config'] = ConfigurationService()
            await self.services['config'].initialize()

            self.services['data'] = DataService()
            await self.services['data'].initialize()

            self.services['model'] = ModelService()
            await self.services['model'].initialize()

            self.services['evaluation'] = EvaluationService()
            await self.services['evaluation'].initialize()

            # Initialize workflow engine
            await self.workflow_engine.initialize(self.services)

            return ServiceResponse(
                success=True,
                data={"initialized_services": len(self.services)}
            )

        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def create_experiment(self, config_path: str,
                              experiment_name: Optional[str] = None) -> ServiceResponse:
        """Create a new experiment from configuration"""

        try:
            # Generate experiment ID
            experiment_id = f"exp_{int(time.time())}_{hash(config_path) % 1000:03d}"

            # Load configuration
            config_response = await self.services['config'].load_experiment_config(config_path)
            if not config_response.success:
                return ServiceResponse(
                    success=False,
                    error=f"Failed to load configuration: {config_response.error}"
                )

            experiment_config = config_response.data

            # Create experiment context
            context = ExperimentContext(
                experiment_id=experiment_id,
                name=experiment_name or experiment_config.get('name', f'Experiment {experiment_id}'),
                config=experiment_config,
                status=ExperimentStatus.CREATED,
                created_at=datetime.now(),
                services=self.services
            )

            self.experiments[experiment_id] = context

            return ServiceResponse(
                success=True,
                data={
                    "experiment_id": experiment_id,
                    "experiment_name": context.name,
                    "models_count": len(experiment_config.get('models', [])),
                    "datasets_count": len(experiment_config.get('datasets', []))
                }
            )

        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def start_experiment(self, experiment_id: str,
                             background: bool = True) -> ServiceResponse:
        """Start experiment execution"""

        if experiment_id not in self.experiments:
            return ServiceResponse(
                success=False,
                error=f"Experiment {experiment_id} not found"
            )

        context = self.experiments[experiment_id]

        if context.status != ExperimentStatus.CREATED:
            return ServiceResponse(
                success=False,
                error=f"Experiment {experiment_id} is not in CREATED status"
            )

        if background:
            # Start experiment in background
            asyncio.create_task(self._run_experiment_workflow(experiment_id))

            return ServiceResponse(
                success=True,
                data={"message": "Experiment started in background"}
            )
        else:
            # Run experiment synchronously
            try:
                result = await self._run_experiment_workflow(experiment_id)
                return ServiceResponse(success=True, data=result)
            except Exception as e:
                return ServiceResponse(success=False, error=str(e))

    async def _run_experiment_workflow(self, experiment_id: str) -> ExperimentResult:
        """Run the complete experiment workflow"""

        context = self.experiments[experiment_id]
        context.status = ExperimentStatus.INITIALIZING
        context.started_at = datetime.now()

        try:
            # Create workflow steps
            workflow_steps = [
                DataLoadingStep(),
                ModelLoadingStep(),
                EvaluationExecutionStep(),
                ResultsAggregationStep()
            ]

            # Execute workflow
            workflow_result = await self.workflow_engine.execute_workflow(
                workflow_steps, context
            )

            # Update context with results
            context.status = ExperimentStatus.COMPLETED
            context.completed_at = datetime.now()
            context.results = workflow_result

            # Create experiment result
            duration = (context.completed_at - context.started_at).total_seconds()

            return ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=context.name,
                status=ExperimentStatus.COMPLETED,
                started_at=context.started_at.isoformat(),
                completed_at=context.completed_at.isoformat(),
                total_duration_seconds=duration,
                models_evaluated=list(context.loaded_models.keys()),
                datasets_used=list(context.loaded_datasets.keys()),
                evaluation_results=workflow_result,
                metadata={"workflow_steps": len(workflow_steps)}
            )

        except Exception as e:
            context.status = ExperimentStatus.FAILED
            context.error_message = str(e)

            return ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=context.name,
                status=ExperimentStatus.FAILED,
                started_at=context.started_at.isoformat(),
                completed_at=None,
                total_duration_seconds=0,
                models_evaluated=[],
                datasets_used=[],
                evaluation_results={},
                error_message=str(e)
            )

    async def get_experiment_progress(self, experiment_id: str) -> ServiceResponse:
        """Get current experiment progress"""

        if experiment_id not in self.experiments:
            return ServiceResponse(
                success=False,
                error=f"Experiment {experiment_id} not found"
            )

        context = self.experiments[experiment_id]

        # Calculate progress
        elapsed_time = 0
        if context.started_at:
            elapsed_time = (datetime.now() - context.started_at).total_seconds()

        progress = ExperimentProgress(
            experiment_id=experiment_id,
            status=context.status,
            current_step=context.current_step or "Not started",
            total_steps=context.total_steps,
            completed_steps=context.completed_steps,
            percentage=context.completed_steps / context.total_steps * 100 if context.total_steps > 0 else 0,
            elapsed_time_seconds=elapsed_time,
            estimated_remaining_seconds=self._estimate_remaining_time(context),
            error_message=context.error_message
        )

        return ServiceResponse(success=True, data=progress)

    def _estimate_remaining_time(self, context: 'ExperimentContext') -> Optional[float]:
        """Estimate remaining execution time"""

        if context.completed_steps == 0 or context.total_steps == 0:
            return None

        if not context.started_at:
            return None

        elapsed_time = (datetime.now() - context.started_at).total_seconds()
        progress_ratio = context.completed_steps / context.total_steps

        if progress_ratio > 0:
            estimated_total_time = elapsed_time / progress_ratio
            remaining_time = estimated_total_time - elapsed_time
            return max(0, remaining_time)

        return None

    async def cancel_experiment(self, experiment_id: str) -> ServiceResponse:
        """Cancel running experiment"""

        if experiment_id not in self.experiments:
            return ServiceResponse(
                success=False,
                error=f"Experiment {experiment_id} not found"
            )

        context = self.experiments[experiment_id]

        if context.status not in [ExperimentStatus.INITIALIZING, ExperimentStatus.RUNNING]:
            return ServiceResponse(
                success=False,
                error=f"Experiment {experiment_id} cannot be cancelled (status: {context.status.value})"
            )

        try:
            # Set cancellation flag
            context.cancel_requested = True
            context.status = ExperimentStatus.CANCELLED

            # Cleanup resources
            await self._cleanup_experiment_resources(context)

            return ServiceResponse(
                success=True,
                data={"message": "Experiment cancelled successfully"}
            )

        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    async def _cleanup_experiment_resources(self, context: 'ExperimentContext') -> None:
        """Clean up experiment resources"""

        # Cleanup loaded models
        if 'model' in self.services:
            for model_id in context.loaded_models:
                try:
                    await self.services['model'].cleanup_model(model_id)
                except:
                    pass  # Ignore cleanup errors

        # Clear context data
        context.loaded_datasets.clear()
        context.loaded_models.clear()

    async def list_experiments(self, status_filter: Optional[ExperimentStatus] = None) -> ServiceResponse:
        """List all experiments with optional status filter"""

        experiments_list = []

        for exp_id, context in self.experiments.items():
            if status_filter is None or context.status == status_filter:
                experiments_list.append({
                    'experiment_id': exp_id,
                    'name': context.name,
                    'status': context.status.value,
                    'created_at': context.created_at.isoformat(),
                    'started_at': context.started_at.isoformat() if context.started_at else None,
                    'completed_at': context.completed_at.isoformat() if context.completed_at else None
                })

        return ServiceResponse(success=True, data=experiments_list)

    async def health_check(self) -> HealthCheck:
        """Check orchestration service health"""

        try:
            # Check all dependent services
            service_health = {}
            overall_status = ServiceStatus.HEALTHY

            for service_name, service in self.services.items():
                health = await service.health_check()
                service_health[service_name] = health.status.value

                if health.status == ServiceStatus.UNHEALTHY:
                    overall_status = ServiceStatus.UNHEALTHY
                elif health.status == ServiceStatus.DEGRADED and overall_status == ServiceStatus.HEALTHY:
                    overall_status = ServiceStatus.DEGRADED

            return HealthCheck(
                service_name="orchestration_service",
                status=overall_status,
                timestamp=datetime.now().isoformat(),
                details={
                    "dependent_services": service_health,
                    "active_experiments": len([exp for exp in self.experiments.values()
                                             if exp.status in [ExperimentStatus.RUNNING, ExperimentStatus.INITIALIZING]]),
                    "total_experiments": len(self.experiments)
                }
            )

        except Exception as e:
            return HealthCheck(
                service_name="orchestration_service",
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )

    async def shutdown(self) -> ServiceResponse:
        """Graceful shutdown of orchestration service"""

        try:
            # Cancel all running experiments
            for exp_id, context in self.experiments.items():
                if context.status in [ExperimentStatus.RUNNING, ExperimentStatus.INITIALIZING]:
                    await self.cancel_experiment(exp_id)

            # Shutdown all services
            for service in self.services.values():
                await service.shutdown()

            return ServiceResponse(success=True, data={"message": "Orchestration service shut down"})

        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

@dataclass
class ExperimentContext:
    """Context for tracking experiment execution"""
    experiment_id: str
    name: str
    config: Dict[str, Any]
    status: ExperimentStatus
    created_at: datetime
    services: Dict[str, BaseService]

    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    total_steps: int = 0
    completed_steps: int = 0

    # Resources
    loaded_datasets: Dict[str, Any] = None
    loaded_models: Dict[str, Any] = None

    # Results and errors
    results: Dict[str, Any] = None
    error_message: Optional[str] = None
    cancel_requested: bool = False

    def __post_init__(self):
        if self.loaded_datasets is None:
            self.loaded_datasets = {}
        if self.loaded_models is None:
            self.loaded_models = {}
        if self.results is None:
            self.results = {}

# workflow/workflow_engine.py
import asyncio
from typing import List, Dict, Any

class WorkflowEngine:
    """Engine for executing workflow steps in dependency order"""

    def __init__(self):
        self.services = {}

    async def initialize(self, services: Dict[str, BaseService]) -> None:
        """Initialize workflow engine with services"""
        self.services = services

    async def execute_workflow(self, steps: List[WorkflowStep],
                             context: ExperimentContext) -> Dict[str, Any]:
        """Execute workflow steps in order"""

        context.total_steps = len(steps)
        context.completed_steps = 0
        context.status = ExperimentStatus.RUNNING

        workflow_results = {}

        for i, step in enumerate(steps):
            if context.cancel_requested:
                raise Exception("Experiment cancelled by user")

            context.current_step = step.get_step_name()

            try:
                # Check dependencies
                await self._check_step_dependencies(step)

                # Execute step
                step_result = await step.execute(context.__dict__)
                workflow_results[step.get_step_name()] = step_result

                context.completed_steps += 1

            except Exception as e:
                raise Exception(f"Workflow step '{step.get_step_name()}' failed: {e}")

        return workflow_results

    async def _check_step_dependencies(self, step: WorkflowStep) -> None:
        """Check that step dependencies are available"""

        required_services = step.get_dependencies()

        for service_name in required_services:
            if service_name not in self.services:
                raise Exception(f"Required service '{service_name}' not available")

            # Check service health
            health = await self.services[service_name].health_check()
            if health.status == ServiceStatus.UNHEALTHY:
                raise Exception(f"Required service '{service_name}' is unhealthy")
```

✅ **Tests**:
Create `tests/unit/test_orchestration_service.py`:
- Test service initialization and dependency management
- Test experiment creation and lifecycle management
- Test progress tracking and status updates
- Test experiment cancellation and cleanup
- Test error handling and recovery

Create `tests/integration/test_orchestration_integration.py`:
- Test complete workflow execution
- Test integration with all dependent services
- Test resource management and cleanup

🔍 **Validation**:
- Orchestration service initializes all dependent services correctly
- Experiment lifecycle is managed properly (create → start → monitor → complete)
- Progress tracking provides accurate status updates
- Experiment cancellation works and cleans up resources
- Error handling prevents system crashes
- Integration with all services works correctly

#### Prompt 6.1.2: Create Workflow Steps Implementation
🎯 **Goal**: Implement concrete workflow steps for data loading, model loading, evaluation, and results aggregation

📁 **Files**:
- `src/benchmark/workflow/steps/data_loading.py`
- `src/benchmark/workflow/steps/model_loading.py`
- `src/benchmark/workflow/steps/evaluation_execution.py`
- `src/benchmark/workflow/steps/results_aggregation.py`

🔧 **Task**:
Create concrete implementations of workflow steps that orchestrate the complete experiment pipeline.

Requirements:
- Data loading step that loads and prepares all configured datasets
- Model loading step that loads and validates all configured models
- Evaluation execution step that runs evaluations across all model-dataset combinations
- Results aggregation step that collects and organizes all evaluation results
- Proper error handling and progress reporting for each step

```python
# workflow/steps/data_loading.py
from typing import Dict, List, Any
import asyncio

from benchmark.interfaces.orchestration_interfaces import WorkflowStep
from benchmark.services.data_service import DataService

class DataLoadingStep(WorkflowStep):
    """Workflow step for loading and preparing datasets"""

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data loading step"""

        data_service: DataService = context['services']['data']
        config = context['config']

        datasets_config = config.get('datasets', [])
        if not datasets_config:
            raise ValueError("No datasets configured for experiment")

        loaded_datasets = {}
        loading_results = {}

        # Load datasets in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent dataset loading

        async def load_single_dataset(dataset_config):
            async with semaphore:
                dataset_id = dataset_config['name']

                try:
                    # Load dataset
                    load_response = await data_service.load_dataset(dataset_id, dataset_config)

                    if not load_response.success:
                        raise Exception(f"Failed to load dataset: {load_response.error}")

                    dataset = load_response.data
                    loaded_datasets[dataset_id] = dataset

                    # Get dataset info
                    info_response = await data_service.get_dataset_info(dataset_id)
                    dataset_info = info_response.data if info_response.success else {}

                    loading_results[dataset_id] = {
                        'status': 'success',
                        'samples_count': dataset_info.get('samples_count', len(dataset.get('samples', []))),
                        'data_splits': dataset_info.get('splits', {}),
                        'preprocessing_applied': dataset_config.get('preprocessing', [])
                    }

                    return dataset_id, dataset

                except Exception as e:
                    loading_results[dataset_id] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    raise Exception(f"Failed to load dataset '{dataset_id}': {e}")

        # Execute dataset loading tasks
        loading_tasks = [load_single_dataset(ds_config) for ds_config in datasets_config]

        try:
            dataset_results = await asyncio.gather(*loading_tasks)

            # Update context with loaded datasets
            context['loaded_datasets'] = loaded_datasets

            return {
                'datasets_loaded': len(loaded_datasets),
                'loading_results': loading_results,
                'total_samples': sum(result.get('samples_count', 0)
                                   for result in loading_results.values()
                                   if result['status'] == 'success')
            }

        except Exception as e:
            # Cleanup any partially loaded datasets
            for dataset_id in loaded_datasets:
                try:
                    await data_service.cleanup_dataset(dataset_id)
                except:
                    pass  # Ignore cleanup errors

            raise e

    def get_step_name(self) -> str:
        return "Data Loading"

    def get_dependencies(self) -> List[str]:
        return ["data"]

# workflow/steps/model_loading.py
from typing import Dict, List, Any
import asyncio

from benchmark.interfaces.orchestration_interfaces import WorkflowStep
from benchmark.services.model_service import ModelService

class ModelLoadingStep(WorkflowStep):
    """Workflow step for loading and validating models"""

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model loading step"""

        model_service: ModelService = context['services']['model']
        config = context['config']

        models_config = config.get('models', [])
        if not models_config:
            raise ValueError("No models configured for experiment")

        loaded_models = {}
        loading_results = {}

        # Load models in sequence to manage memory usage
        for model_config in models_config:
            model_name = model_config['name']

            try:
                # Check if model can be loaded (resource constraints)
                resource_check = await model_service.resource_manager.can_load_model(model_config)

                if not resource_check.can_load:
                    # Try to free up memory by unloading unused models
                    if loaded_models:  # If we have loaded models, try to manage resources
                        candidates = await model_service.resource_manager.suggest_model_unload_candidates(
                            resource_check.estimated_memory_gb
                        )

                        for candidate_id in candidates:
                            if candidate_id in loaded_models:
                                await model_service.cleanup_model(candidate_id)
                                del loaded_models[candidate_id]

                        # Re-check after cleanup
                        resource_check = await model_service.resource_manager.can_load_model(model_config)

                if not resource_check.can_load:
                    raise Exception(f"Insufficient resources to load model. {resource_check.recommendations}")

                # Load model
                model_id = await model_service.load_model(model_config)
                loaded_models[model_name] = model_id

                # Get model info
                model_info_response = await model_service.get_model_info(model_id)
                model_info = model_info_response.data if model_info_response.success else {}

                loading_results[model_name] = {
                    'status': 'success',
                    'model_id': model_id,
                    'model_type': model_config.get('type'),
                    'memory_usage_gb': resource_check.estimated_memory_gb,
                    'model_info': model_info
                }

            except Exception as e:
                loading_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }

                # Continue loading other models even if one fails
                print(f"Warning: Failed to load model '{model_name}': {e}")

        if not loaded_models:
            raise Exception("No models could be loaded successfully")

        # Update context with loaded models
        context['loaded_models'] = loaded_models

        # Apply hardware optimizations
        await model_service.optimize_for_hardware()

        return {
            'models_loaded': len(loaded_models),
            'loading_results': loading_results,
            'total_memory_usage_gb': sum(
                result.get('memory_usage_gb', 0)
                for result in loading_results.values()
                if result['status'] == 'success'
            ),
            'optimization_applied': True
        }

    def get_step_name(self) -> str:
        return "Model Loading"

    def get_dependencies(self) -> List[str]:
        return ["model"]

# workflow/steps/evaluation_execution.py
from typing import Dict, List, Any
import asyncio
from itertools import product

from benchmark.interfaces.orchestration_interfaces import WorkflowStep
from benchmark.interfaces.evaluation_interfaces import EvaluationRequest, MetricType
from benchmark.services.model_service import ModelService
from benchmark.services.data_service import DataService
from benchmark.services.evaluation_service import EvaluationService

class EvaluationExecutionStep(WorkflowStep):
    """Workflow step for executing evaluations across all model-dataset combinations"""

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluation step"""

        model_service: ModelService = context['services']['model']
        data_service: DataService = context['services']['data']
        evaluation_service: EvaluationService = context['services']['evaluation']

        config = context['config']
        loaded_models = context.get('loaded_models', {})
        loaded_datasets = context.get('loaded_datasets', {})

        if not loaded_models:
            raise ValueError("No models loaded for evaluation")

        if not loaded_datasets:
            raise ValueError("No datasets loaded for evaluation")

        # Get evaluation configuration
        evaluation_config = config.get('evaluation', {})
        metrics = evaluation_config.get('metrics', ['accuracy', 'performance'])

        # Convert metric names to MetricType enums
        metric_types = []
        for metric_name in metrics:
            if metric_name == 'accuracy':
                metric_types.append(MetricType.ACCURACY)
            elif metric_name == 'performance':
                metric_types.append(MetricType.PERFORMANCE)
            elif metric_name == 'false_positive_rate' or metric_name == 'fpr':
                metric_types.append(MetricType.FALSE_POSITIVE_RATE)

        if not metric_types:
            metric_types = [MetricType.ACCURACY]  # Default

        # Create all model-dataset combinations
        evaluation_tasks = []
        for model_name, model_id in loaded_models.items():
            for dataset_name, dataset in loaded_datasets.items():
                task_id = f"{model_name}_{dataset_name}"
                evaluation_tasks.append({
                    'task_id': task_id,
                    'model_name': model_name,
                    'model_id': model_id,
                    'dataset_name': dataset_name,
                    'dataset': dataset,
                    'metrics': metric_types
                })

        print(f"Starting evaluation of {len(evaluation_tasks)} model-dataset combinations...")

        evaluation_results = {}

        # Execute evaluations with concurrency control
        semaphore = asyncio.Semaphore(evaluation_config.get('parallel_jobs', 2))

        async def execute_single_evaluation(task):
            async with semaphore:
                task_id = task['task_id']

                try:
                    # Generate predictions for this model-dataset combination
                    predictions_result = await self._generate_predictions(
                        model_service, task['model_id'], task['dataset']
                    )

                    if not predictions_result['success']:
                        raise Exception(predictions_result['error'])

                    predictions = predictions_result['predictions']
                    ground_truth = predictions_result['ground_truth']

                    # Create evaluation request
                    evaluation_request = EvaluationRequest(
                        experiment_id=context['experiment_id'],
                        model_id=task['model_id'],
                        dataset_id=task['dataset_name'],
                        predictions=predictions,
                        ground_truth=ground_truth,
                        metrics=task['metrics'],
                        metadata={
                            'model_name': task['model_name'],
                            'dataset_name': task['dataset_name'],
                            'task_id': task_id
                        }
                    )

                    # Execute evaluation
                    evaluation_id = await evaluation_service.evaluate_and_store(evaluation_request)

                    evaluation_results[task_id] = {
                        'status': 'success',
                        'evaluation_id': evaluation_id,
                        'model_name': task['model_name'],
                        'dataset_name': task['dataset_name'],
                        'predictions_count': len(predictions),
                        'metrics_evaluated': [m.value for m in task['metrics']]
                    }

                    print(f"Completed evaluation: {task_id}")
                    return task_id, evaluation_results[task_id]

                except Exception as e:
                    error_msg = str(e)
                    evaluation_results[task_id] = {
                        'status': 'failed',
                        'error': error_msg,
                        'model_name': task['model_name'],
                        'dataset_name': task['dataset_name']
                    }

                    print(f"Failed evaluation: {task_id} - {error_msg}")
                    return task_id, evaluation_results[task_id]

        # Execute all evaluation tasks
        tasks = [execute_single_evaluation(task) for task in evaluation_tasks]

        try:
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes and failures
            successful_evaluations = sum(1 for task_id, result in completed_tasks
                                       if not isinstance(result, Exception) and result['status'] == 'success')
            failed_evaluations = len(completed_tasks) - successful_evaluations

            if successful_evaluations == 0:
                raise Exception("All evaluations failed")

            return {
                'total_evaluations': len(evaluation_tasks),
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': failed_evaluations,
                'evaluation_results': evaluation_results,
                'metrics_evaluated': [m.value for m in metric_types]
            }

        except Exception as e:
            raise Exception(f"Evaluation execution failed: {e}")

    async def _generate_predictions(self, model_service: ModelService,
                                  model_id: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions for a model on a dataset"""

        try:
            # Extract test samples from dataset
            samples = dataset.get('samples', [])
            if not samples:
                return {'success': False, 'error': 'No samples in dataset'}

            # Extract input texts and ground truth labels
            input_texts = []
            ground_truth = []

            for sample in samples:
                if isinstance(sample, dict):
                    input_text = sample.get('input_text', sample.get('input', str(sample)))
                    label = sample.get('label', sample.get('ground_truth', 'UNKNOWN'))
                else:
                    input_text = str(sample)
                    label = 'UNKNOWN'

                input_texts.append(input_text)
                ground_truth.append({'label': label})

            # Generate predictions
            predictions = await model_service.predict_batch(model_id, input_texts)

            # Ensure predictions have required fields
            processed_predictions = []
            for i, pred in enumerate(predictions):
                if isinstance(pred, dict):
                    processed_pred = {
                        'prediction': pred.get('prediction', 'UNKNOWN'),
                        'confidence': pred.get('confidence', 0.5),
                        'explanation': pred.get('explanation', ''),
                        'inference_time_ms': pred.get('inference_time_ms', 0.0)
                    }
                else:
                    processed_pred = {
                        'prediction': str(pred),
                        'confidence': 0.5,
                        'explanation': '',
                        'inference_time_ms': 0.0
                    }

                processed_predictions.append(processed_pred)

            return {
                'success': True,
                'predictions': processed_predictions,
                'ground_truth': ground_truth
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_step_name(self) -> str:
        return "Evaluation Execution"

    def get_dependencies(self) -> List[str]:
        return ["model", "data", "evaluation"]

# workflow/steps/results_aggregation.py
from typing import Dict, List, Any
from datetime import datetime

from benchmark.interfaces.orchestration_interfaces import WorkflowStep
from benchmark.services.evaluation_service import EvaluationService

class ResultsAggregationStep(WorkflowStep):
    """Workflow step for aggregating and organizing evaluation results"""

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute results aggregation step"""

        evaluation_service: EvaluationService = context['services']['evaluation']
        experiment_id = context['experiment_id']

        # Get all evaluation results for this experiment
        results_response = await evaluation_service.results_storage.get_evaluation_results(
            experiment_id=experiment_id
        )

        if not results_response:
            raise Exception("No evaluation results found")

        evaluation_results = results_response

        # Generate comprehensive report
        comprehensive_report = await evaluation_service.generate_comprehensive_report(experiment_id)

        # Create model comparison for key metrics
        loaded_models = context.get('loaded_models', {})
        model_ids = list(loaded_models.values())

        model_comparisons = {}
        if len(model_ids) > 1:
            # Compare models on key metrics
            key_metrics = ['f1_score', 'accuracy', 'precision', 'recall']

            for metric in key_metrics:
                try:
                    comparison = await evaluation_service.compare_model_results(model_ids, metric)
                    if comparison.get('results'):
                        model_comparisons[metric] = comparison
                except Exception as e:
                    print(f"Warning: Failed to compare models on {metric}: {e}")

        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(evaluation_results)

        # Generate insights and recommendations
        insights = self._generate_experiment_insights(
            evaluation_results,
            context.get('config', {}),
            summary_stats
        )

        # Prepare final aggregated results
        aggregated_results = {
            'experiment_summary': {
                'experiment_id': experiment_id,
                'experiment_name': context.get('name', ''),
                'total_evaluations': len(evaluation_results),
                'models_evaluated': len(set(r.get('model_id', '') for r in evaluation_results)),
                'datasets_used': len(set(r.get('dataset_id', '') for r in evaluation_results)),
                'completed_at': datetime.now().isoformat()
            },
            'comprehensive_report': comprehensive_report,
            'model_comparisons': model_comparisons,
            'summary_statistics': summary_stats,
            'insights_and_recommendations': insights,
            'raw_evaluation_results': evaluation_results
        }

        return aggregated_results

    def _calculate_summary_statistics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all evaluations"""

        stats = {
            'metrics_summary': {},
            'model_performance_summary': {},
            'dataset_difficulty_analysis': {}
        }

        # Group results by metric
        metrics_data = {}
        for result in evaluation_results:
            metric_name = result.get('metric_name')
            metric_value = result.get('value')

            if metric_name and metric_value is not None:
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(float(metric_value))

        # Calculate statistics for each metric
        for metric_name, values in metrics_data.items():
            if values:
                stats['metrics_summary'][metric_name] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': self._calculate_std(values)
                }

        # Model performance summary
        model_scores = {}
        for result in evaluation_results:
            if result.get('metric_name') == 'f1_score':  # Use F1 as primary metric
                model_id = result.get('model_id')
                if model_id:
                    if model_id not in model_scores:
                        model_scores[model_id] = []
                    model_scores[model_id].append(result.get('value', 0))

        for model_id, scores in model_scores.items():
            if scores:
                stats['model_performance_summary'][model_id] = {
                    'avg_f1_score': sum(scores) / len(scores),
                    'evaluations_count': len(scores)
                }

        return stats

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _generate_experiment_insights(self, evaluation_results: List[Dict[str, Any]],
                                    config: Dict[str, Any],
                                    summary_stats: Dict[str, Any]) -> List[str]:
        """Generate insights from experiment results"""

        insights = []

        # Performance insights
        f1_scores = summary_stats.get('metrics_summary', {}).get('f1_score', {})
        if f1_scores:
            avg_f1 = f1_scores['mean']
            if avg_f1 > 0.9:
                insights.append(f"Excellent overall performance achieved (avg F1: {avg_f1:.3f})")
            elif avg_f1 > 0.8:
                insights.append(f"Good overall performance achieved (avg F1: {avg_f1:.3f})")
            elif avg_f1 > 0.6:
                insights.append(f"Moderate performance achieved (avg F1: {avg_f1:.3f}) - consider model tuning")
            else:
                insights.append(f"Performance below expectations (avg F1: {avg_f1:.3f}) - review model selection")

        # Model consistency insights
        model_performance = summary_stats.get('model_performance_summary', {})
        if len(model_performance) > 1:
            scores = [perf['avg_f1_score'] for perf in model_performance.values()]
            score_range = max(scores) - min(scores)

            if score_range > 0.2:
                insights.append(f"High variability between models (range: {score_range:.3f}) - significant model differences detected")
            else:
                insights.append(f"Consistent performance across models (range: {score_range:.3f})")

        # Speed vs accuracy insights
        inference_times = summary_stats.get('metrics_summary', {}).get('avg_inference_time_ms', {})
        if inference_times and f1_scores:
            avg_time = inference_times['mean']
            avg_f1 = f1_scores['mean']

            if avg_time < 200 and avg_f1 > 0.8:
                insights.append(f"Excellent speed-accuracy balance achieved ({avg_time:.0f}ms, {avg_f1:.3f} F1)")
            elif avg_time > 1000:
                insights.append(f"Slow inference detected ({avg_time:.0f}ms avg) - consider optimization")

        # False positive insights
        fpr_stats = summary_stats.get('metrics_summary', {}).get('false_positive_rate', {})
        if fpr_stats:
            avg_fpr = fpr_stats['mean']
            if avg_fpr > 0.1:
                insights.append(f"High false positive rate detected ({avg_fpr:.3f}) - may impact operational efficiency")
            elif avg_fpr < 0.05:
                insights.append(f"Excellent false positive control achieved ({avg_fpr:.3f})")

        # Configuration insights
        models_count = len(config.get('models', []))
        datasets_count = len(config.get('datasets', []))

        if models_count == 1 and f1_scores and f1_scores['mean'] < 0.7:
            insights.append("Single model evaluation with suboptimal performance - consider testing additional models")

        if datasets_count == 1:
            insights.append("Single dataset evaluation - consider testing on multiple datasets for robustness validation")

        return insights

    def get_step_name(self) -> str:
        return "Results Aggregation"

    def get_dependencies(self) -> List[str]:
        return ["evaluation"]
```

✅ **Tests**:
Create `tests/unit/test_workflow_steps.py`:
- Test each workflow step individually with mock services
- Test error handling and recovery for each step
- Test resource management and cleanup

Create `tests/integration/test_complete_workflow.py`:
- Test complete workflow execution with real services
- Test workflow with various configurations
- Test error scenarios and recovery

🔍 **Validation**:
- Data loading step successfully loads and validates all configured datasets
- Model loading step manages resources correctly and loads models efficiently
- Evaluation execution step runs all model-dataset combinations
- Results aggregation step produces comprehensive analysis and insights
- All steps handle errors gracefully and provide useful error messages
- Workflow execution completes successfully end-to-end

#### Prompt 6.1.3: Create Orchestration Service Integration Tests
🎯 **Goal**: Create comprehensive integration tests for the complete orchestration service

📁 **Files**: `tests/integration/test_orchestration_service_complete.py`

🔧 **Task**:
Create thorough integration tests that validate the complete orchestration service functionality with realistic experiment scenarios.

Requirements:
- Test complete experiment workflows from creation to completion
- Test various experiment configurations and scenarios
- Test resource management and cleanup
- Test error handling and recovery
- Test concurrent experiment execution
- Test realistic cybersecurity evaluation scenarios

```python
# tests/integration/test_orchestration_service_complete.py
import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from benchmark.services.orchestration_service import OrchestrationService
from benchmark.interfaces.orchestration_interfaces import ExperimentStatus

class TestOrchestrationServiceComplete:
    """Comprehensive integration tests for Orchestration Service"""

    @pytest.fixture
    async def orchestration_service(self):
        """Fully configured orchestration service"""
        service = OrchestrationService()
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.fixture
    def sample_experiment_config(self):
        """Sample experiment configuration for testing"""
        return {
            'name': 'Integration Test Experiment',
            'description': 'Testing complete orchestration workflow',
            'datasets': [
                {
                    'name': 'test_dataset_1',
                    'source': 'local',
                    'path': 'test://sample_data.json',
                    'max_samples': 100,
                    'preprocessing': ['normalize']
                }
            ],
            'models': [
                {
                    'name': 'test_model_1',
                    'type': 'mlx_local',
                    'path': 'test://mock_model',
                    'max_tokens': 256,
                    'temperature': 0.1
                }
            ],
            'evaluation': {
                'metrics': ['accuracy', 'performance'],
                'parallel_jobs': 2,
                'timeout_minutes': 30
            }
        }

    @pytest.fixture
    def multi_model_config(self):
        """Multi-model experiment configuration"""
        return {
            'name': 'Multi-Model Comparison',
            'description': 'Compare multiple models on cybersecurity tasks',
            'datasets': [
                {
                    'name': 'network_logs',
                    'source': 'local',
                    'path': 'test://network_data.json',
                    'max_samples': 500
                },
                {
                    'name': 'email_data',
                    'source': 'local',
                    'path': 'test://email_data.json',
                    'max_samples': 300
                }
            ],
            'models': [
                {
                    'name': 'cybersec_bert',
                    'type': 'huggingface',
                    'path': 'test://cybersec_bert_model',
                    'max_tokens': 512
                },
                {
                    'name': 'llama_3b',
                    'type': 'mlx_local',
                    'path': 'test://llama_3b_model',
                    'max_tokens': 256
                },
                {
                    'name': 'gpt_4o_mini',
                    'type': 'openai_api',
                    'path': 'gpt-4o-mini',
                    'max_tokens': 512
                }
            ],
            'evaluation': {
                'metrics': ['accuracy', 'performance', 'false_positive_rate'],
                'parallel_jobs': 3,
                'timeout_minutes': 60
            }
        }

    async def test_complete_experiment_workflow(self, orchestration_service, sample_experiment_config):
        """Test complete experiment workflow from creation to completion"""

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_experiment_config, f)
            config_path = f.name

        try:
            # Mock the services to simulate successful operations
            with patch.multiple(
                orchestration_service.services['data'],
                load_dataset=MagicMock(return_value=self._mock_service_response(True, {'samples': self._generate_mock_dataset()})),
                get_dataset_info=MagicMock(return_value=self._mock_service_response(True, {'samples_count': 100}))
            ):
                with patch.multiple(
                    orchestration_service.services['model'],
                    resource_manager=MagicMock(),
                    load_model=MagicMock(return_value='mock_model_id'),
                    get_model_info=MagicMock(return_value=self._mock_service_response(True, {'model_type': 'test'})),
                    predict_batch=MagicMock(return_value=self._generate_mock_predictions(100)),
                    optimize_for_hardware=MagicMock()
                ):
                    # Configure resource manager mock
                    orchestration_service.services['model'].resource_manager.can_load_model.return_value = \
                        MagicMock(can_load=True, estimated_memory_gb=4.0, recommendations=[])

                    with patch.multiple(
                        orchestration_service.services['evaluation'],
                        evaluate_and_store=MagicMock(return_value='eval_123'),
                        results_storage=MagicMock()
                    ):
                        # Configure evaluation service mock
                        orchestration_service.services['evaluation'].results_storage.get_evaluation_results.return_value = \
                            self._generate_mock_evaluation_results()
                        orchestration_service.services['evaluation'].generate_comprehensive_report.return_value = \
                            {'experiment_summary': {'status': 'completed'}}

                        # Create experiment
                        create_response = await orchestration_service.create_experiment(config_path)

                        assert create_response.success
                        experiment_id = create_response.data['experiment_id']

                        # Start experiment synchronously for testing
                        start_response = await orchestration_service.start_experiment(experiment_id, background=False)

                        assert start_response.success

                        # Check experiment status
                        progress_response = await orchestration_service.get_experiment_progress(experiment_id)
                        assert progress_response.success

                        progress = progress_response.data
                        assert progress.status == ExperimentStatus.COMPLETED
                        assert progress.completed_steps == progress.total_steps
                        assert progress.percentage == 100.0

        finally:
            Path(config_path).unlink(missing_ok=True)

    async def test_multi_model_experiment(self, orchestration_service, multi_model_config):
        """Test experiment with multiple models and datasets"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(multi_model_config, f)
            config_path = f.name

        try:
            # Mock services for multi-model scenario
            with patch.multiple(
                orchestration_service.services['data'],
                load_dataset=MagicMock(side_effect=self._mock_dataset_loading),
                get_dataset_info=MagicMock(return_value=self._mock_service_response(True, {'samples_count': 400}))
            ):
                with patch.multiple(
                    orchestration_service.services['model'],
                    resource_manager=MagicMock(),
                    load_model=MagicMock(side_effect=['model_1', 'model_2', 'model_3']),
                    get_model_info=MagicMock(return_value=self._mock_service_response(True, {'model_type': 'test'})),
                    predict_batch=MagicMock(return_value=self._generate_mock_predictions(400)),
                    optimize_for_hardware=MagicMock(),
                    cleanup_model=MagicMock()
                ):
                    # Configure resource manager for multiple models
                    orchestration_service.services['model'].resource_manager.can_load_model.return_value = \
                        MagicMock(can_load=True, estimated_memory_gb=6.0, recommendations=[])
                    orchestration_service.services['model'].resource_manager.suggest_model_unload_candidates.return_value = []

                    with patch.multiple(
                        orchestration_service.services['evaluation'],
                        evaluate_and_store=MagicMock(side_effect=['eval_1', 'eval_2', 'eval_3', 'eval_4', 'eval_5', 'eval_6']),
                        results_storage=MagicMock(),
                        generate_comprehensive_report=MagicMock(return_value={'models_compared': 3}),
                        compare_model_results=MagicMock(return_value={'best_model': 'model_1'})
                    ):
                        orchestration_service.services['evaluation'].results_storage.get_evaluation_results.return_value = \
                            self._generate_mock_evaluation_results(6)  # 3 models × 2 datasets

                        # Create and run experiment
                        create_response = await orchestration_service.create_experiment(config_path, "Multi-Model Test")
                        assert create_response.success

                        experiment_id = create_response.data['experiment_id']

                        # Start in background
                        start_response = await orchestration_service.start_experiment(experiment_id, background=True)
                        assert start_response.success

                        # Wait for completion (with timeout)
                        max_wait_time = 30  # seconds
                        wait_time = 0

                        while wait_time < max_wait_time:
                            await asyncio.sleep(1)
                            wait_time += 1

                            progress_response = await orchestration_service.get_experiment_progress(experiment_id)
                            if progress_response.success:
                                progress = progress_response.data
                                if progress.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
                                    break

                        # Verify completion
                        final_progress = await orchestration_service.get_experiment_progress(experiment_id)
                        assert final_progress.success
                        assert final_progress.data.status == ExperimentStatus.COMPLETED

                        # Verify all model-dataset combinations were evaluated
                        # 3 models × 2 datasets = 6 evaluations expected
                        assert orchestration_service.services['evaluation'].evaluate_and_store.call_count == 6

        finally:
            Path(config_path).unlink(missing_ok=True)

    async def test_experiment_cancellation(self, orchestration_service, sample_experiment_config):
        """Test experiment cancellation and cleanup"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_experiment_config, f)
            config_path = f.name

        try:
            # Mock services with delayed responses to simulate long-running experiment
            async def slow_model_loading(*args, **kwargs):
                await asyncio.sleep(2)  # Simulate slow model loading
                return 'slow_model_id'

            with patch.multiple(
                orchestration_service.services['data'],
                load_dataset=MagicMock(return_value=self._mock_service_response(True, {'samples': self._generate_mock_dataset()})),
                get_dataset_info=MagicMock(return_value=self._mock_service_response(True, {'samples_count': 100}))
            ):
                with patch.multiple(
                    orchestration_service.services['model'],
                    resource_manager=MagicMock(),
                    load_model=MagicMock(side_effect=slow_model_loading),
                    cleanup_model=MagicMock()
                ):
                    orchestration_service.services['model'].resource_manager.can_load_model.return_value = \
                        MagicMock(can_load=True, estimated_memory_gb=4.0, recommendations=[])

                    # Create experiment
                    create_response = await orchestration_service.create_experiment(config_path)
                    assert create_response.success

                    experiment_id = create_response.data['experiment_id']

                    # Start experiment in background
                    start_response = await orchestration_service.start_experiment(experiment_id, background=True)
                    assert start_response.success

                    # Wait a moment for experiment to start
                    await asyncio.sleep(0.5)

                    # Check that experiment is running
                    progress_response = await orchestration_service.get_experiment_progress(experiment_id)
                    assert progress_response.success
                    assert progress_response.data.status in [ExperimentStatus.INITIALIZING, ExperimentStatus.RUNNING]

                    # Cancel experiment
                    cancel_response = await orchestration_service.cancel_experiment(experiment_id)
                    assert cancel_response.success

                    # Verify experiment is cancelled
                    final_progress = await orchestration_service.get_experiment_progress(experiment_id)
                    assert final_progress.success
                    assert final_progress.data.status == ExperimentStatus.CANCELLED

                    # Verify cleanup was called
                    orchestration_service.services['model'].cleanup_model.assert_called()

        finally:
            Path(config_path).unlink(missing_ok=True)

    async def test_error_handling_and_recovery(self, orchestration_service):
        """Test error handling in various failure scenarios"""

        # Test with invalid configuration
        invalid_config = {'invalid': 'config'}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            invalid_config_path = f.name

        try:
            # Should fail to create experiment with invalid config
            create_response = await orchestration_service.create_experiment(invalid_config_path)
            assert not create_response.success
            assert "configuration" in create_response.error.lower()
        finally:
            Path(invalid_config_path).unlink(missing_ok=True)

        # Test with model loading failure
        failing_config = {
            'name': 'Failing Experiment',
            'datasets': [{'name': 'test_dataset', 'source': 'local', 'path': 'test://data.json'}],
            'models': [{'name': 'failing_model', 'type': 'invalid_type', 'path': 'invalid://model'}],
            'evaluation': {'metrics': ['accuracy']}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(failing_config, f)
            failing_config_path = f.name

        try:
            with patch.multiple(
                orchestration_service.services['data'],
                load_dataset=MagicMock(return_value=self._mock_service_response(True, {'samples': self._generate_mock_dataset()})),
                get_dataset_info=MagicMock(return_value=self._mock_service_response(True, {'samples_count': 100}))
            ):
                with patch.multiple(
                    orchestration_service.services['model'],
                    resource_manager=MagicMock(),
                    load_model=MagicMock(side_effect=Exception("Model loading failed"))
                ):
                    orchestration_service.services['model'].resource_manager.can_load_model.return_value = \
                        MagicMock(can_load=True, estimated_memory_gb=4.0, recommendations=[])

                    create_response = await orchestration_service.create_experiment(failing_config_path)
                    assert create_response.success

                    experiment_id = create_response.data['experiment_id']

                    # Experiment should fail during execution
                    start_response = await orchestration_service.start_experiment(experiment_id, background=False)
                    assert not start_response.success

                    # Check that experiment status shows failure
                    progress_response = await orchestration_service.get_experiment_progress(experiment_id)
                    assert progress_response.success
                    assert progress_response.data.status == ExperimentStatus.FAILED
                    assert progress_response.data.error_message is not None

        finally:
            Path(failing_config_path).unlink(missing_ok=True)

    async def test_concurrent_experiments(self, orchestration_service, sample_experiment_config):
        """Test running multiple experiments concurrently"""

        # Create multiple config files
        config_paths = []
        for i in range(3):
            config = sample_experiment_config.copy()
            config['name'] = f'Concurrent Experiment {i+1}'

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                config_paths.append(f.name)

        try:
            # Mock services for concurrent execution
            with patch.multiple(
                orchestration_service.services['data'],
                load_dataset=MagicMock(return_value=self._mock_service_response(True, {'samples': self._generate_mock_dataset()})),
                get_dataset_info=MagicMock(return_value=self._mock_service_response(True, {'samples_count': 100}))
            ):
                with patch.multiple(
                    orchestration_service.services['model'],
                    resource_manager=MagicMock(),
                    load_model=MagicMock(side_effect=['model_1', 'model_2', 'model_3']),
                    get_model_info=MagicMock(return_value=self._mock_service_response(True, {'model_type': 'test'})),
                    predict_batch=MagicMock(return_value=self._generate_mock_predictions(100)),
                    optimize_for_hardware=MagicMock(),
                    cleanup_model=MagicMock()
                ):
                    orchestration_service.services['model'].resource_manager.can_load_model.return_value = \
                        MagicMock(can_load=True, estimated_memory_gb=4.0, recommendations=[])

                    with patch.multiple(
                        orchestration_service.services['evaluation'],
                        evaluate_and_store=MagicMock(side_effect=['eval_1', 'eval_2', 'eval_3']),
                        results_storage=MagicMock(),
                        generate_comprehensive_report=MagicMock(return_value={'status': 'completed'})
                    ):
                        orchestration_service.services['evaluation'].results_storage.get_evaluation_results.return_value = \
                            self._generate_mock_evaluation_results(1)

                        # Create all experiments
                        experiment_ids = []
                        for config_path in config_paths:
                            create_response = await orchestration_service.create_experiment(config_path)
                            assert create_response.success
                            experiment_ids.append(create_response.data['experiment_id'])

                        # Start all experiments concurrently
                        start_tasks = [
                            orchestration_service.start_experiment(exp_id, background=True)
                            for exp_id in experiment_ids
                        ]

                        start_responses = await asyncio.gather(*start_tasks)
                        for response in start_responses:
                            assert response.success

                        # Wait for all to complete
                        max_wait_time = 15  # seconds
                        completed_experiments = 0

                        for _ in range(max_wait_time):
                            await asyncio.sleep(1)

                            completed_count = 0
                            for exp_id in experiment_ids:
                                progress_response = await orchestration_service.get_experiment_progress(exp_id)
                                if progress_response.success and progress_response.data.status == ExperimentStatus.COMPLETED:
                                    completed_count += 1

                            if completed_count == len(experiment_ids):
                                completed_experiments = completed_count
                                break

                        # Verify all experiments completed
                        assert completed_experiments == len(experiment_ids)

        finally:
            for config_path in config_paths:
                Path(config_path).unlink(missing_ok=True)

    async def test_service_health_monitoring(self, orchestration_service):
        """Test orchestration service health monitoring"""

        health = await orchestration_service.health_check()

        assert health.service_name == "orchestration_service"
        assert health.status is not None
        assert 'dependent_services' in health.details
        assert 'active_experiments' in health.details
        assert 'total_experiments' in health.details

        # Check that all dependent services are reported
        dependent_services = health.details['dependent_services']
        expected_services = ['config', 'data', 'model', 'evaluation']

        for service_name in expected_services:
            assert service_name in dependent_services

    # Helper methods for mocking
    def _mock_service_response(self, success: bool, data: Any = None, error: str = None):
        """Create mock service response"""
        class MockResponse:
            def __init__(self, success, data, error):
                self.success = success
                self.data = data
                self.error = error

        return MockResponse(success, data, error)

    def _generate_mock_dataset(self, size: int = 100) -> list:
        """Generate mock dataset for testing"""
        return [
            {
                'input_text': f'Sample network log entry {i}',
                'label': 'ATTACK' if i % 3 == 0 else 'BENIGN',
                'attack_type': 'malware' if i % 3 == 0 else None
            }
            for i in range(size)
        ]

    def _generate_mock_predictions(self, size: int) -> list:
        """Generate mock predictions for testing"""
        return [
            {
                'prediction': 'ATTACK' if i % 3 == 0 else 'BENIGN',
                'confidence': 0.85 + (i % 10) * 0.01,
                'explanation': f'Analysis of sample {i}',
                'inference_time_ms': 100.0 + (i % 50)
            }
            for i in range(size)
        ]

    def _generate_mock_evaluation_results(self, count: int = 1) -> list:
        """Generate mock evaluation results"""
        results = []
        for i in range(count):
            base_results = [
                {'metric_name': 'accuracy', 'value': 0.85 + i * 0.02, 'metric_type': 'accuracy'},
                {'metric_name': 'precision', 'value': 0.83 + i * 0.02, 'metric_type': 'accuracy'},
                {'metric_name': 'recall', 'value': 0.87 + i * 0.02, 'metric_type': 'accuracy'},
                {'metric_name': 'f1_score', 'value': 0.85 + i * 0.02, 'metric_type': 'accuracy'},
                {'metric_name': 'avg_inference_time_ms', 'value': 150.0 - i * 10, 'metric_type': 'performance'}
            ]
            results.extend(base_results)

        return results

    async def _mock_dataset_loading(self, dataset_id, config):
        """Mock dataset loading with different datasets"""
        if 'network' in dataset_id:
            return self._mock_service_response(True, {'samples': self._generate_mock_dataset(500)})
        elif 'email' in dataset_id:
            return self._mock_service_response(True, {'samples': self._generate_mock_dataset(300)})
        else:
            return self._mock_service_response(True, {'samples': self._generate_mock_dataset(100)})
```

✅ **Tests**:
The entire file is comprehensive integration testing. Additional unit tests should be created:

Create `tests/unit/test_orchestration_service_unit.py`:
- Test individual orchestration service methods
- Test experiment context management
- Test service initialization and shutdown

🔍 **Validation**:
- Complete experiment workflows execute successfully from creation to completion
- Multi-model experiments handle resource management correctly
- Experiment cancellation works and cleans up resources properly
- Error handling prevents crashes and provides useful error messages
- Concurrent experiments can run without conflicts
- Service health monitoring reports accurate status across all dependent services
- All workflow steps execute in correct order with proper dependency management
- Results aggregation produces comprehensive analysis and insights

---

## Phase 7: Advanced Evaluation Metrics - Explainability (Weeks 10-12)

*Building on the basic evaluation service to add sophisticated explainability evaluation*

### 7.1: Explainability Evaluation Foundation

#### Prompt 7.1.1: Create Explainability Evaluator Base
🎯 **Goal**: Create the foundation for explainability evaluation using multiple approaches

📁 **Files**:
- `src/benchmark/evaluation/metrics/explainability.py`
- `src/benchmark/evaluation/explainability/llm_judge.py`
- `src/benchmark/evaluation/explainability/automated_metrics.py`

🔧 **Task**:
Create a comprehensive explainability evaluator that uses multiple approaches to assess the quality of model explanations.

Requirements:
- LLM-as-judge evaluation for semantic quality
- Automated metrics (BLEU, ROUGE, BERTScore) for comparison with reference explanations
- Technical accuracy assessment for cybersecurity domain
- Consistency checking across explanations
- Integration with the existing evaluation service

```python
# src/benchmark/evaluation/metrics/explainability.py
from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass

from benchmark.evaluation.base_evaluator import BaseEvaluator
from benchmark.evaluation.explainability.llm_judge import LLMJudgeEvaluator
from benchmark.evaluation.explainability.automated_metrics import AutomatedMetricsEvaluator

@dataclass
class ExplanationQualityScore:
    """Container for explanation quality assessment"""
    overall_score: float  # 0.0 to 1.0
    technical_accuracy: float
    logical_consistency: float
    completeness: float
    clarity: float
    domain_relevance: float
    detailed_feedback: str

class ExplainabilityEvaluator(BaseEvaluator):
    """Comprehensive explainability evaluation using multiple approaches"""

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        self.llm_judge = LLMJudgeEvaluator(judge_model)
        self.automated_metrics = AutomatedMetricsEvaluator()

        self.metric_names = [
            'avg_explanation_quality', 'technical_accuracy_score',
            'logical_consistency_score', 'completeness_score',
            'clarity_score', 'domain_relevance_score',
            'bleu_score', 'rouge_l_score', 'bert_score',
            'explanation_consistency', 'ioc_accuracy',
            'mitre_coverage', 'explanation_length_avg'
        ]

    async def evaluate(self, predictions: List[Dict[str, Any]],
                      ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate explanation quality using multiple methods"""

        self.validate_input_data(predictions, ground_truth)

        # Extract explanations
        explanations = self._extract_explanations(predictions)
        if not explanations:
            raise ValueError("No explanations found in predictions")

        # Run all evaluation methods in parallel
        evaluation_tasks = []

        # LLM-as-judge evaluation
        evaluation_tasks.append(('llm_judge', self._evaluate_with_llm_judge(predictions, ground_truth)))

        # Automated metrics evaluation
        evaluation_tasks.append(('automated', self._evaluate_with_automated_metrics(predictions, ground_truth)))

        # Domain-specific evaluation
        evaluation_tasks.append(('domain', self._evaluate_domain_specific(predictions, ground_truth)))

        # Execute all evaluations
        results = {}
        for eval_name, task in evaluation_tasks:
            try:
                eval_result = await task
                results.update(eval_result)
            except Exception as e:
                print(f"Warning: {eval_name} evaluation failed: {e}")
                # Continue with other evaluations

        return results

    def _extract_explanations(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Extract explanations from predictions"""
        explanations = []

        for pred in predictions:
            explanation = pred.get('explanation', '')
            if not explanation:
                # Try alternative field names
                explanation = pred.get('reasoning', pred.get('rationale', ''))

            explanations.append(str(explanation) if explanation else '')

        return explanations

    async def _evaluate_with_llm_judge(self, predictions: List[Dict[str, Any]],
                                     ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate explanations using LLM-as-judge"""

        judge_results = []

        # Process in batches to manage API rate limits
        batch_size = 10
        for i in range(0, len(predictions), batch_size):
            batch_predictions = predictions[i:i + batch_size]
            batch_ground_truth = ground_truth[i:i + batch_size]

            batch_tasks = []
            for pred, gt in zip(batch_predictions, batch_ground_truth):
                task = self.llm_judge.judge_explanation(
                    input_sample=gt.get('input_text', ''),
                    explanation=pred.get('explanation', ''),
                    prediction=pred.get('prediction', ''),
                    ground_truth_label=gt.get('label', ''),
                    ground_truth_explanation=gt.get('explanation', '')
                )
                batch_tasks.append(task)

            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        # Use default scores for failed evaluations
                        judge_results.append(ExplanationQualityScore(
                            overall_score=0.5,
                            technical_accuracy=0.5,
                            logical_consistency=0.5,
                            completeness=0.5,
                            clarity=0.5,
                            domain_relevance=0.5,
                            detailed_feedback="Evaluation failed"
                        ))
                    else:
                        judge_results.append(result)

            except Exception as e:
                print(f"Warning: Batch LLM evaluation failed: {e}")
                # Add default scores for this batch
                for _ in range(len(batch_tasks)):
                    judge_results.append(ExplanationQualityScore(
                        overall_score=0.5,
                        technical_accuracy=0.5,
                        logical_consistency=0.5,
                        completeness=0.5,
                        clarity=0.5,
                        domain_relevance=0.5,
                        detailed_feedback="Evaluation failed"
                    ))

        # Aggregate LLM judge results
        if judge_results:
            return {
                'avg_explanation_quality': sum(r.overall_score for r in judge_results) / len(judge_results),
                'technical_accuracy_score': sum(r.technical_accuracy for r in judge_results) / len(judge_results),
                'logical_consistency_score': sum(r.logical_consistency for r in judge_results) / len(judge_results),
                'completeness_score': sum(r.completeness for r in judge_results) / len(judge_results),
                'clarity_score': sum(r.clarity for r in judge_results) / len(judge_results),
                'domain_relevance_score': sum(r.domain_relevance for r in judge_results) / len(judge_results)
            }
        else:
            return {}

    async def _evaluate_with_automated_metrics(self, predictions: List[Dict[str, Any]],
                                             ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate explanations using automated metrics"""

        # Extract explanations and references
        candidate_explanations = [pred.get('explanation', '') for pred in predictions]
        reference_explanations = [gt.get('explanation', '') for gt in ground_truth]

        # Only evaluate if we have reference explanations
        if any(ref.strip() for ref in reference_explanations):
            return await self.automated_metrics.calculate_metrics(
                candidate_explanations, reference_explanations
            )
        else:
            # If no reference explanations, use basic metrics
            return await self.automated_metrics.calculate_intrinsic_metrics(candidate_explanations)

    async def _evaluate_domain_specific(self, predictions: List[Dict[str, Any]],
                                      ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate domain-specific aspects of explanations"""

        domain_results = {
            'explanation_consistency': 0.0,
            'ioc_accuracy': 0.0,
            'mitre_coverage': 0.0,
            'explanation_length_avg': 0.0
        }

        explanations = [pred.get('explanation', '') for pred in predictions]

        if not explanations:
            return domain_results

        # Calculate explanation consistency
        domain_results['explanation_consistency'] = self._calculate_explanation_consistency(
            predictions, ground_truth
        )

        # Analyze IoC (Indicators of Compromise) accuracy
        domain_results['ioc_accuracy'] = self._analyze_ioc_accuracy(predictions, ground_truth)

        # Check MITRE ATT&CK technique coverage
        domain_results['mitre_coverage'] = self._analyze_mitre_coverage(explanations)

        # Calculate average explanation length
        valid_explanations = [exp for exp in explanations if exp.strip()]
        if valid_explanations:
            avg_length = sum(len(exp.split()) for exp in valid_explanations) / len(valid_explanations)
            domain_results['explanation_length_avg'] = avg_length

        return domain_results

    def _calculate_explanation_consistency(self, predictions: List[Dict[str, Any]],
                                         ground_truth: List[Dict[str, Any]]) -> float:
        """Calculate consistency between explanations and predictions"""

        consistent_count = 0
        total_count = 0

        for pred, gt in zip(predictions, ground_truth):
            explanation = pred.get('explanation', '').lower()
            prediction = pred.get('prediction', '').lower()

            if not explanation:
                continue

            total_count += 1

            # Check if explanation supports the prediction
            if prediction == 'attack':
                # Look for attack-related keywords in explanation
                attack_keywords = ['attack', 'malicious', 'threat', 'intrusion', 'suspicious',
                                 'compromise', 'exploit', 'malware', 'breach', 'unauthorized']
                if any(keyword in explanation for keyword in attack_keywords):
                    consistent_count += 1
            elif prediction == 'benign':
                # Look for benign-related keywords or absence of attack keywords
                benign_keywords = ['normal', 'legitimate', 'benign', 'safe', 'expected', 'routine']
                attack_keywords = ['attack', 'malicious', 'threat', 'intrusion', 'suspicious']

                has_benign_keywords = any(keyword in explanation for keyword in benign_keywords)
                has_attack_keywords = any(keyword in explanation for keyword in attack_keywords)

                if has_benign_keywords or not has_attack_keywords:
                    consistent_count += 1

        return consistent_count / total_count if total_count > 0 else 0.0

    def _analyze_ioc_accuracy(self, predictions: List[Dict[str, Any]],
                            ground_truth: List[Dict[str, Any]]) -> float:
        """Analyze accuracy of Indicators of Compromise mentioned in explanations"""

        import re

        accurate_count = 0
        total_ioc_count = 0

        # Common IoC patterns
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        hash_pattern = r'\b[a-fA-F0-9]{32,64}\b'
        domain_pattern = r'\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b'

        for pred, gt in zip(predictions, ground_truth):
            explanation = pred.get('explanation', '')
            input_text = gt.get('input_text', '')

            if not explanation:
                continue

            # Extract IoCs from explanation
            explanation_ips = set(re.findall(ip_pattern, explanation))
            explanation_hashes = set(re.findall(hash_pattern, explanation))
            explanation_domains = set(re.findall(domain_pattern, explanation))

            # Extract IoCs from input
            input_ips = set(re.findall(ip_pattern, input_text))
            input_hashes = set(re.findall(hash_pattern, input_text))
            input_domains = set(re.findall(domain_pattern, input_text))

            # Check accuracy of mentioned IoCs
            for ioc_set, input_set in [
                (explanation_ips, input_ips),
                (explanation_hashes, input_hashes),
                (explanation_domains, input_domains)
            ]:
                for ioc in ioc_set:
                    total_ioc_count += 1
                    if ioc in input_set:
                        accurate_count += 1

        return accurate_count / total_ioc_count if total_ioc_count > 0 else 1.0

    def _analyze_mitre_coverage(self, explanations: List[str]) -> float:
        """Analyze coverage of MITRE ATT&CK techniques in explanations"""

        # Common MITRE ATT&CK techniques relevant to cybersecurity
        mitre_techniques = [
            'reconnaissance', 'initial access', 'execution', 'persistence',
            'privilege escalation', 'defense evasion', 'credential access',
            'discovery', 'lateral movement', 'collection', 'command and control',
            'exfiltration', 'impact', 'phishing', 'spearphishing', 'brute force',
            'sql injection', 'buffer overflow', 'code injection', 'dos'
        ]

        technique_mentions = 0
        total_explanations = len([exp for exp in explanations if exp.strip()])

        if total_explanations == 0:
            return 0.0

        for explanation in explanations:
            if not explanation.strip():
                continue

            explanation_lower = explanation.lower()
            for technique in mitre_techniques:
                if technique in explanation_lower:
                    technique_mentions += 1
                    break  # Count max one technique per explanation

        return technique_mentions / total_explanations

    def get_metric_names(self) -> List[str]:
        """Get list of metrics this evaluator produces"""
        return self.metric_names

    def get_required_prediction_fields(self) -> List[str]:
        """Get required fields in prediction data"""
        return ['explanation']

    def get_required_ground_truth_fields(self) -> List[str]:
        """Get required fields in ground truth data"""
        return []  # Ground truth explanation is optional

# src/benchmark/evaluation/explainability/llm_judge.py
import asyncio
import json
from typing import Dict, Any, Optional
from openai import AsyncOpenAI

class LLMJudgeEvaluator:
    """LLM-as-judge evaluator for explanation quality"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = AsyncOpenAI()
        self.rate_limiter = asyncio.Semaphore(10)  # Limit concurrent API calls

    async def judge_explanation(self, input_sample: str, explanation: str,
                              prediction: str, ground_truth_label: str,
                              ground_truth_explanation: str = "") -> ExplanationQualityScore:
        """Judge explanation quality using LLM"""

        async with self.rate_limiter:
            try:
                prompt = self._create_judge_prompt(
                    input_sample, explanation, prediction,
                    ground_truth_label, ground_truth_explanation
                )

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert cybersecurity analyst evaluating the quality of security analysis explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )

                result_text = response.choices[0].message.content
                return self._parse_judge_response(result_text)

            except Exception as e:
                print(f"LLM judge evaluation failed: {e}")
                return ExplanationQualityScore(
                    overall_score=0.5,
                    technical_accuracy=0.5,
                    logical_consistency=0.5,
                    completeness=0.5,
                    clarity=0.5,
                    domain_relevance=0.5,
                    detailed_feedback=f"Evaluation failed: {e}"
                )

    def _create_judge_prompt(self, input_sample: str, explanation: str,
                           prediction: str, ground_truth_label: str,
                           ground_truth_explanation: str = "") -> str:
        """Create prompt for LLM judge evaluation"""

        prompt = f"""
Evaluate the quality of this cybersecurity analysis explanation:

INPUT DATA:
{input_sample}

PREDICTED LABEL: {prediction}
CORRECT LABEL: {ground_truth_label}

EXPLANATION TO EVALUATE:
{explanation}
"""

        if ground_truth_explanation:
            prompt += f"""
REFERENCE EXPLANATION:
{ground_truth_explanation}
"""

        prompt += """
Please evaluate this explanation on the following criteria (score 0.0 to 1.0):

1. TECHNICAL ACCURACY: Are the cybersecurity concepts and terminology used correctly?
2. LOGICAL CONSISTENCY: Does the explanation logically support the prediction?
3. COMPLETENESS: Does the explanation cover the key aspects of the analysis?
4. CLARITY: Is the explanation clear and understandable?
5. DOMAIN RELEVANCE: Are cybersecurity-specific details appropriately included?

Respond in JSON format:
{
    "technical_accuracy": 0.0-1.0,
    "logical_consistency": 0.0-1.0,
    "completeness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "domain_relevance": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "feedback": "Brief explanation of the scores"
}
"""

        return prompt

    def _parse_judge_response(self, response_text: str) -> ExplanationQualityScore:
        """Parse LLM judge response into structured score"""

        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)

                return ExplanationQualityScore(
                    overall_score=float(scores.get('overall_score', 0.5)),
                    technical_accuracy=float(scores.get('technical_accuracy', 0.5)),
                    logical_consistency=float(scores.get('logical_consistency', 0.5)),
                    completeness=float(scores.get('completeness', 0.5)),
                    clarity=float(scores.get('clarity', 0.5)),
                    domain_relevance=float(scores.get('domain_relevance', 0.5)),
                    detailed_feedback=scores.get('feedback', 'No detailed feedback')
                )
            else:
                # Fallback: try to extract scores from text
                return self._extract_scores_from_text(response_text)

        except Exception as e:
            print(f"Failed to parse judge response: {e}")
            return ExplanationQualityScore(
                overall_score=0.5,
                technical_accuracy=0.5,
                logical_consistency=0.5,
                completeness=0.5,
                clarity=0.5,
                domain_relevance=0.5,
                detailed_feedback="Failed to parse judge response"
            )

    def _extract_scores_from_text(self, text: str) -> ExplanationQualityScore:
        """Extract scores from non-JSON text response"""

        import re

        # Try to find numeric scores in text
        score_patterns = {
            'technical_accuracy': r'technical\s*accuracy[:\s]*([0-9.]+)',
            'logical_consistency': r'logical\s*consistency[:\s]*([0-9.]+)',
            'completeness': r'completeness[:\s]*([0-9.]+)',
            'clarity': r'clarity[:\s]*([0-9.]+)',
            'domain_relevance': r'domain\s*relevance[:\s]*([0-9.]+)',
            'overall_score': r'overall[:\s]*([0-9.]+)'
        }

        scores = {}
        for metric, pattern in score_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize score to 0-1 range if needed
                    if score > 1.0 and score <= 10.0:
                        score /= 10.0
                    scores[metric] = max(0.0, min(1.0, score))
                except:
                    scores[metric] = 0.5
            else:
                scores[metric] = 0.5

        return ExplanationQualityScore(
            overall_score=scores.get('overall_score', 0.5),
            technical_accuracy=scores.get('technical_accuracy', 0.5),
            logical_consistency=scores.get('logical_consistency', 0.5),
            completeness=scores.get('completeness', 0.5),
            clarity=scores.get('clarity', 0.5),
            domain_relevance=scores.get('domain_relevance', 0.5),
            detailed_feedback="Scores extracted from text analysis"
        )

# src/benchmark/evaluation/explainability/automated_metrics.py
from typing import List, Dict, Any
import asyncio

class AutomatedMetricsEvaluator:
    """Automated metrics for explanation evaluation"""

    def __init__(self):
        # Import these only when needed to avoid dependency issues
        self.nltk_available = False
        self.bert_score_available = False

        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu
            from rouge_score import rouge_scorer
            self.nltk_available = True
        except ImportError:
            print("Warning: NLTK not available for BLEU/ROUGE scores")

        try:
            from bert_score import score as bert_score
            self.bert_score_available = True
        except ImportError:
            print("Warning: BERTScore not available")

    async def calculate_metrics(self, candidate_explanations: List[str],
                              reference_explanations: List[str]) -> Dict[str, float]:
        """Calculate automated metrics comparing candidates to references"""

        metrics = {}

        if not candidate_explanations or not reference_explanations:
            return metrics

        # Calculate BLEU scores
        if self.nltk_available:
            bleu_scores = await self._calculate_bleu_scores(candidate_explanations, reference_explanations)
            metrics.update(bleu_scores)

        # Calculate ROUGE scores
        if self.nltk_available:
            rouge_scores = await self._calculate_rouge_scores(candidate_explanations, reference_explanations)
            metrics.update(rouge_scores)

        # Calculate BERTScore
        if self.bert_score_available:
            bert_scores = await self._calculate_bert_scores(candidate_explanations, reference_explanations)
            metrics.update(bert_scores)

        return metrics

    async def calculate_intrinsic_metrics(self, explanations: List[str]) -> Dict[str, float]:
        """Calculate intrinsic metrics that don't require reference explanations"""

        metrics = {}

        if not explanations:
            return metrics

        # Calculate diversity metrics
        metrics.update(self._calculate_diversity_metrics(explanations))

        # Calculate length and structure metrics
        metrics.update(self._calculate_structural_metrics(explanations))

        return metrics

    async def _calculate_bleu_scores(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores"""

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

            bleu_scores = []
            smoothing = SmoothingFunction()

            for candidate, reference in zip(candidates, references):
                if not candidate.strip() or not reference.strip():
                    bleu_scores.append(0.0)
                    continue

                # Tokenize
                candidate_tokens = candidate.lower().split()
                reference_tokens = [reference.lower().split()]

                # Calculate BLEU score
                score = sentence_bleu(reference_tokens, candidate_tokens,
                                    smoothing_function=smoothing.method1)
                bleu_scores.append(score)

            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

            return {'bleu_score': avg_bleu}

        except Exception as e:
            print(f"BLEU calculation failed: {e}")
            return {}

    async def _calculate_rouge_scores(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""

        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for candidate, reference in zip(candidates, references):
                if not candidate.strip() or not reference.strip():
                    rouge1_scores.append(0.0)
                    rouge2_scores.append(0.0)
                    rougeL_scores.append(0.0)
                    continue

                scores = scorer.score(reference, candidate)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)

            return {
                'rouge1_score': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
                'rouge2_score': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
                'rouge_l_score': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
            }

        except Exception as e:
            print(f"ROUGE calculation failed: {e}")
            return {}

    async def _calculate_bert_scores(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore"""

        try:
            from bert_score import score

            # Filter out empty strings
            valid_pairs = [(c, r) for c, r in zip(candidates, references)
                          if c.strip() and r.strip()]

            if not valid_pairs:
                return {'bert_score': 0.0}

            valid_candidates = [pair[0] for pair in valid_pairs]
            valid_references = [pair[1] for pair in valid_pairs]

            # Calculate BERTScore
            P, R, F1 = score(valid_candidates, valid_references, lang='en', verbose=False)

            avg_bert_score = F1.mean().item()

            return {'bert_score': avg_bert_score}

        except Exception as e:
            print(f"BERTScore calculation failed: {e}")
            return {}

    def _calculate_diversity_metrics(self, explanations: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for explanations"""

        if not explanations:
            return {}

        # Remove empty explanations
        valid_explanations = [exp.strip().lower() for exp in explanations if exp.strip()]

        if not valid_explanations:
            return {}

        # Calculate uniqueness ratio
        unique_explanations = set(valid_explanations)
        uniqueness_ratio = len(unique_explanations) / len(valid_explanations)

        # Calculate vocabulary diversity
        all_words = []
        for explanation in valid_explanations:
            all_words.extend(explanation.split())

        if all_words:
            unique_words = set(all_words)
            vocabulary_diversity = len(unique_words) / len(all_words)
        else:
            vocabulary_diversity = 0.0

        return {
            'explanation_uniqueness': uniqueness_ratio,
            'vocabulary_diversity': vocabulary_diversity
        }

    def _calculate_structural_metrics(self, explanations: List[str]) -> Dict[str, float]:
        """Calculate structural metrics for explanations"""

        if not explanations:
            return {}

        valid_explanations = [exp.strip() for exp in explanations if exp.strip()]

        if not valid_explanations:
            return {}

        # Calculate average length in words
        word_counts = [len(exp.split()) for exp in valid_explanations]
        avg_length = sum(word_counts) / len(word_counts)

        # Calculate average sentence count
        sentence_counts = [len([s for s in exp.split('.') if s.strip()]) for exp in valid_explanations]
        avg_sentences = sum(sentence_counts) / len(sentence_counts)

        # Calculate complexity score (based on average word length)
        word_lengths = []
        for exp in valid_explanations:
            words = exp.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                word_lengths.append(avg_word_length)

        complexity_score = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0

        return {
            'avg_explanation_length': avg_length,
            'avg_sentences_per_explanation': avg_sentences,
            'explanation_complexity': complexity_score / 10.0  # Normalize to 0-1 range
        }
```

✅ **Tests**:
Create `tests/unit/test_explainability_evaluator.py`:
- Test LLM-as-judge evaluation with mock API responses
- Test automated metrics calculation
- Test domain-specific analysis
- Test consistency checking
- Test error handling for failed API calls

Create `tests/integration/test_explainability_integration.py`:
- Test with actual LLM API calls (if available)
- Test with various explanation formats
- Test performance with large datasets

🔍 **Validation**:
- LLM-as-judge provides meaningful quality scores
- Automated metrics (BLEU, ROUGE, BERTScore) calculate correctly
- Domain-specific analysis identifies cybersecurity concepts
- Consistency checking works between explanations and predictions
- Error handling gracefully manages API failures
- All metrics produce values in expected ranges

#### Prompt 7.1.2: Integrate Explainability Evaluator with Evaluation Service
🎯 **Goal**: Integrate the explainability evaluator with the main evaluation service

📁 **Files**: Modify `src/benchmark/services/evaluation_service.py`

🔧 **Task**:
Integrate the explainability evaluator with the existing evaluation service and add it to the MetricType enum.

Requirements:
- Add EXPLAINABILITY to MetricType enum
- Register explainability evaluator in evaluation service
- Update validation to handle explainability-specific requirements
- Add explainability-specific configuration options
- Ensure explainability evaluation works with existing workflow

```python
# Add to interfaces/evaluation_interfaces.py:
class MetricType(Enum):
    ACCURACY = "accuracy"
    EXPLAINABILITY = "explainability"  # Add this line
    PERFORMANCE = "performance"
    FALSE_POSITIVE_RATE = "false_positive_rate"

# Modify services/evaluation_service.py:
async def _register_default_evaluators(self) -> None:
    """Register all available metric evaluators"""
    from benchmark.evaluation.metrics.accuracy import AccuracyEvaluator
    from benchmark.evaluation.metrics.performance import PerformanceEvaluator
    from benchmark.evaluation.metrics.false_positive_rate import FalsePositiveRateEvaluator
    from benchmark.evaluation.metrics.explainability import ExplainabilityEvaluator

    self.evaluators[MetricType.ACCURACY] = AccuracyEvaluator()
    self.evaluators[MetricType.PERFORMANCE] = PerformanceEvaluator()
    self.evaluators[MetricType.FALSE_POSITIVE_RATE] = FalsePositiveRateEvaluator()
    self.evaluators[MetricType.EXPLAINABILITY] = ExplainabilityEvaluator()

    # Initialize results storage
    from benchmark.storage.results_storage import ResultsStorage
    self.results_storage = ResultsStorage()
    await self.results_storage.initialize()

    # Initialize evaluation cache
    self.evaluation_cache = EvaluationCache()

# Add configuration support for explainability evaluation
class ExplainabilityConfig:
    """Configuration for explainability evaluation"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        config = config_dict or {}

        self.judge_model = config.get('judge_model', 'gpt-4o-mini')
        self.use_reference_explanations = config.get('use_reference_explanations', True)
        self.automated_metrics = config.get('automated_metrics', ['bleu', 'rouge', 'bertscore'])
        self.domain_analysis = config.get('domain_analysis', True)
        self.batch_size = config.get('batch_size', 10)  # For LLM-as-judge API calls
        self.timeout_seconds = config.get('timeout_seconds', 300)

# Update the _validate_evaluation_data method to handle explainability requirements:
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
                            # Special handling for explainability - explanation field is critical
                            if metric_type == MetricType.EXPLAINABILITY and field == 'explanation':
                                explanation = pred.get('explanation', pred.get('reasoning', pred.get('rationale', '')))
                                if not explanation:
                                    return ServiceResponse(
                                        success=False,
                                        error=f"Explainability evaluation requires 'explanation' field in prediction {i}"
                                    )
                            else:
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

# Add explainability-specific evaluation method:
async def evaluate_explainability_detailed(self, request: EvaluationRequest,
                                          config: ExplainabilityConfig = None) -> Dict[str, Any]:
    """Perform detailed explainability evaluation with custom configuration"""

    if MetricType.EXPLAINABILITY not in request.metrics:
        raise ValueError("Explainability metric not requested")

    # Use custom configuration if provided
    if config:
        # Temporarily update the evaluator configuration
        explainability_evaluator = self.evaluators[MetricType.EXPLAINABILITY]
        original_judge_model = explainability_evaluator.llm_judge.model

        try:
            # Update configuration
            explainability_evaluator.llm_judge.model = config.judge_model

            # Run evaluation
            result = await explainability_evaluator.evaluate(request.predictions, request.ground_truth)

            # Add configuration metadata
            result['evaluation_config'] = {
                'judge_model': config.judge_model,
                'automated_metrics': config.automated_metrics,
                'domain_analysis': config.domain_analysis,
                'batch_size': config.batch_size
            }

            return result

        finally:
            # Restore original configuration
            explainability_evaluator.llm_judge.model = original_judge_model
    else:
        # Use default configuration
        explainability_evaluator = self.evaluators[MetricType.EXPLAINABILITY]
        return await explainability_evaluator.evaluate(request.predictions, request.ground_truth)

# Add method to generate explainability report:
async def generate_explainability_report(self, experiment_id: str) -> Dict[str, Any]:
    """Generate detailed explainability report for experiment"""

    # Get explainability results
    results = await self.results_storage.get_evaluation_results(
        experiment_id=experiment_id,
        metric_type='explainability'
    )

    if not results:
        return {'error': 'No explainability results found for experiment'}

    # Organize results by model
    model_results = {}
    for result in results:
        model_id = result.get('model_id', 'unknown')
        metric_name = result.get('metric_name', '')
        metric_value = result.get('value', 0)

        if model_id not in model_results:
            model_results[model_id] = {}

        model_results[model_id][metric_name] = metric_value

    # Generate insights
    insights = []

    # Overall explanation quality insights
    quality_scores = [result.get('value', 0) for result in results
                     if result.get('metric_name') == 'avg_explanation_quality']

    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        if avg_quality > 0.8:
            insights.append(f"Excellent explanation quality achieved (avg: {avg_quality:.3f})")
        elif avg_quality > 0.6:
            insights.append(f"Good explanation quality achieved (avg: {avg_quality:.3f})")
        else:
            insights.append(f"Explanation quality needs improvement (avg: {avg_quality:.3f})")

    # Technical accuracy insights
    accuracy_scores = [result.get('value', 0) for result in results
                      if result.get('metric_name') == 'technical_accuracy_score']

    if accuracy_scores:
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        if avg_accuracy < 0.6:
            insights.append(f"Low technical accuracy in explanations ({avg_accuracy:.3f}) - review domain knowledge")

    # Consistency insights
    consistency_scores = [result.get('value', 0) for result in results
                         if result.get('metric_name') == 'explanation_consistency']

    if consistency_scores:
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        if avg_consistency < 0.7:
            insights.append(f"Low explanation consistency ({avg_consistency:.3f}) - predictions don't align with explanations")

    return {
        'model_results': model_results,
        'insights': insights,
        'summary_statistics': {
            'models_evaluated': len(model_results),
            'total_metrics': len(results),
            'avg_explanation_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0
        }
    }
```

✅ **Tests**:
Create `tests/integration/test_explainability_service_integration.py`:
- Test explainability evaluator registration
- Test explainability evaluation through the main service
- Test custom explainability configuration
- Test explainability report generation
- Test integration with workflow steps

🔍 **Validation**:
- Explainability evaluator is properly registered in evaluation service
- Explainability metrics are calculated and stored correctly
- Custom explainability configuration works
- Explainability reports provide useful insights
- Integration with existing workflow doesn't break other metrics

#### Prompt 7.1.3: Create Advanced Explainability Analysis Features
🎯 **Goal**: Add advanced analysis features for explainability evaluation

📁 **Files**:
- `src/benchmark/evaluation/explainability/advanced_analysis.py`
- `src/benchmark/evaluation/explainability/explanation_templates.py`

🔧 **Task**:
Create advanced analysis features that provide deeper insights into explanation quality and generate explanation templates for different attack types.

Requirements:
- Explanation clustering and pattern analysis
- Template generation for different cybersecurity scenarios
- Comparative analysis across models
- Explanation improvement suggestions
- Advanced statistical analysis of explanation quality

```python
# src/benchmark/evaluation/explainability/advanced_analysis.py
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

@dataclass
class ExplanationCluster:
    cluster_id: int
    pattern_description: str
    example_explanations: List[str]
    common_phrases: List[str]
    quality_score: float
    frequency: int

@dataclass
class ModelComparisonResult:
    model_a: str
    model_b: str
    quality_difference: float
    consistency_difference: float
    technical_accuracy_difference: float
    better_model: str
    statistical_significance: float

class AdvancedExplainabilityAnalyzer:
    """Advanced analysis tools for explainability evaluation"""

    def __init__(self):
        self.attack_type_keywords = {
            'malware': ['malware', 'virus', 'trojan', 'ransomware', 'backdoor', 'rootkit'],
            'intrusion': ['intrusion', 'penetration', 'unauthorized', 'breach', 'infiltration'],
            'dos': ['denial of service', 'dos', 'ddos', 'flooding', 'overload'],
            'phishing': ['phishing', 'spoofing', 'social engineering', 'credential theft'],
            'reconnaissance': ['reconnaissance', 'scanning', 'enumeration', 'discovery'],
            'data_exfiltration': ['exfiltration', 'data theft', 'data extraction', 'information gathering']
        }

    def analyze_explanation_patterns(self, predictions: List[Dict[str, Any]],
                                   ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in explanations across the dataset"""

        explanations = [pred.get('explanation', '') for pred in predictions]
        labels = [gt.get('label', '') for gt in ground_truth]

        analysis_results = {}

        # Pattern analysis by attack type
        analysis_results['attack_type_patterns'] = self._analyze_attack_type_patterns(
            explanations, labels, predictions
        )

        # Clustering analysis
        analysis_results['explanation_clusters'] = self._cluster_explanations(explanations)

        # Quality distribution analysis
        analysis_results['quality_distribution'] = self._analyze_quality_distribution(
            explanations, predictions
        )

        # Common issues identification
        analysis_results['common_issues'] = self._identify_common_issues(explanations, predictions)

        return analysis_results

    def _analyze_attack_type_patterns(self, explanations: List[str], labels: List[str],
                                    predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze explanation patterns for different attack types"""

        attack_type_analysis = {}

        # Group by attack type
        attack_groups = defaultdict(list)
        for i, (exp, label, pred) in enumerate(zip(explanations, labels, predictions)):
            attack_type = pred.get('attack_type', 'unknown')
            if label == 'ATTACK' and attack_type != 'unknown':
                attack_groups[attack_type].append(exp)

        # Analyze each attack type
        for attack_type, type_explanations in attack_groups.items():
            if len(type_explanations) < 2:  # Need at least 2 examples
                continue

            # Find common phrases
            common_phrases = self._find_common_phrases(type_explanations)

            # Calculate average quality metrics
            avg_length = np.mean([len(exp.split()) for exp in type_explanations])

            # Check keyword coverage
            relevant_keywords = self.attack_type_keywords.get(attack_type, [])
            keyword_coverage = self._calculate_keyword_coverage(type_explanations, relevant_keywords)

            attack_type_analysis[attack_type] = {
                'sample_count': len(type_explanations),
                'avg_explanation_length': avg_length,
                'common_phrases': common_phrases,
                'keyword_coverage': keyword_coverage,
                'example_explanation': type_explanations[0] if type_explanations else ''
            }

        return attack_type_analysis

    def _cluster_explanations(self, explanations: List[str]) -> List[ExplanationCluster]:
        """Cluster explanations by similarity"""

        # Simple clustering based on common phrases and length
        clusters = []
        processed_indices = set()

        for i, explanation in enumerate(explanations):
            if i in processed_indices or not explanation.strip():
                continue

            # Find similar explanations
            cluster_explanations = [explanation]
            cluster_indices = {i}

            for j, other_explanation in enumerate(explanations[i+1:], start=i+1):
                if j in processed_indices:
                    continue

                similarity = self._calculate_text_similarity(explanation, other_explanation)
                if similarity > 0.6:  # Similarity threshold
                    cluster_explanations.append(other_explanation)
                    cluster_indices.add(j)

            if len(cluster_explanations) >= 2:  # Only create clusters with multiple items
                processed_indices.update(cluster_indices)

                # Generate cluster description
                common_phrases = self._find_common_phrases(cluster_explanations)
                pattern_description = f"Pattern with phrases: {', '.join(common_phrases[:3])}"

                cluster = ExplanationCluster(
                    cluster_id=len(clusters),
                    pattern_description=pattern_description,
                    example_explanations=cluster_explanations[:3],  # Top 3 examples
                    common_phrases=common_phrases,
                    quality_score=self._estimate_cluster_quality(cluster_explanations),
                    frequency=len(cluster_explanations)
                )

                clusters.append(cluster)

        # Sort by frequency
        clusters.sort(key=lambda c: c.frequency, reverse=True)

        return clusters[:10]  # Return top 10 clusters

    def _analyze_quality_distribution(self, explanations: List[str],
                                    predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of explanation quality metrics"""

        lengths = [len(exp.split()) for exp in explanations if exp.strip()]

        quality_distribution = {
            'length_statistics': {
                'mean': np.mean(lengths) if lengths else 0,
                'std': np.std(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'percentiles': {
                    '25': np.percentile(lengths, 25) if lengths else 0,
                    '50': np.percentile(lengths, 50) if lengths else 0,
                    '75': np.percentile(lengths, 75) if lengths else 0
                }
            }
        }

        # Analyze explanation completeness
        complete_explanations = 0
        incomplete_explanations = 0

        for explanation in explanations:
            if not explanation.strip():
                incomplete_explanations += 1
                continue

            # Check for completeness indicators
            completeness_indicators = ['because', 'due to', 'indicates', 'shows', 'evidence']
            if any(indicator in explanation.lower() for indicator in completeness_indicators):
                complete_explanations += 1
            else:
                incomplete_explanations += 1

        quality_distribution['completeness_analysis'] = {
            'complete_explanations': complete_explanations,
            'incomplete_explanations': incomplete_explanations,
            'completeness_ratio': complete_explanations / (complete_explanations + incomplete_explanations)
                                 if (complete_explanations + incomplete_explanations) > 0 else 0
        }

        return quality_distribution

    def _identify_common_issues(self, explanations: List[str],
                               predictions: List[Dict[str, Any]]) -> List[str]:
        """Identify common issues in explanations"""

        issues = []

        # Check for empty explanations
        empty_count = sum(1 for exp in explanations if not exp.strip())
        if empty_count > len(explanations) * 0.1:  # More than 10% empty
            issues.append(f"High number of empty explanations ({empty_count}/{len(explanations)})")

        # Check for very short explanations
        short_count = sum(1 for exp in explanations if len(exp.split()) < 5)
        if short_count > len(explanations) * 0.2:  # More than 20% very short
            issues.append(f"Many explanations are too short ({short_count} with <5 words)")

        # Check for repetitive explanations
        unique_explanations = set(exp.strip().lower() for exp in explanations if exp.strip())
        if len(unique_explanations) < len(explanations) * 0.5:  # Less than 50% unique
            issues.append("Low explanation diversity - many explanations are repetitive")

        # Check for lack of technical terms
        technical_terms = ['malware', 'intrusion', 'vulnerability', 'exploit', 'attack', 'threat']
        explanations_with_tech_terms = sum(
            1 for exp in explanations
            if any(term in exp.lower() for term in technical_terms)
        )

        if explanations_with_tech_terms < len(explanations) * 0.3:  # Less than 30% with technical terms
            issues.append("Explanations lack cybersecurity technical terminology")

        return issues

    def compare_model_explanations(self, model_a_predictions: List[Dict[str, Any]],
                                 model_b_predictions: List[Dict[str, Any]],
                                 ground_truth: List[Dict[str, Any]],
                                 model_a_name: str = "Model A",
                                 model_b_name: str = "Model B") -> ModelComparisonResult:
        """Compare explanation quality between two models"""

        # Extract explanations
        explanations_a = [pred.get('explanation', '') for pred in model_a_predictions]
        explanations_b = [pred.get('explanation', '') for pred in model_b_predictions]

        # Calculate quality metrics for each model
        quality_a = self._calculate_avg_explanation_quality(explanations_a)
        quality_b = self._calculate_avg_explanation_quality(explanations_b)

        # Calculate consistency (simplified)
        consistency_a = self._calculate_explanation_consistency(model_a_predictions, ground_truth)
        consistency_b = self._calculate_explanation_consistency(model_b_predictions, ground_truth)

        # Technical accuracy (simplified analysis)
        tech_accuracy_a = self._estimate_technical_accuracy(explanations_a)
        tech_accuracy_b = self._estimate_technical_accuracy(explanations_b)

        # Determine better model
        quality_diff = quality_b - quality_a
        consistency_diff = consistency_b - consistency_a
        tech_diff = tech_accuracy_b - tech_accuracy_a

        # Weighted score (quality: 40%, consistency: 30%, technical: 30%)
        overall_diff = 0.4 * quality_diff + 0.3 * consistency_diff + 0.3 * tech_diff

        better_model = model_b_name if overall_diff > 0 else model_a_name

        # Simple statistical significance (would need proper testing in practice)
        significance = abs(overall_diff)

        return ModelComparisonResult(
            model_a=model_a_name,
            model_b=model_b_name,
            quality_difference=quality_diff,
            consistency_difference=consistency_diff,
            technical_accuracy_difference=tech_diff,
            better_model=better_model,
            statistical_significance=significance
        )

    def generate_improvement_suggestions(self, predictions: List[Dict[str, Any]],
                                       analysis_results: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving explanation quality"""

        suggestions = []

        # Analyze common issues
        common_issues = analysis_results.get('common_issues', [])

        for issue in common_issues:
            if "empty explanations" in issue:
                suggestions.append("Ensure all models provide explanations for their predictions")
            elif "too short" in issue:
                suggestions.append("Encourage more detailed explanations with specific reasoning")
            elif "repetitive" in issue:
                suggestions.append("Improve explanation diversity by incorporating more contextual details")
            elif "technical terminology" in issue:
                suggestions.append("Include more cybersecurity-specific technical terms and concepts")

        # Analyze quality distribution
        quality_dist = analysis_results.get('quality_distribution', {})
        length_stats = quality_dist.get('length_statistics', {})

        if length_stats.get('mean', 0) < 10:  # Very short explanations
            suggestions.append("Increase average explanation length to provide more comprehensive reasoning")

        completeness = quality_dist.get('completeness_analysis', {})
        if completeness.get('completeness_ratio', 0) < 0.5:
            suggestions.append("Improve explanation completeness by including causal reasoning (because, due to, etc.)")

        # Analyze attack type patterns
        attack_patterns = analysis_results.get('attack_type_patterns', {})

        for attack_type, pattern_info in attack_patterns.items():
            keyword_coverage = pattern_info.get('keyword_coverage', 0)
            if keyword_coverage < 0.3:  # Low keyword coverage
                relevant_keywords = self.attack_type_keywords.get(attack_type, [])
                suggestions.append(f"For {attack_type} attacks, include more specific terms like: {', '.join(relevant_keywords[:3])}")

        return suggestions

    # Helper methods
    def _find_common_phrases(self, texts: List[str]) -> List[str]:
        """Find common phrases across texts"""

        # Simple n-gram analysis
        all_phrases = []
        for text in texts:
            words = text.lower().split()
            # Extract 2-grams and 3-grams
            for i in range(len(words) - 1):
                all_phrases.append(' '.join(words[i:i+2]))
            for i in range(len(words) - 2):
                all_phrases.append(' '.join(words[i:i+3]))

        # Count phrase frequency
        phrase_counts = Counter(all_phrases)

        # Return most common phrases that appear in at least 2 texts
        common_phrases = [phrase for phrase, count in phrase_counts.most_common(10)
                         if count >= 2]

        return common_phrases

    def _calculate_keyword_coverage(self, explanations: List[str], keywords: List[str]) -> float:
        """Calculate coverage of relevant keywords in explanations"""

        if not keywords or not explanations:
            return 0.0

        coverage_count = 0
        for explanation in explanations:
            explanation_lower = explanation.lower()
            if any(keyword in explanation_lower for keyword in keywords):
                coverage_count += 1

        return coverage_count / len(explanations)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _estimate_cluster_quality(self, explanations: List[str]) -> float:
        """Estimate quality of explanation cluster"""

        # Simple quality estimation based on length and diversity
        avg_length = np.mean([len(exp.split()) for exp in explanations])
        length_score = min(1.0, avg_length / 15.0)  # Normalize to 0-1

        # Diversity within cluster
        unique_words = set()
        for exp in explanations:
            unique_words.update(exp.lower().split())

        total_words = sum(len(exp.split()) for exp in explanations)
        diversity_score = len(unique_words) / total_words if total_words > 0 else 0

        return (length_score + diversity_score) / 2

    def _calculate_avg_explanation_quality(self, explanations: List[str]) -> float:
        """Calculate average explanation quality (simplified)"""

        if not explanations:
            return 0.0

        quality_scores = []
        for explanation in explanations:
            if not explanation.strip():
                quality_scores.append(0.0)
                continue

            # Simple quality factors
            length_score = min(1.0, len(explanation.split()) / 20.0)
            completeness_score = 1.0 if any(word in explanation.lower()
                                          for word in ['because', 'due to', 'indicates']) else 0.5
            technical_score = 1.0 if any(term in explanation.lower()
                                       for term in ['malware', 'attack', 'threat', 'intrusion']) else 0.5

            quality = (length_score + completeness_score + technical_score) / 3
            quality_scores.append(quality)

        return sum(quality_scores) / len(quality_scores)

    def _calculate_explanation_consistency(self, predictions: List[Dict[str, Any]],
                                         ground_truth: List[Dict[str, Any]]) -> float:
        """Calculate explanation consistency with predictions"""

        consistent_count = 0
        total_count = 0

        for pred, gt in zip(predictions, ground_truth):
            explanation = pred.get('explanation', '').lower()
            prediction = pred.get('prediction', '').lower()

            if not explanation:
                continue

            total_count += 1

            if prediction == 'attack':
                attack_keywords = ['attack', 'malicious', 'threat', 'suspicious']
                if any(keyword in explanation for keyword in attack_keywords):
                    consistent_count += 1
            elif prediction == 'benign':
                benign_keywords = ['normal', 'legitimate', 'benign', 'safe']
                attack_keywords = ['attack', 'malicious', 'threat']

                has_benign = any(keyword in explanation for keyword in benign_keywords)
                has_attack = any(keyword in explanation for keyword in attack_keywords)

                if has_benign or not has_attack:
                    consistent_count += 1

        return consistent_count / total_count if total_count > 0 else 0.0

    def _estimate_technical_accuracy(self, explanations: List[str]) -> float:
        """Estimate technical accuracy of explanations"""

        if not explanations:
            return 0.0

        technical_scores = []

        for explanation in explanations:
            if not explanation.strip():
                technical_scores.append(0.0)
                continue

            # Count technical terms
            technical_terms = [
                'malware', 'virus', 'trojan', 'ransomware', 'intrusion', 'vulnerability',
                'exploit', 'payload', 'backdoor', 'phishing', 'dos', 'ddos', 'injection',
                'buffer overflow', 'privilege escalation', 'lateral movement'
            ]

            explanation_lower = explanation.lower()
            technical_count = sum(1 for term in technical_terms if term in explanation_lower)

            # Normalize score
            score = min(1.0, technical_count / 3.0)  # Max score at 3+ technical terms
            technical_scores.append(score)

        return sum(technical_scores) / len(technical_scores)

# src/benchmark/evaluation/explainability/explanation_templates.py
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ExplanationTemplate:
    attack_type: str
    template: str
    required_elements: List[str]
    example_explanation: str

class ExplanationTemplateGenerator:
    """Generate explanation templates for different cybersecurity scenarios"""

    def __init__(self):
        self.templates = {
            'malware': ExplanationTemplate(
                attack_type='malware',
                template="Detected {malware_type} based on {indicators}. The file/process shows {behavioral_patterns} which indicates {threat_level} threat.",
                required_elements=['malware_type', 'indicators', 'behavioral_patterns', 'threat_level'],
                example_explanation="Detected trojan malware based on suspicious file hash and network connections. The file shows encrypted communications and registry modifications which indicates high threat."
            ),
            'intrusion': ExplanationTemplate(
                attack_type='intrusion',
                template="Unauthorized access detected through {attack_vector}. The activity shows {access_patterns} attempting to {objective}.",
                required_elements=['attack_vector', 'access_patterns', 'objective'],
                example_explanation="Unauthorized access detected through brute force login attempts. The activity shows multiple failed authentication attempts from suspicious IP attempting to gain system access."
            ),
            'dos': ExplanationTemplate(
                attack_type='dos',
                template="Denial of Service attack identified via {dos_method}. Traffic analysis shows {traffic_patterns} resulting in {impact}.",
                required_elements=['dos_method', 'traffic_patterns', 'impact'],
                example_explanation="Denial of Service attack identified via TCP flood. Traffic analysis shows abnormally high request volume from multiple sources resulting in service degradation."
            ),
            'phishing': ExplanationTemplate(
                attack_type='phishing',
                template="Phishing attempt detected in {medium} using {deception_method}. The content contains {suspicious_elements} designed to {goal}.",
                required_elements=['medium', 'deception_method', 'suspicious_elements', 'goal'],
                example_explanation="Phishing attempt detected in email using domain spoofing. The content contains suspicious links and urgent language designed to steal credentials."
            )
        }

    def generate_template_for_attack(self, attack_type: str) -> ExplanationTemplate:
        """Get explanation template for specific attack type"""
        return self.templates.get(attack_type.lower(), self.templates.get('general'))

    def evaluate_explanation_against_template(self, explanation: str, attack_type: str) -> Dict[str, Any]:
        """Evaluate explanation against appropriate template"""

        template = self.generate_template_for_attack(attack_type)
        if not template:
            return {'score': 0.5, 'missing_elements': [], 'feedback': 'No template available'}

        explanation_lower = explanation.lower()

        # Check for required elements
        present_elements = []
        missing_elements = []

        for element in template.required_elements:
            # Simple keyword matching (in practice, would use more sophisticated NLP)
            element_keywords = self._get_element_keywords(element)

            if any(keyword in explanation_lower for keyword in element_keywords):
                present_elements.append(element)
            else:
                missing_elements.append(element)

        # Calculate score
        score = len(present_elements) / len(template.required_elements) if template.required_elements else 0.5

        # Generate feedback
        feedback = []
        if missing_elements:
            feedback.append(f"Missing elements: {', '.join(missing_elements)}")
        if present_elements:
            feedback.append(f"Present elements: {', '.join(present_elements)}")

        return {
            'score': score,
            'present_elements': present_elements,
            'missing_elements': missing_elements,
            'feedback': '; '.join(feedback),
            'template_used': template.attack_type
        }

    def _get_element_keywords(self, element: str) -> List[str]:
        """Get keywords associated with template elements"""

        element_keywords = {
            'malware_type': ['malware', 'virus', 'trojan', 'ransomware', 'spyware', 'adware'],
            'indicators': ['hash', 'signature', 'behavior', 'network', 'file', 'process'],
            'behavioral_patterns': ['communication', 'modification', 'execution', 'persistence'],
            'threat_level': ['high', 'medium', 'low', 'critical', 'severe'],
            'attack_vector': ['brute force', 'sql injection', 'buffer overflow', 'social engineering'],
            'access_patterns': ['login', 'authentication', 'credential', 'session'],
            'objective': ['access', 'data', 'system', 'privilege', 'information'],
            'dos_method': ['flood', 'amplification', 'exhaustion', 'volumetric'],
            'traffic_patterns': ['volume', 'frequency', 'source', 'bandwidth'],
            'impact': ['degradation', 'unavailable', 'slow', 'blocked'],
            'medium': ['email', 'web', 'sms', 'social media'],
            'deception_method': ['spoofing', 'impersonation', 'fake', 'mimicking'],
            'suspicious_elements': ['link', 'attachment', 'domain', 'urgency'],
            'goal': ['steal', 'credential', 'information', 'money', 'data']
        }

        return element_keywords.get(element, [element])

    def get_all_templates(self) -> Dict[str, ExplanationTemplate]:
        """Get all available explanation templates"""
        return self.templates.copy()

    def add_custom_template(self, template: ExplanationTemplate) -> None:
        """Add custom explanation template"""
        self.templates[template.attack_type] = template
```

✅ **Tests**:
Create `tests/unit/test_advanced_explainability.py`:
- Test explanation pattern analysis
- Test clustering functionality
- Test model comparison features
- Test template generation and evaluation
- Test improvement suggestion generation

🔍 **Validation**:
- Pattern analysis identifies meaningful explanation patterns
- Clustering groups similar explanations effectively
- Model comparison provides actionable insights
- Templates help evaluate explanation completeness
- Improvement suggestions are practical and helpful

---

This completes development_prompts_3.md covering weeks 9-12 of the project. The document provides 15 detailed, sequential prompts that cover:

**Phase 6 - Orchestration Service (Weeks 9-10):**
- Complete workflow orchestration coordinating all services
- Concrete workflow steps for data, model, evaluation, and results
- Comprehensive integration testing and error handling

**Phase 7 - Advanced Explainability Metrics (Weeks 10-12):**
- Comprehensive explainability evaluation using LLM-as-judge
- Automated metrics (BLEU, ROUGE, BERTScore) for comparison
- Advanced analysis features including pattern recognition and model comparison
- Template-based evaluation for different cybersecurity scenarios

## Key Features Implemented:

1. **Complete Orchestration System**: Manages entire experiment lifecycle with proper service coordination
2. **Advanced Explainability Evaluation**: Multi-faceted approach combining LLM judgment with automated metrics
3. **Sophisticated Analysis**: Pattern recognition, clustering, and comparative analysis of explanations
4. **Template-Based Evaluation**: Structured approach to evaluating explanation completeness
5. **Production-Ready Error Handling**: Robust error recovery and graceful degradation

The prompts maintain the same high-quality standards with comprehensive testing, clear validation criteria, and AI-assistant friendly structure. This positions the project perfectly for the final implementation phases covering API Gateway, CLI Interface, Reporting Service, and Final Integration.

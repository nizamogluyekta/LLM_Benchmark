"""Comprehensive integration tests for the complete orchestration service."""

import asyncio
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmark.core.base import ServiceResponse
from benchmark.interfaces.orchestration_interfaces import ExperimentStatus
from benchmark.services.orchestration_service import OrchestrationService


class TestOrchestrationServiceComplete:
    """Comprehensive integration tests for Orchestration Service."""

    @pytest.fixture
    async def orchestration_service(self):
        """Fully configured orchestration service."""
        service = OrchestrationService()

        # Mock all dependent services for integration testing
        service.services = {
            "config": AsyncMock(),
            "data": AsyncMock(),
            "model": AsyncMock(),
            "evaluation": AsyncMock(),
        }

        # Initialize the service
        for service_name, mock_service in service.services.items():
            mock_service.initialize.return_value = ServiceResponse(
                success=True, message=f"{service_name} initialized"
            )
            mock_service.health_check.return_value = MagicMock(
                status=MagicMock(value="healthy"), message=f"{service_name} healthy"
            )
            mock_service.shutdown.return_value = ServiceResponse(
                success=True, message=f"{service_name} shutdown"
            )

        # Initialize workflow engine
        await service.workflow_engine.initialize(service.services)
        service._initialized = True

        yield service

        # Cleanup
        with contextlib.suppress(Exception):
            await service.shutdown()

    @pytest.fixture
    def sample_experiment_config(self):
        """Sample experiment configuration for testing."""
        return {
            "name": "Integration Test Experiment",
            "description": "Testing complete orchestration workflow",
            "datasets": [
                {
                    "id": "test_dataset_1",
                    "loader_type": "local",
                    "config": {"path": "test://sample_data.json", "max_samples": 100},
                    "preprocessing": ["normalize"],
                }
            ],
            "models": [
                {
                    "id": "test_model_1",
                    "plugin_type": "mlx_local",
                    "config": {"path": "test://mock_model", "max_tokens": 256, "temperature": 0.1},
                }
            ],
            "evaluation": {
                "metrics": ["accuracy", "performance"],
                "parallel_jobs": 2,
                "timeout_minutes": 30,
            },
        }

    @pytest.fixture
    def multi_model_config(self):
        """Multi-model experiment configuration."""
        return {
            "name": "Multi-Model Comparison",
            "description": "Compare multiple models on cybersecurity tasks",
            "datasets": [
                {
                    "id": "network_logs",
                    "loader_type": "local",
                    "config": {"path": "test://network_data.json", "max_samples": 500},
                },
                {
                    "id": "email_data",
                    "loader_type": "local",
                    "config": {"path": "test://email_data.json", "max_samples": 300},
                },
            ],
            "models": [
                {
                    "id": "cybersec_bert",
                    "plugin_type": "huggingface",
                    "config": {"path": "test://cybersec_bert_model", "max_tokens": 512},
                },
                {
                    "id": "llama_3b",
                    "plugin_type": "mlx_local",
                    "config": {"path": "test://llama_3b_model", "max_tokens": 256},
                },
                {
                    "id": "gpt_4o_mini",
                    "plugin_type": "openai_api",
                    "config": {"path": "gpt-4o-mini", "max_tokens": 512},
                },
            ],
            "evaluation": {
                "metrics": ["accuracy", "performance", "false_positive_rate"],
                "parallel_jobs": 3,
                "timeout_minutes": 60,
            },
        }

    @pytest.fixture
    def cybersecurity_config(self):
        """Realistic cybersecurity evaluation configuration."""
        return {
            "name": "Cybersecurity Detection Benchmark",
            "description": "Evaluate models on various cybersecurity detection tasks",
            "datasets": [
                {
                    "id": "network_intrusion",
                    "loader_type": "local",
                    "config": {
                        "path": "test://network_intrusion.json",
                        "max_samples": 1000,
                        "features": ["src_ip", "dst_ip", "protocol", "payload"],
                    },
                },
                {
                    "id": "malware_detection",
                    "loader_type": "local",
                    "config": {
                        "path": "test://malware_samples.json",
                        "max_samples": 800,
                        "features": ["file_hash", "behavior_patterns", "api_calls"],
                    },
                },
                {
                    "id": "phishing_emails",
                    "loader_type": "local",
                    "config": {
                        "path": "test://phishing_emails.json",
                        "max_samples": 600,
                        "features": ["subject", "body", "sender", "links"],
                    },
                },
            ],
            "models": [
                {
                    "id": "security_llama",
                    "plugin_type": "mlx_local",
                    "config": {
                        "path": "test://security_tuned_llama",
                        "max_tokens": 512,
                        "system_prompt": "You are a cybersecurity expert. Analyze the given data for threats.",
                    },
                },
                {
                    "id": "bert_security",
                    "plugin_type": "huggingface",
                    "config": {
                        "path": "test://bert_security_model",
                        "max_tokens": 256,
                        "classification_threshold": 0.7,
                    },
                },
            ],
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"],
                "parallel_jobs": 2,
                "timeout_minutes": 120,
                "cross_validation": False,
            },
        }

    @pytest.mark.asyncio
    async def test_complete_experiment_workflow(
        self, orchestration_service, sample_experiment_config
    ):
        """Test complete experiment workflow from creation to completion."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_experiment_config, f)
            config_path = f.name

        try:
            # Mock service responses
            self._setup_successful_service_mocks(orchestration_service, sample_experiment_config)

            # Create experiment
            experiment_id = await orchestration_service.create_experiment(config_path)
            assert experiment_id is not None
            assert experiment_id.startswith("exp_")

            # Verify experiment is in CREATED state
            progress = await orchestration_service.get_experiment_progress(experiment_id)
            assert progress.status == ExperimentStatus.CREATED

            # Start experiment synchronously for testing
            result = await orchestration_service.start_experiment(experiment_id, background=False)

            # Verify experiment completed successfully
            assert result["experiment_id"] == experiment_id
            assert result["status"] == ExperimentStatus.COMPLETED.value

            # Check final experiment result
            experiment_result = await orchestration_service.get_experiment_result(experiment_id)
            assert experiment_result.status == ExperimentStatus.COMPLETED
            assert len(experiment_result.models_evaluated) == 1
            assert len(experiment_result.datasets_used) == 1
            assert experiment_result.evaluation_results is not None

            # Verify all workflow steps completed
            assert "data_loading" in experiment_result.evaluation_results
            assert "model_loading" in experiment_result.evaluation_results
            assert "evaluation_execution" in experiment_result.evaluation_results
            assert "results_aggregation" in experiment_result.evaluation_results

        finally:
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_multi_model_experiment(self, orchestration_service, multi_model_config):
        """Test experiment with multiple models and datasets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(multi_model_config, f)
            config_path = f.name

        try:
            # Mock services for multi-model scenario
            self._setup_multi_model_service_mocks(orchestration_service, multi_model_config)

            # Create and run experiment
            experiment_id = await orchestration_service.create_experiment(
                config_path, "Multi-Model Test"
            )
            assert experiment_id is not None

            # Start in background
            result = await orchestration_service.start_experiment(experiment_id, background=True)
            assert "message" in result

            # Wait for completion with timeout
            completed = await self._wait_for_experiment_completion(
                orchestration_service, experiment_id, timeout=30
            )
            assert completed

            # Verify completion
            final_progress = await orchestration_service.get_experiment_progress(experiment_id)
            assert final_progress.status == ExperimentStatus.COMPLETED

            # Verify all model-dataset combinations were evaluated
            experiment_result = await orchestration_service.get_experiment_result(experiment_id)
            eval_results = experiment_result.evaluation_results.get("evaluation_execution", {})

            if "total_evaluations" in eval_results:
                # 3 models × 2 datasets = 6 evaluations expected
                assert eval_results["total_evaluations"] == 6
                assert eval_results["successful_evaluations"] > 0

        finally:
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cybersecurity_evaluation_scenario(
        self, orchestration_service, cybersecurity_config
    ):
        """Test realistic cybersecurity evaluation scenario."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cybersecurity_config, f)
            config_path = f.name

        try:
            # Mock services for cybersecurity scenario
            self._setup_cybersecurity_service_mocks(orchestration_service, cybersecurity_config)

            # Create cybersecurity experiment
            experiment_id = await orchestration_service.create_experiment(
                config_path, "Cybersecurity Benchmark"
            )
            assert experiment_id is not None

            # Start experiment
            result = await orchestration_service.start_experiment(experiment_id, background=False)
            assert result["status"] == ExperimentStatus.COMPLETED.value

            # Verify cybersecurity-specific results
            experiment_result = await orchestration_service.get_experiment_result(experiment_id)

            # Check that all cybersecurity datasets were used
            assert len(experiment_result.datasets_used) == 3
            assert "network_intrusion" in experiment_result.datasets_used
            assert "malware_detection" in experiment_result.datasets_used
            assert "phishing_emails" in experiment_result.datasets_used

            # Check that security models were evaluated
            assert len(experiment_result.models_evaluated) == 2
            assert "security_llama" in experiment_result.models_evaluated
            assert "bert_security" in experiment_result.models_evaluated

            # Verify security-specific metrics were calculated
            eval_results = experiment_result.evaluation_results.get("evaluation_execution", {})
            if "metrics_used" in eval_results:
                metrics = eval_results["metrics_used"]
                assert "false_positive_rate" in metrics
                assert "precision" in metrics
                assert "recall" in metrics

        finally:
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_experiment_cancellation(self, orchestration_service, sample_experiment_config):
        """Test experiment cancellation and cleanup."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_experiment_config, f)
            config_path = f.name

        try:
            # Mock services with delayed responses to simulate long-running experiment
            self._setup_slow_service_mocks(orchestration_service, sample_experiment_config)

            # Create experiment
            experiment_id = await orchestration_service.create_experiment(config_path)
            assert experiment_id is not None

            # Start experiment in background
            result = await orchestration_service.start_experiment(experiment_id, background=True)
            assert "message" in result

            # Wait a moment for experiment to start
            await asyncio.sleep(0.5)

            # Check that experiment is running
            progress = await orchestration_service.get_experiment_progress(experiment_id)
            assert progress.status in [ExperimentStatus.INITIALIZING, ExperimentStatus.RUNNING]

            # Cancel experiment
            cancelled = await orchestration_service.cancel_experiment(experiment_id)
            assert cancelled is True

            # Verify experiment is cancelled
            final_progress = await orchestration_service.get_experiment_progress(experiment_id)
            assert final_progress.status == ExperimentStatus.CANCELLED

        finally:
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestration_service):
        """Test error handling in various failure scenarios."""
        # Test with invalid configuration
        invalid_config = {"invalid": "config"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            invalid_config_path = f.name

        try:
            # Mock config service to return failure
            orchestration_service.services[
                "config"
            ].load_experiment_config.return_value = ServiceResponse(
                success=False, message="Invalid configuration", error="Invalid configuration"
            )

            # Should fail to create experiment with invalid config
            with pytest.raises(Exception, match="Failed to load configuration"):
                await orchestration_service.create_experiment(invalid_config_path)

        finally:
            Path(invalid_config_path).unlink(missing_ok=True)

        # Test with model loading failure
        failing_config = {
            "name": "Failing Experiment",
            "datasets": [
                {
                    "id": "test_dataset",
                    "loader_type": "local",
                    "config": {"path": "test://data.json"},
                }
            ],
            "models": [
                {
                    "id": "failing_model",
                    "plugin_type": "invalid_type",
                    "config": {"path": "invalid://model"},
                }
            ],
            "evaluation": {"metrics": ["accuracy"]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(failing_config, f)
            failing_config_path = f.name

        try:
            # Setup failing service mocks
            self._setup_failing_service_mocks(orchestration_service, failing_config)

            experiment_id = await orchestration_service.create_experiment(failing_config_path)
            assert experiment_id is not None

            # Experiment should fail during execution
            result = await orchestration_service.start_experiment(experiment_id, background=False)
            assert result["status"] == ExperimentStatus.FAILED.value

            # Check that experiment status shows failure
            progress = await orchestration_service.get_experiment_progress(experiment_id)
            assert progress.status == ExperimentStatus.FAILED
            assert progress.error_message is not None

        finally:
            Path(failing_config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_concurrent_experiments(self, orchestration_service, sample_experiment_config):
        """Test running multiple experiments concurrently."""
        # Create multiple config files
        config_paths = []
        for i in range(3):
            config = sample_experiment_config.copy()
            config["name"] = f"Concurrent Experiment {i + 1}"

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(config, f)
                config_paths.append(f.name)

        try:
            # Mock services for concurrent execution
            self._setup_concurrent_service_mocks(orchestration_service, sample_experiment_config)

            # Create all experiments
            experiment_ids = []
            for config_path in config_paths:
                experiment_id = await orchestration_service.create_experiment(config_path)
                assert experiment_id is not None
                experiment_ids.append(experiment_id)

            # Start all experiments concurrently
            start_tasks = [
                orchestration_service.start_experiment(exp_id, background=True)
                for exp_id in experiment_ids
            ]

            start_results = await asyncio.gather(*start_tasks)
            for result in start_results:
                assert "message" in result

            # Wait for all to complete
            completed_experiments = 0
            for exp_id in experiment_ids:
                completed = await self._wait_for_experiment_completion(
                    orchestration_service, exp_id, timeout=15
                )
                if completed:
                    completed_experiments += 1

            # Verify all experiments completed
            assert completed_experiments == len(experiment_ids)

            # Verify experiments list shows all experiments
            all_experiments = await orchestration_service.list_experiments()
            assert len(all_experiments) >= len(experiment_ids)

        finally:
            for config_path in config_paths:
                Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, orchestration_service):
        """Test orchestration service health monitoring."""
        health = await orchestration_service.health_check()

        assert health.status is not None
        assert "dependent_services" in health.checks
        assert "active_experiments" in health.checks
        assert "total_experiments" in health.checks

        # Check that all dependent services are reported
        dependent_services = health.checks["dependent_services"]
        expected_services = ["config", "data", "model", "evaluation"]

        for service_name in expected_services:
            assert service_name in dependent_services

    @pytest.mark.asyncio
    async def test_experiment_progress_tracking(
        self, orchestration_service, sample_experiment_config
    ):
        """Test detailed experiment progress tracking."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_experiment_config, f)
            config_path = f.name

        try:
            # Setup mocks for progress tracking
            self._setup_progress_tracking_mocks(orchestration_service, sample_experiment_config)

            experiment_id = await orchestration_service.create_experiment(config_path)
            assert experiment_id is not None

            # Check initial progress
            initial_progress = await orchestration_service.get_experiment_progress(experiment_id)
            assert initial_progress.status == ExperimentStatus.CREATED
            assert initial_progress.percentage == 0.0
            assert initial_progress.completed_steps == 0

            # Start experiment and track progress
            await orchestration_service.start_experiment(experiment_id, background=True)

            # Wait for completion while tracking progress
            max_wait = 10
            for _ in range(max_wait):
                await asyncio.sleep(0.5)
                progress = await orchestration_service.get_experiment_progress(experiment_id)

                # Verify progress fields are properly set
                assert progress.experiment_id == experiment_id
                assert isinstance(progress.percentage, float)
                assert progress.percentage >= 0.0 and progress.percentage <= 100.0
                assert progress.completed_steps <= progress.total_steps

                if progress.status == ExperimentStatus.COMPLETED:
                    assert progress.percentage == 100.0
                    assert progress.completed_steps == progress.total_steps
                    break

        finally:
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_resource_management(self, orchestration_service, multi_model_config):
        """Test resource management across multiple models."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(multi_model_config, f)
            config_path = f.name

        try:
            # Setup resource management mocks
            self._setup_resource_management_mocks(orchestration_service, multi_model_config)

            experiment_id = await orchestration_service.create_experiment(config_path)
            result = await orchestration_service.start_experiment(experiment_id, background=False)

            assert result["status"] == ExperimentStatus.COMPLETED.value

            # Verify resource management calls
            model_service = orchestration_service.services["model"]

            # Should have checked resources for each model
            assert model_service.resource_manager.can_load_model.call_count == 3

            # Should have optimized hardware
            model_service.optimize_for_hardware.assert_called()

        finally:
            Path(config_path).unlink(missing_ok=True)

    # Helper methods for setting up mocks

    def _setup_successful_service_mocks(self, orchestration_service, config):
        """Setup service mocks for successful experiment execution."""
        # Config service mock
        orchestration_service.services[
            "config"
        ].load_experiment_config.return_value = ServiceResponse(
            success=True, message="Config loaded", data=config
        )

        # Data service mock
        orchestration_service.services["data"].load_dataset.return_value = ServiceResponse(
            success=True,
            message="Dataset loaded",
            data={"samples": self._generate_mock_dataset(100)},
        )
        orchestration_service.services["data"].get_dataset_info.return_value = ServiceResponse(
            success=True,
            message="Dataset info",
            data={"samples_count": 100, "features": ["text", "label"]},
        )

        # Model service mock
        orchestration_service.services["model"].load_model.return_value = ServiceResponse(
            success=True, message="Model loaded", data={"model_id": "test_model"}
        )
        orchestration_service.services["model"].get_model_info.return_value = ServiceResponse(
            success=True, message="Model info", data={"memory_usage_mb": 512, "parameters": "7B"}
        )
        orchestration_service.services[
            "model"
        ].predict_batch.return_value = self._generate_mock_predictions(100)
        orchestration_service.services["model"].optimize_for_hardware = AsyncMock()

        # Resource manager mock
        resource_check = MagicMock()
        resource_check.can_load = True
        resource_check.estimated_memory_gb = 1.0
        orchestration_service.services[
            "model"
        ].resource_manager.can_load_model.return_value = resource_check

        # Evaluation service mock
        orchestration_service.services["evaluation"].evaluate_model.return_value = ServiceResponse(
            success=True,
            message="Evaluation completed",
            data={"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85},
        )

    def _setup_multi_model_service_mocks(self, orchestration_service, config):
        """Setup service mocks for multi-model experiment."""
        self._setup_successful_service_mocks(orchestration_service, config)

        # Override for multiple models
        model_responses = [
            ServiceResponse(success=True, message="Model 1 loaded", data={"model_id": "model_1"}),
            ServiceResponse(success=True, message="Model 2 loaded", data={"model_id": "model_2"}),
            ServiceResponse(success=True, message="Model 3 loaded", data={"model_id": "model_3"}),
        ]
        orchestration_service.services["model"].load_model.side_effect = model_responses

        # Multiple dataset responses
        dataset_responses = [
            ServiceResponse(
                success=True,
                message="Dataset 1 loaded",
                data={"samples": self._generate_mock_dataset(500)},
            ),
            ServiceResponse(
                success=True,
                message="Dataset 2 loaded",
                data={"samples": self._generate_mock_dataset(300)},
            ),
        ]
        orchestration_service.services["data"].load_dataset.side_effect = dataset_responses

    def _setup_cybersecurity_service_mocks(self, orchestration_service, config):
        """Setup service mocks for cybersecurity scenario."""
        self._setup_successful_service_mocks(orchestration_service, config)

        # Cybersecurity-specific dataset responses
        def mock_cybersec_dataset_loading(*args, **kwargs):
            dataset_id = args[0] if args else kwargs.get("dataset_id", "unknown")
            if "network" in dataset_id:
                return ServiceResponse(
                    success=True,
                    message="Network data loaded",
                    data={"samples": self._generate_cybersec_dataset("network", 1000)},
                )
            elif "malware" in dataset_id:
                return ServiceResponse(
                    success=True,
                    message="Malware data loaded",
                    data={"samples": self._generate_cybersec_dataset("malware", 800)},
                )
            else:  # phishing
                return ServiceResponse(
                    success=True,
                    message="Phishing data loaded",
                    data={"samples": self._generate_cybersec_dataset("phishing", 600)},
                )

        orchestration_service.services[
            "data"
        ].load_dataset.side_effect = mock_cybersec_dataset_loading

        # Cybersecurity-specific evaluation results
        orchestration_service.services["evaluation"].evaluate_model.return_value = ServiceResponse(
            success=True,
            message="Security evaluation completed",
            data={
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91,
                "false_positive_rate": 0.03,
            },
        )

    def _setup_slow_service_mocks(self, orchestration_service, config):
        """Setup service mocks with delays for cancellation testing."""
        self._setup_successful_service_mocks(orchestration_service, config)

        # Make model loading slow
        async def slow_model_loading(*_args, **_kwargs):
            await asyncio.sleep(2)  # Simulate slow loading
            return ServiceResponse(
                success=True, message="Model loaded slowly", data={"model_id": "slow_model"}
            )

        orchestration_service.services["model"].load_model.side_effect = slow_model_loading

    def _setup_failing_service_mocks(self, orchestration_service, config):
        """Setup service mocks for failure scenarios."""
        # Config service succeeds
        orchestration_service.services[
            "config"
        ].load_experiment_config.return_value = ServiceResponse(
            success=True, message="Config loaded", data=config
        )

        # Data service succeeds
        orchestration_service.services["data"].load_dataset.return_value = ServiceResponse(
            success=True,
            message="Dataset loaded",
            data={"samples": self._generate_mock_dataset(100)},
        )
        orchestration_service.services["data"].get_dataset_info.return_value = ServiceResponse(
            success=True, message="Dataset info", data={"samples_count": 100}
        )

        # Model service fails
        orchestration_service.services["model"].load_model.return_value = ServiceResponse(
            success=False, message="Model loading failed", error="Model not found"
        )

        # Resource manager
        resource_check = MagicMock()
        resource_check.can_load = True
        resource_check.estimated_memory_gb = 1.0
        orchestration_service.services[
            "model"
        ].resource_manager.can_load_model.return_value = resource_check

    def _setup_concurrent_service_mocks(self, orchestration_service, config):
        """Setup service mocks for concurrent execution."""
        self._setup_successful_service_mocks(orchestration_service, config)

        # Make responses slightly different for each call
        call_count = [0]

        def mock_model_loading(*_args, **_kwargs):
            call_count[0] += 1
            return ServiceResponse(
                success=True,
                message=f"Model {call_count[0]} loaded",
                data={"model_id": f"model_{call_count[0]}"},
            )

        orchestration_service.services["model"].load_model.side_effect = mock_model_loading

    def _setup_progress_tracking_mocks(self, orchestration_service, config):
        """Setup service mocks for progress tracking."""
        self._setup_successful_service_mocks(orchestration_service, config)

        # Add small delays to simulate realistic progress
        async def delayed_data_loading(*_args, **_kwargs):
            await asyncio.sleep(0.1)
            return ServiceResponse(
                success=True,
                message="Dataset loaded",
                data={"samples": self._generate_mock_dataset(100)},
            )

        async def delayed_model_loading(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return ServiceResponse(
                success=True, message="Model loaded", data={"model_id": "test_model"}
            )

        async def delayed_evaluation(*_args, **_kwargs):
            await asyncio.sleep(0.3)
            return ServiceResponse(
                success=True,
                message="Evaluation completed",
                data={"accuracy": 0.85, "f1_score": 0.83},
            )

        orchestration_service.services["data"].load_dataset.side_effect = delayed_data_loading
        orchestration_service.services["model"].load_model.side_effect = delayed_model_loading
        orchestration_service.services["evaluation"].evaluate_model.side_effect = delayed_evaluation

    def _setup_resource_management_mocks(self, orchestration_service, config):
        """Setup service mocks for resource management testing."""
        self._setup_multi_model_service_mocks(orchestration_service, config)

        # Enhanced resource manager
        orchestration_service.services[
            "model"
        ].resource_manager.suggest_model_unload_candidates.return_value = []
        orchestration_service.services["model"].cleanup_model = AsyncMock()

    # Helper methods for generating mock data

    def _generate_mock_dataset(self, size: int = 100) -> list[dict[str, Any]]:
        """Generate mock dataset for testing."""
        return [
            {
                "input_text": f"Sample network log entry {i}",
                "label": "ATTACK" if i % 3 == 0 else "BENIGN",
                "attack_type": "malware" if i % 3 == 0 else None,
            }
            for i in range(size)
        ]

    def _generate_cybersec_dataset(self, dataset_type: str, size: int) -> list[dict[str, Any]]:
        """Generate cybersecurity-specific mock dataset."""
        if dataset_type == "network":
            return [
                {
                    "src_ip": f"192.168.1.{i % 255}",
                    "dst_ip": f"10.0.0.{i % 100}",
                    "protocol": "TCP" if i % 2 == 0 else "UDP",
                    "payload": f"network_payload_{i}",
                    "label": "INTRUSION" if i % 4 == 0 else "NORMAL",
                }
                for i in range(size)
            ]
        elif dataset_type == "malware":
            return [
                {
                    "file_hash": f"hash_{i:08x}",
                    "behavior_patterns": [f"behavior_{j}" for j in range(i % 5)],
                    "api_calls": [f"api_call_{j}" for j in range(i % 10)],
                    "label": "MALWARE" if i % 5 == 0 else "BENIGN",
                }
                for i in range(size)
            ]
        else:  # phishing
            return [
                {
                    "subject": f"Email subject {i}",
                    "body": f"Email body content {i}",
                    "sender": f"sender{i}@example.com",
                    "links": [f"http://link{j}.com" for j in range(i % 3)],
                    "label": "PHISHING" if i % 6 == 0 else "LEGITIMATE",
                }
                for i in range(size)
            ]

    def _generate_mock_predictions(self, size: int) -> list[dict[str, Any]]:
        """Generate mock predictions for testing."""
        return [
            {
                "prediction": "ATTACK" if i % 3 == 0 else "BENIGN",
                "confidence": 0.85 + (i % 10) * 0.01,
                "explanation": f"Analysis of sample {i}",
                "inference_time_ms": 100.0 + (i % 50),
            }
            for i in range(size)
        ]

    async def _wait_for_experiment_completion(
        self, orchestration_service, experiment_id: str, timeout: int = 30
    ) -> bool:
        """Wait for experiment completion with timeout."""
        for _ in range(timeout):
            await asyncio.sleep(1)

            try:
                progress = await orchestration_service.get_experiment_progress(experiment_id)
                if progress.status in [
                    ExperimentStatus.COMPLETED,
                    ExperimentStatus.FAILED,
                    ExperimentStatus.CANCELLED,
                ]:
                    return progress.status == ExperimentStatus.COMPLETED
            except Exception:
                continue  # Continue waiting on errors

        return False  # Timeout reached

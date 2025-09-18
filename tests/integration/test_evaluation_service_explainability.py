"""
Integration tests for explainability evaluation service integration.

This module tests the complete integration between the evaluation service
and the explainability evaluator with realistic scenarios.
"""

from unittest.mock import patch

import pytest
import pytest_asyncio

from benchmark.interfaces.evaluation_interfaces import EvaluationRequest, MetricType
from benchmark.services.evaluation_service import EvaluationService


@pytest_asyncio.fixture
async def evaluation_service():
    """Create evaluation service with explainability configuration."""
    service = EvaluationService()

    # Configure explainability settings
    config = {
        "explainability": {
            "judge_model": "gpt-4o-mini",
            "batch_size": 5,
            "fail_on_missing_explanation": False,
            "min_explanation_length": 5,
            "max_explanation_length": 500,
        }
    }

    await service.initialize(config)
    yield service
    await service.shutdown()


@pytest.fixture
def sample_predictions():
    """Sample predictions with explanations for testing."""
    return [
        {
            "prediction": "attack",
            "explanation": "This traffic shows suspicious patterns including multiple failed authentication attempts from the same IP address 192.168.1.100, which indicates a potential brute force attack targeting the SSH service.",
            "confidence": 0.95,
        },
        {
            "prediction": "benign",
            "explanation": "Normal HTTPS traffic to legitimate domain google.com with standard headers and expected response codes. No indicators of compromise detected.",
            "confidence": 0.88,
        },
        {
            "prediction": "attack",
            "explanation": "SQL injection attempt detected in the query parameter with payload 'UNION SELECT * FROM users' targeting the login endpoint.",
            "confidence": 0.92,
        },
    ]


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth for testing."""
    return [
        {
            "label": "attack",
            "input_text": "Multiple SSH login failures from 192.168.1.100 targeting admin account",
            "explanation": "Brute force attack pattern with repeated authentication failures",
        },
        {
            "label": "benign",
            "input_text": "HTTPS request to google.com with standard browser headers",
            "explanation": "Legitimate web browsing traffic",
        },
        {
            "label": "attack",
            "input_text": "HTTP POST to /login with SQL injection payload in username field",
            "explanation": "SQL injection attack attempting to bypass authentication",
        },
    ]


class TestExplainabilityServiceIntegration:
    """Test explainability evaluation service integration."""

    @pytest.mark.asyncio
    async def test_service_initialization_with_explainability_config(self, evaluation_service):
        """Test service initialization includes explainability configuration."""
        assert evaluation_service.explainability_config is not None
        assert evaluation_service.explainability_config.judge_model == "gpt-4o-mini"
        assert evaluation_service.explainability_config.batch_size == 5
        assert evaluation_service.explainability_config.fail_on_missing_explanation is False

    @pytest.mark.asyncio
    async def test_explainability_evaluator_registration(self, evaluation_service):
        """Test explainability evaluator is properly registered."""
        assert MetricType.EXPLAINABILITY in evaluation_service.evaluators
        evaluator = evaluation_service.evaluators[MetricType.EXPLAINABILITY]
        assert hasattr(evaluator, "llm_judge")
        assert hasattr(evaluator, "automated_metrics")

    @pytest.mark.asyncio
    async def test_get_explainability_config(self, evaluation_service):
        """Test getting explainability configuration."""
        response = await evaluation_service.get_explainability_config()
        assert response.success is True
        assert "judge_model" in response.data
        assert response.data["judge_model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_explainability_data_validation_success(
        self, evaluation_service, sample_predictions, sample_ground_truth
    ):
        """Test successful explainability data validation."""
        request = EvaluationRequest(
            experiment_id="test_exp_001",
            model_id="test_model",
            dataset_id="test_dataset",
            predictions=sample_predictions,
            ground_truth=sample_ground_truth,
            metrics=[MetricType.EXPLAINABILITY],
            metadata={},
        )

        validation_result = evaluation_service._validate_explainability_data(request)
        assert validation_result.success is True

    @pytest.mark.asyncio
    async def test_explainability_data_validation_no_explanations(
        self, evaluation_service, sample_ground_truth
    ):
        """Test explainability validation with missing explanations."""
        predictions_no_explanations = [
            {"prediction": "attack", "confidence": 0.95},
            {"prediction": "benign", "confidence": 0.88},
        ]

        request = EvaluationRequest(
            experiment_id="test_exp_002",
            model_id="test_model",
            dataset_id="test_dataset",
            predictions=predictions_no_explanations,
            ground_truth=sample_ground_truth[:2],
            metrics=[MetricType.EXPLAINABILITY],
            metadata={},
        )

        # Should pass with warning since fail_on_missing_explanation is False
        validation_result = evaluation_service._validate_explainability_data(request)
        assert validation_result.success is True

    @pytest.mark.asyncio
    async def test_explainability_data_validation_strict_mode(self):
        """Test explainability validation in strict mode."""
        service = EvaluationService()
        config = {
            "explainability": {
                "fail_on_missing_explanation": True,
            }
        }
        await service.initialize(config)

        try:
            predictions_no_explanations = [
                {"prediction": "attack", "confidence": 0.95},
            ]
            ground_truth = [
                {"label": "attack", "input_text": "test input"},
            ]

            request = EvaluationRequest(
                experiment_id="test_exp_003",
                model_id="test_model",
                dataset_id="test_dataset",
                predictions=predictions_no_explanations,
                ground_truth=ground_truth,
                metrics=[MetricType.EXPLAINABILITY],
                metadata={},
            )

            validation_result = service._validate_explainability_data(request)
            assert validation_result.success is False
            assert "No explanations found" in validation_result.error

        finally:
            await service.shutdown()

    @pytest.mark.asyncio
    @patch("benchmark.evaluation.explainability.llm_judge.LLMJudgeEvaluator.judge_explanation")
    async def test_evaluate_explainability_method(
        self, mock_judge, evaluation_service, sample_predictions, sample_ground_truth
    ):
        """Test dedicated explainability evaluation method."""
        # Mock LLM judge responses
        from benchmark.evaluation.metrics.explainability import ExplanationQualityScore

        mock_score = ExplanationQualityScore(
            overall_score=0.85,
            technical_accuracy=0.88,
            logical_consistency=0.82,
            completeness=0.86,
            clarity=0.84,
            domain_relevance=0.87,
            detailed_feedback="Good explanation with technical details",
        )
        mock_judge.return_value = mock_score

        response = await evaluation_service.evaluate_explainability(
            sample_predictions, sample_ground_truth
        )

        assert response.success is True
        assert "metrics" in response.data
        assert "execution_time_seconds" in response.data
        assert "predictions_evaluated" in response.data
        assert response.data["predictions_evaluated"] == 3

    @pytest.mark.asyncio
    @patch("benchmark.evaluation.explainability.llm_judge.LLMJudgeEvaluator.judge_explanation")
    async def test_evaluate_explainability_with_config_override(
        self, mock_judge, evaluation_service, sample_predictions, sample_ground_truth
    ):
        """Test explainability evaluation with configuration override."""
        from benchmark.evaluation.metrics.explainability import ExplanationQualityScore

        mock_score = ExplanationQualityScore(
            overall_score=0.80,
            technical_accuracy=0.82,
            logical_consistency=0.78,
            completeness=0.81,
            clarity=0.79,
            domain_relevance=0.83,
            detailed_feedback="Adequate explanation",
        )
        mock_judge.return_value = mock_score

        config_override = {
            "judge_model": "gpt-4",
            "batch_size": 2,
        }

        response = await evaluation_service.evaluate_explainability(
            sample_predictions, sample_ground_truth, config_override
        )

        assert response.success is True
        assert "metrics" in response.data

    @pytest.mark.asyncio
    @patch("benchmark.evaluation.explainability.llm_judge.LLMJudgeEvaluator.judge_explanation")
    async def test_full_evaluation_pipeline_with_explainability(
        self, mock_judge, evaluation_service, sample_predictions, sample_ground_truth
    ):
        """Test complete evaluation pipeline including explainability."""
        from benchmark.evaluation.metrics.explainability import ExplanationQualityScore

        # Mock LLM judge responses
        mock_score = ExplanationQualityScore(
            overall_score=0.75,
            technical_accuracy=0.78,
            logical_consistency=0.72,
            completeness=0.76,
            clarity=0.74,
            domain_relevance=0.77,
            detailed_feedback="Satisfactory explanation quality",
        )
        mock_judge.return_value = mock_score

        request = EvaluationRequest(
            experiment_id="test_exp_004",
            model_id="test_model",
            dataset_id="test_dataset",
            predictions=sample_predictions,
            ground_truth=sample_ground_truth,
            metrics=[MetricType.EXPLAINABILITY],
            metadata={"test_run": "integration_test"},
        )

        result = await evaluation_service.evaluate_predictions(request)

        assert result.success is True
        assert result.experiment_id == "test_exp_004"
        assert result.model_id == "test_model"
        assert result.dataset_id == "test_dataset"
        assert len(result.metrics) > 0
        assert "avg_explanation_quality" in result.metrics
        assert MetricType.EXPLAINABILITY.value in result.detailed_results

    @pytest.mark.asyncio
    async def test_evaluate_explainability_evaluator_not_registered(self):
        """Test explainability evaluation when evaluator is not registered."""
        service = EvaluationService()
        await service.initialize()  # Initialize without explainability config

        try:
            # Remove explainability evaluator to simulate it not being registered
            if MetricType.EXPLAINABILITY in service.evaluators:
                del service.evaluators[MetricType.EXPLAINABILITY]

            response = await service.evaluate_explainability([], [])

            assert response.success is False
            assert "not registered" in response.error

        finally:
            await service.shutdown()

    @pytest.mark.asyncio
    async def test_evaluate_explainability_data_incompatible(self, evaluation_service):
        """Test explainability evaluation with incompatible data."""
        # Use empty predictions and ground truth with different lengths
        predictions = []
        ground_truth = [{"label": "attack"}]

        response = await evaluation_service.evaluate_explainability(predictions, ground_truth)

        assert response.success is False
        assert "not compatible" in response.error

    @pytest.mark.asyncio
    async def test_available_metrics_includes_explainability(self, evaluation_service):
        """Test that available metrics includes explainability."""
        response = await evaluation_service.get_available_metrics()

        assert response.success is True
        assert "metrics" in response.data
        assert MetricType.EXPLAINABILITY.value in response.data["metrics"]

        explainability_info = response.data["metrics"][MetricType.EXPLAINABILITY.value]
        assert "metric_names" in explainability_info
        assert "avg_explanation_quality" in explainability_info["metric_names"]

    @pytest.mark.asyncio
    async def test_health_check_with_explainability(self, evaluation_service):
        """Test health check includes explainability evaluator."""
        health = await evaluation_service.health_check()

        assert health.checks["evaluators_count"] >= 1  # At least explainability evaluator

    @pytest.mark.asyncio
    async def test_evaluation_history_with_explainability(
        self, evaluation_service, sample_predictions, sample_ground_truth
    ):
        """Test evaluation history stores explainability results."""
        with patch(
            "benchmark.evaluation.explainability.llm_judge.LLMJudgeEvaluator.judge_explanation"
        ) as mock_judge:
            from benchmark.evaluation.metrics.explainability import ExplanationQualityScore

            mock_score = ExplanationQualityScore(
                overall_score=0.70,
                technical_accuracy=0.72,
                logical_consistency=0.68,
                completeness=0.71,
                clarity=0.69,
                domain_relevance=0.73,
                detailed_feedback="Basic explanation quality",
            )
            mock_judge.return_value = mock_score

            request = EvaluationRequest(
                experiment_id="test_exp_005",
                model_id="test_model",
                dataset_id="test_dataset",
                predictions=sample_predictions,
                ground_truth=sample_ground_truth,
                metrics=[MetricType.EXPLAINABILITY],
                metadata={},
            )

            # Perform evaluation
            await evaluation_service.evaluate_predictions(request)

            # Check history
            history_response = await evaluation_service.get_evaluation_history(
                experiment_id="test_exp_005"
            )

            assert history_response.success is True
            assert len(history_response.data["results"]) == 1

            stored_result = history_response.data["results"][0]
            assert stored_result.experiment_id == "test_exp_005"
            assert len(stored_result.metrics) > 0

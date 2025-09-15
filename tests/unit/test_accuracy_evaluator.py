"""
Unit tests for the AccuracyEvaluator.

Tests comprehensive accuracy evaluation including basic metrics, probability-based
metrics, per-class analysis, confusion matrix statistics, and edge cases.
"""

import pytest
import pytest_asyncio

from benchmark.evaluation.metrics.accuracy import AccuracyEvaluator
from benchmark.interfaces.evaluation_interfaces import MetricType
from tests.fixtures.accuracy_test_data import (
    ALTERNATIVE_FIELDS_GROUND_TRUTH,
    ALTERNATIVE_FIELDS_PREDICTIONS,
    DEFAULT_CONFIDENCE_GROUND_TRUTH,
    DEFAULT_CONFIDENCE_PREDICTIONS,
    HIGH_CONFIDENCE_GROUND_TRUTH,
    HIGH_CONFIDENCE_PREDICTIONS,
    IMPERFECT_BINARY_EXPECTED_METRICS,
    IMPERFECT_BINARY_GROUND_TRUTH,
    IMPERFECT_BINARY_PREDICTIONS,
    LOW_CONFIDENCE_GROUND_TRUTH,
    LOW_CONFIDENCE_PREDICTIONS,
    MIXED_CONFIDENCE_GROUND_TRUTH,
    MIXED_CONFIDENCE_PREDICTIONS,
    MULTICLASS_GROUND_TRUTH,
    MULTICLASS_IMPERFECT_GROUND_TRUTH,
    MULTICLASS_IMPERFECT_PREDICTIONS,
    MULTICLASS_PREDICTIONS,
    PERFECT_BINARY_GROUND_TRUTH,
    PERFECT_BINARY_PREDICTIONS,
    SINGLE_CLASS_GROUND_TRUTH,
    SINGLE_CLASS_PREDICTIONS,
    TEST_SCENARIOS,
)


class TestAccuracyEvaluator:
    """Test cases for AccuracyEvaluator."""

    @pytest_asyncio.fixture
    async def evaluator(self):
        """Create an AccuracyEvaluator instance for testing."""
        return AccuracyEvaluator()

    def test_evaluator_initialization(self, evaluator):
        """Test that the evaluator initializes correctly."""
        assert evaluator.get_metric_type() == MetricType.ACCURACY
        assert isinstance(evaluator.get_metric_names(), list)
        assert len(evaluator.get_metric_names()) > 0
        assert "accuracy" in evaluator.get_metric_names()
        assert "precision" in evaluator.get_metric_names()
        assert "recall" in evaluator.get_metric_names()
        assert "f1_score" in evaluator.get_metric_names()

    def test_required_fields(self, evaluator):
        """Test required field specifications."""
        pred_fields = evaluator.get_required_prediction_fields()
        gt_fields = evaluator.get_required_ground_truth_fields()

        assert "prediction" in pred_fields
        assert "label" in gt_fields

    @pytest.mark.asyncio
    async def test_perfect_binary_classification(self, evaluator):
        """Test evaluation with perfect binary classification."""
        metrics = await evaluator.evaluate(PERFECT_BINARY_PREDICTIONS, PERFECT_BINARY_GROUND_TRUTH)

        # Check that all expected metrics are present
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            assert metric_name in metrics
            assert metrics[metric_name] == 1.0

        # Check confusion matrix metrics
        assert metrics["true_positive_rate"] == 1.0
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["specificity"] == 1.0

        # Check that we have per-class metrics
        assert "attack_precision" in metrics
        assert "benign_precision" in metrics

    @pytest.mark.asyncio
    async def test_imperfect_binary_classification(self, evaluator):
        """Test evaluation with imperfect binary classification."""
        metrics = await evaluator.evaluate(
            IMPERFECT_BINARY_PREDICTIONS, IMPERFECT_BINARY_GROUND_TRUTH
        )

        # Check basic metrics match expected values
        assert abs(metrics["accuracy"] - IMPERFECT_BINARY_EXPECTED_METRICS["accuracy"]) < 0.001
        assert metrics["true_positives"] == IMPERFECT_BINARY_EXPECTED_METRICS["true_positives"]
        assert metrics["false_positives"] == IMPERFECT_BINARY_EXPECTED_METRICS["false_positives"]
        assert metrics["true_negatives"] == IMPERFECT_BINARY_EXPECTED_METRICS["true_negatives"]
        assert metrics["false_negatives"] == IMPERFECT_BINARY_EXPECTED_METRICS["false_negatives"]

        # Check that precision and recall are between 0 and 1
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_multiclass_perfect_classification(self, evaluator):
        """Test evaluation with perfect multi-class classification."""
        metrics = await evaluator.evaluate(MULTICLASS_PREDICTIONS, MULTICLASS_GROUND_TRUTH)

        # Perfect classification should have accuracy = 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

        # Check per-class metrics exist
        assert "malware_precision" in metrics
        assert "intrusion_precision" in metrics
        assert "dos_precision" in metrics
        assert "benign_precision" in metrics
        assert "phishing_precision" in metrics

        # All per-class metrics should be 1.0 for perfect classification
        assert metrics["malware_precision"] == 1.0
        assert metrics["intrusion_recall"] == 1.0
        assert metrics["dos_f1"] == 1.0

    @pytest.mark.asyncio
    async def test_multiclass_imperfect_classification(self, evaluator):
        """Test evaluation with imperfect multi-class classification."""
        metrics = await evaluator.evaluate(
            MULTICLASS_IMPERFECT_PREDICTIONS, MULTICLASS_IMPERFECT_GROUND_TRUTH
        )

        # Imperfect classification should have accuracy < 1.0
        assert 0.0 <= metrics["accuracy"] < 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

        # Should have per-class metrics
        assert "malware_precision" in metrics
        assert "intrusion_precision" in metrics
        assert "dos_precision" in metrics
        assert "benign_precision" in metrics

    @pytest.mark.asyncio
    async def test_single_class_edge_case(self, evaluator):
        """Test evaluation with single class (edge case)."""
        metrics = await evaluator.evaluate(SINGLE_CLASS_PREDICTIONS, SINGLE_CLASS_GROUND_TRUTH)

        # Single class should have perfect accuracy
        assert metrics["accuracy"] == 1.0

        # Should handle single class gracefully
        assert "attack_precision" in metrics
        assert metrics["attack_precision"] == 1.0

    @pytest.mark.asyncio
    async def test_probability_metrics_high_confidence(self, evaluator):
        """Test probability-based metrics with high confidence predictions."""
        metrics = await evaluator.evaluate(
            HIGH_CONFIDENCE_PREDICTIONS, HIGH_CONFIDENCE_GROUND_TRUTH
        )

        # Should have ROC-AUC and PR-AUC metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

        # High confidence perfect predictions should have high AUC
        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0

    @pytest.mark.asyncio
    async def test_probability_metrics_mixed_confidence(self, evaluator):
        """Test probability-based metrics with mixed confidence and errors."""
        metrics = await evaluator.evaluate(
            MIXED_CONFIDENCE_PREDICTIONS, MIXED_CONFIDENCE_GROUND_TRUTH
        )

        # Should have ROC-AUC and PR-AUC metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

        # Mixed confidence with errors should have AUC < 1.0 but > 0.5 (better than random)
        assert 0.5 < metrics["roc_auc"] < 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    @pytest.mark.asyncio
    async def test_default_confidence_handling(self, evaluator):
        """Test handling of predictions without explicit confidence scores."""
        metrics = await evaluator.evaluate(
            DEFAULT_CONFIDENCE_PREDICTIONS, DEFAULT_CONFIDENCE_GROUND_TRUTH
        )

        # Should still calculate basic metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # ROC-AUC should be 0.0 when no real confidence is provided
        assert metrics["roc_auc"] == 0.0
        assert metrics["pr_auc"] == 0.0

    @pytest.mark.asyncio
    async def test_low_confidence_predictions(self, evaluator):
        """Test evaluation with low confidence predictions."""
        metrics = await evaluator.evaluate(LOW_CONFIDENCE_PREDICTIONS, LOW_CONFIDENCE_GROUND_TRUTH)

        # Should still calculate all metrics
        assert "accuracy" in metrics
        assert "roc_auc" in metrics

        # Low confidence perfect predictions should still have perfect accuracy
        assert metrics["accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_matthews_correlation_coefficient(self, evaluator):
        """Test Matthews correlation coefficient calculation."""
        # Perfect predictions
        metrics = await evaluator.evaluate(PERFECT_BINARY_PREDICTIONS, PERFECT_BINARY_GROUND_TRUTH)
        assert metrics["matthews_corr"] == 1.0

        # Imperfect predictions
        metrics = await evaluator.evaluate(
            IMPERFECT_BINARY_PREDICTIONS, IMPERFECT_BINARY_GROUND_TRUTH
        )
        assert -1.0 <= metrics["matthews_corr"] <= 1.0

    @pytest.mark.asyncio
    async def test_macro_metrics(self, evaluator):
        """Test macro-averaged metrics calculation."""
        metrics = await evaluator.evaluate(MULTICLASS_PREDICTIONS, MULTICLASS_GROUND_TRUTH)

        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics

        # For perfect classification, macro metrics should be 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["f1_macro"] == 1.0

    @pytest.mark.asyncio
    async def test_confusion_matrix_metrics_binary(self, evaluator):
        """Test confusion matrix derived metrics for binary classification."""
        metrics = await evaluator.evaluate(
            IMPERFECT_BINARY_PREDICTIONS, IMPERFECT_BINARY_GROUND_TRUTH
        )

        # Should have all confusion matrix components
        assert "true_positives" in metrics
        assert "false_positives" in metrics
        assert "true_negatives" in metrics
        assert "false_negatives" in metrics

        # Should have derived metrics
        assert "true_positive_rate" in metrics
        assert "false_positive_rate" in metrics
        assert "specificity" in metrics
        assert "sensitivity" in metrics

        # Specificity should equal true negative rate
        assert metrics["specificity"] == metrics["true_negative_rate"]
        assert metrics["sensitivity"] == metrics["true_positive_rate"]

    @pytest.mark.asyncio
    async def test_confusion_matrix_metrics_multiclass(self, evaluator):
        """Test confusion matrix derived metrics for multi-class classification."""
        metrics = await evaluator.evaluate(
            MULTICLASS_IMPERFECT_PREDICTIONS, MULTICLASS_IMPERFECT_GROUND_TRUTH
        )

        # Should have averaged metrics for multi-class
        assert "true_positive_rate" in metrics
        assert "false_positive_rate" in metrics
        assert "specificity" in metrics

        # All rates should be between 0 and 1
        assert 0.0 <= metrics["true_positive_rate"] <= 1.0
        assert 0.0 <= metrics["false_positive_rate"] <= 1.0
        assert 0.0 <= metrics["specificity"] <= 1.0

    @pytest.mark.asyncio
    async def test_alternative_field_names(self, evaluator):
        """Test handling of alternative field names in input data."""
        # Should handle 'label' instead of 'prediction' in predictions
        # and 'true_label'/'ground_truth' instead of 'label' in ground truth
        # This test validates the base evaluator's extract_labels method
        pred_labels, true_labels = evaluator.extract_labels(
            ALTERNATIVE_FIELDS_PREDICTIONS, ALTERNATIVE_FIELDS_GROUND_TRUTH
        )

        assert len(pred_labels) == 2
        assert len(true_labels) == 2
        assert pred_labels[0] == "ATTACK"
        assert true_labels[0] == "ATTACK"

    @pytest.mark.asyncio
    async def test_data_validation_errors(self, evaluator):
        """Test that data validation catches invalid input."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            await evaluator.evaluate(
                [{"prediction": "ATTACK"}], [{"label": "ATTACK"}, {"label": "BENIGN"}]
            )

        # Empty datasets
        with pytest.raises(ValueError, match="empty"):
            await evaluator.evaluate([], [])

    @pytest.mark.asyncio
    async def test_per_class_metrics_naming(self, evaluator):
        """Test that per-class metrics are named correctly."""
        metrics = await evaluator.evaluate(MULTICLASS_PREDICTIONS, MULTICLASS_GROUND_TRUTH)

        # Check that class names are properly sanitized for metric names
        expected_classes = ["malware", "intrusion", "dos", "benign", "phishing"]
        for class_name in expected_classes:
            assert f"{class_name}_precision" in metrics
            assert f"{class_name}_recall" in metrics
            assert f"{class_name}_f1" in metrics
            assert f"{class_name}_support" in metrics

    @pytest.mark.asyncio
    async def test_detailed_report_generation(self, evaluator):
        """Test detailed classification report generation."""
        report = evaluator.generate_detailed_report(
            PERFECT_BINARY_PREDICTIONS, PERFECT_BINARY_GROUND_TRUTH
        )

        assert isinstance(report, str)
        assert "Cybersecurity Classification Report" in report
        assert "Confusion Matrix" in report
        assert "precision" in report.lower()
        assert "recall" in report.lower()

    @pytest.mark.asyncio
    async def test_all_test_scenarios(self, evaluator):
        """Test all predefined test scenarios."""
        for _scenario_name, scenario_data in TEST_SCENARIOS.items():
            metrics = await evaluator.evaluate(
                scenario_data["predictions"], scenario_data["ground_truth"]
            )

            # Basic validation that metrics are calculated
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics

            # All metrics should be between 0 and 1 (or valid ranges)
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0
            assert 0.0 <= metrics["f1_score"] <= 1.0

            # ROC-AUC and PR-AUC should be present and in valid range
            assert "roc_auc" in metrics
            assert "pr_auc" in metrics
            assert 0.0 <= metrics["roc_auc"] <= 1.0
            assert 0.0 <= metrics["pr_auc"] <= 1.0

    @pytest.mark.asyncio
    async def test_metric_consistency(self, evaluator):
        """Test consistency between different metric calculations."""
        metrics = await evaluator.evaluate(
            IMPERFECT_BINARY_PREDICTIONS, IMPERFECT_BINARY_GROUND_TRUTH
        )

        # True positive rate should equal sensitivity
        assert metrics["true_positive_rate"] == metrics["sensitivity"]

        # Specificity should equal true negative rate
        assert metrics["specificity"] == metrics["true_negative_rate"]

        # False positive rate + specificity should equal 1
        assert abs((metrics["false_positive_rate"] + metrics["specificity"]) - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_support_calculation(self, evaluator):
        """Test that support values (number of samples per class) are calculated correctly."""
        metrics = await evaluator.evaluate(MULTICLASS_PREDICTIONS, MULTICLASS_GROUND_TRUTH)

        # Support should be positive integers (converted to float)
        for class_name in ["malware", "intrusion", "dos", "benign", "phishing"]:
            support_key = f"{class_name}_support"
            assert support_key in metrics
            assert metrics[support_key] >= 0
            assert metrics[support_key] == int(metrics[support_key])  # Should be whole number

    @pytest.mark.asyncio
    async def test_zero_division_handling(self, evaluator):
        """Test handling of zero division cases."""
        # Create a scenario where one class has no predictions
        predictions = [
            {"prediction": "ATTACK", "confidence": 0.9},
            {"prediction": "ATTACK", "confidence": 0.8},
        ]
        ground_truth = [
            {"label": "ATTACK"},
            {"label": "BENIGN"},  # BENIGN has no predictions -> precision = 0
        ]

        metrics = await evaluator.evaluate(predictions, ground_truth)

        # Should handle zero division gracefully
        assert "benign_precision" in metrics
        assert metrics["benign_precision"] == 0.0  # No predicted positives for BENIGN

    @pytest.mark.asyncio
    async def test_probability_metrics_multiclass(self, evaluator):
        """Test probability metrics calculation for multi-class scenarios."""
        metrics = await evaluator.evaluate(MULTICLASS_PREDICTIONS, MULTICLASS_GROUND_TRUTH)

        # Multi-class should have ROC-AUC
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0

        # PR-AUC for multi-class is set to 0.0 (simplified implementation)
        assert metrics["pr_auc"] == 0.0

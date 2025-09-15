"""
Unit tests for the BaseEvaluator class.

Tests input data validation, label and confidence extraction,
statistical calculations, and utility methods.
"""

import pytest

from benchmark.evaluation.base_evaluator import BaseEvaluator
from benchmark.interfaces.evaluation_interfaces import MetricType


class TestEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator for testing."""

    def __init__(self, metric_type: MetricType = MetricType.ACCURACY):
        super().__init__(metric_type)

    async def evaluate(self, predictions, ground_truth):
        """Mock evaluation method."""
        self.validate_input_data(predictions, ground_truth)
        return {"test_metric": 0.85}

    def get_metric_names(self):
        return ["test_metric"]

    def get_required_prediction_fields(self):
        return ["prediction"]

    def get_required_ground_truth_fields(self):
        return ["label"]


class TestBaseEvaluator:
    """Test cases for BaseEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create a test evaluator instance."""
        return TestEvaluator()

    @pytest.fixture
    def valid_predictions(self):
        """Valid prediction data for testing."""
        return [
            {"prediction": "ATTACK", "confidence": 0.9},
            {"prediction": "BENIGN", "confidence": 0.8},
            {"prediction": "ATTACK", "confidence": 0.7},
            {"prediction": "BENIGN", "confidence": 0.6},
        ]

    @pytest.fixture
    def valid_ground_truth(self):
        """Valid ground truth data for testing."""
        return [
            {"label": "ATTACK"},
            {"label": "BENIGN"},
            {"label": "ATTACK"},
            {"label": "BENIGN"},
        ]

    @pytest.fixture
    def predictions_with_probabilities(self):
        """Predictions with probability distributions."""
        return [
            {
                "prediction": "ATTACK",
                "confidence": 0.9,
                "probabilities": {"ATTACK": 0.9, "BENIGN": 0.1},
            },
            {
                "prediction": "BENIGN",
                "confidence": 0.8,
                "probabilities": {"ATTACK": 0.2, "BENIGN": 0.8},
            },
        ]

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = TestEvaluator(MetricType.PRECISION)

        assert evaluator.get_metric_type() == MetricType.PRECISION
        assert evaluator.metric_type == MetricType.PRECISION

    def test_validate_input_data_success(self, evaluator, valid_predictions, valid_ground_truth):
        """Test successful input data validation."""
        # Should not raise any exception
        evaluator.validate_input_data(valid_predictions, valid_ground_truth)

    def test_validate_input_data_length_mismatch(self, evaluator, valid_predictions):
        """Test validation with mismatched lengths."""
        ground_truth = [{"label": "ATTACK"}]  # Different length

        with pytest.raises(ValueError) as exc_info:
            evaluator.validate_input_data(valid_predictions, ground_truth)

        assert "must have same length" in str(exc_info.value)

    def test_validate_input_data_empty(self, evaluator):
        """Test validation with empty data."""
        with pytest.raises(ValueError) as exc_info:
            evaluator.validate_input_data([], [])

        assert "empty prediction set" in str(exc_info.value)

    def test_validate_input_data_missing_prediction_field(self, evaluator, valid_ground_truth):
        """Test validation with missing prediction fields."""
        invalid_predictions = [
            {"confidence": 0.9},  # Missing 'prediction' field
            {"prediction": "BENIGN"},
        ]

        with pytest.raises(ValueError) as exc_info:
            evaluator.validate_input_data(invalid_predictions, valid_ground_truth[:2])

        assert "Missing required prediction field 'prediction'" in str(exc_info.value)

    def test_validate_input_data_missing_ground_truth_field(self, evaluator, valid_predictions):
        """Test validation with missing ground truth fields."""
        invalid_ground_truth = [
            {"incorrect_field": "ATTACK"},  # Missing 'label' field
            {"label": "BENIGN"},
        ]

        with pytest.raises(ValueError) as exc_info:
            evaluator.validate_input_data(valid_predictions[:2], invalid_ground_truth)

        assert "Missing required ground truth field 'label'" in str(exc_info.value)

    def test_validate_input_data_non_dict_prediction(self, evaluator):
        """Test validation with non-dictionary prediction."""
        invalid_predictions = ["not_a_dict"]
        ground_truth = [{"label": "ATTACK"}]

        with pytest.raises(ValueError) as exc_info:
            evaluator.validate_input_data(invalid_predictions, ground_truth)

        assert "must be a dictionary" in str(exc_info.value)

    def test_validate_input_data_non_dict_ground_truth(self, evaluator):
        """Test validation with non-dictionary ground truth."""
        predictions = [{"prediction": "ATTACK"}]
        invalid_ground_truth = ["not_a_dict"]

        with pytest.raises(ValueError) as exc_info:
            evaluator.validate_input_data(predictions, invalid_ground_truth)

        assert "must be a dictionary" in str(exc_info.value)

    def test_extract_labels_standard_fields(self, evaluator, valid_predictions, valid_ground_truth):
        """Test extracting labels with standard field names."""
        pred_labels, true_labels = evaluator.extract_labels(valid_predictions, valid_ground_truth)

        assert pred_labels == ["ATTACK", "BENIGN", "ATTACK", "BENIGN"]
        assert true_labels == ["ATTACK", "BENIGN", "ATTACK", "BENIGN"]

    def test_extract_labels_alternative_prediction_field(self, evaluator):
        """Test extracting labels with alternative prediction field name."""
        predictions = [{"label": "ATTACK"}, {"label": "BENIGN"}]
        ground_truth = [{"label": "ATTACK"}, {"label": "BENIGN"}]

        pred_labels, true_labels = evaluator.extract_labels(predictions, ground_truth)

        assert pred_labels == ["ATTACK", "BENIGN"]
        assert true_labels == ["ATTACK", "BENIGN"]

    def test_extract_labels_alternative_ground_truth_fields(self, evaluator):
        """Test extracting labels with alternative ground truth field names."""
        predictions = [{"prediction": "ATTACK"}, {"prediction": "BENIGN"}]

        # Test with 'true_label' field
        ground_truth1 = [{"true_label": "ATTACK"}, {"true_label": "BENIGN"}]
        pred_labels, true_labels = evaluator.extract_labels(predictions, ground_truth1)
        assert true_labels == ["ATTACK", "BENIGN"]

        # Test with 'ground_truth' field
        ground_truth2 = [{"ground_truth": "ATTACK"}, {"ground_truth": "BENIGN"}]
        pred_labels, true_labels = evaluator.extract_labels(predictions, ground_truth2)
        assert true_labels == ["ATTACK", "BENIGN"]

    def test_extract_labels_missing_prediction_field(self, evaluator):
        """Test extracting labels when prediction field is missing."""
        predictions = [{"confidence": 0.9}]  # No prediction or label field
        ground_truth = [{"label": "ATTACK"}]

        with pytest.raises(ValueError) as exc_info:
            evaluator.extract_labels(predictions, ground_truth)

        assert "No prediction or label field found" in str(exc_info.value)

    def test_extract_labels_missing_ground_truth_field(self, evaluator):
        """Test extracting labels when ground truth field is missing."""
        predictions = [{"prediction": "ATTACK"}]
        ground_truth = [{"incorrect_field": "ATTACK"}]  # No label field

        with pytest.raises(ValueError) as exc_info:
            evaluator.extract_labels(predictions, ground_truth)

        assert "No label field found" in str(exc_info.value)

    def test_extract_confidences_with_values(self, evaluator, valid_predictions):
        """Test extracting confidence scores when present."""
        confidences = evaluator.extract_confidences(valid_predictions)

        assert confidences == [0.9, 0.8, 0.7, 0.6]

    def test_extract_confidences_missing_values(self, evaluator):
        """Test extracting confidence scores when missing."""
        predictions = [
            {"prediction": "ATTACK"},  # No confidence
            {"prediction": "BENIGN", "confidence": 0.8},
        ]

        confidences = evaluator.extract_confidences(predictions)

        assert confidences == [0.5, 0.8]  # Default 0.5 for missing

    def test_extract_confidences_invalid_values(self, evaluator):
        """Test extracting confidence scores with invalid values."""
        predictions = [
            {"prediction": "ATTACK", "confidence": "invalid"},
            {"prediction": "BENIGN", "confidence": None},
        ]

        confidences = evaluator.extract_confidences(predictions)

        assert confidences == [0.5, 0.5]  # Defaults for invalid values

    def test_extract_probabilities_with_dict(self, evaluator, predictions_with_probabilities):
        """Test extracting probabilities when provided as dictionary."""
        probabilities = evaluator.extract_probabilities(predictions_with_probabilities)

        assert len(probabilities) == 2
        assert probabilities[0] == {"ATTACK": 0.9, "BENIGN": 0.1}
        assert probabilities[1] == {"ATTACK": 0.2, "BENIGN": 0.8}

    def test_extract_probabilities_from_confidence(self, evaluator):
        """Test extracting probabilities from confidence scores."""
        predictions = [
            {"prediction": "ATTACK", "confidence": 0.8},
            {"prediction": "BENIGN", "confidence": 0.7},
        ]
        class_labels = ["ATTACK", "BENIGN"]

        probabilities = evaluator.extract_probabilities(predictions, class_labels)

        assert len(probabilities) == 2
        assert probabilities[0]["ATTACK"] == 0.8
        assert probabilities[0]["BENIGN"] == 0.2  # Complement for binary
        assert probabilities[1]["BENIGN"] == 0.7
        assert probabilities[1]["ATTACK"] == 0.3

    def test_extract_probabilities_missing(self, evaluator):
        """Test extracting probabilities when not provided."""
        predictions = [{"prediction": "ATTACK"}]

        probabilities = evaluator.extract_probabilities(predictions)

        assert len(probabilities) == 1
        assert probabilities[0] == {"ATTACK": 0.5}  # Default confidence

    def test_extract_probabilities_invalid_format(self, evaluator):
        """Test extracting probabilities with invalid format."""
        predictions = [
            {"prediction": "ATTACK", "probabilities": "invalid_format"},
        ]

        probabilities = evaluator.extract_probabilities(predictions)

        assert len(probabilities) == 1
        assert probabilities[0] == {"ATTACK": 0.5}  # Fallback to confidence

    def test_calculate_basic_stats_valid_data(self, evaluator):
        """Test calculating statistics with valid data."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = evaluator.calculate_basic_stats(values)

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0
        assert stats["std"] > 0

    def test_calculate_basic_stats_empty_data(self, evaluator):
        """Test calculating statistics with empty data."""
        stats = evaluator.calculate_basic_stats([])

        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_calculate_basic_stats_invalid_values(self, evaluator):
        """Test calculating statistics with invalid values."""
        values = [1.0, "invalid", 3.0, None, 5.0]

        stats = evaluator.calculate_basic_stats(values)

        assert stats["count"] == 3  # Only valid numeric values
        assert stats["mean"] == 3.0  # (1 + 3 + 5) / 3

    def test_calculate_class_distribution(self, evaluator):
        """Test calculating class distribution."""
        labels = ["ATTACK", "BENIGN", "ATTACK", "ATTACK", "BENIGN"]

        distribution = evaluator.calculate_class_distribution(labels)

        assert "ATTACK" in distribution
        assert "BENIGN" in distribution
        assert distribution["ATTACK"]["count"] == 3
        assert distribution["BENIGN"]["count"] == 2
        assert distribution["ATTACK"]["percentage"] == 60.0
        assert distribution["BENIGN"]["percentage"] == 40.0

    def test_calculate_class_distribution_empty(self, evaluator):
        """Test calculating class distribution with empty data."""
        distribution = evaluator.calculate_class_distribution([])

        assert distribution == {}

    def test_normalize_predictions(self, evaluator, valid_predictions, valid_ground_truth):
        """Test normalizing prediction and ground truth data."""
        norm_pred, norm_gt = evaluator.normalize_predictions(valid_predictions, valid_ground_truth)

        assert len(norm_pred) == len(valid_predictions)
        assert len(norm_gt) == len(valid_ground_truth)

        # Check that all required fields are present
        for i, pred in enumerate(norm_pred):
            assert "prediction" in pred
            assert "confidence" in pred
            assert "sample_index" in pred
            assert pred["sample_index"] == i

        for i, gt in enumerate(norm_gt):
            assert "label" in gt
            assert "sample_index" in gt
            assert gt["sample_index"] == i

    def test_normalize_predictions_alternative_fields(self, evaluator):
        """Test normalizing data with alternative field names."""
        predictions = [{"label": "ATTACK"}]  # Using 'label' instead of 'prediction'
        ground_truth = [{"true_label": "ATTACK"}]  # Using 'true_label' instead of 'label'

        norm_pred, norm_gt = evaluator.normalize_predictions(predictions, ground_truth)

        assert norm_pred[0]["prediction"] == "ATTACK"
        assert norm_gt[0]["label"] == "ATTACK"
        assert norm_pred[0]["confidence"] == 0.5  # Default confidence added

    def test_format_results(self, evaluator):
        """Test formatting evaluation results."""
        metrics = {"accuracy": 0.85, "precision": 0.80}
        additional_data = {"samples_evaluated": 100}

        results = evaluator.format_results(metrics, additional_data)

        assert results["metrics"] == metrics
        assert results["evaluator_type"] == evaluator.metric_type.value
        assert results["metric_names"] == evaluator.get_metric_names()
        assert results["samples_evaluated"] == 100
        assert "timestamp" in results

    def test_safe_divide_normal_case(self, evaluator):
        """Test safe division with normal values."""
        result = evaluator.safe_divide(10, 2)
        assert result == 5.0

    def test_safe_divide_zero_denominator(self, evaluator):
        """Test safe division with zero denominator."""
        result = evaluator.safe_divide(10, 0)
        assert result == 0.0  # Default value

    def test_safe_divide_custom_default(self, evaluator):
        """Test safe division with custom default value."""
        result = evaluator.safe_divide(10, 0, default=-1.0)
        assert result == -1.0

    def test_safe_divide_invalid_types(self, evaluator):
        """Test safe division with invalid types."""
        result = evaluator.safe_divide("invalid", "types")
        assert result == 0.0  # Default value

    def test_create_confusion_matrix_dict(self, evaluator):
        """Test creating confusion matrix dictionary."""
        true_labels = ["ATTACK", "BENIGN", "ATTACK", "BENIGN"]
        pred_labels = ["ATTACK", "ATTACK", "ATTACK", "BENIGN"]

        cm_dict = evaluator.create_confusion_matrix_dict(true_labels, pred_labels)

        assert "matrix" in cm_dict
        assert "labels" in cm_dict
        assert "total_samples" in cm_dict

        matrix = cm_dict["matrix"]
        assert matrix["ATTACK"]["ATTACK"] == 2  # True positives for ATTACK
        assert matrix["BENIGN"]["ATTACK"] == 1  # False positives for ATTACK
        assert matrix["ATTACK"]["BENIGN"] == 0  # False negatives for ATTACK
        assert matrix["BENIGN"]["BENIGN"] == 1  # True positives for BENIGN

        assert cm_dict["total_samples"] == 4
        assert set(cm_dict["labels"]) == {"ATTACK", "BENIGN"}

    def test_create_confusion_matrix_dict_length_mismatch(self, evaluator):
        """Test confusion matrix with mismatched label lengths."""
        true_labels = ["ATTACK", "BENIGN"]
        pred_labels = ["ATTACK"]  # Different length

        with pytest.raises(ValueError) as exc_info:
            evaluator.create_confusion_matrix_dict(true_labels, pred_labels)

        assert "same length" in str(exc_info.value)

    def test_calculate_per_class_metrics(self, evaluator):
        """Test calculating per-class precision, recall, and F1 scores."""
        true_labels = ["ATTACK", "BENIGN", "ATTACK", "BENIGN", "ATTACK"]
        pred_labels = ["ATTACK", "ATTACK", "ATTACK", "BENIGN", "BENIGN"]

        per_class_metrics = evaluator.calculate_per_class_metrics(true_labels, pred_labels)

        assert "ATTACK" in per_class_metrics
        assert "BENIGN" in per_class_metrics

        # Check ATTACK metrics
        attack_metrics = per_class_metrics["ATTACK"]
        assert "precision" in attack_metrics
        assert "recall" in attack_metrics
        assert "f1_score" in attack_metrics
        assert "support" in attack_metrics
        assert "true_positives" in attack_metrics
        assert "false_positives" in attack_metrics
        assert "false_negatives" in attack_metrics

        # ATTACK: TP=2, FP=1, FN=1
        assert attack_metrics["true_positives"] == 2
        assert attack_metrics["false_positives"] == 1
        assert attack_metrics["false_negatives"] == 1
        assert attack_metrics["support"] == 3  # Total ATTACK samples

        # Check that precision and recall are calculated correctly
        expected_precision = 2 / (2 + 1)  # TP / (TP + FP) = 2/3
        expected_recall = 2 / (2 + 1)  # TP / (TP + FN) = 2/3

        assert abs(attack_metrics["precision"] - expected_precision) < 0.001
        assert abs(attack_metrics["recall"] - expected_recall) < 0.001

    def test_calculate_per_class_metrics_perfect_classification(self, evaluator):
        """Test per-class metrics with perfect classification."""
        true_labels = ["ATTACK", "BENIGN", "ATTACK", "BENIGN"]
        pred_labels = ["ATTACK", "BENIGN", "ATTACK", "BENIGN"]

        per_class_metrics = evaluator.calculate_per_class_metrics(true_labels, pred_labels)

        # Perfect classification should have precision=1, recall=1, f1=1
        for class_label in ["ATTACK", "BENIGN"]:
            metrics = per_class_metrics[class_label]
            assert metrics["precision"] == 1.0
            assert metrics["recall"] == 1.0
            assert metrics["f1_score"] == 1.0
            assert metrics["false_positives"] == 0
            assert metrics["false_negatives"] == 0

    def test_calculate_per_class_metrics_single_class(self, evaluator):
        """Test per-class metrics with single class."""
        true_labels = ["ATTACK", "ATTACK", "ATTACK"]
        pred_labels = ["ATTACK", "ATTACK", "BENIGN"]

        per_class_metrics = evaluator.calculate_per_class_metrics(true_labels, pred_labels)

        # ATTACK class: TP=2, FP=0, FN=1
        attack_metrics = per_class_metrics["ATTACK"]
        assert attack_metrics["true_positives"] == 2
        assert attack_metrics["false_positives"] == 0
        assert attack_metrics["false_negatives"] == 1
        assert attack_metrics["precision"] == 1.0  # 2 / (2 + 0)
        assert abs(attack_metrics["recall"] - (2 / 3)) < 0.001  # 2 / (2 + 1)

        # BENIGN class: TP=0, FP=1, FN=0
        benign_metrics = per_class_metrics["BENIGN"]
        assert benign_metrics["true_positives"] == 0
        assert benign_metrics["false_positives"] == 1
        assert benign_metrics["false_negatives"] == 0
        assert benign_metrics["precision"] == 0.0  # 0 / (0 + 1)
        assert benign_metrics["recall"] == 0.0  # 0 / (0 + 0) -> safe_divide returns 0

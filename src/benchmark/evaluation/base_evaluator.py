"""
Base evaluator class with common evaluation utilities.

This module provides a base implementation of the MetricEvaluator interface
with common utility methods for data validation, statistical calculations,
and result formatting.
"""

from abc import ABC
from typing import Any

import numpy as np

from benchmark.core.logging import get_logger
from benchmark.interfaces.evaluation_interfaces import MetricEvaluator, MetricType


class BaseEvaluator(MetricEvaluator, ABC):
    """Base class with common evaluation utilities."""

    def __init__(self, metric_type: MetricType) -> None:
        """
        Initialize the base evaluator.

        Args:
            metric_type: The type of metric this evaluator handles
        """
        self.metric_type = metric_type
        self.logger = get_logger(f"evaluator_{metric_type.value}")

    def get_metric_type(self) -> MetricType:
        """Get the metric type this evaluator handles."""
        return self.metric_type

    def validate_input_data(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> None:
        """
        Validate input data format and requirements.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Raises:
            ValueError: If input data is invalid
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) must have same length"
            )

        if len(predictions) == 0:
            raise ValueError("Cannot evaluate empty prediction set")

        # Validate required fields in predictions
        required_pred_fields = self.get_required_prediction_fields()
        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                raise ValueError(f"Prediction {i} must be a dictionary")

            for field in required_pred_fields:
                if field not in pred:
                    raise ValueError(f"Missing required prediction field '{field}' in sample {i}")

        # Validate required fields in ground truth
        required_gt_fields = self.get_required_ground_truth_fields()
        for i, gt in enumerate(ground_truth):
            if not isinstance(gt, dict):
                raise ValueError(f"Ground truth {i} must be a dictionary")

            for field in required_gt_fields:
                if field not in gt:
                    raise ValueError(f"Missing required ground truth field '{field}' in sample {i}")

        self.logger.debug(
            f"Validated {len(predictions)} samples for {self.metric_type.value} evaluation"
        )

    def extract_labels(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """
        Extract prediction and ground truth labels.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            Tuple of (predicted_labels, true_labels)

        Raises:
            ValueError: If required fields are missing
        """
        try:
            pred_labels = []
            for i, pred in enumerate(predictions):
                if "prediction" in pred:
                    pred_labels.append(str(pred["prediction"]))
                elif "label" in pred:
                    pred_labels.append(str(pred["label"]))
                else:
                    raise ValueError(f"No prediction or label field found in prediction {i}")

            true_labels = []
            for i, gt in enumerate(ground_truth):
                if "label" in gt:
                    true_labels.append(str(gt["label"]))
                elif "true_label" in gt:
                    true_labels.append(str(gt["true_label"]))
                elif "ground_truth" in gt:
                    true_labels.append(str(gt["ground_truth"]))
                else:
                    raise ValueError(f"No label field found in ground truth {i}")

            return pred_labels, true_labels

        except Exception as e:
            self.logger.error(f"Failed to extract labels: {e}")
            raise

    def extract_confidences(self, predictions: list[dict[str, Any]]) -> list[float]:
        """
        Extract confidence scores from predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            List of confidence scores (defaults to 0.5 if not provided)
        """
        confidences = []
        for pred in predictions:
            confidence = pred.get("confidence")
            if confidence is not None:
                try:
                    confidences.append(float(confidence))
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid confidence value: {confidence}, using 0.5")
                    confidences.append(0.5)
            else:
                confidences.append(0.5)  # Default confidence

        return confidences

    def extract_probabilities(
        self, predictions: list[dict[str, Any]], class_labels: list[str] | None = None
    ) -> list[dict[str, float]]:
        """
        Extract class probabilities from predictions.

        Args:
            predictions: List of prediction dictionaries
            class_labels: Optional list of expected class labels

        Returns:
            List of probability dictionaries for each prediction
        """
        probabilities = []
        for i, pred in enumerate(predictions):
            if "probabilities" in pred:
                prob_dict = pred["probabilities"]
                if isinstance(prob_dict, dict):
                    # Ensure all probabilities are floats
                    normalized_probs = {}
                    for label, prob in prob_dict.items():
                        try:
                            normalized_probs[str(label)] = float(prob)
                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"Invalid probability for label {label} in prediction {i}"
                            )
                            normalized_probs[str(label)] = 0.0
                    probabilities.append(normalized_probs)
                else:
                    self.logger.warning(f"Probabilities not in dict format for prediction {i}")
                    probabilities.append({})
            else:
                # Create probability dict from confidence if available
                confidence = pred.get("confidence", 0.5)
                prediction_label = str(pred.get("prediction", "unknown"))
                prob_dict = {prediction_label: float(confidence)}

                # Add complement probability if binary classification
                if class_labels and len(class_labels) == 2:
                    other_label = [label for label in class_labels if label != prediction_label]
                    if other_label:
                        prob_dict[other_label[0]] = 1.0 - float(confidence)

                probabilities.append(prob_dict)

        return probabilities

    def calculate_basic_stats(self, values: list[float]) -> dict[str, float]:
        """
        Calculate basic statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with statistical measures
        """
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }

        try:
            # Filter out non-numeric values
            numeric_values = []
            for val in values:
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    continue

            if not numeric_values:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                }

            return {
                "count": len(numeric_values),
                "mean": float(np.mean(numeric_values)),
                "std": float(np.std(numeric_values)),
                "min": float(np.min(numeric_values)),
                "max": float(np.max(numeric_values)),
                "median": float(np.median(numeric_values)),
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate statistics: {e}")
            return {
                "count": float(len(values)),
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }

    def calculate_class_distribution(self, labels: list[str]) -> dict[str, dict[str, float]]:
        """
        Calculate class distribution statistics.

        Args:
            labels: List of class labels

        Returns:
            Dictionary with class distribution information
        """
        if not labels:
            return {}

        # Count occurrences
        class_counts: dict[str, int] = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        total_samples = len(labels)

        # Calculate percentages and create distribution
        distribution = {}
        for class_label, count in class_counts.items():
            distribution[class_label] = {
                "count": count,
                "percentage": (count / total_samples) * 100.0,
                "frequency": count / total_samples,
            }

        return distribution

    def normalize_predictions(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Normalize prediction and ground truth data for consistent processing.

        Args:
            predictions: Raw prediction data
            ground_truth: Raw ground truth data

        Returns:
            Tuple of (normalized_predictions, normalized_ground_truth)
        """
        self.validate_input_data(predictions, ground_truth)

        normalized_pred = []
        normalized_gt = []

        for i in range(len(predictions)):
            pred = predictions[i].copy()
            gt = ground_truth[i].copy()

            # Ensure prediction field is standardized
            if "prediction" not in pred and "label" in pred:
                pred["prediction"] = pred["label"]

            # Ensure ground truth label field is standardized
            if "label" not in gt:
                if "true_label" in gt:
                    gt["label"] = gt["true_label"]
                elif "ground_truth" in gt:
                    gt["label"] = gt["ground_truth"]

            # Ensure confidence is present
            if "confidence" not in pred:
                pred["confidence"] = 0.5

            # Add sample index for tracking
            pred["sample_index"] = i
            gt["sample_index"] = i

            normalized_pred.append(pred)
            normalized_gt.append(gt)

        return normalized_pred, normalized_gt

    def format_results(
        self, metrics: dict[str, float], additional_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Format evaluation results with standard structure.

        Args:
            metrics: Dictionary of computed metrics
            additional_data: Optional additional data to include

        Returns:
            Formatted results dictionary
        """
        results = {
            "metrics": metrics,
            "evaluator_type": self.metric_type.value,
            "metric_names": self.get_metric_names(),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

        if additional_data:
            results.update(additional_data)

        return results

    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, returning default if denominator is zero.

        Args:
            numerator: The numerator
            denominator: The denominator
            default: Value to return if denominator is zero

        Returns:
            Division result or default value
        """
        try:
            if denominator == 0:
                return default
            return float(numerator) / float(denominator)
        except (ValueError, TypeError, ZeroDivisionError):
            return default

    def create_confusion_matrix_dict(
        self, true_labels: list[str], pred_labels: list[str]
    ) -> dict[str, Any]:
        """
        Create a confusion matrix as a dictionary.

        Args:
            true_labels: List of true class labels
            pred_labels: List of predicted class labels

        Returns:
            Dictionary representation of confusion matrix
        """
        if len(true_labels) != len(pred_labels):
            raise ValueError("True and predicted labels must have same length")

        # Get unique labels
        all_labels = sorted(set(true_labels + pred_labels))

        # Initialize matrix
        matrix: dict[str, dict[str, int]] = {}
        for true_label in all_labels:
            matrix[true_label] = {}
            for pred_label in all_labels:
                matrix[true_label][pred_label] = 0

        # Populate matrix
        for true_label, pred_label in zip(true_labels, pred_labels, strict=False):
            matrix[true_label][pred_label] += 1

        return {
            "matrix": matrix,
            "labels": all_labels,
            "total_samples": len(true_labels),
        }

    def calculate_per_class_metrics(
        self, true_labels: list[str], pred_labels: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Calculate precision, recall, and F1 score for each class.

        Args:
            true_labels: List of true class labels
            pred_labels: List of predicted class labels

        Returns:
            Dictionary with per-class metrics
        """
        confusion_matrix = self.create_confusion_matrix_dict(true_labels, pred_labels)
        matrix = confusion_matrix["matrix"]
        labels = confusion_matrix["labels"]

        per_class_metrics = {}

        for label in labels:
            # True positives
            tp = matrix[label][label]

            # False positives (predicted as this label but actually other labels)
            fp = sum(matrix[other_label][label] for other_label in labels if other_label != label)

            # False negatives (actually this label but predicted as other labels)
            fn = sum(matrix[label][other_label] for other_label in labels if other_label != label)

            # Calculate metrics
            precision = self.safe_divide(tp, tp + fp)
            recall = self.safe_divide(tp, tp + fn)
            f1_score = self.safe_divide(2 * precision * recall, precision + recall)

            per_class_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": sum(matrix[label].values()),  # Total samples for this class
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }

        return per_class_metrics

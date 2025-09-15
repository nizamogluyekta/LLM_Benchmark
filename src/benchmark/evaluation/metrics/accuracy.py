"""
Comprehensive accuracy evaluation for cybersecurity classification.

This module provides an AccuracyEvaluator that calculates various classification
metrics relevant to cybersecurity attack detection, including standard metrics,
ROC-AUC, per-class analysis, and confusion matrix statistics.
"""

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

from benchmark.evaluation.base_evaluator import BaseEvaluator
from benchmark.interfaces.evaluation_interfaces import MetricType


class AccuracyEvaluator(BaseEvaluator):
    """Comprehensive accuracy evaluation for cybersecurity classification."""

    def __init__(self) -> None:
        """Initialize the accuracy evaluator."""
        super().__init__(MetricType.ACCURACY)
        self.metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "matthews_corr",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "true_positive_rate",
            "false_positive_rate",
            "specificity",
        ]

    async def evaluate(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Calculate comprehensive accuracy metrics.

        Args:
            predictions: List of prediction dictionaries with 'prediction' and 'confidence'
            ground_truth: List of ground truth dictionaries with 'label'

        Returns:
            Dictionary mapping metric names to values

        Raises:
            ValueError: If input data is invalid
        """
        # Validate input data
        self.validate_input_data(predictions, ground_truth)

        # Extract labels and confidences
        pred_labels, true_labels = self.extract_labels(predictions, ground_truth)
        confidences = self.extract_confidences(predictions)

        # Calculate metrics
        metrics: dict[str, float] = {}

        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(pred_labels, true_labels))

        # Probability-based metrics (if confidences available)
        if any(conf != 0.5 for conf in confidences):  # Check if real confidences provided
            metrics.update(
                self._calculate_probability_metrics(
                    pred_labels, true_labels, confidences, predictions
                )
            )
        else:
            # Add zero values for probability metrics when no real confidences
            metrics.update({"roc_auc": 0.0, "pr_auc": 0.0})

        # Per-class metrics
        metrics.update(self._calculate_per_class_metrics(pred_labels, true_labels))

        # Confusion matrix analysis
        metrics.update(self._analyze_confusion_matrix(pred_labels, true_labels))

        return metrics

    def _calculate_basic_metrics(
        self, pred_labels: list[str], true_labels: list[str]
    ) -> dict[str, float]:
        """
        Calculate basic classification metrics.

        Args:
            pred_labels: List of predicted labels
            true_labels: List of true labels

        Returns:
            Dictionary with basic metrics
        """
        # Handle binary vs multi-class
        unique_labels = sorted(set(true_labels + pred_labels))
        is_binary = len(unique_labels) <= 2

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)

        if is_binary:
            # Binary classification metrics
            # For binary, use 'ATTACK' as positive class if available, else use first label
            pos_label = "ATTACK" if "ATTACK" in unique_labels else unique_labels[0]
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average="binary", pos_label=pos_label
            )
            # Also calculate macro averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average="macro"
            )
        else:
            # Multi-class classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average="weighted"
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average="macro"
            )

        # Matthews correlation coefficient
        matthews_corr = matthews_corrcoef(true_labels, pred_labels)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "matthews_corr": float(matthews_corr),
        }

    def _calculate_probability_metrics(
        self,
        pred_labels: list[str],
        true_labels: list[str],
        confidences: list[float],
        predictions: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Calculate probability-based metrics (ROC-AUC, PR-AUC).

        Args:
            pred_labels: List of predicted labels
            true_labels: List of true labels
            confidences: List of confidence scores
            predictions: Original prediction dictionaries

        Returns:
            Dictionary with probability-based metrics
        """
        try:
            # Convert labels to binary for AUC calculation
            unique_labels = sorted(set(true_labels))

            if len(unique_labels) == 2:
                # Binary classification
                # Use 'ATTACK' as positive class if available, else use first unique label
                pos_label = "ATTACK" if "ATTACK" in unique_labels else unique_labels[0]
                binary_true = [1 if label == pos_label else 0 for label in true_labels]

                # Adjust confidences: high confidence for positive class, low for negative
                adjusted_confidences = []
                for i, pred in enumerate(predictions):
                    pred_label = pred.get("prediction", "")
                    if pred_label == pos_label:
                        adjusted_confidences.append(confidences[i])
                    else:
                        adjusted_confidences.append(1.0 - confidences[i])

                # Calculate AUC metrics
                roc_auc = roc_auc_score(binary_true, adjusted_confidences)
                pr_auc = average_precision_score(binary_true, adjusted_confidences)

                return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}
            else:
                # Multi-class: use one-vs-rest approach
                lb = LabelBinarizer()
                binary_true = lb.fit_transform(true_labels)

                # For multi-class, we need probabilities for each class
                # This is simplified - in practice, would need confidence per class
                # For now, use a simple heuristic based on prediction confidence
                n_classes = len(unique_labels)
                class_probs = np.zeros((len(predictions), n_classes))

                for i, pred in enumerate(predictions):
                    pred_label = pred.get("prediction", "")
                    if pred_label in unique_labels:
                        pred_idx = unique_labels.index(pred_label)
                        # Assign confidence to predicted class, distribute remainder
                        class_probs[i, pred_idx] = confidences[i]
                        remaining_prob = (1.0 - confidences[i]) / (n_classes - 1)
                        for j in range(n_classes):
                            if j != pred_idx:
                                class_probs[i, j] = remaining_prob

                # Calculate ROC-AUC for multi-class
                roc_auc = roc_auc_score(
                    binary_true, class_probs, multi_class="ovr", average="macro"
                )

                return {"roc_auc": float(roc_auc), "pr_auc": 0.0}  # PR-AUC complex for multi-class

        except Exception as e:
            # If AUC calculation fails, return zeros
            warnings.warn(f"Failed to calculate probability metrics: {e}", stacklevel=2)
            return {"roc_auc": 0.0, "pr_auc": 0.0}

    def _calculate_per_class_metrics(
        self, pred_labels: list[str], true_labels: list[str]
    ) -> dict[str, float]:
        """
        Calculate per-class precision, recall, and F1 scores.

        Args:
            pred_labels: List of predicted labels
            true_labels: List of true labels

        Returns:
            Dictionary with per-class metrics
        """
        unique_labels = sorted(set(true_labels + pred_labels))

        # Get per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = (
            precision_recall_fscore_support(
                true_labels, pred_labels, labels=unique_labels, average=None, zero_division=0
            )
        )

        per_class_metrics: dict[str, float] = {}

        for i, label in enumerate(unique_labels):
            # Use label directly for key (sanitized)
            safe_label = label.lower().replace(" ", "_").replace("-", "_")
            per_class_metrics[f"{safe_label}_precision"] = float(precision_per_class[i])
            per_class_metrics[f"{safe_label}_recall"] = float(recall_per_class[i])
            per_class_metrics[f"{safe_label}_f1"] = float(f1_per_class[i])
            per_class_metrics[f"{safe_label}_support"] = float(support[i])

        return per_class_metrics

    def _analyze_confusion_matrix(
        self, pred_labels: list[str], true_labels: list[str]
    ) -> dict[str, float]:
        """
        Analyze confusion matrix and calculate derived metrics.

        Args:
            pred_labels: List of predicted labels
            true_labels: List of true labels

        Returns:
            Dictionary with confusion matrix derived metrics
        """
        unique_labels = sorted(set(true_labels + pred_labels))
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

        metrics: dict[str, float] = {}

        if len(unique_labels) == 2:
            # Binary classification confusion matrix analysis
            # Determine which class should be positive
            pos_label = "ATTACK" if "ATTACK" in unique_labels else unique_labels[0]
            pos_idx = unique_labels.index(pos_label)
            neg_idx = 1 - pos_idx

            # Extract confusion matrix values based on positive class
            tp = cm[pos_idx, pos_idx]  # True positives
            fn = cm[pos_idx, neg_idx]  # False negatives
            fp = cm[neg_idx, pos_idx]  # False positives
            tn = cm[neg_idx, neg_idx]  # True negatives

            # Calculate additional metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

            metrics.update(
                {
                    "true_positive_rate": float(tpr),
                    "false_positive_rate": float(fpr),
                    "true_negative_rate": float(tnr),
                    "false_negative_rate": float(fnr),
                    "specificity": float(tnr),
                    "sensitivity": float(tpr),
                    "true_positives": float(tp),
                    "false_positives": float(fp),
                    "true_negatives": float(tn),
                    "false_negatives": float(fn),
                }
            )
        else:
            # Multi-class confusion matrix - calculate macro averages
            total_samples = cm.sum()

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

            metrics.update(
                {
                    "true_positive_rate": float(np.mean(tprs)),
                    "false_positive_rate": float(np.mean(fprs)),
                    "specificity": float(1.0 - np.mean(fprs)),
                }
            )

        return metrics

    def get_metric_names(self) -> list[str]:
        """Get list of metrics this evaluator produces."""
        return self.metric_names.copy()

    def get_required_prediction_fields(self) -> list[str]:
        """Get required fields in prediction data."""
        return ["prediction"]

    def get_required_ground_truth_fields(self) -> list[str]:
        """Get required fields in ground truth data."""
        return ["label"]

    def generate_detailed_report(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            Detailed classification report as string
        """
        pred_labels, true_labels = self.extract_labels(predictions, ground_truth)

        # Generate sklearn classification report
        report = classification_report(true_labels, pred_labels, output_dict=False, zero_division=0)

        # Add confusion matrix
        unique_labels = sorted(set(true_labels + pred_labels))
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

"""
Sample test data for accuracy evaluation.

This module provides various test data scenarios for testing the AccuracyEvaluator,
including perfect predictions, imperfect predictions, and multi-class scenarios.
"""

# Perfect binary classification - all predictions correct
PERFECT_BINARY_PREDICTIONS = [
    {"prediction": "ATTACK", "confidence": 0.95},
    {"prediction": "BENIGN", "confidence": 0.85},
    {"prediction": "ATTACK", "confidence": 0.90},
    {"prediction": "BENIGN", "confidence": 0.88},
    {"prediction": "ATTACK", "confidence": 0.92},
]

PERFECT_BINARY_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
]

# Imperfect binary classification - some errors
IMPERFECT_BINARY_PREDICTIONS = [
    {"prediction": "ATTACK", "confidence": 0.75},  # Correct
    {"prediction": "ATTACK", "confidence": 0.65},  # False positive
    {"prediction": "BENIGN", "confidence": 0.80},  # False negative
    {"prediction": "BENIGN", "confidence": 0.90},  # Correct
    {"prediction": "ATTACK", "confidence": 0.70},  # Correct
]

IMPERFECT_BINARY_GROUND_TRUTH = [
    {"label": "ATTACK"},  # TP
    {"label": "BENIGN"},  # FP
    {"label": "ATTACK"},  # FN
    {"label": "BENIGN"},  # TN
    {"label": "ATTACK"},  # TP
]

# Multi-class classification - cybersecurity attack types
MULTICLASS_PREDICTIONS = [
    {"prediction": "malware", "confidence": 0.90},
    {"prediction": "intrusion", "confidence": 0.85},
    {"prediction": "dos", "confidence": 0.75},
    {"prediction": "benign", "confidence": 0.95},
    {"prediction": "malware", "confidence": 0.88},
    {"prediction": "intrusion", "confidence": 0.82},
    {"prediction": "phishing", "confidence": 0.78},
    {"prediction": "dos", "confidence": 0.73},
]

MULTICLASS_GROUND_TRUTH = [
    {"label": "malware"},
    {"label": "intrusion"},
    {"label": "dos"},
    {"label": "benign"},
    {"label": "malware"},
    {"label": "intrusion"},
    {"label": "phishing"},
    {"label": "dos"},
]

# Multi-class with some errors
MULTICLASS_IMPERFECT_PREDICTIONS = [
    {"prediction": "malware", "confidence": 0.90},  # Correct
    {"prediction": "dos", "confidence": 0.85},  # Wrong (should be intrusion)
    {"prediction": "dos", "confidence": 0.75},  # Correct
    {"prediction": "benign", "confidence": 0.95},  # Correct
    {"prediction": "phishing", "confidence": 0.88},  # Wrong (should be malware)
    {"prediction": "intrusion", "confidence": 0.82},  # Correct
]

MULTICLASS_IMPERFECT_GROUND_TRUTH = [
    {"label": "malware"},
    {"label": "intrusion"},
    {"label": "dos"},
    {"label": "benign"},
    {"label": "malware"},
    {"label": "intrusion"},
]

# Edge case: Single class predictions
SINGLE_CLASS_PREDICTIONS = [
    {"prediction": "ATTACK", "confidence": 0.95},
    {"prediction": "ATTACK", "confidence": 0.85},
    {"prediction": "ATTACK", "confidence": 0.90},
]

SINGLE_CLASS_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "ATTACK"},
    {"label": "ATTACK"},
]

# Edge case: Default confidences (all 0.5)
DEFAULT_CONFIDENCE_PREDICTIONS = [
    {"prediction": "ATTACK"},  # No confidence provided
    {"prediction": "BENIGN"},
    {"prediction": "ATTACK"},
]

DEFAULT_CONFIDENCE_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
]

# High confidence predictions for probability metrics testing
HIGH_CONFIDENCE_PREDICTIONS = [
    {"prediction": "ATTACK", "confidence": 0.98},
    {"prediction": "BENIGN", "confidence": 0.95},
    {"prediction": "ATTACK", "confidence": 0.96},
    {"prediction": "BENIGN", "confidence": 0.97},
]

HIGH_CONFIDENCE_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
]

# Low confidence predictions (near threshold)
LOW_CONFIDENCE_PREDICTIONS = [
    {"prediction": "ATTACK", "confidence": 0.52},
    {"prediction": "BENIGN", "confidence": 0.51},
    {"prediction": "ATTACK", "confidence": 0.53},
    {"prediction": "BENIGN", "confidence": 0.54},
]

LOW_CONFIDENCE_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
]

# Mixed confidence with errors for ROC analysis
MIXED_CONFIDENCE_PREDICTIONS = [
    {"prediction": "ATTACK", "confidence": 0.95},  # TP - high conf
    {"prediction": "ATTACK", "confidence": 0.55},  # FP - low conf
    {"prediction": "BENIGN", "confidence": 0.90},  # TN - high conf
    {"prediction": "BENIGN", "confidence": 0.60},  # FN - low conf
    {"prediction": "ATTACK", "confidence": 0.85},  # TP - high conf
    {"prediction": "BENIGN", "confidence": 0.75},  # TN - high conf
]

MIXED_CONFIDENCE_GROUND_TRUTH = [
    {"label": "ATTACK"},  # TP
    {"label": "BENIGN"},  # FP
    {"label": "BENIGN"},  # TN
    {"label": "ATTACK"},  # FN
    {"label": "ATTACK"},  # TP
    {"label": "BENIGN"},  # TN
]

# Alternative field names for testing robustness
ALTERNATIVE_FIELDS_PREDICTIONS = [
    {"label": "ATTACK", "confidence": 0.95},  # Using 'label' instead of 'prediction'
    {"label": "BENIGN", "confidence": 0.85},
]

ALTERNATIVE_FIELDS_GROUND_TRUTH = [
    {"true_label": "ATTACK"},  # Using 'true_label' instead of 'label'
    {"ground_truth": "BENIGN"},  # Using 'ground_truth' instead of 'label'
]

# Expected results for validation
PERFECT_BINARY_EXPECTED_METRICS = {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "true_positive_rate": 1.0,
    "false_positive_rate": 0.0,
    "specificity": 1.0,
}

IMPERFECT_BINARY_EXPECTED_METRICS = {
    "accuracy": 0.6,  # 3/5 correct
    "precision": 0.6666666666666666,  # 2/3 (2 TP out of 3 predicted attacks)
    "recall": 0.6666666666666666,  # 2/3 (2 TP out of 3 actual attacks)
    "true_positives": 2.0,
    "false_positives": 1.0,
    "true_negatives": 1.0,
    "false_negatives": 1.0,
}

# Test scenarios collection
TEST_SCENARIOS = {
    "perfect_binary": {
        "predictions": PERFECT_BINARY_PREDICTIONS,
        "ground_truth": PERFECT_BINARY_GROUND_TRUTH,
        "description": "Perfect binary classification",
    },
    "imperfect_binary": {
        "predictions": IMPERFECT_BINARY_PREDICTIONS,
        "ground_truth": IMPERFECT_BINARY_GROUND_TRUTH,
        "description": "Imperfect binary classification with errors",
    },
    "multiclass_perfect": {
        "predictions": MULTICLASS_PREDICTIONS,
        "ground_truth": MULTICLASS_GROUND_TRUTH,
        "description": "Perfect multi-class classification",
    },
    "multiclass_imperfect": {
        "predictions": MULTICLASS_IMPERFECT_PREDICTIONS,
        "ground_truth": MULTICLASS_IMPERFECT_GROUND_TRUTH,
        "description": "Imperfect multi-class classification",
    },
    "single_class": {
        "predictions": SINGLE_CLASS_PREDICTIONS,
        "ground_truth": SINGLE_CLASS_GROUND_TRUTH,
        "description": "Single class edge case",
    },
    "high_confidence": {
        "predictions": HIGH_CONFIDENCE_PREDICTIONS,
        "ground_truth": HIGH_CONFIDENCE_GROUND_TRUTH,
        "description": "High confidence predictions for probability metrics",
    },
    "mixed_confidence": {
        "predictions": MIXED_CONFIDENCE_PREDICTIONS,
        "ground_truth": MIXED_CONFIDENCE_GROUND_TRUTH,
        "description": "Mixed confidence with errors for ROC analysis",
    },
}

"""
Sample test data for performance evaluation.

This module provides various test data scenarios for testing the PerformanceEvaluator,
including consistent performance, variable performance, and edge cases.
"""

# Consistent performance data - low variance
CONSISTENT_PERFORMANCE_DATA = [
    {"prediction": "ATTACK", "inference_time_ms": 100.0, "output_tokens": 20},
    {"prediction": "BENIGN", "inference_time_ms": 102.0, "output_tokens": 18},
    {"prediction": "ATTACK", "inference_time_ms": 98.0, "output_tokens": 22},
    {"prediction": "BENIGN", "inference_time_ms": 101.0, "output_tokens": 19},
    {"prediction": "ATTACK", "inference_time_ms": 99.0, "output_tokens": 21},
]

CONSISTENT_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
]

# Variable performance data - high variance with outliers
VARIABLE_PERFORMANCE_DATA = [
    {"prediction": "ATTACK", "inference_time_ms": 50.0},
    {"prediction": "BENIGN", "inference_time_ms": 200.0},  # Outlier
    {"prediction": "ATTACK", "inference_time_ms": 75.0},
    {"prediction": "BENIGN", "inference_time_ms": 80.0},
    {"prediction": "ATTACK", "inference_time_ms": 300.0},  # Major outlier
    {"prediction": "BENIGN", "inference_time_ms": 65.0},
    {"prediction": "ATTACK", "inference_time_ms": 70.0},
    {"prediction": "BENIGN", "inference_time_ms": 85.0},
]

VARIABLE_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
]

# Performance data with timestamps for trend analysis
PERFORMANCE_WITH_TIMESTAMPS = [
    {
        "prediction": "ATTACK",
        "inference_time_ms": 100.0,
        "timestamp": "2024-01-01T10:00:00",
        "output_tokens": 20,
    },
    {
        "prediction": "BENIGN",
        "inference_time_ms": 110.0,
        "timestamp": "2024-01-01T10:01:00",
        "output_tokens": 18,
    },
    {
        "prediction": "ATTACK",
        "inference_time_ms": 120.0,
        "timestamp": "2024-01-01T10:02:00",
        "output_tokens": 22,
    },
    {
        "prediction": "BENIGN",
        "inference_time_ms": 130.0,
        "timestamp": "2024-01-01T10:03:00",
        "output_tokens": 19,
    },
    {
        "prediction": "ATTACK",
        "inference_time_ms": 140.0,
        "timestamp": "2024-01-01T10:04:00",
        "output_tokens": 21,
    },
    {
        "prediction": "BENIGN",
        "inference_time_ms": 150.0,
        "timestamp": "2024-01-01T10:05:00",
        "output_tokens": 23,
    },
    {
        "prediction": "ATTACK",
        "inference_time_ms": 160.0,
        "timestamp": "2024-01-01T10:06:00",
        "output_tokens": 25,
    },
    {
        "prediction": "BENIGN",
        "inference_time_ms": 170.0,
        "timestamp": "2024-01-01T10:07:00",
        "output_tokens": 20,
    },
    {
        "prediction": "ATTACK",
        "inference_time_ms": 180.0,
        "timestamp": "2024-01-01T10:08:00",
        "output_tokens": 24,
    },
    {
        "prediction": "BENIGN",
        "inference_time_ms": 190.0,
        "timestamp": "2024-01-01T10:09:00",
        "output_tokens": 22,
    },
]

TIMESTAMPS_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
]

# Fast performance data - optimized model
FAST_PERFORMANCE_DATA = [
    {"prediction": "ATTACK", "inference_time_ms": 10.0, "tokens": 50},
    {"prediction": "BENIGN", "inference_time_ms": 12.0, "tokens": 45},
    {"prediction": "ATTACK", "inference_time_ms": 8.0, "tokens": 55},
    {"prediction": "BENIGN", "inference_time_ms": 11.0, "tokens": 48},
    {"prediction": "ATTACK", "inference_time_ms": 9.0, "tokens": 52},
]

FAST_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
]

# Slow performance data - heavy model
SLOW_PERFORMANCE_DATA = [
    {"prediction": "ATTACK", "inference_time_ms": 2000.0, "response_tokens": 100},
    {"prediction": "BENIGN", "inference_time_ms": 2100.0, "response_tokens": 95},
    {"prediction": "ATTACK", "inference_time_ms": 1900.0, "response_tokens": 105},
    {"prediction": "BENIGN", "inference_time_ms": 2050.0, "response_tokens": 98},
]

SLOW_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
]

# Alternative timing field names
ALTERNATIVE_TIMING_FIELDS = [
    {"prediction": "ATTACK", "inference_time_sec": 0.1, "output_tokens": 20},  # seconds
    {"prediction": "BENIGN", "processing_time_ms": 150.0, "output_tokens": 18},  # processing_time
    {"prediction": "ATTACK", "latency_ms": 120.0, "output_tokens": 22},  # latency
    {"prediction": "BENIGN", "inference_time_ms": 110.0, "output_tokens": 19},  # standard field
]

ALTERNATIVE_TIMING_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
]

# Edge case: No timing data
NO_TIMING_DATA = [
    {"prediction": "ATTACK", "confidence": 0.9},
    {"prediction": "BENIGN", "confidence": 0.8},
    {"prediction": "ATTACK", "confidence": 0.85},
]

NO_TIMING_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
]

# Edge case: Single sample
SINGLE_SAMPLE_DATA = [{"prediction": "ATTACK", "inference_time_ms": 100.0, "output_tokens": 20}]

SINGLE_SAMPLE_GROUND_TRUTH = [{"label": "ATTACK"}]

# Edge case: Invalid timing values
INVALID_TIMING_DATA = [
    {"prediction": "ATTACK", "inference_time_ms": -10.0},  # Negative time
    {"prediction": "BENIGN", "inference_time_ms": 0.0},  # Zero time
    {"prediction": "ATTACK", "inference_time_ms": None},  # None value
    {"prediction": "BENIGN", "inference_time_ms": 150.0},  # Valid time
]

INVALID_TIMING_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
]

# Token performance data - focus on token-based metrics
TOKEN_FOCUSED_DATA = [
    {"prediction": "ATTACK", "inference_time_ms": 100.0, "tokens_generated": 10},
    {
        "prediction": "BENIGN",
        "inference_time_ms": 200.0,
        "tokens_generated": 40,
    },  # More tokens, longer time
    {"prediction": "ATTACK", "inference_time_ms": 150.0, "tokens_generated": 15},
    {"prediction": "BENIGN", "inference_time_ms": 300.0, "tokens_generated": 60},
    {"prediction": "ATTACK", "inference_time_ms": 250.0, "tokens_generated": 25},
]

TOKEN_FOCUSED_GROUND_TRUTH = [
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
    {"label": "BENIGN"},
    {"label": "ATTACK"},
]

# Expected metrics for validation
CONSISTENT_PERFORMANCE_EXPECTED = {
    "avg_inference_time_ms": 100.0,  # Mean of [100, 102, 98, 101, 99]
    "median_inference_time_ms": 100.0,  # Median of sorted values
    "min_inference_time_ms": 98.0,
    "max_inference_time_ms": 102.0,
    "outlier_percentage": 0.0,  # No outliers in consistent data
}

VARIABLE_PERFORMANCE_EXPECTED = {
    "min_inference_time_ms": 50.0,
    "max_inference_time_ms": 300.0,
    # Outlier percentage should be > 0 due to 200ms and 300ms outliers
}

FAST_PERFORMANCE_EXPECTED = {
    "avg_inference_time_ms": 10.0,  # Mean of [10, 12, 8, 11, 9]
    "throughput_samples_per_sec": 100.0,  # High throughput due to fast inference
}

# Performance scenarios collection
PERFORMANCE_SCENARIOS = {
    "consistent": {
        "predictions": CONSISTENT_PERFORMANCE_DATA,
        "ground_truth": CONSISTENT_GROUND_TRUTH,
        "description": "Consistent performance with low variance",
    },
    "variable": {
        "predictions": VARIABLE_PERFORMANCE_DATA,
        "ground_truth": VARIABLE_GROUND_TRUTH,
        "description": "Variable performance with outliers",
    },
    "fast": {
        "predictions": FAST_PERFORMANCE_DATA,
        "ground_truth": FAST_GROUND_TRUTH,
        "description": "Fast inference performance",
    },
    "slow": {
        "predictions": SLOW_PERFORMANCE_DATA,
        "ground_truth": SLOW_GROUND_TRUTH,
        "description": "Slow inference performance",
    },
    "token_focused": {
        "predictions": TOKEN_FOCUSED_DATA,
        "ground_truth": TOKEN_FOCUSED_GROUND_TRUTH,
        "description": "Token generation performance analysis",
    },
    "alternative_fields": {
        "predictions": ALTERNATIVE_TIMING_FIELDS,
        "ground_truth": ALTERNATIVE_TIMING_GROUND_TRUTH,
        "description": "Alternative timing field names",
    },
    "with_timestamps": {
        "predictions": PERFORMANCE_WITH_TIMESTAMPS,
        "ground_truth": TIMESTAMPS_GROUND_TRUTH,
        "description": "Performance data with timestamps for trend analysis",
    },
}

# Degrading performance trend for trend analysis
DEGRADING_TREND_DATA = []
for i in range(20):
    # Performance gradually degrades over time
    base_time = 100.0 + i * 10.0  # Increases by 10ms each iteration
    DEGRADING_TREND_DATA.append(
        {
            "prediction": "ATTACK" if i % 2 == 0 else "BENIGN",
            "inference_time_ms": base_time + (i % 3) * 5.0,  # Add some variance
            "timestamp": f"2024-01-01T10:{i:02d}:00",
            "output_tokens": 20 + (i % 5),
        }
    )

DEGRADING_TREND_GROUND_TRUTH = [{"label": "ATTACK" if i % 2 == 0 else "BENIGN"} for i in range(20)]

# Improving performance trend
IMPROVING_TREND_DATA = []
for i in range(20):
    # Performance gradually improves over time
    base_time = 200.0 - i * 5.0  # Decreases by 5ms each iteration
    IMPROVING_TREND_DATA.append(
        {
            "prediction": "ATTACK" if i % 2 == 0 else "BENIGN",
            "inference_time_ms": max(50.0, base_time + (i % 3) * 2.0),  # Min 50ms
            "timestamp": f"2024-01-01T10:{i:02d}:00",
            "output_tokens": 20 + (i % 5),
        }
    )

IMPROVING_TREND_GROUND_TRUTH = [{"label": "ATTACK" if i % 2 == 0 else "BENIGN"} for i in range(20)]

# Stable performance trend
STABLE_TREND_DATA = []
for i in range(20):
    # Performance remains stable over time
    base_time = 100.0
    STABLE_TREND_DATA.append(
        {
            "prediction": "ATTACK" if i % 2 == 0 else "BENIGN",
            "inference_time_ms": base_time + (i % 5) * 2.0,  # Small random variance
            "timestamp": f"2024-01-01T10:{i:02d}:00",
            "output_tokens": 20 + (i % 3),
        }
    )

STABLE_TREND_GROUND_TRUTH = [{"label": "ATTACK" if i % 2 == 0 else "BENIGN"} for i in range(20)]

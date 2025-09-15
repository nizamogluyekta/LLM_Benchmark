"""
Comprehensive test suite for evaluation results storage system.

Tests storage, retrieval, querying, export functionality, and data integrity
validation for the evaluation results storage system.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from benchmark.evaluation.result_models import (
    EvaluationResult,
    ModelPerformanceHistory,
    ResultsQuery,
)
from benchmark.evaluation.results_storage import ResultsStorage, StorageError


class TestEvaluationResultModels:
    """Test cases for evaluation result data models."""

    def test_evaluation_result_creation(self):
        """Test creating EvaluationResult with valid data."""
        result = EvaluationResult(
            evaluation_id="test_001",
            model_name="test_model",
            task_type="classification",
            dataset_name="test_dataset",
            metrics={"accuracy": 0.95, "f1_score": 0.93},
            timestamp=datetime.now(),
            configuration={"learning_rate": 0.001},
            raw_responses=[{"input": "test", "output": "result"}],
            processing_time=10.5,
        )

        assert result.evaluation_id == "test_001"
        assert result.model_name == "test_model"
        assert result.metrics["accuracy"] == 0.95
        assert result.success_rate == 1.0  # default value
        assert result.error_count == 0  # default value

    def test_evaluation_result_validation(self):
        """Test data validation in EvaluationResult."""
        # Test empty evaluation_id
        with pytest.raises(ValueError, match="evaluation_id cannot be empty"):
            EvaluationResult(
                evaluation_id="",
                model_name="test_model",
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": 0.95},
                timestamp=datetime.now(),
                configuration={},
                raw_responses=[],
                processing_time=10.0,
            )

        # Test negative processing time
        with pytest.raises(ValueError, match="processing_time cannot be negative"):
            EvaluationResult(
                evaluation_id="test_001",
                model_name="test_model",
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": 0.95},
                timestamp=datetime.now(),
                configuration={},
                raw_responses=[],
                processing_time=-1.0,
            )

        # Test invalid success rate
        with pytest.raises(ValueError, match="success_rate must be between 0 and 1"):
            EvaluationResult(
                evaluation_id="test_001",
                model_name="test_model",
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": 0.95},
                timestamp=datetime.now(),
                configuration={},
                raw_responses=[],
                processing_time=10.0,
                success_rate=1.5,
            )

    def test_evaluation_result_serialization(self):
        """Test serialization and deserialization of EvaluationResult."""
        original = EvaluationResult(
            evaluation_id="test_001",
            model_name="test_model",
            task_type="classification",
            dataset_name="test_dataset",
            metrics={"accuracy": 0.95, "f1_score": 0.93},
            timestamp=datetime.now(),
            configuration={"learning_rate": 0.001, "batch_size": 32},
            raw_responses=[{"input": "test", "output": "result"}],
            processing_time=10.5,
            tags=["experiment_1", "baseline"],
            notes="Test evaluation run",
        )

        # Test dict conversion
        result_dict = original.to_dict()
        assert result_dict["evaluation_id"] == "test_001"
        assert result_dict["metrics"]["accuracy"] == 0.95
        assert isinstance(result_dict["timestamp"], str)

        # Test round-trip conversion
        restored = EvaluationResult.from_dict(result_dict)
        assert restored.evaluation_id == original.evaluation_id
        assert restored.metrics == original.metrics
        assert restored.tags == original.tags
        assert restored.notes == original.notes

    def test_evaluation_result_primary_metric(self):
        """Test getting primary metric from evaluation result."""
        result = EvaluationResult(
            evaluation_id="test_001",
            model_name="test_model",
            task_type="classification",
            dataset_name="test_dataset",
            metrics={"precision": 0.92, "recall": 0.89, "f1_score": 0.90},
            timestamp=datetime.now(),
            configuration={},
            raw_responses=[],
            processing_time=10.0,
        )

        # Should return f1_score as it's in the primary metrics list
        assert result.get_primary_metric() == 0.90

        # Test with only custom metrics
        result_custom = EvaluationResult(
            evaluation_id="test_002",
            model_name="test_model",
            task_type="custom",
            dataset_name="test_dataset",
            metrics={"custom_score": 0.85},
            timestamp=datetime.now(),
            configuration={},
            raw_responses=[],
            processing_time=10.0,
        )

        # Should return the custom metric
        assert result_custom.get_primary_metric() == 0.85

    def test_model_performance_history(self):
        """Test ModelPerformanceHistory functionality."""
        history = ModelPerformanceHistory("test_model")

        # Create test results
        result1 = EvaluationResult(
            evaluation_id="test_001",
            model_name="test_model",
            task_type="classification",
            dataset_name="dataset_1",
            metrics={"accuracy": 0.85},
            timestamp=datetime.now() - timedelta(days=2),
            configuration={},
            raw_responses=[],
            processing_time=10.0,
        )

        result2 = EvaluationResult(
            evaluation_id="test_002",
            model_name="test_model",
            task_type="classification",
            dataset_name="dataset_1",
            metrics={"accuracy": 0.90},
            timestamp=datetime.now() - timedelta(days=1),
            configuration={},
            raw_responses=[],
            processing_time=8.0,
        )

        # Add evaluations
        history.add_evaluation(result1)
        history.add_evaluation(result2)

        # Test latest result
        latest = history.get_latest_result()
        assert latest.evaluation_id == "test_002"

        # Test best result
        best = history.get_best_result("accuracy")
        assert best.evaluation_id == "test_002"
        assert best.metrics["accuracy"] == 0.90

        # Test average metric
        avg_accuracy = history.get_average_metric("accuracy")
        assert avg_accuracy == 0.875  # (0.85 + 0.90) / 2

        # Test metric trends
        trends = history.get_metric_trend("accuracy")
        assert len(trends) == 2
        assert trends[0][1] == 0.85  # First evaluation (oldest)
        assert trends[1][1] == 0.90  # Second evaluation (newest)

    def test_results_query(self):
        """Test ResultsQuery parameter handling."""
        query = ResultsQuery(
            model_name="test_model",
            task_type="classification",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            tags=["experiment_1"],
            min_success_rate=0.8,
            metric_filters={"accuracy": (0.7, 1.0)},
            limit=10,
            sort_by="timestamp",
            sort_order="desc",
        )

        query_dict = query.to_dict()
        assert query_dict["model_name"] == "test_model"
        assert query_dict["task_type"] == "classification"
        assert isinstance(query_dict["start_date"], str)
        assert query_dict["tags"] == ["experiment_1"]
        assert query_dict["metric_filters"]["accuracy"] == (0.7, 1.0)


class TestResultsStorage:
    """Test cases for ResultsStorage functionality."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def storage(self, temp_storage_path):
        """Create ResultsStorage instance with temporary directory."""
        return ResultsStorage(temp_storage_path)

    @pytest.fixture
    def sample_result(self):
        """Create sample evaluation result for testing."""
        return EvaluationResult(
            evaluation_id="test_eval_001",
            model_name="bert_base",
            task_type="text_classification",
            dataset_name="imdb_sentiment",
            metrics={
                "accuracy": 0.892,
                "precision": 0.885,
                "recall": 0.898,
                "f1_score": 0.891,
            },
            timestamp=datetime.now(),
            configuration={
                "learning_rate": 0.0001,
                "batch_size": 16,
                "epochs": 3,
                "optimizer": "adam",
            },
            raw_responses=[
                {"input": "This movie is great!", "predicted": "positive", "actual": "positive"},
                {"input": "Terrible film", "predicted": "negative", "actual": "negative"},
            ],
            processing_time=45.2,
            model_version="1.0",
            dataset_version="2.1",
            experiment_name="sentiment_analysis_exp",
            tags=["baseline", "bert"],
            notes="Baseline model evaluation",
            error_count=0,
            success_rate=1.0,
        )

    def test_storage_initialization(self, temp_storage_path):
        """Test storage initialization and database creation."""
        storage = ResultsStorage(temp_storage_path)

        # Check that database file was created
        assert storage.db_path.exists()

        # Check that tables were created
        import sqlite3

        with sqlite3.connect(storage.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                "evaluation_results",
                "result_metrics",
                "result_configurations",
                "raw_responses",
                "result_tags",
            }
            assert expected_tables.issubset(tables)

    def test_store_evaluation_result(self, storage, sample_result):
        """Test storing evaluation results."""
        # Store the result
        result_id = storage.store_evaluation_result(sample_result)
        assert result_id == sample_result.evaluation_id

        # Verify it was stored correctly
        retrieved = storage.get_evaluation_by_id(sample_result.evaluation_id)
        assert retrieved is not None
        assert retrieved.evaluation_id == sample_result.evaluation_id
        assert retrieved.model_name == sample_result.model_name
        assert retrieved.metrics == sample_result.metrics
        assert retrieved.configuration == sample_result.configuration
        assert retrieved.raw_responses == sample_result.raw_responses
        assert retrieved.tags == sample_result.tags

    def test_store_duplicate_evaluation(self, storage, sample_result):
        """Test storing duplicate evaluation (should update)."""
        # Store original result
        storage.store_evaluation_result(sample_result)

        # Modify the result
        sample_result.metrics["accuracy"] = 0.95
        sample_result.notes = "Updated evaluation"

        # Store again (should update)
        result_id = storage.store_evaluation_result(sample_result)
        assert result_id == sample_result.evaluation_id

        # Verify update
        retrieved = storage.get_evaluation_by_id(sample_result.evaluation_id)
        assert retrieved.metrics["accuracy"] == 0.95
        assert retrieved.notes == "Updated evaluation"

    def test_query_results_basic(self, storage, sample_result):
        """Test basic result querying."""
        storage.store_evaluation_result(sample_result)

        # Query all results
        all_results = storage.query_results()
        assert len(all_results) == 1
        assert all_results[0].evaluation_id == sample_result.evaluation_id

        # Query by model name
        model_results = storage.query_results({"model_name": "bert_base"})
        assert len(model_results) == 1

        # Query by non-existent model
        no_results = storage.query_results({"model_name": "nonexistent"})
        assert len(no_results) == 0

    def test_query_results_advanced(self, storage):
        """Test advanced querying with multiple filters."""
        # Create multiple test results
        results = []
        for i in range(5):
            result = EvaluationResult(
                evaluation_id=f"test_{i:03d}",
                model_name=f"model_{i % 2}",  # alternating models
                task_type="classification",
                dataset_name="test_dataset",
                metrics={"accuracy": 0.8 + i * 0.02},  # increasing accuracy
                timestamp=datetime.now() - timedelta(days=i),
                configuration={"param": i},
                raw_responses=[],
                processing_time=10.0 + i,
                tags=["tag_a"] if i % 2 == 0 else ["tag_b"],
            )
            results.append(result)
            storage.store_evaluation_result(result)

        # Test filtering by model
        model_0_results = storage.query_results({"model_name": "model_0"})
        assert len(model_0_results) == 3  # indices 0, 2, 4

        # Test filtering by tags
        tag_a_results = storage.query_results({"tags": ["tag_a"]})
        assert len(tag_a_results) == 3

        # Test metric filtering (values are 0.80, 0.82, 0.84, 0.86, 0.88)
        high_accuracy_results = storage.query_results({"metric_filters": {"accuracy": (0.85, 1.0)}})
        assert len(high_accuracy_results) == 2  # only last 2 results (0.86, 0.88) are >= 0.85

        # Test date filtering (timestamps go: today, yesterday, 2 days ago, 3 days ago, 4 days ago)
        recent_results = storage.query_results(
            {
                "start_date": datetime.now()
                - timedelta(days=1.5)  # Get results from last ~1.5 days
            }
        )
        assert len(recent_results) == 2  # today and yesterday

        # Test sorting
        sorted_results = storage.query_results({"sort_by": "metric:accuracy", "sort_order": "desc"})
        assert sorted_results[0].metrics["accuracy"] == 0.88  # highest accuracy
        assert sorted_results[-1].metrics["accuracy"] == 0.80  # lowest accuracy

        # Test pagination
        page_1 = storage.query_results({"limit": 2, "offset": 0})
        page_2 = storage.query_results({"limit": 2, "offset": 2})
        assert len(page_1) == 2
        assert len(page_2) == 2
        assert page_1[0].evaluation_id != page_2[0].evaluation_id

    def test_model_performance_history(self, storage):
        """Test model performance history tracking."""
        # Create multiple evaluations for the same model
        model_name = "test_model"
        for i in range(3):
            result = EvaluationResult(
                evaluation_id=f"eval_{i}",
                model_name=model_name,
                task_type="classification",
                dataset_name=f"dataset_{i}",
                metrics={"accuracy": 0.7 + i * 0.1},
                timestamp=datetime.now() - timedelta(days=2 - i),  # chronological order
                configuration={},
                raw_responses=[],
                processing_time=10.0,
            )
            storage.store_evaluation_result(result)

        # Get performance history
        history = storage.get_model_performance_history(model_name)

        assert history.model_name == model_name
        assert len(history.evaluations) == 3

        # Check chronological ordering
        assert history.evaluations[0].evaluation_id == "eval_0"
        assert history.evaluations[-1].evaluation_id == "eval_2"

        # Test best performance
        best = history.get_best_result("accuracy")
        assert best.evaluation_id == "eval_2"
        assert abs(best.metrics["accuracy"] - 0.9) < 0.0001  # Handle floating point precision

        # Test average performance
        avg_accuracy = history.get_average_metric("accuracy")
        assert abs(avg_accuracy - 0.8) < 0.0001  # (0.7 + 0.8 + 0.9) / 3

    def test_export_json(self, storage, sample_result):
        """Test JSON export functionality."""
        storage.store_evaluation_result(sample_result)

        # Export to JSON string
        json_data = storage.export_results(format="json")

        # Verify JSON structure
        export_dict = json.loads(json_data)
        assert "export_timestamp" in export_dict
        assert export_dict["result_count"] == 1
        assert len(export_dict["results"]) == 1

        result_data = export_dict["results"][0]
        assert result_data["evaluation_id"] == sample_result.evaluation_id
        assert result_data["model_name"] == sample_result.model_name
        assert result_data["metrics"] == sample_result.metrics

    def test_export_csv(self, storage, sample_result):
        """Test CSV export functionality."""
        storage.store_evaluation_result(sample_result)

        # Export to CSV string
        csv_data = storage.export_results(format="csv")

        # Verify CSV structure
        lines = csv_data.strip().split("\n")
        assert len(lines) >= 2  # header + data row

        header = lines[0]
        assert "evaluation_id" in header
        assert "model_name" in header
        assert "metric_accuracy" in header
        assert "config_learning_rate" in header

        data_row = lines[1]
        assert sample_result.evaluation_id in data_row
        assert sample_result.model_name in data_row

    def test_export_to_file(self, storage, sample_result, temp_storage_path):
        """Test exporting results to file."""
        storage.store_evaluation_result(sample_result)

        # Export to file
        output_path = Path(temp_storage_path) / "export.json"
        result_path = storage.export_results(format="json", output_path=str(output_path))

        assert result_path == str(output_path)
        assert output_path.exists()

        # Verify file content
        with open(output_path) as f:
            export_data = json.load(f)

        assert export_data["result_count"] == 1
        assert len(export_data["results"]) == 1

    def test_delete_evaluation(self, storage, sample_result):
        """Test deleting evaluation results."""
        # Store result
        storage.store_evaluation_result(sample_result)

        # Verify it exists
        retrieved = storage.get_evaluation_by_id(sample_result.evaluation_id)
        assert retrieved is not None

        # Delete it
        success = storage.delete_evaluation(sample_result.evaluation_id)
        assert success is True

        # Verify it's gone
        retrieved = storage.get_evaluation_by_id(sample_result.evaluation_id)
        assert retrieved is None

        # Try to delete again
        success = storage.delete_evaluation(sample_result.evaluation_id)
        assert success is False

    def test_storage_stats(self, storage, sample_result):
        """Test storage statistics."""
        # Initially empty
        stats = storage.get_storage_stats()
        assert stats["total_evaluations"] == 0

        # Add some data
        storage.store_evaluation_result(sample_result)

        # Check updated stats
        stats = storage.get_storage_stats()
        assert stats["total_evaluations"] == 1
        assert stats["unique_models"] == 1
        assert stats["unique_task_types"] == 1
        assert stats["unique_datasets"] == 1
        assert stats["total_metrics"] == len(sample_result.metrics)
        assert stats["total_responses"] == len(sample_result.raw_responses)
        assert stats["database_size_bytes"] > 0

    def test_data_integrity_validation(self, storage, sample_result):
        """Test data integrity validation."""
        # Store valid result
        storage.store_evaluation_result(sample_result)

        # Run integrity validation
        validation_result = storage.validate_data_integrity()

        assert validation_result["integrity_ok"] is True
        assert validation_result["statistics"]["total_checked"] == 1
        assert validation_result["statistics"]["issues_found"] == 0
        assert len(validation_result["issues"]) == 0

    def test_data_integrity_with_issues(self, storage, sample_result):
        """Test data integrity validation with problematic data."""
        storage.store_evaluation_result(sample_result)

        # Manually corrupt data to test validation
        import sqlite3

        with sqlite3.connect(storage.db_path) as conn:
            # Insert invalid success rate
            conn.execute(
                "UPDATE evaluation_results SET success_rate = 1.5 WHERE evaluation_id = ?",
                (sample_result.evaluation_id,),
            )
            conn.commit()

        # Run validation
        validation_result = storage.validate_data_integrity()

        assert validation_result["integrity_ok"] is False
        assert validation_result["statistics"]["issues_found"] > 0
        assert any("invalid success rates" in issue for issue in validation_result["issues"])

    def test_error_handling(self, storage):
        """Test error handling for invalid operations."""
        # Test storing invalid result - create without validation first
        try:
            invalid_result = EvaluationResult(
                evaluation_id="",  # Invalid empty ID
                model_name="test",
                task_type="test",
                dataset_name="test",
                metrics={},
                timestamp=datetime.now(),
                configuration={},
                raw_responses=[],
                processing_time=10.0,
            )
        except ValueError:
            # Expected - validation fails at creation
            pass
        else:
            # If somehow validation didn't trigger, test storage
            with pytest.raises(StorageError):
                storage.store_evaluation_result(invalid_result)

        # Test invalid export format
        with pytest.raises(ValueError, match="Unsupported export format"):
            storage.export_results(format="xml")

    def test_query_with_results_query_object(self, storage, sample_result):
        """Test querying with ResultsQuery object instead of dict."""
        storage.store_evaluation_result(sample_result)

        # Create ResultsQuery object
        query = ResultsQuery(model_name="bert_base", task_type="text_classification", limit=10)

        # Query with object
        results = storage.query_results(query)
        assert len(results) == 1
        assert results[0].evaluation_id == sample_result.evaluation_id


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @pytest.fixture
    def storage(self):
        """Create storage with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ResultsStorage(temp_dir)

    def test_model_comparison_scenario(self, storage):
        """Test scenario: comparing multiple models on same dataset."""
        models = ["bert_base", "roberta_base", "distilbert"]
        dataset = "imdb_sentiment"

        # Store results for each model
        for i, model in enumerate(models):
            result = EvaluationResult(
                evaluation_id=f"{model}_eval",
                model_name=model,
                task_type="sentiment_analysis",
                dataset_name=dataset,
                metrics={
                    "accuracy": 0.85 + i * 0.02,
                    "f1_score": 0.83 + i * 0.025,
                },
                timestamp=datetime.now(),
                configuration={"epochs": 3, "learning_rate": 0.0001},
                raw_responses=[],
                processing_time=30.0 + i * 10,
                experiment_name="model_comparison",
                tags=["comparison", "baseline"],
            )
            storage.store_evaluation_result(result)

        # Query results for comparison
        comparison_results = storage.query_results(
            {
                "dataset_name": dataset,
                "experiment_name": "model_comparison",
                "sort_by": "metric:accuracy",
                "sort_order": "desc",
            }
        )

        assert len(comparison_results) == 3
        assert comparison_results[0].model_name == "distilbert"  # highest accuracy
        assert comparison_results[-1].model_name == "bert_base"  # lowest accuracy

        # Export comparison results
        comparison_report = storage.export_results(
            format="json", filters={"experiment_name": "model_comparison"}
        )

        report_data = json.loads(comparison_report)
        assert report_data["result_count"] == 3

    def test_longitudinal_tracking_scenario(self, storage):
        """Test scenario: tracking model performance over time."""
        model_name = "production_model"

        # Simulate weekly evaluations over a month
        for week in range(4):
            result = EvaluationResult(
                evaluation_id=f"weekly_eval_week_{week + 1}",
                model_name=model_name,
                task_type="classification",
                dataset_name="production_data",
                metrics={
                    "accuracy": 0.90 - week * 0.01,  # Degrading performance
                    "latency_ms": 50 + week * 5,  # Increasing latency
                },
                timestamp=datetime.now() - timedelta(weeks=3 - week),
                configuration={"version": f"1.{week}"},
                raw_responses=[],
                processing_time=20.0,
                experiment_name="production_monitoring",
                tags=["production", "weekly"],
                notes=f"Week {week + 1} production evaluation",
            )
            storage.store_evaluation_result(result)

        # Get performance history
        history = storage.get_model_performance_history(model_name)

        assert len(history.evaluations) == 4
        assert history.get_latest_result().evaluation_id == "weekly_eval_week_4"

        # Analyze trends
        accuracy_trend = history.get_metric_trend("accuracy")
        assert len(accuracy_trend) == 4
        assert accuracy_trend[0][1] == 0.90  # Week 1 accuracy
        assert accuracy_trend[-1][1] == 0.87  # Week 4 accuracy (degraded)

        # Export longitudinal data
        longitudinal_data = storage.export_results(
            format="csv",
            filters={"model_name": model_name, "sort_by": "timestamp", "sort_order": "asc"},
        )

        lines = longitudinal_data.strip().split("\n")
        assert len(lines) == 5  # header + 4 data rows

    def test_experiment_management_scenario(self, storage):
        """Test scenario: managing multiple experiments."""
        experiments = [
            ("hyperparameter_tuning", ["model_v1", "model_v2", "model_v3"]),
            ("data_augmentation", ["baseline", "augmented_v1", "augmented_v2"]),
            ("architecture_comparison", ["transformer", "cnn", "lstm"]),
        ]

        # Store results for multiple experiments
        eval_id = 0
        for exp_name, models in experiments:
            for model in models:
                result = EvaluationResult(
                    evaluation_id=f"eval_{eval_id:03d}",
                    model_name=model,
                    task_type="text_classification",
                    dataset_name="benchmark_dataset",
                    metrics={"accuracy": 0.75 + (eval_id % 5) * 0.02},
                    timestamp=datetime.now() - timedelta(days=eval_id),
                    configuration={"experiment": exp_name},
                    raw_responses=[],
                    processing_time=25.0,
                    experiment_name=exp_name,
                    tags=[exp_name.split("_")[0]],
                )
                storage.store_evaluation_result(result)
                eval_id += 1

        # Query by experiment
        hp_tuning_results = storage.query_results({"experiment_name": "hyperparameter_tuning"})
        assert len(hp_tuning_results) == 3

        # Get best result per experiment
        best_results = {}
        for exp_name, _ in experiments:
            exp_results = storage.query_results(
                {
                    "experiment_name": exp_name,
                    "sort_by": "metric:accuracy",
                    "sort_order": "desc",
                    "limit": 1,
                }
            )
            if exp_results:
                best_results[exp_name] = exp_results[0]

        assert len(best_results) == 3

        # Export experiment summary
        all_results = storage.query_results()
        stats = storage.get_storage_stats()

        assert stats["total_evaluations"] == 9
        assert stats["unique_models"] == 9  # All models are unique
        assert len({r.experiment_name for r in all_results}) == 3

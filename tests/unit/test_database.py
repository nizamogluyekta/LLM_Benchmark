"""
Unit tests for database models and utilities.

Tests database schema creation, model relationships, constraints,
and basic CRUD operations.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from benchmark.core.database import (
    Dataset,
    Evaluation,
    EvaluationResult,
    Experiment,
    Model,
    Prediction,
    create_database_engine,
    create_database_tables,
    create_session_factory,
    get_database_schema_info,
)


class TestDatabaseSchema:
    """Test database schema creation and structure."""

    def test_create_database_engine(self):
        """Test database engine creation."""
        # Test in-memory SQLite database
        engine = create_database_engine("sqlite:///:memory:")
        assert engine is not None
        assert str(engine.url).startswith("sqlite://")

    def test_create_database_tables(self):
        """Test database table creation."""
        engine = create_database_engine("sqlite:///:memory:")

        # Create tables
        create_database_tables(engine)

        # Check that all tables exist
        inspector = engine.dialect.get_table_names(engine.connect())
        expected_tables = {
            "experiments",
            "datasets",
            "models",
            "evaluations",
            "evaluation_results",
            "predictions",
        }

        # Convert to set for comparison (inspector might return additional system tables)
        actual_tables = set(inspector)
        assert expected_tables.issubset(actual_tables)

    def test_session_factory_creation(self):
        """Test session factory creation."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)

        session_factory = create_session_factory(engine)
        assert session_factory is not None

        # Test creating a session
        with session_factory() as session:
            assert isinstance(session, Session)

    def test_database_schema_info(self):
        """Test database schema information utility."""
        schema_info = get_database_schema_info()

        assert "tables" in schema_info
        assert "total_tables" in schema_info
        assert schema_info["total_tables"] == 6

        # Check all expected tables are documented
        expected_tables = {
            "experiments",
            "datasets",
            "models",
            "evaluations",
            "evaluation_results",
            "predictions",
        }
        assert set(schema_info["tables"].keys()) == expected_tables

        # Check each table has required metadata
        for _table_name, table_info in schema_info["tables"].items():
            assert "description" in table_info
            assert "key_fields" in table_info
            assert "relationships" in table_info


class TestExperimentModel:
    """Test Experiment model functionality."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            yield session

    def test_experiment_creation(self, db_session):
        """Test basic experiment creation."""
        experiment = Experiment(
            name="test-experiment",
            description="Test experiment description",
            version="1.0",
            config={"test": "config"},
            output_dir="/tmp/test-results",
        )

        db_session.add(experiment)
        db_session.commit()

        # Verify experiment was saved
        assert experiment.id is not None
        assert experiment.status == "pending"  # default value
        assert experiment.created_at is not None

    def test_experiment_status_update(self, db_session):
        """Test experiment status tracking."""
        experiment = Experiment(name="status-test", config={}, output_dir="/tmp/test")

        db_session.add(experiment)
        db_session.commit()

        # Update status
        experiment.status = "running"
        experiment.started_at = datetime.now(UTC)
        db_session.commit()

        # Verify update
        retrieved = db_session.get(Experiment, experiment.id)
        assert retrieved.status == "running"
        assert retrieved.started_at is not None

    def test_experiment_config_json(self, db_session):
        """Test JSON configuration storage."""
        complex_config = {
            "datasets": [{"name": "test", "path": "/data"}],
            "models": [{"type": "api", "config": {"key": "value"}}],
            "nested": {"deep": {"value": 123}},
        }

        experiment = Experiment(name="config-test", config=complex_config, output_dir="/tmp/test")

        db_session.add(experiment)
        db_session.commit()

        # Verify JSON serialization/deserialization
        retrieved = db_session.get(Experiment, experiment.id)
        assert retrieved.config == complex_config

    def test_experiment_unique_name_per_version(self, db_session):
        """Test that experiment names should be unique (business logic test)."""
        # Create first experiment
        exp1 = Experiment(name="duplicate-name", config={}, output_dir="/tmp/test1")
        db_session.add(exp1)
        db_session.commit()

        # Create second experiment with same name (should be allowed by schema)
        exp2 = Experiment(name="duplicate-name", config={}, output_dir="/tmp/test2")
        db_session.add(exp2)
        db_session.commit()

        # Both should exist (no unique constraint on name alone)
        experiments = db_session.scalars(
            select(Experiment).where(Experiment.name == "duplicate-name")
        ).all()
        assert len(experiments) == 2


class TestDatasetModel:
    """Test Dataset model functionality."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            yield session

    def test_dataset_creation(self, db_session):
        """Test basic dataset creation."""
        dataset = Dataset(
            name="test-dataset",
            description="Test cybersecurity dataset",
            source="kaggle",
            path="/data/test.csv",
            format="csv",
            total_samples=1000,
            preprocessing=["normalize", "clean"],
        )

        db_session.add(dataset)
        db_session.commit()

        assert dataset.id is not None
        assert dataset.created_at is not None
        assert dataset.updated_at is not None

    def test_dataset_unique_name_constraint(self, db_session):
        """Test unique name constraint on datasets."""
        # Create first dataset
        dataset1 = Dataset(name="unique-dataset", source="local", path="/data/test1.csv")
        db_session.add(dataset1)
        db_session.commit()

        # Try to create second dataset with same name
        dataset2 = Dataset(name="unique-dataset", source="kaggle", path="/data/test2.csv")
        db_session.add(dataset2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_dataset_preprocessing_json(self, db_session):
        """Test preprocessing steps as JSON list."""
        preprocessing_steps = [
            "normalize_text",
            "remove_duplicates",
            "handle_missing_values",
            "feature_engineering",
        ]

        dataset = Dataset(
            name="preprocessing-test",
            source="huggingface",
            path="test/dataset",
            preprocessing=preprocessing_steps,
        )

        db_session.add(dataset)
        db_session.commit()

        retrieved = db_session.get(Dataset, dataset.id)
        assert retrieved.preprocessing == preprocessing_steps

    def test_dataset_metadata_storage(self, db_session):
        """Test metadata JSON field."""
        metadata = {
            "source_url": "https://example.com/dataset",
            "license": "MIT",
            "tags": ["cybersecurity", "malware", "detection"],
            "statistics": {
                "class_distribution": {"benign": 0.7, "malware": 0.3},
                "feature_types": {"numerical": 45, "categorical": 5},
            },
        }

        dataset = Dataset(
            name="metadata-test",
            source="remote",
            path="https://example.com/data.csv",
            dataset_metadata=metadata,
        )

        db_session.add(dataset)
        db_session.commit()

        retrieved = db_session.get(Dataset, dataset.id)
        assert retrieved.dataset_metadata == metadata


class TestModelModel:
    """Test Model model functionality."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            yield session

    def test_model_creation(self, db_session):
        """Test basic model creation."""
        model = Model(
            name="gpt-4-cybersec",
            description="GPT-4 for cybersecurity analysis",
            type="openai_api",
            path="gpt-4",
            version="1.0",
            max_tokens=1024,
            temperature=0.1,
            config={"api_key": "test-key"},
            system_prompt="You are a cybersecurity expert.",
        )

        db_session.add(model)
        db_session.commit()

        assert model.id is not None
        assert model.is_available is True  # default value
        assert model.created_at is not None

    def test_model_unique_name_constraint(self, db_session):
        """Test unique name constraint on models."""
        # Create first model
        model1 = Model(name="unique-model", type="openai_api", path="gpt-3.5-turbo")
        db_session.add(model1)
        db_session.commit()

        # Try to create second model with same name
        model2 = Model(name="unique-model", type="anthropic_api", path="claude-3-sonnet")
        db_session.add(model2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_model_config_json(self, db_session):
        """Test model configuration as JSON."""
        model_config = {
            "api_key": "test-key",
            "api_base": "https://api.openai.com/v1",
            "rate_limit": {"requests_per_minute": 60, "tokens_per_minute": 30000},
            "retry_config": {"max_retries": 3, "backoff_factor": 2.0},
        }

        model = Model(
            name="config-test", type="openai_api", path="gpt-3.5-turbo", config=model_config
        )

        db_session.add(model)
        db_session.commit()

        retrieved = db_session.get(Model, model.id)
        assert retrieved.config == model_config

    def test_model_performance_metrics(self, db_session):
        """Test model performance tracking fields."""
        model = Model(
            name="performance-test",
            type="mlx_local",
            path="/models/llama-7b",
            parameter_count=7000000000,
            model_size_gb=13.5,
            context_length=2048,
            avg_response_time_ms=150.5,
            tokens_per_second=45.2,
            memory_usage_gb=8.1,
        )

        db_session.add(model)
        db_session.commit()

        retrieved = db_session.get(Model, model.id)
        assert retrieved.parameter_count == 7000000000
        assert retrieved.avg_response_time_ms == 150.5
        assert retrieved.tokens_per_second == 45.2


class TestEvaluationModel:
    """Test Evaluation model functionality."""

    @pytest.fixture
    def db_session_with_data(self):
        """Create a test database session with experiment, dataset, and model."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            # Create test experiment
            experiment = Experiment(name="test-exp", config={}, output_dir="/tmp/test")
            session.add(experiment)

            # Create test dataset
            dataset = Dataset(name="test-dataset", source="local", path="/data/test.csv")
            session.add(dataset)

            # Create test model
            model = Model(name="test-model", type="openai_api", path="gpt-3.5-turbo")
            session.add(model)

            session.commit()

            yield session, experiment, dataset, model

    def test_evaluation_creation(self, db_session_with_data):
        """Test basic evaluation creation."""
        session, experiment, dataset, model = db_session_with_data

        evaluation = Evaluation(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_id=model.id,
            batch_size=16,
            max_samples=1000,
            random_seed=42,
            metrics=["accuracy", "f1_score", "precision", "recall"],
        )

        session.add(evaluation)
        session.commit()

        assert evaluation.id is not None
        assert evaluation.status == "pending"  # default
        assert evaluation.error_count == 0  # default

    def test_evaluation_unique_constraint(self, db_session_with_data):
        """Test unique constraint on experiment-dataset-model combination."""
        session, experiment, dataset, model = db_session_with_data

        # Create first evaluation
        eval1 = Evaluation(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_id=model.id,
            metrics=["accuracy"],
        )
        session.add(eval1)
        session.commit()

        # Try to create duplicate evaluation
        eval2 = Evaluation(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_id=model.id,
            metrics=["precision"],
        )
        session.add(eval2)

        with pytest.raises(IntegrityError):
            session.commit()

    def test_evaluation_relationships(self, db_session_with_data):
        """Test evaluation relationships with other models."""
        session, experiment, dataset, model = db_session_with_data

        evaluation = Evaluation(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_id=model.id,
            metrics=["accuracy"],
        )
        session.add(evaluation)
        session.commit()

        # Test relationships
        assert evaluation.experiment.name == "test-exp"
        assert evaluation.dataset.name == "test-dataset"
        assert evaluation.model.name == "test-model"

    def test_evaluation_metrics_json(self, db_session_with_data):
        """Test metrics as JSON list."""
        session, experiment, dataset, model = db_session_with_data

        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        evaluation = Evaluation(
            experiment_id=experiment.id, dataset_id=dataset.id, model_id=model.id, metrics=metrics
        )
        session.add(evaluation)
        session.commit()

        retrieved = session.get(Evaluation, evaluation.id)
        assert retrieved.metrics == metrics


class TestEvaluationResultModel:
    """Test EvaluationResult model functionality."""

    @pytest.fixture
    def db_session_with_evaluation(self):
        """Create a test database session with a complete evaluation setup."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            # Create test data
            experiment = Experiment(name="test-exp", config={}, output_dir="/tmp/test")
            dataset = Dataset(name="test-dataset", source="local", path="/data/test.csv")
            model = Model(name="test-model", type="openai_api", path="gpt-3.5-turbo")

            session.add_all([experiment, dataset, model])
            session.commit()

            evaluation = Evaluation(
                experiment_id=experiment.id,
                dataset_id=dataset.id,
                model_id=model.id,
                metrics=["accuracy", "f1_score"],
            )
            session.add(evaluation)
            session.commit()

            yield session, evaluation

    def test_evaluation_result_creation(self, db_session_with_evaluation):
        """Test basic evaluation result creation."""
        session, evaluation = db_session_with_evaluation

        result = EvaluationResult(
            evaluation_id=evaluation.id,
            metric_name="accuracy",
            metric_value=0.85,
            metric_data={"samples_correct": 850, "total_samples": 1000},
            confidence_interval={"lower": 0.82, "upper": 0.88},
            standard_error=0.015,
            sample_size=1000,
        )

        session.add(result)
        session.commit()

        assert result.id is not None
        assert result.created_at is not None

    def test_evaluation_result_unique_constraint(self, db_session_with_evaluation):
        """Test unique constraint on evaluation-metric combination."""
        session, evaluation = db_session_with_evaluation

        # Create first result
        result1 = EvaluationResult(
            evaluation_id=evaluation.id, metric_name="accuracy", metric_value=0.85
        )
        session.add(result1)
        session.commit()

        # Try to create duplicate result for same metric
        result2 = EvaluationResult(
            evaluation_id=evaluation.id, metric_name="accuracy", metric_value=0.90
        )
        session.add(result2)

        with pytest.raises(IntegrityError):
            session.commit()

    def test_evaluation_result_metric_data_json(self, db_session_with_evaluation):
        """Test metric data as JSON."""
        session, evaluation = db_session_with_evaluation

        metric_data = {
            "confusion_matrix": [[950, 50], [100, 900]],
            "classification_report": {
                "class_0": {"precision": 0.90, "recall": 0.95, "f1-score": 0.92},
                "class_1": {"precision": 0.95, "recall": 0.90, "f1-score": 0.92},
            },
            "additional_metrics": {"true_positive_rate": 0.90, "false_positive_rate": 0.05},
        }

        result = EvaluationResult(
            evaluation_id=evaluation.id, metric_name="confusion_matrix", metric_data=metric_data
        )

        session.add(result)
        session.commit()

        retrieved = session.get(EvaluationResult, result.id)
        assert retrieved.metric_data == metric_data

    def test_evaluation_result_baseline_comparison(self, db_session_with_evaluation):
        """Test baseline comparison functionality."""
        session, evaluation = db_session_with_evaluation

        # Create baseline result
        baseline = EvaluationResult(
            evaluation_id=evaluation.id, metric_name="baseline_accuracy", metric_value=0.75
        )
        session.add(baseline)
        session.commit()

        # Create comparison result
        comparison = EvaluationResult(
            evaluation_id=evaluation.id,
            metric_name="model_accuracy",
            metric_value=0.85,
            baseline_metric_id=baseline.id,
            p_value=0.001,
            is_statistically_significant=True,
        )
        session.add(comparison)
        session.commit()

        # Test relationship
        assert comparison.baseline_metric.metric_value == 0.75
        assert comparison.is_statistically_significant is True


class TestPredictionModel:
    """Test Prediction model functionality."""

    @pytest.fixture
    def db_session_with_evaluation(self):
        """Create a test database session with a complete evaluation setup."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            # Create test data
            experiment = Experiment(name="test-exp", config={}, output_dir="/tmp/test")
            dataset = Dataset(name="test-dataset", source="local", path="/data/test.csv")
            model = Model(name="test-model", type="openai_api", path="gpt-3.5-turbo")

            session.add_all([experiment, dataset, model])
            session.commit()

            evaluation = Evaluation(
                experiment_id=experiment.id,
                dataset_id=dataset.id,
                model_id=model.id,
                metrics=["accuracy"],
            )
            session.add(evaluation)
            session.commit()

            yield session, evaluation

    def test_prediction_creation(self, db_session_with_evaluation):
        """Test basic prediction creation."""
        session, evaluation = db_session_with_evaluation

        prediction = Prediction(
            evaluation_id=evaluation.id,
            sample_id="sample_001",
            sample_index=0,
            input_text="Suspicious network traffic detected from 192.168.1.100",
            true_label="malicious",
            predicted_label="malicious",
            confidence_score=0.92,
            is_correct=True,
            response_time_ms=150.5,
        )

        session.add(prediction)
        session.commit()

        assert prediction.id is not None
        assert prediction.created_at is not None
        assert prediction.retry_count == 0  # default

    def test_prediction_unique_constraint(self, db_session_with_evaluation):
        """Test unique constraint on evaluation-sample combination."""
        session, evaluation = db_session_with_evaluation

        # Create first prediction
        pred1 = Prediction(
            evaluation_id=evaluation.id,
            sample_id="sample_001",
            sample_index=0,
            predicted_label="benign",
        )
        session.add(pred1)
        session.commit()

        # Try to create duplicate prediction for same sample
        pred2 = Prediction(
            evaluation_id=evaluation.id,
            sample_id="sample_001",
            sample_index=1,  # different index but same sample_id
            predicted_label="malicious",
        )
        session.add(pred2)

        with pytest.raises(IntegrityError):
            session.commit()

    def test_prediction_class_probabilities(self, db_session_with_evaluation):
        """Test class probabilities as JSON."""
        session, evaluation = db_session_with_evaluation

        probabilities = {"benign": 0.15, "malware": 0.75, "phishing": 0.08, "spam": 0.02}

        prediction = Prediction(
            evaluation_id=evaluation.id,
            sample_id="sample_multiclass",
            sample_index=0,
            predicted_label="malware",
            class_probabilities=probabilities,
            confidence_score=0.75,
        )

        session.add(prediction)
        session.commit()

        retrieved = session.get(Prediction, prediction.id)
        assert retrieved.class_probabilities == probabilities

    def test_prediction_input_data_json(self, db_session_with_evaluation):
        """Test input data storage as JSON."""
        session, evaluation = db_session_with_evaluation

        input_data = {
            "features": {
                "packet_size": 1500,
                "source_ip": "192.168.1.100",
                "destination_port": 80,
                "protocol": "TCP",
                "flags": ["SYN", "ACK"],
            },
            "metadata": {"timestamp": "2024-01-15T10:30:00Z", "session_id": "sess_12345"},
        }

        prediction = Prediction(
            evaluation_id=evaluation.id,
            sample_id="sample_detailed",
            sample_index=0,
            input_data=input_data,
            predicted_label="benign",
        )

        session.add(prediction)
        session.commit()

        retrieved = session.get(Prediction, prediction.id)
        assert retrieved.input_data == input_data

    def test_prediction_error_handling(self, db_session_with_evaluation):
        """Test prediction error tracking."""
        session, evaluation = db_session_with_evaluation

        prediction = Prediction(
            evaluation_id=evaluation.id,
            sample_id="failed_sample",
            sample_index=0,
            error_message="API timeout after 30 seconds",
            retry_count=3,
            error_type="timeout",
            is_correct=None,  # No prediction made due to error
        )

        session.add(prediction)
        session.commit()

        retrieved = session.get(Prediction, prediction.id)
        assert retrieved.error_message == "API timeout after 30 seconds"
        assert retrieved.retry_count == 3
        assert retrieved.error_type == "timeout"
        assert retrieved.is_correct is None


class TestDatabaseIntegration:
    """Test database integration and complex operations."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        engine = create_database_engine("sqlite:///:memory:")
        create_database_tables(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            yield session

    def test_complete_experiment_workflow(self, db_session):
        """Test a complete experiment workflow with all models."""
        # Create experiment
        experiment = Experiment(
            name="integration-test",
            description="Complete workflow test",
            config={"test": True},
            output_dir="/tmp/integration",
        )
        db_session.add(experiment)

        # Create dataset
        dataset = Dataset(
            name="integration-dataset",
            source="synthetic",
            path="/data/synthetic.csv",
            total_samples=1000,
            preprocessing=["normalize", "balance"],
        )
        db_session.add(dataset)

        # Create model
        model = Model(
            name="integration-model",
            type="openai_api",
            path="gpt-3.5-turbo",
            config={"api_key": "test-key"},
        )
        db_session.add(model)

        db_session.commit()

        # Create evaluation
        evaluation = Evaluation(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_id=model.id,
            metrics=["accuracy", "f1_score"],
            status="completed",
        )
        db_session.add(evaluation)
        db_session.commit()

        # Add evaluation results
        accuracy_result = EvaluationResult(
            evaluation_id=evaluation.id, metric_name="accuracy", metric_value=0.87, sample_size=1000
        )

        f1_result = EvaluationResult(
            evaluation_id=evaluation.id, metric_name="f1_score", metric_value=0.85, sample_size=1000
        )

        db_session.add_all([accuracy_result, f1_result])

        # Add some predictions
        predictions = [
            Prediction(
                evaluation_id=evaluation.id,
                sample_id=f"sample_{i:03d}",
                sample_index=i,
                true_label="benign" if i % 2 == 0 else "malicious",
                predicted_label="benign" if i % 2 == 0 else "malicious",
                is_correct=True,
                confidence_score=0.9,
            )
            for i in range(10)
        ]

        db_session.add_all(predictions)
        db_session.commit()

        # Verify the complete workflow
        retrieved_experiment = db_session.get(Experiment, experiment.id)
        assert retrieved_experiment is not None
        assert len(retrieved_experiment.evaluations) == 1

        retrieved_evaluation = retrieved_experiment.evaluations[0]
        assert len(retrieved_evaluation.evaluation_results) == 2
        assert len(retrieved_evaluation.predictions) == 10

        # Test cascading relationships
        accuracy_results = [
            r for r in retrieved_evaluation.evaluation_results if r.metric_name == "accuracy"
        ]
        assert len(accuracy_results) == 1
        assert accuracy_results[0].metric_value == 0.87

    def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraint enforcement."""
        # Enable foreign key constraints for SQLite
        if "sqlite" in str(db_session.bind.url):
            db_session.execute(text("PRAGMA foreign_keys=ON"))

        # Try to create evaluation with non-existent experiment
        evaluation = Evaluation(
            experiment_id=999,  # Non-existent
            dataset_id=999,  # Non-existent
            model_id=999,  # Non-existent
            metrics=["accuracy"],
        )

        db_session.add(evaluation)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_cascade_deletion(self, db_session):
        """Test cascade deletion behavior."""
        # Create experiment with evaluation
        experiment = Experiment(name="cascade-test", config={}, output_dir="/tmp/test")
        dataset = Dataset(name="cascade-dataset", source="local", path="/data/test.csv")
        model = Model(name="cascade-model", type="openai_api", path="gpt-3.5-turbo")

        db_session.add_all([experiment, dataset, model])
        db_session.commit()

        evaluation = Evaluation(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_id=model.id,
            metrics=["accuracy"],
        )
        db_session.add(evaluation)
        db_session.commit()

        # Add evaluation result and prediction
        result = EvaluationResult(
            evaluation_id=evaluation.id, metric_name="accuracy", metric_value=0.8
        )

        prediction = Prediction(
            evaluation_id=evaluation.id,
            sample_id="test_sample",
            sample_index=0,
            predicted_label="benign",
        )

        db_session.add_all([result, prediction])
        db_session.commit()

        # Delete experiment (should cascade to evaluations, results, and predictions)
        db_session.delete(experiment)
        db_session.commit()

        # Verify cascaded deletion
        remaining_evaluations = db_session.scalars(select(Evaluation)).all()
        remaining_results = db_session.scalars(select(EvaluationResult)).all()
        remaining_predictions = db_session.scalars(select(Prediction)).all()

        assert len(remaining_evaluations) == 0
        assert len(remaining_results) == 0
        assert len(remaining_predictions) == 0

        # Dataset and model should still exist (RESTRICT constraint)
        remaining_datasets = db_session.scalars(select(Dataset)).all()
        remaining_models = db_session.scalars(select(Model)).all()

        assert len(remaining_datasets) == 1
        assert len(remaining_models) == 1

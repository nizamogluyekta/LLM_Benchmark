"""
SQLAlchemy database models for the LLM Cybersecurity Benchmark system.

This module defines the complete database schema for storing experiment data,
including experiments, datasets, models, evaluations, and results.
"""

from datetime import UTC, datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

# Type: ignore for SQLAlchemy DeclarativeBase to avoid MyPy issues in some environments


class Base(DeclarativeBase):  # type: ignore
    """Base class for all database models."""

    pass


class Experiment(Base):
    """
    Experiments table - stores high-level experiment metadata.

    Each experiment can contain multiple model-dataset combinations
    and represents a complete benchmark run.
    """

    __tablename__ = "experiments"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Basic information
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0")

    # Configuration and metadata
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    output_dir: Mapped[str] = mapped_column(String(512), nullable=False)

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )  # pending, running, completed, failed, cancelled

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Execution metadata
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Resource usage tracking
    max_memory_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_api_calls: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    total_api_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)

    # Relationships
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation", back_populates="experiment", cascade="all, delete-orphan"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_experiment_status_created", "status", "created_at"),
        Index("idx_experiment_name_version", "name", "version"),
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name='{self.name}', status='{self.status}')>"


class Dataset(Base):
    """
    Datasets table - stores dataset metadata and configuration.

    Reusable dataset configurations that can be referenced by multiple experiments.
    """

    __tablename__ = "datasets"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Dataset identification
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Source information
    source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    format: Mapped[str] = mapped_column(String(50), nullable=False, default="csv")

    # Dataset characteristics
    total_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feature_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Split configuration
    test_split: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    validation_split: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)
    stratify_column: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Data preprocessing
    preprocessing: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Metadata and quality metrics
    dataset_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    data_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    evaluations: Mapped[list["Evaluation"]] = relationship("Evaluation", back_populates="dataset")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_dataset_source_format", "source", "format"),
        Index("idx_dataset_created_updated", "created_at", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}', source='{self.source}')>"


class Model(Base):
    """
    Models table - stores model metadata and configuration.

    Reusable model configurations that can be referenced by multiple experiments.
    """

    __tablename__ = "models"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Model identification
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Model configuration
    type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False, default="1.0")

    # Generation parameters
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=512)
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)
    top_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Model-specific configuration
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Model characteristics
    parameter_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_size_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Performance characteristics
    avg_response_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_usage_gb: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cost information (for API models)
    cost_per_input_token: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_output_token: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Status and availability
    is_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    last_health_check: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    evaluations: Mapped[list["Evaluation"]] = relationship("Evaluation", back_populates="model")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_model_type_available", "type", "is_available"),
        Index("idx_model_performance", "avg_response_time_ms", "tokens_per_second"),
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name='{self.name}', type='{self.type}')>"


class Evaluation(Base):
    """
    Evaluations table - stores individual model-dataset evaluation runs.

    Each evaluation represents one model being evaluated on one dataset
    within an experiment.
    """

    __tablename__ = "evaluations"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key relationships
    experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="RESTRICT"), nullable=False, index=True
    )

    # Evaluation configuration
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=32)
    max_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)

    # Evaluation metrics configuration
    metrics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )  # pending, running, completed, failed, cancelled

    # Execution timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Sample processing statistics
    total_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    successful_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failed_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Performance metrics
    avg_response_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error information
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Metadata
    eval_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    # Relationships
    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="evaluations")
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="evaluations")
    model: Mapped[Model] = relationship("Model", back_populates="evaluations")

    evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
        "EvaluationResult", back_populates="evaluation", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="evaluation", cascade="all, delete-orphan"
    )

    # Constraints and indexes
    __table_args__ = (
        # Ensure unique evaluations per experiment-dataset-model combination
        UniqueConstraint(
            "experiment_id", "dataset_id", "model_id", name="uq_evaluation_combination"
        ),
        Index("idx_evaluation_status_created", "status", "created_at"),
        Index("idx_evaluation_experiment_status", "experiment_id", "status"),
        Index("idx_evaluation_performance", "avg_response_time_ms", "total_cost_usd"),
    )

    def __repr__(self) -> str:
        return f"<Evaluation(id={self.id}, experiment_id={self.experiment_id}, status='{self.status}')>"


class EvaluationResult(Base):
    """
    Evaluation results table - stores computed metrics for each evaluation.

    Each row represents one metric computed for an evaluation.
    """

    __tablename__ = "evaluation_results"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key relationship
    evaluation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Metric information
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Metric metadata
    confidence_interval: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    standard_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Statistical significance (for comparisons)
    baseline_metric_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("evaluation_results.id", ondelete="SET NULL"), nullable=True
    )
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_statistically_significant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Computation metadata
    computation_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    computation_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    # Relationships
    evaluation: Mapped[Evaluation] = relationship("Evaluation", back_populates="evaluation_results")
    baseline_metric: Mapped[Optional["EvaluationResult"]] = relationship(
        "EvaluationResult", remote_side=[id], post_update=True
    )

    # Constraints and indexes
    __table_args__ = (
        # Ensure unique metric per evaluation
        UniqueConstraint("evaluation_id", "metric_name", name="uq_evaluation_metric"),
        Index("idx_result_metric_value", "metric_name", "metric_value"),
        Index("idx_result_evaluation_metric", "evaluation_id", "metric_name"),
    )

    def __repr__(self) -> str:
        return f"<EvaluationResult(id={self.id}, metric='{self.metric_name}', value={self.metric_value})>"


class Prediction(Base):
    """
    Predictions table - stores individual model predictions.

    Each row represents one model prediction for one input sample.
    Useful for detailed analysis and error investigation.
    """

    __tablename__ = "predictions"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key relationship
    evaluation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Sample identification
    sample_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Input data (optionally stored for analysis)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Ground truth
    true_label: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    true_class_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Model prediction
    predicted_label: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    predicted_class_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Prediction probabilities (for multi-class)
    class_probabilities: Mapped[dict[str, float]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Raw model output
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_response_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Prediction correctness
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True, index=True)
    error_type: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Timing and performance
    response_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error information (if prediction failed)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    # Relationships
    evaluation: Mapped[Evaluation] = relationship("Evaluation", back_populates="predictions")

    # Constraints and indexes
    __table_args__ = (
        # Ensure unique prediction per evaluation-sample combination
        UniqueConstraint("evaluation_id", "sample_id", name="uq_prediction_sample"),
        Index("idx_prediction_correctness", "evaluation_id", "is_correct"),
        Index("idx_prediction_confidence", "evaluation_id", "confidence_score"),
        Index("idx_prediction_performance", "response_time_ms", "cost_usd"),
        Index("idx_prediction_error_type", "error_type", "is_correct"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, sample_id='{self.sample_id}', correct={self.is_correct})>"
        )


# Database utility functions
def create_database_engine(
    database_url: str = "sqlite:///benchmark.db", echo: bool = False
) -> Engine:
    """
    Create a SQLAlchemy engine for the benchmark database.

    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL statements (for debugging)

    Returns:
        SQLAlchemy Engine instance
    """
    engine = create_engine(database_url, echo=echo)
    return engine


def create_database_tables(engine: Engine) -> None:
    """
    Create all database tables.

    Args:
        engine: SQLAlchemy Engine instance
    """
    Base.metadata.create_all(engine)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """
    Create a sessionmaker for database sessions.

    Args:
        engine: SQLAlchemy Engine instance

    Returns:
        SQLAlchemy sessionmaker
    """
    return sessionmaker(bind=engine)


def get_database_schema_info() -> dict[str, Any]:
    """
    Get information about the database schema.

    Returns:
        Dictionary containing schema information
    """
    return {
        "tables": {
            "experiments": {
                "description": "High-level experiment metadata and status",
                "key_fields": ["name", "status", "created_at"],
                "relationships": ["evaluations"],
            },
            "datasets": {
                "description": "Dataset metadata and configuration",
                "key_fields": ["name", "source", "path"],
                "relationships": ["evaluations"],
            },
            "models": {
                "description": "Model metadata and configuration",
                "key_fields": ["name", "type", "is_available"],
                "relationships": ["evaluations"],
            },
            "evaluations": {
                "description": "Individual model-dataset evaluation runs",
                "key_fields": ["experiment_id", "dataset_id", "model_id", "status"],
                "relationships": [
                    "experiment",
                    "dataset",
                    "model",
                    "evaluation_results",
                    "predictions",
                ],
            },
            "evaluation_results": {
                "description": "Computed metrics for evaluations",
                "key_fields": ["evaluation_id", "metric_name", "metric_value"],
                "relationships": ["evaluation", "baseline_metric"],
            },
            "predictions": {
                "description": "Individual model predictions",
                "key_fields": ["evaluation_id", "sample_id", "is_correct"],
                "relationships": ["evaluation"],
            },
        },
        "total_tables": 6,
        "sqlalchemy_version": "2.0+",
    }

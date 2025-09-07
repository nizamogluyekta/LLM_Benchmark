"""
Data models for datasets, samples, and metadata.

This module defines comprehensive Pydantic models for representing datasets,
samples, and metadata throughout the LLM Cybersecurity Benchmark system.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class DatasetSample(BaseModel):
    """A single data sample with cybersecurity context."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique sample identifier")
    input_text: str = Field(..., min_length=1, description="The input text content")
    label: str = Field(..., description="Classification label: 'ATTACK' or 'BENIGN'")
    attack_type: str | None = Field(None, description="Type of attack if applicable")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime | None = Field(None, description="Sample timestamp")
    confidence_score: float | None = Field(None, ge=0.0, le=1.0, description="Label confidence")
    source_file: str | None = Field(None, description="Source file path")
    line_number: int | None = Field(None, ge=1, description="Line number in source file")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate that label is either ATTACK or BENIGN."""
        if v.upper() not in ["ATTACK", "BENIGN"]:
            raise ValueError("Label must be either 'ATTACK' or 'BENIGN'")
        return v.upper()

    @field_validator("attack_type")
    @classmethod
    def validate_attack_type(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate attack_type is provided for ATTACK labels."""
        if hasattr(info, "data") and info.data.get("label", "").upper() == "ATTACK" and not v:
            # Allow None for ATTACK samples as attack_type might be unknown
            pass
        return v.lower() if v else v

    @model_validator(mode="after")
    def validate_sample_consistency(self) -> "DatasetSample":
        """Validate consistency between label and attack_type."""
        if self.label == "BENIGN" and self.attack_type:
            raise ValueError("BENIGN samples should not have an attack_type")
        return self


class DatasetInfo(BaseModel):
    """Information and metadata about a dataset."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique dataset identifier")
    name: str = Field(..., min_length=1, description="Dataset name")
    source: str = Field(..., description="Data source (local, kaggle, huggingface, etc.)")
    total_samples: int = Field(..., ge=0, description="Total number of samples")
    attack_samples: int = Field(..., ge=0, description="Number of attack samples")
    benign_samples: int = Field(..., ge=0, description="Number of benign samples")
    attack_types: list[str] = Field(default_factory=list, description="Types of attacks present")
    schema_version: str = Field(default="1.0", description="Schema version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    description: str | None = Field(None, description="Dataset description")
    size_bytes: int | None = Field(None, ge=0, description="Dataset size in bytes")
    format: str | None = Field(None, description="Data format (json, csv, etc.)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="after")
    def validate_sample_counts(self) -> "DatasetInfo":
        """Validate that sample counts are consistent."""
        if self.attack_samples + self.benign_samples != self.total_samples:
            raise ValueError(
                f"Attack samples ({self.attack_samples}) + Benign samples ({self.benign_samples}) "
                f"must equal total samples ({self.total_samples})"
            )
        return self

    @field_validator("attack_types")
    @classmethod
    def validate_attack_types(cls, v: list[str]) -> list[str]:
        """Normalize attack types to lowercase."""
        return [attack_type.lower() for attack_type in v]


class DataSplits(BaseModel):
    """Data splits for training, testing, and validation."""

    train: list[DatasetSample] = Field(default_factory=list, description="Training samples")
    test: list[DatasetSample] = Field(default_factory=list, description="Testing samples")
    validation: list[DatasetSample] | None = Field(None, description="Validation samples")
    split_strategy: str = Field(default="random", description="Splitting strategy used")
    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "test": 0.2, "validation": 0.1},
        description="Split ratios",
    )
    split_metadata: dict[str, Any] = Field(default_factory=dict, description="Split metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Split creation time")

    @property
    def train_count(self) -> int:
        """Number of training samples."""
        return len(self.train)

    @property
    def test_count(self) -> int:
        """Number of test samples."""
        return len(self.test)

    @property
    def validation_count(self) -> int:
        """Number of validation samples."""
        return len(self.validation) if self.validation else 0

    @property
    def total_count(self) -> int:
        """Total number of samples across all splits."""
        return self.train_count + self.test_count + self.validation_count

    @model_validator(mode="after")
    def validate_splits(self) -> "DataSplits":
        """Validate that splits are reasonable."""
        if not self.train and not self.test:
            raise ValueError("At least one of train or test splits must be non-empty")

        # Validate split ratios sum to approximately 1.0
        total_ratio = sum(self.split_ratios.values())
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        return self


class Dataset(BaseModel):
    """Complete dataset with information, samples, and optional splits."""

    info: DatasetInfo = Field(..., description="Dataset information and metadata")
    samples: list[DatasetSample] = Field(default_factory=list, description="Dataset samples")
    splits: DataSplits | None = Field(None, description="Data splits if available")
    version: str = Field(default="1.0", description="Dataset version")
    checksum: str | None = Field(None, description="Dataset checksum for integrity")

    @property
    def sample_count(self) -> int:
        """Number of samples in the dataset."""
        return len(self.samples)

    @property
    def attack_samples(self) -> list[DatasetSample]:
        """Get all attack samples."""
        return [s for s in self.samples if s.label == "ATTACK"]

    @property
    def benign_samples(self) -> list[DatasetSample]:
        """Get all benign samples."""
        return [s for s in self.samples if s.label == "BENIGN"]

    @property
    def attack_types_present(self) -> list[str]:
        """Get unique attack types present in the dataset."""
        attack_types = set()
        for sample in self.attack_samples:
            if sample.attack_type:
                attack_types.add(sample.attack_type)
        return sorted(attack_types)

    @model_validator(mode="after")
    def validate_dataset_consistency(self) -> "Dataset":
        """Validate consistency between dataset info and samples."""
        actual_total = len(self.samples)
        actual_attacks = len(self.attack_samples)
        actual_benign = len(self.benign_samples)

        # Update info counts if they don't match (allow auto-correction)
        if (
            self.info.total_samples != actual_total
            or self.info.attack_samples != actual_attacks
            or self.info.benign_samples != actual_benign
        ):
            self.info.total_samples = actual_total
            self.info.attack_samples = actual_attacks
            self.info.benign_samples = actual_benign

        # Update attack types in info if they don't match
        actual_attack_types = self.attack_types_present
        if set(self.info.attack_types) != set(actual_attack_types):
            self.info.attack_types = actual_attack_types

        return self

    def get_samples_by_type(self, attack_type: str) -> list[DatasetSample]:
        """Get all samples of a specific attack type."""
        return [s for s in self.samples if s.attack_type == attack_type.lower()]

    def get_sample_by_id(self, sample_id: str) -> DatasetSample | None:
        """Get a sample by its ID."""
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None


class DataBatch(BaseModel):
    """A batch of dataset samples for processing."""

    samples: list[DatasetSample] = Field(..., description="Samples in the batch")
    batch_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique batch identifier"
    )
    dataset_id: str = Field(..., description="Source dataset identifier")
    offset: int = Field(..., ge=0, description="Starting offset in the dataset")
    total_batches: int = Field(..., ge=1, description="Total number of batches")
    batch_size: int = Field(..., ge=1, description="Target batch size")
    created_at: datetime = Field(default_factory=datetime.now, description="Batch creation time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Batch metadata")

    @property
    def actual_size(self) -> int:
        """Actual number of samples in the batch."""
        return len(self.samples)

    @property
    def attack_count(self) -> int:
        """Number of attack samples in the batch."""
        return len([s for s in self.samples if s.label == "ATTACK"])

    @property
    def benign_count(self) -> int:
        """Number of benign samples in the batch."""
        return len([s for s in self.samples if s.label == "BENIGN"])

    @property
    def attack_ratio(self) -> float:
        """Ratio of attack samples in the batch."""
        return self.attack_count / self.actual_size if self.actual_size > 0 else 0.0

    @model_validator(mode="after")
    def validate_batch(self) -> "DataBatch":
        """Validate batch consistency."""
        if self.actual_size > self.batch_size:
            raise ValueError(
                f"Actual batch size ({self.actual_size}) cannot exceed "
                f"target batch size ({self.batch_size})"
            )

        if self.offset < 0:
            raise ValueError("Batch offset cannot be negative")

        return self

    def get_samples_by_label(self, label: str) -> list[DatasetSample]:
        """Get all samples with a specific label."""
        return [s for s in self.samples if s.label.upper() == label.upper()]


# Type aliases for convenience
SampleBatch = DataBatch
DatasetMetadata = dict[str, Any]
SampleMetadata = dict[str, Any]

# Constants for validation
VALID_LABELS = ["ATTACK", "BENIGN"]
COMMON_ATTACK_TYPES = [
    "malware",
    "intrusion",
    "ddos",
    "phishing",
    "sql_injection",
    "xss",
    "csrf",
    "privilege_escalation",
    "data_breach",
    "ransomware",
]

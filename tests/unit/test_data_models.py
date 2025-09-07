"""
Unit tests for data models.

This module tests the data model functionality including validation,
serialization, relationships, and field constraints.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from benchmark.data.models import (
    COMMON_ATTACK_TYPES,
    VALID_LABELS,
    DataBatch,
    Dataset,
    DatasetInfo,
    DatasetSample,
    DataSplits,
)


class TestDatasetSample:
    """Test the DatasetSample model."""

    def test_sample_creation_minimal(self):
        """Test creating a sample with minimal required fields."""
        sample = DatasetSample(input_text="Suspicious network activity detected", label="ATTACK")

        assert sample.input_text == "Suspicious network activity detected"
        assert sample.label == "ATTACK"
        assert len(sample.id) > 0  # UUID should be generated
        assert sample.attack_type is None
        assert sample.metadata == {}
        assert sample.timestamp is None

    def test_sample_creation_full(self):
        """Test creating a sample with all fields."""
        timestamp = datetime.now()
        metadata = {"source": "test", "severity": "high"}

        sample = DatasetSample(
            id="test-sample-001",
            input_text="Malware detected in file.exe",
            label="attack",  # Test case normalization
            attack_type="MALWARE",  # Test case normalization
            metadata=metadata,
            timestamp=timestamp,
            confidence_score=0.95,
            source_file="/data/samples.jsonl",
            line_number=42,
        )

        assert sample.id == "test-sample-001"
        assert sample.input_text == "Malware detected in file.exe"
        assert sample.label == "ATTACK"  # Should be normalized to uppercase
        assert sample.attack_type == "malware"  # Should be normalized to lowercase
        assert sample.metadata == metadata
        assert sample.timestamp == timestamp
        assert sample.confidence_score == 0.95
        assert sample.source_file == "/data/samples.jsonl"
        assert sample.line_number == 42

    def test_sample_label_validation(self):
        """Test label validation."""
        # Valid labels should work
        for label in ["ATTACK", "attack", "BENIGN", "benign"]:
            sample = DatasetSample(input_text="test", label=label)
            assert sample.label in ["ATTACK", "BENIGN"]

        # Invalid labels should raise ValidationError
        with pytest.raises(ValidationError, match="Label must be either"):
            DatasetSample(input_text="test", label="INVALID")

    def test_sample_benign_attack_type_validation(self):
        """Test that BENIGN samples cannot have attack_type."""
        with pytest.raises(ValidationError, match="BENIGN samples should not have an attack_type"):
            DatasetSample(input_text="Normal traffic", label="BENIGN", attack_type="malware")

    def test_sample_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid scores
        for score in [0.0, 0.5, 1.0]:
            sample = DatasetSample(input_text="test", label="ATTACK", confidence_score=score)
            assert sample.confidence_score == score

        # Invalid scores
        for score in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValidationError):
                DatasetSample(input_text="test", label="ATTACK", confidence_score=score)

    def test_sample_line_number_validation(self):
        """Test line number validation."""
        # Valid line numbers
        sample = DatasetSample(input_text="test", label="ATTACK", line_number=1)
        assert sample.line_number == 1

        # Invalid line numbers
        with pytest.raises(ValidationError):
            DatasetSample(input_text="test", label="ATTACK", line_number=0)

    def test_sample_serialization(self):
        """Test sample JSON serialization/deserialization."""
        timestamp = datetime.now()
        sample = DatasetSample(
            input_text="Test sample",
            label="ATTACK",
            attack_type="malware",
            timestamp=timestamp,
            metadata={"test": True},
        )

        # Serialize to JSON
        json_data = sample.model_dump()
        json_str = sample.model_dump_json()

        assert json_data["input_text"] == "Test sample"
        assert json_data["label"] == "ATTACK"
        assert json_data["attack_type"] == "malware"
        assert "timestamp" in json_data

        # Deserialize from JSON
        sample_from_dict = DatasetSample.model_validate(json_data)
        sample_from_json = DatasetSample.model_validate_json(json_str)

        assert sample_from_dict.input_text == sample.input_text
        assert sample_from_dict.label == sample.label
        assert sample_from_json.input_text == sample.input_text
        assert sample_from_json.label == sample.label


class TestDatasetInfo:
    """Test the DatasetInfo model."""

    def test_dataset_info_creation(self):
        """Test creating dataset info with required fields."""
        info = DatasetInfo(
            name="Test Dataset",
            source="local",
            total_samples=100,
            attack_samples=60,
            benign_samples=40,
        )

        assert info.name == "Test Dataset"
        assert info.source == "local"
        assert info.total_samples == 100
        assert info.attack_samples == 60
        assert info.benign_samples == 40
        assert len(info.id) > 0  # UUID should be generated
        assert info.attack_types == []
        assert info.schema_version == "1.0"
        assert isinstance(info.created_at, datetime)

    def test_dataset_info_full(self):
        """Test creating dataset info with all fields."""
        created_at = datetime.now()
        attack_types = ["malware", "phishing"]
        metadata = {"source_url": "https://example.com", "license": "MIT"}

        info = DatasetInfo(
            id="dataset-001",
            name="Cybersec Dataset",
            source="kaggle",
            total_samples=1000,
            attack_samples=700,
            benign_samples=300,
            attack_types=attack_types,
            schema_version="2.0",
            created_at=created_at,
            description="Test cybersecurity dataset",
            size_bytes=1024000,
            format="jsonl",
            metadata=metadata,
        )

        assert info.id == "dataset-001"
        assert info.name == "Cybersec Dataset"
        assert info.attack_types == attack_types
        assert info.schema_version == "2.0"
        assert info.created_at == created_at
        assert info.description == "Test cybersecurity dataset"
        assert info.size_bytes == 1024000
        assert info.format == "jsonl"
        assert info.metadata == metadata

    def test_sample_count_validation(self):
        """Test that sample counts must be consistent."""
        # Valid counts
        info = DatasetInfo(
            name="Test", source="local", total_samples=100, attack_samples=60, benign_samples=40
        )
        assert info.total_samples == 100

        # Invalid counts
        with pytest.raises(ValidationError, match="must equal total samples"):
            DatasetInfo(
                name="Test",
                source="local",
                total_samples=100,
                attack_samples=70,
                benign_samples=40,  # 70 + 40 = 110, not 100
            )

    def test_attack_types_normalization(self):
        """Test that attack types are normalized to lowercase."""
        info = DatasetInfo(
            name="Test",
            source="local",
            total_samples=10,
            attack_samples=5,
            benign_samples=5,
            attack_types=["MALWARE", "Phishing", "SQL_INJECTION"],
        )

        assert info.attack_types == ["malware", "phishing", "sql_injection"]

    def test_dataset_info_serialization(self):
        """Test dataset info serialization."""
        info = DatasetInfo(
            name="Test Dataset",
            source="local",
            total_samples=50,
            attack_samples=30,
            benign_samples=20,
            attack_types=["malware"],
        )

        json_data = info.model_dump()
        json_str = info.model_dump_json()

        # Deserialize
        info_from_dict = DatasetInfo.model_validate(json_data)
        info_from_json = DatasetInfo.model_validate_json(json_str)

        assert info_from_dict.name == info.name
        assert info_from_dict.total_samples == info.total_samples
        assert info_from_json.name == info.name
        assert info_from_json.attack_types == info.attack_types


class TestDataSplits:
    """Test the DataSplits model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing splits."""
        return [
            DatasetSample(id=f"sample_{i}", input_text=f"Sample {i}", label="ATTACK")
            for i in range(10)
        ]

    def test_splits_creation(self, sample_data):
        """Test creating data splits."""
        splits = DataSplits(
            train=sample_data[:7], test=sample_data[7:9], validation=sample_data[9:10]
        )

        assert splits.train_count == 7
        assert splits.test_count == 2
        assert splits.validation_count == 1
        assert splits.total_count == 10
        assert splits.split_strategy == "random"

    def test_splits_properties(self, sample_data):
        """Test split properties."""
        splits = DataSplits(
            train=sample_data[:5], test=sample_data[5:8], validation=sample_data[8:]
        )

        assert splits.train_count == 5
        assert splits.test_count == 3
        assert splits.validation_count == 2
        assert splits.total_count == 10

    def test_splits_without_validation(self, sample_data):
        """Test splits without validation set."""
        splits = DataSplits(train=sample_data[:7], test=sample_data[7:])

        assert splits.train_count == 7
        assert splits.test_count == 3
        assert splits.validation_count == 0
        assert splits.total_count == 10

    def test_splits_validation_empty(self):
        """Test that at least one split must be non-empty."""
        with pytest.raises(ValidationError, match="At least one of train or test"):
            DataSplits(train=[], test=[])

    def test_splits_ratios_validation(self, sample_data):
        """Test split ratios validation."""
        # Valid ratios
        splits = DataSplits(
            train=sample_data[:7], test=sample_data[7:], split_ratios={"train": 0.7, "test": 0.3}
        )
        assert splits.split_ratios["train"] == 0.7

        # Invalid ratios (don't sum to 1.0)
        with pytest.raises(ValidationError, match="Split ratios must sum to 1.0"):
            DataSplits(
                train=sample_data[:5],
                test=sample_data[5:],
                split_ratios={"train": 0.8, "test": 0.3},  # Sum = 1.1
            )

    def test_splits_serialization(self, sample_data):
        """Test splits serialization."""
        splits = DataSplits(
            train=sample_data[:5], test=sample_data[5:], split_strategy="stratified"
        )

        json_data = splits.model_dump()
        json_str = splits.model_dump_json()

        assert json_data["split_strategy"] == "stratified"
        assert len(json_data["train"]) == 5
        assert len(json_data["test"]) == 5

        # Deserialize
        splits_from_dict = DataSplits.model_validate(json_data)
        splits_from_json = DataSplits.model_validate_json(json_str)

        assert splits_from_dict.split_strategy == "stratified"
        assert splits_from_dict.train_count == 5
        assert splits_from_json.test_count == 5


class TestDataset:
    """Test the Dataset model."""

    @pytest.fixture
    def dataset_info(self):
        """Create dataset info for testing."""
        return DatasetInfo(
            name="Test Dataset",
            source="local",
            total_samples=10,
            attack_samples=6,
            benign_samples=4,
            attack_types=["malware", "phishing"],
        )

    @pytest.fixture
    def sample_list(self):
        """Create sample list for testing."""
        samples = []
        # Add attack samples
        for i in range(6):
            attack_type = "malware" if i < 3 else "phishing"
            samples.append(
                DatasetSample(
                    id=f"attack_{i}",
                    input_text=f"Attack sample {i}",
                    label="ATTACK",
                    attack_type=attack_type,
                )
            )

        # Add benign samples
        for i in range(4):
            samples.append(
                DatasetSample(id=f"benign_{i}", input_text=f"Benign sample {i}", label="BENIGN")
            )

        return samples

    def test_dataset_creation(self, dataset_info, sample_list):
        """Test creating a dataset."""
        dataset = Dataset(info=dataset_info, samples=sample_list)

        assert dataset.info == dataset_info
        assert len(dataset.samples) == 10
        assert dataset.sample_count == 10
        assert dataset.version == "1.0"

    def test_dataset_properties(self, dataset_info, sample_list):
        """Test dataset properties."""
        dataset = Dataset(info=dataset_info, samples=sample_list)

        assert len(dataset.attack_samples) == 6
        assert len(dataset.benign_samples) == 4
        assert set(dataset.attack_types_present) == {"malware", "phishing"}

    def test_dataset_consistency_validation(self, dataset_info, sample_list):
        """Test that dataset auto-corrects info counts."""
        # Create info with wrong counts
        wrong_info = DatasetInfo(
            name="Test Dataset",
            source="local",
            total_samples=5,  # Wrong count
            attack_samples=2,  # Wrong count
            benign_samples=3,  # Wrong count
            attack_types=["ddos"],  # Wrong types
        )

        dataset = Dataset(info=wrong_info, samples=sample_list)

        # Should auto-correct the counts and attack types
        assert dataset.info.total_samples == 10
        assert dataset.info.attack_samples == 6
        assert dataset.info.benign_samples == 4
        assert set(dataset.info.attack_types) == {"malware", "phishing"}

    def test_dataset_get_methods(self, dataset_info, sample_list):
        """Test dataset getter methods."""
        dataset = Dataset(info=dataset_info, samples=sample_list)

        # Test get_samples_by_type
        malware_samples = dataset.get_samples_by_type("malware")
        assert len(malware_samples) == 3
        assert all(s.attack_type == "malware" for s in malware_samples)

        phishing_samples = dataset.get_samples_by_type("phishing")
        assert len(phishing_samples) == 3
        assert all(s.attack_type == "phishing" for s in phishing_samples)

        # Test get_sample_by_id
        sample = dataset.get_sample_by_id("attack_0")
        assert sample is not None
        assert sample.id == "attack_0"

        missing_sample = dataset.get_sample_by_id("nonexistent")
        assert missing_sample is None

    def test_dataset_with_splits(self, dataset_info, sample_list):
        """Test dataset with splits."""
        splits = DataSplits(
            train=sample_list[:7], test=sample_list[7:9], validation=sample_list[9:]
        )

        dataset = Dataset(info=dataset_info, samples=sample_list, splits=splits)

        assert dataset.splits is not None
        assert dataset.splits.train_count == 7
        assert dataset.splits.test_count == 2
        assert dataset.splits.validation_count == 1

    def test_dataset_serialization(self, dataset_info, sample_list):
        """Test dataset serialization."""
        dataset = Dataset(
            info=dataset_info,
            samples=sample_list[:3],  # Use fewer samples for faster testing
            version="2.0",
            checksum="abc123",
        )

        json_data = dataset.model_dump()
        json_str = dataset.model_dump_json()

        assert json_data["version"] == "2.0"
        assert json_data["checksum"] == "abc123"
        assert len(json_data["samples"]) == 3

        # Deserialize
        dataset_from_dict = Dataset.model_validate(json_data)
        dataset_from_json = Dataset.model_validate_json(json_str)

        assert dataset_from_dict.version == "2.0"
        assert dataset_from_dict.sample_count == 3
        assert dataset_from_json.checksum == "abc123"


class TestDataBatch:
    """Test the DataBatch model."""

    @pytest.fixture
    def batch_samples(self):
        """Create samples for batch testing."""
        return [
            DatasetSample(
                id=f"sample_{i}",
                input_text=f"Sample {i}",
                label="ATTACK" if i % 2 == 0 else "BENIGN",
            )
            for i in range(5)
        ]

    def test_batch_creation(self, batch_samples):
        """Test creating a data batch."""
        batch = DataBatch(
            samples=batch_samples,
            dataset_id="test-dataset",
            offset=0,
            total_batches=2,
            batch_size=5,
        )

        assert len(batch.samples) == 5
        assert batch.dataset_id == "test-dataset"
        assert batch.offset == 0
        assert batch.total_batches == 2
        assert batch.batch_size == 5
        assert batch.actual_size == 5
        assert len(batch.batch_id) > 0  # UUID should be generated

    def test_batch_properties(self, batch_samples):
        """Test batch properties."""
        batch = DataBatch(
            samples=batch_samples,
            dataset_id="test-dataset",
            offset=0,
            total_batches=1,
            batch_size=5,
        )

        assert batch.actual_size == 5
        assert batch.attack_count == 3  # samples 0, 2, 4
        assert batch.benign_count == 2  # samples 1, 3
        assert batch.attack_ratio == 0.6

    def test_batch_size_validation(self, batch_samples):
        """Test batch size validation."""
        # Valid batch (actual <= target)
        batch = DataBatch(
            samples=batch_samples[:3],
            dataset_id="test-dataset",
            offset=0,
            total_batches=1,
            batch_size=5,  # Target size larger than actual
        )
        assert batch.actual_size == 3

        # Invalid batch (actual > target)
        with pytest.raises(ValidationError, match="cannot exceed target batch size"):
            DataBatch(
                samples=batch_samples,
                dataset_id="test-dataset",
                offset=0,
                total_batches=1,
                batch_size=3,  # Target size smaller than actual
            )

    def test_batch_offset_validation(self, batch_samples):
        """Test batch offset validation."""
        # Valid offset
        batch = DataBatch(
            samples=batch_samples,
            dataset_id="test-dataset",
            offset=10,
            total_batches=1,
            batch_size=5,
        )
        assert batch.offset == 10

        # Invalid offset
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DataBatch(
                samples=batch_samples,
                dataset_id="test-dataset",
                offset=-1,
                total_batches=1,
                batch_size=5,
            )

    def test_batch_get_samples_by_label(self, batch_samples):
        """Test getting samples by label."""
        batch = DataBatch(
            samples=batch_samples,
            dataset_id="test-dataset",
            offset=0,
            total_batches=1,
            batch_size=5,
        )

        attack_samples = batch.get_samples_by_label("ATTACK")
        benign_samples = batch.get_samples_by_label("benign")  # Test case insensitive

        assert len(attack_samples) == 3
        assert len(benign_samples) == 2
        assert all(s.label == "ATTACK" for s in attack_samples)
        assert all(s.label == "BENIGN" for s in benign_samples)

    def test_batch_serialization(self, batch_samples):
        """Test batch serialization."""
        batch = DataBatch(
            samples=batch_samples[:3],
            batch_id="batch-001",
            dataset_id="test-dataset",
            offset=0,
            total_batches=1,
            batch_size=3,
            metadata={"processor": "test"},
        )

        json_data = batch.model_dump()
        json_str = batch.model_dump_json()

        assert json_data["batch_id"] == "batch-001"
        assert json_data["dataset_id"] == "test-dataset"
        assert len(json_data["samples"]) == 3
        assert json_data["metadata"]["processor"] == "test"

        # Deserialize
        batch_from_dict = DataBatch.model_validate(json_data)
        batch_from_json = DataBatch.model_validate_json(json_str)

        assert batch_from_dict.batch_id == "batch-001"
        assert batch_from_dict.actual_size == 3
        assert batch_from_json.dataset_id == "test-dataset"


class TestModelRelationships:
    """Test relationships between different models."""

    def test_dataset_with_complete_structure(self):
        """Test a complete dataset structure with all components."""
        # Create samples
        samples = []
        for i in range(20):
            label = "ATTACK" if i < 12 else "BENIGN"
            attack_type = "malware" if i < 6 else ("phishing" if i < 12 else None)
            samples.append(
                DatasetSample(
                    id=f"sample_{i}",
                    input_text=f"Sample content {i}",
                    label=label,
                    attack_type=attack_type,
                )
            )

        # Create dataset info
        info = DatasetInfo(
            name="Complete Test Dataset",
            source="synthetic",
            total_samples=20,
            attack_samples=12,
            benign_samples=8,
            attack_types=["malware", "phishing"],
        )

        # Create splits
        splits = DataSplits(
            train=samples[:14],
            test=samples[14:18],
            validation=samples[18:],
            split_strategy="stratified",
        )

        # Create complete dataset
        dataset = Dataset(info=info, samples=samples, splits=splits)

        # Test relationships
        assert dataset.sample_count == 20
        assert dataset.info.total_samples == 20
        assert dataset.splits.total_count == 20
        assert len(dataset.attack_samples) == 12
        assert len(dataset.benign_samples) == 8
        assert set(dataset.attack_types_present) == {"malware", "phishing"}

        # Create batches from dataset
        batch1 = DataBatch(
            samples=dataset.samples[:5],
            dataset_id=dataset.info.id,
            offset=0,
            total_batches=4,
            batch_size=5,
        )

        batch2 = DataBatch(
            samples=dataset.samples[5:10],
            dataset_id=dataset.info.id,
            offset=5,
            total_batches=4,
            batch_size=5,
        )

        assert batch1.dataset_id == dataset.info.id
        assert batch2.dataset_id == dataset.info.id
        assert batch1.actual_size == 5
        assert batch2.actual_size == 5

    def test_model_consistency_across_serialization(self):
        """Test model consistency across JSON serialization/deserialization."""
        # Create original dataset
        samples = [
            DatasetSample(
                input_text="Test sample",
                label="ATTACK",
                attack_type="malware",
                metadata={"source": "test"},
            )
        ]

        info = DatasetInfo(
            name="Serialization Test",
            source="test",
            total_samples=1,
            attack_samples=1,
            benign_samples=0,
            attack_types=["malware"],
        )

        original_dataset = Dataset(info=info, samples=samples)

        # Serialize to JSON and back
        json_str = original_dataset.model_dump_json()
        deserialized_dataset = Dataset.model_validate_json(json_str)

        # Test consistency
        assert deserialized_dataset.info.name == original_dataset.info.name
        assert deserialized_dataset.sample_count == original_dataset.sample_count
        assert deserialized_dataset.samples[0].input_text == original_dataset.samples[0].input_text
        assert deserialized_dataset.samples[0].label == original_dataset.samples[0].label
        assert (
            deserialized_dataset.samples[0].attack_type == original_dataset.samples[0].attack_type
        )


class TestModelConstants:
    """Test model constants and utilities."""

    def test_valid_labels_constant(self):
        """Test VALID_LABELS constant."""
        assert "ATTACK" in VALID_LABELS
        assert "BENIGN" in VALID_LABELS
        assert len(VALID_LABELS) == 2

    def test_common_attack_types_constant(self):
        """Test COMMON_ATTACK_TYPES constant."""
        expected_types = [
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

        for attack_type in expected_types:
            assert attack_type in COMMON_ATTACK_TYPES

        assert len(COMMON_ATTACK_TYPES) >= len(expected_types)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_input_text(self):
        """Test that empty input text is rejected."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            DatasetSample(input_text="", label="ATTACK")

    def test_zero_samples_dataset_info(self):
        """Test dataset info with zero samples."""
        info = DatasetInfo(
            name="Empty Dataset", source="test", total_samples=0, attack_samples=0, benign_samples=0
        )
        assert info.total_samples == 0
        assert info.attack_samples == 0
        assert info.benign_samples == 0

    def test_large_batch_numbers(self):
        """Test with large batch numbers."""
        samples = [DatasetSample(input_text="test", label="ATTACK")]

        batch = DataBatch(
            samples=samples, dataset_id="test", offset=1000000, total_batches=2000000, batch_size=1
        )

        assert batch.offset == 1000000
        assert batch.total_batches == 2000000

    def test_unicode_text_handling(self):
        """Test handling of Unicode text."""
        unicode_text = "攻撃パターン detected 🚨"

        sample = DatasetSample(input_text=unicode_text, label="ATTACK", attack_type="malware")

        assert sample.input_text == unicode_text

        # Test serialization with Unicode
        json_str = sample.model_dump_json()
        deserialized = DatasetSample.model_validate_json(json_str)

        assert deserialized.input_text == unicode_text

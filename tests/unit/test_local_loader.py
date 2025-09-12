"""
Unit tests for LocalFileDataLoader.

This module tests the local file data loader functionality including
loading different file formats, field mapping, validation, and error handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from benchmark.data.loaders.base_loader import DataLoadError, FieldMapping
from benchmark.data.loaders.local_loader import LocalFileDataLoader
from benchmark.data.models import Dataset, DatasetSample


class TestFieldMapping:
    """Test FieldMapping class functionality."""

    def test_field_mapping_defaults(self):
        """Test default field mapping configuration."""
        mapping = FieldMapping()

        assert mapping.input_text_field == "input_text"
        assert mapping.label_field == "label"
        assert mapping.attack_type_field == "attack_type"
        assert mapping.metadata_fields == []
        assert mapping.required_fields == ["input_text", "label"]

    def test_field_mapping_custom(self):
        """Test custom field mapping configuration."""
        mapping = FieldMapping(
            input_text_field="request_text",
            label_field="classification",
            attack_type_field="threat_category",
            metadata_fields=["source", "severity"],
            required_fields=["request_text", "classification", "source"],
        )

        assert mapping.input_text_field == "request_text"
        assert mapping.label_field == "classification"
        assert mapping.attack_type_field == "threat_category"
        assert mapping.metadata_fields == ["source", "severity"]
        assert mapping.required_fields == ["request_text", "classification", "source"]

    def test_validate_fields_success(self):
        """Test successful field validation."""
        mapping = FieldMapping()
        sample = {"input_text": "test input", "label": "ATTACK", "attack_type": "malware"}

        # Should not raise exception
        mapping.validate_fields(sample)

    def test_validate_fields_missing_required(self):
        """Test field validation with missing required fields."""
        mapping = FieldMapping()
        sample = {
            "input_text": "test input"
            # Missing label field
        }

        with pytest.raises(ValueError, match="Missing required fields"):
            mapping.validate_fields(sample)

    def test_extract_metadata(self):
        """Test metadata extraction."""
        mapping = FieldMapping(metadata_fields=["source", "severity", "timestamp"])
        sample = {
            "input_text": "test input",
            "label": "ATTACK",
            "source": "test_data",
            "severity": "high",
            "timestamp": "2024-01-15T10:30:00Z",
            "other_field": "should_not_be_included",
        }

        metadata = mapping.extract_metadata(sample)

        assert metadata == {
            "source": "test_data",
            "severity": "high",
            "timestamp": "2024-01-15T10:30:00Z",
        }


class TestLocalFileDataLoader:
    """Test LocalFileDataLoader functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def loader(self):
        """Create LocalFileDataLoader instance."""
        return LocalFileDataLoader()

    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create sample JSON file for testing."""
        data = [
            {
                "input_text": "SELECT * FROM users WHERE id = 1",
                "label": "ATTACK",
                "attack_type": "sql_injection",
                "source": "test_data",
            },
            {"input_text": "GET /api/users HTTP/1.1", "label": "BENIGN", "source": "test_data"},
        ]

        file_path = temp_dir / "test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        return file_path

    @pytest.fixture
    def sample_csv_file(self, temp_dir):
        """Create sample CSV file for testing."""
        csv_content = """input_text,label,attack_type,source
"SELECT * FROM users WHERE id = 1",ATTACK,sql_injection,test_data
"GET /api/users HTTP/1.1",BENIGN,,test_data"""

        file_path = temp_dir / "test.csv"
        with open(file_path, "w") as f:
            f.write(csv_content)

        return file_path

    @pytest.fixture
    def malformed_json_file(self, temp_dir):
        """Create malformed JSON file for testing."""
        file_path = temp_dir / "malformed.json"
        with open(file_path, "w") as f:
            f.write('{"invalid": "json" "missing_comma": true}')

        return file_path

    def test_get_supported_formats(self, loader):
        """Test getting supported file formats."""
        formats = loader.get_supported_formats()

        assert "json" in formats
        assert "csv" in formats
        # parquet might not be available without pandas

    def test_detect_format_json(self, loader):
        """Test format detection for JSON files."""
        assert loader._detect_format(Path("test.json")) == "json"
        assert loader._detect_format(Path("test.jsonl")) == "json"

    def test_detect_format_csv(self, loader):
        """Test format detection for CSV files."""
        assert loader._detect_format(Path("test.csv")) == "csv"
        assert loader._detect_format(Path("test.tsv")) == "csv"

    def test_detect_format_parquet(self, loader):
        """Test format detection for Parquet files."""
        if "parquet" in loader.get_supported_formats():
            assert loader._detect_format(Path("test.parquet")) == "parquet"
            assert loader._detect_format(Path("test.pq")) == "parquet"

    def test_detect_format_unsupported(self, loader):
        """Test format detection for unsupported files."""
        with pytest.raises(DataLoadError, match="Cannot detect format"):
            loader._detect_format(Path("test.txt"))

    @pytest.mark.asyncio
    async def test_validate_source_existing_file(self, loader, sample_json_file):
        """Test source validation with existing file."""
        config = {"file_path": str(sample_json_file)}

        is_valid = await loader.validate_source(config)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_source_nonexistent_file(self, loader):
        """Test source validation with non-existent file."""
        config = {"file_path": "/nonexistent/path/file.json"}

        is_valid = await loader.validate_source(config)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_source_directory(self, loader, temp_dir):
        """Test source validation with directory instead of file."""
        config = {"file_path": str(temp_dir)}

        is_valid = await loader.validate_source(config)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_load_json_file(self, loader, sample_json_file):
        """Test loading JSON file."""
        config = {"file_path": str(sample_json_file), "name": "Test Dataset"}

        dataset = await loader.load(config)

        assert isinstance(dataset, Dataset)
        assert dataset.info.name == "Test Dataset"
        assert dataset.sample_count == 2
        assert len(dataset.attack_samples) == 1
        assert len(dataset.benign_samples) == 1

        # Check first sample
        attack_sample = dataset.attack_samples[0]
        assert attack_sample.input_text == "SELECT * FROM users WHERE id = 1"
        assert attack_sample.label == "ATTACK"
        assert attack_sample.attack_type == "sql_injection"
        assert (
            attack_sample.metadata["source"] == "test_data"
        )  # Should be in metadata automatically

    @pytest.mark.asyncio
    async def test_load_csv_file(self, loader, sample_csv_file):
        """Test loading CSV file."""
        config = {"file_path": str(sample_csv_file), "name": "CSV Test Dataset"}

        dataset = await loader.load(config)

        assert isinstance(dataset, Dataset)
        assert dataset.info.name == "CSV Test Dataset"
        assert dataset.sample_count == 2
        assert len(dataset.attack_samples) == 1
        assert len(dataset.benign_samples) == 1

    @pytest.mark.asyncio
    async def test_load_custom_field_mapping(self, loader, temp_dir):
        """Test loading with custom field mapping."""
        # Create CSV with custom fields
        csv_content = """request_text,classification,threat_category,data_source
"SELECT * FROM users",malicious,sql_injection,web_logs
"GET /index.html",benign,,web_logs"""

        file_path = temp_dir / "custom_fields.csv"
        with open(file_path, "w") as f:
            f.write(csv_content)

        config = {
            "file_path": str(file_path),
            "field_mapping": {
                "input_text_field": "request_text",
                "label_field": "classification",
                "attack_type_field": "threat_category",
                "metadata_fields": ["data_source"],
            },
        }

        dataset = await loader.load(config)

        assert dataset.sample_count == 2

        # Check that field mapping worked
        samples = dataset.samples
        assert samples[0].input_text == "SELECT * FROM users"
        assert samples[0].label == "ATTACK"  # 'malicious' should be normalized to 'ATTACK'
        assert samples[0].attack_type == "sql_injection"
        assert samples[0].metadata["data_source"] == "web_logs"

    @pytest.mark.asyncio
    async def test_load_malformed_json(self, loader, malformed_json_file):
        """Test loading malformed JSON file."""
        config = {"file_path": str(malformed_json_file)}

        with pytest.raises(DataLoadError, match="Invalid JSON format"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_missing_file(self, loader):
        """Test loading non-existent file."""
        config = {"file_path": "/nonexistent/file.json"}

        with pytest.raises(DataLoadError, match="Source file not accessible"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_unsupported_format(self, loader, temp_dir):
        """Test loading unsupported file format."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("some text content")

        config = {"file_path": str(file_path)}

        with pytest.raises(DataLoadError, match="Cannot detect format"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_csv_custom_options(self, loader, temp_dir):
        """Test loading CSV with custom options."""
        # Create TSV file (tab-separated)
        tsv_content = "input_text\tlabel\tattack_type\n"
        tsv_content += "SQL injection attempt\tATTACK\tsql_injection\n"
        tsv_content += "Normal request\tBENIGN\t\n"

        file_path = temp_dir / "test.tsv"
        with open(file_path, "w") as f:
            f.write(tsv_content)

        config = {
            "file_path": str(file_path),
            "format": "csv",  # Override detected format
            "csv_options": {"delimiter": "\t"},
        }

        dataset = await loader.load(config)

        assert dataset.sample_count == 2
        assert "SQL injection attempt" in [s.input_text for s in dataset.samples]

    def test_map_fields_success(self, loader):
        """Test successful field mapping."""
        raw_data = [
            {
                "input_text": "test input",
                "label": "ATTACK",
                "attack_type": "malware",
                "source": "test_data",
            }
        ]

        field_mapping = FieldMapping(metadata_fields=["source"])
        samples = loader._map_fields(raw_data, field_mapping)

        assert len(samples) == 1
        sample = samples[0]

        assert isinstance(sample, DatasetSample)
        assert sample.input_text == "test input"
        assert sample.label == "ATTACK"
        assert sample.attack_type == "malware"
        assert sample.metadata["source"] == "test_data"
        assert sample.metadata["source_index"] == 0

    def test_map_fields_missing_required(self, loader):
        """Test field mapping with missing required fields."""
        raw_data = [
            {
                "input_text": "test input"
                # Missing label field
            }
        ]

        field_mapping = FieldMapping()

        with pytest.raises(DataLoadError, match="Failed to process record"):
            loader._map_fields(raw_data, field_mapping)

    def test_create_dataset_info(self, loader):
        """Test dataset info creation."""
        file_path = Path("/test/path/dataset.json")
        samples = [
            DatasetSample(input_text="test1", label="ATTACK", attack_type="malware"),
            DatasetSample(input_text="test2", label="BENIGN"),
            DatasetSample(input_text="test3", label="ATTACK", attack_type="phishing"),
        ]

        # Mock the stat method result with proper signature
        def mock_stat(follow_symlinks=True):  # noqa: ARG001
            mock_result = Mock()
            mock_result.st_size = 1024
            return mock_result

        with patch.object(Path, "stat", mock_stat):
            info = loader._create_dataset_info(
                file_path=file_path,
                samples=samples,
                name="Test Dataset",
                description="Test description",
                format_name="json",
            )

        assert info.name == "Test Dataset"
        assert info.description == "Test description"
        assert info.total_samples == 3
        assert info.attack_samples == 2
        assert info.benign_samples == 1
        assert set(info.attack_types) == {"malware", "phishing"}
        assert info.format == "json"
        assert info.size_bytes == 1024
        assert "dataset.json" in info.source


class TestLoadProgressTracking:
    """Test progress tracking functionality."""

    @pytest.fixture
    def loader(self):
        """Create loader with progress tracking enabled."""
        return LocalFileDataLoader(show_progress=True)

    def test_progress_callback_registration(self, loader):
        """Test progress callback registration."""
        callback = Mock()
        loader.add_progress_callback(callback)

        assert callback in loader.progress_callbacks

    @pytest.mark.asyncio
    async def test_progress_tracking_large_json(self, loader, temp_dir):
        """Test progress tracking for large JSON files."""
        # Create a JSON file with enough data to exceed threshold
        # Make each sample larger to ensure file size exceeds PROGRESS_THRESHOLD_BYTES (10MB)
        large_text = "A" * 1000  # 1KB of text per sample
        large_data = [
            {
                "input_text": f"Sample {i}: {large_text}",
                "label": "BENIGN" if i % 2 == 0 else "ATTACK",
                "attack_type": "malware" if i % 2 == 1 else None,
            }
            for i in range(15000)  # 15K samples * 1KB each = ~15MB
        ]

        file_path = temp_dir / "large.json"
        with open(file_path, "w") as f:
            json.dump(large_data, f)

        # Verify file is actually large enough
        actual_size = file_path.stat().st_size
        assert actual_size > loader.PROGRESS_THRESHOLD_BYTES, (
            f"File size {actual_size} should exceed {loader.PROGRESS_THRESHOLD_BYTES}"
        )

        # Add progress callback to track calls
        progress_callback = Mock()
        loader.add_progress_callback(progress_callback)

        config = {"file_path": str(file_path)}
        dataset = await loader.load(config)

        assert dataset.sample_count == 15000
        # Progress callback should have been called at least once
        # (start and complete calls)
        assert progress_callback.call_count >= 2


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def loader(self):
        """Create LocalFileDataLoader instance."""
        return LocalFileDataLoader()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_permission_error(self, loader, temp_dir):
        """Test handling of permission errors."""
        file_path = temp_dir / "test.json"
        file_path.write_text('[{"input_text": "test", "label": "BENIGN"}]')

        config = {"file_path": str(file_path)}

        # Mock permission error during file reading
        with (
            patch("builtins.open", side_effect=PermissionError("Permission denied")),
            pytest.raises(DataLoadError, match="Source file not accessible"),
        ):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_invalid_config(self, loader):
        """Test handling of invalid configuration."""
        config = {}  # Missing file_path

        with pytest.raises(ValueError, match="Missing required configuration keys"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_corrupted_csv_data(self, loader, temp_dir):
        """Test handling of corrupted CSV data."""
        # Create CSV with inconsistent columns that will cause validation errors
        csv_content = "input_text,label\n"
        csv_content += "Valid row,BENIGN\n"
        csv_content += "Invalid row with extra,columns,and,more\n"

        file_path = temp_dir / "corrupted.csv"
        with open(file_path, "w") as f:
            f.write(csv_content)

        config = {"file_path": str(file_path)}

        # Should fail due to invalid label format in the corrupted row
        with pytest.raises(DataLoadError, match="Failed to process record"):
            await loader.load(config)

    def test_data_load_error_creation(self):
        """Test DataLoadError exception creation."""
        source_path = Path("/test/path")
        cause = ValueError("Original error")

        error = DataLoadError("Test error", source_path, cause)

        assert "Test error" in str(error)
        assert str(source_path) in str(error)
        assert "Original error" in str(error)


class TestJSONFileStructures:
    """Test different JSON file structures."""

    @pytest.fixture
    def loader(self):
        """Create LocalFileDataLoader instance."""
        return LocalFileDataLoader()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_json_list_structure(self, loader, temp_dir):
        """Test JSON file with list structure."""
        data = [
            {"input_text": "test1", "label": "ATTACK"},
            {"input_text": "test2", "label": "BENIGN"},
        ]

        file_path = temp_dir / "list.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        config = {"file_path": str(file_path)}
        dataset = await loader.load(config)

        assert dataset.sample_count == 2

    @pytest.mark.asyncio
    async def test_json_dict_with_samples_key(self, loader, temp_dir):
        """Test JSON file with dict structure containing 'samples' key."""
        data = {
            "metadata": {"source": "test"},
            "samples": [
                {"input_text": "test1", "label": "ATTACK"},
                {"input_text": "test2", "label": "BENIGN"},
            ],
        }

        file_path = temp_dir / "dict_samples.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        config = {"file_path": str(file_path)}
        dataset = await loader.load(config)

        assert dataset.sample_count == 2

    @pytest.mark.asyncio
    async def test_json_dict_with_data_key(self, loader, temp_dir):
        """Test JSON file with dict structure containing 'data' key."""
        data = {
            "info": {"name": "test dataset"},
            "data": [
                {"input_text": "test1", "label": "ATTACK"},
                {"input_text": "test2", "label": "BENIGN"},
            ],
        }

        file_path = temp_dir / "dict_data.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        config = {"file_path": str(file_path)}
        dataset = await loader.load(config)

        assert dataset.sample_count == 2

    @pytest.mark.asyncio
    async def test_json_single_dict(self, loader, temp_dir):
        """Test JSON file with single dictionary."""
        data = {"input_text": "single sample", "label": "BENIGN"}

        file_path = temp_dir / "single.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        config = {"file_path": str(file_path)}
        dataset = await loader.load(config)

        assert dataset.sample_count == 1

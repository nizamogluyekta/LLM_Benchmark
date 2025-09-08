"""
Unit tests for HuggingFaceDataLoader.

This module tests the HuggingFace data loader functionality including
loading datasets, field mapping, streaming mode, validation, and error handling.
"""

from unittest.mock import Mock, patch

import pytest

from benchmark.data.loaders.base_loader import DataLoadError, FieldMapping


class TestHuggingFaceDataLoaderImport:
    """Test HuggingFaceDataLoader import behavior."""

    def test_import_without_datasets(self):
        """Test importing loader without datasets library installed."""
        with (
            patch.dict("sys.modules", {"datasets": None}),
            patch("benchmark.data.loaders.huggingface_loader.DATASETS_AVAILABLE", False),
        ):
            from benchmark.data.loaders.huggingface_loader import HuggingFaceDataLoader

            with pytest.raises(ImportError, match="HuggingFace datasets library is required"):
                HuggingFaceDataLoader()

    def test_import_with_datasets(self):
        """Test importing loader with datasets library available."""
        with patch("benchmark.data.loaders.huggingface_loader.DATASETS_AVAILABLE", True):
            from benchmark.data.loaders.huggingface_loader import HuggingFaceDataLoader

            loader = HuggingFaceDataLoader()
            assert loader is not None


@pytest.fixture
def mock_datasets():
    """Mock the datasets library."""
    # Create mock objects
    mock_load = Mock()
    mock_infos = Mock()
    mock_configs = Mock()
    mock_hf_dataset = Mock()
    mock_dataset_dict = Mock()
    mock_iterable_dataset = Mock()

    # Create a mock datasets module
    mock_datasets_module = Mock()
    mock_datasets_module.Dataset = mock_hf_dataset
    mock_datasets_module.DatasetDict = mock_dataset_dict
    mock_datasets_module.IterableDataset = mock_iterable_dataset
    mock_datasets_module.load_dataset = mock_load
    mock_datasets_module.get_dataset_infos = mock_infos
    mock_datasets_module.get_dataset_config_names = mock_configs

    with (
        patch.dict("sys.modules", {"datasets": mock_datasets_module}),
        patch("benchmark.data.loaders.huggingface_loader.DATASETS_AVAILABLE", True),
    ):
        # Reimport the module to pick up the mocked datasets
        import importlib

        import benchmark.data.loaders.huggingface_loader

        importlib.reload(benchmark.data.loaders.huggingface_loader)

        yield {
            "load_dataset": mock_load,
            "get_dataset_infos": mock_infos,
            "get_dataset_config_names": mock_configs,
        }


@pytest.fixture
def loader(mock_datasets):  # noqa: ARG001
    """Create HuggingFaceDataLoader instance with mocked dependencies."""
    from benchmark.data.loaders.huggingface_loader import HuggingFaceDataLoader

    return HuggingFaceDataLoader()


@pytest.fixture
def sample_hf_dataset():
    """Create sample HuggingFace dataset mock."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=3)
    mock_dataset.__iter__ = Mock(
        return_value=iter(
            [
                {
                    "text": "SELECT * FROM users WHERE id = 1",
                    "label": "ATTACK",
                    "attack_type": "sql_injection",
                    "source": "test_data",
                    "extra_field": "extra_value",
                },
                {"text": "GET /api/users HTTP/1.1", "label": "BENIGN", "source": "test_data"},
                {
                    "text": "<script>alert('xss')</script>",
                    "label": "ATTACK",
                    "attack_type": "xss",
                    "severity": "high",
                },
            ]
        )
    )
    return mock_dataset


@pytest.fixture
def sample_streaming_dataset():
    """Create sample streaming HuggingFace dataset mock."""
    mock_dataset = Mock()
    # Streaming datasets don't have __len__
    mock_dataset.__iter__ = Mock(
        return_value=iter(
            [
                {
                    "text": "SELECT * FROM users WHERE id = 1",
                    "label": 1,  # Numeric label to test normalization
                    "attack_type": "sql_injection",
                },
                {
                    "text": "GET /api/users HTTP/1.1",
                    "label": 0,  # Numeric label to test normalization
                },
            ]
        )
    )
    return mock_dataset


class TestHuggingFaceDataLoader:
    """Test HuggingFaceDataLoader functionality."""

    def test_init_default(self, loader):
        """Test loader initialization with default parameters."""
        assert loader.field_mapping.input_text_field == "input_text"
        assert loader.field_mapping.label_field == "label"
        assert loader.streaming is False
        assert loader.cache_dir is None
        assert loader.show_progress is True

    def test_init_custom(self, mock_datasets):
        """Test loader initialization with custom parameters."""
        from benchmark.data.loaders.huggingface_loader import HuggingFaceDataLoader

        field_mapping = FieldMapping(
            input_text_field="request", label_field="classification", attack_type_field="threat"
        )

        loader = HuggingFaceDataLoader(
            field_mapping=field_mapping, streaming=True, cache_dir="/tmp/cache", show_progress=False
        )

        assert loader.field_mapping.input_text_field == "request"
        assert loader.streaming is True
        assert loader.cache_dir == "/tmp/cache"
        assert loader.show_progress is False

    def test_get_supported_formats(self, loader):
        """Test getting supported formats."""
        formats = loader.get_supported_formats()
        assert "huggingface" in formats
        assert "hf" in formats

    def test_add_progress_callback(self, loader):
        """Test adding progress callback."""
        callback = Mock()
        loader.add_progress_callback(callback)
        assert callback in loader.progress_callbacks

    def test_get_field_mapping_explicit(self, loader):
        """Test getting field mapping from explicit config."""
        config = {
            "field_mapping": {
                "input_text_field": "request_text",
                "label_field": "classification",
                "attack_type_field": "threat_type",
                "metadata_fields": ["severity", "source"],
                "required_fields": ["request_text", "classification"],
            }
        }

        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "request_text"
        assert mapping.label_field == "classification"
        assert mapping.attack_type_field == "threat_type"
        assert mapping.metadata_fields == ["severity", "source"]
        assert mapping.required_fields == ["request_text", "classification"]

    def test_get_field_mapping_auto_detect(self, loader):
        """Test auto-detection of field mapping based on dataset name."""
        # Test malware dataset detection
        config = {"dataset_path": "cybersec/malware-detection"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "text"
        assert mapping.label_field == "malware"
        assert mapping.attack_type_field == "family"

        # Test phishing dataset detection
        config = {"dataset_path": "security/phishing-urls"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "url"
        assert mapping.label_field == "label"
        assert mapping.attack_type_field == "type"

        # Test network dataset detection
        config = {"dataset_path": "network-intrusion-detection"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "packet"
        assert mapping.label_field == "label"
        assert mapping.attack_type_field == "attack"

    def test_get_field_mapping_default(self, loader):
        """Test default field mapping fallback."""
        config = {"dataset_path": "some/unknown/dataset"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "text"
        assert mapping.label_field == "label"
        assert mapping.attack_type_field == "attack_type"

    def test_normalize_label_attack_variants(self, loader):
        """Test label normalization for attack variants."""
        attack_labels = [
            "attack",
            "malicious",
            "malware",
            "threat",
            "bad",
            "positive",
            "1",
            "true",
            "anomaly",
            "intrusion",
            "phishing",
            "spam",
            "fraud",
            "suspicious",
            "harmful",
        ]

        for label in attack_labels:
            assert loader._normalize_label(label) == "ATTACK"
            assert loader._normalize_label(label.upper()) == "ATTACK"
            assert loader._normalize_label(f" {label} ") == "ATTACK"

    def test_normalize_label_benign_variants(self, loader):
        """Test label normalization for benign variants."""
        benign_labels = [
            "benign",
            "normal",
            "clean",
            "safe",
            "good",
            "negative",
            "0",
            "false",
            "legitimate",
            "ham",
            "ok",
            "valid",
            "trusted",
        ]

        for label in benign_labels:
            assert loader._normalize_label(label) == "BENIGN"
            assert loader._normalize_label(label.upper()) == "BENIGN"
            assert loader._normalize_label(f" {label} ") == "BENIGN"

    def test_normalize_label_unknown(self, loader):
        """Test label normalization for unknown labels."""
        assert loader._normalize_label("unknown_label") == "UNKNOWN_LABEL"
        assert loader._normalize_label("custom") == "CUSTOM"
        assert loader._normalize_label("") == ""

    @pytest.mark.asyncio
    async def test_get_dataset_info_success(self, loader, mock_datasets):
        """Test getting dataset info successfully."""
        # Mock dataset info object
        mock_info = Mock()
        mock_info.__dict__ = {
            "description": "Test dataset",
            "splits": {"train": {"num_examples": 1000}},
            "features": {"text": "string", "label": "int"},
        }

        mock_datasets["get_dataset_infos"].return_value = {"default": mock_info}

        info = await loader._get_dataset_info("test/dataset")

        assert info["description"] == "Test dataset"
        assert "splits" in info
        assert "features" in info
        mock_datasets["get_dataset_infos"].assert_called_once_with("test/dataset")

    @pytest.mark.asyncio
    async def test_get_dataset_info_with_config_name(self, loader, mock_datasets):
        """Test getting dataset info with specific config name."""
        mock_info = Mock()
        mock_info.__dict__ = {"description": "Config-specific dataset"}

        mock_datasets["get_dataset_infos"].return_value = {
            "default": Mock(),
            "specific_config": mock_info,
        }

        info = await loader._get_dataset_info("test/dataset", "specific_config")

        assert info["description"] == "Config-specific dataset"
        mock_datasets["get_dataset_infos"].assert_called_once_with("test/dataset")

    @pytest.mark.asyncio
    async def test_get_dataset_info_config_not_found(self, loader, mock_datasets):
        """Test getting dataset info with non-existent config name."""
        mock_info = Mock()
        mock_info.__dict__ = {"description": "Default config"}
        # Reset and set up the mock for this specific test
        mock_datasets["get_dataset_infos"].reset_mock()
        mock_datasets["get_dataset_infos"].return_value = {"default": mock_info}

        with pytest.raises(ValueError, match="Dataset config 'nonexistent' not found"):
            await loader._get_dataset_info("test/dataset", "nonexistent")

    @pytest.mark.asyncio
    async def test_get_dataset_info_fallback_to_configs(self, loader, mock_datasets):
        """Test fallback to config names when dataset infos fail."""
        mock_datasets["get_dataset_infos"].side_effect = Exception("Info not available")
        mock_datasets["get_dataset_config_names"].return_value = ["config1", "config2"]

        info = await loader._get_dataset_info("test/dataset")

        assert info == {"configs": ["config1", "config2"]}

    @pytest.mark.asyncio
    async def test_get_dataset_info_complete_failure(self, loader, mock_datasets):
        """Test complete failure to get any dataset info."""
        mock_datasets["get_dataset_infos"].side_effect = Exception("Info not available")
        mock_datasets["get_dataset_config_names"].side_effect = Exception("Configs not available")

        info = await loader._get_dataset_info("test/dataset")

        assert info == {}

    @pytest.mark.asyncio
    async def test_validate_source_success(self, loader, mock_datasets):
        """Test successful source validation."""
        mock_info = Mock()
        mock_info.__dict__ = {"description": "Valid dataset"}
        mock_datasets["get_dataset_infos"].return_value = {"default": mock_info}

        config = {"dataset_path": "test/dataset"}
        is_valid = await loader.validate_source(config)

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_source_missing_path(self, loader):
        """Test source validation with missing dataset path."""
        config = {}
        is_valid = await loader.validate_source(config)

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_source_dataset_not_found(self, loader, mock_datasets):
        """Test source validation with non-existent dataset."""
        # Reset the mocks to fail for this specific test
        mock_datasets["get_dataset_infos"].reset_mock(side_effect=True)
        mock_datasets["get_dataset_config_names"].reset_mock(side_effect=True)
        mock_datasets["get_dataset_infos"].side_effect = Exception("Dataset not found")
        mock_datasets["get_dataset_config_names"].side_effect = Exception("Dataset not found")

        config = {"dataset_path": "nonexistent/dataset"}
        is_valid = await loader.validate_source(config)

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_load_dataset_streaming(self, loader, mock_datasets, sample_streaming_dataset):
        """Test loading dataset in streaming mode."""
        mock_datasets["load_dataset"].return_value = sample_streaming_dataset

        result = await loader._load_dataset_streaming("test/dataset", None, "train", {})

        assert result == sample_streaming_dataset
        mock_datasets["load_dataset"].assert_called_once_with(
            path="test/dataset", split="train", streaming=True, cache_dir=None
        )

    @pytest.mark.asyncio
    async def test_load_dataset_streaming_with_config(
        self, loader, mock_datasets, sample_streaming_dataset
    ):
        """Test loading dataset in streaming mode with config name."""
        mock_datasets["load_dataset"].return_value = sample_streaming_dataset

        await loader._load_dataset_streaming("test/dataset", "specific_config", "train", {})

        mock_datasets["load_dataset"].assert_called_once_with(
            path="test/dataset",
            name="specific_config",
            split="train",
            streaming=True,
            cache_dir=None,
        )

    @pytest.mark.asyncio
    async def test_load_dataset_streaming_error(self, loader, mock_datasets):
        """Test error handling in streaming mode."""
        mock_datasets["load_dataset"].side_effect = Exception("Loading failed")

        with pytest.raises(DataLoadError, match="Failed to load streaming dataset"):
            await loader._load_dataset_streaming("test/dataset", None, "train", {})

    @pytest.mark.asyncio
    async def test_load_dataset_full(self, loader, mock_datasets, sample_hf_dataset):
        """Test loading full dataset into memory."""
        mock_datasets["load_dataset"].return_value = sample_hf_dataset

        result = await loader._load_dataset_full("test/dataset", None, "train", {})

        assert result == sample_hf_dataset
        mock_datasets["load_dataset"].assert_called_once_with(
            path="test/dataset", split="train", cache_dir=None
        )

    @pytest.mark.asyncio
    async def test_load_dataset_full_error(self, loader, mock_datasets):
        """Test error handling in full loading mode."""
        mock_datasets["load_dataset"].side_effect = Exception("Loading failed")

        with pytest.raises(DataLoadError, match="Failed to load full dataset"):
            await loader._load_dataset_full("test/dataset", None, "train", {})

    @pytest.mark.asyncio
    async def test_map_huggingface_fields_success(self, loader, sample_hf_dataset):
        """Test successful field mapping from HuggingFace dataset."""
        field_mapping = FieldMapping(
            input_text_field="text",
            label_field="label",
            attack_type_field="attack_type",
            metadata_fields=["source"],
        )

        samples = await loader._map_huggingface_fields(sample_hf_dataset, field_mapping, {})

        assert len(samples) == 3

        # Check first sample (attack)
        assert samples[0].input_text == "SELECT * FROM users WHERE id = 1"
        assert samples[0].label == "ATTACK"
        assert samples[0].attack_type == "sql_injection"
        # The loader overwrites the 'source' field with 'huggingface'
        # Original 'source' should be preserved under a different key since it's not in metadata_fields
        assert samples[0].metadata["source"] == "huggingface"  # This is set by the loader
        assert samples[0].metadata["hf_index"] == 0
        assert samples[0].metadata["extra_field"] == "extra_value"
        # The original sample's 'source' field should still be preserved in metadata since it wasn't mapped

        # Check second sample (benign)
        assert samples[1].input_text == "GET /api/users HTTP/1.1"
        assert samples[1].label == "BENIGN"
        assert samples[1].attack_type is None
        assert samples[1].metadata["source"] == "huggingface"  # This is set by the loader

        # Check third sample (attack with severity)
        assert samples[2].input_text == "<script>alert('xss')</script>"
        assert samples[2].label == "ATTACK"
        assert samples[2].attack_type == "xss"
        assert samples[2].metadata["severity"] == "high"

    @pytest.mark.asyncio
    async def test_map_huggingface_fields_with_max_samples(self, loader, sample_hf_dataset):
        """Test field mapping with max samples limit."""
        field_mapping = FieldMapping(input_text_field="text", label_field="label")

        samples = await loader._map_huggingface_fields(
            sample_hf_dataset, field_mapping, {"max_samples": 2}
        )

        assert len(samples) == 2

    @pytest.mark.asyncio
    async def test_map_huggingface_fields_invalid_sample(self, loader):
        """Test field mapping with invalid sample data."""
        # Create dataset with invalid sample (missing required field)
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(
            return_value=iter(
                [
                    {"text": "valid sample", "label": "ATTACK"},
                    {"label": "ATTACK"},  # Missing text field
                ]
            )
        )

        field_mapping = FieldMapping(input_text_field="text", label_field="label")

        # Should continue processing despite invalid sample
        samples = await loader._map_huggingface_fields(mock_dataset, field_mapping, {})

        # Only valid sample should be included
        assert len(samples) == 1
        assert samples[0].input_text == "valid sample"

    @pytest.mark.asyncio
    async def test_map_huggingface_fields_streaming_dataset(self, loader, sample_streaming_dataset):
        """Test field mapping with streaming dataset."""
        field_mapping = FieldMapping(
            input_text_field="text", label_field="label", attack_type_field="attack_type"
        )

        samples = await loader._map_huggingface_fields(sample_streaming_dataset, field_mapping, {})

        assert len(samples) == 2
        assert samples[0].label == "ATTACK"  # Normalized from 1
        assert samples[1].label == "BENIGN"  # Normalized from 0

    def test_create_dataset_info(self, loader):
        """Test creating dataset info from samples and metadata."""
        from benchmark.data.models import DatasetSample

        samples = [
            DatasetSample(
                input_text="attack sample 1", label="ATTACK", attack_type="sql_injection"
            ),
            DatasetSample(input_text="benign sample 1", label="BENIGN"),
            DatasetSample(input_text="attack sample 2", label="ATTACK", attack_type="xss"),
        ]

        hf_info = {
            "description": "Test cybersecurity dataset",
            "splits": {"train": {"num_examples": 1000}},
        }

        info = loader._create_dataset_info(
            dataset_path="test/cybersec-dataset",
            dataset_name="default",
            samples=samples,
            name="Custom Dataset Name",
            description="Custom description",
            hf_info=hf_info,
            split="train",
        )

        assert info.name == "Custom Dataset Name"
        assert info.source == "huggingface:test/cybersec-dataset/default#train"
        assert info.total_samples == 3
        assert info.attack_samples == 2
        assert info.benign_samples == 1
        assert set(info.attack_types) == {"sql_injection", "xss"}
        assert info.description == "Custom description"
        assert info.format == "huggingface"
        assert info.metadata["dataset_path"] == "test/cybersec-dataset"
        assert info.metadata["dataset_name"] == "default"
        assert info.metadata["split"] == "train"
        assert info.metadata["loader"] == "HuggingFaceDataLoader"
        assert info.metadata["huggingface_info"] == hf_info

    def test_create_dataset_info_minimal(self, loader):
        """Test creating dataset info with minimal parameters."""
        samples = []

        info = loader._create_dataset_info(
            dataset_path="simple/dataset", dataset_name=None, samples=samples
        )

        assert info.name == "simple_dataset_train"
        assert info.source == "huggingface:simple/dataset#train"
        assert info.total_samples == 0
        assert info.attack_samples == 0
        assert info.benign_samples == 0
        assert info.attack_types == []
        assert info.metadata["dataset_name"] is None

    @pytest.mark.asyncio
    async def test_load_success_full_mode(self, loader, mock_datasets, sample_hf_dataset):
        """Test successful dataset loading in full mode."""
        # Mock dataset info
        mock_info = Mock()
        mock_info.__dict__ = {
            "description": "Test dataset",
            "splits": {"train": {"num_examples": 100}},  # Below streaming threshold
        }
        mock_datasets["get_dataset_infos"].return_value = {"default": mock_info}
        mock_datasets["load_dataset"].return_value = sample_hf_dataset

        config = {"dataset_path": "test/dataset", "split": "train"}

        dataset = await loader.load(config)

        assert dataset is not None
        assert len(dataset.samples) == 3
        assert dataset.info.source == "huggingface:test/dataset#train"
        assert dataset.info.format == "huggingface"

    @pytest.mark.asyncio
    async def test_load_success_streaming_mode_auto(
        self, loader, mock_datasets, sample_streaming_dataset
    ):
        """Test automatic streaming mode for large datasets."""
        # Mock large dataset info to trigger streaming
        mock_info = Mock()
        mock_info.__dict__ = {
            "description": "Large dataset",
            "splits": {"train": {"num_examples": 200000}},  # Above streaming threshold
        }
        mock_datasets["get_dataset_infos"].return_value = {"default": mock_info}
        mock_datasets["load_dataset"].return_value = sample_streaming_dataset

        config = {"dataset_path": "test/large-dataset", "split": "train"}

        dataset = await loader.load(config)

        # Should have called with streaming=True
        mock_datasets["load_dataset"].assert_called_once_with(
            path="test/large-dataset", split="train", streaming=True, cache_dir=None
        )

        assert len(dataset.samples) == 2

    @pytest.mark.asyncio
    async def test_load_success_streaming_mode_forced(
        self, loader, mock_datasets, sample_streaming_dataset
    ):
        """Test forced streaming mode via config."""
        # Mock dataset info to return something that passes validation
        mock_info = Mock()
        mock_info.__dict__ = {"description": "Test dataset"}
        mock_datasets["get_dataset_infos"].return_value = {"default": mock_info}
        mock_datasets["load_dataset"].return_value = sample_streaming_dataset

        config = {"dataset_path": "test/dataset", "split": "train", "streaming": True}

        await loader.load(config)

        # Should have called with streaming=True
        mock_datasets["load_dataset"].assert_called_once_with(
            path="test/dataset", split="train", streaming=True, cache_dir=None
        )

    @pytest.mark.asyncio
    async def test_load_missing_dataset_path(self, loader):
        """Test loading with missing dataset_path."""
        config = {}

        with pytest.raises(ValueError, match="Missing required configuration keys"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_invalid_source(self, loader, mock_datasets):
        """Test loading with invalid source."""
        # Set up mocks to simulate dataset not found in validation
        mock_datasets["get_dataset_infos"].side_effect = Exception("Dataset not found")
        mock_datasets["get_dataset_config_names"].side_effect = Exception("Dataset not found")

        # Mock the validate_source method to return False
        with patch.object(loader, "validate_source", return_value=False):
            config = {"dataset_path": "invalid/dataset"}

            with pytest.raises(DataLoadError, match="HuggingFace dataset not accessible"):
                await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_dataset_loading_error(self, loader, mock_datasets):
        """Test error during dataset loading."""
        mock_info = Mock()
        mock_info.__dict__ = {"description": "Test dataset"}
        mock_datasets["get_dataset_infos"].return_value = {"default": mock_info}
        mock_datasets["load_dataset"].side_effect = Exception("Network error")

        config = {"dataset_path": "test/dataset"}

        with pytest.raises(DataLoadError, match="Failed to load HuggingFace dataset"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_with_all_config_options(self, loader, mock_datasets, sample_hf_dataset):
        """Test loading with all configuration options."""
        # Mock dataset info to return valid config
        mock_info = Mock()
        mock_info.__dict__ = {"description": "Config dataset"}
        mock_datasets["get_dataset_infos"].return_value = {"custom_config": mock_info}
        mock_datasets["load_dataset"].return_value = sample_hf_dataset

        config = {
            "dataset_path": "test/dataset",
            "dataset_name": "custom_config",
            "split": "validation",
            "streaming": False,
            "max_samples": 2,
            "name": "Custom Dataset",
            "description": "Custom description",
            "field_mapping": {
                "input_text_field": "text",
                "label_field": "label",
                "attack_type_field": "attack_type",
                "metadata_fields": ["source"],
                "required_fields": ["text", "label"],
            },
        }

        dataset = await loader.load(config)

        assert dataset.info.name == "Custom Dataset"
        assert dataset.info.description == "Custom description"
        assert len(dataset.samples) == 2  # Limited by max_samples
        assert dataset.info.source == "huggingface:test/dataset/custom_config#validation"

        mock_datasets["load_dataset"].assert_called_once_with(
            path="test/dataset", name="custom_config", split="validation", cache_dir=None
        )

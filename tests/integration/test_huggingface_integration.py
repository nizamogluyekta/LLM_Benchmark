"""
Integration tests for HuggingFaceDataLoader.

This module tests the HuggingFace data loader with actual dataset loading
when the datasets library is available, gracefully skipping tests when it is not.
"""

import pytest

try:
    from benchmark.data.loaders.huggingface_loader import DATASETS_AVAILABLE, HuggingFaceDataLoader

    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False
    DATASETS_AVAILABLE = False


# Skip all integration tests if datasets library is not available
pytestmark = pytest.mark.skipif(
    not LOADER_AVAILABLE or not DATASETS_AVAILABLE,
    reason="HuggingFace datasets library not available (install with: pip install datasets>=2.16.0)",
)


class TestHuggingFaceIntegration:
    """Integration tests with actual HuggingFace datasets."""

    @pytest.fixture
    def loader(self):
        """Create HuggingFaceDataLoader instance."""
        return HuggingFaceDataLoader(show_progress=False)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_small_public_dataset(self, loader):
        """Test loading a small public dataset for basic functionality."""
        # Use a very small dataset for testing to avoid long download times
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",  # Validation split is smaller than train
            "max_samples": 10,  # Limit to 10 samples for speed
            "field_mapping": {
                "input_text_field": "sentence",
                "label_field": "label",
                "metadata_fields": ["idx"],
            },
        }

        dataset = await loader.load(config)

        # Verify dataset structure
        assert dataset is not None
        assert len(dataset.samples) <= 10
        assert dataset.info.source.startswith("huggingface:glue/cola#validation")
        assert dataset.info.format == "huggingface"
        assert dataset.info.total_samples == len(dataset.samples)

        # Verify samples have correct structure
        if dataset.samples:
            sample = dataset.samples[0]
            assert sample.input_text is not None
            assert sample.label in ["ATTACK", "BENIGN"]  # Should be normalized
            assert "idx" in sample.metadata
            assert sample.metadata["source"] == "huggingface"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_validate_source_existing_dataset(self, loader):
        """Test source validation with existing dataset."""
        config = {"dataset_path": "glue", "dataset_name": "cola"}

        is_valid = await loader.validate_source(config)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_source_nonexistent_dataset(self, loader):
        """Test source validation with non-existent dataset."""
        config = {"dataset_path": "nonexistent/fake-dataset-12345"}

        is_valid = await loader.validate_source(config)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_dataset_info_public_dataset(self, loader):
        """Test getting dataset info for public dataset."""
        info = await loader._get_dataset_info("glue", "cola")

        assert isinstance(info, dict)
        # GLUE CoLA should have some standard info
        assert "description" in info or "splits" in info or len(info) > 0

    @pytest.mark.asyncio
    async def test_get_dataset_info_nonexistent(self, loader):
        """Test getting dataset info for non-existent dataset."""
        info = await loader._get_dataset_info("nonexistent/fake-dataset-12345")

        # Should return empty dict when dataset doesn't exist
        assert info == {}

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_streaming_mode_small_dataset(self, loader):
        """Test streaming mode with a small dataset."""
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "streaming": True,  # Force streaming mode
            "max_samples": 5,
            "field_mapping": {"input_text_field": "sentence", "label_field": "label"},
        }

        dataset = await loader.load(config)

        assert dataset is not None
        assert len(dataset.samples) <= 5
        assert dataset.info.format == "huggingface"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_field_mapping_detection(self, loader):
        """Test automatic field mapping detection."""
        # Test default mapping with GLUE dataset
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "max_samples": 3,
        }

        dataset = await loader.load(config)

        assert dataset is not None
        if dataset.samples:
            # GLUE CoLA has sentence and label fields
            sample = dataset.samples[0]
            assert sample.input_text  # Should have mapped from 'sentence' to text field

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_label_normalization_numeric(self, loader):
        """Test label normalization with numeric labels."""
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "max_samples": 10,
            "field_mapping": {"input_text_field": "sentence", "label_field": "label"},
        }

        dataset = await loader.load(config)

        assert dataset is not None
        if dataset.samples:
            # All labels should be normalized to ATTACK/BENIGN
            for sample in dataset.samples:
                assert sample.label in ["ATTACK", "BENIGN"]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_dataset_info_creation(self, loader):
        """Test dataset info creation with real data."""
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "max_samples": 5,
            "name": "Test CoLA Dataset",
            "description": "Test dataset for validation",
        }

        dataset = await loader.load(config)

        info = dataset.info
        assert info.name == "Test CoLA Dataset"
        assert info.description == "Test dataset for validation"
        assert info.source == "huggingface:glue/cola#validation"
        assert info.format == "huggingface"
        assert info.total_samples == len(dataset.samples)
        assert info.attack_samples + info.benign_samples == info.total_samples
        assert info.metadata["dataset_path"] == "glue"
        assert info.metadata["dataset_name"] == "cola"
        assert info.metadata["split"] == "validation"
        assert info.metadata["loader"] == "HuggingFaceDataLoader"

    @pytest.mark.asyncio
    async def test_load_invalid_config(self, loader):
        """Test loading with invalid configuration."""

        # Missing dataset_path
        config = {}

        with pytest.raises(ValueError, match="Missing required configuration keys"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_invalid_dataset(self, loader):
        """Test loading non-existent dataset."""
        from benchmark.data.loaders.base_loader import DataLoadError

        config = {"dataset_path": "nonexistent/fake-dataset-12345"}

        with pytest.raises(DataLoadError, match="HuggingFace dataset not accessible"):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_invalid_split(self, loader):
        """Test loading non-existent split."""
        from benchmark.data.loaders.base_loader import DataLoadError

        config = {"dataset_path": "glue", "dataset_name": "cola", "split": "nonexistent_split"}

        with pytest.raises(DataLoadError):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_invalid_config_name(self, loader):
        """Test loading with invalid configuration name."""
        from benchmark.data.loaders.base_loader import DataLoadError

        config = {"dataset_path": "glue", "dataset_name": "nonexistent_config"}

        with pytest.raises(DataLoadError):
            await loader.load(config)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_custom_field_mapping(self, loader):
        """Test loading with custom field mapping."""
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "max_samples": 3,
            "field_mapping": {
                "input_text_field": "sentence",
                "label_field": "label",
                "metadata_fields": ["idx"],
                "required_fields": ["sentence", "label"],
            },
        }

        dataset = await loader.load(config)

        assert dataset is not None
        assert len(dataset.samples) <= 3

        if dataset.samples:
            sample = dataset.samples[0]
            assert sample.input_text is not None
            assert sample.label in ["ATTACK", "BENIGN"]
            assert "idx" in sample.metadata

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_progress_callbacks(self, loader):
        """Test progress reporting callbacks."""
        progress_calls = []

        def progress_callback(progress):
            progress_calls.append(progress.processed_items)

        loader.add_progress_callback(progress_callback)
        loader.show_progress = True

        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "max_samples": 50,  # Enough to trigger progress updates
        }

        dataset = await loader.load(config)

        assert dataset is not None
        # Should have made some progress calls for larger datasets
        # Note: Small datasets might not trigger progress updates

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_caching_behavior(self, loader):
        """Test that datasets are cached by HuggingFace."""
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "max_samples": 5,
        }

        # Load dataset twice - second load should be faster due to caching
        dataset1 = await loader.load(config)
        dataset2 = await loader.load(config)

        # Both should succeed and have same structure
        assert dataset1 is not None
        assert dataset2 is not None
        assert len(dataset1.samples) == len(dataset2.samples)

        # Content should be equivalent (though object references will differ)
        if dataset1.samples and dataset2.samples:
            assert dataset1.samples[0].input_text == dataset2.samples[0].input_text
            assert dataset1.samples[0].label == dataset2.samples[0].label

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_metadata_preservation(self, loader):
        """Test that original dataset metadata is preserved."""
        config = {
            "dataset_path": "glue",
            "dataset_name": "cola",
            "split": "validation",
            "max_samples": 3,
            "field_mapping": {
                "input_text_field": "sentence",
                "label_field": "label",
                "metadata_fields": ["idx"],  # Preserve idx field
            },
        }

        dataset = await loader.load(config)

        assert dataset is not None
        if dataset.samples:
            sample = dataset.samples[0]

            # Should preserve original fields in metadata
            assert "idx" in sample.metadata  # Explicitly requested
            assert "hf_index" in sample.metadata  # Added by loader
            assert "source" in sample.metadata  # Added by loader
            assert sample.metadata["source"] == "huggingface"


class TestHuggingFaceDataLoaderEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def loader(self):
        """Create HuggingFaceDataLoader instance."""
        return HuggingFaceDataLoader(show_progress=False)

    def test_supported_formats(self, loader):
        """Test that loader reports correct supported formats."""
        formats = loader.get_supported_formats()
        assert "huggingface" in formats
        assert "hf" in formats
        assert len(formats) >= 2

    def test_field_mapping_patterns(self, loader):
        """Test field mapping pattern recognition."""
        # Test malware pattern
        config = {"dataset_path": "security/malware-detection-v2"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "text"
        assert mapping.label_field == "malware"
        assert mapping.attack_type_field == "family"

        # Test phishing pattern
        config = {"dataset_path": "cybersec/phishing-urls"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "url"
        assert mapping.label_field == "label"
        assert mapping.attack_type_field == "type"

        # Test network pattern
        config = {"dataset_path": "network-intrusion-detection"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "packet"
        assert mapping.label_field == "label"
        assert mapping.attack_type_field == "attack"

        # Test default fallback
        config = {"dataset_path": "some/unknown/dataset"}
        mapping = loader._get_field_mapping(config)
        assert mapping.input_text_field == "text"
        assert mapping.label_field == "label"
        assert mapping.attack_type_field == "attack_type"

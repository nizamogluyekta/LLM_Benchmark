"""
Unit tests for the Data Service.

This module tests the data service functionality including plugin registration,
data loading, caching behavior, and error handling.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.core.base import ServiceStatus
from benchmark.core.config import DatasetConfig
from benchmark.core.exceptions import DataLoadingError
from benchmark.interfaces.data_interfaces import (
    DataFormat,
    DataLoader,
    DataSample,
    Dataset,
    DatasetInfo,
    DataSource,
    DataSplitter,
)
from benchmark.services.data_service import (
    DataService,
    LocalDataLoader,
    MemoryDataCache,
    RandomDataSplitter,
)


class TestMemoryDataCache:
    """Test the memory data cache implementation."""

    @pytest.fixture
    def cache(self):
        """Create a memory data cache for testing."""
        return MemoryDataCache(max_size=5, max_memory_mb=1, ttl_default=60)

    @pytest.mark.asyncio
    async def test_cache_basic_operations(self, cache):
        """Test basic cache operations."""
        # Test set and get
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

        # Test non-existent key
        result = await cache.get("nonexistent")
        assert result is None

        # Test delete
        deleted = await cache.delete("key1")
        assert deleted is True

        # Test get after delete
        result = await cache.get("key1")
        assert result is None

        # Test delete non-existent
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Verify all items are cached
        for i in range(5):
            result = await cache.get(f"key{i}")
            assert result == f"value{i}"

        # Add one more item to trigger eviction
        await cache.set("key5", "value5")

        # First item should be evicted (LRU)
        result = await cache.get("key0")
        assert result is None

        # Other items should still be there
        for i in range(1, 6):
            result = await cache.get(f"key{i}")
            assert result == f"value{i}"

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, cache):
        """Test TTL expiration."""
        # Set item with very short TTL for fast testing
        await cache.set("key1", "value1", ttl=0.05)

        # Should be available immediately
        result = await cache.get("key1")
        assert result == "value1"

        # Wait for expiration
        await asyncio.sleep(0.1)

        # Should be expired
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """Test cache clearing."""
        # Add some items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Verify items exist
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"

        # Clear cache
        await cache.clear()

        # Items should be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_cache_info(self, cache):
        """Test cache information retrieval."""
        # Add some items
        await cache.set("key1", "value1")
        await cache.set("key2", {"data": "complex"})

        info = await cache.get_cache_info()

        assert info["cache_size"] == 2
        assert info["max_size"] == 5
        assert info["memory_usage_bytes"] > 0
        assert info["memory_usage_mb"] > 0
        assert info["memory_utilization"] > 0
        assert "key1" in info["entries"]
        assert "key2" in info["entries"]


class TestLocalDataLoader:
    """Test the local data loader implementation."""

    @pytest.fixture
    def loader(self):
        """Create a local data loader for testing."""
        return LocalDataLoader()

    @pytest.fixture
    def sample_jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Sample attack log", "label": "ATTACK"}\n')
            f.write('{"text": "Normal traffic", "label": "BENIGN"}\n')
            f.write('{"text": "Malware detected", "label": "ATTACK"}\n')
            temp_path = f.name

        yield Path(temp_path)
        Path(temp_path).unlink()  # Clean up

    @pytest.fixture
    def sample_json_file(self):
        """Create a temporary JSON file for testing."""
        data = [
            {"text": "Attack pattern", "label": "ATTACK"},
            {"text": "Normal behavior", "label": "BENIGN"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        yield Path(temp_path)
        Path(temp_path).unlink()  # Clean up

    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("Attack detected,ATTACK\n")
            f.write("Normal activity,BENIGN\n")
            temp_path = f.name

        yield Path(temp_path)
        Path(temp_path).unlink()  # Clean up

    def test_supported_formats(self, loader):
        """Test that loader reports correct supported formats."""
        formats = loader.get_supported_formats()
        assert DataFormat.JSON in formats
        assert DataFormat.JSONL in formats
        assert DataFormat.CSV in formats
        assert DataFormat.TSV in formats

    def test_source_type(self, loader):
        """Test that loader reports correct source type."""
        assert loader.get_source_type() == DataSource.LOCAL

    @pytest.mark.asyncio
    async def test_validate_source_exists(self, loader, sample_jsonl_file):
        """Test source validation for existing file."""
        config = DatasetConfig(name="test", source="local", path=str(sample_jsonl_file))

        is_valid = await loader.validate_source(config)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_source_missing(self, loader):
        """Test source validation for missing file."""
        config = DatasetConfig(name="test", source="local", path="/nonexistent/file.jsonl")

        is_valid = await loader.validate_source(config)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_load_jsonl_file(self, loader, sample_jsonl_file):
        """Test loading JSONL file."""
        config = DatasetConfig(name="test_jsonl", source="local", path=str(sample_jsonl_file))

        dataset = await loader.load(config)

        assert dataset.dataset_id == "test_jsonl"
        assert dataset.info.name == "test_jsonl"
        assert dataset.info.source == DataSource.LOCAL
        assert dataset.info.format == DataFormat.JSONL
        assert len(dataset.samples) == 3

        # Check first sample
        sample = dataset.samples[0]
        assert sample.data["text"] == "Sample attack log"
        assert sample.label == "ATTACK"
        assert "line_number" in sample.metadata

    @pytest.mark.asyncio
    async def test_load_json_file(self, loader, sample_json_file):
        """Test loading JSON file."""
        config = DatasetConfig(name="test_json", source="local", path=str(sample_json_file))

        dataset = await loader.load(config)

        assert dataset.dataset_id == "test_json"
        assert dataset.info.format == DataFormat.JSON
        assert len(dataset.samples) == 2

        # Check samples
        assert dataset.samples[0].data["text"] == "Attack pattern"
        assert dataset.samples[0].label == "ATTACK"
        assert dataset.samples[1].data["text"] == "Normal behavior"
        assert dataset.samples[1].label == "BENIGN"

    @pytest.mark.asyncio
    async def test_load_csv_file(self, loader, sample_csv_file):
        """Test loading CSV file."""
        config = DatasetConfig(name="test_csv", source="local", path=str(sample_csv_file))

        dataset = await loader.load(config)

        assert dataset.dataset_id == "test_csv"
        assert dataset.info.format == DataFormat.CSV
        assert len(dataset.samples) == 2

        # Check samples
        assert dataset.samples[0].data["text"] == "Attack detected"
        assert dataset.samples[0].label == "ATTACK"

    @pytest.mark.asyncio
    async def test_load_with_max_samples(self, loader, sample_jsonl_file):
        """Test loading with max_samples limit."""
        config = DatasetConfig(
            name="test_limited", source="local", path=str(sample_jsonl_file), max_samples=2
        )

        dataset = await loader.load(config)

        assert len(dataset.samples) == 2  # Limited from 3 to 2

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, loader):
        """Test loading non-existent file."""
        config = DatasetConfig(name="test_missing", source="local", path="/nonexistent/file.jsonl")

        with pytest.raises(DataLoadingError):
            await loader.load(config)

    @pytest.mark.asyncio
    async def test_load_unsupported_format(self, loader):
        """Test loading unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"some data")
            temp_path = f.name

        try:
            config = DatasetConfig(name="test_unsupported", source="local", path=temp_path)

            with pytest.raises(DataLoadingError):
                await loader.load(config)
        finally:
            Path(temp_path).unlink()


class TestRandomDataSplitter:
    """Test the random data splitter implementation."""

    @pytest.fixture
    def splitter(self):
        """Create a random data splitter for testing."""
        return RandomDataSplitter(seed=42)

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for splitting."""
        samples = []
        for i in range(100):
            sample = DataSample(
                sample_id=f"sample_{i}",
                data={"text": f"Sample {i}"},
                label="ATTACK" if i % 2 == 0 else "BENIGN",
            )
            samples.append(sample)

        info = DatasetInfo(dataset_id="test_dataset", name="Test Dataset", num_samples=100)

        return Dataset(dataset_id="test_dataset", info=info, samples=samples)

    def test_split_strategy(self, splitter):
        """Test that splitter reports correct strategy."""
        assert splitter.get_split_strategy() == "random"

    @pytest.mark.asyncio
    async def test_create_splits_default(self, splitter, sample_dataset):
        """Test creating splits with default ratios."""
        config = DatasetConfig(name="test", source="local", path="/dummy")

        splits = await splitter.create_splits(sample_dataset, config)

        assert splits.dataset_id == "test_dataset"
        assert len(splits.train_samples) == 70  # 100 - 20 (test) - 10 (val)
        assert len(splits.validation_samples) == 10  # 10% of 100
        assert len(splits.test_samples) == 20  # 20% of 100

        # Check that all samples are accounted for
        all_samples = set(splits.train_samples + splits.validation_samples + splits.test_samples)
        expected_samples = {f"sample_{i}" for i in range(100)}
        assert all_samples == expected_samples

    @pytest.mark.asyncio
    async def test_create_splits_custom_ratios(self, splitter, sample_dataset):
        """Test creating splits with custom ratios."""
        config = DatasetConfig(
            name="test", source="local", path="/dummy", test_split=0.3, validation_split=0.2
        )

        splits = await splitter.create_splits(sample_dataset, config)

        assert len(splits.train_samples) == 50  # 100 - 30 (test) - 20 (val)
        assert len(splits.validation_samples) == 20  # 20% of 100
        assert len(splits.test_samples) == 30  # 30% of 100

    @pytest.mark.asyncio
    async def test_splits_reproducible(self, sample_dataset):
        """Test that splits are reproducible with same seed."""
        config = DatasetConfig(name="test", source="local", path="/dummy")

        splitter1 = RandomDataSplitter(seed=123)
        splitter2 = RandomDataSplitter(seed=123)

        splits1 = await splitter1.create_splits(sample_dataset, config)
        splits2 = await splitter2.create_splits(sample_dataset, config)

        assert splits1.train_samples == splits2.train_samples
        assert splits1.validation_samples == splits2.validation_samples
        assert splits1.test_samples == splits2.test_samples


class TestDataService:
    """Test the Data Service implementation."""

    @pytest.fixture
    def service(self):
        """Create a data service for testing."""
        return DataService(cache_max_size=10, cache_max_memory_mb=64, cache_ttl=300)

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a sample dataset configuration."""
        # Create a test data file
        data_file = tmp_path / "test_data.jsonl"
        with open(data_file, "w") as f:
            f.write('{"text": "Attack pattern", "label": "ATTACK"}\n')
            f.write('{"text": "Normal traffic", "label": "BENIGN"}\n')

        return DatasetConfig(
            name="test_dataset",
            source="local",
            path=str(data_file),
            test_split=0.3,
            validation_split=0.2,
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initialization."""
        response = await service.initialize()

        assert response.success is True
        assert service.status == ServiceStatus.HEALTHY

        # Check that default plugins are registered
        assert DataSource.LOCAL in service.loaders
        assert "random" in service.splitters

    @pytest.mark.asyncio
    async def test_service_health_check(self, service):
        """Test service health check."""
        await service.initialize()

        health = await service.health_check()

        assert health.status == ServiceStatus.HEALTHY.value
        assert "cache_status" in health.checks
        assert "registered_loaders" in health.checks

    @pytest.mark.asyncio
    async def test_service_shutdown(self, service):
        """Test service shutdown."""
        await service.initialize()

        response = await service.shutdown()

        assert response.success is True
        assert service.status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_register_custom_loader(self, service):
        """Test registering a custom data loader."""
        await service.initialize()

        # Create mock loader
        mock_loader = MagicMock(spec=DataLoader)

        # Register loader
        await service.register_loader(DataSource.KAGGLE, mock_loader)

        # Verify registration
        assert DataSource.KAGGLE in service.loaders
        assert service.loaders[DataSource.KAGGLE] == mock_loader

    @pytest.mark.asyncio
    async def test_register_custom_splitter(self, service):
        """Test registering a custom data splitter."""
        await service.initialize()

        # Create mock splitter
        mock_splitter = MagicMock(spec=DataSplitter)

        # Register splitter
        await service.register_splitter("stratified", mock_splitter)

        # Verify registration
        assert "stratified" in service.splitters
        assert service.splitters["stratified"] == mock_splitter

    @pytest.mark.asyncio
    async def test_load_dataset_success(self, service, sample_config):
        """Test successful dataset loading."""
        await service.initialize()

        dataset = await service.load_dataset(sample_config)

        assert dataset.dataset_id == "test_dataset"
        assert dataset.info.name == "test_dataset"
        assert len(dataset.samples) == 2
        assert dataset.samples[0].data["text"] == "Attack pattern"
        assert dataset.samples[0].label == "ATTACK"

    @pytest.mark.asyncio
    async def test_load_dataset_caching(self, service, sample_config):
        """Test that datasets are cached correctly."""
        await service.initialize()

        # Load dataset first time
        dataset1 = await service.load_dataset(sample_config)

        # Load dataset second time (should use cache)
        dataset2 = await service.load_dataset(sample_config)

        # Should be equal (content-wise, not necessarily same object reference)
        assert dataset1 == dataset2
        assert dataset1.info.name == dataset2.info.name
        assert len(dataset1.samples) == len(dataset2.samples)

    @pytest.mark.asyncio
    async def test_load_dataset_unsupported_source(self, service):
        """Test loading dataset with unsupported source."""
        await service.initialize()

        # Create a config with a supported source but unregistered loader
        config = DatasetConfig(name="test", source="kaggle", path="/dummy")

        with pytest.raises(DataLoadingError):
            await service.load_dataset(config)

    @pytest.mark.asyncio
    async def test_get_dataset_info(self, service, sample_config):
        """Test getting dataset information."""
        await service.initialize()

        # Load dataset first
        await service.load_dataset(sample_config)

        # Get info
        info = await service.get_dataset_info("test_dataset")

        assert info.dataset_id == "test_dataset"
        assert info.name == "test_dataset"
        assert info.num_samples == 2

    @pytest.mark.asyncio
    async def test_get_dataset_info_not_found(self, service):
        """Test getting info for non-existent dataset."""
        await service.initialize()

        with pytest.raises(DataLoadingError):
            await service.get_dataset_info("nonexistent")

    @pytest.mark.asyncio
    async def test_create_data_splits(self, service, sample_config):
        """Test creating data splits."""
        await service.initialize()

        # Load dataset first
        await service.load_dataset(sample_config)

        # Create splits
        splits = await service.create_data_splits("test_dataset", sample_config)

        assert splits.dataset_id == "test_dataset"
        # With only 2 samples and integer rounding, we expect:
        # test: int(2 * 0.3) = 0
        # val: int(2 * 0.2) = 0
        # train: 2 - 0 - 0 = 2
        assert splits.train_size == 2
        assert splits.test_size == 0
        assert splits.validation_size == 0

    @pytest.mark.asyncio
    async def test_create_data_splits_caching(self, service, sample_config):
        """Test that data splits are cached."""
        await service.initialize()

        # Load dataset
        await service.load_dataset(sample_config)

        # Create splits first time
        splits1 = await service.create_data_splits("test_dataset", sample_config)

        # Create splits second time (should use cache)
        splits2 = await service.create_data_splits("test_dataset", sample_config)

        # Should be equal (content-wise, not necessarily same object reference)
        assert splits1 == splits2
        assert splits1.dataset_id == splits2.dataset_id
        if splits1 and splits2:
            assert splits1.train_size == splits2.train_size
            assert splits1.test_size == splits2.test_size
            assert splits1.validation_size == splits2.validation_size

    @pytest.mark.asyncio
    async def test_get_batch(self, service, sample_config):
        """Test getting data batches."""
        await service.initialize()

        # Load dataset
        await service.load_dataset(sample_config)

        # Get batch
        batch = await service.get_batch("test_dataset", batch_size=1, offset=0)

        assert batch.batch_id == "test_dataset_batch_0_1"
        assert len(batch.samples) == 1
        assert batch.batch_info["dataset_id"] == "test_dataset"
        assert batch.batch_info["batch_size"] == 1
        assert batch.batch_info["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_batch_beyond_dataset(self, service, sample_config):
        """Test getting batch beyond dataset size."""
        await service.initialize()

        # Load dataset
        await service.load_dataset(sample_config)

        # Get batch beyond dataset size
        batch = await service.get_batch("test_dataset", batch_size=5, offset=1)

        # Should return only available samples
        assert len(batch.samples) == 1  # Only 1 sample left from offset 1
        assert batch.batch_info["actual_size"] == 1

    @pytest.mark.asyncio
    async def test_get_batch_dataset_not_loaded(self, service):
        """Test getting batch from non-loaded dataset."""
        await service.initialize()

        with pytest.raises(DataLoadingError):
            await service.get_batch("nonexistent", batch_size=1, offset=0)

    @pytest.mark.asyncio
    async def test_cache_stats(self, service, sample_config):
        """Test getting cache statistics."""
        await service.initialize()

        # Load dataset to populate cache
        await service.load_dataset(sample_config)

        stats = await service.get_cache_stats()

        assert "cache_size" in stats
        assert "memory_usage_mb" in stats
        assert stats["cache_size"] >= 1  # At least the dataset we loaded

    @pytest.mark.asyncio
    async def test_clear_cache(self, service, sample_config):
        """Test clearing cache."""
        await service.initialize()

        # Load dataset to populate cache
        await service.load_dataset(sample_config)

        # Verify cache has items
        stats = await service.get_cache_stats()
        assert stats["cache_size"] > 0

        # Clear cache
        await service.clear_cache()

        # Verify cache is empty
        stats = await service.get_cache_stats()
        assert stats["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_initialization(self):
        """Test error handling during initialization."""
        with patch.object(
            DataService, "_register_default_plugins", side_effect=Exception("Test error")
        ):
            service = DataService()
            response = await service.initialize()

            assert response.success is False
            assert service.status == ServiceStatus.ERROR

    @pytest.mark.asyncio
    async def test_concurrent_loading(self, service, sample_config):
        """Test concurrent dataset loading."""
        await service.initialize()

        # Create multiple concurrent load tasks
        tasks = [service.load_dataset(sample_config) for _ in range(5)]

        datasets = await asyncio.gather(*tasks)

        # All should return equal datasets (cached or equivalent)
        for dataset in datasets[1:]:
            assert dataset == datasets[0]
            assert dataset.info.name == datasets[0].info.name
            assert len(dataset.samples) == len(datasets[0].samples)

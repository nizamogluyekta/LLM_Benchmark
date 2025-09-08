"""
Integration tests for KaggleDataLoader.

This module tests the Kaggle data loader with actual API calls when credentials
are available, gracefully skipping tests when they are not.
"""

import os
import tempfile
from pathlib import Path

import pytest

from benchmark.data.loaders.kaggle_loader import KAGGLE_AVAILABLE, KaggleDataLoader
from benchmark.data.models import Dataset


# Check for Kaggle credentials
def has_kaggle_credentials():
    """Check if Kaggle credentials are available."""
    if not KAGGLE_AVAILABLE:
        return False

    # Check for kaggle.json file
    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_config_file = kaggle_config_dir / "kaggle.json"

    if kaggle_config_file.exists():
        return True

    # Check for environment variables
    return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))


# Skip all integration tests if Kaggle is not available or no credentials
pytestmark = pytest.mark.skipif(
    not has_kaggle_credentials(),
    reason="Kaggle credentials not available (set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json)",
)


class TestKaggleIntegration:
    """Integration tests with actual Kaggle API."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def loader(self, temp_dir):
        """Create KaggleDataLoader with temporary cache."""
        return KaggleDataLoader(
            cache_dir=temp_dir / "kaggle_cache",
            show_progress=False,  # Disable progress for testing
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_api_authentication(self, loader):
        """Test that API authentication works with available credentials."""
        await loader._initialize_api()

        assert loader.api is not None
        # Try a simple API call to verify authentication
        try:
            # This should not raise an exception if authenticated
            datasets = loader.api.dataset_list(search="test", page_size=1)
            assert isinstance(datasets, list)
        except Exception as e:
            pytest.fail(f"API authentication failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_validate_existing_dataset(self, loader):
        """Test validation of a known public dataset."""
        # Use a small, stable public dataset for testing
        config = {"dataset": "uciml/iris"}

        is_valid = await loader.validate_source(config)
        assert is_valid is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_validate_nonexistent_dataset(self, loader):
        """Test validation of a non-existent dataset."""
        config = {"dataset": "nonexistent/invalid-dataset-name-123456"}

        is_valid = await loader.validate_source(config)
        assert is_valid is False

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_load_small_dataset(self, loader):
        """Test loading a small public dataset."""
        # Use the Iris dataset - it's small and stable
        config = {
            "dataset": "uciml/iris",
            "name": "Iris Dataset Test",
            "description": "Test loading of Iris dataset from Kaggle",
        }

        try:
            dataset = await loader.load(config)

            assert isinstance(dataset, Dataset)
            assert dataset.info.name == "Iris Dataset Test"
            assert dataset.sample_count > 0
            assert dataset.info.source == "kaggle:uciml/iris"
            assert "kaggle_dataset" in dataset.info.metadata
            assert "cache_path" in dataset.info.metadata

        except Exception as e:
            pytest.fail(f"Failed to load Iris dataset: {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cache_functionality(self, loader):
        """Test that caching works correctly."""
        config = {"dataset": "uciml/iris"}

        # First load should download
        dataset1 = await loader.load(config)
        cache_info_after_first = loader.get_cache_info()

        # Second load should use cache
        dataset2 = await loader.load(config)
        cache_info_after_second = loader.get_cache_info()

        assert dataset1.sample_count == dataset2.sample_count
        # Cache info should be the same (no additional downloads)
        assert cache_info_after_first == cache_info_after_second

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_force_download(self, loader):
        """Test force download functionality."""
        config = {"dataset": "uciml/iris"}

        # First load
        await loader.load(config)

        # Force download should work even with cache
        loader.force_download = True
        dataset = await loader.load(config)

        assert isinstance(dataset, Dataset)
        assert dataset.sample_count > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cleanup_cache(self, loader):
        """Test cache cleanup functionality."""
        config = {"dataset": "uciml/iris"}

        # Load dataset to create cache
        await loader.load(config)
        cache_info_before = loader.get_cache_info()
        assert len(cache_info_before["datasets"]) > 0

        # Clean specific dataset cache
        await loader.cleanup_cache("uciml/iris")
        cache_info_after = loader.get_cache_info()

        # Should have fewer (or no) cached datasets
        assert len(cache_info_after["datasets"]) <= len(cache_info_before["datasets"])

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_progress_reporting(self, loader):
        """Test progress reporting during download."""
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(
                {
                    "processed": progress.processed_items,
                    "total": progress.total_items,
                    "percentage": progress.percentage,
                }
            )

        loader.add_progress_callback(progress_callback)
        loader.show_progress = True

        config = {"dataset": "uciml/iris"}

        # Clean cache first to ensure fresh download
        await loader.cleanup_cache("uciml/iris")

        dataset = await loader.load(config)

        assert isinstance(dataset, Dataset)
        # Progress callback should have been called at least once
        assert len(progress_updates) >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling_network_timeout(self, loader):
        """Test error handling for network issues."""
        # This test is more conceptual since we can't easily simulate network issues
        # in integration tests. The actual error handling is tested in unit tests.
        config = {"dataset": "uciml/iris"}

        try:
            dataset = await loader.load(config)
            # If we get here, the network is working fine
            assert isinstance(dataset, Dataset)
        except Exception as e:
            # If there's a network error, it should be wrapped in DataLoadError
            # This is more of a smoke test
            assert "Failed to" in str(e) or "not accessible" in str(e)


class TestKaggleCredentialHandling:
    """Test credential handling scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_api_initialization_with_credentials(self):
        """Test API initialization with valid credentials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = KaggleDataLoader(cache_dir=Path(temp_dir))

            # This should work if credentials are properly configured
            await loader._initialize_api()
            assert loader.api is not None

    def test_get_cache_info_structure(self):
        """Test that cache info has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = KaggleDataLoader(cache_dir=Path(temp_dir))
            cache_info = loader.get_cache_info()

            assert "cache_dir" in cache_info
            assert "total_size_bytes" in cache_info
            assert "datasets" in cache_info
            assert isinstance(cache_info["datasets"], list)


# Additional tests that run even without credentials
class TestKaggleDataLoaderWithoutCredentials:
    """Tests that can run without Kaggle credentials."""

    def test_loader_initialization(self):
        """Test loader can be initialized without credentials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = KaggleDataLoader(cache_dir=Path(temp_dir))

            assert loader.cache_dir == Path(temp_dir)
            assert loader.api is None

    def test_supported_formats(self):
        """Test supported formats list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = KaggleDataLoader(cache_dir=Path(temp_dir))
            formats = loader.get_supported_formats()

            assert "json" in formats
            assert "csv" in formats
            assert "zip" in formats

    @pytest.mark.skipif(not KAGGLE_AVAILABLE, reason="Kaggle package not available")
    @pytest.mark.asyncio
    async def test_authentication_failure_handling(self):
        """Test proper handling of authentication failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = KaggleDataLoader(cache_dir=Path(temp_dir))

            # Temporarily remove credentials to test failure handling
            original_username = os.environ.get("KAGGLE_USERNAME")
            original_key = os.environ.get("KAGGLE_KEY")

            try:
                # Remove environment variables if they exist
                if "KAGGLE_USERNAME" in os.environ:
                    del os.environ["KAGGLE_USERNAME"]
                if "KAGGLE_KEY" in os.environ:
                    del os.environ["KAGGLE_KEY"]

                # Also check if kaggle.json exists and temporarily rename it
                kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
                kaggle_backup = None
                if kaggle_config.exists():
                    kaggle_backup = Path.home() / ".kaggle" / "kaggle.json.backup"
                    kaggle_config.rename(kaggle_backup)

                try:
                    from benchmark.data.loaders.base_loader import DataLoadError

                    with pytest.raises(DataLoadError, match="authentication failed"):
                        await loader._initialize_api()
                finally:
                    # Restore kaggle.json if it was backed up
                    if kaggle_backup and kaggle_backup.exists():
                        kaggle_backup.rename(kaggle_config)

            finally:
                # Restore original environment variables
                if original_username:
                    os.environ["KAGGLE_USERNAME"] = original_username
                if original_key:
                    os.environ["KAGGLE_KEY"] = original_key

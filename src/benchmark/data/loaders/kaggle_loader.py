"""
Kaggle dataset loader for the LLM Cybersecurity Benchmark system.

This module provides a data loader for datasets from Kaggle using the Kaggle API,
with support for authentication, downloading, extraction, and progress reporting.
"""

import asyncio
import shutil
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaggle.api.kaggle_api_extended import KaggleApi as KaggleApiType
else:
    KaggleApiType = None

try:
    # Check if kaggle is available without triggering authentication
    import importlib.util

    kaggle_spec = importlib.util.find_spec("kaggle")
    KAGGLE_AVAILABLE = kaggle_spec is not None

    # Only import when actually needed to avoid authentication on import
    KaggleApi = None
except ImportError:
    KAGGLE_AVAILABLE = False
    KaggleApi = None

from benchmark.data.loaders.base_loader import DataLoader, DataLoadError, LoadProgress
from benchmark.data.loaders.local_loader import LocalFileDataLoader
from benchmark.data.models import Dataset


class KaggleDataLoader(DataLoader):
    """
    Data loader for Kaggle datasets.

    Downloads datasets from Kaggle using the Kaggle API, handles authentication,
    extracts compressed files, and loads the data using the LocalFileDataLoader.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        show_progress: bool = True,
        force_download: bool = False,
    ):
        """
        Initialize the Kaggle data loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
            show_progress: Whether to show download progress
            force_download: Whether to force re-download existing files
        """
        super().__init__()
        self.api: KaggleApiType | None = None
        self.cache_dir = cache_dir or Path.home() / ".kaggle" / "datasets"
        self.show_progress = show_progress
        self.force_download = force_download
        self.progress_callbacks: list[Any] = []

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize local loader for actual data loading
        self.local_loader = LocalFileDataLoader()

    def add_progress_callback(self, callback: Any) -> None:
        """Add a callback function for progress updates."""
        self.progress_callbacks.append(callback)

    async def load(self, config: dict[str, Any]) -> Dataset:
        """
        Load dataset from Kaggle.

        Args:
            config: Configuration containing:
                - dataset: Kaggle dataset identifier (e.g., 'username/dataset-name')
                - file_path: Optional specific file within dataset to load
                - field_mapping: Optional field mapping configuration
                - name: Optional dataset name
                - description: Optional dataset description

        Returns:
            Loaded dataset

        Raises:
            ValueError: If configuration is invalid
            DataLoadError: If dataset cannot be loaded
        """
        self._validate_config(config, ["dataset"])

        dataset_id = config["dataset"]
        specific_file = config.get("file_path")

        self.logger.info(f"Loading Kaggle dataset: {dataset_id}")

        # Initialize API if not already done
        await self._initialize_api()

        # Validate source
        if not await self.validate_source(config):
            raise DataLoadError(f"Kaggle dataset not accessible: {dataset_id}")

        # Download dataset
        download_path = await self._download_dataset(dataset_id)

        # Extract if compressed
        extracted_path = await self._extract_files(download_path)

        # Find data files
        data_files = self._find_data_files(extracted_path, specific_file)

        if not data_files:
            raise DataLoadError(f"No data files found in dataset: {dataset_id}", extracted_path)

        # Load the first suitable data file using LocalFileDataLoader
        data_file = data_files[0]
        self.logger.info(f"Loading data from file: {data_file}")

        # Prepare config for local loader
        local_config = {
            "file_path": str(data_file),
            "name": config.get("name") or f"kaggle_{dataset_id.replace('/', '_')}",
            "description": config.get("description") or f"Kaggle dataset: {dataset_id}",
            "field_mapping": config.get("field_mapping"),
        }

        # Load using local loader
        dataset = await self.local_loader.load(local_config)

        # Update dataset info with Kaggle metadata
        dataset.info.source = f"kaggle:{dataset_id}"
        dataset.info.metadata.update(
            {
                "kaggle_dataset": dataset_id,
                "source_file": str(data_file),
                "loader": "KaggleDataLoader",
                "cache_path": str(extracted_path),
            }
        )

        self.logger.info(
            f"Successfully loaded Kaggle dataset: {len(dataset.samples)} samples "
            f"({len(dataset.attack_samples)} attacks, {len(dataset.benign_samples)} benign)"
        )

        return dataset

    async def validate_source(self, config: dict[str, Any]) -> bool:
        """
        Validate that Kaggle dataset exists and is accessible.

        Args:
            config: Configuration containing dataset identifier

        Returns:
            True if dataset is valid and accessible, False otherwise
        """
        try:
            dataset_id = config.get("dataset", "")
            if not dataset_id:
                self.logger.error("No dataset identifier provided")
                return False

            # Initialize API if not already done
            await self._initialize_api()

            # Try to get dataset info
            parts = dataset_id.split("/")
            if len(parts) != 2:
                self.logger.error(
                    f"Invalid dataset format: {dataset_id}. Expected 'owner/dataset-name'"
                )
                return False

            owner, dataset_name = parts

            # Check if dataset exists and is accessible
            try:
                if self.api is not None:
                    dataset_info = self.api.dataset_metadata(owner, dataset_name)
                    self.logger.debug(f"Dataset found: {dataset_info.get('title', dataset_id)}")
                    return True
                else:
                    self.logger.error("API not initialized")
                    return False
            except Exception as e:
                self.logger.error(f"Dataset not accessible: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error validating Kaggle source: {e}")
            return False

    def get_supported_formats(self) -> list[str]:
        """Get list of supported data formats."""
        # Support same formats as LocalFileDataLoader plus zip archives
        return self.local_loader.get_supported_formats() + ["zip"]

    async def _initialize_api(self) -> None:
        """Initialize Kaggle API with authentication."""
        if not KAGGLE_AVAILABLE:
            raise DataLoadError("Kaggle package not available. Install with: pip install kaggle")

        if self.api is not None:
            return

        try:
            # Import KaggleApi only when needed to avoid import-time authentication
            from kaggle.api.kaggle_api_extended import KaggleApi

            self.api = KaggleApi()

            # Try to authenticate
            self.api.authenticate()
            self.logger.debug("Kaggle API authentication successful")

        except Exception as e:
            error_msg = (
                f"Kaggle API authentication failed: {e}. "
                "Please ensure you have valid Kaggle credentials. "
                "You can set up authentication by:\n"
                "1. Creating a kaggle.json file with your API credentials\n"
                "2. Setting KAGGLE_USERNAME and KAGGLE_KEY environment variables"
            )
            raise DataLoadError(error_msg) from e

    async def _download_dataset(self, dataset_id: str) -> Path:
        """
        Download dataset from Kaggle.

        Args:
            dataset_id: Kaggle dataset identifier (owner/dataset-name)

        Returns:
            Path to downloaded dataset
        """
        owner, dataset_name = dataset_id.split("/")
        download_dir = self.cache_dir / owner / dataset_name

        # Check if already downloaded and not forcing re-download
        if download_dir.exists() and not self.force_download:
            self.logger.info(f"Using cached dataset: {download_dir}")
            return download_dir

        # Create download directory
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Set up progress reporting
            progress = None
            if self.show_progress:
                progress = LoadProgress()
                for callback in self.progress_callbacks:
                    progress.add_callback(callback)
                progress.start()

            self.logger.info(f"Downloading dataset from Kaggle: {dataset_id}")

            # Download dataset - run in thread pool to avoid blocking
            if self.api is not None:
                api = self.api  # Capture for lambda to help mypy
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: api.dataset_download_files(
                        dataset=dataset_id,
                        path=str(download_dir),
                        unzip=False,  # We'll handle extraction ourselves
                    ),
                )
            else:
                raise DataLoadError("API not initialized")

            if progress:
                progress.complete()

            self.logger.info(f"Dataset downloaded to: {download_dir}")
            return download_dir

        except Exception as e:
            raise DataLoadError(f"Failed to download Kaggle dataset {dataset_id}: {e}") from e

    async def _extract_files(self, download_path: Path) -> Path:
        """
        Extract compressed dataset files.

        Args:
            download_path: Path to downloaded files

        Returns:
            Path to extracted files directory
        """
        # Look for zip files in download directory
        zip_files = list(download_path.glob("*.zip"))

        if not zip_files:
            # No zip files, return download path directly
            return download_path

        # Extract the first zip file (most datasets have one main zip)
        zip_file = zip_files[0]
        extract_dir = download_path / "extracted"

        # Check if already extracted
        if extract_dir.exists() and not self.force_download:
            self.logger.debug(f"Using existing extraction: {extract_dir}")
            return extract_dir

        try:
            extract_dir.mkdir(exist_ok=True)

            self.logger.info(f"Extracting dataset: {zip_file}")

            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            self.logger.info(f"Dataset extracted to: {extract_dir}")
            return extract_dir

        except Exception as e:
            raise DataLoadError(f"Failed to extract dataset files: {e}") from e

    def _find_data_files(self, extract_dir: Path, specific_file: str | None = None) -> list[Path]:
        """
        Find actual data files in extracted directory.

        Args:
            extract_dir: Directory containing extracted files
            specific_file: Optional specific file to look for

        Returns:
            List of data file paths
        """
        data_files = []

        # Supported data file extensions
        supported_extensions = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".pq"}

        if specific_file:
            # Look for specific file
            specific_path = extract_dir / specific_file
            if specific_path.exists():
                data_files.append(specific_path)
            else:
                # Try finding it recursively
                for file_path in extract_dir.rglob(specific_file):
                    data_files.append(file_path)
        else:
            # Find all data files recursively
            for file_path in extract_dir.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in supported_extensions
                    and not file_path.name.startswith(".")
                    and file_path.name.lower()
                    not in {
                        "readme.md",
                        "license.txt",
                        "metadata.json",
                        "dataset_description.txt",
                    }
                ):
                    data_files.append(file_path)

        # Sort by file size (largest first) to prioritize main data files
        data_files.sort(key=lambda p: p.stat().st_size, reverse=True)

        self.logger.debug(f"Found {len(data_files)} data files: {[f.name for f in data_files[:5]]}")
        return data_files

    async def cleanup_cache(self, dataset_id: str | None = None) -> None:
        """
        Clean up cached datasets.

        Args:
            dataset_id: Optional specific dataset to clean. If None, cleans all cache.
        """
        if dataset_id:
            owner, dataset_name = dataset_id.split("/")
            cache_path = self.cache_dir / owner / dataset_name
            if cache_path.exists():
                shutil.rmtree(cache_path)
                self.logger.info(f"Cleaned cache for dataset: {dataset_id}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Cleaned entire dataset cache")

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about cached datasets.

        Returns:
            Dictionary with cache information
        """
        cache_info: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "total_size_bytes": 0,
            "datasets": [],
        }

        if not self.cache_dir.exists():
            return cache_info

        for owner_dir in self.cache_dir.iterdir():
            if owner_dir.is_dir():
                for dataset_dir in owner_dir.iterdir():
                    if dataset_dir.is_dir():
                        dataset_id = f"{owner_dir.name}/{dataset_dir.name}"
                        size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
                        cache_info["datasets"].append(
                            {
                                "id": dataset_id,
                                "path": str(dataset_dir),
                                "size_bytes": size,
                                "files": len(list(dataset_dir.rglob("*"))),
                            }
                        )
                        cache_info["total_size_bytes"] = cache_info["total_size_bytes"] + size

        return cache_info

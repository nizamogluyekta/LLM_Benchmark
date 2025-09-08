"""
HuggingFace datasets data loader for the LLM Cybersecurity Benchmark system.

This module provides a data loader for HuggingFace datasets with support for
streaming large datasets, automatic field mapping, and progress reporting.
"""

import asyncio
from typing import Any, Union

from benchmark.data.loaders.base_loader import DataLoader, DataLoadError, FieldMapping, LoadProgress
from benchmark.data.models import Dataset, DatasetInfo, DatasetSample

try:
    from datasets import (
        Dataset as HFDataset,
    )
    from datasets import (
        DatasetDict,
        IterableDataset,
        get_dataset_config_names,
        get_dataset_infos,
        load_dataset,
    )

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    HFDataset = None
    DatasetDict = None
    IterableDataset = None


class HuggingFaceDataLoader(DataLoader):
    """
    Data loader for HuggingFace datasets.

    Supports streaming large datasets, automatic field mapping for common
    dataset formats, progress reporting, and built-in HuggingFace caching.
    """

    # Default field mappings for common cybersecurity datasets
    COMMON_FIELD_MAPPINGS = {
        "default": {
            "input_text_field": "text",
            "label_field": "label",
            "attack_type_field": "attack_type",
        },
        "malware": {
            "input_text_field": "text",
            "label_field": "malware",
            "attack_type_field": "family",
        },
        "phishing": {
            "input_text_field": "url",
            "label_field": "label",
            "attack_type_field": "type",
        },
        "network": {
            "input_text_field": "packet",
            "label_field": "label",
            "attack_type_field": "attack",
        },
    }

    # Streaming threshold - datasets larger than this use streaming mode
    STREAMING_THRESHOLD_SAMPLES = 100000

    def __init__(
        self,
        field_mapping: FieldMapping | None = None,
        streaming: bool = False,
        cache_dir: str | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize the HuggingFace data loader.

        Args:
            field_mapping: Configuration for mapping HF fields to dataset fields
            streaming: Force streaming mode for large datasets
            cache_dir: Directory for HuggingFace cache (uses default if None)
            show_progress: Whether to show progress for dataset loading
        """
        super().__init__()

        if not DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library is required. "
                "Install with: pip install datasets>=2.16.0"
            )

        self.field_mapping = field_mapping or FieldMapping()
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.show_progress = show_progress
        self.progress_callbacks: list[Any] = []

    def add_progress_callback(self, callback: Any) -> None:
        """Add a callback function for progress updates."""
        self.progress_callbacks.append(callback)

    async def load(self, config: dict[str, Any]) -> Dataset:
        """
        Load dataset from HuggingFace.

        Args:
            config: Configuration containing:
                - dataset_path: HuggingFace dataset path (e.g., "squad", "glue")
                - dataset_name: Optional dataset configuration name
                - split: Dataset split to load (train/test/validation)
                - streaming: Override streaming mode
                - field_mapping: Optional field mapping configuration
                - max_samples: Optional limit on number of samples
                - name: Optional dataset name override
                - description: Optional dataset description

        Returns:
            Loaded dataset

        Raises:
            ValueError: If configuration is invalid
            DataLoadError: If dataset cannot be loaded
        """
        self._validate_config(config, ["dataset_path"])

        dataset_path = config["dataset_path"]
        dataset_name = config.get("dataset_name")
        split = config.get("split", "train")

        # Override streaming mode if specified in config
        use_streaming = config.get("streaming", self.streaming)

        self.logger.info(f"Loading HuggingFace dataset: {dataset_path}")
        if dataset_name:
            self.logger.info(f"Dataset configuration: {dataset_name}")

        # Validate source first
        if not await self.validate_source(config):
            raise DataLoadError(f"HuggingFace dataset not accessible: {dataset_path}")

        # Override field mapping if provided in config
        field_mapping = self._get_field_mapping(config)

        # Get dataset info for size estimation
        try:
            dataset_info = await self._get_dataset_info(dataset_path, dataset_name)

            # Determine if we should use streaming based on dataset size
            if not config.get("streaming") and not self.streaming:
                estimated_size = (
                    dataset_info.get("splits", {}).get(split, {}).get("num_examples", 0)
                )
                if estimated_size > self.STREAMING_THRESHOLD_SAMPLES:
                    use_streaming = True
                    self.logger.info(
                        f"Dataset has {estimated_size} samples, enabling streaming mode"
                    )

        except Exception as e:
            self.logger.warning(f"Could not get dataset info: {e}")
            dataset_info = {}

        # Load dataset
        try:
            if use_streaming:
                hf_dataset = await self._load_dataset_streaming(
                    dataset_path, dataset_name, split, config
                )
            else:
                hf_dataset = await self._load_dataset_full(
                    dataset_path, dataset_name, split, config
                )
        except Exception as e:
            raise DataLoadError(f"Failed to load HuggingFace dataset: {e}", cause=e) from e

        # Map HuggingFace fields to our format
        samples = await self._map_huggingface_fields(hf_dataset, field_mapping, config)

        # Create dataset info
        info = self._create_dataset_info(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            samples=samples,
            name=config.get("name"),
            description=config.get("description"),
            hf_info=dataset_info,
            split=split,
        )

        # Create and return dataset
        dataset = Dataset(info=info, samples=samples, splits=None, checksum=None)

        self.logger.info(
            f"Successfully loaded HuggingFace dataset: {len(samples)} samples "
            f"({len(dataset.attack_samples)} attacks, {len(dataset.benign_samples)} benign)"
        )

        return dataset

    async def validate_source(self, config: dict[str, Any]) -> bool:
        """
        Validate that HuggingFace dataset exists and is accessible.

        Args:
            config: Configuration containing dataset_path and optional dataset_name

        Returns:
            True if dataset is valid and accessible, False otherwise
        """
        try:
            dataset_path = config.get("dataset_path", "")
            dataset_name = config.get("dataset_name")

            if not dataset_path:
                self.logger.error("No dataset_path provided")
                return False

            # Try to get dataset info to verify it exists
            info = await self._get_dataset_info(dataset_path, dataset_name)
            # If we got meaningful info back, dataset exists
            return len(info) > 0 and not (len(info) == 1 and "configs" in info)

        except Exception as e:
            self.logger.error(f"Error validating HuggingFace dataset: {e}")
            return False

    def get_supported_formats(self) -> list[str]:
        """Get list of supported data formats."""
        return ["huggingface", "hf"]

    async def _load_dataset_streaming(
        self,
        dataset_path: str,
        dataset_name: str | None,
        split: str,
        config: dict[str, Any],
    ) -> IterableDataset:
        """
        Load dataset in streaming mode for large datasets.

        Args:
            dataset_path: HuggingFace dataset path
            dataset_name: Optional dataset configuration name
            split: Dataset split to load
            config: Additional configuration

        Returns:
            Streaming dataset

        Raises:
            DataLoadError: If dataset cannot be loaded in streaming mode
        """
        try:
            self.logger.info("Loading dataset in streaming mode")

            load_kwargs = {
                "path": dataset_path,
                "split": split,
                "streaming": True,
                "cache_dir": self.cache_dir,
            }

            if dataset_name:
                load_kwargs["name"] = dataset_name

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            hf_dataset = await loop.run_in_executor(None, lambda: load_dataset(**load_kwargs))

            return hf_dataset

        except Exception as e:
            raise DataLoadError(f"Failed to load streaming dataset: {e}", cause=e) from e

    async def _load_dataset_full(
        self,
        dataset_path: str,
        dataset_name: str | None,
        split: str,
        config: dict[str, Any],
    ) -> HFDataset:
        """
        Load full dataset into memory.

        Args:
            dataset_path: HuggingFace dataset path
            dataset_name: Optional dataset configuration name
            split: Dataset split to load
            config: Additional configuration

        Returns:
            Full dataset in memory

        Raises:
            DataLoadError: If dataset cannot be loaded
        """
        try:
            self.logger.info("Loading full dataset into memory")

            load_kwargs = {
                "path": dataset_path,
                "split": split,
                "cache_dir": self.cache_dir,
            }

            if dataset_name:
                load_kwargs["name"] = dataset_name

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            hf_dataset = await loop.run_in_executor(None, lambda: load_dataset(**load_kwargs))

            return hf_dataset

        except Exception as e:
            raise DataLoadError(f"Failed to load full dataset: {e}", cause=e) from e

    async def _map_huggingface_fields(
        self,
        hf_dataset: Union[HFDataset, IterableDataset],  # noqa: UP007
        field_mapping: FieldMapping,
        config: dict[str, Any],
    ) -> list[DatasetSample]:
        """
        Map HuggingFace dataset fields to our DatasetSample format.

        Args:
            hf_dataset: HuggingFace dataset (regular or iterable)
            field_mapping: Field mapping configuration
            config: Additional configuration

        Returns:
            List of DatasetSample objects

        Raises:
            DataLoadError: If field mapping fails
        """
        samples = []
        max_samples = config.get("max_samples")

        try:
            # Setup progress tracking
            progress = None
            if self.show_progress:
                total_samples = None
                if hasattr(hf_dataset, "__len__"):
                    total_samples = len(hf_dataset)
                elif max_samples:
                    total_samples = max_samples

                if total_samples:
                    progress = LoadProgress(total_samples)
                    for callback in self.progress_callbacks:
                        progress.add_callback(callback)
                    progress.start()

            # Process samples
            processed_count = 0
            for i, sample in enumerate(hf_dataset):
                try:
                    # Check max samples limit
                    if max_samples and processed_count >= max_samples:
                        break

                    # Validate required fields exist
                    field_mapping.validate_fields(sample)

                    # Extract main fields
                    input_text = sample.get(field_mapping.input_text_field)
                    label = sample.get(field_mapping.label_field)
                    attack_type = sample.get(field_mapping.attack_type_field)

                    # Normalize label to standard format
                    label = self._normalize_label(str(label) if label is not None else "")

                    # Extract metadata
                    metadata = field_mapping.extract_metadata(sample)
                    metadata["hf_index"] = i
                    metadata["source"] = "huggingface"

                    # Add any extra fields to metadata
                    for key, value in sample.items():
                        if (
                            key
                            not in [
                                field_mapping.input_text_field,
                                field_mapping.label_field,
                                field_mapping.attack_type_field,
                            ]
                            and key not in field_mapping.metadata_fields
                            and key not in metadata
                            and value is not None
                        ):
                            metadata[key] = value

                    # Create DatasetSample
                    dataset_sample = DatasetSample(
                        input_text=str(input_text) if input_text is not None else "",
                        label=label,
                        attack_type=attack_type,
                        metadata=metadata,
                        timestamp=None,
                        confidence_score=None,
                        source_file=None,
                        line_number=None,
                    )

                    samples.append(dataset_sample)
                    processed_count += 1

                    # Update progress
                    if progress and processed_count % 100 == 0:
                        progress.update(100)

                except Exception as e:
                    self.logger.warning(f"Failed to process sample {i}: {e}")
                    continue

            if progress:
                progress.complete()

            return samples

        except Exception as e:
            raise DataLoadError(f"Failed to map HuggingFace fields: {e}", cause=e) from e

    async def _get_dataset_info(
        self, dataset_path: str, dataset_name: str | None = None
    ) -> dict[str, Any]:
        """
        Get information about HuggingFace dataset.

        Args:
            dataset_path: HuggingFace dataset path
            dataset_name: Optional dataset configuration name

        Returns:
            Dictionary containing dataset information

        Raises:
            Exception: If dataset info cannot be retrieved
        """
        loop = asyncio.get_event_loop()

        try:
            # Get dataset infos
            dataset_infos = await loop.run_in_executor(
                None, lambda: get_dataset_infos(dataset_path)
            )

            # If specific config name provided, get that config
            if dataset_name:
                if dataset_name in dataset_infos:
                    return dict(dataset_infos[dataset_name].__dict__)
                else:
                    # This is a user error, not a dataset access error, so re-raise
                    raise ValueError(f"Dataset config '{dataset_name}' not found")

            # Otherwise, return the first available config
            if dataset_infos:
                first_config = list(dataset_infos.values())[0]
                return dict(first_config.__dict__)

            return {}

        except ValueError:
            # Re-raise ValueError for config not found
            raise
        except Exception as e:
            # If get_dataset_infos fails, try to get config names at least
            try:
                config_names = await loop.run_in_executor(
                    None, lambda: get_dataset_config_names(dataset_path)
                )
                return {"configs": config_names}
            except Exception:
                # If everything fails, return empty dict
                self.logger.warning(f"Could not retrieve dataset info for {dataset_path}: {e}")
                return {}

    def _get_field_mapping(self, config: dict[str, Any]) -> FieldMapping:
        """
        Get field mapping configuration, with automatic detection for common datasets.

        Args:
            config: Configuration dictionary

        Returns:
            FieldMapping instance
        """
        # If explicit field mapping provided in config, use it
        if "field_mapping" in config:
            field_mapping_config = config["field_mapping"]
            return FieldMapping(
                input_text_field=field_mapping_config.get("input_text_field", "text"),
                label_field=field_mapping_config.get("label_field", "label"),
                attack_type_field=field_mapping_config.get("attack_type_field", "attack_type"),
                metadata_fields=field_mapping_config.get("metadata_fields", []),
                required_fields=field_mapping_config.get("required_fields"),
            )

        # Try to auto-detect based on dataset path
        dataset_path = config.get("dataset_path", "").lower()

        # Check for known dataset patterns
        for pattern, mapping in self.COMMON_FIELD_MAPPINGS.items():
            if pattern != "default" and pattern in dataset_path:
                self.logger.info(f"Using {pattern} field mapping for dataset")
                return FieldMapping(
                    input_text_field=mapping["input_text_field"],
                    label_field=mapping["label_field"],
                    attack_type_field=mapping["attack_type_field"],
                )

        # Fall back to default mapping
        default_mapping = self.COMMON_FIELD_MAPPINGS["default"]
        return FieldMapping(
            input_text_field=default_mapping["input_text_field"],
            label_field=default_mapping["label_field"],
            attack_type_field=default_mapping["attack_type_field"],
        )

    def _normalize_label(self, label: str) -> str:
        """
        Normalize label to standard ATTACK/BENIGN format.

        Args:
            label: Raw label value

        Returns:
            Normalized label (ATTACK or BENIGN)
        """
        if not label:
            return label

        label_lower = label.lower().strip()

        # Map common attack label variations
        attack_labels = {
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
        }

        # Map common benign label variations
        benign_labels = {
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
        }

        if label_lower in attack_labels:
            return "ATTACK"
        elif label_lower in benign_labels:
            return "BENIGN"
        else:
            # Return original if no mapping found - let DatasetSample validation handle it
            return label.upper()

    def _create_dataset_info(
        self,
        dataset_path: str,
        dataset_name: str | None,
        samples: list[DatasetSample],
        name: str | None = None,
        description: str | None = None,
        hf_info: dict[str, Any] | None = None,
        split: str = "train",
    ) -> DatasetInfo:
        """
        Create DatasetInfo from loaded samples and HuggingFace metadata.

        Args:
            dataset_path: HuggingFace dataset path
            dataset_name: Optional dataset configuration name
            samples: List of dataset samples
            name: Optional dataset name override
            description: Optional dataset description
            hf_info: HuggingFace dataset info
            split: Dataset split loaded

        Returns:
            DatasetInfo object
        """
        # Count attack types
        attack_types = set()
        attack_count = 0
        benign_count = 0

        for sample in samples:
            if sample.label == "ATTACK":
                attack_count += 1
                if sample.attack_type:
                    attack_types.add(sample.attack_type)
            elif sample.label == "BENIGN":
                benign_count += 1

        # Create dataset source identifier
        source_id = f"huggingface:{dataset_path}"
        if dataset_name:
            source_id += f"/{dataset_name}"
        source_id += f"#{split}"

        # Extract description from HF info if available
        if not description and hf_info:
            description = str(hf_info.get("description", "")).strip()

        # Create metadata
        metadata: dict[str, Any] = {
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "split": split,
            "loader": "HuggingFaceDataLoader",
        }

        if hf_info is not None:
            metadata["huggingface_info"] = hf_info

        return DatasetInfo(
            name=name or f"{dataset_path.replace('/', '_')}_{split}",
            source=source_id,
            total_samples=len(samples),
            attack_samples=attack_count,
            benign_samples=benign_count,
            attack_types=sorted(attack_types),
            description=description,
            size_bytes=None,  # HF doesn't provide easy access to size
            format="huggingface",
            metadata=metadata,
            updated_at=None,
        )

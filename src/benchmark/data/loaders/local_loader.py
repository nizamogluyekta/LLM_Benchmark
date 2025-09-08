"""
Local file data loader for the LLM Cybersecurity Benchmark system.

This module provides a data loader for local files in various formats
including JSON, CSV, and Parquet files commonly used for cybersecurity datasets.
"""

import csv
import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from benchmark.data.loaders.base_loader import DataLoader, DataLoadError, FieldMapping, LoadProgress
from benchmark.data.models import Dataset, DatasetInfo, DatasetSample


class LocalFileDataLoader(DataLoader):
    """
    Data loader for local files in JSON, CSV, and Parquet formats.

    Supports automatic format detection, configurable field mapping,
    data validation, and progress reporting for large files.
    """

    # File size threshold for progress reporting (10MB)
    PROGRESS_THRESHOLD_BYTES = 10 * 1024 * 1024

    # Chunk size for processing large files
    CHUNK_SIZE = 1000

    def __init__(
        self,
        field_mapping: FieldMapping | None = None,
        chunk_size: int = CHUNK_SIZE,
        show_progress: bool = True,
    ):
        """
        Initialize the local file data loader.

        Args:
            field_mapping: Configuration for mapping source fields to dataset fields
            chunk_size: Number of records to process in each chunk
            show_progress: Whether to show progress for large files
        """
        super().__init__()
        self.field_mapping = field_mapping or FieldMapping()
        self.chunk_size = chunk_size
        self.show_progress = show_progress
        self.progress_callbacks: list[Any] = []

    def add_progress_callback(self, callback: Any) -> None:
        """Add a callback function for progress updates."""
        self.progress_callbacks.append(callback)

    async def load(self, config: dict[str, Any]) -> Dataset:
        """
        Load dataset from local file.

        Args:
            config: Configuration containing:
                - file_path: Path to the data file
                - field_mapping: Optional field mapping configuration
                - format: Optional format override (auto-detected if not provided)
                - name: Optional dataset name
                - description: Optional dataset description

        Returns:
            Loaded dataset

        Raises:
            ValueError: If configuration is invalid
            DataLoadError: If file cannot be loaded or parsed
        """
        self._validate_config(config, ["file_path"])

        file_path = Path(config["file_path"])

        # Validate source first
        if not await self.validate_source(config):
            raise DataLoadError(f"Source file not accessible: {file_path}")

        # Override field mapping if provided in config
        field_mapping = self.field_mapping
        if "field_mapping" in config:
            field_mapping_config = config["field_mapping"]
            field_mapping = FieldMapping(
                input_text_field=field_mapping_config.get("input_text_field", "input_text"),
                label_field=field_mapping_config.get("label_field", "label"),
                attack_type_field=field_mapping_config.get("attack_type_field", "attack_type"),
                metadata_fields=field_mapping_config.get("metadata_fields", []),
                required_fields=field_mapping_config.get("required_fields", None),
            )

        # Detect format
        format_name = config.get("format") or self._detect_format(file_path)

        self.logger.info(f"Loading {format_name.upper()} file: {file_path}")

        # Load raw data based on format
        try:
            if format_name == "json":
                raw_data = await self._load_json(file_path)
            elif format_name == "csv":
                raw_data = await self._load_csv(file_path, config.get("csv_options", {}))
            elif format_name == "parquet":
                raw_data = await self._load_parquet(file_path)
            else:
                raise DataLoadError(f"Unsupported format: {format_name}", file_path)
        except Exception as e:
            if isinstance(e, DataLoadError):
                raise
            raise DataLoadError(f"Failed to load {format_name} file: {e}", file_path, e) from e

        # Map fields and create dataset samples
        samples = self._map_fields(raw_data, field_mapping)

        # Create dataset info
        info = self._create_dataset_info(
            file_path=file_path,
            samples=samples,
            name=config.get("name"),
            description=config.get("description"),
            format_name=format_name,
        )

        # Create and return dataset
        dataset = Dataset(info=info, samples=samples, splits=None, checksum=None)

        self.logger.info(
            f"Successfully loaded dataset: {len(samples)} samples "
            f"({len(dataset.attack_samples)} attacks, {len(dataset.benign_samples)} benign)"
        )

        return dataset

    async def validate_source(self, config: dict[str, Any]) -> bool:
        """
        Validate that local file exists and is readable.

        Args:
            config: Configuration containing file_path

        Returns:
            True if file is valid and accessible, False otherwise
        """
        try:
            file_path = Path(config.get("file_path", ""))

            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return False

            if not file_path.is_file():
                self.logger.error(f"Path is not a file: {file_path}")
                return False

            # Check if file is readable
            try:
                with open(file_path, encoding="utf-8") as f:
                    f.read(1)  # Try to read first byte
                return True
            except PermissionError:
                self.logger.error(f"No permission to read file: {file_path}")
                return False
            except UnicodeDecodeError:
                # Might be a binary format like Parquet
                if file_path.suffix.lower() == ".parquet":
                    return PANDAS_AVAILABLE
                return False
        except Exception as e:
            self.logger.error(f"Error validating source: {e}")
            return False

    def get_supported_formats(self) -> list[str]:
        """Get list of supported data formats."""
        formats = ["json", "csv"]
        if PANDAS_AVAILABLE:
            formats.append("parquet")
        return formats

    async def _load_json(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Load data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of data records

        Raises:
            DataLoadError: If JSON file cannot be parsed
        """
        try:
            # Check file size for progress reporting
            file_size = file_path.stat().st_size
            show_progress = self.show_progress and file_size > self.PROGRESS_THRESHOLD_BYTES

            progress = None
            if show_progress:
                progress = LoadProgress()
                for callback in self.progress_callbacks:
                    progress.add_callback(callback)
                progress.start()

            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse JSON
            data = json.loads(content)

            # Handle different JSON structures
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Look for common dataset structure keys
                if "samples" in data:
                    records = data["samples"]
                elif "data" in data:
                    records = data["data"]
                elif "records" in data:
                    records = data["records"]
                else:
                    # Treat single dict as single record
                    records = [data]
            else:
                raise DataLoadError("Unexpected JSON structure: expected list or dict", file_path)

            if progress:
                progress.complete()

            self.logger.debug(f"Loaded {len(records)} records from JSON file")
            return records

        except json.JSONDecodeError as e:
            raise DataLoadError(f"Invalid JSON format: {e}", file_path, e) from e
        except Exception as e:
            raise DataLoadError(f"Failed to load JSON file: {e}", file_path, e) from e

    async def _load_csv(self, file_path: Path, csv_options: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            csv_options: CSV parsing options (delimiter, quotechar, etc.)

        Returns:
            List of data records

        Raises:
            DataLoadError: If CSV file cannot be parsed
        """
        try:
            # Set default CSV options
            delimiter = csv_options.get("delimiter", ",")
            quotechar = csv_options.get("quotechar", '"')
            encoding = csv_options.get("encoding", "utf-8")
            has_header = csv_options.get("has_header", True)

            # Check file size for progress reporting
            file_size = file_path.stat().st_size
            show_progress = self.show_progress and file_size > self.PROGRESS_THRESHOLD_BYTES

            records = []

            with open(file_path, encoding=encoding) as f:
                # Create CSV reader
                reader = (
                    csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
                    if has_header
                    else csv.reader(f, delimiter=delimiter, quotechar=quotechar)
                )

                progress = None
                if show_progress:
                    # Estimate total rows for progress (rough estimate)
                    f.seek(0, 2)  # Seek to end
                    total_bytes = f.tell()
                    f.seek(0)  # Seek back to beginning

                    # Skip header if present
                    if has_header:
                        next(reader)

                    # Estimate rows based on first few lines
                    sample_lines = []
                    for i, row in enumerate(reader):
                        if i >= 10:  # Sample first 10 lines
                            break
                        sample_lines.append(row)

                    if sample_lines:
                        avg_line_length = total_bytes / (
                            len(sample_lines) + (1 if has_header else 0)
                        )
                        estimated_rows = int(total_bytes / avg_line_length)
                    else:
                        estimated_rows = 0

                    f.seek(0)  # Reset to beginning
                    reader = (
                        csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
                        if has_header
                        else csv.reader(f, delimiter=delimiter, quotechar=quotechar)
                    )

                    progress = LoadProgress(estimated_rows)
                    for callback in self.progress_callbacks:
                        progress.add_callback(callback)
                    progress.start()

                # Process rows
                if has_header:
                    for row_num, row in enumerate(reader):
                        records.append(dict(row))  # type: ignore[arg-type]

                        if progress and row_num % 100 == 0:  # Update progress every 100 rows
                            progress.update(100)
                else:
                    # Handle CSV without header - create generic field names
                    first_row = next(reader, None)
                    if first_row:
                        headers = [f"column_{i}" for i in range(len(first_row))]
                        records.append(dict(zip(headers, first_row, strict=False)))

                        for row in reader:
                            if len(row) == len(headers):
                                records.append(dict(zip(headers, row, strict=False)))

            if progress:
                progress.complete()

            self.logger.debug(f"Loaded {len(records)} records from CSV file")
            return records

        except Exception as e:
            raise DataLoadError(f"Failed to load CSV file: {e}", file_path, e) from e

    async def _load_parquet(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Load data from Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            List of data records

        Raises:
            DataLoadError: If Parquet file cannot be parsed or pandas is not available
        """
        if not PANDAS_AVAILABLE:
            raise DataLoadError("Parquet support requires pandas to be installed", file_path)

        try:
            # Check file size for progress reporting
            file_size = file_path.stat().st_size
            show_progress = self.show_progress and file_size > self.PROGRESS_THRESHOLD_BYTES

            progress = None
            if show_progress:
                progress = LoadProgress()
                for callback in self.progress_callbacks:
                    progress.add_callback(callback)
                progress.start()

            # Load parquet file
            df = pd.read_parquet(file_path)

            # Convert to list of dictionaries
            records = df.to_dict("records")

            if progress:
                progress.complete()

            self.logger.debug(f"Loaded {len(records)} records from Parquet file")
            return records  # type: ignore[no-any-return]

        except Exception as e:
            raise DataLoadError(f"Failed to load Parquet file: {e}", file_path, e) from e

    def _map_fields(
        self, raw_data: list[dict[str, Any]], field_mapping: FieldMapping
    ) -> list[DatasetSample]:
        """
        Map raw data fields to DatasetSample objects.

        Args:
            raw_data: List of raw data records
            field_mapping: Field mapping configuration

        Returns:
            List of DatasetSample objects

        Raises:
            DataLoadError: If required fields are missing or data validation fails
        """
        samples = []

        for i, record in enumerate(raw_data):
            try:
                # Validate required fields
                field_mapping.validate_fields(record)

                # Extract main fields
                input_text = record.get(field_mapping.input_text_field)
                label = record.get(field_mapping.label_field)
                attack_type = record.get(field_mapping.attack_type_field)

                # Normalize label to standard format
                label = self._normalize_label(str(label) if label is not None else "")

                # Extract metadata
                metadata = field_mapping.extract_metadata(record)

                # Add original record index to metadata
                metadata["source_index"] = i

                # Add any extra fields not in the standard mapping to metadata
                for key, value in record.items():
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
                sample = DatasetSample(
                    input_text=str(input_text) if input_text is not None else "",
                    label=label,
                    attack_type=attack_type,
                    metadata=metadata,
                    timestamp=None,
                    confidence_score=None,
                    source_file=None,
                    line_number=None,
                )

                samples.append(sample)

            except Exception as e:
                raise DataLoadError(f"Failed to process record {i}: {e}", cause=e) from e

        return samples

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
        }

        if label_lower in attack_labels:
            return "ATTACK"
        elif label_lower in benign_labels:
            return "BENIGN"
        else:
            # Return original if no mapping found - let DatasetSample validation handle it
            return label.upper()

    def _detect_format(self, file_path: Path) -> str:
        """
        Detect file format based on file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected format name

        Raises:
            DataLoadError: If format cannot be detected
        """
        suffix = file_path.suffix.lower()

        format_map = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
        }

        if suffix in format_map:
            format_name = format_map[suffix]

            # Check if format is supported
            if format_name not in self.get_supported_formats():
                raise DataLoadError(
                    f"Format '{format_name}' is not supported (missing dependencies?)", file_path
                )

            return format_name

        raise DataLoadError(f"Cannot detect format from extension: {suffix}", file_path)

    def _create_dataset_info(
        self,
        file_path: Path,
        samples: list[DatasetSample],
        name: str | None = None,
        description: str | None = None,
        format_name: str = "unknown",
    ) -> DatasetInfo:
        """
        Create DatasetInfo from loaded samples and metadata.

        Args:
            file_path: Path to source file
            samples: List of dataset samples
            name: Optional dataset name
            description: Optional dataset description
            format_name: File format name

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

        # Get file size
        file_size = file_path.stat().st_size

        return DatasetInfo(
            name=name or file_path.stem,
            source=f"local_file:{file_path}",
            total_samples=len(samples),
            attack_samples=attack_count,
            benign_samples=benign_count,
            attack_types=sorted(attack_types),
            description=description,
            size_bytes=file_size,
            format=format_name,
            metadata={"source_file": str(file_path), "loader": "LocalFileDataLoader"},
            updated_at=None,
        )

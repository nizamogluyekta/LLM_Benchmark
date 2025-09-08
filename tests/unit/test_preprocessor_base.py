"""
Unit tests for preprocessing base classes and utilities.

This module tests the base preprocessor interface, common utilities,
and preprocessing operations functionality.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from benchmark.data.models import DatasetSample
from benchmark.data.preprocessors.base import (
    DataPreprocessor,
    PreprocessingProgress,
)
from benchmark.data.preprocessors.common import PreprocessingUtilities


class TestPreprocessingProgress:
    """Test PreprocessingProgress functionality."""

    def test_init_default(self):
        """Test progress initialization with defaults."""
        progress = PreprocessingProgress()

        assert progress.total_items == 0
        assert progress.processed_items == 0
        assert progress.failed_items == 0
        assert progress.start_time is None
        assert progress.callbacks == []

    def test_init_with_total(self):
        """Test progress initialization with total items."""
        progress = PreprocessingProgress(100)

        assert progress.total_items == 100
        assert progress.processed_items == 0
        assert progress.failed_items == 0

    def test_add_callback(self):
        """Test adding progress callbacks."""
        progress = PreprocessingProgress()
        callback = Mock()

        progress.add_callback(callback)
        assert callback in progress.callbacks

    def test_start_progress(self):
        """Test starting progress tracking."""
        progress = PreprocessingProgress()
        callback = Mock()
        progress.add_callback(callback)

        progress.start()

        assert progress.start_time is not None
        callback.assert_called_once_with(progress)

    def test_update_progress(self):
        """Test updating progress."""
        progress = PreprocessingProgress(100)
        callback = Mock()
        progress.add_callback(callback)

        progress.start()
        progress.update(10, 2)

        assert progress.processed_items == 10
        assert progress.failed_items == 2
        assert callback.call_count == 2  # start() + update()

    def test_complete_progress(self):
        """Test completing progress."""
        progress = PreprocessingProgress(100)
        callback = Mock()
        progress.add_callback(callback)

        progress.start()
        progress.complete()

        assert progress.processed_items == 100
        assert callback.call_count == 2  # start() + complete()

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        progress = PreprocessingProgress(100)

        # No progress
        assert progress.percentage == 0.0

        # Partial progress
        progress.update(25)
        assert progress.percentage == 0.25

        # Complete
        progress.update(75)
        assert progress.percentage == 1.0

    def test_percentage_zero_total(self):
        """Test percentage with zero total items."""
        progress = PreprocessingProgress(0)
        assert progress.percentage == 0.0

    def test_success_rate(self):
        """Test success rate calculation."""
        progress = PreprocessingProgress(100)

        # No items processed
        assert progress.success_rate == 1.0

        # Some items processed with failures
        progress.update(10, 2)
        assert progress.success_rate == 0.8  # (10-2)/10

        # All successful
        progress.update(10, 0)
        assert progress.success_rate == 0.9  # (20-2)/20

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        progress = PreprocessingProgress()

        # Not started
        assert progress.elapsed_time == 0.0

        # Started
        progress.start()
        assert progress.elapsed_time >= 0.0

    def test_callback_error_handling(self):
        """Test that callback errors don't break progress."""
        progress = PreprocessingProgress()

        def bad_callback(_):
            raise ValueError("Callback error")

        progress.add_callback(bad_callback)

        # Should not raise exception
        progress.start()
        progress.update(10)


class MockPreprocessor(DataPreprocessor):
    """Mock preprocessor for testing base functionality."""

    def __init__(self, required_fields=None, config_keys=None):
        super().__init__()
        self._required_fields = required_fields or ["input_text"]
        self._config_keys = config_keys if config_keys is not None else ["test_option"]

    async def process(self, samples, config):
        """Simple mock processing that adds metadata."""
        processed = []
        for sample in samples:
            new_sample = sample.model_copy()
            new_sample.metadata["processed"] = True
            new_sample.metadata.update(config)
            processed.append(new_sample)
        return processed

    def get_required_fields(self):
        return self._required_fields

    def validate_config(self, config):
        warnings = []
        if "deprecated_option" in config:
            warnings.append("deprecated_option is deprecated")
        return warnings

    def get_supported_config_keys(self):
        return self._config_keys


class TestDataPreprocessor:
    """Test DataPreprocessor base class functionality."""

    def test_init_default_name(self):
        """Test preprocessor initialization with default name."""
        preprocessor = MockPreprocessor()
        assert preprocessor.name == "MockPreprocessor"

    def test_init_custom_name(self):
        """Test preprocessor initialization with custom name."""
        preprocessor = MockPreprocessor()
        preprocessor.name = "CustomName"
        assert preprocessor.name == "CustomName"

    @pytest.mark.asyncio
    async def test_process_basic(self):
        """Test basic processing functionality."""
        preprocessor = MockPreprocessor()
        samples = [
            DatasetSample(input_text="test 1", label="ATTACK"),
            DatasetSample(input_text="test 2", label="BENIGN"),
        ]
        config = {"test_option": "value"}

        result = await preprocessor.process(samples, config)

        assert len(result) == 2
        assert result[0].metadata["processed"] is True
        assert result[0].metadata["test_option"] == "value"

    def test_process_single(self):
        """Test single sample processing."""
        preprocessor = MockPreprocessor()
        sample = DatasetSample(input_text="test", label="ATTACK")
        config = {"test_option": "value"}

        result = preprocessor.process_single(sample, config)

        assert result.metadata["processed"] is True
        assert result.metadata["test_option"] == "value"

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing with progress reporting."""
        preprocessor = MockPreprocessor()
        samples = [DatasetSample(input_text=f"test {i}", label="ATTACK") for i in range(10)]
        config = {"test_option": "value"}

        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress.processed_items)

        result = await preprocessor.process_batch(
            samples, config, batch_size=3, progress_callback=progress_callback
        )

        assert len(result) == 10
        assert all(sample.metadata["processed"] for sample in result)
        # Should have progress updates
        assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_process_batch_empty(self):
        """Test batch processing with empty sample list."""
        preprocessor = MockPreprocessor()

        result = await preprocessor.process_batch([], {})

        assert result == []

    @pytest.mark.asyncio
    async def test_process_batch_warnings(self):
        """Test batch processing with configuration warnings."""
        preprocessor = MockPreprocessor()
        samples = [DatasetSample(input_text="test", label="ATTACK")]
        config = {"deprecated_option": "value"}

        # Should not raise exception, just log warnings
        result = await preprocessor.process_batch(samples, config)

        assert len(result) == 1

    def test_validate_samples_success(self):
        """Test successful sample validation."""
        preprocessor = MockPreprocessor(required_fields=["input_text", "label"])
        samples = [
            DatasetSample(input_text="test 1", label="ATTACK"),
            DatasetSample(input_text="test 2", label="BENIGN"),
        ]

        # Should not raise exception
        preprocessor._validate_samples(samples)

    def test_validate_samples_missing_field(self):
        """Test sample validation with missing fields."""
        preprocessor = MockPreprocessor(required_fields=["input_text", "nonexistent_field"])
        samples = [DatasetSample(input_text="test", label="ATTACK")]

        with pytest.raises(ValueError, match="missing required fields"):
            preprocessor._validate_samples(samples)

    def test_validate_config_keys_warnings(self):
        """Test configuration key validation."""
        preprocessor = MockPreprocessor(config_keys=["valid_key"])
        config = {"valid_key": "value", "unknown_key": "value"}

        warnings = preprocessor._validate_config_keys(config)

        assert len(warnings) == 1
        assert "unknown_key" in warnings[0]

    def test_validate_config_keys_no_supported_keys(self):
        """Test config validation with no supported keys defined."""
        preprocessor = MockPreprocessor(config_keys=[])
        config = {"any_key": "value"}

        warnings = preprocessor._validate_config_keys(config)

        assert len(warnings) == 0

    def test_str_representation(self):
        """Test string representation of preprocessor."""
        preprocessor = MockPreprocessor()
        str_repr = str(preprocessor)

        assert "MockPreprocessor" in str_repr
        assert "input_text" in str_repr

    def test_repr_representation(self):
        """Test detailed representation of preprocessor."""
        preprocessor = MockPreprocessor()
        repr_str = repr(preprocessor)

        assert "MockPreprocessor" in repr_str
        assert "required_fields" in repr_str
        assert "supported_config" in repr_str


class TestPreprocessingUtilities:
    """Test PreprocessingUtilities functionality."""

    def test_clean_text_default(self):
        """Test text cleaning with default options."""
        text = "  Hello\n\tWorld!  \r\n  "
        result = PreprocessingUtilities.clean_text(text)

        assert result == "Hello World!"

    def test_clean_text_preserve_newlines(self):
        """Test text cleaning preserving newlines."""
        text = "Line 1\n  Line 2  \n\nLine 3"
        result = PreprocessingUtilities.clean_text(text, {"preserve_newlines": True})

        assert result == "Line 1\nLine 2\n\nLine 3"

    def test_clean_text_lowercase(self):
        """Test text cleaning with lowercase option."""
        text = "Hello WORLD"
        result = PreprocessingUtilities.clean_text(text, {"lowercase": True})

        assert result == "hello world"

    def test_clean_text_unicode_normalization(self):
        """Test unicode normalization."""
        text = "café"  # Contains combining characters
        result = PreprocessingUtilities.clean_text(text, {"normalize_unicode": True})

        assert result == "café"

    def test_clean_text_empty_input(self):
        """Test cleaning empty or None input."""
        assert PreprocessingUtilities.clean_text("") == ""
        assert PreprocessingUtilities.clean_text(None) == ""
        assert PreprocessingUtilities.clean_text(123) == ""

    def test_normalize_timestamp_valid_formats(self):
        """Test timestamp parsing with valid formats."""
        test_cases = [
            "2024-01-15 10:30:00",
            "2024-01-15T10:30:00",
            "2024-01-15T10:30:00Z",
            "2024/01/15 10:30:00",
            "15/01/2024 10:30:00",
            "2024-01-15",
        ]

        for timestamp_str in test_cases:
            result = PreprocessingUtilities.normalize_timestamp(timestamp_str)
            assert isinstance(result, datetime)

    def test_normalize_timestamp_invalid_format(self):
        """Test timestamp parsing with invalid format."""
        result = PreprocessingUtilities.normalize_timestamp("invalid timestamp")
        assert result is None

    def test_normalize_timestamp_empty_input(self):
        """Test timestamp parsing with empty input."""
        assert PreprocessingUtilities.normalize_timestamp("") is None
        assert PreprocessingUtilities.normalize_timestamp(None) is None

    def test_normalize_timestamp_custom_formats(self):
        """Test timestamp parsing with custom formats."""
        timestamp_str = "15-01-2024 10:30"
        custom_formats = ["%d-%m-%Y %H:%M"]

        result = PreprocessingUtilities.normalize_timestamp(timestamp_str, custom_formats)

        assert isinstance(result, datetime)
        assert result.day == 15
        assert result.month == 1

    def test_extract_ip_addresses_ipv4(self):
        """Test IPv4 address extraction."""
        text = "Server at 192.168.1.1 and 10.0.0.1 are down. Check 256.1.1.1 too."
        result = PreprocessingUtilities.extract_ip_addresses(text)

        # 256.1.1.1 is invalid IP, should not be included
        assert "192.168.1.1" in result
        assert "10.0.0.1" in result
        assert "256.1.1.1" not in result

    def test_extract_ip_addresses_exclude_private(self):
        """Test IP address extraction excluding private addresses."""
        text = "Public: 8.8.8.8, Private: 192.168.1.1"
        result = PreprocessingUtilities.extract_ip_addresses(text, include_private=False)

        assert "8.8.8.8" in result
        assert "192.168.1.1" not in result

    def test_extract_urls_basic(self):
        """Test URL extraction."""
        text = "Visit https://example.com or http://test.org for more info."
        result = PreprocessingUtilities.extract_urls(text)

        assert "https://example.com" in result
        assert "http://test.org" in result

    def test_extract_urls_complex(self):
        """Test URL extraction with complex URLs."""
        text = "API: https://api.example.com/v1/users?id=123&type=admin#section"
        result = PreprocessingUtilities.extract_urls(text)

        assert len(result) == 1
        assert "https://api.example.com/v1/users?id=123&type=admin#section" in result

    def test_extract_urls_no_validation(self):
        """Test URL extraction without validation."""
        text = "Malformed: http://bad..url"
        result = PreprocessingUtilities.extract_urls(text, validate=False)

        assert "http://bad..url" in result

    def test_extract_email_addresses(self):
        """Test email address extraction."""
        text = "Contact admin@example.com or support@test.org for help."
        result = PreprocessingUtilities.extract_email_addresses(text)

        assert "admin@example.com" in result
        assert "support@test.org" in result

    def test_extract_hashes_md5(self):
        """Test MD5 hash extraction."""
        text = "File hash: 5d41402abc4b2a76b9719d911017c592"
        result = PreprocessingUtilities.extract_hashes(text, ["md5"])

        assert "5d41402abc4b2a76b9719d911017c592" in result["md5"]

    def test_extract_hashes_sha256(self):
        """Test SHA256 hash extraction."""
        text = "SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        result = PreprocessingUtilities.extract_hashes(text, ["sha256"])

        assert (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" in result["sha256"]
        )

    def test_extract_hashes_multiple_types(self):
        """Test extraction of multiple hash types."""
        text = """
        MD5: 5d41402abc4b2a76b9719d911017c592
        SHA1: aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
        SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        """
        result = PreprocessingUtilities.extract_hashes(text)

        assert len(result["md5"]) == 1
        assert len(result["sha1"]) == 1
        assert len(result["sha256"]) == 1

    def test_extract_domains(self):
        """Test domain extraction."""
        text = "Visit example.com, test.org, and subdomain.example.net."
        result = PreprocessingUtilities.extract_domains(text)

        assert "example.com" in result
        assert "test.org" in result
        assert "subdomain.example.net" in result

    def test_extract_domains_exclude_ips(self):
        """Test domain extraction excluding IPs."""
        text = "Domains: example.com, 192.168.1.1"
        result = PreprocessingUtilities.extract_domains(text, exclude_ips=True)

        assert "example.com" in result
        assert "192.168.1.1" not in result

    def test_normalize_attack_labels_sql_injection(self):
        """Test attack label normalization for SQL injection variants."""
        test_cases = [
            "SQL Injection",
            "sqli",
            "sql-injection",
            "SQL_INJECTION",
        ]

        for label in test_cases:
            result = PreprocessingUtilities.normalize_attack_labels(label)
            assert result == "sql_injection"

    def test_normalize_attack_labels_xss(self):
        """Test attack label normalization for XSS variants."""
        test_cases = [
            "XSS",
            "cross-site-scripting",
            "Cross Site Scripting",
        ]

        for label in test_cases:
            result = PreprocessingUtilities.normalize_attack_labels(label)
            assert result == "cross_site_scripting"

    def test_normalize_attack_labels_custom_mappings(self):
        """Test attack label normalization with custom mappings."""
        custom_mappings = {"custom attack": "custom_attack_type"}

        result = PreprocessingUtilities.normalize_attack_labels("custom attack", custom_mappings)

        assert result == "custom_attack_type"

    def test_normalize_attack_labels_unknown(self):
        """Test attack label normalization for unknown types."""
        result = PreprocessingUtilities.normalize_attack_labels("Unknown Attack Type")
        assert result == "unknown_attack_type"

    def test_normalize_attack_labels_empty(self):
        """Test attack label normalization with empty input."""
        assert PreprocessingUtilities.normalize_attack_labels("") == ""
        assert PreprocessingUtilities.normalize_attack_labels(None) == ""

    @pytest.mark.asyncio
    async def test_process_batch_utility(self):
        """Test utility batch processing function."""
        samples = [DatasetSample(input_text=f"test {i}", label="ATTACK") for i in range(5)]

        def add_metadata(sample):
            new_sample = sample.model_copy()
            new_sample.metadata["processed"] = True
            return new_sample

        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress.processed_items)

        result = await PreprocessingUtilities.process_batch(
            samples, add_metadata, batch_size=2, progress_callback=progress_callback
        )

        assert len(result) == 5
        assert all(sample.metadata.get("processed") for sample in result)
        assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_process_batch_utility_with_failures(self):
        """Test utility batch processing with some failures."""
        samples = [DatasetSample(input_text=f"test {i}", label="ATTACK") for i in range(5)]

        def failing_processor(sample):
            if "test 2" in sample.input_text:
                raise ValueError("Processing failed")
            new_sample = sample.model_copy()
            new_sample.metadata["processed"] = True
            return new_sample

        result = await PreprocessingUtilities.process_batch(
            samples, failing_processor, batch_size=2
        )

        # Should have 4 samples (one failed)
        assert len(result) == 4
        assert all(sample.metadata.get("processed") for sample in result)

    def test_extract_features_comprehensive(self):
        """Test comprehensive feature extraction."""
        text = """
        Email: admin@example.com
        URL: https://malware.example.com/payload
        IP: 192.168.1.100
        Domain: malicious-site.com
        Hash: 5d41402abc4b2a76b9719d911017c592
        """

        result = PreprocessingUtilities.extract_features(text)

        assert "admin@example.com" in result["email_addresses"]
        assert "https://malware.example.com/payload" in result["urls"]
        assert "192.168.1.100" in result["ip_addresses"]
        assert "malicious-site.com" in result["domains"]
        assert "5d41402abc4b2a76b9719d911017c592" in result["hashes"]["md5"]

    def test_extract_features_selective(self):
        """Test selective feature extraction."""
        text = "Email: admin@example.com, URL: https://example.com"

        result = PreprocessingUtilities.extract_features(text, feature_types=["emails"])

        assert "email_addresses" in result
        assert "urls" not in result

    def test_extract_features_empty_text(self):
        """Test feature extraction with empty text."""
        result = PreprocessingUtilities.extract_features("")
        assert result == {}

        result = PreprocessingUtilities.extract_features(None)
        assert result == {}

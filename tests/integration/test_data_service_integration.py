"""
Integration tests for the enhanced Data Service.

This module tests the complete data service functionality including auto-registration
of loaders, dataset discovery, quality validation, statistics computation, and
concurrent operations.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from benchmark.core.config import DatasetConfig
from benchmark.core.exceptions import DataLoadingError
from benchmark.data.models import DataQualityReport, DatasetStatistics
from benchmark.interfaces.data_interfaces import DataSource
from benchmark.services.data_service import DataService


class TestDataServiceIntegration:
    """Integration tests for the enhanced Data Service."""

    @pytest_asyncio.fixture
    async def data_service(self):
        """Create a data service instance for testing."""
        service = DataService(cache_max_size=10, cache_max_memory_mb=64, cache_ttl=300)
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.fixture
    def sample_dataset_files(self):
        """Create sample dataset files for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create a sample JSON dataset
        json_data = [
            {"text": "This is a normal email", "label": "BENIGN"},
            {
                "text": "URGENT: Your account is suspended!",
                "label": "ATTACK",
                "attack_type": "phishing",
            },
            {"text": "Meeting tomorrow at 2 PM", "label": "BENIGN"},
            {
                "text": "Click here to verify your account",
                "label": "ATTACK",
                "attack_type": "phishing",
            },
            {"text": "", "label": "BENIGN"},  # Empty sample for quality testing
        ]

        json_file = temp_dir / "sample_dataset.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f)

        # Create a JSONL dataset
        jsonl_file = temp_dir / "sample_dataset.jsonl"
        with open(jsonl_file, "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")

        # Create a CSV dataset
        csv_file = temp_dir / "sample_dataset.csv"
        with open(csv_file, "w") as f:
            f.write("text,label,attack_type\n")
            for item in json_data:
                attack_type = item.get("attack_type", "")
                f.write(f'"{item["text"]}",{item["label"]},{attack_type}\n')

        return {
            "temp_dir": temp_dir,
            "json_file": json_file,
            "jsonl_file": jsonl_file,
            "csv_file": csv_file,
            "expected_samples": len(json_data),
        }

    @pytest.mark.asyncio
    async def test_auto_registration_of_loaders(self, data_service):
        """Test that all available loaders are auto-registered."""
        registered_loaders = await data_service.get_registered_loaders()

        # Should at least have local loader registered
        assert "local" in registered_loaders
        assert registered_loaders["local"]["class_name"] == "LocalDataLoader"
        assert "json" in registered_loaders["local"]["supported_formats"]

        # Check that the service knows about available loaders
        assert len(registered_loaders) >= 1

    @pytest.mark.asyncio
    async def test_dataset_discovery(self, data_service, sample_dataset_files):
        """Test dataset discovery functionality."""
        temp_dir = sample_dataset_files["temp_dir"]

        # Discover datasets in the temp directory
        discovered = await data_service.discover_datasets([str(temp_dir)])

        # Should discover dataset files (may vary based on duplicates)
        assert len(discovered) >= 1

        # Check that datasets were added to registry
        available_datasets = await data_service.list_available_datasets()
        assert len(available_datasets) >= 1

        # Verify discovery metadata
        json_dataset = next((d for d in discovered if d.name == "sample_dataset"), None)
        assert json_dataset is not None
        assert json_dataset.source == DataSource.LOCAL
        assert json_dataset.metadata.get("auto_discovered") is True

    @pytest.mark.asyncio
    async def test_dataset_loading_and_quality_validation(self, data_service, sample_dataset_files):
        """Test dataset loading and quality validation."""
        json_file = sample_dataset_files["json_file"]
        expected_samples = sample_dataset_files["expected_samples"]

        # Create config and load dataset
        config = DatasetConfig(
            name="test_dataset", path=str(json_file), source="local", format="json"
        )

        dataset = await data_service.load_dataset(config)
        assert dataset.size == expected_samples

        # Validate dataset quality
        quality_report = await data_service.validate_dataset_quality("test_dataset")

        assert isinstance(quality_report, DataQualityReport)
        assert quality_report.dataset_id == "test_dataset"
        assert quality_report.total_samples == expected_samples
        assert quality_report.empty_samples == 1  # One empty sample in test data
        # Since empty sample is skipped in analysis, ratios are based on valid samples only
        # Out of 5 total: 1 empty (skipped), 2 attack, 2 benign with valid labels
        assert quality_report.attack_ratio == 0.4  # 2 out of 5 total samples
        assert quality_report.benign_ratio == 0.4  # 2 out of 5 total samples
        assert 0.0 <= quality_report.quality_score <= 1.0

        # Test cached quality report
        cached_report = await data_service.get_dataset_quality_report("test_dataset")
        assert cached_report.dataset_id == quality_report.dataset_id
        assert cached_report.total_samples == quality_report.total_samples

    @pytest.mark.asyncio
    async def test_dataset_statistics_computation(self, data_service, sample_dataset_files):
        """Test comprehensive dataset statistics computation."""
        json_file = sample_dataset_files["json_file"]

        # Create config and load dataset
        config = DatasetConfig(
            name="stats_test_dataset", path=str(json_file), source="local", format="json"
        )

        dataset = await data_service.load_dataset(config)

        # Compute statistics
        stats = await data_service.compute_dataset_statistics("stats_test_dataset")

        assert isinstance(stats, DatasetStatistics)
        assert stats.dataset_id == "stats_test_dataset"
        assert stats.total_samples == dataset.size
        assert stats.attack_samples == 2
        assert stats.benign_samples == 3
        assert stats.attack_ratio == 2 / 5
        assert stats.benign_ratio == 3 / 5
        assert 0.0 <= stats.label_balance <= 1.0

        # Check content statistics
        assert "mean" in stats.content_length_stats
        assert "std" in stats.content_length_stats
        assert "min" in stats.content_length_stats
        assert "max" in stats.content_length_stats

        assert "mean" in stats.word_count_stats
        assert "ascii" in stats.detected_languages or "non_ascii" in stats.detected_languages

        # Test attack types counting
        assert "phishing" in stats.attack_types
        assert stats.attack_types["phishing"] == 2

        # Test cached statistics
        cached_stats = await data_service.get_dataset_statistics("stats_test_dataset")
        assert cached_stats.dataset_id == stats.dataset_id
        assert cached_stats.total_samples == stats.total_samples

    @pytest.mark.asyncio
    async def test_concurrent_dataset_loading(self, data_service, sample_dataset_files):
        """Test concurrent dataset loading."""
        json_file = sample_dataset_files["json_file"]
        jsonl_file = sample_dataset_files["jsonl_file"]
        csv_file = sample_dataset_files["csv_file"]

        # Create configs for different datasets
        configs = [
            DatasetConfig(
                name="concurrent_json", path=str(json_file), source="local", format="json"
            ),
            DatasetConfig(
                name="concurrent_jsonl", path=str(jsonl_file), source="local", format="jsonl"
            ),
            DatasetConfig(name="concurrent_csv", path=str(csv_file), source="local", format="csv"),
        ]

        # Load datasets concurrently
        tasks = [data_service.load_dataset(config) for config in configs]
        datasets = await asyncio.gather(*tasks)

        # Verify all datasets loaded successfully
        assert len(datasets) == 3
        for dataset in datasets:
            assert dataset.size > 0

        # Test concurrent quality validation
        quality_tasks = [
            data_service.validate_dataset_quality("concurrent_json"),
            data_service.validate_dataset_quality("concurrent_jsonl"),
            data_service.validate_dataset_quality("concurrent_csv"),
        ]

        quality_reports = await asyncio.gather(*quality_tasks)
        assert len(quality_reports) == 3
        for report in quality_reports:
            assert isinstance(report, DataQualityReport)
            assert report.total_samples > 0

    @pytest.mark.asyncio
    async def test_caching_across_loaders(self, data_service, sample_dataset_files):
        """Test caching behavior across different loaders."""
        json_file = sample_dataset_files["json_file"]

        config = DatasetConfig(
            name="cache_test", path=str(json_file), source="local", format="json"
        )

        # Load dataset first time
        dataset1 = await data_service.load_dataset(config)

        # Check cache stats
        cache_stats = await data_service.get_cache_stats()
        assert cache_stats["cache_size"] > 0

        # Load same dataset again (should use cache)
        dataset2 = await data_service.load_dataset(config)

        # Should be the same instance from cache
        assert dataset1.dataset_id == dataset2.dataset_id
        assert dataset1.size == dataset2.size

        # Test quality report caching
        report1 = await data_service.validate_dataset_quality("cache_test")
        report2 = await data_service.get_dataset_quality_report("cache_test")

        assert report1.dataset_id == report2.dataset_id
        assert report1.quality_score == report2.quality_score

        # Test statistics caching
        stats1 = await data_service.compute_dataset_statistics("cache_test")
        stats2 = await data_service.get_dataset_statistics("cache_test")

        assert stats1.dataset_id == stats2.dataset_id
        assert stats1.total_samples == stats2.total_samples

    @pytest.mark.asyncio
    async def test_quality_validation_edge_cases(self, data_service):
        """Test quality validation with edge cases."""
        # Create a dataset with quality issues
        temp_dir = Path(tempfile.mkdtemp())

        problematic_data = [
            {"text": "Normal sample", "label": "BENIGN"},
            {"text": "", "label": "BENIGN"},  # Empty content
            {"text": "Sample without label"},  # Missing label
            {"text": "Invalid label sample", "label": "INVALID"},  # Invalid label
            {"text": "Normal sample", "label": "BENIGN"},  # Duplicate content
            {"text": "Attack sample", "label": "ATTACK", "attack_type": "malware"},
        ]

        problem_file = temp_dir / "problematic.json"
        with open(problem_file, "w") as f:
            json.dump(problematic_data, f)

        config = DatasetConfig(
            name="problematic_dataset", path=str(problem_file), source="local", format="json"
        )

        await data_service.load_dataset(config)
        quality_report = await data_service.validate_dataset_quality("problematic_dataset")

        # Should detect quality issues
        assert quality_report.has_issues
        assert quality_report.empty_samples >= 1
        assert quality_report.missing_labels >= 1
        assert quality_report.invalid_labels >= 1
        assert quality_report.duplicate_samples >= 1
        assert len(quality_report.validation_errors) > 0
        assert quality_report.quality_score < 1.0
        assert quality_report.clean_sample_ratio < 1.0

    @pytest.mark.asyncio
    async def test_dataset_not_found_error(self, data_service):
        """Test error handling for non-existent datasets."""
        with pytest.raises(DataLoadingError):
            await data_service.get_dataset_quality_report("non_existent_dataset")

        with pytest.raises(DataLoadingError):
            await data_service.get_dataset_statistics("non_existent_dataset")

    @pytest.mark.asyncio
    async def test_unified_interface_across_sources(self, data_service, sample_dataset_files):
        """Test that the unified interface works consistently across different sources."""
        json_file = sample_dataset_files["json_file"]

        config = DatasetConfig(
            name="unified_test", path=str(json_file), source="local", format="json"
        )

        # Test the complete workflow
        dataset = await data_service.load_dataset(config)
        quality_report = await data_service.validate_dataset_quality("unified_test")
        statistics = await data_service.compute_dataset_statistics("unified_test")

        # All operations should work consistently
        assert dataset.size == quality_report.total_samples == statistics.total_samples
        assert dataset.dataset_id == quality_report.dataset_id == statistics.dataset_id

        # Statistics should include quality report
        assert statistics.quality_report is not None
        assert statistics.quality_report.dataset_id == quality_report.dataset_id

    @pytest.mark.asyncio
    async def test_service_health_with_enhanced_features(self, data_service):
        """Test service health check includes enhanced features."""
        health = await data_service.health_check()

        assert health.status == data_service.status.value
        assert "cache_status" in health.checks
        assert "registered_loaders" in health.checks
        assert "loaded_datasets" in health.checks

        # Should report registered loaders count
        assert health.checks["registered_loaders"] >= 1

    @pytest.mark.asyncio
    async def test_memory_management_under_load(self, data_service):
        """Test memory management under load with multiple datasets."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create multiple datasets
        datasets_info = []
        for i in range(5):
            data = [
                {
                    "text": f"Sample {j} from dataset {i}",
                    "label": "BENIGN" if j % 2 == 0 else "ATTACK",
                }
                for j in range(100)  # 100 samples per dataset
            ]

            file_path = temp_dir / f"dataset_{i}.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

            datasets_info.append(
                {
                    "name": f"memory_test_{i}",
                    "path": str(file_path),
                }
            )

        # Load all datasets
        for info in datasets_info:
            config = DatasetConfig(
                name=info["name"], path=info["path"], source="local", format="json"
            )
            await data_service.load_dataset(config)

        # Check cache behavior under load
        cache_stats = await data_service.get_cache_stats()
        assert cache_stats["cache_size"] > 0
        assert cache_stats["memory_usage_mb"] > 0

        # Should be able to generate statistics for all
        for info in datasets_info:
            stats = await data_service.compute_dataset_statistics(info["name"])
            assert stats.total_samples == 100


@pytest.mark.asyncio
async def test_service_initialization_and_shutdown():
    """Test service initialization and shutdown process."""
    service = DataService()

    # Test initialization
    init_response = await service.initialize()
    assert init_response.success is True
    assert "registered_loaders" in init_response.data

    # Test shutdown
    shutdown_response = await service.shutdown()
    assert shutdown_response.success is True

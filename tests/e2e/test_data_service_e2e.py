"""
End-to-end tests for the complete data service functionality.

This module provides comprehensive end-to-end tests that validate the entire
data service functionality with real-world scenarios including multiple data
sources, realistic dataset sizes, error recovery, and performance under load.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio

from benchmark.core.config import DatasetConfig
from benchmark.core.exceptions import DataLoadingError
from benchmark.data.models import DataQualityReport, DatasetStatistics
from benchmark.services.data_service import DataService


class TestDataServiceE2E:
    """Comprehensive end-to-end tests for the data service."""

    @pytest_asyncio.fixture
    async def data_service(self):
        """Create a fully optimized data service instance for E2E testing."""
        service = DataService(
            cache_max_size=100,
            cache_max_memory_mb=512,
            cache_ttl=600,
            enable_compression=True,
            enable_hardware_optimization=True,
        )
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.fixture
    def realistic_datasets(self):
        """Create realistic cybersecurity datasets for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        datasets = {}

        # Create UNSW-NB15 style network traffic dataset
        unsw_data = []
        for i in range(10000):  # 10K samples for realistic size
            srcip = f"192.168.{(i // 255) % 255 + 1}.{i % 255 + 1}"
            dstip = f"10.{(i // 1000) % 255}.{(i // 100) % 255}.{(i + 50) % 255 + 1}"
            srcport = 1024 + (i % 30000)
            dstport = 80 + (i % 1000)
            proto = "tcp" if i % 2 == 0 else "udp"
            service = "http" if i % 3 == 0 else "dns" if i % 3 == 1 else "ssl"

            sample = {
                "srcip": srcip,
                "dstip": dstip,
                "srcport": srcport,
                "dstport": dstport,
                "proto": proto,
                "dur": round((i % 1000) / 100.0, 3),
                "sbytes": 1024 + (i % 10000),
                "dbytes": 512 + (i % 5000),
                "sttl": 64,
                "dttl": 64,
                "sloss": 0,
                "dloss": 0,
                "service": service,
                "sload": round((i % 100) / 10.0, 2),
                "dload": round((i % 50) / 10.0, 2),
                "spkts": 10 + (i % 100),
                "dpkts": 8 + (i % 80),
                "swin": 8192,
                "dwin": 8192,
                "stcpb": 0,
                "dtcpb": 0,
                "smeansz": 64.0,
                "dmeansz": 32.0,
                "trans_depth": 1,
                "res_bdy_len": 0,
                "sjit": 0.01,
                "djit": 0.01,
                "stime": f"2024-01-{(i % 28) + 1:02d} {(i % 24):02d}:{(i % 60):02d}:{(i % 60):02d}",
                "ltime": f"2024-01-{(i % 28) + 1:02d} {(i % 24):02d}:{(i % 60):02d}:{(i % 60 + 1):02d}",
                "sintpkt": 0.001,
                "dintpkt": 0.001,
                "tcprtt": 0.1,
                "synack": 0.05,
                "ackdat": 0.02,
                "is_sm_ips_ports": 0,
                "ct_state_ttl": 1,
                "ct_flw_http_mthd": 1 if i % 3 == 0 else 0,
                "is_ftp_login": 0,
                "ct_ftp_cmd": 0,
                "ct_srv_src": 1,
                "ct_srv_dst": 1,
                "ct_dst_ltm": 1,
                "ct_src_ltm": 1,
                "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 1,
                "ct_dst_src_ltm": 1,
                "text": f"Network flow {i}: {srcip}:{srcport} -> {dstip}:{dstport} [{proto}] {service} dur:{round((i % 1000) / 100.0, 3)}s",
                "label": "ATTACK" if i % 5 == 0 else "BENIGN",
                "attack_type": (
                    "DoS"
                    if i % 10 == 0
                    else "Probe"
                    if i % 15 == 0
                    else "R2L"
                    if i % 20 == 0
                    else "U2R"
                    if i % 25 == 0
                    else None
                )
                if i % 5 == 0
                else None,
            }
            unsw_data.append(sample)

        unsw_file = temp_dir / "unsw_nb15_sample.json"
        with open(unsw_file, "w") as f:
            json.dump(unsw_data, f)
        datasets["unsw_nb15"] = {"file": unsw_file, "size": len(unsw_data)}

        # Create phishing email dataset
        phishing_data = []
        for i in range(5000):  # 5K samples
            sample = {
                "subject": f"Subject {i}: {'URGENT: Verify your account!' if i % 4 == 0 else f'Meeting scheduled for {i}'}",
                "sender": f"{'noreply@suspicious-bank.com' if i % 4 == 0 else f'colleague{i}@company.com'}",
                "body": (
                    f"URGENT: Your account will be suspended! Click here to verify: http://fake-bank{i}.com/verify"
                    if i % 4 == 0
                    else f"Hi, this is a normal email about topic {i}. Please find the attachment."
                ),
                "has_links": 1 if i % 4 == 0 else 0,
                "has_attachments": 1 if i % 6 == 0 else 0,
                "sender_reputation": "suspicious" if i % 4 == 0 else "trusted",
                "domain_age": 1 if i % 4 == 0 else 365 + (i % 1000),
                "text": "",  # Will be set after sample creation
                "label": "ATTACK" if i % 4 == 0 else "BENIGN",
                "attack_type": "phishing" if i % 4 == 0 else None,
                "confidence": 0.95 if i % 4 == 0 else 0.1,
            }
            # Set text field after sample is created
            sample["text"] = (
                f"Subject: {sample['subject']} From: {sample['sender']} Body: {sample['body']}"
            )
            phishing_data.append(sample)

        phishing_file = temp_dir / "phishing_emails.json"
        with open(phishing_file, "w") as f:
            json.dump(phishing_data, f)
        datasets["phishing"] = {"file": phishing_file, "size": len(phishing_data)}

        # Create web traffic logs
        web_logs = []
        for i in range(7500):  # 7.5K samples
            sample = {
                "timestamp": f"2024-01-0{(i % 9) + 1} {(i % 24):02d}:{(i % 60):02d}:{(i % 60):02d}",
                "client_ip": f"192.168.{(i % 10) + 1}.{(i % 255) + 1}",
                "method": "GET" if i % 3 == 0 else "POST" if i % 3 == 1 else "PUT",
                "url": (
                    "/admin/login?user=admin&pass=admin123"
                    if i % 10 == 0
                    else f"/api/v1/users/{i}/delete"
                    if i % 15 == 0
                    else "/../../../etc/passwd"
                    if i % 20 == 0
                    else f"/images/photo{i}.jpg"
                ),
                "status_code": 200
                if i % 8 != 0
                else 404
                if i % 8 == 1
                else 403
                if i % 8 == 2
                else 500,
                "response_size": 1024 + (i % 50000),
                "user_agent": (
                    "sqlmap/1.0"
                    if i % 20 == 0
                    else "Nikto/2.1.6"
                    if i % 25 == 0
                    else "Mozilla/5.0 (compatible; bot)"
                ),
                "referer": f"http://example{i % 100}.com" if i % 5 == 0 else "",
                "text": "",  # Will be set after creation
                "label": "BENIGN",  # Will be set after creation
                "attack_type": None,  # Will be set after creation
            }
            # Set text field after sample is created
            sample["text"] = f"HTTP {sample['method']} {sample['url']} from {sample['client_ip']}"

            # Set label and attack type based on content
            if any(
                [
                    "admin" in sample["url"],
                    "../" in sample["url"],
                    "delete" in sample["url"],
                    "sqlmap" in sample["user_agent"],
                    "Nikto" in sample["user_agent"],
                ]
            ):
                sample["label"] = "ATTACK"
                if "admin" in sample["url"] or "sqlmap" in sample["user_agent"]:
                    sample["attack_type"] = "SQLi"
                elif "../" in sample["url"]:
                    sample["attack_type"] = "Directory_Traversal"
                elif "delete" in sample["url"] or "Nikto" in sample["user_agent"]:
                    sample["attack_type"] = "Web_Attack"

            web_logs.append(sample)

        web_file = temp_dir / "web_traffic_logs.jsonl"
        with open(web_file, "w") as f:
            for log in web_logs:
                f.write(json.dumps(log) + "\n")
        datasets["web_logs"] = {"file": web_file, "size": len(web_logs)}

        # Create malware dataset (CSV format)
        malware_data = [
            "filename,size,entropy,pe_sections,imports_count,exports_count,strings_count,label,attack_type,family"
        ]
        for i in range(3000):  # 3K samples
            row = (
                f"file_{i}.exe,"
                f"{10000 + (i % 1000000)},"
                f"{round(7.5 + (i % 100) / 100, 2)},"
                f"{3 + (i % 5)},"
                f"{50 + (i % 200)},"
                f"{0 if i % 3 == 0 else 1 + (i % 10)},"
                f"{100 + (i % 5000)},"
                f"{'ATTACK' if i % 3 == 0 else 'BENIGN'},"
                f"{'malware' if i % 3 == 0 else ''},"
                f"{'Trojan' if i % 9 == 0 else 'Worm' if i % 12 == 0 else 'Ransomware' if i % 15 == 0 else ''}"
            )
            malware_data.append(row)

        malware_file = temp_dir / "malware_samples.csv"
        with open(malware_file, "w") as f:
            f.write("\n".join(malware_data))
        datasets["malware"] = {"file": malware_file, "size": len(malware_data) - 1}

        return {"temp_dir": temp_dir, "datasets": datasets}

    @pytest.fixture
    def corrupted_datasets(self):
        """Create datasets with various types of corruption for error recovery testing."""
        temp_dir = Path(tempfile.mkdtemp())
        datasets = {}

        # Invalid JSON
        invalid_json_file = temp_dir / "invalid.json"
        with open(invalid_json_file, "w") as f:
            f.write('{"incomplete": "json file without closing brace"')
        datasets["invalid_json"] = invalid_json_file

        # Missing fields
        missing_fields_data = [
            {"text": "Sample 1"},  # Missing label
            {"label": "ATTACK"},  # Missing text
            {"text": "Sample 3", "label": "INVALID_LABEL"},  # Invalid label
        ]
        missing_fields_file = temp_dir / "missing_fields.json"
        with open(missing_fields_file, "w") as f:
            json.dump(missing_fields_data, f)
        datasets["missing_fields"] = missing_fields_file

        # Empty file
        empty_file = temp_dir / "empty.json"
        empty_file.touch()
        datasets["empty"] = empty_file

        # Non-existent file path
        datasets["non_existent"] = temp_dir / "does_not_exist.json"

        # Extremely large file (simulated)
        large_file = temp_dir / "large_dataset.json"
        large_data = [{"text": f"Sample {i}", "label": "BENIGN"} for i in range(50000)]
        with open(large_file, "w") as f:
            json.dump(large_data, f)
        datasets["large"] = large_file

        return {"temp_dir": temp_dir, "datasets": datasets}

    @pytest.mark.asyncio
    async def test_complete_dataset_pipeline(self, data_service, realistic_datasets):
        """Test complete pipeline: config -> load -> preprocess -> validate -> cache."""
        datasets = realistic_datasets["datasets"]

        # Test pipeline with UNSW-NB15 dataset
        config = DatasetConfig(
            name="unsw_pipeline_test",
            path=str(datasets["unsw_nb15"]["file"]),
            source="local",
            format="json",
        )

        # Step 1: Load dataset
        start_time = time.time()
        dataset = await data_service.load_dataset(config)
        load_time = time.time() - start_time

        assert dataset.size == datasets["unsw_nb15"]["size"]
        assert dataset.dataset_id == "unsw_pipeline_test"

        # Step 2: Validate quality
        quality_report = await data_service.validate_dataset_quality("unsw_pipeline_test")
        assert isinstance(quality_report, DataQualityReport)
        assert quality_report.total_samples == dataset.size
        assert (
            quality_report.quality_score > 0.5
        )  # Should be reasonable quality (allowing for some duplicates in network data)

        # Step 3: Compute statistics
        stats = await data_service.compute_dataset_statistics("unsw_pipeline_test")
        assert isinstance(stats, DatasetStatistics)
        assert stats.total_samples == dataset.size
        assert stats.attack_samples > 0
        assert stats.benign_samples > 0
        assert len(stats.attack_types) > 0

        # Step 4: Verify caching
        cache_stats = await data_service.get_cache_performance_stats()
        assert cache_stats["cache_size"] > 0

        # Step 5: Test cached access (should be faster)
        cached_start = time.time()
        cached_dataset = await data_service.load_dataset(config)
        cached_time = time.time() - cached_start

        assert cached_dataset.dataset_id == dataset.dataset_id
        assert cached_time < load_time / 2  # Should be significantly faster

        print(f"Initial load: {load_time:.3f}s, Cached load: {cached_time:.3f}s")
        print(f"Quality score: {quality_report.quality_score:.3f}")
        print(f"Attack ratio: {stats.attack_ratio:.3f}")

    @pytest.mark.asyncio
    async def test_multi_source_dataset_loading(self, data_service, realistic_datasets):
        """Test loading datasets from multiple sources simultaneously."""
        datasets = realistic_datasets["datasets"]

        # Create configs for different datasets
        configs = [
            DatasetConfig(
                name="multi_unsw",
                path=str(datasets["unsw_nb15"]["file"]),
                source="local",
                format="json",
            ),
            DatasetConfig(
                name="multi_phishing",
                path=str(datasets["phishing"]["file"]),
                source="local",
                format="json",
            ),
            DatasetConfig(
                name="multi_web",
                path=str(datasets["web_logs"]["file"]),
                source="local",
                format="jsonl",
            ),
            DatasetConfig(
                name="multi_malware",
                path=str(datasets["malware"]["file"]),
                source="local",
                format="csv",
            ),
        ]

        # Load all datasets concurrently
        start_time = time.time()
        tasks = [data_service.load_dataset(config) for config in configs]
        loaded_datasets = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time

        # Verify all datasets loaded correctly
        assert len(loaded_datasets) == 4
        for i, dataset in enumerate(loaded_datasets):
            expected_name = configs[i].name
            assert dataset.dataset_id == expected_name
            assert dataset.size > 0

        # Verify different attack types are present
        all_attack_types = set()
        total_samples = 0
        for dataset in loaded_datasets:
            stats = await data_service.compute_dataset_statistics(dataset.dataset_id)
            all_attack_types.update(stats.attack_types.keys())
            total_samples += stats.total_samples

        # Should have various attack types from different domains
        assert len(all_attack_types) >= 5  # DoS, Probe, phishing, SQLi, etc.
        assert total_samples > 20000  # Total samples from all datasets

        print(f"Loaded {len(loaded_datasets)} datasets concurrently in {concurrent_time:.3f}s")
        print(f"Total samples: {total_samples}")
        print(f"Attack types found: {sorted(all_attack_types)}")

    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, data_service, realistic_datasets):
        """Test with large datasets approaching memory limits."""
        datasets = realistic_datasets["datasets"]

        # Load the largest dataset
        config = DatasetConfig(
            name="large_dataset_test",
            path=str(datasets["unsw_nb15"]["file"]),  # 10K samples
            source="local",
            format="json",
        )

        # Monitor memory before loading
        initial_memory = await data_service.get_memory_status()

        # Load dataset
        dataset = await data_service.load_dataset(config)

        # Check memory after loading
        after_load_memory = await data_service.get_memory_status()

        # Verify dataset loaded correctly
        assert dataset.size == 10000

        # Test streaming processing for large dataset
        batch_count = 0
        sample_count = 0

        async for batch in data_service.stream_dataset_batches(config, batch_size=500):
            batch_count += 1
            sample_count += len(batch.samples)

            # Verify batch properties
            assert len(batch.samples) <= 500
            assert hasattr(batch, "batch_info")

            # Stop after processing a portion to test streaming
            if batch_count >= 10:  # Process 5000 samples
                break

        # Test memory cleanup
        cleanup_stats = await data_service.cleanup_memory()
        await data_service.get_memory_status()  # Check final memory state

        # Verify memory management
        memory_increase = (
            after_load_memory["memory_status"]["process_memory_mb"]
            - initial_memory["memory_status"]["process_memory_mb"]
        )

        print(f"Memory increase after loading: {memory_increase:.2f}MB")
        print(f"Processed {sample_count} samples in {batch_count} batches")
        print(f"Cleanup freed: {cleanup_stats.get('memory_freed_mb', 0):.2f}MB")

        # Verify streaming worked correctly
        assert sample_count == batch_count * 500
        assert memory_increase > 0  # Should use some memory

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, data_service, corrupted_datasets):
        """Test recovery from various error conditions."""
        datasets = corrupted_datasets["datasets"]

        error_scenarios = [
            {
                "name": "invalid_json",
                "path": datasets["invalid_json"],
                "expected_error": DataLoadingError,
                "description": "Invalid JSON format",
            },
            {
                "name": "non_existent",
                "path": datasets["non_existent"],
                "expected_error": DataLoadingError,
                "description": "Non-existent file",
            },
            {
                "name": "empty_file",
                "path": datasets["empty"],
                "expected_error": DataLoadingError,
                "description": "Empty file",
            },
        ]

        error_count = 0
        recovery_count = 0

        for scenario in error_scenarios:
            config = DatasetConfig(
                name=f"error_test_{scenario['name']}",
                path=str(scenario["path"]),
                source="local",
                format="json",
            )

            # Test that error is properly raised
            try:
                await data_service.load_dataset(config)
                raise AssertionError(
                    f"Expected {scenario['expected_error'].__name__} for {scenario['description']}"
                )
            except scenario["expected_error"] as e:
                error_count += 1
                print(f"✓ Correctly handled {scenario['description']}: {str(e)[:100]}")

            # Verify service is still operational after error
            try:
                health = await data_service.health_check()
                # The status might be a ServiceStatus enum or string
                status_value = (
                    health.status.value if hasattr(health.status, "value") else str(health.status)
                )
                if status_value == "healthy" or status_value == "HEALTHY":
                    recovery_count += 1
                    print(f"✓ Service healthy after {scenario['description']}")
                else:
                    print(f"✗ Service status {status_value} after {scenario['description']}")
            except Exception as e:
                print(f"✗ Service not healthy after {scenario['description']}: {e}")

        # Test handling dataset with quality issues
        config = DatasetConfig(
            name="quality_issues_test",
            path=str(datasets["missing_fields"]),
            source="local",
            format="json",
        )

        try:
            dataset = await data_service.load_dataset(config)
            quality_report = await data_service.validate_dataset_quality("quality_issues_test")

            # Should load but have quality issues
            assert dataset.size > 0
            assert quality_report.has_issues
            assert quality_report.missing_labels > 0
            assert quality_report.invalid_labels > 0
            assert quality_report.quality_score < 0.8

            print(
                f"✓ Handled dataset with quality issues: score={quality_report.quality_score:.3f}"
            )
            recovery_count += 1
        except Exception as e:
            print(f"✗ Failed to handle quality issues: {e}")

        assert error_count == len(error_scenarios), "Not all error scenarios were handled"
        assert recovery_count >= len(error_scenarios), "Service didn't recover from all errors"

        print(
            f"Successfully handled {error_count} error scenarios with {recovery_count} recoveries"
        )

    @pytest.mark.asyncio
    async def test_data_service_under_load(self, data_service, realistic_datasets):
        """Test data service performance under high concurrent load."""
        datasets = realistic_datasets["datasets"]

        # Create multiple concurrent operations
        concurrent_tasks = []
        num_concurrent = 10

        # Mix of different operations
        for i in range(num_concurrent):
            dataset_key = list(datasets.keys())[i % len(datasets)]
            dataset_info = datasets[dataset_key]

            config = DatasetConfig(
                name=f"load_test_{i}_{dataset_key}",
                path=str(dataset_info["file"]),
                source="local",
                format="json"
                if dataset_key != "web_logs"
                else "jsonl"
                if dataset_key != "malware"
                else "csv",
            )

            # Different types of operations
            if i % 3 == 0:
                # Dataset loading
                concurrent_tasks.append(data_service.load_dataset(config))
            elif i % 3 == 1:
                # Load + quality validation
                async def load_and_validate(cfg=config):
                    dataset = await data_service.load_dataset(cfg)
                    quality = await data_service.validate_dataset_quality(dataset.dataset_id)
                    return dataset, quality

                concurrent_tasks.append(load_and_validate())
            else:
                # Load + statistics
                async def load_and_stats(cfg=config):
                    dataset = await data_service.load_dataset(cfg)
                    stats = await data_service.compute_dataset_statistics(dataset.dataset_id)
                    return dataset, stats

                concurrent_tasks.append(load_and_stats())

        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time

        # Analyze results
        successful_operations = 0
        failed_operations = 0
        total_samples_processed = 0

        for result in results:
            if isinstance(result, Exception):
                failed_operations += 1
                print(f"Failed operation: {result}")
            else:
                successful_operations += 1
                if isinstance(result, tuple):
                    # Operations that return multiple values
                    dataset = result[0]
                    total_samples_processed += dataset.size
                else:
                    # Simple dataset loading
                    total_samples_processed += result.size

        # Check memory and cache status after load
        memory_status = await data_service.get_memory_status()
        cache_stats = await data_service.get_cache_performance_stats()

        # Verify performance
        avg_time_per_operation = concurrent_time / num_concurrent
        throughput = total_samples_processed / concurrent_time

        print(f"Concurrent operations: {num_concurrent}")
        print(f"Total time: {concurrent_time:.3f}s")
        print(f"Average time per operation: {avg_time_per_operation:.3f}s")
        print(f"Successful: {successful_operations}, Failed: {failed_operations}")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Throughput: {throughput:.1f} samples/second")
        print(f"Cache utilization: {cache_stats['memory_utilization']:.1f}%")

        # Verify most operations succeeded
        assert successful_operations >= num_concurrent * 0.8  # At least 80% success rate
        assert throughput > 1000  # Should process at least 1000 samples/second
        assert not memory_status["memory_pressure"]  # Should not be under memory pressure

    @pytest.mark.asyncio
    async def test_realistic_cybersecurity_workflows(self, data_service, realistic_datasets):
        """Test with realistic cybersecurity dataset workflows."""
        datasets = realistic_datasets["datasets"]

        # Workflow 1: Network Traffic Analysis
        print("=== Network Traffic Analysis Workflow ===")

        # Load network dataset
        unsw_config = DatasetConfig(
            name="network_analysis",
            path=str(datasets["unsw_nb15"]["file"]),
            source="local",
            format="json",
        )

        network_dataset = await data_service.load_dataset(unsw_config)
        network_stats = await data_service.compute_dataset_statistics("network_analysis")

        # Verify network-specific attack types
        expected_network_attacks = {"DoS", "Probe", "R2L", "U2R"}
        found_attacks = set(network_stats.attack_types.keys())
        assert len(found_attacks.intersection(expected_network_attacks)) >= 2

        print(f"Network dataset: {network_dataset.size} flows")
        print(f"Attack types: {list(network_stats.attack_types.keys())}")
        print(f"Attack ratio: {network_stats.attack_ratio:.3f}")

        # Workflow 2: Email Security Analysis
        print("\n=== Email Security Analysis Workflow ===")

        phishing_config = DatasetConfig(
            name="email_security",
            path=str(datasets["phishing"]["file"]),
            source="local",
            format="json",
        )

        email_dataset = await data_service.load_dataset(phishing_config)
        email_quality = await data_service.validate_dataset_quality("email_security")
        email_stats = await data_service.compute_dataset_statistics("email_security")

        # Verify phishing detection
        assert "phishing" in email_stats.attack_types
        assert email_stats.attack_types["phishing"] > 0
        assert email_quality.quality_score > 0.9  # Email data should be high quality

        print(f"Email dataset: {email_dataset.size} emails")
        print(f"Phishing emails: {email_stats.attack_types.get('phishing', 0)}")
        print(f"Quality score: {email_quality.quality_score:.3f}")

        # Workflow 3: Web Application Security
        print("\n=== Web Application Security Workflow ===")

        web_config = DatasetConfig(
            name="web_security",
            path=str(datasets["web_logs"]["file"]),
            source="local",
            format="jsonl",
        )

        web_dataset = await data_service.load_dataset(web_config)
        web_stats = await data_service.compute_dataset_statistics("web_security")

        # Verify web attack types
        expected_web_attacks = {"SQLi", "Directory_Traversal", "Web_Attack"}
        web_attacks = set(web_stats.attack_types.keys())
        assert len(web_attacks.intersection(expected_web_attacks)) >= 2

        print(f"Web logs: {web_dataset.size} requests")
        print(f"Web attack types: {list(web_stats.attack_types.keys())}")

        # Workflow 4: Malware Analysis
        print("\n=== Malware Analysis Workflow ===")

        malware_config = DatasetConfig(
            name="malware_analysis",
            path=str(datasets["malware"]["file"]),
            source="local",
            format="csv",
        )

        malware_dataset = await data_service.load_dataset(malware_config)
        malware_stats = await data_service.compute_dataset_statistics("malware_analysis")

        # Verify malware families
        assert "malware" in malware_stats.attack_types
        assert malware_stats.attack_types["malware"] > 0

        print(f"Malware samples: {malware_dataset.size} files")
        print(f"Malware samples: {malware_stats.attack_types.get('malware', 0)}")

        # Workflow 5: Multi-domain Analysis
        print("\n=== Multi-domain Analysis Workflow ===")

        # Process all datasets for comprehensive analysis
        all_stats = [network_stats, email_stats, web_stats, malware_stats]
        total_samples = sum(stats.total_samples for stats in all_stats)
        total_attacks = sum(stats.attack_samples for stats in all_stats)
        all_attack_types = set()
        for stats in all_stats:
            all_attack_types.update(stats.attack_types.keys())

        overall_attack_ratio = total_attacks / total_samples if total_samples > 0 else 0

        print(f"Total samples across all domains: {total_samples}")
        print(f"Total attacks: {total_attacks}")
        print(f"Overall attack ratio: {overall_attack_ratio:.3f}")
        print(f"Unique attack types: {len(all_attack_types)}")
        print(f"Attack types: {sorted(all_attack_types)}")

        # Verify comprehensive coverage
        assert total_samples > 25000  # Should have processed substantial data
        assert len(all_attack_types) >= 6  # Should have diverse attack types
        assert 0.15 <= overall_attack_ratio <= 0.35  # Realistic attack ratio

    @pytest.mark.asyncio
    async def test_integration_with_preprocessing_pipeline(self, data_service, realistic_datasets):
        """Test data service integration with preprocessing pipeline."""
        datasets = realistic_datasets["datasets"]

        # Test with phishing email dataset
        config = DatasetConfig(
            name="preprocessing_integration",
            path=str(datasets["phishing"]["file"]),
            source="local",
            format="json",
        )

        # Load and analyze dataset
        dataset = await data_service.load_dataset(config)

        # Test batching for ML pipeline
        batch_sizes = [32, 64, 128]
        for batch_size in batch_sizes:
            batch_count = 0
            sample_count = 0

            # Test optimized batching
            optimized_batch = await data_service.get_optimized_batch(
                "preprocessing_integration", batch_size=batch_size
            )
            assert len(optimized_batch.samples) <= batch_size

            # Test streaming for preprocessing pipeline
            async for batch in data_service.stream_dataset_batches(config, batch_size=batch_size):
                batch_count += 1
                sample_count += len(batch.samples)

                # Verify batch is suitable for ML preprocessing
                assert all(hasattr(sample, "data") for sample in batch.samples)
                assert all(hasattr(sample, "label") for sample in batch.samples)

                # Test only a few batches to verify streaming works
                if batch_count >= 3:
                    break

        # Verify preprocessing-ready data structure
        stats = await data_service.compute_dataset_statistics("preprocessing_integration")
        assert stats.total_samples == dataset.size
        assert stats.attack_samples > 0
        assert stats.benign_samples > 0

        # Test data quality for ML training
        quality = await data_service.validate_dataset_quality("preprocessing_integration")
        assert quality.quality_score > 0.8  # Should be high quality for ML
        assert quality.empty_samples == 0  # No empty samples for training
        assert quality.missing_labels == 0  # All samples should have labels

        print(f"Dataset ready for preprocessing: {dataset.size} samples")
        print(f"Quality score: {quality.quality_score:.3f}")
        print(
            f"Label distribution - Attack: {stats.attack_ratio:.3f}, Benign: {stats.benign_ratio:.3f}"
        )

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, data_service, realistic_datasets):
        """Test performance benchmarks to validate acceptable performance."""
        datasets = realistic_datasets["datasets"]

        performance_results = {}

        # Benchmark 1: Large dataset loading
        config = DatasetConfig(
            name="perf_large",
            path=str(datasets["unsw_nb15"]["file"]),
            source="local",
            format="json",
        )

        start_time = time.time()
        dataset = await data_service.load_dataset(config)
        load_time = time.time() - start_time

        performance_results["large_dataset_load"] = {
            "samples": dataset.size,
            "time_seconds": load_time,
            "samples_per_second": dataset.size / load_time,
        }

        # Benchmark 2: Quality validation
        start_time = time.time()
        await data_service.validate_dataset_quality("perf_large")
        validation_time = time.time() - start_time

        performance_results["quality_validation"] = {
            "samples": dataset.size,
            "time_seconds": validation_time,
            "samples_per_second": dataset.size / validation_time,
        }

        # Benchmark 3: Statistics computation
        start_time = time.time()
        await data_service.compute_dataset_statistics("perf_large")
        stats_time = time.time() - start_time

        performance_results["statistics_computation"] = {
            "samples": dataset.size,
            "time_seconds": stats_time,
            "samples_per_second": dataset.size / stats_time,
        }

        # Benchmark 4: Batch streaming
        start_time = time.time()
        batch_count = 0
        async for _batch in data_service.stream_dataset_batches(config, batch_size=100):
            batch_count += 1
            if batch_count >= 50:  # Stream 5000 samples
                break
        streaming_time = time.time() - start_time

        performance_results["batch_streaming"] = {
            "batches": batch_count,
            "samples": batch_count * 100,
            "time_seconds": streaming_time,
            "samples_per_second": (batch_count * 100) / streaming_time,
        }

        # Benchmark 5: Memory efficiency
        memory_status = await data_service.get_memory_status()
        cache_stats = await data_service.get_cache_performance_stats()

        performance_results["memory_efficiency"] = {
            "cache_utilization_percent": cache_stats["memory_utilization"],
            "compression_ratio": cache_stats.get("compression_ratio", 1.0),
            "memory_pressure": memory_status["memory_pressure"],
        }

        # Print results
        print("=== Performance Benchmark Results ===")
        for benchmark, results in performance_results.items():
            print(f"\n{benchmark.replace('_', ' ').title()}:")
            for metric, value in results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")

        # Validate acceptable performance
        assert performance_results["large_dataset_load"]["samples_per_second"] > 500
        assert performance_results["quality_validation"]["samples_per_second"] > 1000
        assert performance_results["statistics_computation"]["samples_per_second"] > 1000
        assert performance_results["batch_streaming"]["samples_per_second"] > 5000
        assert not performance_results["memory_efficiency"]["memory_pressure"]

        # Verify compression effectiveness if enabled
        if cache_stats.get("compression_enabled"):
            assert performance_results["memory_efficiency"]["compression_ratio"] > 5.0

    @pytest.mark.asyncio
    async def test_service_resilience_and_recovery(
        self, data_service, realistic_datasets, corrupted_datasets
    ):
        """Test service resilience and recovery from various failure scenarios."""
        datasets = realistic_datasets["datasets"]
        corrupted = corrupted_datasets["datasets"]

        # Start with successful operations to establish baseline
        good_config = DatasetConfig(
            name="resilience_baseline",
            path=str(datasets["phishing"]["file"]),
            source="local",
            format="json",
        )

        baseline_dataset = await data_service.load_dataset(good_config)
        assert baseline_dataset.size > 0

        initial_health = await data_service.health_check()
        status_value = (
            initial_health.status.value
            if hasattr(initial_health.status, "value")
            else str(initial_health.status)
        )
        assert status_value.lower() == "healthy"

        # Test resilience scenarios
        resilience_tests = [
            {
                "name": "corrupted_data_recovery",
                "config": DatasetConfig(
                    name="corrupted_test",
                    path=str(corrupted["invalid_json"]),
                    source="local",
                    format="json",
                ),
                "should_fail": True,
            },
            {
                "name": "non_existent_file_recovery",
                "config": DatasetConfig(
                    name="missing_test",
                    path=str(corrupted["non_existent"]),
                    source="local",
                    format="json",
                ),
                "should_fail": True,
            },
            {
                "name": "good_operation_after_failure",
                "config": DatasetConfig(
                    name="recovery_test",
                    path=str(datasets["web_logs"]["file"]),
                    source="local",
                    format="jsonl",
                ),
                "should_fail": False,
            },
        ]

        recovery_count = 0

        for test in resilience_tests:
            print(f"\n--- Testing {test['name']} ---")

            if test["should_fail"]:
                # Operation should fail
                try:
                    await data_service.load_dataset(test["config"])
                    raise AssertionError(f"Expected failure for {test['name']}")
                except DataLoadingError:
                    print(f"✓ Correctly failed: {test['name']}")
            else:
                # Operation should succeed
                try:
                    dataset = await data_service.load_dataset(test["config"])
                    assert dataset.size > 0
                    print(f"✓ Successful operation: {test['name']} ({dataset.size} samples)")
                except Exception as e:
                    raise AssertionError(f"Unexpected failure for {test['name']}: {e}") from e

            # Verify service health after each operation
            health = await data_service.health_check()
            status_value = (
                health.status.value if hasattr(health.status, "value") else str(health.status)
            )
            if status_value.lower() == "healthy":
                recovery_count += 1
                print(f"✓ Service healthy after {test['name']}")
            else:
                print(
                    f"✗ Service not healthy after {test['name']}: {health.message if hasattr(health, 'message') else status_value}"
                )

        # Verify service can still perform complex operations
        final_config = DatasetConfig(
            name="final_resilience_test",
            path=str(datasets["unsw_nb15"]["file"]),
            source="local",
            format="json",
        )

        final_dataset = await data_service.load_dataset(final_config)
        final_stats = await data_service.compute_dataset_statistics("final_resilience_test")
        final_quality = await data_service.validate_dataset_quality("final_resilience_test")

        # Verify cache is still working
        cache_stats = await data_service.get_cache_performance_stats()
        memory_status = await data_service.get_memory_status()

        print("\n=== Final Service State ===")
        print(f"Service recovered from {recovery_count}/{len(resilience_tests)} scenarios")
        print(f"Final dataset loaded: {final_dataset.size} samples")
        print(f"Cache entries: {cache_stats['cache_size']}")
        print(f"Memory pressure: {memory_status['memory_pressure']}")

        # Assertions
        assert recovery_count == len(resilience_tests)
        assert final_dataset.size > 0
        assert final_stats.total_samples == final_dataset.size
        assert final_quality.quality_score > 0
        assert not memory_status["memory_pressure"]


# Additional helper functions for realistic data generation
def _generate_network_flow_features():
    """Generate realistic network flow features for testing."""

    # Common network services and their typical ports
    services = {
        "http": {"port": 80, "proto": "tcp"},
        "https": {"port": 443, "proto": "tcp"},
        "dns": {"port": 53, "proto": "udp"},
        "ssh": {"port": 22, "proto": "tcp"},
        "ftp": {"port": 21, "proto": "tcp"},
        "smtp": {"port": 25, "proto": "tcp"},
        "pop3": {"port": 110, "proto": "tcp"},
        "imap": {"port": 143, "proto": "tcp"},
    }

    return services


def _generate_email_content_variants():
    """Generate realistic email content variants for testing."""
    benign_subjects = [
        "Weekly team meeting",
        "Project update",
        "Quarterly report",
        "Holiday schedule",
        "Training session",
    ]

    phishing_subjects = [
        "URGENT: Account verification required",
        "Your account has been suspended",
        "Security alert for your account",
        "Immediate action required",
        "Confirm your identity",
    ]

    return benign_subjects, phishing_subjects

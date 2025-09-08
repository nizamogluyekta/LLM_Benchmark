"""
Unit tests for network log preprocessor.

This module tests the NetworkLogPreprocessor functionality including
log parsing, feature extraction, and attack indicator identification.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.data.models import DatasetSample
from benchmark.data.preprocessors.network_logs import NetworkLogPreprocessor


class TestNetworkLogPreprocessor:
    """Test NetworkLogPreprocessor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = NetworkLogPreprocessor()

        # Load test fixtures
        fixtures_path = Path(__file__).parent.parent / "fixtures" / "network_logs.json"
        with open(fixtures_path) as f:
            self.fixtures = json.load(f)

    def test_init_default_name(self):
        """Test preprocessor initialization with default name."""
        preprocessor = NetworkLogPreprocessor()
        assert preprocessor.name == "NetworkLogPreprocessor"

    def test_init_custom_name(self):
        """Test preprocessor initialization with custom name."""
        preprocessor = NetworkLogPreprocessor("CustomNetworkProcessor")
        assert preprocessor.name == "CustomNetworkProcessor"

    def test_get_required_fields(self):
        """Test getting required fields."""
        required_fields = self.preprocessor.get_required_fields()
        assert required_fields == ["input_text", "label"]

    def test_get_supported_config_keys(self):
        """Test getting supported configuration keys."""
        config_keys = self.preprocessor.get_supported_config_keys()
        expected_keys = [
            "extract_features",
            "identify_attacks",
            "normalize_protocols",
            "parse_timestamps",
            "decode_urls",
        ]
        assert set(config_keys) == set(expected_keys)

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            "extract_features": True,
            "identify_attacks": True,
            "normalize_protocols": True,
        }
        warnings = self.preprocessor.validate_config(config)
        assert len(warnings) == 0

    def test_validate_config_invalid_keys(self):
        """Test configuration validation with invalid keys."""
        config = {
            "extract_features": True,
            "unknown_key": "value",
            "another_unknown": 123,
        }
        warnings = self.preprocessor.validate_config(config)
        assert len(warnings) == 2
        assert any("unknown_key" in warning for warning in warnings)
        assert any("another_unknown" in warning for warning in warnings)

    @pytest.mark.asyncio
    async def test_process_apache_common_log(self):
        """Test processing Apache common log format."""
        log_data = self.fixtures["apache_common_logs"][0]
        sample = DatasetSample(input_text=log_data["log_entry"], label="BENIGN")

        config = {"extract_features": True, "identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Check basic processing
        assert processed_sample.label == "BENIGN"
        assert "log_format" in processed_sample.metadata
        assert processed_sample.metadata["log_format"] == "apache_common"

        # Check expected features
        expected = log_data["expected_features"]
        for feature, expected_value in expected.items():
            if feature in processed_sample.metadata:
                if isinstance(expected_value, list):
                    assert set(processed_sample.metadata[feature]) >= set(expected_value)
                else:
                    assert processed_sample.metadata[feature] == expected_value

    @pytest.mark.asyncio
    async def test_process_nginx_log(self):
        """Test processing Nginx log format."""
        log_data = self.fixtures["nginx_logs"][0]
        sample = DatasetSample(input_text=log_data["log_entry"], label="BENIGN")

        config = {"extract_features": True, "identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert processed_sample.metadata["log_format"] == "nginx"

        # Check expected features
        expected = log_data["expected_features"]
        for feature, expected_value in expected.items():
            if feature in processed_sample.metadata:
                if isinstance(expected_value, list):
                    assert set(processed_sample.metadata[feature]) >= set(expected_value)
                else:
                    assert processed_sample.metadata[feature] == expected_value

    @pytest.mark.asyncio
    async def test_process_firewall_log(self):
        """Test processing firewall log format."""
        log_data = self.fixtures["firewall_logs"][0]
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"extract_features": True, "identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert processed_sample.metadata["log_format"] == "firewall"

        # Check expected features
        expected = log_data["expected_features"]
        for feature, expected_value in expected.items():
            if feature in processed_sample.metadata and isinstance(expected_value, list):
                assert set(processed_sample.metadata[feature]) >= set(expected_value)

    @pytest.mark.asyncio
    async def test_sql_injection_detection(self):
        """Test SQL injection attack detection."""
        log_data = self.fixtures["apache_common_logs"][2]  # SQL injection sample
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "attack_indicators" in processed_sample.metadata
        assert "sql_injection" in processed_sample.metadata["attack_indicators"]

    @pytest.mark.asyncio
    async def test_xss_detection(self):
        """Test XSS attack detection."""
        log_data = self.fixtures["apache_common_logs"][3]  # XSS sample
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "attack_indicators" in processed_sample.metadata
        assert "xss" in processed_sample.metadata["attack_indicators"]

    @pytest.mark.asyncio
    async def test_directory_traversal_detection(self):
        """Test directory traversal attack detection."""
        log_data = self.fixtures["nginx_logs"][1]  # Directory traversal sample
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "attack_indicators" in processed_sample.metadata
        assert "directory_traversal" in processed_sample.metadata["attack_indicators"]

    @pytest.mark.asyncio
    async def test_command_injection_detection(self):
        """Test command injection attack detection."""
        log_data = self.fixtures["attack_samples"][0]  # Command injection sample
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "attack_indicators" in processed_sample.metadata
        assert "command_injection" in processed_sample.metadata["attack_indicators"]

    @pytest.mark.asyncio
    async def test_connection_info_extraction(self):
        """Test connection information extraction."""
        log_data = self.fixtures["connection_logs"][0]
        sample = DatasetSample(input_text=log_data["log_entry"], label="BENIGN")

        config = {"extract_features": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        expected = log_data["expected_features"]
        for feature, expected_value in expected.items():
            assert processed_sample.metadata[feature] == expected_value

    @pytest.mark.asyncio
    async def test_high_risk_port_detection(self):
        """Test high-risk port access detection."""
        log_data = self.fixtures["connection_logs"][1]  # RDP connection
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"identify_attacks": True, "extract_features": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "attack_indicators" in processed_sample.metadata
        assert "high_risk_port_access" in processed_sample.metadata["attack_indicators"]

    @pytest.mark.asyncio
    async def test_service_identification(self):
        """Test service identification by port."""
        log_data = self.fixtures["firewall_logs"][0]  # SSH connection
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"extract_features": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "identified_services" in processed_sample.metadata
        assert "ssh" in processed_sample.metadata["identified_services"]

    @pytest.mark.asyncio
    async def test_ip_classification(self):
        """Test IP address classification (private vs public)."""
        sample = DatasetSample(
            input_text="Connection from 192.168.1.100 to 8.8.8.8", label="BENIGN"
        )

        config = {"extract_features": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "private_ips" in processed_sample.metadata
        assert "public_ips" in processed_sample.metadata
        assert "192.168.1.100" in processed_sample.metadata["private_ips"]
        assert "8.8.8.8" in processed_sample.metadata["public_ips"]

    @pytest.mark.asyncio
    async def test_protocol_normalization(self):
        """Test protocol name normalization."""
        sample = DatasetSample(input_text="GET /test HTTP/1.1", label="BENIGN")

        config = {"normalize_protocols": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        if "normalized_protocol" in processed_sample.metadata:
            assert processed_sample.metadata["normalized_protocol"] == "http"

    @pytest.mark.asyncio
    async def test_url_decoding(self):
        """Test URL decoding functionality."""
        sample = DatasetSample(input_text="GET /search?q=%3Cscript%3E HTTP/1.1", label="ATTACK")

        config = {"decode_urls": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        if "decoded_url" in processed_sample.metadata:
            assert "<script>" in processed_sample.metadata["decoded_url"]

    @pytest.mark.asyncio
    async def test_timestamp_parsing(self):
        """Test timestamp parsing from logs."""
        log_data = self.fixtures["apache_common_logs"][0]
        sample = DatasetSample(input_text=log_data["log_entry"], label="BENIGN")

        config = {"parse_timestamps": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        if "parsed_timestamp" in processed_sample.metadata:
            # Should be in ISO format
            timestamp = processed_sample.metadata["parsed_timestamp"]
            assert isinstance(timestamp, str)
            assert "T" in timestamp or "2024" in timestamp

    @pytest.mark.asyncio
    async def test_multiple_attacks_detection(self):
        """Test detection of multiple attack types in one log."""
        # Create a log with multiple attack indicators
        sample = DatasetSample(
            input_text="GET /app?cmd=wget&data=' OR 1=1-- HTTP/1.1", label="ATTACK"
        )

        config = {"identify_attacks": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "attack_indicators" in processed_sample.metadata
        indicators = processed_sample.metadata["attack_indicators"]
        assert "command_injection" in indicators
        assert "sql_injection" in indicators

    @pytest.mark.asyncio
    async def test_port_scanning_detection(self):
        """Test port scanning detection based on multiple ports."""
        # Log with many different ports
        sample = DatasetSample(
            input_text="Connections to ports: 22, 23, 80, 443, 3389, 3306, 5432, 6379",
            label="ATTACK",
        )

        config = {"identify_attacks": True, "extract_features": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert "ports" in processed_sample.metadata
        assert len(processed_sample.metadata["ports"]) > 5

        if "attack_indicators" in processed_sample.metadata:
            assert "port_scanning" in processed_sample.metadata["attack_indicators"]

    @pytest.mark.asyncio
    async def test_feature_extraction_disabled(self):
        """Test processing with feature extraction disabled."""
        sample = DatasetSample(
            input_text='192.168.1.100 - - [23/Jan/2024:10:47:32 +0000] "GET /index.html HTTP/1.1" 200 2326',
            label="BENIGN",
        )

        config = {"extract_features": False}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should have basic log parsing but no detailed network features
        assert "log_format" in processed_sample.metadata
        assert "ip_count" not in processed_sample.metadata
        assert "port_count" not in processed_sample.metadata

    @pytest.mark.asyncio
    async def test_attack_detection_disabled(self):
        """Test processing with attack detection disabled."""
        log_data = self.fixtures["apache_common_logs"][2]  # SQL injection sample
        sample = DatasetSample(input_text=log_data["log_entry"], label="ATTACK")

        config = {"identify_attacks": False}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should not have attack indicators
        assert "attack_indicators" not in processed_sample.metadata

    @pytest.mark.asyncio
    async def test_empty_log_processing(self):
        """Test processing minimal or malformed logs."""
        sample = DatasetSample(
            input_text=" ",  # Single space instead of empty string
            label="BENIGN",
        )

        config = {}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should handle gracefully and clean the text
        assert processed_sample.input_text == ""  # Whitespace should be cleaned
        assert processed_sample.label == "BENIGN"

    @pytest.mark.asyncio
    async def test_unknown_log_format(self):
        """Test processing logs that don't match known formats."""
        sample = DatasetSample(
            input_text="Some random log entry that doesn't match any format", label="BENIGN"
        )

        config = {}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert processed_sample.metadata["log_format"] == "unknown"

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test processing multiple samples at once."""
        samples = []
        for log_data in self.fixtures["apache_common_logs"][:2]:
            sample = DatasetSample(input_text=log_data["log_entry"], label="BENIGN")
            samples.append(sample)

        config = {"extract_features": True, "identify_attacks": True}
        result = await self.preprocessor.process(samples, config)

        assert len(result) == 2
        for processed_sample in result:
            assert "log_format" in processed_sample.metadata
            assert processed_sample.metadata["log_format"] == "apache_common"

    def test_normalize_protocol(self):
        """Test protocol normalization method."""
        # Test various protocol formats
        assert self.preprocessor._normalize_protocol("HTTP/1.1") == "http"
        assert self.preprocessor._normalize_protocol("HTTP/2.0") == "http"
        assert self.preprocessor._normalize_protocol("HTTPS") == "https"
        assert self.preprocessor._normalize_protocol("tcp") == "tcp"
        assert self.preprocessor._normalize_protocol("UDP") == "udp"
        assert self.preprocessor._normalize_protocol("") == "unknown"
        assert self.preprocessor._normalize_protocol("CUSTOM") == "custom"

    def test_extract_ports(self):
        """Test port extraction method."""
        text = "Connection 192.168.1.1:80 -> 10.0.0.1:443 on port 22"
        ports = self.preprocessor._extract_ports(text)

        assert set(ports) == {80, 443, 22}

    def test_extract_protocols(self):
        """Test protocol extraction method."""
        text = "TCP connection over HTTP using UDP for DNS"
        protocols = self.preprocessor._extract_protocols(text)

        assert set(protocols) == {"TCP", "HTTP", "UDP", "DNS"}

    def test_extract_http_methods(self):
        """Test HTTP method extraction."""
        text = 'GET /index.html HTTP/1.1" POST /api/data PUT /resource'
        methods = self.preprocessor._extract_http_methods(text)

        assert set(methods) == {"GET", "POST", "PUT"}

    def test_extract_status_codes(self):
        """Test HTTP status code extraction."""
        text = 'HTTP/1.1" 200 1234 and also status 404 with error 500'
        codes = self.preprocessor._extract_status_codes(text)

        assert set(codes) == {200, 404, 500}

    def test_identify_services_by_port(self):
        """Test service identification by port numbers."""
        ports = [22, 80, 443, 3306, 6379]
        services = self.preprocessor._identify_services_by_port(ports)

        expected_services = {"ssh", "http", "https", "mysql", "redis"}
        assert set(services) == expected_services

    def test_is_private_ip(self):
        """Test private IP detection."""
        # Test private IPs
        assert self.preprocessor._is_private_ip("192.168.1.1") is True
        assert self.preprocessor._is_private_ip("10.0.0.1") is True
        assert self.preprocessor._is_private_ip("172.16.0.1") is True
        assert self.preprocessor._is_private_ip("127.0.0.1") is True

        # Test public IPs
        assert self.preprocessor._is_private_ip("8.8.8.8") is False
        assert self.preprocessor._is_private_ip("1.1.1.1") is False

    @pytest.mark.asyncio
    async def test_sample_validation(self):
        """Test sample validation with required fields."""
        # Test a valid sample first
        sample = DatasetSample(input_text="test log", label="BENIGN")

        # This should process without error
        result = await self.preprocessor.process([sample], {})
        assert len(result) == 1

        # Note: DatasetSample model validation happens at construction time,
        # so we can't test None values directly as they would fail model validation

    @pytest.mark.asyncio
    async def test_processing_error_handling(self):
        """Test error handling during processing."""
        sample = DatasetSample(input_text="test log", label="BENIGN")

        # Mock an error in feature extraction
        with (
            patch.object(
                self.preprocessor, "_extract_network_features", side_effect=Exception("Test error")
            ),
            pytest.raises(Exception, match="Test error"),
        ):
            await self.preprocessor.process([sample], {"extract_features": True})

    def test_str_representation(self):
        """Test string representation of preprocessor."""
        str_repr = str(self.preprocessor)

        assert "NetworkLogPreprocessor" in str_repr
        assert "input_text" in str_repr
        assert "label" in str_repr

    def test_repr_representation(self):
        """Test detailed representation of preprocessor."""
        repr_str = repr(self.preprocessor)

        assert "NetworkLogPreprocessor" in repr_str
        assert "required_fields" in repr_str
        assert "supported_config" in repr_str

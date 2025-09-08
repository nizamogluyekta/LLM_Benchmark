"""
Network log preprocessor for the LLM Cybersecurity Benchmark system.

This module provides specialized preprocessing for network log data commonly found
in cybersecurity datasets, including Apache, Nginx, firewall logs, and UNSW-NB15 format.
"""

import re
from datetime import datetime
from typing import Any
from urllib.parse import unquote

from benchmark.data.models import DatasetSample
from benchmark.data.preprocessors.base import DataPreprocessor, PreprocessorError
from benchmark.data.preprocessors.common import PreprocessingUtilities


class NetworkLogPreprocessor(DataPreprocessor):
    """
    Specialized preprocessor for network log data.

    Handles parsing of common network log formats and extraction of
    network-specific features for cybersecurity analysis.
    """

    # Common log format patterns
    APACHE_COMMON_LOG_PATTERN = re.compile(
        r'^(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<url>\S+) (?P<protocol>[^"]*)" '
        r"(?P<status>\d+) (?P<size>\S+)$"
    )

    NGINX_LOG_PATTERN = re.compile(
        r'^(?P<ip>\S+) - \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<url>\S+) (?P<protocol>[^"]*)" '
        r'(?P<status>\d+) (?P<size>\d+) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"$'
    )

    FIREWALL_LOG_PATTERN = re.compile(
        r"(?P<timestamp>\w+ \d+ \d+:\d+:\d+) .*?"
        r"(?:SRC=(?P<src_ip>[\d.]+)).*?"
        r"(?:DST=(?P<dst_ip>[\d.]+)).*?"
        r"(?:PROTO=(?P<protocol>\w+)).*?"
        r"(?:SPT=(?P<src_port>\d+)).*?"
        r"(?:DPT=(?P<dst_port>\d+))?"
    )

    # Network connection patterns
    CONNECTION_PATTERN = re.compile(
        r"(?P<src_ip>(?:\d{1,3}\.){3}\d{1,3}):(?P<src_port>\d+)\s*->\s*"
        r"(?P<dst_ip>(?:\d{1,3}\.){3}\d{1,3}):(?P<dst_port>\d+)"
    )

    # Port patterns for different services
    PORT_PATTERNS = {
        "http": [80, 8080, 8000, 8008],
        "https": [443, 8443],
        "ssh": [22],
        "ftp": [21, 20],
        "telnet": [23],
        "smtp": [25, 587],
        "dns": [53],
        "pop3": [110, 995],
        "imap": [143, 993],
        "snmp": [161, 162],
        "ldap": [389, 636],
        "mysql": [3306],
        "postgresql": [5432],
        "redis": [6379],
        "mongodb": [27017],
    }

    # Attack indicator patterns
    ATTACK_INDICATORS = {
        "sql_injection": [
            re.compile(
                r"(?i)(union\s+select|or\s+1\s*=\s*1|drop\s+table|exec\s*\()", re.IGNORECASE
            ),
            re.compile(r"(?i)(insert\s+into|delete\s+from|update\s+set)", re.IGNORECASE),
            re.compile(r"(?i)(information_schema|sysobjects|sys\.tables)", re.IGNORECASE),
        ],
        "xss": [
            re.compile(r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)", re.IGNORECASE),
            re.compile(r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()", re.IGNORECASE),
        ],
        "directory_traversal": [
            re.compile(r"(?i)(\.\./|\.\.\\|%2e%2e%2f|%2e%2e\\)", re.IGNORECASE),
            re.compile(r"(?i)(etc/passwd|boot\.ini|win\.ini)", re.IGNORECASE),
        ],
        "command_injection": [
            re.compile(r"(?i)(\||\|\||&|&&|;|`|\$\()", re.IGNORECASE),
            re.compile(r"(?i)(wget|curl|nc|netcat|ping)", re.IGNORECASE),
        ],
        "brute_force": [
            re.compile(r"(?i)(admin|administrator|root|test|guest)", re.IGNORECASE),
            re.compile(r"(?i)(password|passwd|login|auth)", re.IGNORECASE),
        ],
        "shellcode": [
            re.compile(r"(?i)(\\x[0-9a-f]{2}|%u[0-9a-f]{4}|\\u[0-9a-f]{4})", re.IGNORECASE),
            re.compile(r"(?i)(nop|shellcode|metasploit)", re.IGNORECASE),
        ],
        "dos_attack": [
            re.compile(r"(?i)(flood|ddos|dos\s+attack)", re.IGNORECASE),
            re.compile(r"(?i)(slowloris|hulk|goldeneye)", re.IGNORECASE),
        ],
    }

    def __init__(self, name: str | None = None):
        """Initialize the network log preprocessor."""
        super().__init__(name)

    async def process(
        self, samples: list[DatasetSample], config: dict[str, Any]
    ) -> list[DatasetSample]:
        """
        Process network log samples.

        Args:
            samples: List of samples to process
            config: Configuration dictionary with options:
                - extract_features: Extract network features (default: True)
                - identify_attacks: Identify attack indicators (default: True)
                - normalize_protocols: Normalize protocol names (default: True)
                - parse_timestamps: Parse and normalize timestamps (default: True)
                - decode_urls: URL decode request paths (default: True)

        Returns:
            List of processed samples with network-specific metadata

        Raises:
            PreprocessorError: If processing fails
        """
        # Validate samples
        self._validate_samples(samples)

        processed_samples = []

        for sample in samples:
            try:
                # Create a copy of the sample
                processed_sample = sample.model_copy()

                # Parse the log entry
                log_features = await self._parse_log_entry(sample.input_text, config)

                # Add extracted features to metadata
                processed_sample.metadata.update(log_features)

                # Extract network-specific features if enabled
                if config.get("extract_features", True):
                    network_features = self._extract_network_features(sample.input_text)
                    processed_sample.metadata.update(network_features)

                # Identify attack indicators if enabled
                if config.get("identify_attacks", True):
                    attack_indicators = self._identify_attack_indicators(
                        sample.input_text, processed_sample.metadata
                    )
                    if attack_indicators:
                        processed_sample.metadata["attack_indicators"] = attack_indicators

                # Clean and normalize the input text
                cleaned_text = PreprocessingUtilities.clean_text(
                    sample.input_text,
                    {
                        "remove_extra_whitespace": True,
                        "preserve_newlines": False,
                        "normalize_unicode": True,
                    },
                )
                processed_sample.input_text = cleaned_text

                processed_samples.append(processed_sample)

            except Exception as e:
                error_msg = f"Failed to process network log sample: {e}"
                self.logger.error(error_msg)
                raise PreprocessorError(error_msg, sample_id=sample.id, cause=e) from e

        return processed_samples

    def get_required_fields(self) -> list[str]:
        """Get list of required fields for network log preprocessing."""
        return ["input_text", "label"]

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """
        Validate configuration and return warnings.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of warning messages (empty if no warnings)
        """
        warnings = []

        # Check for unknown configuration keys
        warnings.extend(self._validate_config_keys(config))

        return warnings

    def get_supported_config_keys(self) -> list[str]:
        """Get list of supported configuration keys."""
        return [
            "extract_features",
            "identify_attacks",
            "normalize_protocols",
            "parse_timestamps",
            "decode_urls",
        ]

    async def _parse_log_entry(self, log_text: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Parse individual log entry and extract features.

        Args:
            log_text: Log entry text to parse
            config: Configuration dictionary

        Returns:
            Dictionary of extracted features
        """
        features: dict[str, Any] = {}

        # Try parsing with different log format patterns
        parsed_data = None

        # Try Nginx Log Format first (more specific with mandatory referer and user agent)
        match = self.NGINX_LOG_PATTERN.match(log_text)
        if match:
            parsed_data = match.groupdict()
            features["log_format"] = "nginx"

        # Try Apache Common Log Format (more permissive)
        if not parsed_data:
            match = self.APACHE_COMMON_LOG_PATTERN.match(log_text)
            if match:
                parsed_data = match.groupdict()
                features["log_format"] = "apache_common"

        # Try Firewall Log Format
        if not parsed_data:
            match = self.FIREWALL_LOG_PATTERN.search(log_text)
            if match:
                parsed_data = match.groupdict()
                features["log_format"] = "firewall"

        if parsed_data:
            # Parse timestamp if present
            if config.get("parse_timestamps", True) and "timestamp" in parsed_data:
                timestamp = self._parse_timestamp(parsed_data["timestamp"])
                if timestamp:
                    features["parsed_timestamp"] = timestamp.isoformat()

            # Extract connection information
            connection_info = self._extract_connection_info(log_text)
            features.update(connection_info)

            # Normalize protocol if present
            if config.get("normalize_protocols", True) and "protocol" in parsed_data:
                features["normalized_protocol"] = self._normalize_protocol(parsed_data["protocol"])

            # URL decode if present and enabled
            if config.get("decode_urls", True) and "url" in parsed_data:
                features["decoded_url"] = unquote(parsed_data["url"])

            # Add raw parsed data
            features["parsed_log_data"] = parsed_data
        else:
            # If no specific format matches, try to extract basic network info
            connection_info = self._extract_connection_info(log_text)
            features.update(connection_info)
            features["log_format"] = "unknown"

        return features

    def _extract_network_features(self, log_text: str) -> dict[str, Any]:
        """
        Extract network-specific features.

        Args:
            log_text: Log text to analyze

        Returns:
            Dictionary of network features
        """
        features: dict[str, Any] = {}

        # Extract IP addresses
        ip_addresses = PreprocessingUtilities.extract_ip_addresses(log_text)
        if ip_addresses:
            features["ip_addresses"] = ip_addresses
            features["ip_count"] = len(ip_addresses)

            # Classify IP addresses
            private_ips = []
            public_ips = []
            for ip in ip_addresses:
                if self._is_private_ip(ip):
                    private_ips.append(ip)
                else:
                    public_ips.append(ip)

            features["private_ips"] = private_ips
            features["public_ips"] = public_ips

        # Extract URLs
        urls = PreprocessingUtilities.extract_urls(log_text)
        if urls:
            features["urls"] = urls
            features["url_count"] = len(urls)

        # Extract ports
        ports = self._extract_ports(log_text)
        if ports:
            features["ports"] = ports
            features["port_count"] = len(ports)

            # Identify services by port
            services = self._identify_services_by_port(ports)
            if services:
                features["identified_services"] = services

        # Extract protocols
        protocols = self._extract_protocols(log_text)
        if protocols:
            features["protocols"] = protocols

        # Extract request/response sizes
        sizes = self._extract_sizes(log_text)
        if sizes:
            features.update(sizes)

        # Extract HTTP methods
        http_methods = self._extract_http_methods(log_text)
        if http_methods:
            features["http_methods"] = http_methods

        # Extract status codes
        status_codes = self._extract_status_codes(log_text)
        if status_codes:
            features["status_codes"] = status_codes

        return features

    def _identify_attack_indicators(self, log_text: str, features: dict[str, Any]) -> list[str]:
        """
        Identify potential attack indicators in log.

        Args:
            log_text: Log text to analyze
            features: Already extracted features

        Returns:
            List of identified attack types
        """
        indicators = []

        # Check each attack type
        for attack_type, patterns in self.ATTACK_INDICATORS.items():
            for pattern in patterns:
                if pattern.search(log_text):
                    indicators.append(attack_type)
                    break  # Only add each attack type once

        # Additional heuristics based on features
        if "status_codes" in features:
            # Look for suspicious status codes
            suspicious_codes = [400, 401, 403, 404, 500, 503]
            if (
                any(code in features["status_codes"] for code in suspicious_codes)
                and len(features["status_codes"]) > 1
            ):
                # Multiple failed requests might indicate scanning
                indicators.append("scanning_activity")

        if "ports" in features:
            # Check for port scanning indicators
            if len(features["ports"]) > 5:  # Many ports accessed
                indicators.append("port_scanning")

            # Check for access to unusual/high-risk ports
            high_risk_ports = [23, 135, 139, 445, 1433, 3389, 5432]
            if any(port in features["ports"] for port in high_risk_ports):
                indicators.append("high_risk_port_access")

        return list(set(indicators))  # Remove duplicates

    def _normalize_protocol(self, protocol: str) -> str:
        """
        Normalize protocol names to standard format.

        Args:
            protocol: Protocol name to normalize

        Returns:
            Normalized protocol name
        """
        if not protocol:
            return "unknown"

        protocol_lower = protocol.lower().strip()

        # Common protocol mappings
        protocol_mappings = {
            "http/1.0": "http",
            "http/1.1": "http",
            "http/2.0": "http",
            "https": "https",
            "tcp": "tcp",
            "udp": "udp",
            "icmp": "icmp",
            "ssh": "ssh",
            "ftp": "ftp",
            "smtp": "smtp",
            "dns": "dns",
            "dhcp": "dhcp",
        }

        return protocol_mappings.get(protocol_lower, protocol_lower)

    def _extract_connection_info(self, log_text: str) -> dict[str, Any]:
        """
        Extract source/destination IP, ports, etc.

        Args:
            log_text: Log text to analyze

        Returns:
            Dictionary of connection information
        """
        connection_info = {}

        # Try to match connection pattern
        match = self.CONNECTION_PATTERN.search(log_text)
        if match:
            connection_info.update(
                {
                    "source_ip": match.group("src_ip"),
                    "source_port": int(match.group("src_port")),
                    "destination_ip": match.group("dst_ip"),
                    "destination_port": int(match.group("dst_port")),
                }
            )

        return connection_info

    def _parse_timestamp(self, timestamp_str: str) -> datetime | None:
        """
        Parse timestamp from log entry.

        Args:
            timestamp_str: Timestamp string to parse

        Returns:
            Parsed datetime object or None if parsing fails
        """
        # Common timestamp formats in network logs
        log_timestamp_formats = [
            "%d/%b/%Y:%H:%M:%S %z",  # Apache format
            "%d/%b/%Y:%H:%M:%S",  # Apache without timezone
            "%Y-%m-%d %H:%M:%S",  # Standard format
            "%b %d %H:%M:%S",  # Syslog format
            "%Y-%m-%dT%H:%M:%S",  # ISO format
        ]

        return PreprocessingUtilities.normalize_timestamp(timestamp_str, log_timestamp_formats)

    def _extract_ports(self, text: str) -> list[int]:
        """Extract port numbers from text."""
        ports = []

        # First, look for connection patterns (IP:port format)
        connection_matches = self.CONNECTION_PATTERN.finditer(text)
        for match in connection_matches:
            src_port = int(match.group("src_port"))
            dst_port = int(match.group("dst_port"))
            if self._is_valid_port(src_port):
                ports.append(src_port)
            if self._is_valid_port(dst_port):
                ports.append(dst_port)

        # Look for firewall log patterns (more specific)
        spt_matches = re.finditer(r"\bSPT=(\d+)\b", text, re.IGNORECASE)
        for match in spt_matches:
            port = int(match.group(1))
            if self._is_valid_port(port):
                ports.append(port)

        dpt_matches = re.finditer(r"\bDPT=(\d+)\b", text, re.IGNORECASE)
        for match in dpt_matches:
            port = int(match.group(1))
            if self._is_valid_port(port):
                ports.append(port)

        # Look for explicit port mentions
        port_mentions = re.finditer(r"\bports?[:\s]*(\d+(?:,\s*\d+)*)\b", text, re.IGNORECASE)
        for match in port_mentions:
            port_str = match.group(1)
            if "," in port_str:
                for p in port_str.split(","):
                    try:
                        port = int(p.strip())
                        if self._is_valid_port(port):
                            ports.append(port)
                    except ValueError:
                        continue
            else:
                try:
                    port = int(port_str)
                    if self._is_valid_port(port):
                        ports.append(port)
                except ValueError:
                    continue

        # Look for individual port mentions
        individual_ports = re.finditer(r"\bport\s+(\d+)\b", text, re.IGNORECASE)
        for match in individual_ports:
            try:
                port = int(match.group(1))
                if self._is_valid_port(port):
                    ports.append(port)
            except ValueError:
                continue

        return list(set(ports))  # Remove duplicates

    def _is_valid_port(self, port: int) -> bool:
        """Check if a port number is valid and relevant."""
        # Accept high ports (registered/dynamic)
        if 1024 <= port <= 65535:
            return True
        # Accept common well-known ports
        return port in [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]

    def _extract_protocols(self, text: str) -> list[str]:
        """Extract protocol names from text."""
        protocol_pattern = re.compile(
            r"\b(TCP|UDP|ICMP|HTTP|HTTPS|SSH|FTP|SMTP|DNS)\b", re.IGNORECASE
        )
        protocols = []

        for match in protocol_pattern.finditer(text):
            protocols.append(match.group(1).upper())

        return list(set(protocols))

    def _extract_sizes(self, text: str) -> dict[str, Any]:
        """Extract request/response sizes from text."""
        sizes: dict[str, Any] = {}

        # Look for size indicators
        size_pattern = re.compile(r"\b(\d+)\s*(?:bytes?|B)\b", re.IGNORECASE)
        matches = size_pattern.findall(text)

        if matches:
            size_values = [int(match) for match in matches]
            sizes["request_sizes"] = size_values
            sizes["total_size"] = sum(size_values)
            sizes["avg_size"] = sum(size_values) / len(size_values)
            sizes["max_size"] = max(size_values)

        return sizes

    def _extract_http_methods(self, text: str) -> list[str]:
        """Extract HTTP methods from text."""
        method_pattern = re.compile(r"\b(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE|CONNECT)\b")
        methods = []

        for match in method_pattern.finditer(text):
            methods.append(match.group(1))

        return list(set(methods))

    def _extract_status_codes(self, text: str) -> list[int]:
        """Extract HTTP status codes from text."""
        status_pattern = re.compile(r"\b([1-5]\d{2})\b")
        codes = []

        for match in status_pattern.finditer(text):
            code = int(match.group(1))
            codes.append(code)

        return list(set(codes))

    def _identify_services_by_port(self, ports: list[int]) -> list[str]:
        """Identify services based on port numbers."""
        services = []

        for port in ports:
            for service, service_ports in self.PORT_PATTERNS.items():
                if port in service_ports:
                    services.append(service)

        return list(set(services))

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private."""
        try:
            import ipaddress

            return ipaddress.IPv4Address(ip).is_private
        except (ImportError, ValueError):
            # Fallback to simple string matching for private ranges
            # RFC 1918 private ranges: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            # Plus localhost 127.0.0.0/8
            return (
                ip.startswith("192.168.")
                or ip.startswith("10.")
                or ip.startswith("127.")
                or
                # 172.16.0.0 to 172.31.255.255
                any(ip.startswith(f"172.{i}.") for i in range(16, 32))
            )

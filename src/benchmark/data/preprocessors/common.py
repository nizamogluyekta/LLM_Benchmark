"""
Common preprocessing utilities for the LLM Cybersecurity Benchmark system.

This module provides reusable utilities for text cleaning, normalization,
feature extraction, and other common preprocessing operations.
"""

import asyncio
import re
import unicodedata
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from benchmark.data.models import DatasetSample
from benchmark.data.preprocessors.base import PreprocessingProgress


class PreprocessingUtilities:
    """
    Collection of common preprocessing utility functions.

    These functions are designed to be used across different preprocessors
    to ensure consistent data cleaning and normalization.
    """

    # Common timestamp formats
    TIMESTAMP_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%Y %H:%M:%S",
        "%Y%m%d %H%M%S",
        "%Y%m%d_%H%M%S",
    ]

    # IP address patterns
    IPV4_PATTERN = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )

    IPV6_PATTERN = re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
        r"\b::1\b|"
        r"\b::ffff:[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b"
    )

    # URL pattern
    URL_PATTERN = re.compile(
        r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?",
        re.IGNORECASE,
    )

    # Email pattern
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    # Hash patterns
    MD5_PATTERN = re.compile(r"\b[a-fA-F0-9]{32}\b")
    SHA1_PATTERN = re.compile(r"\b[a-fA-F0-9]{40}\b")
    SHA256_PATTERN = re.compile(r"\b[a-fA-F0-9]{64}\b")

    # Domain pattern
    DOMAIN_PATTERN = re.compile(
        r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"
    )

    @staticmethod
    def clean_text(text: str, options: dict[str, Any] | None = None) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Text to clean
            options: Cleaning options:
                - remove_extra_whitespace: Remove extra whitespace (default: True)
                - normalize_unicode: Normalize unicode characters (default: True)
                - lowercase: Convert to lowercase (default: False)
                - remove_non_printable: Remove non-printable characters (default: True)
                - preserve_newlines: Keep newline characters (default: False)

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Default options
        opts = {
            "remove_extra_whitespace": True,
            "normalize_unicode": True,
            "lowercase": False,
            "remove_non_printable": True,
            "preserve_newlines": False,
        }
        if options:
            opts.update(options)

        cleaned = text

        # Normalize unicode
        if opts["normalize_unicode"]:
            cleaned = unicodedata.normalize("NFKC", cleaned)

        # Remove non-printable characters
        if opts["remove_non_printable"]:
            if opts["preserve_newlines"]:
                # Keep newlines and tabs
                cleaned = "".join(c for c in cleaned if c.isprintable() or c in "\n\t")
            else:
                # Replace non-printable whitespace with spaces, remove other non-printable
                result = []
                for c in cleaned:
                    if c.isprintable():
                        result.append(c)
                    elif c.isspace():
                        result.append(" ")
                cleaned = "".join(result)

        # Convert to lowercase
        if opts["lowercase"]:
            cleaned = cleaned.lower()

        # Remove extra whitespace
        if opts["remove_extra_whitespace"]:
            if opts["preserve_newlines"]:
                # Preserve line structure but clean up spaces
                lines = cleaned.split("\n")
                lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
                cleaned = "\n".join(lines)
            else:
                cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    @staticmethod
    def normalize_timestamp(
        timestamp_str: str, formats: list[str] | None = None, default_timezone: str = "UTC"
    ) -> datetime | None:
        """
        Parse timestamp from various formats.

        Args:
            timestamp_str: Timestamp string to parse
            formats: List of formats to try (uses default formats if None)
            default_timezone: Default timezone for naive timestamps

        Returns:
            Parsed datetime object or None if parsing fails
        """
        if not timestamp_str or not isinstance(timestamp_str, str):
            return None

        # Use provided formats or default ones
        formats_to_try = formats or PreprocessingUtilities.TIMESTAMP_FORMATS

        # Clean the timestamp string
        timestamp_str = timestamp_str.strip()

        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                # Add timezone info if naive
                if dt.tzinfo is None:
                    import pytz  # type: ignore[import-untyped]

                    tz = pytz.timezone(default_timezone)
                    dt = tz.localize(dt)
                return dt
            except (ValueError, ImportError):
                continue

        # Try parsing with dateutil as fallback
        try:
            from dateutil import parser  # type: ignore[import-untyped]

            return parser.parse(timestamp_str)  # type: ignore[no-any-return]
        except (ImportError, ValueError):
            pass

        return None

    @staticmethod
    def extract_ip_addresses(text: str, include_private: bool = True) -> list[str]:
        """
        Extract IP addresses from text.

        Args:
            text: Text to extract IPs from
            include_private: Whether to include private IP addresses

        Returns:
            List of unique IP addresses found
        """
        if not text:
            return []

        ips = set()

        # Find IPv4 addresses
        ipv4_matches = PreprocessingUtilities.IPV4_PATTERN.findall(text)
        for ip in ipv4_matches:
            if include_private or not PreprocessingUtilities._is_private_ipv4(ip):
                ips.add(ip)

        # Find IPv6 addresses
        ipv6_matches = PreprocessingUtilities.IPV6_PATTERN.findall(text)
        for ip in ipv6_matches:
            if include_private or not PreprocessingUtilities._is_private_ipv6(ip):
                ips.add(ip)

        return sorted(ips)

    @staticmethod
    def extract_urls(text: str, validate: bool = True) -> list[str]:
        """
        Extract URLs from text.

        Args:
            text: Text to extract URLs from
            validate: Whether to validate URL structure

        Returns:
            List of unique URLs found
        """
        if not text:
            return []

        urls = set()
        matches = PreprocessingUtilities.URL_PATTERN.findall(text)

        for url in matches:
            if validate:
                try:
                    parsed = urlparse(url)
                    if parsed.netloc and parsed.scheme:
                        urls.add(url)
                except Exception:
                    continue
            else:
                urls.add(url)

        return sorted(urls)

    @staticmethod
    def extract_email_addresses(text: str) -> list[str]:
        """
        Extract email addresses from text.

        Args:
            text: Text to extract emails from

        Returns:
            List of unique email addresses found
        """
        if not text:
            return []

        emails = set(PreprocessingUtilities.EMAIL_PATTERN.findall(text))
        return sorted(emails)

    @staticmethod
    def extract_hashes(text: str, hash_types: list[str] | None = None) -> dict[str, list[str]]:
        """
        Extract hash values from text.

        Args:
            text: Text to extract hashes from
            hash_types: List of hash types to extract ('md5', 'sha1', 'sha256')

        Returns:
            Dictionary mapping hash types to lists of found hashes
        """
        if not text:
            return {}

        hash_types = hash_types or ["md5", "sha1", "sha256"]
        results = {}

        for hash_type in hash_types:
            if hash_type.lower() == "md5":
                results["md5"] = list(set(PreprocessingUtilities.MD5_PATTERN.findall(text)))
            elif hash_type.lower() == "sha1":
                results["sha1"] = list(set(PreprocessingUtilities.SHA1_PATTERN.findall(text)))
            elif hash_type.lower() == "sha256":
                results["sha256"] = list(set(PreprocessingUtilities.SHA256_PATTERN.findall(text)))

        return results

    @staticmethod
    def extract_domains(text: str, exclude_ips: bool = True) -> list[str]:
        """
        Extract domain names from text.

        Args:
            text: Text to extract domains from
            exclude_ips: Whether to exclude IP addresses

        Returns:
            List of unique domain names found
        """
        if not text:
            return []

        domains = set(PreprocessingUtilities.DOMAIN_PATTERN.findall(text))

        if exclude_ips:
            # Remove any IP addresses that might match domain pattern
            domains = {
                domain
                for domain in domains
                if not PreprocessingUtilities.IPV4_PATTERN.match(domain)
            }

        return sorted(domains)

    @staticmethod
    def normalize_attack_labels(label: str, custom_mappings: dict[str, str] | None = None) -> str:
        """
        Normalize attack type labels to standard format.

        Args:
            label: Attack label to normalize
            custom_mappings: Additional custom mappings

        Returns:
            Normalized attack label
        """
        if not label:
            return ""

        # Default mappings for common attack types
        default_mappings = {
            # SQL Injection variants
            "sqli": "sql_injection",
            "sql-injection": "sql_injection",
            "sql injection": "sql_injection",
            "sqlinjection": "sql_injection",
            # Cross-site scripting variants
            "xss": "cross_site_scripting",
            "cross-site-scripting": "cross_site_scripting",
            "cross site scripting": "cross_site_scripting",
            "crosssitescripting": "cross_site_scripting",
            # Command injection variants
            "cmd injection": "command_injection",
            "command-injection": "command_injection",
            "commandinjection": "command_injection",
            "code injection": "command_injection",
            # DDoS variants
            "ddos": "distributed_denial_of_service",
            "dos": "denial_of_service",
            "denial-of-service": "denial_of_service",
            "distributed-dos": "distributed_denial_of_service",
            # Malware variants
            "malware": "malware",
            "virus": "malware",
            "trojan": "malware",
            "ransomware": "ransomware",
            "spyware": "malware",
            "adware": "malware",
            # Phishing variants
            "phish": "phishing",
            "phishing": "phishing",
            "social engineering": "social_engineering",
            # Brute force variants
            "brute-force": "brute_force",
            "brute force": "brute_force",
            "bruteforce": "brute_force",
            "password attack": "brute_force",
            # Network intrusion variants
            "intrusion": "network_intrusion",
            "network intrusion": "network_intrusion",
            "network-intrusion": "network_intrusion",
            "unauthorized access": "unauthorized_access",
        }

        # Apply custom mappings if provided
        mappings = default_mappings.copy()
        if custom_mappings:
            mappings.update(custom_mappings)

        # Normalize input
        normalized = label.lower().strip().replace("_", " ").replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)

        # Check direct mappings
        if normalized in mappings:
            return mappings[normalized]

        # Check if any mapping key is contained in the label
        for key, value in mappings.items():
            if key in normalized or normalized in key:
                return value

        # Return cleaned version if no mapping found
        return normalized.replace(" ", "_")

    @staticmethod
    async def process_batch(
        samples: list[DatasetSample],
        processor_func: Callable[[DatasetSample], DatasetSample],
        batch_size: int = 100,
        progress_callback: Callable[[PreprocessingProgress], None] | None = None,
    ) -> list[DatasetSample]:
        """
        Process samples in batches with progress reporting.

        Args:
            samples: List of samples to process
            processor_func: Function to apply to each sample
            batch_size: Number of samples to process in each batch
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed samples
        """
        if not samples:
            return []

        # Setup progress tracking
        progress = PreprocessingProgress(len(samples))
        if progress_callback:
            progress.add_callback(progress_callback)
        progress.start()

        processed_samples = []
        failed_samples = 0

        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            batch_processed = []
            batch_failed = 0

            for sample in batch:
                try:
                    processed_sample = processor_func(sample)
                    batch_processed.append(processed_sample)
                except Exception:
                    # Skip failed samples but count them
                    batch_failed += 1
                    continue

            processed_samples.extend(batch_processed)
            failed_samples += batch_failed

            # Update progress
            progress.update(len(batch_processed), batch_failed)

            # Yield control to allow other async operations
            await asyncio.sleep(0)

        progress.complete()
        return processed_samples

    @staticmethod
    def _is_private_ipv4(ip: str) -> bool:
        """Check if IPv4 address is private."""
        try:
            import ipaddress

            return ipaddress.IPv4Address(ip).is_private
        except (ImportError, ValueError):
            # Fallback to simple string matching
            return (
                ip.startswith("192.168.")
                or ip.startswith("10.")
                or ip.startswith("172.16.")
                or ip.startswith("172.17.")
                or ip.startswith("172.18.")
                or ip.startswith("172.19.")
                or ip.startswith("172.2")
                or ip.startswith("172.30.")
                or ip.startswith("172.31.")
                or ip.startswith("127.")
            )

    @staticmethod
    def _is_private_ipv6(ip: str) -> bool:
        """Check if IPv6 address is private."""
        try:
            import ipaddress

            addr = ipaddress.IPv6Address(ip)
            return addr.is_private or addr.is_link_local or addr.is_loopback
        except (ImportError, ValueError):
            # Fallback - assume all IPv6 are public for simplicity
            return False

    @staticmethod
    def extract_features(
        text: str, feature_types: list[str] | None = None, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Extract multiple types of features from text in one pass.

        Args:
            text: Text to extract features from
            feature_types: Types of features to extract
            options: Additional options for feature extraction

        Returns:
            Dictionary mapping feature types to extracted values
        """
        if not text:
            return {}

        feature_types = feature_types or ["ips", "urls", "emails", "domains", "hashes"]
        options = options or {}

        features: dict[str, Any] = {}

        if "ips" in feature_types:
            features["ip_addresses"] = PreprocessingUtilities.extract_ip_addresses(
                text, include_private=options.get("include_private_ips", True)
            )

        if "urls" in feature_types:
            features["urls"] = PreprocessingUtilities.extract_urls(
                text, validate=options.get("validate_urls", True)
            )

        if "emails" in feature_types:
            features["email_addresses"] = PreprocessingUtilities.extract_email_addresses(text)

        if "domains" in feature_types:
            features["domains"] = PreprocessingUtilities.extract_domains(
                text, exclude_ips=options.get("exclude_ip_domains", True)
            )

        if "hashes" in feature_types:
            features["hashes"] = PreprocessingUtilities.extract_hashes(
                text, hash_types=options.get("hash_types", ["md5", "sha1", "sha256"])
            )

        return features

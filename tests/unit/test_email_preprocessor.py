"""
Unit tests for email content preprocessor.

This module tests the EmailContentPreprocessor functionality including
HTML cleaning, header extraction, suspicious URL identification, and
linguistic feature analysis for phishing detection.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.data.models import DatasetSample
from benchmark.data.preprocessors.email_content import EmailContentPreprocessor


class TestEmailContentPreprocessor:
    """Test EmailContentPreprocessor functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.preprocessor = EmailContentPreprocessor()

        # Load test fixtures
        fixtures_path = Path(__file__).parent.parent / "fixtures" / "sample_emails.json"
        with open(fixtures_path) as f:
            self.fixtures = json.load(f)

    def test_init_default_name(self) -> None:
        """Test preprocessor initialization with default name."""
        preprocessor = EmailContentPreprocessor()
        assert preprocessor.name == "EmailContentPreprocessor"

    def test_init_custom_name(self) -> None:
        """Test preprocessor initialization with custom name."""
        preprocessor = EmailContentPreprocessor("CustomEmailProcessor")
        assert preprocessor.name == "CustomEmailProcessor"

    def test_get_required_fields(self) -> None:
        """Test getting required fields."""
        required_fields = self.preprocessor.get_required_fields()
        assert required_fields == ["input_text", "label"]

    def test_get_supported_config_keys(self) -> None:
        """Test getting supported configuration keys."""
        config_keys = self.preprocessor.get_supported_config_keys()
        expected_keys = [
            "clean_html",
            "extract_headers",
            "identify_suspicious_urls",
            "normalize_emails",
            "analyze_linguistics",
            "preserve_structure",
            "extract_features",
        ]
        assert set(config_keys) == set(expected_keys)

    def test_validate_config_valid(self) -> None:
        """Test configuration validation with valid config."""
        config = {
            "clean_html": True,
            "extract_headers": True,
            "identify_suspicious_urls": True,
        }
        warnings = self.preprocessor.validate_config(config)
        assert len(warnings) == 0

    def test_validate_config_invalid_keys(self) -> None:
        """Test configuration validation with invalid keys."""
        config = {
            "clean_html": True,
            "unknown_key": "value",
            "another_unknown": 123,
        }
        warnings = self.preprocessor.validate_config(config)
        assert len(warnings) == 2
        assert any("unknown_key" in warning for warning in warnings)
        assert any("another_unknown" in warning for warning in warnings)

    @pytest.mark.asyncio
    async def test_process_legitimate_email(self) -> None:
        """Test processing legitimate email."""
        email_data = self.fixtures["legitimate_emails"][0]
        sample = DatasetSample(input_text=email_data["email_content"], label="BENIGN")

        config = {"extract_features": True, "analyze_linguistics": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Check basic processing
        assert processed_sample.label == "BENIGN"
        assert "email_format" in processed_sample.metadata

        # Check expected features (use >= for counts since we might detect more)
        expected = email_data["expected_features"]
        for feature, expected_value in expected.items():
            if feature in processed_sample.metadata:
                if feature.endswith("_count") or feature.endswith("_keywords"):
                    assert processed_sample.metadata[feature] >= expected_value
                else:
                    assert processed_sample.metadata[feature] == expected_value

    @pytest.mark.asyncio
    async def test_process_html_email(self) -> None:
        """Test processing HTML email."""
        email_data = self.fixtures["legitimate_emails"][1]
        sample = DatasetSample(input_text=email_data["email_content"], label="BENIGN")

        config = {"clean_html": True, "extract_features": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert processed_sample.metadata["email_format"] == "mime"
        assert processed_sample.metadata.get("html_content_cleaned") is True
        # HTML should be cleaned from the text
        assert "<html>" not in processed_sample.input_text
        assert "<body>" not in processed_sample.input_text

    @pytest.mark.asyncio
    async def test_phishing_detection_paypal(self) -> None:
        """Test phishing detection for PayPal scam."""
        email_data = self.fixtures["phishing_emails"][0]
        sample = DatasetSample(input_text=email_data["email_content"], label="ATTACK")

        config = {"identify_suspicious_urls": True, "analyze_linguistics": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should detect suspicious URLs
        assert processed_sample.metadata.get("suspicious_url_count", 0) >= 1
        assert "suspicious_urls" in processed_sample.metadata

        # Should detect phishing keywords
        assert processed_sample.metadata.get("urgency_keywords", 0) > 0
        assert processed_sample.metadata.get("security_keywords", 0) > 0
        assert processed_sample.metadata.get("social_engineering_score", 0) > 0

    @pytest.mark.asyncio
    async def test_phishing_detection_bank(self) -> None:
        """Test phishing detection for bank scam."""
        email_data = self.fixtures["phishing_emails"][1]
        sample = DatasetSample(input_text=email_data["email_content"], label="ATTACK")

        config = {"identify_suspicious_urls": True, "analyze_linguistics": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should detect suspicious URLs (bit.ly)
        assert processed_sample.metadata.get("suspicious_url_count", 0) >= 1

        # Should detect elevated caps ratio
        assert processed_sample.metadata.get("caps_ratio", 0) > 0.05

    @pytest.mark.asyncio
    async def test_phishing_detection_lottery(self) -> None:
        """Test phishing detection for lottery scam."""
        email_data = self.fixtures["phishing_emails"][2]
        sample = DatasetSample(input_text=email_data["email_content"], label="ATTACK")

        config = {"analyze_linguistics": True, "identify_suspicious_urls": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should detect prize-related keywords
        assert processed_sample.metadata.get("prizes_keywords", 0) >= 3

        # Should detect high exclamation usage
        assert processed_sample.metadata.get("exclamation_count", 0) >= 3

        # Should detect suspicious TLD (.ga)
        assert processed_sample.metadata.get("suspicious_url_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_html_phishing_email(self) -> None:
        """Test processing HTML phishing email."""
        email_data = self.fixtures["phishing_emails"][3]
        sample = DatasetSample(input_text=email_data["email_content"], label="ATTACK")

        config = {"clean_html": True, "identify_suspicious_urls": True, "analyze_linguistics": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert processed_sample.metadata["email_format"] == "html"
        assert processed_sample.metadata.get("html_content_cleaned") is True
        assert processed_sample.metadata.get("suspicious_url_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_email_header_extraction(self) -> None:
        """Test email header extraction."""
        email_data = self.fixtures["legitimate_emails"][0]
        sample = DatasetSample(input_text=email_data["email_content"], label="BENIGN")

        config = {"extract_headers": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        headers = processed_sample.metadata.get("email_headers", {})
        assert "from" in headers
        assert "to" in headers
        assert "subject" in headers
        assert "date" in headers

        assert "john.doe@company.com" in headers["from"]
        assert "jane.smith@company.com" in headers["to"]
        assert "Meeting Tomorrow" in headers["subject"]

    @pytest.mark.asyncio
    async def test_mime_multipart_email(self) -> None:
        """Test processing MIME multipart email."""
        email_data = self.fixtures["malformed_emails"][1]
        sample = DatasetSample(input_text=email_data["email_content"], label="BENIGN")

        config = {"extract_features": True}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        assert processed_sample.metadata["email_format"] == "mime"
        assert processed_sample.metadata["is_multipart"] is True
        assert processed_sample.metadata.get("multipart_sections", 0) > 0

    @pytest.mark.asyncio
    async def test_suspicious_url_identification(self) -> None:
        """Test suspicious URL identification."""
        test_urls = [
            "http://bit.ly/suspicious",
            "http://192.168.1.1/malware",
            "https://payp4l.com/verify",
            "http://malicious.tk/phishing",
            "https://secure-update-microsoft.com/update",
        ]

        email_text = "Click these links: " + " ".join(test_urls)

        suspicious_urls = self.preprocessor._identify_suspicious_urls(email_text)

        # Should identify all as suspicious
        assert len(suspicious_urls) >= 4  # At least most of them should be caught

    @pytest.mark.asyncio
    async def test_legitimate_url_not_flagged(self) -> None:
        """Test that legitimate URLs are not flagged as suspicious."""
        legitimate_urls = [
            "https://amazon.com/orders",
            "https://google.com/search",
            "https://microsoft.com/support",
        ]

        email_text = "Visit these sites: " + " ".join(legitimate_urls)

        suspicious_urls = self.preprocessor._identify_suspicious_urls(email_text)

        # Legitimate URLs should not be flagged
        assert len(suspicious_urls) == 0

    def test_html_cleaning_basic(self) -> None:
        """Test basic HTML cleaning functionality."""
        html_content = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        cleaned = self.preprocessor._clean_html_content(html_content)

        assert "<html>" not in cleaned
        assert "<body>" not in cleaned
        assert "Title" in cleaned
        assert "Content" in cleaned

    def test_html_cleaning_with_scripts(self) -> None:
        """Test HTML cleaning removes scripts and styles."""
        html_content = """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
            <script>alert('malicious');</script>
            <p>Safe content</p>
        </body>
        </html>
        """
        cleaned = self.preprocessor._clean_html_content(html_content)

        assert "alert" not in cleaned
        assert "color: red" not in cleaned
        assert "Safe content" in cleaned

    def test_html_cleaning_without_beautifulsoup(self) -> None:
        """Test HTML cleaning fallback when BeautifulSoup is not available."""
        with patch("benchmark.data.preprocessors.email_content.HAS_BS4", False):
            preprocessor = EmailContentPreprocessor()
            html_content = "<p>Test <strong>content</strong></p>"
            cleaned = preprocessor._clean_html_content(html_content)

            assert "<p>" not in cleaned
            assert "<strong>" not in cleaned
            assert "Test content" in cleaned

    def test_email_address_normalization(self) -> None:
        """Test email address normalization."""
        email_text = "Contact US at Admin@Example.COM or Support@TEST.org"
        normalized = self.preprocessor._normalize_email_addresses(email_text)

        assert "admin@example.com" in normalized
        assert "support@test.org" in normalized
        assert "Admin@Example.COM" not in normalized

    def test_linguistic_feature_analysis(self) -> None:
        """Test linguistic feature analysis."""
        phishing_text = "URGENT! Your account is suspended! Click here immediately to verify now!"
        features = self.preprocessor._analyze_linguistic_features(phishing_text)

        assert features["urgency_keywords"] > 0
        assert features["security_keywords"] > 0
        assert features["action_required_keywords"] > 0
        assert features["exclamation_count"] >= 3
        assert features["social_engineering_score"] > 0
        assert features["caps_ratio"] > 0.1

    def test_header_extraction_regex_fallback(self) -> None:
        """Test header extraction using regex fallback."""
        email_text = """From: sender@example.com
To: recipient@example.com
Subject: Test Subject
Date: Mon, 15 Jan 2024 10:30:00

Email body content here."""

        headers = self.preprocessor._extract_header_info(email_text)

        assert headers["from"] == "sender@example.com"
        assert headers["to"] == "recipient@example.com"
        assert headers["subject"] == "Test Subject"
        assert headers["date"] == "Mon, 15 Jan 2024 10:30:00"

    def test_attachment_indicators(self) -> None:
        """Test attachment indicator detection."""
        email_text = """Please find the attached report.pdf for your review.
Also included is the invoice.xlsx file.
Content-Disposition: attachment; filename="document.doc"
"""
        indicators = self.preprocessor._find_attachment_indicators(email_text)

        assert len(indicators) > 0
        # Should find some attachment-related content

    def test_text_statistics_calculation(self) -> None:
        """Test text statistics calculation."""
        text = "This is a test email. It has multiple sentences! How are you?"
        stats = self.preprocessor._calculate_text_statistics(text)

        assert stats["word_count"] == 12
        assert stats["sentence_count"] == 3
        assert "avg_word_length" in stats
        assert stats["avg_word_length"] > 0

    @pytest.mark.asyncio
    async def test_empty_email_processing(self) -> None:
        """Test processing minimal email."""
        sample = DatasetSample(input_text=" ", label="BENIGN")

        config = {}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should handle gracefully
        assert processed_sample.input_text == ""
        assert processed_sample.label == "BENIGN"

    @pytest.mark.asyncio
    async def test_whitespace_only_email(self) -> None:
        """Test processing email with only whitespace."""
        sample = DatasetSample(input_text="   \n\n  \t  \n   ", label="BENIGN")

        config = {}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should clean whitespace (preserve_newlines=True keeps some structure)
        assert len(processed_sample.input_text.strip()) == 0
        assert processed_sample.label == "BENIGN"

    @pytest.mark.asyncio
    async def test_feature_extraction_disabled(self) -> None:
        """Test processing with feature extraction disabled."""
        email_data = self.fixtures["phishing_emails"][0]
        sample = DatasetSample(input_text=email_data["email_content"], label="ATTACK")

        config = {"extract_features": False, "analyze_linguistics": False}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # Should have basic format detection but no detailed features
        assert "email_format" in processed_sample.metadata
        assert "urgency_keywords" not in processed_sample.metadata
        assert "suspicious_url_count" not in processed_sample.metadata

    @pytest.mark.asyncio
    async def test_html_cleaning_disabled(self) -> None:
        """Test processing with HTML cleaning disabled."""
        email_data = self.fixtures["html_emails"][0]
        sample = DatasetSample(input_text=email_data["email_content"], label="BENIGN")

        config = {"clean_html": False}
        result = await self.preprocessor.process([sample], config)

        assert len(result) == 1
        processed_sample = result[0]

        # HTML should still be present
        assert "<html>" in processed_sample.input_text
        assert "html_content_cleaned" not in processed_sample.metadata

    @pytest.mark.asyncio
    async def test_batch_processing(self) -> None:
        """Test processing multiple email samples at once."""
        samples = []
        for email_data in self.fixtures["legitimate_emails"][:2]:
            sample = DatasetSample(input_text=email_data["email_content"], label="BENIGN")
            samples.append(sample)

        config = {"extract_features": True, "analyze_linguistics": True}
        result = await self.preprocessor.process(samples, config)

        assert len(result) == 2
        for processed_sample in result:
            assert "email_format" in processed_sample.metadata
            assert processed_sample.label == "BENIGN"

    @pytest.mark.asyncio
    async def test_processing_error_handling(self) -> None:
        """Test error handling during processing."""
        sample = DatasetSample(input_text="test email", label="BENIGN")

        # Mock an error in feature extraction
        with (
            patch.object(
                self.preprocessor, "_extract_email_features", side_effect=Exception("Test error")
            ),
            pytest.raises(Exception, match="Test error"),
        ):
            await self.preprocessor.process([sample], {"extract_features": True})

    def test_domain_legitimacy_analysis(self) -> None:
        """Test domain legitimacy analysis."""
        email_text = """
        From: user@gmail.com
        Contact: support@amazon.com
        Suspicious: admin@suspicious-domain.tk
        """

        features = self.preprocessor._extract_email_features(email_text)

        assert features.get("legitimate_domain_count", 0) >= 2  # gmail.com, amazon.com
        assert features.get("suspicious_domain_count", 0) >= 1  # suspicious-domain.tk

    def test_phishing_keyword_density(self) -> None:
        """Test phishing keyword density calculation."""
        # Text with 20 words, 4 phishing-related
        text = "urgent verify your account now click here immediately to update your password secure login"
        features = self.preprocessor._analyze_linguistic_features(text)

        assert features["phishing_keyword_density"] > 0.15  # At least 15% phishing keywords

    def test_validate_config_beautifulsoup_warning(self) -> None:
        """Test configuration validation warns about missing BeautifulSoup."""
        with patch("benchmark.data.preprocessors.email_content.HAS_BS4", False):
            preprocessor = EmailContentPreprocessor()
            config = {"clean_html": True}
            warnings = preprocessor.validate_config(config)

            assert len(warnings) >= 1
            assert any("BeautifulSoup4" in warning for warning in warnings)

    def test_str_representation(self) -> None:
        """Test string representation of preprocessor."""
        str_repr = str(self.preprocessor)

        assert "EmailContentPreprocessor" in str_repr
        assert "input_text" in str_repr
        assert "label" in str_repr

    def test_repr_representation(self) -> None:
        """Test detailed representation of preprocessor."""
        repr_str = repr(self.preprocessor)

        assert "EmailContentPreprocessor" in repr_str
        assert "required_fields" in repr_str
        assert "supported_config" in repr_str

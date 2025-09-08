"""
Email content preprocessor for the LLM Cybersecurity Benchmark system.

This module provides specialized preprocessing for email content, focusing on
phishing detection use cases. It handles HTML cleaning, header extraction,
suspicious URL identification, and linguistic analysis.
"""

import email
import email.message
import re
import statistics
from email.header import decode_header
from typing import Any
from urllib.parse import urlparse

from benchmark.data.models import DatasetSample
from benchmark.data.preprocessors.base import DataPreprocessor, PreprocessorError
from benchmark.data.preprocessors.common import PreprocessingUtilities

try:
    from bs4 import BeautifulSoup, NavigableString

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class EmailContentPreprocessor(DataPreprocessor):
    """
    Specialized preprocessor for email content analysis.

    Handles cleaning HTML content, extracting email headers, identifying
    suspicious URLs, and analyzing linguistic features for phishing detection.
    """

    # Suspicious URL patterns for phishing detection
    SUSPICIOUS_URL_PATTERNS = [
        re.compile(
            r"^(bit\.ly|tinyurl\.com|short\.link|t\.co)$", re.IGNORECASE
        ),  # URL shorteners (domain only)
        re.compile(
            r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", re.IGNORECASE
        ),  # IP addresses
        re.compile(r"[a-z0-9-]+\.tk|\.ml|\.ga|\.cf", re.IGNORECASE),  # Suspicious TLDs
        re.compile(
            r"\.(exe|scr|bat|pif)(?:/|$)", re.IGNORECASE
        ),  # Executable file extensions (not .com domain)
        re.compile(r"secure[_-]?update|account[_-]?verification|urgent[_-]?action", re.IGNORECASE),
        re.compile(
            r"payp[a4]l(?!\.com)|g[o0][o0]gle(?!\.com)|micr[o0]s[o0]ft(?!\.com)|[a4]m[a4]z[o0]n(?!\.com)",
            re.IGNORECASE,
        ),  # Typosquatting (but not legitimate domains)
    ]

    # Phishing keywords for linguistic analysis
    PHISHING_KEYWORDS = {
        "urgency": [
            "urgent",
            "immediate",
            "expire",
            "suspended",
            "deadline",
            "asap",
            "limited time",
            "act now",
            "final notice",
            "last chance",
        ],
        "financial": [
            "bank",
            "credit",
            "payment",
            "transaction",
            "refund",
            "billing",
            "invoice",
            "account",
            "verify",
            "update payment",
        ],
        "security": [
            "security alert",
            "breach",
            "compromised",
            "unauthorized",
            "suspicious",
            "locked",
            "verify identity",
            "confirm",
            "validate",
            "suspended",
            "security",
        ],
        "action_required": [
            "click here",
            "verify now",
            "update now",
            "confirm identity",
            "login",
            "sign in",
            "download",
            "install",
        ],
        "prizes": [
            "winner",
            "congratulations",
            "prize",
            "lottery",
            "free",
            "gift",
            "reward",
            "bonus",
            "cash",
            "money",
        ],
    }

    # Common legitimate email domains
    LEGITIMATE_DOMAINS = {
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "aol.com",
        "amazon.com",
        "microsoft.com",
        "google.com",
        "apple.com",
        "facebook.com",
        "paypal.com",
        "ebay.com",
        "twitter.com",
        "linkedin.com",
        "instagram.com",
        "github.com",
    }

    def __init__(self, name: str | None = None):
        """Initialize the email content preprocessor."""
        super().__init__(name)

        if not HAS_BS4:
            self.logger.warning(
                "BeautifulSoup4 not available. HTML parsing will be limited. "
                "Install with: pip install beautifulsoup4 lxml"
            )

    async def process(
        self, samples: list[DatasetSample], config: dict[str, Any]
    ) -> list[DatasetSample]:
        """
        Process email content samples.

        Args:
            samples: List of samples to process
            config: Configuration dictionary with options:
                - clean_html: Clean HTML content (default: True)
                - extract_headers: Extract email headers (default: True)
                - identify_suspicious_urls: Identify suspicious URLs (default: True)
                - normalize_emails: Normalize email addresses (default: True)
                - analyze_linguistics: Analyze linguistic features (default: True)
                - preserve_structure: Preserve email structure info (default: True)

        Returns:
            List of processed samples with email-specific metadata

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

                # Parse email content and extract basic structure
                email_features = await self._parse_email_content(sample.input_text, config)

                # Add extracted features to metadata
                processed_sample.metadata.update(email_features)

                # Extract email headers if enabled
                if config.get("extract_headers", True):
                    headers = self._extract_header_info(sample.input_text)
                    if headers:
                        processed_sample.metadata["email_headers"] = headers

                # Clean HTML content if enabled
                cleaned_text = sample.input_text
                if config.get("clean_html", True):
                    cleaned_text = self._clean_html_content(sample.input_text)
                    if cleaned_text != sample.input_text:
                        processed_sample.metadata["html_content_cleaned"] = True

                # Extract email-specific features if enabled
                if config.get("extract_features", True):
                    email_specific_features = self._extract_email_features(cleaned_text)
                    processed_sample.metadata.update(email_specific_features)

                # Identify suspicious URLs if enabled (also requires extract_features to be enabled)
                if config.get("identify_suspicious_urls", True) and config.get(
                    "extract_features", True
                ):
                    suspicious_urls = self._identify_suspicious_urls(sample.input_text)
                    if suspicious_urls:
                        processed_sample.metadata["suspicious_urls"] = suspicious_urls
                        processed_sample.metadata["suspicious_url_count"] = len(suspicious_urls)

                # Normalize email addresses if enabled
                if config.get("normalize_emails", True):
                    cleaned_text = self._normalize_email_addresses(cleaned_text)

                # Analyze linguistic features if enabled
                if config.get("analyze_linguistics", True):
                    linguistic_features = self._analyze_linguistic_features(cleaned_text)
                    processed_sample.metadata.update(linguistic_features)

                # Final text cleaning
                final_text = PreprocessingUtilities.clean_text(
                    cleaned_text,
                    {
                        "remove_extra_whitespace": True,
                        "preserve_newlines": True,  # Keep structure for emails
                        "normalize_unicode": True,
                    },
                )
                processed_sample.input_text = final_text

                processed_samples.append(processed_sample)

            except Exception as e:
                error_msg = f"Failed to process email sample: {e}"
                self.logger.error(error_msg)
                raise PreprocessorError(error_msg, sample_id=sample.id, cause=e) from e

        return processed_samples

    def get_required_fields(self) -> list[str]:
        """Get list of required fields for email preprocessing."""
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

        # Check for BeautifulSoup4 dependency if HTML cleaning is enabled
        if config.get("clean_html", True) and not HAS_BS4:
            warnings.append(
                "clean_html is enabled but BeautifulSoup4 is not installed. "
                "HTML parsing will be limited. Install with: pip install beautifulsoup4 lxml"
            )

        return warnings

    def get_supported_config_keys(self) -> list[str]:
        """Get list of supported configuration keys."""
        return [
            "clean_html",
            "extract_headers",
            "identify_suspicious_urls",
            "normalize_emails",
            "analyze_linguistics",
            "preserve_structure",
            "extract_features",
        ]

    def _clean_html_content(self, html_content: str) -> str:
        """
        Clean HTML and extract plain text.

        Args:
            html_content: HTML content to clean

        Returns:
            Plain text content with HTML removed
        """
        if not html_content or not html_content.strip():
            return html_content

        # If BeautifulSoup is available, use it for proper HTML parsing
        if HAS_BS4:
            try:
                soup = BeautifulSoup(html_content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text and preserve some structure
                text_parts = []

                # Process each element to preserve some structure
                for element in soup.find_all(string=True):
                    if isinstance(element, NavigableString):
                        text = str(element).strip()
                        if text:
                            # Add extra newlines for block elements
                            parent = element.parent
                            if parent and parent.name in [
                                "p",
                                "div",
                                "br",
                                "h1",
                                "h2",
                                "h3",
                                "h4",
                                "h5",
                                "h6",
                            ]:
                                text_parts.append(text + "\n")
                            else:
                                text_parts.append(text + " ")

                return " ".join(text_parts).strip()

            except Exception as e:
                self.logger.warning(f"Failed to parse HTML with BeautifulSoup: {e}")
                # Fall back to regex-based cleaning

        # Fallback regex-based HTML cleaning
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html_content)

        # Decode HTML entities
        html_entities = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&nbsp;": " ",
            "&copy;": "©",
            "&reg;": "®",
            "&trade;": "™",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_email_features(self, email_text: str) -> dict[str, Any]:
        """
        Extract email-specific features.

        Args:
            email_text: Email text to analyze

        Returns:
            Dictionary of email features
        """
        features: dict[str, Any] = {}

        # Extract email addresses
        email_addresses = PreprocessingUtilities.extract_email_addresses(email_text)
        if email_addresses:
            features["email_addresses"] = email_addresses
            features["email_count"] = len(email_addresses)

            # Analyze sender domain legitimacy
            domains = [addr.split("@")[1].lower() for addr in email_addresses if "@" in addr]
            legitimate_domains = [d for d in domains if d in self.LEGITIMATE_DOMAINS]
            suspicious_domains = [d for d in domains if d not in self.LEGITIMATE_DOMAINS]

            features["legitimate_domain_count"] = len(legitimate_domains)
            features["suspicious_domain_count"] = len(suspicious_domains)
            if suspicious_domains:
                features["suspicious_domains"] = list(set(suspicious_domains))

        # Extract URLs
        urls = PreprocessingUtilities.extract_urls(email_text)
        if urls:
            features["urls"] = urls
            features["url_count"] = len(urls)

        # Analyze text statistics
        text_stats = self._calculate_text_statistics(email_text)
        features.update(text_stats)

        # Look for attachment indicators
        attachment_indicators = self._find_attachment_indicators(email_text)
        if attachment_indicators:
            features["attachment_indicators"] = attachment_indicators

        return features

    def _identify_suspicious_urls(self, email_text: str) -> list[str]:
        """
        Identify potentially suspicious URLs.

        Args:
            email_text: Email text to analyze

        Returns:
            List of suspicious URLs found
        """
        suspicious_urls = []

        # Extract all URLs first
        urls = PreprocessingUtilities.extract_urls(email_text)

        for url in urls:
            is_suspicious = False

            # Parse URL to get domain for certain checks
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
            except Exception:
                # If URL parsing fails, consider it suspicious
                is_suspicious = True
                continue

            # Check against suspicious patterns
            for i, pattern in enumerate(self.SUSPICIOUS_URL_PATTERNS):
                if i == 0:  # First pattern is for URL shortener domains
                    if pattern.search(domain):
                        is_suspicious = True
                        break
                else:  # Other patterns check full URL
                    if pattern.search(url):
                        is_suspicious = True
                        break

            # Additional checks
            if (
                not is_suspicious
                and domain not in self.LEGITIMATE_DOMAINS
                and (
                    len(domain) > 50  # Very long domain
                    or domain.count("-") > 3  # Many hyphens
                    or domain.count(".") > 4  # Many subdomains
                    or any(char.isdigit() for char in domain.replace(".", ""))
                    and sum(1 for char in domain if char.isdigit()) > 5  # Many numbers
                )
            ):
                is_suspicious = True

            if is_suspicious:
                suspicious_urls.append(url)

        return suspicious_urls

    def _extract_header_info(self, email_text: str) -> dict[str, str]:
        """
        Extract email headers if present.

        Args:
            email_text: Email text potentially containing headers

        Returns:
            Dictionary of extracted headers
        """
        headers: dict[str, str] = {}

        try:
            # Try to parse as email message
            msg = email.message_from_string(email_text)

            # Extract common headers
            header_fields = [
                "From",
                "To",
                "Subject",
                "Date",
                "Reply-To",
                "Return-Path",
                "Message-ID",
                "X-Mailer",
                "User-Agent",
                "Received",
            ]

            for field in header_fields:
                value = msg.get(field)
                if value:
                    # Decode header if needed
                    try:
                        decoded_parts = decode_header(value)
                        decoded_value = ""
                        for part, encoding in decoded_parts:
                            if isinstance(part, bytes):
                                decoded_value += part.decode(encoding or "ascii", errors="ignore")
                            else:
                                decoded_value += part
                        headers[field.lower()] = decoded_value.strip()
                    except Exception:
                        headers[field.lower()] = str(value).strip()

        except Exception as e:
            self.logger.debug(f"Could not parse email headers: {e}")

            # Fallback: try to extract headers using regex
            header_patterns = {
                "from": re.compile(r"^From:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
                "to": re.compile(r"^To:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
                "subject": re.compile(r"^Subject:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
                "date": re.compile(r"^Date:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
            }

            for header_name, pattern in header_patterns.items():
                match = pattern.search(email_text)
                if match:
                    headers[header_name] = match.group(1).strip()

        return headers

    def _analyze_linguistic_features(self, text: str) -> dict[str, Any]:
        """
        Analyze linguistic features relevant to phishing.

        Args:
            text: Text content to analyze

        Returns:
            Dictionary of linguistic features
        """
        features: dict[str, Any] = {}

        text_lower = text.lower()

        # Count phishing-related keywords
        keyword_counts = {}
        total_phishing_words = 0

        for category, keywords in self.PHISHING_KEYWORDS.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_counts[f"{category}_keywords"] = count
            total_phishing_words += count

        features.update(keyword_counts)
        features["total_phishing_keywords"] = total_phishing_words

        # Calculate phishing keyword density
        words = text_lower.split()
        if words:
            features["phishing_keyword_density"] = total_phishing_words / len(words)
        else:
            features["phishing_keyword_density"] = 0.0

        # Analyze punctuation patterns
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")
        features["caps_ratio"] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        # Look for social engineering patterns
        social_patterns = [
            r"\b(?:click|tap)\s+(?:here|now|below)\b",
            r"\b(?:verify|confirm|update)\s+(?:now|immediately|asap)\b",
            r"\b(?:limited|final|last)\s+(?:time|chance|offer)\b",
            r"\b(?:urgent|immediate)\s+(?:action|attention)\b",
        ]

        social_engineering_score = 0
        for pattern in social_patterns:
            matches = len(re.findall(pattern, text_lower))
            social_engineering_score += matches

        features["social_engineering_score"] = social_engineering_score

        return features

    def _normalize_email_addresses(self, email_text: str) -> str:
        """
        Normalize email addresses in content.

        Args:
            email_text: Text containing email addresses

        Returns:
            Text with normalized email addresses
        """
        # Extract email addresses
        emails = PreprocessingUtilities.extract_email_addresses(email_text)

        # Normalize each email address
        normalized_text = email_text
        for email_addr in emails:
            # Convert to lowercase and remove extra spaces
            normalized = email_addr.lower().strip()

            # Replace in text if different
            if normalized != email_addr:
                normalized_text = normalized_text.replace(email_addr, normalized)

        return normalized_text

    async def _parse_email_content(self, email_text: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Parse email content and extract basic structure information.

        Args:
            email_text: Email content to parse
            config: Configuration dictionary

        Returns:
            Dictionary of email structure features
        """
        features: dict[str, Any] = {}

        # Detect email format (check MIME first, then HTML, then plain text)
        if "Content-Type:" in email_text or "MIME-Version:" in email_text:
            features["email_format"] = "mime"
        elif "<html" in email_text.lower() or "<body" in email_text.lower():
            features["email_format"] = "html"
        else:
            features["email_format"] = "plain_text"

        # Check for multipart content
        if "boundary=" in email_text:
            features["is_multipart"] = True
            # Count parts
            boundary_count = email_text.count("--")
            features["multipart_sections"] = max(0, (boundary_count // 2) - 1)
        else:
            features["is_multipart"] = False

        # Basic content analysis
        features["content_length"] = len(email_text)
        features["line_count"] = email_text.count("\n") + 1

        return features

    def _calculate_text_statistics(self, text: str) -> dict[str, Any]:
        """Calculate basic text statistics."""
        stats: dict[str, Any] = {}

        if not text:
            return stats

        # Basic counts
        stats["character_count"] = len(text)
        stats["word_count"] = len(text.split())
        stats["sentence_count"] = len(re.findall(r"[.!?]+", text))

        # Calculate averages
        words = text.split()
        if words:
            word_lengths = [len(word) for word in words]
            stats["avg_word_length"] = statistics.mean(word_lengths)
            stats["max_word_length"] = max(word_lengths)

        sentences = re.split(r"[.!?]+", text)
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths:
                stats["avg_sentence_length"] = statistics.mean(sentence_lengths)
                stats["max_sentence_length"] = max(sentence_lengths)

        return stats

    def _find_attachment_indicators(self, email_text: str) -> list[str]:
        """Find indicators of email attachments."""
        indicators = []

        # Common attachment patterns
        patterns = [
            r"(?i)\battach(?:ed|ment)\b.*?\.(?:pdf|doc|docx|xls|xlsx|zip|rar|exe|scr)",
            r"(?i)content-disposition:\s*attachment",
            r'(?i)filename\s*=\s*["\']?([^"\'>\s]+\.[a-z0-9]{2,4})',
            r"(?i)please\s+(?:find|see)\s+(?:the\s+)?attach",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, email_text)
            indicators.extend(matches)

        return list(set(indicators))  # Remove duplicates

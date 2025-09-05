"""
Tests for cybersecurity data generators.

This module validates that the data generators create realistic and valid
cybersecurity test data matching expected schemas and formats.
"""
# mypy: disable-error-code="no-untyped-def,var-annotated"

import re
import sys
from ipaddress import AddressValueError, IPv4Address
from pathlib import Path

# Add tests directory to Python path
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

# Import after path modification
from utils.data_generators import CybersecurityDataGenerator  # noqa: E402


class TestCybersecurityDataGenerator:
    """Test the main data generator class."""

    def test_generator_initialization(self):
        """Test generator can be initialized with and without seed."""
        # Without seed
        generator = CybersecurityDataGenerator()
        assert generator is not None

        # With seed for reproducible results
        generator_seeded = CybersecurityDataGenerator(seed=42)
        assert generator_seeded is not None

        # Test reproducibility with same seed
        gen1 = CybersecurityDataGenerator(seed=123)
        gen2 = CybersecurityDataGenerator(seed=123)

        sample1 = gen1.generate_network_log()
        sample2 = gen2.generate_network_log()

        # Should generate identical samples with same seed
        assert sample1["src_ip"] == sample2["src_ip"]
        assert sample1["timestamp"] == sample2["timestamp"]

    def test_ip_address_generation(self):
        """Test IP address generation."""
        generator = CybersecurityDataGenerator(seed=42)

        # Test private IP generation
        private_ip = generator.generate_ip_address(private=True)
        assert self._is_valid_ipv4(private_ip)
        assert self._is_private_ip(private_ip)

        # Test public IP generation
        public_ip = generator.generate_ip_address(private=False)
        assert self._is_valid_ipv4(public_ip)
        assert not self._is_private_ip(public_ip)

    def test_timestamp_generation(self):
        """Test timestamp generation."""
        generator = CybersecurityDataGenerator(seed=42)

        # Test default timestamp (30 days back)
        timestamp = generator.generate_timestamp()
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", timestamp)

        # Test custom days back
        timestamp_recent = generator.generate_timestamp(days_back=1)
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", timestamp_recent)

    def _is_valid_ipv4(self, ip: str) -> bool:
        """Check if string is valid IPv4 address."""
        try:
            IPv4Address(ip)
            return True
        except AddressValueError:
            return False

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is in private ranges."""
        ip_obj = IPv4Address(ip)
        return ip_obj.is_private


class TestNetworkLogGeneration:
    """Test network log generation functionality."""

    def test_benign_network_log_generation(self):
        """Test generation of benign network logs."""
        generator = CybersecurityDataGenerator(seed=42)

        log = generator.generate_network_log(is_attack=False)

        # Validate required fields
        required_fields = [
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "bytes_sent",
            "bytes_received",
            "text",
            "label",
        ]
        for field in required_fields:
            assert field in log, f"Missing required field: {field}"

        # Validate content
        assert log["label"] == "BENIGN"
        assert log["attack_type"] is None
        assert log["severity"] == "LOW"
        assert 0.9 <= log["confidence"] <= 0.99
        assert self._is_valid_ipv4(log["src_ip"])
        assert self._is_valid_ipv4(log["dst_ip"])
        assert 1 <= log["src_port"] <= 65535
        assert 1 <= log["dst_port"] <= 65535
        assert log["protocol"] == "TCP"

    def test_attack_network_log_generation(self):
        """Test generation of attack network logs."""
        generator = CybersecurityDataGenerator(seed=42)

        attack_types = ["malware", "intrusion", "dos", "reconnaissance"]

        for attack_type in attack_types:
            log = generator.generate_network_log(is_attack=True, attack_type=attack_type)

            # Validate basic structure
            assert log["label"] == "ATTACK"
            assert log["attack_type"] == attack_type
            assert log["attack_subtype"] in generator.ATTACK_TYPES[attack_type]
            assert log["severity"] in ["MEDIUM", "HIGH", "CRITICAL"]
            assert 0.6 <= log["confidence"] <= 0.98

            # Validate network fields
            assert self._is_valid_ipv4(log["src_ip"])
            assert self._is_valid_ipv4(log["dst_ip"])
            assert "additional_data" in log

    def test_malware_log_specifics(self):
        """Test malware-specific log generation."""
        generator = CybersecurityDataGenerator(seed=42)

        log = generator.generate_network_log(is_attack=True, attack_type="malware")

        assert log["attack_type"] == "malware"
        assert log["attack_subtype"] in generator.ATTACK_TYPES["malware"]
        assert "suspicious_domain" in log["additional_data"]
        assert "file_hash" in log["additional_data"]
        assert "user_agent" in log["additional_data"]
        assert len(log["additional_data"]["file_hash"]) == 64  # SHA-256 style

    def test_intrusion_log_specifics(self):
        """Test intrusion-specific log generation."""
        generator = CybersecurityDataGenerator(seed=42)

        log = generator.generate_network_log(is_attack=True, attack_type="intrusion")

        assert log["attack_type"] == "intrusion"
        assert log["attack_subtype"] in generator.ATTACK_TYPES["intrusion"]
        assert "failed_login_attempts" in log["additional_data"]
        assert "target_service" in log["additional_data"]
        assert "username_attempts" in log["additional_data"]
        assert isinstance(log["additional_data"]["username_attempts"], list)

    def test_dos_log_specifics(self):
        """Test DoS-specific log generation."""
        generator = CybersecurityDataGenerator(seed=42)

        log = generator.generate_network_log(is_attack=True, attack_type="dos")

        assert log["attack_type"] == "dos"
        assert log["attack_subtype"] in generator.ATTACK_TYPES["dos"]
        assert "request_rate" in log["additional_data"]
        assert "packet_size" in log["additional_data"]
        assert "duration_seconds" in log["additional_data"]
        assert "amplification_factor" in log["additional_data"]

    def test_reconnaissance_log_specifics(self):
        """Test reconnaissance-specific log generation."""
        generator = CybersecurityDataGenerator(seed=42)

        log = generator.generate_network_log(is_attack=True, attack_type="reconnaissance")

        assert log["attack_type"] == "reconnaissance"
        assert log["attack_subtype"] in generator.ATTACK_TYPES["reconnaissance"]
        assert "scanned_ports" in log["additional_data"]
        assert "scan_type" in log["additional_data"]
        assert "scan_duration" in log["additional_data"]
        assert "response_analysis" in log["additional_data"]
        assert isinstance(log["additional_data"]["scanned_ports"], list)

    def _is_valid_ipv4(self, ip: str) -> bool:
        """Check if string is valid IPv4 address."""
        try:
            IPv4Address(ip)
            return True
        except AddressValueError:
            return False


class TestEmailGeneration:
    """Test email sample generation functionality."""

    def test_benign_email_generation(self):
        """Test generation of benign emails."""
        generator = CybersecurityDataGenerator(seed=42)

        email = generator.generate_email_sample(is_phishing=False)

        # Validate required fields
        required_fields = [
            "message_id",
            "timestamp",
            "sender",
            "recipient",
            "subject",
            "body",
            "text",
            "label",
        ]
        for field in required_fields:
            assert field in email, f"Missing required field: {field}"

        # Validate content
        assert email["label"] == "BENIGN"
        assert email["attack_type"] is None
        assert 0.85 <= email["confidence"] <= 0.99
        assert "@" in email["sender"]
        assert "@" in email["recipient"]
        assert email["message_id"].startswith("<") and email["message_id"].endswith(">")

    def test_phishing_email_generation(self):
        """Test generation of phishing emails."""
        generator = CybersecurityDataGenerator(seed=42)

        phishing_types = generator.ATTACK_TYPES["phishing"]

        for phishing_type in phishing_types:
            email = generator.generate_email_sample(is_phishing=True, phishing_type=phishing_type)

            # Validate basic structure
            assert email["label"] == "ATTACK"
            assert email["attack_type"] == "phishing"
            assert email["attack_subtype"] == phishing_type
            assert 0.75 <= email["confidence"] <= 0.95

            # Validate phishing-specific fields
            assert "additional_data" in email
            assert "sender_domain" in email["additional_data"]
            assert "suspicious_urls" in email["additional_data"]
            assert "urgency_keywords" in email["additional_data"]
            assert isinstance(email["additional_data"]["suspicious_urls"], list)

    def test_spear_phishing_specifics(self):
        """Test spear phishing specific generation."""
        generator = CybersecurityDataGenerator(seed=42)

        email = generator.generate_email_sample(is_phishing=True, phishing_type="spear_phishing")

        assert email["attack_subtype"] == "spear_phishing"
        # Spear phishing typically has urgency indicators
        urgency_words = ["urgent", "immediate", "suspend", "verify", "action required"]
        subject_lower = email["subject"].lower()
        assert any(word in subject_lower for word in urgency_words)

    def test_whaling_specifics(self):
        """Test whaling (CEO fraud) specific generation."""
        generator = CybersecurityDataGenerator(seed=42)

        email = generator.generate_email_sample(is_phishing=True, phishing_type="whaling")

        assert email["attack_subtype"] == "whaling"
        # Whaling typically involves executive impersonation
        executive_words = ["ceo", "board", "executive", "confidential", "wire", "transfer"]
        content_lower = (email["subject"] + " " + email["body"]).lower()
        assert any(word in content_lower for word in executive_words)


class TestModelPredictionGeneration:
    """Test model prediction generation functionality."""

    def test_accurate_predictions(self):
        """Test generation of accurate predictions."""
        generator = CybersecurityDataGenerator(seed=42)

        # Test high accuracy predictions
        for ground_truth in ["ATTACK", "BENIGN"]:
            prediction = generator.generate_model_prediction(ground_truth, accuracy=0.95)

            # Validate structure
            required_fields = [
                "sample_id",
                "prediction",
                "confidence",
                "ground_truth",
                "is_correct",
                "class_probabilities",
                "inference_time_ms",
            ]
            for field in required_fields:
                assert field in prediction

            # Validate content
            assert prediction["ground_truth"] == ground_truth
            assert prediction["prediction"] in ["ATTACK", "BENIGN"]
            assert 0.0 <= prediction["confidence"] <= 1.0
            assert isinstance(prediction["is_correct"], bool)
            assert "ATTACK" in prediction["class_probabilities"]
            assert "BENIGN" in prediction["class_probabilities"]
            assert prediction["inference_time_ms"] > 0

    def test_prediction_accuracy_distribution(self):
        """Test that prediction accuracy follows expected distribution."""
        generator = CybersecurityDataGenerator(seed=42)

        # Generate many predictions with 80% accuracy
        predictions = []
        for _ in range(100):
            pred = generator.generate_model_prediction("ATTACK", accuracy=0.8)
            predictions.append(pred)

        # Check that roughly 80% are correct
        correct_count = sum(1 for p in predictions if p["is_correct"])
        accuracy_rate = correct_count / len(predictions)

        # Allow some variance due to randomness
        assert 0.7 <= accuracy_rate <= 0.9

    def test_confidence_scores_realistic(self):
        """Test that confidence scores are realistic."""
        generator = CybersecurityDataGenerator(seed=42)

        # Correct predictions should have higher confidence
        correct_pred = generator.generate_model_prediction("ATTACK", accuracy=1.0)
        assert correct_pred["is_correct"]
        assert correct_pred["confidence"] >= 0.7

        # Test class probabilities sum to 1.0
        probs = correct_pred["class_probabilities"]
        assert abs(probs["ATTACK"] + probs["BENIGN"] - 1.0) < 0.001


class TestExplanationGeneration:
    """Test explanation generation functionality."""

    def test_attack_explanations(self):
        """Test generation of attack explanations."""
        generator = CybersecurityDataGenerator(seed=42)

        attack_types = ["malware", "phishing", "dos", "reconnaissance"]

        for attack_type in attack_types:
            explanation = generator.generate_explanation("ATTACK", attack_type)

            # Validate explanation structure
            assert isinstance(explanation, str)
            assert len(explanation) > 50  # Should be reasonably detailed
            assert "." in explanation  # Should have proper sentences

            # Should contain relevant keywords for attack type
            explanation_lower = explanation.lower()
            if attack_type == "malware":
                assert any(
                    word in explanation_lower
                    for word in ["malware", "hash", "signature", "payload"]
                )
            elif attack_type == "phishing":
                assert any(
                    word in explanation_lower for word in ["phishing", "email", "url", "sender"]
                )
            elif attack_type == "dos":
                assert any(
                    word in explanation_lower
                    for word in ["volume", "requests", "flooding", "traffic"]
                )
            elif attack_type == "reconnaissance":
                assert any(
                    word in explanation_lower
                    for word in ["scan", "port", "enumeration", "discovery"]
                )

    def test_benign_explanations(self):
        """Test generation of benign explanations."""
        generator = CybersecurityDataGenerator(seed=42)

        explanation = generator.generate_explanation("BENIGN")

        # Validate explanation structure
        assert isinstance(explanation, str)
        assert len(explanation) > 50
        assert "." in explanation

        # Should contain positive indicators
        explanation_lower = explanation.lower()
        positive_words = ["normal", "legitimate", "standard", "successful", "passed"]
        assert any(word in explanation_lower for word in positive_words)

    def test_explanation_technical_details(self):
        """Test that explanations contain technical details."""
        generator = CybersecurityDataGenerator(seed=42)

        explanation = generator.generate_explanation("ATTACK", "malware")

        # Should contain technical information
        explanation_lower = explanation.lower()
        technical_terms = ["feature", "score", "analysis", "pattern", "risk"]
        assert any(term in explanation_lower for term in technical_terms)


class TestPerformanceDataGeneration:
    """Test performance data generation functionality."""

    def test_performance_data_structure(self):
        """Test structure of generated performance data."""
        generator = CybersecurityDataGenerator(seed=42)

        performance_data = generator.generate_performance_data(num_samples=10)

        assert len(performance_data) == 10

        for sample in performance_data:
            required_fields = [
                "sample_id",
                "model_size",
                "inference_time_ms",
                "preprocessing_time_ms",
                "postprocessing_time_ms",
                "total_time_ms",
                "memory_usage_mb",
                "cpu_usage_percent",
                "gpu_usage_percent",
                "batch_size",
                "throughput_samples_per_second",
                "timestamp",
            ]

            for field in required_fields:
                assert field in sample, f"Missing field: {field}"

    def test_performance_data_ranges(self):
        """Test that performance data falls within realistic ranges."""
        generator = CybersecurityDataGenerator(seed=42)

        performance_data = generator.generate_performance_data(num_samples=50)

        for sample in performance_data:
            # Validate ranges
            assert sample["inference_time_ms"] > 0
            assert sample["preprocessing_time_ms"] > 0
            assert sample["postprocessing_time_ms"] > 0
            assert sample["total_time_ms"] > sample["inference_time_ms"]
            assert 0 <= sample["memory_usage_mb"] <= 500
            assert 0 <= sample["cpu_usage_percent"] <= 100
            assert 0 <= sample["gpu_usage_percent"] <= 100
            assert sample["batch_size"] in [1, 4, 8, 16, 32]
            assert sample["throughput_samples_per_second"] > 0

    def test_model_size_performance_correlation(self):
        """Test that larger models generally have longer inference times."""
        generator = CybersecurityDataGenerator(seed=42)

        performance_data = generator.generate_performance_data(num_samples=100)

        # Group by model size
        size_times = {"small": [], "medium": [], "large": [], "xlarge": []}

        for sample in performance_data:
            size_times[sample["model_size"]].append(sample["inference_time_ms"])

        # Calculate averages (if we have samples for each size)
        if all(times for times in size_times.values()):
            avg_times = {size: sum(times) / len(times) for size, times in size_times.items()}

            # Generally, larger models should take longer (with some variance allowed)
            assert avg_times["medium"] >= avg_times["small"] * 0.8
            assert avg_times["large"] >= avg_times["medium"] * 0.8
            assert avg_times["xlarge"] >= avg_times["large"] * 0.8


class TestBatchGeneration:
    """Test batch sample generation functionality."""

    def test_batch_sample_generation(self):
        """Test generation of mixed batch samples."""
        generator = CybersecurityDataGenerator(seed=42)

        samples = generator.generate_batch_samples(num_samples=100, attack_ratio=0.3)

        assert len(samples) == 100

        # Count attack vs benign
        attack_count = sum(1 for s in samples if s["label"] == "ATTACK")
        benign_count = sum(1 for s in samples if s["label"] == "BENIGN")

        # Should be roughly 30% attacks (allow some variance)
        assert 25 <= attack_count <= 35
        assert benign_count == 100 - attack_count

        # All samples should have sample_id
        for i, sample in enumerate(samples):
            assert "sample_id" in sample
            assert sample["sample_id"] == str(i + 1)

    def test_attack_type_distribution(self):
        """Test that specified attack types are used in batch generation."""
        generator = CybersecurityDataGenerator(seed=42)

        # Generate batch with specific attack types
        attack_types = ["malware", "phishing"]
        samples = generator.generate_batch_samples(
            num_samples=50, attack_ratio=0.5, attack_types=attack_types
        )

        # Check that only specified attack types are used
        attack_samples = [s for s in samples if s["label"] == "ATTACK"]

        for sample in attack_samples:
            assert sample["attack_type"] in attack_types

    def test_mixed_sample_types(self):
        """Test that batch contains both network logs and emails."""
        generator = CybersecurityDataGenerator(seed=42)

        samples = generator.generate_batch_samples(num_samples=100, attack_ratio=0.5)

        # Should have mix of sample types (some with email-specific fields)
        has_network_logs = any("src_ip" in s for s in samples)
        has_emails = any("sender" in s for s in samples)

        # With 100 samples, we should get both types
        assert has_network_logs
        assert has_emails


class TestDataValidation:
    """Test that generated data matches expected schemas."""

    def test_network_log_schema_compliance(self):
        """Test that network logs comply with expected schema."""
        generator = CybersecurityDataGenerator(seed=42)

        # Test both attack and benign logs
        for is_attack in [True, False]:
            log = generator.generate_network_log(is_attack=is_attack)

            # Check required string fields
            string_fields = ["timestamp", "src_ip", "dst_ip", "protocol", "text", "label"]
            for field in string_fields:
                assert isinstance(log[field], str)
                assert len(log[field]) > 0

            # Check numeric fields
            numeric_fields = ["src_port", "dst_port", "bytes_sent", "bytes_received"]
            for field in numeric_fields:
                assert isinstance(log[field], int)
                assert log[field] >= 0

            # Check confidence is float
            assert isinstance(log["confidence"], float)
            assert 0.0 <= log["confidence"] <= 1.0

            # Check attack_type field
            if is_attack:
                assert log["attack_type"] in generator.ATTACK_TYPES
            else:
                assert log["attack_type"] is None

    def test_email_schema_compliance(self):
        """Test that emails comply with expected schema."""
        generator = CybersecurityDataGenerator(seed=42)

        for is_phishing in [True, False]:
            email = generator.generate_email_sample(is_phishing=is_phishing)

            # Check required string fields
            string_fields = [
                "message_id",
                "timestamp",
                "sender",
                "recipient",
                "subject",
                "body",
                "text",
                "label",
            ]
            for field in string_fields:
                assert isinstance(email[field], str)
                assert len(email[field]) > 0

            # Check confidence
            assert isinstance(email["confidence"], float)
            assert 0.0 <= email["confidence"] <= 1.0

            # Check email format
            assert "@" in email["sender"]
            assert "@" in email["recipient"]
            assert email["message_id"].startswith("<")
            assert email["message_id"].endswith(">")

    def test_prediction_schema_compliance(self):
        """Test that predictions comply with expected schema."""
        generator = CybersecurityDataGenerator(seed=42)

        prediction = generator.generate_model_prediction("ATTACK")

        # Check string fields
        assert isinstance(prediction["sample_id"], str)
        assert isinstance(prediction["prediction"], str)
        assert isinstance(prediction["ground_truth"], str)
        assert isinstance(prediction["model_version"], str)
        assert isinstance(prediction["timestamp"], str)

        # Check numeric fields
        assert isinstance(prediction["confidence"], float)
        assert isinstance(prediction["inference_time_ms"], float)
        assert isinstance(prediction["is_correct"], bool)

        # Check class probabilities
        assert isinstance(prediction["class_probabilities"], dict)
        assert "ATTACK" in prediction["class_probabilities"]
        assert "BENIGN" in prediction["class_probabilities"]

        # Probabilities should sum to 1.0
        prob_sum = sum(prediction["class_probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.001

    def test_performance_data_schema_compliance(self):
        """Test that performance data complies with expected schema."""
        generator = CybersecurityDataGenerator(seed=42)

        performance_data = generator.generate_performance_data(num_samples=5)

        for sample in performance_data:
            # Check string fields
            assert isinstance(sample["sample_id"], str)
            assert isinstance(sample["model_size"], str)
            assert isinstance(sample["timestamp"], str)

            # Check numeric fields
            numeric_fields = [
                "inference_time_ms",
                "preprocessing_time_ms",
                "postprocessing_time_ms",
                "total_time_ms",
                "memory_usage_mb",
                "cpu_usage_percent",
                "gpu_usage_percent",
                "throughput_samples_per_second",
            ]

            for field in numeric_fields:
                assert isinstance(sample[field], int | float)
                assert sample[field] >= 0

            # Check integer fields
            assert isinstance(sample["batch_size"], int)
            assert sample["batch_size"] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_attack_type_handling(self):
        """Test handling of invalid attack types."""
        generator = CybersecurityDataGenerator(seed=42)

        # Should handle unknown attack types gracefully
        log = generator.generate_network_log(is_attack=True, attack_type="unknown_type")

        assert log["label"] == "ATTACK"
        assert log["attack_type"] == "unknown_type"
        # Should fallback to generic attack log

    def test_zero_samples_batch(self):
        """Test batch generation with zero samples."""
        generator = CybersecurityDataGenerator(seed=42)

        samples = generator.generate_batch_samples(num_samples=0)
        assert len(samples) == 0
        assert isinstance(samples, list)

    def test_extreme_accuracy_values(self):
        """Test prediction generation with extreme accuracy values."""
        generator = CybersecurityDataGenerator(seed=42)

        # 100% accuracy
        perfect_pred = generator.generate_model_prediction("ATTACK", accuracy=1.0)
        assert perfect_pred["is_correct"] is True
        assert perfect_pred["prediction"] == "ATTACK"

        # 0% accuracy (should always be wrong)
        wrong_pred = generator.generate_model_prediction("ATTACK", accuracy=0.0)
        assert wrong_pred["is_correct"] is False
        assert wrong_pred["prediction"] == "BENIGN"

    def test_single_sample_performance_data(self):
        """Test performance data generation with single sample."""
        generator = CybersecurityDataGenerator(seed=42)

        performance_data = generator.generate_performance_data(num_samples=1)

        assert len(performance_data) == 1
        sample = performance_data[0]
        assert sample["sample_id"] == "1"
        assert all(
            field in sample
            for field in ["inference_time_ms", "memory_usage_mb", "throughput_samples_per_second"]
        )

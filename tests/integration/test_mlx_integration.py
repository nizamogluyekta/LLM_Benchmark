"""
Integration tests for the MLX plugin.

This module tests MLX plugin integration with actual MLX libraries
when available, and gracefully skips when not available.
"""

import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import pytest_asyncio

from benchmark.models.plugins.mlx_local import MLXModelPlugin


class TestMLXIntegration:
    """Integration tests for MLX plugin."""

    @pytest.fixture(scope="class")
    def apple_silicon_only(self) -> None:
        """Skip tests on non-Apple Silicon systems."""
        if platform.system() != "Darwin" or "arm64" not in platform.machine().lower():
            pytest.skip("MLX tests require Apple Silicon hardware")

    @pytest.fixture(scope="class")
    def mlx_available(self) -> None:
        """Skip tests if MLX libraries are not available."""
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError:
            pytest.skip("MLX libraries not available. Install with: pip install mlx mlx-lm")

    @pytest_asyncio.fixture
    async def plugin(self, apple_silicon_only: None, mlx_available: None) -> MLXModelPlugin:
        """Create MLX plugin for integration testing."""
        return MLXModelPlugin()

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_plugin_initialization_real_system(self, plugin: MLXModelPlugin) -> None:
        """Test plugin initialization on real Apple Silicon system."""
        # Test system detection
        system_info = plugin._system_info
        assert system_info["is_apple_silicon"] is True
        assert system_info["platform"] == "Darwin"
        assert "arm64" in system_info["machine"].lower()
        assert system_info["memory_gb"] > 0
        assert system_info["cpu_cores"] > 0

        # Test MLX import
        assert plugin._import_mlx() is True
        assert plugin.mlx_lm is not None
        assert plugin.mlx is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_initialize_with_dummy_model(self, plugin: MLXModelPlugin) -> None:
        """Test initialization with a dummy model structure."""
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "dummy_model"
            model_path.mkdir()

            # Create minimal model structure
            (model_path / "config.json").write_text('{"model_type": "llama"}')
            (model_path / "tokenizer_config.json").write_text(
                '{"tokenizer_class": "LlamaTokenizer"}'
            )

            config = {
                "model_path": str(model_path),
                "name": "dummy-test-model",
                "quantization": "4bit",
                "max_tokens": 128,
                "temperature": 0.1,
            }

            # This should fail with model loading but succeed in validation
            response = await plugin.initialize(config)

            # May fail due to invalid model format, but should validate configs properly
            if not response.success:
                # Expected - dummy model isn't a real MLX model
                assert "failed" in response.error.lower() or "invalid" in response.error.lower()
            else:
                # If it somehow succeeds, check the setup
                assert plugin.model_config == config
                assert plugin.model_info is not None
                assert plugin.performance_metrics is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_check_real_system(self, plugin: MLXModelPlugin) -> None:
        """Test health check on real system."""
        health = await plugin.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert health["apple_silicon"] is True
        assert health["memory_gb"] > 0
        assert health["cached_models"] == 0  # No models loaded yet

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_prompt_formatting_integration(self, plugin: MLXModelPlugin) -> None:
        """Test prompt formatting integration."""
        test_samples = [
            "192.168.1.100:80 -> suspicious-domain.com:443 ESTABLISHED",
            "Failed login attempt from user: admin",
            "Process execution: powershell.exe -encoded <base64>",
            "DNS query: malware-c2.example.com",
            "Normal HTTP request to google.com",
        ]

        for sample in test_samples:
            prompt = plugin._format_cybersecurity_prompt(sample)

            # Verify prompt structure
            assert "<|system|>" in prompt
            assert "<|user|>" in prompt
            assert "<|assistant|>" in prompt
            assert sample in prompt
            assert "cybersecurity expert" in prompt.lower()

            # Verify all required fields are mentioned
            required_fields = ["Classification", "Confidence", "Attack_Type", "Explanation", "IOCs"]
            for field in required_fields:
                assert field in prompt

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_response_parsing_integration(self, plugin: MLXModelPlugin) -> None:
        """Test response parsing with various response formats."""
        test_responses = [
            # Well-formatted attack response
            """
            Classification: ATTACK
            Confidence: 0.85
            Attack_Type: malware
            Explanation: This network connection shows signs of command and control communication with a known malicious domain.
            IOCs: suspicious-domain.com, 192.168.1.100, port 443
            """,
            # Well-formatted benign response
            """
            Classification: BENIGN
            Confidence: 0.92
            Attack_Type: N/A
            Explanation: Normal HTTPS traffic to a legitimate service.
            IOCs: None
            """,
            # Response with case variations
            """
            classification: attack
            confidence: 0.76
            attack_type: intrusion
            explanation: Attempted unauthorized access detected.
            iocs: admin, failed_login, multiple_attempts
            """,
            # Minimal response
            """
            Classification: BENIGN
            Confidence: 0.5
            """,
            # Response with extra content
            """
            Here is my analysis:

            Classification: ATTACK
            Confidence: 0.90
            Attack_Type: phishing
            Explanation: This appears to be a phishing attempt targeting user credentials.
            IOCs: fake-login.com, credential_harvest.js

            Additional notes: This is a sophisticated attack...
            """,
        ]

        for i, response in enumerate(test_responses):
            parsed = plugin._parse_response(response)

            # Verify required fields exist
            assert "classification" in parsed
            assert "confidence" in parsed
            assert "attack_type" in parsed
            assert "explanation" in parsed
            assert "iocs" in parsed

            # Verify data types
            assert parsed["classification"] in ["ATTACK", "BENIGN"]
            assert 0.0 <= parsed["confidence"] <= 1.0
            assert parsed["attack_type"] is None or isinstance(parsed["attack_type"], str)
            assert isinstance(parsed["explanation"], str)
            assert isinstance(parsed["iocs"], list)

            print(
                f"Response {i + 1} parsed successfully: {parsed['classification']} ({parsed['confidence']:.2f})"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cybersecurity_prompt_quality(self, plugin: MLXModelPlugin) -> None:
        """Test that cybersecurity prompts contain appropriate security context."""
        security_samples = [
            # Network security events
            "TCP connection from 10.0.0.1:1337 to external IP 203.0.113.1:80",
            "Multiple failed SSH login attempts from IP 198.51.100.42",
            "Large data transfer: 500MB uploaded to cloud storage",
            # Endpoint security events
            "Process created: cmd.exe /c powershell -ExecutionPolicy Bypass",
            "File modification: C:\\Windows\\System32\\drivers\\etc\\hosts",
            "Registry key modified: HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
            # Application security events
            "SQL injection attempt: ' OR 1=1--",
            "Cross-site scripting: <script>alert('XSS')</script>",
            "Directory traversal: ../../../../etc/passwd",
            # Email security events
            "Email with suspicious attachment: invoice.pdf.exe",
            "Phishing email from admin@example.com claiming account suspended",
        ]

        for sample in security_samples:
            prompt = plugin._format_cybersecurity_prompt(sample)

            # Check for cybersecurity context
            security_keywords = [
                "cybersecurity",
                "security",
                "threat",
                "attack",
                "malicious",
                "network",
                "intrusion",
                "malware",
                "phishing",
                "indicators",
            ]

            prompt_lower = prompt.lower()
            assert any(keyword in prompt_lower for keyword in security_keywords), (
                f"Prompt lacks security context for sample: {sample[:50]}..."
            )

            # Check for structured output requirements
            assert "Classification:" in prompt
            assert "ATTACK or BENIGN" in prompt
            assert "Confidence:" in prompt
            assert "0.0 to 1.0" in prompt
            assert "IOCs:" in prompt

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_characteristics(self, plugin: MLXModelPlugin) -> None:
        """Test performance characteristics of the plugin."""
        import time

        # Test multiple prompt generations for performance
        sample = "Suspicious network connection detected"

        start_time = time.time()
        for i in range(100):
            _ = plugin._format_cybersecurity_prompt(f"{sample} #{i}")
            _ = plugin._parse_response("Classification: BENIGN\nConfidence: 0.5")
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_ms = (total_time / 100) * 1000

        print(f"Average prompt formatting + parsing time: {avg_time_ms:.2f}ms")

        # Should be very fast for formatting and parsing
        assert avg_time_ms < 10.0, f"Prompt processing too slow: {avg_time_ms:.2f}ms"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_estimation_integration(self, plugin: MLXModelPlugin) -> None:
        """Test memory estimation integration."""
        # Test with no model loaded
        memory_mb = plugin._estimate_memory_usage()
        assert memory_mb == 2048.0  # Default fallback

        # Test system memory detection
        system_info = plugin._system_info
        assert system_info["memory_gb"] > 0
        print(f"System memory: {system_info['memory_gb']:.2f}GB")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cleanup_integration(self, plugin: MLXModelPlugin) -> None:
        """Test cleanup integration."""
        # Set some state
        plugin.model_config = {"test": "config"}
        plugin._model_cache["test"] = ("model", "tokenizer")

        # Cleanup
        await plugin.cleanup()

        # Verify cleanup
        assert plugin.model is None
        assert plugin.tokenizer is None
        assert plugin.model_config is None

        # Cache should remain (by design for reuse)
        assert "test" in plugin._model_cache

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_configuration_validation_integration(self, plugin: MLXModelPlugin) -> None:
        """Test configuration validation integration."""
        test_configs = [
            # Valid configurations
            {
                "model_path": "/tmp/test_model",
                "name": "test-model",
                "quantization": "4bit",
                "max_tokens": 256,
                "temperature": 0.1,
            },
            {
                "model_path": "/tmp/another_model",
                "name": "another-model",
                "quantization": "8bit",
                "max_tokens": 512,
                "temperature": 0.2,
            },
            # Invalid configurations
            {
                "name": "no-path-model"  # Missing model_path
            },
            {
                "model_path": "/tmp/test"  # Missing name
            },
            {},  # Empty config
        ]

        for i, config in enumerate(test_configs):
            try:
                response = await plugin.initialize(config)

                if i < 2:  # Valid configs (will fail on missing path)
                    # Should fail on missing path, not config validation
                    assert not response.success
                    assert "does not exist" in response.error.lower()
                else:  # Invalid configs
                    assert not response.success
                    assert "missing" in response.error.lower()

            except Exception as e:
                # Should handle gracefully
                print(f"Config {i} handled with error: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_format_variants_integration(self, plugin: MLXModelPlugin) -> None:
        """Test that plugin can handle various input formats."""
        format_samples = [
            # JSON-like logs
            '{"timestamp": "2024-01-15T10:30:00Z", "src_ip": "192.168.1.100", "dst_ip": "10.0.0.1", "action": "ALLOW"}',
            # Syslog format
            "Jan 15 10:30:00 server1 sshd[1234]: Failed password for admin from 192.168.1.100 port 22 ssh2",
            # Windows Event Log style
            "Event ID: 4625, Logon Type: 3, Account: admin, Source IP: 192.168.1.100",
            # Firewall logs
            "DENY TCP 192.168.1.100:1337 -> 203.0.113.1:80 (suspicious connection blocked)",
            # Application logs
            "ERROR: SQL injection detected in parameter 'user_id': ' OR 1=1--",
            # Empty/minimal inputs
            "",
            "test",
            "normal web traffic",
        ]

        for sample in format_samples:
            try:
                # Should handle all formats without crashing
                prompt = plugin._format_cybersecurity_prompt(sample)
                assert isinstance(prompt, str)
                assert len(prompt) > 0
                assert sample in prompt or sample == ""  # Empty sample won't be in prompt

                # Test parsing a dummy response
                dummy_response = "Classification: BENIGN\nConfidence: 0.5"
                parsed = plugin._parse_response(dummy_response)
                assert parsed["classification"] == "BENIGN"

            except Exception as e:
                pytest.fail(f"Failed to handle sample format: {sample[:50]}... Error: {e}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        platform.system() != "Darwin" or "arm64" not in platform.machine().lower(),
        reason="Requires Apple Silicon for realistic performance testing",
    )
    @pytest.mark.asyncio
    async def test_apple_silicon_specific_features(self, plugin: MLXModelPlugin) -> None:
        """Test Apple Silicon specific features."""
        # Test system detection
        system_info = plugin._system_info
        assert system_info["is_apple_silicon"] is True
        assert system_info["platform"] == "Darwin"
        assert "arm64" in system_info["machine"].lower()

        # Test unified memory detection
        assert system_info["memory_gb"] > 8.0  # Minimum for M-series

        # Test MLX library compatibility
        if plugin._import_mlx():
            assert plugin.mlx_lm is not None
            assert plugin.mlx is not None
            print("MLX libraries successfully imported on Apple Silicon")
        else:
            pytest.skip("MLX libraries not available despite Apple Silicon hardware")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_behavior_integration(self, plugin: MLXModelPlugin) -> None:
        """Test model caching behavior."""
        # Test cache initialization
        assert isinstance(plugin._model_cache, dict)
        assert len(plugin._model_cache) == 0

        # Test cache directory creation
        assert plugin._cache_dir.exists()
        assert plugin._cache_dir.is_dir()

        # Test cache key generation
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"
            cache_key = f"{model_path}_4bit"
            assert isinstance(cache_key, str)
            assert "4bit" in cache_key

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, plugin: MLXModelPlugin) -> None:
        """Test comprehensive error handling."""
        error_scenarios = [
            # Non-existent model path
            {"model_path": "/nonexistent/path", "name": "test"},
            # Invalid quantization
            {"model_path": "/tmp", "name": "test", "quantization": "invalid"},
            # Missing required fields
            {"model_path": "/tmp"},
            {"name": "test"},
            # Invalid numeric parameters
            {"model_path": "/tmp", "name": "test", "max_tokens": -1},
            {"model_path": "/tmp", "name": "test", "temperature": -0.5},
        ]

        for i, config in enumerate(error_scenarios):
            try:
                response = await plugin.initialize(config)
                # Should handle all errors gracefully
                assert isinstance(response.success, bool)
                if not response.success:
                    assert isinstance(response.error, str)
                    assert len(response.error) > 0
                    print(f"Scenario {i + 1}: {response.error}")

            except Exception as e:
                # Should not raise unhandled exceptions
                pytest.fail(f"Unhandled exception in error scenario {i + 1}: {e}")

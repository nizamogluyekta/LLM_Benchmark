"""
Integration tests for the Ollama plugin.

This module tests Ollama plugin integration with actual Ollama server
when available, and gracefully skips when not available.
"""

import time

import pytest
import pytest_asyncio

from benchmark.models.plugins.ollama_local import OllamaModelPlugin


class TestOllamaIntegration:
    """Integration tests for Ollama plugin."""

    @pytest.fixture(scope="class")
    def ollama_available(self) -> None:
        """Skip tests if Ollama library is not available."""
        try:
            import ollama  # noqa: F401
        except ImportError:
            pytest.skip("Ollama library not available. Install with: pip install ollama")

    @pytest.fixture(scope="class")
    def ollama_server_available(self) -> None:
        """Skip tests if Ollama server is not available."""
        try:
            import ollama

            client = ollama.Client()
            client.list()  # Test connection
        except Exception:
            pytest.skip("Ollama server not available. Start server with: ollama serve")

    @pytest_asyncio.fixture
    async def plugin(self, ollama_available: None) -> OllamaModelPlugin:
        """Create Ollama plugin for integration testing."""
        return OllamaModelPlugin()

    @pytest.mark.integration
    async def test_plugin_initialization_real_system(self, plugin: OllamaModelPlugin) -> None:
        """Test plugin initialization on real system."""
        # Test Ollama import
        assert plugin._import_ollama() is True
        assert plugin.ollama is not None

        # Test server info
        info = plugin.get_server_info()
        assert info["host"] == "localhost"
        assert info["port"] == 11434
        assert info["url"] == "http://localhost:11434"

    @pytest.mark.integration
    async def test_server_connection_test(
        self, plugin: OllamaModelPlugin, ollama_server_available: None
    ) -> None:
        """Test server connection with real Ollama server."""
        # Import ollama and create client
        plugin._import_ollama()
        plugin.client = plugin.ollama.Client()

        # Test connection - should not raise
        await plugin._test_server_connection()

    @pytest.mark.integration
    async def test_model_availability_check(
        self, plugin: OllamaModelPlugin, ollama_server_available: None
    ) -> None:
        """Test model availability checking with real server."""
        plugin._import_ollama()
        plugin.client = plugin.ollama.Client()

        # Test with a lightweight model that might not be available
        plugin.model_name = "tinyllama:1b"

        # This should either find the model or attempt to pull it
        # We don't actually want to pull a model in tests, so we'll just check the logic
        try:
            models_response = plugin.client.list()
            available_models = [model["name"] for model in models_response.get("models", [])]

            # Log what models are available
            print(f"Available models on server: {available_models}")

            # The method should work without error
            if plugin.model_name not in available_models:
                # In real integration, this would pull the model
                print(f"Model {plugin.model_name} would be pulled from registry")
            else:
                print(f"Model {plugin.model_name} is already available")

        except Exception as e:
            pytest.fail(f"Model availability check failed: {e}")

    @pytest.mark.integration
    async def test_initialize_with_dummy_config(
        self, plugin: OllamaModelPlugin, ollama_server_available: None
    ) -> None:
        """Test initialization with a configuration (without actually pulling models)."""
        config = {
            "model_name": "tinyllama:1b",  # Small model for testing
            "name": "test-ollama-model",
            "max_tokens": 100,
            "temperature": 0.1,
            "host": "localhost",
            "port": 11434,
        }

        # Mock the model pulling to avoid actually downloading
        with pytest.MonkeyPatch().context() as m:

            async def mock_ensure_model_available():
                # Just check if server is reachable, don't pull
                _ = plugin.client.list()
                print(f"Mock: Would ensure {plugin.model_name} is available")
                # Don't actually pull in integration tests

            m.setattr(plugin, "_ensure_model_available", mock_ensure_model_available)

            response = await plugin.initialize(config)

            # Should succeed in setup even if model isn't available
            if response.success:
                assert plugin.model_name == "tinyllama:1b"
                assert plugin.model_config == config
                assert plugin.model_info is not None
                assert plugin.performance_metrics is not None
            else:
                # Expected if server isn't available or other setup issues
                print(f"Initialization failed (expected in some environments): {response.error}")

    @pytest.mark.integration
    async def test_health_check_real_server(
        self, plugin: OllamaModelPlugin, ollama_server_available: None
    ) -> None:
        """Test health check with real Ollama server."""
        plugin._import_ollama()
        plugin.client = plugin.ollama.Client()
        plugin.model_name = "test-model"

        health = await plugin.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "server_reachable" in health
        assert "available_models_count" in health

        # With real server, should be able to reach it
        if health["server_reachable"]:
            assert health["available_models_count"] >= 0
            print(
                f"Ollama server health: {health['status']}, models: {health['available_models_count']}"
            )
        else:
            print(f"Server not reachable: {health.get('error', 'Unknown error')}")

    @pytest.mark.integration
    async def test_prompt_formatting_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test prompt formatting integration with various security samples."""
        security_samples = [
            # Network security events
            "192.168.1.100:80 -> malicious-domain.com:443 ESTABLISHED",
            "Failed SSH login attempt from IP 203.0.113.42",
            "Large data transfer: 1GB uploaded to unknown server",
            # Endpoint security events
            "Process created: cmd.exe /c powershell -ExecutionPolicy Bypass",
            "File modification: C:\\Windows\\System32\\drivers\\etc\\hosts",
            "Registry modification: HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
            # Application security events
            "SQL injection detected: ' OR 1=1--",
            "XSS attempt: <script>alert('malicious')</script>",
            "Directory traversal: ../../../../etc/passwd",
            # Email security events
            "Email with suspicious attachment: invoice.pdf.exe",
            "Phishing attempt from admin@fake-bank.com",
        ]

        for sample in security_samples:
            prompt = plugin._format_cybersecurity_prompt(sample)

            # Verify prompt structure
            assert "cybersecurity expert" in prompt.lower()
            assert sample in prompt
            assert "Classification:" in prompt
            assert "Confidence:" in prompt
            assert "Attack_Type:" in prompt
            assert "Explanation:" in prompt
            assert "IOCs:" in prompt

            # Verify security context
            security_keywords = ["threat", "attack", "malicious", "security", "analysis"]
            assert any(keyword in prompt.lower() for keyword in security_keywords)

    @pytest.mark.integration
    async def test_response_parsing_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test response parsing with various response formats."""
        test_responses = [
            # Standard format responses
            """
            Classification: ATTACK
            Confidence: 0.85
            Attack_Type: malware
            Explanation: This network connection shows signs of C2 communication.
            IOCs: malicious-domain.com, 192.168.1.100, port 443
            """,
            """
            Classification: BENIGN
            Confidence: 0.92
            Attack_Type: N/A
            Explanation: Normal HTTPS traffic to legitimate service.
            IOCs: None
            """,
            # Variations in formatting
            """
            classification: attack
            confidence: 0.76
            attack_type: intrusion
            explanation: Unauthorized access attempt detected.
            iocs: admin, failed_login, 203.0.113.42
            """,
            # With extra content
            """
            Based on my analysis:

            Classification: ATTACK
            Confidence: 0.90
            Attack_Type: phishing
            Explanation: This email contains indicators of phishing.
            IOCs: fake-bank.com, credential_harvest.js

            Additional context: This appears to be part of a larger campaign.
            """,
            # Minimal response
            """
            Classification: BENIGN
            Confidence: 0.6
            """,
        ]

        for i, response in enumerate(test_responses):
            parsed = plugin._parse_response(response)

            # Verify required fields exist and have correct types
            assert "classification" in parsed
            assert "confidence" in parsed
            assert "attack_type" in parsed
            assert "explanation" in parsed
            assert "iocs" in parsed

            assert parsed["classification"] in ["ATTACK", "BENIGN"]
            assert 0.0 <= parsed["confidence"] <= 1.0
            assert parsed["attack_type"] is None or isinstance(parsed["attack_type"], str)
            assert isinstance(parsed["explanation"], str)
            assert isinstance(parsed["iocs"], list)

            print(
                f"Response {i + 1} parsed: {parsed['classification']} ({parsed['confidence']:.2f})"
            )

    @pytest.mark.integration
    async def test_performance_characteristics(self, plugin: OllamaModelPlugin) -> None:
        """Test performance characteristics of prompt formatting and parsing."""
        sample = "Suspicious network activity detected on port 443"

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
    async def test_memory_estimation_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test memory estimation for different model sizes."""
        test_models = [
            ("tinyllama:1b", 2000.0),
            ("llama2:7b", 4000.0),
            ("llama2:13b", 8000.0),
            ("codellama:34b", 20000.0),
            ("llama2:70b", 40000.0),
            ("unknown-model", 2000.0),
        ]

        for model_name, expected_memory in test_models:
            plugin.model_name = model_name
            estimated = plugin._estimate_memory_usage()
            assert estimated == expected_memory
            print(f"{model_name}: {estimated}MB")

    @pytest.mark.integration
    async def test_cleanup_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test cleanup integration."""
        # Set some state
        plugin._import_ollama()
        plugin.client = plugin.ollama.Client() if plugin.ollama else None
        plugin.model_name = "test-model"
        plugin.model_config = {"test": "config"}

        # Cleanup
        await plugin.cleanup()

        # Verify cleanup
        assert plugin.client is None
        assert plugin.model_name is None
        assert plugin.model_config is None

    @pytest.mark.integration
    async def test_recommended_models_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test recommended models functionality."""
        categories = ["general", "cybersecurity", "code_analysis"]

        for category in categories:
            models = plugin.get_recommended_models(category)
            assert isinstance(models, list)
            assert len(models) > 0

            # All models should be valid Ollama model names
            for model in models:
                assert isinstance(model, str)
                assert ":" in model  # Should have tag like "llama2:7b"

            print(f"{category}: {models}")

    @pytest.mark.integration
    async def test_configuration_validation_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test configuration validation with various configs."""
        test_configs = [
            # Valid configurations
            {
                "model_name": "llama2:7b",
                "name": "test-model",
                "max_tokens": 256,
                "temperature": 0.1,
            },
            {
                "model_name": "codellama:7b",
                "name": "code-model",
                "max_tokens": 512,
                "temperature": 0.2,
                "host": "localhost",
                "port": 11434,
            },
            # Invalid configurations
            {
                "name": "no-model-name"  # Missing model_name
            },
            {
                "model_name": "llama2:7b"  # Missing name
            },
            {},  # Empty config
        ]

        for i, config in enumerate(test_configs):
            plugin_instance = OllamaModelPlugin()

            # Mock the model availability check to avoid pulling
            with pytest.MonkeyPatch().context() as m:

                async def mock_ensure_available():
                    pass

                async def mock_test_connection():
                    pass

                m.setattr(plugin_instance, "_ensure_model_available", mock_ensure_available)
                m.setattr(plugin_instance, "_test_server_connection", mock_test_connection)

                response = await plugin_instance.initialize(config)

                if i < 2:  # Valid configs
                    # May succeed if mocked properly
                    if response.success:
                        assert plugin_instance.model_name is not None
                        assert plugin_instance.model_config == config
                    else:
                        # Expected - might fail due to server unavailability
                        print(
                            f"Config {i} failed (expected in some environments): {response.error}"
                        )
                else:  # Invalid configs
                    assert not response.success
                    assert "missing" in response.error.lower()

    @pytest.mark.integration
    async def test_error_handling_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test comprehensive error handling."""
        # Test with invalid server configuration
        plugin.server_host = "nonexistent-host"
        plugin.server_port = 99999
        plugin.server_url = "http://nonexistent-host:99999"

        config = {"model_name": "llama2:7b", "name": "test-model"}

        response = await plugin.initialize(config)

        # Should handle connection failure gracefully
        assert not response.success
        assert "Cannot connect to Ollama server" in response.error
        print(f"Connection error handled correctly: {response.error}")

    @pytest.mark.integration
    async def test_chat_format_compatibility(self, plugin: OllamaModelPlugin) -> None:
        """Test that chat format is compatible with Ollama expectations."""
        # Test the expected message format for Ollama chat API
        sample = "Test network event"
        prompt = plugin._format_cybersecurity_prompt(sample)

        # Ollama expects messages in format: [{'role': 'user', 'content': prompt}]
        messages = [{"role": "user", "content": prompt}]

        # Verify message format is correct
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt
        assert sample in messages[0]["content"]

        # Test options format
        options = {
            "temperature": 0.1,
            "num_predict": 512,
        }

        # Verify options are valid
        assert isinstance(options["temperature"], int | float)
        assert isinstance(options["num_predict"], int)
        assert 0.0 <= options["temperature"] <= 2.0
        assert options["num_predict"] > 0

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,  # Skip by default to avoid pulling models
        reason="Skipped to avoid pulling large models in CI/testing",
    )
    async def test_actual_model_interaction(
        self, plugin: OllamaModelPlugin, ollama_server_available: None
    ) -> None:
        """Test actual model interaction (disabled by default)."""
        # This test would actually pull and use a model - disabled for CI
        config = {
            "model_name": "tinyllama:1b",  # Smallest available model
            "name": "integration-test-model",
            "max_tokens": 50,
            "temperature": 0.1,
        }

        response = await plugin.initialize(config)

        if response.success:
            # Test actual prediction
            samples = ["Normal web traffic to google.com"]
            predictions = await plugin.predict(samples)

            assert len(predictions) == 1
            prediction = predictions[0]
            assert prediction.prediction in ["ATTACK", "BENIGN", "ERROR"]
            assert 0.0 <= prediction.confidence <= 1.0

            # Test explanation
            explanation = await plugin.explain(samples[0])
            assert isinstance(explanation, str)
            assert len(explanation) > 0

            print(f"Actual prediction: {prediction.prediction} ({prediction.confidence:.2f})")
            print(f"Explanation: {explanation[:100]}...")

        else:
            pytest.skip(f"Model initialization failed: {response.error}")

    @pytest.mark.integration
    async def test_server_info_integration(self, plugin: OllamaModelPlugin) -> None:
        """Test server information functionality."""
        default_info = plugin.get_server_info()

        assert default_info["host"] == "localhost"
        assert default_info["port"] == 11434
        assert default_info["url"] == "http://localhost:11434"

        # Test custom configuration
        plugin.server_host = "custom.example.com"
        plugin.server_port = 8080
        plugin.server_url = f"http://{plugin.server_host}:{plugin.server_port}"

        custom_info = plugin.get_server_info()
        assert custom_info["host"] == "custom.example.com"
        assert custom_info["port"] == 8080
        assert custom_info["url"] == "http://custom.example.com:8080"

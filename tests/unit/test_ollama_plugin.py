"""
Unit tests for the Ollama plugin.

This module tests Ollama model loading, server connection, prediction,
response parsing, and error handling with comprehensive mocking.
"""

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from benchmark.core.exceptions import BenchmarkError, ErrorCode
from benchmark.interfaces.model_interfaces import ModelInfo, PerformanceMetrics, Prediction
from benchmark.models.plugins.ollama_local import OllamaModelPlugin


class TestOllamaModelPlugin:
    """Test cases for OllamaModelPlugin."""

    @pytest_asyncio.fixture
    async def plugin(self) -> OllamaModelPlugin:
        """Create Ollama plugin for testing."""
        return OllamaModelPlugin()

    @pytest_asyncio.fixture
    async def initialized_plugin(self) -> OllamaModelPlugin:
        """Create initialized Ollama plugin for testing."""
        plugin = OllamaModelPlugin()

        # Mock the ollama import and client
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        plugin.ollama = mock_ollama
        plugin.client = mock_client
        plugin.model_name = "llama2:7b"
        plugin.model_config = {
            "name": "test-model",
            "model_name": "llama2:7b",
            "max_tokens": 512,
            "temperature": 0.1,
        }

        # Mock model info and metrics
        plugin.model_info = ModelInfo(model_id="test-model", name="test-model", type="ollama")
        plugin.performance_metrics = PerformanceMetrics(model_id="test-model")

        return plugin

    def test_plugin_initialization(self, plugin: OllamaModelPlugin) -> None:
        """Test basic plugin initialization."""
        assert plugin.client is None
        assert plugin.model_name is None
        assert plugin.model_config is None
        assert plugin.model_info is None
        assert plugin.performance_metrics is None
        assert plugin.ollama is None
        assert plugin.server_host == "localhost"
        assert plugin.server_port == 11434
        assert plugin.server_url == "http://localhost:11434"
        assert plugin.logger is not None

    def test_ollama_import_success(self, plugin: OllamaModelPlugin) -> None:
        """Test successful Ollama import."""
        with patch("importlib.import_module") as mock_import:
            mock_ollama = MagicMock()

            def side_effect(module_name):
                if module_name == "ollama":
                    return mock_ollama
                return MagicMock()

            mock_import.side_effect = side_effect

            result = plugin._import_ollama()
            assert result is True
            assert plugin.ollama is mock_ollama

    def test_ollama_import_failure(self, plugin: OllamaModelPlugin) -> None:
        """Test Ollama import failure."""
        with patch("importlib.import_module", side_effect=ImportError("Ollama not found")):
            result = plugin._import_ollama()
            assert result is False
            assert plugin.ollama is None

    @pytest.mark.asyncio
    async def test_initialize_ollama_import_failure(self, plugin: OllamaModelPlugin) -> None:
        """Test initialization with Ollama import failure."""
        config = {"model_name": "llama2:7b", "name": "test-model"}

        with patch.object(plugin, "_import_ollama", return_value=False):
            response = await plugin.initialize(config)

            assert response.success is False
            assert "Ollama library not available" in response.error

    @pytest.mark.asyncio
    async def test_initialize_missing_config_fields(self, plugin: OllamaModelPlugin) -> None:
        """Test initialization with missing config fields."""
        config = {"name": "test-model"}  # Missing model_name

        with patch.object(plugin, "_import_ollama", return_value=True):
            response = await plugin.initialize(config)

            assert response.success is False
            assert "Missing required config fields" in response.error

    @pytest.mark.asyncio
    async def test_initialize_server_connection_failure(self, plugin: OllamaModelPlugin) -> None:
        """Test initialization with server connection failure."""
        config = {"model_name": "llama2:7b", "name": "test-model"}

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama.Client.return_value = mock_client

        with (
            patch.object(plugin, "_import_ollama", return_value=True),
            patch.object(
                plugin, "_test_server_connection", side_effect=Exception("Connection failed")
            ),
        ):
            plugin.ollama = mock_ollama

            response = await plugin.initialize(config)

            assert response.success is False
            assert "Cannot connect to Ollama server" in response.error

    @pytest.mark.asyncio
    async def test_initialize_model_availability_failure(self, plugin: OllamaModelPlugin) -> None:
        """Test initialization with model availability check failure."""
        config = {"model_name": "llama2:7b", "name": "test-model"}

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with (
            patch.object(plugin, "_import_ollama", return_value=True),
            patch.object(plugin, "_test_server_connection", return_value=None),
            patch.object(
                plugin, "_ensure_model_available", side_effect=Exception("Model pull failed")
            ),
        ):
            plugin.ollama = mock_ollama

            response = await plugin.initialize(config)

            assert response.success is False
            assert "Failed to ensure model availability" in response.error

    @pytest.mark.asyncio
    async def test_initialize_success(self, plugin: OllamaModelPlugin) -> None:
        """Test successful initialization."""
        config = {
            "model_name": "llama2:7b",
            "name": "test-model",
            "max_tokens": 256,
            "temperature": 0.2,
            "host": "localhost",
            "port": 11434,
        }

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with (
            patch.object(plugin, "_import_ollama", return_value=True),
            patch.object(plugin, "_test_server_connection", return_value=None),
            patch.object(plugin, "_ensure_model_available", return_value=None),
            patch.object(plugin, "_estimate_memory_usage", return_value=4000.0),
        ):
            plugin.ollama = mock_ollama

            response = await plugin.initialize(config)

            assert response.success is True
            assert plugin.client is mock_client
            assert plugin.model_name == "llama2:7b"
            assert plugin.model_config == config
            assert plugin.model_info is not None
            assert plugin.performance_metrics is not None

            # Check model info
            assert plugin.model_info.name == "test-model"
            assert plugin.model_info.type == "ollama"
            assert "model_name" in plugin.model_info.parameters

    @pytest.mark.asyncio
    async def test_test_server_connection_success(self, plugin: OllamaModelPlugin) -> None:
        """Test successful server connection test."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "test-model"}]}
        plugin.client = mock_client

        # Should not raise exception
        await plugin._test_server_connection()
        mock_client.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_server_connection_failure(self, plugin: OllamaModelPlugin) -> None:
        """Test server connection test failure."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        plugin.client = mock_client

        with pytest.raises(Exception, match="Server connection test failed"):
            await plugin._test_server_connection()

    @pytest.mark.asyncio
    async def test_ensure_model_available_already_exists(self, plugin: OllamaModelPlugin) -> None:
        """Test model availability check when model already exists."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama2:7b"}]}
        plugin.client = mock_client
        plugin.model_name = "llama2:7b"

        await plugin._ensure_model_available()

        mock_client.list.assert_called_once()
        mock_client.pull.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_model_available_needs_pull(self, plugin: OllamaModelPlugin) -> None:
        """Test model availability check when model needs to be pulled."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "other-model"}]}
        plugin.client = mock_client
        plugin.model_name = "llama2:7b"

        await plugin._ensure_model_available()

        mock_client.list.assert_called_once()
        mock_client.pull.assert_called_once_with("llama2:7b")

    @pytest.mark.asyncio
    async def test_ensure_model_available_failure(self, plugin: OllamaModelPlugin) -> None:
        """Test model availability check failure."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Server error")
        plugin.client = mock_client
        plugin.model_name = "llama2:7b"

        with pytest.raises(Exception, match="Failed to ensure model availability"):
            await plugin._ensure_model_available()

    def test_memory_estimation(self, plugin: OllamaModelPlugin) -> None:
        """Test memory usage estimation."""
        plugin.model_name = "llama2:7b"
        assert plugin._estimate_memory_usage() == 4000.0

        plugin.model_name = "llama2:13b"
        assert plugin._estimate_memory_usage() == 8000.0

        plugin.model_name = "codellama:34b"
        assert plugin._estimate_memory_usage() == 20000.0

        plugin.model_name = "llama2:70b"
        assert plugin._estimate_memory_usage() == 40000.0

        plugin.model_name = "unknown-model"
        assert plugin._estimate_memory_usage() == 2000.0

    def test_cybersecurity_prompt_formatting(self, plugin: OllamaModelPlugin) -> None:
        """Test cybersecurity prompt formatting."""
        sample = "suspicious network connection to 192.168.1.100"

        prompt = plugin._format_cybersecurity_prompt(sample)

        assert "cybersecurity expert" in prompt.lower()
        assert sample in prompt
        assert "Classification:" in prompt
        assert "Confidence:" in prompt
        assert "Attack_Type:" in prompt
        assert "Explanation:" in prompt
        assert "IOCs:" in prompt

    def test_response_parsing_attack(self, plugin: OllamaModelPlugin) -> None:
        """Test parsing attack response."""
        response = """
        Classification: ATTACK
        Confidence: 0.85
        Attack_Type: malware
        Explanation: This shows signs of malicious activity with suspicious network connections.
        IOCs: 192.168.1.100, suspicious.exe, port 1337
        """

        parsed = plugin._parse_response(response)

        assert parsed["classification"] == "ATTACK"
        assert parsed["confidence"] == 0.85
        assert parsed["attack_type"] == "malware"
        assert "malicious activity" in parsed["explanation"]
        assert "192.168.1.100" in parsed["iocs"]
        assert "suspicious.exe" in parsed["iocs"]
        assert "port 1337" in parsed["iocs"]

    def test_response_parsing_benign(self, plugin: OllamaModelPlugin) -> None:
        """Test parsing benign response."""
        response = """
        Classification: BENIGN
        Confidence: 0.92
        Attack_Type: N/A
        Explanation: Normal system activity, no threats detected.
        IOCs: None
        """

        parsed = plugin._parse_response(response)

        assert parsed["classification"] == "BENIGN"
        assert parsed["confidence"] == 0.92
        assert parsed["attack_type"] is None
        assert "Normal system activity" in parsed["explanation"]
        assert parsed["iocs"] == []

    def test_response_parsing_malformed(self, plugin: OllamaModelPlugin) -> None:
        """Test parsing malformed response."""
        response = "This is a completely malformed response without proper fields"

        parsed = plugin._parse_response(response)

        assert parsed["classification"] == "BENIGN"  # Default
        assert parsed["confidence"] == 0.5  # Default
        assert parsed["attack_type"] is None
        assert parsed["explanation"] == ""
        assert parsed["iocs"] == []

    def test_response_parsing_error(self, plugin: OllamaModelPlugin) -> None:
        """Test response parsing with error."""
        with patch("re.search", side_effect=Exception("Regex error")):
            parsed = plugin._parse_response("test response")

            assert parsed["classification"] == "BENIGN"
            assert parsed["confidence"] == 0.0
            assert "Parse error" in parsed["explanation"]

    @pytest.mark.asyncio
    async def test_predict_not_initialized(self, plugin: OllamaModelPlugin) -> None:
        """Test prediction with uninitialized model."""
        samples = ["test sample"]

        with pytest.raises(BenchmarkError) as exc_info:
            await plugin.predict(samples)

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_predict_success(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test successful prediction."""
        mock_response = {
            "message": {
                "content": """
                Classification: ATTACK
                Confidence: 0.85
                Attack_Type: malware
                Explanation: Suspicious behavior detected
                IOCs: malicious.exe
                """
            }
        }

        initialized_plugin.client.chat.return_value = mock_response
        samples = ["suspicious network activity"]

        with patch("time.time", side_effect=[0.0, 0.5]):  # Mock inference time
            predictions = await initialized_plugin.predict(samples)

        assert len(predictions) == 1
        prediction = predictions[0]

        assert isinstance(prediction, Prediction)
        assert prediction.sample_id == "ollama_0"
        assert prediction.input_text == samples[0]
        assert prediction.prediction == "ATTACK"
        assert prediction.confidence == 0.85
        assert prediction.attack_type == "malware"
        assert prediction.inference_time_ms == 500.0  # 0.5 * 1000
        assert "malicious.exe" in prediction.metadata["iocs"]
        assert prediction.metadata["model_name"] == "llama2:7b"

    @pytest.mark.asyncio
    async def test_predict_with_error(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test prediction with generation error."""
        initialized_plugin.client.chat.side_effect = Exception("Chat failed")
        samples = ["test sample"]

        predictions = await initialized_plugin.predict(samples)

        assert len(predictions) == 1
        prediction = predictions[0]

        assert prediction.prediction == "ERROR"
        assert prediction.confidence == 0.0
        assert "Chat failed" in prediction.explanation
        assert "error" in prediction.metadata

    @pytest.mark.asyncio
    async def test_predict_batch_processing(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test batch prediction processing."""
        mock_response = {"message": {"content": "Classification: BENIGN\nConfidence: 0.9"}}

        initialized_plugin.client.chat.return_value = mock_response
        samples = ["sample 1", "sample 2", "sample 3"]

        predictions = await initialized_plugin.predict(samples)

        assert len(predictions) == 3
        assert all(p.sample_id.startswith("ollama_") for p in predictions)
        assert all(p.prediction == "BENIGN" for p in predictions)
        assert initialized_plugin.client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_explain_success(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test successful explanation generation."""
        mock_response = {"message": {"content": "This event shows suspicious network activity..."}}

        initialized_plugin.client.chat.return_value = mock_response
        sample = "network event"

        explanation = await initialized_plugin.explain(sample)

        assert explanation == "This event shows suspicious network activity..."
        initialized_plugin.client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_not_initialized(self, plugin: OllamaModelPlugin) -> None:
        """Test explanation with uninitialized model."""
        explanation = await plugin.explain("test sample")

        assert explanation == "Model not initialized"

    @pytest.mark.asyncio
    async def test_explain_error(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test explanation with error."""
        initialized_plugin.client.chat.side_effect = Exception("Generation error")

        explanation = await initialized_plugin.explain("test sample")

        assert "Explanation generation failed" in explanation
        assert "Generation error" in explanation

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test getting model info."""
        info = await initialized_plugin.get_model_info()

        assert info is initialized_plugin.model_info
        assert info.name == "test-model"
        assert info.type == "ollama"

    @pytest.mark.asyncio
    async def test_get_model_info_not_initialized(self, plugin: OllamaModelPlugin) -> None:
        """Test getting model info when not initialized."""
        with pytest.raises(BenchmarkError) as exc_info:
            await plugin.get_model_info()

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_get_performance_metrics_success(
        self, initialized_plugin: OllamaModelPlugin
    ) -> None:
        """Test getting performance metrics."""
        metrics = await initialized_plugin.get_performance_metrics()

        assert metrics is initialized_plugin.performance_metrics
        assert metrics.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_get_performance_metrics_not_initialized(self, plugin: OllamaModelPlugin) -> None:
        """Test getting performance metrics when not initialized."""
        with pytest.raises(BenchmarkError) as exc_info:
            await plugin.get_performance_metrics()

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test health check with healthy model and server."""
        # Mock successful server communication
        initialized_plugin.client.list.return_value = {
            "models": [{"name": "llama2:7b"}, {"name": "other-model"}]
        }

        health = await initialized_plugin.health_check()

        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["client_connected"] is True
        assert health["server_reachable"] is True
        assert health["model_available"] is True
        assert health["available_models_count"] == 2

    @pytest.mark.asyncio
    async def test_health_check_model_not_available(
        self, initialized_plugin: OllamaModelPlugin
    ) -> None:
        """Test health check when model is not available on server."""
        # Mock server response without the target model
        initialized_plugin.client.list.return_value = {"models": [{"name": "other-model"}]}

        health = await initialized_plugin.health_check()

        assert health["status"] == "degraded"
        assert health["model_available"] is False
        assert "Model llama2:7b not available on server" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_server_unreachable(
        self, initialized_plugin: OllamaModelPlugin
    ) -> None:
        """Test health check when server is unreachable."""
        initialized_plugin.client.list.side_effect = Exception("Connection refused")

        health = await initialized_plugin.health_check()

        assert health["status"] == "unhealthy"
        assert health["server_reachable"] is False
        assert "Cannot reach Ollama server" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, plugin: OllamaModelPlugin) -> None:
        """Test health check with uninitialized plugin."""
        health = await plugin.health_check()

        assert health["status"] == "unhealthy"
        assert health["model_loaded"] is False
        assert health["client_connected"] is False
        assert "Client not initialized" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_error(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test health check with error."""
        # Mock an error during health check
        with patch.object(
            initialized_plugin, "client", side_effect=Exception("Health check error")
        ):
            health = await initialized_plugin.health_check()

            assert health["status"] in ["error", "degraded", "unhealthy"]
            assert "error" in health or "status" in health

    @pytest.mark.asyncio
    async def test_cleanup_success(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test successful cleanup."""
        await initialized_plugin.cleanup()

        assert initialized_plugin.client is None
        assert initialized_plugin.model_name is None
        assert initialized_plugin.model_config is None

    @pytest.mark.asyncio
    async def test_cleanup_error(self, initialized_plugin: OllamaModelPlugin) -> None:
        """Test cleanup with error (should not raise)."""
        # Mock an error during cleanup by making logger.info raise
        with (
            patch.object(initialized_plugin.logger, "info", side_effect=Exception("Logger error")),
            patch.object(initialized_plugin.logger, "error") as mock_error,
        ):
            await initialized_plugin.cleanup()

            # Should log error but not raise
            mock_error.assert_called()

    def test_get_recommended_models(self, plugin: OllamaModelPlugin) -> None:
        """Test getting recommended models."""
        general_models = plugin.get_recommended_models("general")
        assert "llama2:7b" in general_models
        assert "mistral:7b" in general_models

        cybersecurity_models = plugin.get_recommended_models("cybersecurity")
        assert "llama2:7b" in cybersecurity_models
        assert "codellama:7b" in cybersecurity_models

        code_models = plugin.get_recommended_models("code_analysis")
        assert "codellama:7b" in code_models
        assert "codellama:13b" in code_models

        # Test default fallback
        default_models = plugin.get_recommended_models("unknown_category")
        assert default_models == plugin.recommended_models["general"]

    def test_get_server_info(self, plugin: OllamaModelPlugin) -> None:
        """Test getting server information."""
        info = plugin.get_server_info()

        assert info["host"] == "localhost"
        assert info["port"] == 11434
        assert info["url"] == "http://localhost:11434"

    def test_custom_server_config(self, plugin: OllamaModelPlugin) -> None:
        """Test custom server configuration."""
        plugin.server_host = "custom-host"
        plugin.server_port = 9999
        plugin.server_url = "http://custom-host:9999"

        info = plugin.get_server_info()

        assert info["host"] == "custom-host"
        assert info["port"] == 9999
        assert info["url"] == "http://custom-host:9999"

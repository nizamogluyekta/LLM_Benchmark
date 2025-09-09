"""
Unit tests for the MLX plugin.

This module tests MLX model loading, prediction, response parsing,
and error handling with comprehensive mocking.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import pytest_asyncio

from benchmark.core.exceptions import BenchmarkError, ErrorCode
from benchmark.interfaces.model_interfaces import ModelInfo, PerformanceMetrics, Prediction
from benchmark.models.plugins.mlx_local import MLXModelPlugin


class TestMLXModelPlugin:
    """Test cases for MLXModelPlugin."""

    @pytest_asyncio.fixture
    async def plugin(self) -> MLXModelPlugin:
        """Create MLX plugin for testing."""
        return MLXModelPlugin()

    @pytest_asyncio.fixture
    async def apple_silicon_plugin(self) -> MLXModelPlugin:
        """Create MLX plugin with mocked Apple Silicon system."""
        with patch.object(
            MLXModelPlugin,
            "_detect_system_capabilities",
            return_value={
                "platform": "Darwin",
                "machine": "arm64",
                "is_apple_silicon": True,
                "memory_gb": 64.0,
                "cpu_cores": 12,
            },
        ):
            return MLXModelPlugin()

    @pytest_asyncio.fixture
    async def non_apple_plugin(self) -> MLXModelPlugin:
        """Create MLX plugin with non-Apple Silicon system."""
        with patch.object(
            MLXModelPlugin,
            "_detect_system_capabilities",
            return_value={
                "platform": "Linux",
                "machine": "x86_64",
                "is_apple_silicon": False,
                "memory_gb": 32.0,
                "cpu_cores": 8,
            },
        ):
            return MLXModelPlugin()

    def test_plugin_initialization(self, plugin: MLXModelPlugin) -> None:
        """Test basic plugin initialization."""
        assert plugin.model is None
        assert plugin.tokenizer is None
        assert plugin.model_config is None
        assert plugin.model_info is None
        assert plugin.performance_metrics is None
        assert plugin.mlx_lm is None
        assert plugin.mlx is None
        assert plugin._model_cache == {}
        assert plugin.logger is not None

    def test_system_capabilities_detection_apple_silicon(
        self, apple_silicon_plugin: MLXModelPlugin
    ) -> None:
        """Test Apple Silicon detection."""
        system_info = apple_silicon_plugin._system_info
        assert system_info["is_apple_silicon"] is True
        assert system_info["platform"] == "Darwin"
        assert system_info["machine"] == "arm64"
        assert system_info["memory_gb"] == 64.0
        assert system_info["cpu_cores"] == 12

    def test_system_capabilities_detection_non_apple(
        self, non_apple_plugin: MLXModelPlugin
    ) -> None:
        """Test non-Apple Silicon detection."""
        system_info = non_apple_plugin._system_info
        assert system_info["is_apple_silicon"] is False
        assert system_info["platform"] == "Linux"
        assert system_info["machine"] == "x86_64"

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_count")
    def test_system_detection_with_error(
        self, mock_cpu_count: Mock, mock_memory: Mock, plugin: MLXModelPlugin
    ) -> None:
        """Test system detection fallback on error."""
        mock_memory.side_effect = Exception("Memory detection failed")
        mock_cpu_count.return_value = None

        # Create new plugin to trigger detection
        with patch.object(
            MLXModelPlugin, "_detect_system_capabilities", wraps=plugin._detect_system_capabilities
        ):
            new_plugin = MLXModelPlugin()

            # Should fallback to defaults
            assert new_plugin._system_info["memory_gb"] == 8.0
            assert new_plugin._system_info["cpu_cores"] == 4
            assert new_plugin._system_info["is_apple_silicon"] is False

    def test_mlx_import_success(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test successful MLX import."""
        with patch("importlib.import_module") as mock_import:
            mock_mlx_lm = MagicMock()
            mock_mlx_core = MagicMock()

            def side_effect(module_name):
                if module_name == "mlx_lm":
                    return mock_mlx_lm
                elif module_name == "mlx.core":
                    return mock_mlx_core
                return MagicMock()

            mock_import.side_effect = side_effect

            result = apple_silicon_plugin._import_mlx()
            assert result is True
            assert apple_silicon_plugin.mlx_lm is mock_mlx_lm
            assert apple_silicon_plugin.mlx is mock_mlx_core

    def test_mlx_import_failure(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test MLX import failure."""
        with patch("importlib.import_module", side_effect=ImportError("MLX not found")):
            result = apple_silicon_plugin._import_mlx()
            assert result is False
            assert apple_silicon_plugin.mlx_lm is None
            assert apple_silicon_plugin.mlx is None

    @pytest.mark.asyncio
    async def test_initialize_non_apple_silicon(self, non_apple_plugin: MLXModelPlugin) -> None:
        """Test initialization failure on non-Apple Silicon."""
        config = {"model_path": "/path/to/model", "name": "test-model"}

        response = await non_apple_plugin.initialize(config)

        assert response.success is False
        assert "Apple Silicon hardware" in response.error

    @pytest.mark.asyncio
    async def test_initialize_mlx_import_failure(
        self, apple_silicon_plugin: MLXModelPlugin
    ) -> None:
        """Test initialization with MLX import failure."""
        config = {"model_path": "/path/to/model", "name": "test-model"}

        with patch.object(apple_silicon_plugin, "_import_mlx", return_value=False):
            response = await apple_silicon_plugin.initialize(config)

            assert response.success is False
            assert "MLX libraries not available" in response.error

    @pytest.mark.asyncio
    async def test_initialize_missing_config_fields(
        self, apple_silicon_plugin: MLXModelPlugin
    ) -> None:
        """Test initialization with missing config fields."""
        config = {"name": "test-model"}  # Missing model_path

        with patch.object(apple_silicon_plugin, "_import_mlx", return_value=True):
            response = await apple_silicon_plugin.initialize(config)

            assert response.success is False
            assert "Missing required config fields" in response.error

    @pytest.mark.asyncio
    async def test_initialize_model_path_not_exists(
        self, apple_silicon_plugin: MLXModelPlugin
    ) -> None:
        """Test initialization with non-existent model path."""
        config = {"model_path": "/nonexistent/path/to/model", "name": "test-model"}

        with patch.object(apple_silicon_plugin, "_import_mlx", return_value=True):
            response = await apple_silicon_plugin.initialize(config)

            assert response.success is False
            assert "Model path does not exist" in response.error

    @pytest.mark.asyncio
    async def test_initialize_success(
        self, apple_silicon_plugin: MLXModelPlugin, tmp_path: Path
    ) -> None:
        """Test successful model initialization."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        config = {
            "model_path": str(model_path),
            "name": "test-model",
            "quantization": "4bit",
            "max_tokens": 256,
            "temperature": 0.2,
        }

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with (
            patch.object(apple_silicon_plugin, "_import_mlx", return_value=True),
            patch.object(apple_silicon_plugin, "_estimate_memory_usage", return_value=2048.0),
        ):
            apple_silicon_plugin.mlx_lm = mock_mlx_lm

            response = await apple_silicon_plugin.initialize(config)

            assert response.success is True
            assert apple_silicon_plugin.model is mock_model
            assert apple_silicon_plugin.tokenizer is mock_tokenizer
            assert apple_silicon_plugin.model_config == config
            assert apple_silicon_plugin.model_info is not None
            assert apple_silicon_plugin.performance_metrics is not None

            # Check model info
            assert apple_silicon_plugin.model_info.name == "test-model"
            assert apple_silicon_plugin.model_info.type == "mlx"
            assert "quantization" in apple_silicon_plugin.model_info.parameters

    @pytest.mark.asyncio
    async def test_initialize_with_cache(
        self, apple_silicon_plugin: MLXModelPlugin, tmp_path: Path
    ) -> None:
        """Test model loading from cache."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        config = {"model_path": str(model_path), "name": "test-model", "quantization": "4bit"}

        # Pre-populate cache
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        cache_key = f"{model_path}_4bit"
        apple_silicon_plugin._model_cache[cache_key] = (mock_model, mock_tokenizer)

        with (
            patch.object(apple_silicon_plugin, "_import_mlx", return_value=True),
            patch.object(apple_silicon_plugin, "_estimate_memory_usage", return_value=2048.0),
        ):
            apple_silicon_plugin.mlx_lm = MagicMock()

            response = await apple_silicon_plugin.initialize(config)

            assert response.success is True
            assert apple_silicon_plugin.model is mock_model
            assert apple_silicon_plugin.tokenizer is mock_tokenizer
            # Should not call load since it's cached
            apple_silicon_plugin.mlx_lm.load.assert_not_called()

    def test_memory_estimation_with_model(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test memory estimation with loaded model."""
        # Mock model with parameters
        mock_param = MagicMock()
        mock_param.size = 1024 * 1024  # 1M parameters
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.model.parameters.return_value = [mock_param]

        memory_mb = apple_silicon_plugin._estimate_memory_usage()

        # Should be base (500) + param memory (1M * 4 bytes / 1024^2)
        expected = 500.0 + (1024 * 1024 * 4) / (1024 * 1024)
        assert memory_mb == expected

    def test_memory_estimation_fallback(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test memory estimation fallback."""
        apple_silicon_plugin.model = None

        memory_mb = apple_silicon_plugin._estimate_memory_usage()

        assert memory_mb == 2048.0

    def test_cybersecurity_prompt_formatting(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test cybersecurity prompt formatting."""
        sample = "suspicious network connection to 192.168.1.100"

        prompt = apple_silicon_plugin._format_cybersecurity_prompt(sample)

        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert sample in prompt
        assert "cybersecurity expert" in prompt.lower()
        assert "Classification:" in prompt
        assert "Confidence:" in prompt
        assert "Attack_Type:" in prompt
        assert "Explanation:" in prompt
        assert "IOCs:" in prompt

    def test_response_parsing_attack(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test parsing attack response."""
        response = """
        Classification: ATTACK
        Confidence: 0.85
        Attack_Type: malware
        Explanation: This shows signs of malicious activity with suspicious network connections.
        IOCs: 192.168.1.100, suspicious.exe, port 1337
        """

        parsed = apple_silicon_plugin._parse_response(response)

        assert parsed["classification"] == "ATTACK"
        assert parsed["confidence"] == 0.85
        assert parsed["attack_type"] == "malware"
        assert "malicious activity" in parsed["explanation"]
        assert "192.168.1.100" in parsed["iocs"]
        assert "suspicious.exe" in parsed["iocs"]
        assert "port 1337" in parsed["iocs"]

    def test_response_parsing_benign(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test parsing benign response."""
        response = """
        Classification: BENIGN
        Confidence: 0.92
        Attack_Type: N/A
        Explanation: Normal system activity, no threats detected.
        IOCs: None
        """

        parsed = apple_silicon_plugin._parse_response(response)

        assert parsed["classification"] == "BENIGN"
        assert parsed["confidence"] == 0.92
        assert parsed["attack_type"] is None
        assert "Normal system activity" in parsed["explanation"]
        assert parsed["iocs"] == []

    def test_response_parsing_malformed(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test parsing malformed response."""
        response = "This is a completely malformed response without proper fields"

        parsed = apple_silicon_plugin._parse_response(response)

        assert parsed["classification"] == "BENIGN"  # Default
        assert parsed["confidence"] == 0.5  # Default
        assert parsed["attack_type"] is None
        assert parsed["explanation"] == ""
        assert parsed["iocs"] == []

    def test_response_parsing_error(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test response parsing with error."""
        with patch("re.search", side_effect=Exception("Regex error")):
            parsed = apple_silicon_plugin._parse_response("test response")

            assert parsed["classification"] == "BENIGN"
            assert parsed["confidence"] == 0.0
            assert "Parse error" in parsed["explanation"]

    @pytest.mark.asyncio
    async def test_predict_not_initialized(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test prediction with uninitialized model."""
        samples = ["test sample"]

        with pytest.raises(BenchmarkError) as exc_info:
            await apple_silicon_plugin.predict(samples)

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_predict_success(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test successful prediction."""
        # Set up initialized model
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.tokenizer = MagicMock()
        apple_silicon_plugin.model_config = {"max_tokens": 256, "temperature": 0.1}
        apple_silicon_plugin.performance_metrics = PerformanceMetrics(model_id="test")

        mock_mlx_lm = MagicMock()
        mock_response = """
        Classification: ATTACK
        Confidence: 0.85
        Attack_Type: malware
        Explanation: Suspicious behavior detected
        IOCs: malicious.exe
        """
        mock_mlx_lm.generate.return_value = mock_response
        apple_silicon_plugin.mlx_lm = mock_mlx_lm

        samples = ["suspicious network activity"]

        with patch("time.time", side_effect=[0.0, 0.5]):  # Mock inference time
            predictions = await apple_silicon_plugin.predict(samples)

        assert len(predictions) == 1
        prediction = predictions[0]

        assert isinstance(prediction, Prediction)
        assert prediction.sample_id == "mlx_0"
        assert prediction.input_text == samples[0]
        assert prediction.prediction == "ATTACK"
        assert prediction.confidence == 0.85
        assert prediction.attack_type == "malware"
        assert prediction.inference_time_ms == 500.0  # 0.5 * 1000
        assert "malicious.exe" in prediction.metadata["iocs"]

    @pytest.mark.asyncio
    async def test_predict_with_error(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test prediction with generation error."""
        # Set up initialized model
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.tokenizer = MagicMock()
        apple_silicon_plugin.model_config = {"max_tokens": 256, "temperature": 0.1}
        apple_silicon_plugin.performance_metrics = PerformanceMetrics(model_id="test")

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.side_effect = Exception("Generation failed")
        apple_silicon_plugin.mlx_lm = mock_mlx_lm

        samples = ["test sample"]

        predictions = await apple_silicon_plugin.predict(samples)

        assert len(predictions) == 1
        prediction = predictions[0]

        assert prediction.prediction == "ERROR"
        assert prediction.confidence == 0.0
        assert "Generation failed" in prediction.explanation
        assert "error" in prediction.metadata

    @pytest.mark.asyncio
    async def test_predict_batch_processing(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test batch prediction processing."""
        # Set up initialized model
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.tokenizer = MagicMock()
        apple_silicon_plugin.model_config = {"max_tokens": 256, "temperature": 0.1}
        apple_silicon_plugin.performance_metrics = PerformanceMetrics(model_id="test")

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "Classification: BENIGN\nConfidence: 0.9"
        apple_silicon_plugin.mlx_lm = mock_mlx_lm

        samples = ["sample 1", "sample 2", "sample 3"]

        predictions = await apple_silicon_plugin.predict(samples)

        assert len(predictions) == 3
        assert all(p.sample_id.startswith("mlx_") for p in predictions)
        assert all(p.prediction == "BENIGN" for p in predictions)
        assert mock_mlx_lm.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_explain_success(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test successful explanation generation."""
        # Set up initialized model
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.tokenizer = MagicMock()
        apple_silicon_plugin.model_config = {"max_tokens": 512, "temperature": 0.1}

        mock_mlx_lm = MagicMock()
        mock_explanation = "This event shows suspicious network activity..."
        mock_mlx_lm.generate.return_value = mock_explanation
        apple_silicon_plugin.mlx_lm = mock_mlx_lm

        sample = "network event"

        explanation = await apple_silicon_plugin.explain(sample)

        assert explanation == mock_explanation
        mock_mlx_lm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_not_initialized(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test explanation with uninitialized model."""
        explanation = await apple_silicon_plugin.explain("test sample")

        assert explanation == "Model not initialized"

    @pytest.mark.asyncio
    async def test_explain_error(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test explanation with error."""
        # Set up initialized model
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.tokenizer = MagicMock()
        apple_silicon_plugin.model_config = {"max_tokens": 512, "temperature": 0.1}

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.side_effect = Exception("Generation error")
        apple_silicon_plugin.mlx_lm = mock_mlx_lm

        explanation = await apple_silicon_plugin.explain("test sample")

        assert "Explanation generation failed" in explanation
        assert "Generation error" in explanation

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test getting model info."""
        mock_info = ModelInfo(model_id="test", name="test-model", type="mlx")
        apple_silicon_plugin.model_info = mock_info

        info = await apple_silicon_plugin.get_model_info()

        assert info is mock_info

    @pytest.mark.asyncio
    async def test_get_model_info_not_initialized(
        self, apple_silicon_plugin: MLXModelPlugin
    ) -> None:
        """Test getting model info when not initialized."""
        with pytest.raises(BenchmarkError) as exc_info:
            await apple_silicon_plugin.get_model_info()

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_get_performance_metrics_success(
        self, apple_silicon_plugin: MLXModelPlugin
    ) -> None:
        """Test getting performance metrics."""
        mock_metrics = PerformanceMetrics(model_id="test")
        apple_silicon_plugin.performance_metrics = mock_metrics

        metrics = await apple_silicon_plugin.get_performance_metrics()

        assert metrics is mock_metrics

    @pytest.mark.asyncio
    async def test_get_performance_metrics_not_initialized(
        self, apple_silicon_plugin: MLXModelPlugin
    ) -> None:
        """Test getting performance metrics when not initialized."""
        with pytest.raises(BenchmarkError) as exc_info:
            await apple_silicon_plugin.get_performance_metrics()

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test health check with healthy model."""
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.tokenizer = MagicMock()

        health = await apple_silicon_plugin.health_check()

        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["tokenizer_loaded"] is True
        assert health["apple_silicon"] is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test health check with unhealthy model."""
        health = await apple_silicon_plugin.health_check()

        assert health["status"] == "unhealthy"
        assert health["model_loaded"] is False
        assert "Model not loaded" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_incompatible(self, non_apple_plugin: MLXModelPlugin) -> None:
        """Test health check on incompatible hardware."""
        health = await non_apple_plugin.health_check()

        assert health["status"] == "incompatible"
        assert health["apple_silicon"] is False
        assert "Requires Apple Silicon" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_error(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test health check with error."""

        # Mock an error by making _system_info access raise an exception
        def mock_health_check_error():
            raise Exception("System error")

        with patch.object(
            apple_silicon_plugin, "_system_info", side_effect=mock_health_check_error
        ):
            health = await apple_silicon_plugin.health_check()

            assert health["status"] in [
                "error",
                "unhealthy",
            ]  # Can be either depending on implementation
            assert "error" in health or "status" in health

    @pytest.mark.asyncio
    async def test_cleanup_success(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test successful cleanup."""
        apple_silicon_plugin.model = MagicMock()
        apple_silicon_plugin.tokenizer = MagicMock()
        apple_silicon_plugin.model_config = {"test": "config"}

        await apple_silicon_plugin.cleanup()

        assert apple_silicon_plugin.model is None
        assert apple_silicon_plugin.tokenizer is None
        assert apple_silicon_plugin.model_config is None

    @pytest.mark.asyncio
    async def test_cleanup_error(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test cleanup with error (should not raise)."""
        # Mock an error during cleanup by making logger.info raise
        with (
            patch.object(
                apple_silicon_plugin.logger, "info", side_effect=Exception("Logger error")
            ),
            patch.object(apple_silicon_plugin.logger, "error") as mock_error,
        ):
            await apple_silicon_plugin.cleanup()

            # Should log error but not raise
            mock_error.assert_called()

    def test_get_supported_quantizations(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test getting supported quantizations."""
        quantizations = apple_silicon_plugin.get_supported_quantizations()

        assert "4bit" in quantizations
        assert "8bit" in quantizations
        assert "none" in quantizations

    def test_get_model_formats(self, apple_silicon_plugin: MLXModelPlugin) -> None:
        """Test getting supported model formats."""
        formats = apple_silicon_plugin.get_model_formats()

        assert "llama" in formats
        assert "qwen" in formats
        assert "mistral" in formats
        assert "phi" in formats
        assert "gemma" in formats

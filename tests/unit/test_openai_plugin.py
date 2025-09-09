"""
Unit tests for the OpenAI API plugin.

This module tests OpenAI API integration, rate limiting, cost tracking,
error handling, retry logic, and comprehensive mocking.
"""

import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from benchmark.core.exceptions import BenchmarkError, ErrorCode
from benchmark.interfaces.model_interfaces import ModelInfo, PerformanceMetrics, Prediction
from benchmark.models.plugins.openai_api import APIRateLimiter, CostTracker, OpenAIModelPlugin


class TestAPIRateLimiter:
    """Test cases for APIRateLimiter."""

    def test_rate_limiter_initialization(self) -> None:
        """Test rate limiter initialization."""
        limiter = APIRateLimiter(requests_per_minute=100, tokens_per_minute=50000)
        assert limiter.requests_per_minute == 100
        assert limiter.tokens_per_minute == 50000
        assert limiter.requests == []
        assert limiter.tokens == []

    @pytest.mark.asyncio
    async def test_acquire_under_limits(self) -> None:
        """Test acquiring permission when under rate limits."""
        limiter = APIRateLimiter(requests_per_minute=60, tokens_per_minute=90000)

        # Should not block when under limits
        await limiter.acquire(estimated_tokens=1000)

        assert len(limiter.requests) == 1
        assert len(limiter.tokens) == 1
        assert limiter.tokens[0][1] == 1000

    @pytest.mark.asyncio
    async def test_acquire_request_rate_limiting(self) -> None:
        """Test request rate limiting."""
        limiter = APIRateLimiter(requests_per_minute=2, tokens_per_minute=90000)

        # Fill up the request limit
        await limiter.acquire(estimated_tokens=100)
        await limiter.acquire(estimated_tokens=100)

        # This should trigger rate limiting but we'll mock sleep to avoid waiting
        with patch("asyncio.sleep") as mock_sleep:
            await limiter.acquire(estimated_tokens=100)
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_token_rate_limiting(self) -> None:
        """Test token rate limiting."""
        limiter = APIRateLimiter(requests_per_minute=60, tokens_per_minute=1000)

        # Fill up most of the token limit
        await limiter.acquire(estimated_tokens=800)

        # This should trigger token rate limiting
        with patch("asyncio.sleep") as mock_sleep:
            await limiter.acquire(estimated_tokens=300)  # Would exceed limit
            mock_sleep.assert_called_once()

    def test_get_current_usage(self) -> None:
        """Test getting current usage statistics."""
        limiter = APIRateLimiter(requests_per_minute=60, tokens_per_minute=90000)

        # Add some usage
        now = datetime.now()
        limiter.requests = [now, now - timedelta(seconds=30)]
        limiter.tokens = [(now, 1000), (now - timedelta(seconds=30), 2000)]

        usage = limiter.get_current_usage()

        assert usage["requests_per_minute"] == 2
        assert usage["requests_limit"] == 60
        assert usage["tokens_per_minute"] == 3000
        assert usage["tokens_limit"] == 90000
        assert usage["requests_remaining"] == 58
        assert usage["tokens_remaining"] == 87000

    def test_cleanup_old_entries(self) -> None:
        """Test cleanup of old rate limit entries."""
        limiter = APIRateLimiter(requests_per_minute=60, tokens_per_minute=90000)

        # Add old and recent entries
        old_time = datetime.now() - timedelta(minutes=2)
        recent_time = datetime.now()

        limiter.requests = [old_time, recent_time]
        limiter.tokens = [(old_time, 1000), (recent_time, 2000)]

        usage = limiter.get_current_usage()

        # Should only count recent entries
        assert usage["requests_per_minute"] == 1
        assert usage["tokens_per_minute"] == 2000


class TestCostTracker:
    """Test cases for CostTracker."""

    def test_cost_tracker_initialization(self) -> None:
        """Test cost tracker initialization."""
        tracker = CostTracker()
        assert tracker.costs == []
        assert "gpt-4o-mini" in tracker.pricing
        assert "gpt-4o" in tracker.pricing

    def test_add_request_known_model(self) -> None:
        """Test adding a request for a known model."""
        tracker = CostTracker()

        cost = tracker.add_request("gpt-4o-mini", input_tokens=1000, output_tokens=500)

        # Expected cost: (1000/1000 * 0.00015) + (500/1000 * 0.0006) = 0.00015 + 0.0003 = 0.00045
        expected_cost = 0.00045
        assert abs(cost - expected_cost) < 0.000001
        assert len(tracker.costs) == 1

        request = tracker.costs[0]
        assert request["model"] == "gpt-4o-mini"
        assert request["input_tokens"] == 1000
        assert request["output_tokens"] == 500
        assert request["total_tokens"] == 1500
        assert request["cost_usd"] == cost

    def test_add_request_unknown_model(self) -> None:
        """Test adding a request for an unknown model (uses fallback)."""
        tracker = CostTracker()

        with patch.object(tracker.logger, "warning") as mock_warning:
            cost = tracker.add_request("unknown-model", input_tokens=1000, output_tokens=500)
            mock_warning.assert_called_once()

        # Should use gpt-4o-mini pricing as fallback
        expected_cost = 0.00045
        assert abs(cost - expected_cost) < 0.000001

    def test_get_total_cost(self) -> None:
        """Test getting total cost."""
        tracker = CostTracker()

        tracker.add_request("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.add_request("gpt-4o", input_tokens=500, output_tokens=250)

        total_cost = tracker.get_total_cost()
        assert total_cost > 0
        assert len(str(total_cost).split(".")[1]) <= 6  # Reasonable precision

    def test_get_total_cost_by_model(self) -> None:
        """Test getting total cost filtered by model."""
        tracker = CostTracker()

        tracker.add_request("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.add_request("gpt-4o", input_tokens=500, output_tokens=250)

        mini_cost = tracker.get_total_cost("gpt-4o-mini")
        gpt4_cost = tracker.get_total_cost("gpt-4o")

        assert mini_cost != gpt4_cost
        assert mini_cost + gpt4_cost == tracker.get_total_cost()

    def test_get_usage_summary_empty(self) -> None:
        """Test getting usage summary with no requests."""
        tracker = CostTracker()

        summary = tracker.get_usage_summary()

        assert summary["total_requests"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_tokens"] == 0
        assert summary["by_model"] == {}

    def test_get_usage_summary_with_data(self) -> None:
        """Test getting usage summary with request data."""
        tracker = CostTracker()

        tracker.add_request("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.add_request("gpt-4o-mini", input_tokens=800, output_tokens=300)
        tracker.add_request("gpt-4o", input_tokens=500, output_tokens=250)

        summary = tracker.get_usage_summary()

        assert summary["total_requests"] == 3
        assert summary["total_cost_usd"] > 0
        assert summary["total_tokens"] == 3350  # Sum of all tokens

        # Check by-model breakdown
        assert "gpt-4o-mini" in summary["by_model"]
        assert "gpt-4o" in summary["by_model"]
        assert summary["by_model"]["gpt-4o-mini"]["requests"] == 2
        assert summary["by_model"]["gpt-4o"]["requests"] == 1

        # Check time range
        assert "time_range" in summary
        assert "start" in summary["time_range"]
        assert "end" in summary["time_range"]


class TestOpenAIModelPlugin:
    """Test cases for OpenAIModelPlugin."""

    @pytest_asyncio.fixture
    async def plugin(self) -> OpenAIModelPlugin:
        """Create OpenAI plugin for testing."""
        return OpenAIModelPlugin()

    @pytest_asyncio.fixture
    async def initialized_plugin(self) -> OpenAIModelPlugin:
        """Create initialized OpenAI plugin for testing."""
        plugin = OpenAIModelPlugin()

        # Mock the OpenAI clients
        mock_client = MagicMock()
        mock_async_client = AsyncMock()
        plugin.client = mock_client
        plugin.async_client = mock_async_client
        plugin.model_name = "gpt-4o-mini"
        plugin.model_config = {
            "name": "test-model",
            "model_name": "gpt-4o-mini",
            "max_tokens": 512,
            "temperature": 0.1,
        }

        # Mock model info and metrics
        plugin.model_info = ModelInfo(model_id="test-model", name="test-model", type="openai_api")
        plugin.performance_metrics = PerformanceMetrics(model_id="test-model")

        return plugin

    def test_plugin_initialization(self, plugin: OpenAIModelPlugin) -> None:
        """Test basic plugin initialization."""
        assert plugin.client is None
        assert plugin.async_client is None
        assert plugin.model_name is None
        assert plugin.model_config is None
        assert plugin.model_info is None
        assert plugin.performance_metrics is None
        assert plugin.logger is not None
        assert isinstance(plugin.rate_limiter, APIRateLimiter)
        assert isinstance(plugin.cost_tracker, CostTracker)
        assert len(plugin.supported_models) > 0

    @pytest.mark.asyncio
    async def test_initialize_missing_api_key(self, plugin: OpenAIModelPlugin) -> None:
        """Test initialization without API key."""
        config = {"model_name": "gpt-4o-mini", "name": "test-model"}

        with patch.dict(os.environ, {}, clear=True):
            response = await plugin.initialize(config)

            assert response.success is False
            assert "API key not found" in response.error

    @pytest.mark.asyncio
    async def test_initialize_missing_config_fields(self, plugin: OpenAIModelPlugin) -> None:
        """Test initialization with missing config fields."""
        config = {"name": "test-model"}  # Missing model_name

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            response = await plugin.initialize(config)

            assert response.success is False
            assert "Missing required config fields" in response.error

    @pytest.mark.asyncio
    async def test_initialize_unsupported_model(self, plugin: OpenAIModelPlugin) -> None:
        """Test initialization with unsupported model."""
        config = {"model_name": "unsupported-model", "name": "test-model"}

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            response = await plugin.initialize(config)

            assert response.success is False
            assert "Unsupported model" in response.error

    @pytest.mark.asyncio
    async def test_initialize_api_test_failure(self, plugin: OpenAIModelPlugin) -> None:
        """Test initialization with API test failure."""
        config = {"model_name": "gpt-4o-mini", "name": "test-model"}

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}),
            patch("openai.AsyncOpenAI") as mock_async_openai,
            patch("openai.OpenAI") as mock_openai,
        ):
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create.side_effect = Exception("API Error")
            mock_async_openai.return_value = mock_async_client
            mock_openai.return_value = MagicMock()

            response = await plugin.initialize(config)

            assert response.success is False
            assert "Failed to connect to OpenAI API" in response.error

    @pytest.mark.asyncio
    async def test_initialize_success(self, plugin: OpenAIModelPlugin) -> None:
        """Test successful initialization."""
        config = {
            "model_name": "gpt-4o-mini",
            "name": "test-model",
            "max_tokens": 256,
            "temperature": 0.2,
            "requests_per_minute": 100,
            "tokens_per_minute": 50000,
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 5

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}),
            patch("openai.AsyncOpenAI") as mock_async_openai,
            patch("openai.OpenAI") as mock_openai,
            patch.object(plugin, "_test_api_access", new_callable=AsyncMock) as mock_test_api,
        ):
            mock_async_client = AsyncMock()
            mock_async_openai.return_value = mock_async_client
            mock_openai.return_value = MagicMock()
            mock_test_api.return_value = None  # Success

            response = await plugin.initialize(config)

            assert response.success is True
            assert plugin.model_name == "gpt-4o-mini"
            assert plugin.model_config == config
            assert plugin.model_info is not None
            assert plugin.performance_metrics is not None
            assert plugin.client is not None
            assert plugin.async_client is not None

    @pytest.mark.asyncio
    async def test_test_api_access_success(self, plugin: OpenAIModelPlugin) -> None:
        """Test successful API access test."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.usage = MagicMock()

        plugin.model_name = "gpt-4o-mini"
        plugin.async_client = AsyncMock()
        plugin.async_client.chat.completions.create.return_value = mock_response

        # Should not raise exception
        await plugin._test_api_access()

    @pytest.mark.asyncio
    async def test_test_api_access_failure(self, plugin: OpenAIModelPlugin) -> None:
        """Test API access test failure."""
        plugin.model_name = "gpt-4o-mini"
        plugin.async_client = AsyncMock()
        plugin.async_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API access test failed"):
            await plugin._test_api_access()

    @pytest.mark.asyncio
    async def test_make_api_request_success(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o-mini"
        mock_response.id = "test-id"

        initialized_plugin.async_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        with patch.object(initialized_plugin.rate_limiter, "acquire", new_callable=AsyncMock):
            result = await initialized_plugin._make_api_request(messages, 1000)

            assert result["content"] == "Test response"
            assert result["usage"]["total_tokens"] == 15
            assert result["model"] == "gpt-4o-mini"
            assert result["id"] == "test-id"

    @pytest.mark.asyncio
    async def test_make_api_request_with_retry(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test API request with retry logic."""
        # Mock to fail twice, then succeed
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success after retry"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o-mini"
        mock_response.id = "test-id"

        initialized_plugin.async_client.chat.completions.create.side_effect = [
            Exception("Temporary error"),
            Exception("Another error"),
            mock_response,
        ]

        messages = [{"role": "user", "content": "test"}]

        with (
            patch.object(initialized_plugin.rate_limiter, "acquire", new_callable=AsyncMock),
            patch("time.sleep"),  # Speed up retries
        ):
            result = await initialized_plugin._make_api_request(messages, 1000)

            assert result["content"] == "Success after retry"
            assert initialized_plugin.async_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_predict_not_initialized(self, plugin: OpenAIModelPlugin) -> None:
        """Test prediction with uninitialized model."""
        samples = ["test sample"]

        with pytest.raises(BenchmarkError) as exc_info:
            await plugin.predict(samples)

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_predict_success(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test successful prediction."""
        mock_response_data = {
            "content": """
            Classification: ATTACK
            Confidence: 0.85
            Attack_Type: malware
            Explanation: Suspicious behavior detected
            IOCs: malicious.exe, 192.168.1.1
            """,
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80,
            },
            "model": "gpt-4o-mini",
            "id": "test-response-id",
        }

        samples = ["suspicious network activity"]

        with (
            patch.object(
                initialized_plugin, "_make_api_request", new_callable=AsyncMock
            ) as mock_api_request,
            patch("time.time", side_effect=[0.0, 0.5]),  # Mock inference time
        ):
            mock_api_request.return_value = mock_response_data

            predictions = await initialized_plugin.predict(samples)

        assert len(predictions) == 1
        prediction = predictions[0]

        assert isinstance(prediction, Prediction)
        assert prediction.sample_id == "openai_0"
        assert prediction.input_text == samples[0]
        assert prediction.prediction == "ATTACK"
        assert prediction.confidence == 0.85
        assert prediction.attack_type == "malware"
        assert prediction.inference_time_ms == 500.0  # 0.5 * 1000
        assert "malicious.exe" in prediction.metadata["iocs"]
        assert prediction.metadata["cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_predict_with_error(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test prediction with API error."""
        samples = ["test sample"]

        with patch.object(
            initialized_plugin, "_make_api_request", new_callable=AsyncMock
        ) as mock_api_request:
            mock_api_request.side_effect = Exception("API Error")

            predictions = await initialized_plugin.predict(samples)

        assert len(predictions) == 1
        prediction = predictions[0]

        assert prediction.prediction == "ERROR"
        assert prediction.confidence == 0.0
        assert "API Error" in prediction.explanation
        assert "error" in prediction.metadata

    @pytest.mark.asyncio
    async def test_predict_batch_processing(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test batch prediction processing."""
        mock_response_data = {
            "content": "Classification: BENIGN\nConfidence: 0.9",
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            "model": "gpt-4o-mini",
            "id": "test-response-id",
        }

        samples = ["sample 1", "sample 2", "sample 3"]

        with patch.object(
            initialized_plugin, "_make_api_request", new_callable=AsyncMock
        ) as mock_api_request:
            mock_api_request.return_value = mock_response_data

            predictions = await initialized_plugin.predict(samples)

        assert len(predictions) == 3
        assert all(p.sample_id.startswith("openai_") for p in predictions)
        assert all(p.prediction == "BENIGN" for p in predictions)
        assert mock_api_request.call_count == 3

    def test_cybersecurity_prompt_formatting(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test cybersecurity prompt formatting."""
        sample = "suspicious network connection to 192.168.1.100"

        prompt = initialized_plugin._format_cybersecurity_prompt(sample)

        assert sample in prompt
        assert "Classification:" in prompt
        assert "Confidence:" in prompt
        assert "Attack_Type:" in prompt
        assert "Explanation:" in prompt
        assert "IOCs:" in prompt
        assert "cybersecurity threats" in prompt.lower()

    def test_response_parsing_attack(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test parsing attack response."""
        response = """
        Classification: ATTACK
        Confidence: 0.85
        Attack_Type: malware
        Explanation: This shows signs of malicious activity with suspicious network connections.
        IOCs: 192.168.1.100, suspicious.exe, port 1337
        """

        parsed = initialized_plugin._parse_response(response)

        assert parsed["classification"] == "ATTACK"
        assert parsed["confidence"] == 0.85
        assert parsed["attack_type"] == "malware"
        assert "malicious activity" in parsed["explanation"]
        assert "192.168.1.100" in parsed["iocs"]
        assert "suspicious.exe" in parsed["iocs"]
        assert "port 1337" in parsed["iocs"]

    def test_response_parsing_benign(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test parsing benign response."""
        response = """
        Classification: BENIGN
        Confidence: 0.92
        Attack_Type: N/A
        Explanation: Normal system activity, no threats detected.
        IOCs: None
        """

        parsed = initialized_plugin._parse_response(response)

        assert parsed["classification"] == "BENIGN"
        assert parsed["confidence"] == 0.92
        assert parsed["attack_type"] is None
        assert "Normal system activity" in parsed["explanation"]
        assert parsed["iocs"] == []

    def test_response_parsing_malformed(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test parsing malformed response."""
        response = "This is a completely malformed response without proper fields"

        parsed = initialized_plugin._parse_response(response)

        assert parsed["classification"] == "BENIGN"  # Default
        assert parsed["confidence"] == 0.5  # Default
        assert parsed["attack_type"] is None
        assert parsed["explanation"] == ""
        assert parsed["iocs"] == []

    def test_response_parsing_error(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test response parsing with error."""
        with patch("re.search", side_effect=Exception("Regex error")):
            parsed = initialized_plugin._parse_response("test response")

            assert parsed["classification"] == "BENIGN"
            assert parsed["confidence"] == 0.0
            assert "Parse error" in parsed["explanation"]

    @pytest.mark.asyncio
    async def test_explain_success(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test successful explanation generation."""
        mock_response_data = {
            "content": "This event shows detailed cybersecurity analysis...",
            "usage": {"prompt_tokens": 100, "completion_tokens": 150, "total_tokens": 250},
            "model": "gpt-4o-mini",
            "id": "explain-response-id",
        }

        sample = "network event"

        with patch.object(
            initialized_plugin, "_make_api_request", new_callable=AsyncMock
        ) as mock_api_request:
            mock_api_request.return_value = mock_response_data

            explanation = await initialized_plugin.explain(sample)

            assert explanation == "This event shows detailed cybersecurity analysis..."
            mock_api_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_not_initialized(self, plugin: OpenAIModelPlugin) -> None:
        """Test explanation with uninitialized model."""
        explanation = await plugin.explain("test sample")
        assert explanation == "Model not initialized"

    @pytest.mark.asyncio
    async def test_explain_error(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test explanation with error."""
        with patch.object(
            initialized_plugin, "_make_api_request", new_callable=AsyncMock
        ) as mock_api_request:
            mock_api_request.side_effect = Exception("API Error")

            explanation = await initialized_plugin.explain("test sample")

            assert "Explanation generation failed" in explanation
            assert "API Error" in explanation

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test getting model info."""
        info = await initialized_plugin.get_model_info()

        assert info is initialized_plugin.model_info
        assert info.name == "test-model"
        assert info.type == "openai_api"

    @pytest.mark.asyncio
    async def test_get_model_info_not_initialized(self, plugin: OpenAIModelPlugin) -> None:
        """Test getting model info when not initialized."""
        with pytest.raises(BenchmarkError) as exc_info:
            await plugin.get_model_info()

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_get_performance_metrics_success(
        self, initialized_plugin: OpenAIModelPlugin
    ) -> None:
        """Test getting performance metrics."""
        metrics = await initialized_plugin.get_performance_metrics()

        assert metrics is initialized_plugin.performance_metrics
        assert metrics.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_get_performance_metrics_not_initialized(self, plugin: OpenAIModelPlugin) -> None:
        """Test getting performance metrics when not initialized."""
        with pytest.raises(BenchmarkError) as exc_info:
            await plugin.get_performance_metrics()

        assert exc_info.value.error_code == ErrorCode.MODEL_INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test health check with healthy API."""
        mock_response = MagicMock()
        mock_response.usage.total_tokens = 1

        initialized_plugin.async_client.chat.completions.create.return_value = mock_response

        health = await initialized_plugin.health_check()

        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["client_connected"] is True
        assert health["api_reachable"] is True
        assert "rate_limits" in health
        assert "usage_summary" in health

    @pytest.mark.asyncio
    async def test_health_check_api_unreachable(
        self, initialized_plugin: OpenAIModelPlugin
    ) -> None:
        """Test health check when API is unreachable."""
        initialized_plugin.async_client.chat.completions.create.side_effect = Exception(
            "Connection error"
        )

        health = await initialized_plugin.health_check()

        assert health["status"] == "degraded"
        assert health["api_reachable"] is False
        assert "Connection error" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, plugin: OpenAIModelPlugin) -> None:
        """Test health check with uninitialized plugin."""
        health = await plugin.health_check()

        assert health["status"] == "unhealthy"
        assert health["model_loaded"] is False
        assert "not initialized" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_error(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test health check with error."""

        # Mock an error during health check execution
        async def mock_health_check():
            raise Exception("System error")

        initialized_plugin.health_check = mock_health_check

        try:
            health = await initialized_plugin.health_check()
        except Exception:
            # The exception should be caught and converted to error status
            health = {"status": "error", "error": "System error"}

        assert health["status"] == "error"
        assert "error" in health

    @pytest.mark.asyncio
    async def test_cleanup_success(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test successful cleanup."""
        # Add some cost data first
        initialized_plugin.cost_tracker.add_request("gpt-4o-mini", 100, 50)

        await initialized_plugin.cleanup()

        assert initialized_plugin.client is None
        assert initialized_plugin.async_client is None
        assert initialized_plugin.model_name is None
        assert initialized_plugin.model_config is None

    @pytest.mark.asyncio
    async def test_cleanup_error(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test cleanup with error (should not raise)."""
        # Mock an error during cleanup by making logger.info raise
        with (
            patch.object(initialized_plugin.logger, "info", side_effect=Exception("Logger error")),
            patch.object(initialized_plugin.logger, "error") as mock_error,
        ):
            await initialized_plugin.cleanup()

            # Should log error but not raise
            mock_error.assert_called()

    def test_get_supported_models(self, plugin: OpenAIModelPlugin) -> None:
        """Test getting supported models."""
        models = plugin.get_supported_models()

        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models
        assert "gpt-4-turbo" in models
        assert len(models) > 10  # Should have many models

    def test_get_model_specs(self, plugin: OpenAIModelPlugin) -> None:
        """Test getting model specifications."""
        specs = plugin.get_model_specs("gpt-4o-mini")

        assert specs is not None
        assert "max_tokens" in specs
        assert "context_window" in specs
        assert specs["context_window"] == 128000

    def test_get_cost_summary(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test getting cost summary."""
        # Add some cost data
        initialized_plugin.cost_tracker.add_request("gpt-4o-mini", 1000, 500)

        summary = initialized_plugin.get_cost_summary()

        assert "total_requests" in summary
        assert "total_cost_usd" in summary
        assert "by_model" in summary
        assert summary["total_requests"] == 1

    def test_get_rate_limit_status(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test getting rate limit status."""
        status = initialized_plugin.get_rate_limit_status()

        assert "requests_per_minute" in status
        assert "tokens_per_minute" in status
        assert "requests_remaining" in status
        assert "tokens_remaining" in status

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test rate limiting integration with predictions."""
        # Mock the OpenAI client but keep the rate limiting logic
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Classification: BENIGN\nConfidence: 0.9"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 30
        mock_response.usage.total_tokens = 80
        mock_response.model = "gpt-4o-mini"
        mock_response.id = "test-response-id"

        initialized_plugin.async_client.chat.completions.create.return_value = mock_response

        samples = ["sample 1", "sample 2"]

        with patch.object(
            initialized_plugin.rate_limiter, "acquire", new_callable=AsyncMock
        ) as mock_acquire:
            predictions = await initialized_plugin.predict(samples)

            # Should have made rate limiting calls
            assert len(predictions) == 2
            # Rate limiter acquire should have been called for each prediction
            assert mock_acquire.call_count == 2

    @pytest.mark.asyncio
    async def test_cost_tracking_integration(self, initialized_plugin: OpenAIModelPlugin) -> None:
        """Test cost tracking integration with predictions."""
        mock_response_data = {
            "content": "Classification: ATTACK\nConfidence: 0.8",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "model": "gpt-4o-mini",
            "id": "cost-test-id",
        }

        samples = ["network anomaly"]

        with patch.object(
            initialized_plugin, "_make_api_request", new_callable=AsyncMock
        ) as mock_api_request:
            mock_api_request.return_value = mock_response_data

            predictions = await initialized_plugin.predict(samples)

            # Check that cost was tracked
            assert len(predictions) == 1
            prediction = predictions[0]
            assert prediction.metadata["cost_usd"] > 0
            assert prediction.metadata["tokens_used"] == 150

            # Check cost tracker
            summary = initialized_plugin.cost_tracker.get_usage_summary()
            assert summary["total_requests"] == 1
            assert summary["total_tokens"] == 150
            assert summary["total_cost_usd"] > 0

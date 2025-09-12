"""
Anthropic Claude API plugin for the LLM Cybersecurity Benchmark system.

This module provides an Anthropic Claude API-based plugin with comprehensive rate limiting,
cost tracking, error handling, and retry mechanisms for cybersecurity analysis.
"""

import asyncio
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any

from anthropic import Anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from benchmark.core.base import ServiceResponse
from benchmark.core.exceptions import BenchmarkError, ErrorCode
from benchmark.core.logging import get_logger
from benchmark.interfaces.model_interfaces import (
    ModelInfo,
    ModelPlugin,
    PerformanceMetrics,
    Prediction,
)


class AnthropicRateLimiter:
    """Rate limiter for Anthropic API requests to prevent hitting API limits."""

    def __init__(self, requests_per_minute: int = 50, tokens_per_minute: int = 40000) -> None:
        """Initialize rate limiter with conservative defaults for Anthropic API.

        Args:
            requests_per_minute: Maximum requests per minute (Anthropic default is lower than OpenAI)
            tokens_per_minute: Maximum tokens per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.requests: list[datetime] = []
        self.tokens: list[tuple[datetime, int]] = []
        self.logger = get_logger("anthropic_rate_limiter")

    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Acquire permission to make an API request.

        Args:
            estimated_tokens: Estimated number of tokens for the request
        """
        now = datetime.now()

        # Clean old requests and tokens (older than 1 minute)
        cutoff_time = now - timedelta(minutes=1)
        self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
        self.tokens = [
            (token_time, tokens) for token_time, tokens in self.tokens if token_time > cutoff_time
        ]

        # Check request rate limit
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0]).total_seconds()
            if sleep_time > 0:
                self.logger.info(
                    f"Request rate limit reached, sleeping for {sleep_time:.2f} seconds"
                )
                await asyncio.sleep(sleep_time)

        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self.tokens)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            # Find the earliest token request that would allow this request
            tokens_needed = current_tokens + estimated_tokens - self.tokens_per_minute
            oldest_token_time = None
            cumulative_tokens = 0

            for token_time, tokens in sorted(self.tokens):
                cumulative_tokens += tokens
                if cumulative_tokens >= tokens_needed:
                    oldest_token_time = token_time
                    break

            if oldest_token_time:
                sleep_time = 60 - (now - oldest_token_time).total_seconds()
                if sleep_time > 0:
                    self.logger.info(
                        f"Token rate limit reached, sleeping for {sleep_time:.2f} seconds"
                    )
                    await asyncio.sleep(sleep_time)

        # Record this request
        self.requests.append(now)
        self.tokens.append((now, estimated_tokens))

    def get_current_usage(self) -> dict[str, Any]:
        """Get current rate limit usage statistics."""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)

        recent_requests = [req for req in self.requests if req > cutoff_time]
        recent_tokens = [(time, tokens) for time, tokens in self.tokens if time > cutoff_time]
        current_token_count = sum(tokens for _, tokens in recent_tokens)

        return {
            "requests_per_minute": len(recent_requests),
            "requests_limit": self.requests_per_minute,
            "tokens_per_minute": current_token_count,
            "tokens_limit": self.tokens_per_minute,
            "requests_remaining": max(0, self.requests_per_minute - len(recent_requests)),
            "tokens_remaining": max(0, self.tokens_per_minute - current_token_count),
        }


class AnthropicCostTracker:
    """Track API usage costs for different Anthropic Claude models."""

    def __init__(self) -> None:
        """Initialize cost tracker with current Anthropic pricing."""
        self.costs: list[dict[str, Any]] = []

        # Anthropic pricing as of 2024 (per 1K tokens)
        self.pricing = {
            # Claude 3.5 Sonnet
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
            # Claude 3 Opus
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            # Claude 3 Sonnet
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            # Claude 3 Haiku
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            # Claude 2.1 and 2.0 (legacy)
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024},
            # Claude Instant (legacy)
            "claude-instant-1.2": {"input": 0.0008, "output": 0.0024},
        }

        self.logger = get_logger("anthropic_cost_tracker")

    def add_request(
        self, model: str, input_tokens: int, output_tokens: int, request_id: str | None = None
    ) -> float:
        """Track cost of an API request.

        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_id: Optional request identifier

        Returns:
            Cost of this request in USD
        """
        model_pricing = self.pricing.get(model)
        if not model_pricing:
            # Use Claude 3 Haiku pricing as fallback (cheapest)
            model_pricing = self.pricing["claude-3-haiku-20240307"]
            self.logger.warning(f"Unknown model {model}, using Claude 3 Haiku pricing as fallback")

        cost = (input_tokens / 1000) * model_pricing["input"] + (
            output_tokens / 1000
        ) * model_pricing["output"]

        request_data = {
            "timestamp": datetime.now(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost,
            "request_id": request_id,
        }

        self.costs.append(request_data)
        self.logger.debug(f"Tracked cost: ${cost:.6f} for {input_tokens + output_tokens} tokens")

        return cost

    def get_total_cost(self, model: str | None = None) -> float:
        """Get total cost of all tracked requests.

        Args:
            model: Optional model name to filter by

        Returns:
            Total cost in USD
        """
        if model:
            return sum(req["cost_usd"] for req in self.costs if req["model"] == model)
        return sum(req["cost_usd"] for req in self.costs)

    def get_usage_summary(self) -> dict[str, Any]:
        """Get comprehensive usage summary."""
        if not self.costs:
            return {
                "total_requests": 0,
                "total_cost_usd": 0.0,
                "total_tokens": 0,
                "by_model": {},
            }

        total_cost = sum(req["cost_usd"] for req in self.costs)
        total_tokens = sum(req["total_tokens"] for req in self.costs)

        # Group by model
        by_model = {}
        for req in self.costs:
            model = req["model"]
            if model not in by_model:
                by_model[model] = {
                    "requests": 0,
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

            by_model[model]["requests"] += 1
            by_model[model]["cost_usd"] += req["cost_usd"]
            by_model[model]["input_tokens"] += req["input_tokens"]
            by_model[model]["output_tokens"] += req["output_tokens"]
            by_model[model]["total_tokens"] += req["total_tokens"]

        return {
            "total_requests": len(self.costs),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "by_model": by_model,
            "time_range": {
                "start": min(req["timestamp"] for req in self.costs),
                "end": max(req["timestamp"] for req in self.costs),
            },
        }


class AnthropicModelPlugin(ModelPlugin):
    """
    Anthropic Claude API plugin with comprehensive rate limiting, cost tracking, and error handling.

    Features:
    - Support for all Anthropic Claude models (3.5 Sonnet, 3 Opus, 3 Sonnet, 3 Haiku, etc.)
    - Conservative rate limiting appropriate for Anthropic API
    - Accurate cost tracking with up-to-date Anthropic pricing
    - Robust retry logic with exponential backoff
    - Cybersecurity-focused prompting optimized for Claude
    - Comprehensive error handling and recovery
    """

    def __init__(self) -> None:
        """Initialize the Anthropic plugin."""
        self.client = None
        self.model_name = None
        self.model_config = None
        self.model_info = None
        self.performance_metrics = None
        self.logger = get_logger("anthropic_plugin")

        # Initialize rate limiter and cost tracker with Anthropic-specific defaults
        self.rate_limiter = AnthropicRateLimiter(requests_per_minute=50, tokens_per_minute=40000)
        self.cost_tracker = AnthropicCostTracker()

        # Supported Claude models (updated list)
        self.supported_models = {
            # Claude 3.5 models (latest and most capable)
            "claude-3-5-sonnet-20241022": {"max_tokens": 8192, "context_window": 200000},
            "claude-3-5-sonnet-20240620": {"max_tokens": 8192, "context_window": 200000},
            # Claude 3 models
            "claude-3-opus-20240229": {"max_tokens": 4096, "context_window": 200000},
            "claude-3-sonnet-20240229": {"max_tokens": 4096, "context_window": 200000},
            "claude-3-haiku-20240307": {"max_tokens": 4096, "context_window": 200000},
            # Claude 2 models (legacy but still available)
            "claude-2.1": {"max_tokens": 4096, "context_window": 200000},
            "claude-2.0": {"max_tokens": 4096, "context_window": 100000},
            # Claude Instant (legacy)
            "claude-instant-1.2": {"max_tokens": 4096, "context_window": 100000},
        }

    async def initialize(self, config: dict[str, Any]) -> ServiceResponse:
        """
        Initialize Anthropic client and validate configuration.

        Args:
            config: Model configuration containing API key, model name, etc.

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            # Get API key from environment or config
            api_key = os.getenv("ANTHROPIC_API_KEY") or config.get("api_key")
            if not api_key:
                return ServiceResponse(
                    success=False,
                    message="Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
                    error="Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
                )

            # Validate required config fields
            required_fields = ["model_name", "name"]
            missing_fields = [f for f in required_fields if f not in config]
            if missing_fields:
                return ServiceResponse(
                    success=False,
                    message=f"Missing required config fields: {missing_fields}",
                    error=f"Missing required config fields: {missing_fields}",
                )

            model_name = config["model_name"]

            # Validate model is supported
            if model_name not in self.supported_models:
                return ServiceResponse(
                    success=False,
                    message=f"Unsupported model: {model_name}. Supported: {list(self.supported_models.keys())}",
                    error=f"Unsupported model: {model_name}. Supported: {list(self.supported_models.keys())}",
                )

            # Initialize Anthropic client
            base_url = config.get("base_url")  # For custom endpoints if needed

            self.client = Anthropic(
                api_key=api_key,
                base_url=base_url,
            )

            self.model_name = model_name
            self.model_config = config

            # Configure rate limiter based on model and tier
            requests_per_minute = config.get("requests_per_minute", 50)
            tokens_per_minute = config.get("tokens_per_minute", 40000)
            self.rate_limiter = AnthropicRateLimiter(requests_per_minute, tokens_per_minute)

            self.logger.info(f"Initializing Anthropic API with model: {model_name}")

            # Test API access
            try:
                await self._test_api_access()
            except Exception as e:
                return ServiceResponse(
                    success=False,
                    message=f"Failed to connect to Anthropic API: {str(e)}",
                    error=f"Failed to connect to Anthropic API: {str(e)}",
                )

            # Create model info
            model_specs = self.supported_models[model_name]
            self.model_info = ModelInfo(
                model_id=config["name"],
                name=config["name"],
                type="anthropic_api",
                version=config.get("version"),
                description=f"Anthropic {model_name} via API",
                capabilities=["text-generation", "cybersecurity-analysis", "conversation"],
                parameters={
                    "model_name": model_name,
                    "max_tokens": config.get("max_tokens", 1024),
                    "temperature": config.get("temperature", 0.1),
                    "context_window": model_specs["context_window"],
                    "requests_per_minute": requests_per_minute,
                    "tokens_per_minute": tokens_per_minute,
                },
                memory_usage_mb=0.0,  # API-based, no local memory usage
                status="loaded",
            )

            # Initialize performance metrics
            self.performance_metrics = PerformanceMetrics(
                model_id=config["name"], last_prediction_at=None
            )

            self.logger.info(
                f"Anthropic model initialized successfully: {config['name']} ({model_name})"
            )

            return ServiceResponse(
                success=True,
                message=f"Anthropic model loaded: {config['name']}",
                data={
                    "model_loaded": model_name,
                    "context_window": model_specs["context_window"],
                    "max_tokens": model_specs["max_tokens"],
                    "rate_limits": {
                        "requests_per_minute": requests_per_minute,
                        "tokens_per_minute": tokens_per_minute,
                    },
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic model: {e}")
            return ServiceResponse(
                success=False,
                message=f"Anthropic model initialization failed: {str(e)}",
                error=f"Anthropic model initialization failed: {str(e)}",
            )

    async def _test_api_access(self) -> None:
        """Test API access with a minimal request."""
        try:
            # Use a small request to test connectivity
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1,
                temperature=0,
                messages=[{"role": "user", "content": "Hi"}],
            )

            if message.content and message.usage:
                self.logger.info("API access test successful")
            else:
                raise Exception("Invalid API response format")

        except Exception as e:
            raise Exception(f"API access test failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _make_api_request(
        self, messages: list[dict[str, str]], estimated_tokens: int = 1000
    ) -> dict[str, Any]:
        """Make an API request with retry logic and rate limiting.

        Args:
            messages: Messages for Claude API
            estimated_tokens: Estimated token count for rate limiting

        Returns:
            API response data
        """
        # Apply rate limiting
        await self.rate_limiter.acquire(estimated_tokens)

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.model_config.get("max_tokens", 1024),
                temperature=self.model_config.get("temperature", 0.1),
                messages=messages,
            )

            return {
                "content": message.content[0].text,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                    "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
                },
                "model": message.model,
                "id": message.id,
            }

        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise

    async def predict(self, samples: list[str]) -> list[Prediction]:
        """
        Generate predictions using Anthropic Claude API with cybersecurity analysis.

        Args:
            samples: List of input samples to analyze

        Returns:
            List of Prediction objects with cybersecurity analysis
        """
        if self.client is None or self.model_name is None:
            raise BenchmarkError("Model not initialized", ErrorCode.MODEL_INITIALIZATION_FAILED)

        predictions = []

        for i, sample in enumerate(samples):
            try:
                start_time = time.time()

                # Format cybersecurity prompt
                prompt = self._format_cybersecurity_prompt(sample)

                # Estimate tokens for rate limiting (rough estimate)
                estimated_tokens = len(prompt.split()) * 1.3 + self.model_config.get(
                    "max_tokens", 1024
                )

                # Make API request with retry logic
                messages = [{"role": "user", "content": prompt}]

                response_data = await self._make_api_request(messages, int(estimated_tokens))

                inference_time = time.time() - start_time

                # Parse structured response
                model_response = response_data["content"]
                parsed_response = self._parse_response(model_response)

                # Track costs
                usage = response_data["usage"]
                cost = self.cost_tracker.add_request(
                    model=self.model_name,
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    request_id=response_data["id"],
                )

                # Create prediction object
                prediction = Prediction(
                    sample_id=f"anthropic_{i}",
                    input_text=sample,
                    prediction=parsed_response["classification"],
                    confidence=parsed_response["confidence"],
                    attack_type=parsed_response.get("attack_type"),
                    explanation=parsed_response.get("explanation"),
                    inference_time_ms=inference_time * 1000,
                    metadata={
                        "model_response": model_response,
                        "iocs": parsed_response.get("iocs", []),
                        "model_name": self.model_name,
                        "api_response_id": response_data["id"],
                        "tokens_used": usage["total_tokens"],
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "cost_usd": cost,
                        "prompt_length": len(prompt),
                    },
                    model_version=self.model_config.get("version"),
                )

                predictions.append(prediction)

                # Update performance metrics
                if self.performance_metrics:
                    self.performance_metrics.total_predictions += 1
                    self.performance_metrics.total_inference_time_ms += inference_time * 1000
                    self.performance_metrics.average_inference_time_ms = (
                        self.performance_metrics.total_inference_time_ms
                        / self.performance_metrics.total_predictions
                    )
                    self.performance_metrics.last_prediction_at = datetime.now()

            except Exception as e:
                self.logger.error(f"Prediction failed for sample {i}: {e}")

                # Create error prediction
                error_prediction = Prediction(
                    sample_id=f"anthropic_{i}",
                    input_text=sample,
                    prediction="ERROR",
                    confidence=0.0,
                    attack_type=None,
                    explanation=f"Prediction failed: {str(e)}",
                    inference_time_ms=0.0,
                    metadata={"error": str(e), "model_name": self.model_name},
                    model_version=self.model_config.get("version"),
                )

                predictions.append(error_prediction)

                # Update error count
                if self.performance_metrics:
                    self.performance_metrics.error_count += 1

        # Update success rate
        if self.performance_metrics and self.performance_metrics.total_predictions > 0:
            self.performance_metrics.success_rate = (
                self.performance_metrics.total_predictions - self.performance_metrics.error_count
            ) / self.performance_metrics.total_predictions

        return predictions

    def _format_cybersecurity_prompt(self, sample: str) -> str:
        """
        Format sample for cybersecurity analysis optimized for Claude.

        Args:
            sample: Input sample to analyze

        Returns:
            Formatted prompt for cybersecurity analysis
        """
        return f"""As a cybersecurity expert, analyze the following network log entry or security event for potential threats. Provide a structured analysis with clear reasoning.

Event to analyze:
{sample}

Please provide your analysis in the following structured format:

Classification: [ATTACK or BENIGN]
Confidence: [0.0 to 1.0]
Attack_Type: [malware, intrusion, dos, phishing, data_exfiltration, lateral_movement, or N/A if benign]
Explanation: [Detailed explanation of your reasoning and what specific indicators led to this classification]
IOCs: [List any indicators of compromise found, comma-separated, or None if no IOCs detected]

Analysis:"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """
        Parse structured response from Claude.

        Args:
            response: Raw model response text

        Returns:
            Dictionary with parsed fields
        """
        try:
            parsed = {
                "classification": "BENIGN",  # Default
                "confidence": 0.5,
                "attack_type": None,
                "explanation": "",
                "iocs": [],
            }

            # Extract classification
            classification_match = re.search(
                r"Classification:\s*(ATTACK|BENIGN)", response, re.IGNORECASE
            )
            if classification_match:
                parsed["classification"] = classification_match.group(1).upper()

            # Extract confidence
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                parsed["confidence"] = max(0.0, min(1.0, confidence))

            # Extract attack type
            attack_type_match = re.search(
                r"Attack_Type:\s*(malware|intrusion|dos|phishing|data_exfiltration|lateral_movement|N/A)",
                response,
                re.IGNORECASE,
            )
            if attack_type_match and attack_type_match.group(1).lower() != "n/a":
                parsed["attack_type"] = attack_type_match.group(1).lower()

            # Extract explanation
            explanation_match = re.search(
                r"Explanation:\s*(.+?)(?=\n\s*IOCs:|$)", response, re.IGNORECASE | re.DOTALL
            )
            if explanation_match:
                parsed["explanation"] = explanation_match.group(1).strip()

            # Extract IOCs
            iocs_match = re.search(r"IOCs:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
            if iocs_match:
                iocs_text = iocs_match.group(1).strip()
                if iocs_text and iocs_text.lower() not in ["none", "n/a", "null"]:
                    # Split by comma and clean up
                    iocs = [ioc.strip() for ioc in iocs_text.split(",") if ioc.strip()]
                    parsed["iocs"] = iocs

            return parsed

        except Exception as e:
            self.logger.warning(f"Failed to parse response: {e}")
            return {
                "classification": "BENIGN",
                "confidence": 0.0,
                "attack_type": None,
                "explanation": f"Parse error: {str(e)}",
                "iocs": [],
            }

    async def explain(self, sample: str) -> str:
        """
        Generate detailed explanation for a prediction.

        Args:
            sample: Input sample

        Returns:
            Detailed explanation string
        """
        try:
            if self.client is None or self.model_name is None:
                return "Model not initialized"

            # Generate detailed explanation prompt
            explanation_prompt = f"""As a senior cybersecurity analyst, provide a comprehensive and detailed analysis of this security event:

Event: {sample}

Please provide an in-depth explanation covering:

1. **Initial Assessment**: What you observe in this event and your immediate assessment
2. **Technical Indicators**: Specific technical details that inform your analysis
3. **Threat Classification**: Whether this represents an attack or benign activity and why
4. **Risk Assessment**: Potential impact and severity if this is malicious
5. **Context Analysis**: How this fits into common attack patterns or normal operations
6. **Recommended Actions**: Immediate response steps and mitigation strategies

Detailed Cybersecurity Analysis:"""

            messages = [{"role": "user", "content": explanation_prompt}]

            # Estimate tokens and make request
            estimated_tokens = len(explanation_prompt.split()) * 1.3 + 1000
            response_data = await self._make_api_request(messages, int(estimated_tokens))

            # Track cost for explanation
            usage = response_data["usage"]
            self.cost_tracker.add_request(
                model=self.model_name,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                request_id=response_data["id"],
            )

            return response_data["content"].strip()

        except Exception as e:
            return f"Explanation generation failed: {str(e)}"

    async def get_model_info(self) -> ModelInfo:
        """
        Get information about the model.

        Returns:
            ModelInfo object
        """
        if self.model_info is None:
            raise BenchmarkError("Model not initialized", ErrorCode.MODEL_INITIALIZATION_FAILED)
        return self.model_info

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics for the model.

        Returns:
            PerformanceMetrics object
        """
        if self.performance_metrics is None:
            raise BenchmarkError("Model not initialized", ErrorCode.MODEL_INITIALIZATION_FAILED)
        return self.performance_metrics

    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the Anthropic API connection.

        Returns:
            Dictionary with health status information
        """
        try:
            status = {
                "status": "healthy",
                "model_loaded": self.model_name is not None,
                "client_connected": self.client is not None,
                "model_name": self.model_name,
            }

            # Test API connectivity
            if self.client and self.model_name:
                try:
                    # Make a minimal test request
                    test_message = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=1,
                        temperature=0,
                        messages=[{"role": "user", "content": "test"}],
                    )

                    status["api_reachable"] = True
                    status["test_tokens_used"] = (
                        test_message.usage.input_tokens + test_message.usage.output_tokens
                    )

                    # Add rate limit status
                    rate_status = self.rate_limiter.get_current_usage()
                    status["rate_limits"] = rate_status

                    # Add cost information
                    cost_summary = self.cost_tracker.get_usage_summary()
                    status["usage_summary"] = cost_summary

                except Exception as e:
                    status["status"] = "degraded"
                    status["api_reachable"] = False
                    status["error"] = f"API test failed: {str(e)}"
            else:
                status["status"] = "unhealthy"
                status["error"] = "Client or model not initialized"

            return status

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            self.logger.info("Cleaning up Anthropic API resources")

            # Log final usage summary
            if self.cost_tracker.costs:
                summary = self.cost_tracker.get_usage_summary()
                self.logger.info(
                    f"Session summary: {summary['total_requests']} requests, "
                    f"${summary['total_cost_usd']:.4f} total cost"
                )

            # Clear client references
            self.client = None
            self.model_name = None
            self.model_config = None

            self.logger.info("Anthropic API cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_supported_models(self) -> list[str]:
        """Get list of supported Anthropic models."""
        return list(self.supported_models.keys())

    def get_model_specs(self, model_name: str) -> dict[str, Any] | None:
        """Get specifications for a specific model."""
        return self.supported_models.get(model_name)

    def get_cost_summary(self) -> dict[str, Any]:
        """Get comprehensive cost and usage summary."""
        return self.cost_tracker.get_usage_summary()

    def get_rate_limit_status(self) -> dict[str, Any]:
        """Get current rate limit usage status."""
        return self.rate_limiter.get_current_usage()

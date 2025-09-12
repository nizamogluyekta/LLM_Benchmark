"""
Ollama plugin for local model serving.

This module provides an Ollama-based plugin for running local models
with cybersecurity-focused prompting and analysis via Ollama server.
"""

import re
import time
from datetime import datetime
from typing import Any

from benchmark.core.base import ServiceResponse
from benchmark.core.exceptions import BenchmarkError, ErrorCode
from benchmark.core.logging import get_logger
from benchmark.interfaces.model_interfaces import (
    ModelInfo,
    ModelPlugin,
    PerformanceMetrics,
    Prediction,
)


class OllamaModelPlugin(ModelPlugin):
    """
    Ollama plugin for local model serving with cybersecurity analysis.

    Features:
    - Connect to local Ollama server
    - Support popular cybersecurity and general models
    - Automatic model pulling if not available locally
    - Chat-based inference with streaming support
    - Cybersecurity-focused prompting
    - Robust error handling for server connectivity
    """

    def __init__(self) -> None:
        """Initialize the Ollama plugin."""
        self.client = None
        self.model_name = None
        self.model_config = None
        self.model_info = None
        self.performance_metrics = None
        self.logger = get_logger("ollama_plugin")

        # Ollama client (imported lazily)
        self.ollama = None

        # Default Ollama server settings
        self.server_host = "localhost"
        self.server_port = 11434
        self.server_url = f"http://{self.server_host}:{self.server_port}"

        # Popular models for cybersecurity analysis
        self.recommended_models = {
            "general": ["llama2:7b", "llama2:13b", "mistral:7b", "codellama:7b"],
            "cybersecurity": ["llama2:7b", "codellama:7b", "mistral:7b"],
            "code_analysis": ["codellama:7b", "codellama:13b", "phind-codellama:34b"],
        }

    def _import_ollama(self) -> bool:
        """Lazy import of Ollama library."""
        try:
            if self.ollama is None:
                import importlib

                self.ollama = importlib.import_module("ollama")
            return True
        except ImportError as e:
            self.logger.error(f"Ollama library not available: {e}")
            return False

    async def initialize(self, config: dict[str, Any]) -> ServiceResponse:
        """
        Initialize Ollama client and ensure model is available.

        Args:
            config: Model configuration containing model name, server settings, etc.

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            # Import Ollama library
            if not self._import_ollama():
                return ServiceResponse(
                    success=False,
                    message="Ollama library not available. Install with: pip install ollama",
                    error="Ollama library not available. Install with: pip install ollama",
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

            # Set up server connection
            self.server_host = config.get("host", self.server_host)
            self.server_port = config.get("port", self.server_port)
            self.server_url = f"http://{self.server_host}:{self.server_port}"

            # Initialize client
            self.client = self.ollama.Client(host=self.server_url)
            self.model_name = config["model_name"]  # e.g., "llama2:7b", "codellama:13b"

            self.logger.info(f"Connecting to Ollama server at {self.server_url}")

            # Test server connectivity
            try:
                await self._test_server_connection()
            except Exception as e:
                return ServiceResponse(
                    success=False,
                    message=f"Cannot connect to Ollama server at {self.server_url}: {str(e)}",
                    error=f"Cannot connect to Ollama server at {self.server_url}: {str(e)}",
                )

            # Ensure model is available (pull if necessary)
            try:
                await self._ensure_model_available()
            except Exception as e:
                return ServiceResponse(
                    success=False,
                    message=f"Failed to ensure model availability: {str(e)}",
                    error=f"Failed to ensure model availability: {str(e)}",
                )

            self.model_config = config

            # Create model info
            self.model_info = ModelInfo(
                model_id=config["name"],
                name=config["name"],
                type="ollama",
                version=config.get("version"),
                description=f"Ollama model {self.model_name} via {self.server_url}",
                capabilities=["text-generation", "cybersecurity-analysis", "chat"],
                parameters={
                    "model_name": self.model_name,
                    "max_tokens": config.get("max_tokens", 512),
                    "temperature": config.get("temperature", 0.1),
                    "server_url": self.server_url,
                },
                memory_usage_mb=self._estimate_memory_usage(),
                status="loaded",
            )

            # Initialize performance metrics
            self.performance_metrics = PerformanceMetrics(
                model_id=config["name"], last_prediction_at=None
            )

            self.logger.info(
                f"Ollama model loaded successfully: {config['name']} ({self.model_name})"
            )

            return ServiceResponse(
                success=True,
                message=f"Ollama model loaded: {config['name']}",
                data={
                    "model_loaded": self.model_name,
                    "server_url": self.server_url,
                    "estimated_memory_mb": self.model_info.memory_usage_mb,
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama model: {e}")
            return ServiceResponse(
                success=False,
                message=f"Ollama model initialization failed: {str(e)}",
                error=f"Ollama model initialization failed: {str(e)}",
            )

    async def _test_server_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            # Try to list models to test connectivity
            models = self.client.list()
            self.logger.info(
                f"Successfully connected to Ollama server. Found {len(models.get('models', []))} models."
            )
        except Exception as e:
            raise Exception(f"Server connection test failed: {str(e)}") from e

    async def _ensure_model_available(self) -> None:
        """Check if model is available, pull if necessary."""
        try:
            self.logger.info(f"Checking availability of model: {self.model_name}")

            # List available models
            models_response = self.client.list()
            available_models = [model["name"] for model in models_response.get("models", [])]

            self.logger.info(f"Available models: {available_models}")

            if self.model_name not in available_models:
                self.logger.info(
                    f"Model {self.model_name} not found locally. Pulling from registry..."
                )

                # Pull the model
                self.client.pull(self.model_name)
                self.logger.info(f"Successfully pulled model: {self.model_name}")
            else:
                self.logger.info(f"Model {self.model_name} is already available locally")

        except Exception as e:
            raise Exception(f"Failed to ensure model availability: {str(e)}") from e

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for the model."""
        # Rough estimation based on model name
        try:
            if self.model_name:
                # Extract parameter count from model name if available
                if "7b" in self.model_name.lower():
                    return 4000.0  # ~4GB for 7B models
                elif "13b" in self.model_name.lower():
                    return 8000.0  # ~8GB for 13B models
                elif "34b" in self.model_name.lower():
                    return 20000.0  # ~20GB for 34B models
                elif "70b" in self.model_name.lower():
                    return 40000.0  # ~40GB for 70B models
                else:
                    return 2000.0  # Default for smaller models
            return 2000.0  # Default estimation
        except Exception:
            return 2000.0  # Fallback

    async def predict(self, samples: list[str]) -> list[Prediction]:
        """
        Generate predictions using Ollama with cybersecurity prompting.

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

                # Generate response with Ollama
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.model_config.get("temperature", 0.1),
                        "num_predict": self.model_config.get("max_tokens", 512),
                    },
                )

                inference_time = time.time() - start_time

                # Parse structured response
                model_response = response["message"]["content"]
                parsed_response = self._parse_response(model_response)

                # Create prediction object
                prediction = Prediction(
                    sample_id=f"ollama_{i}",
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
                        "server_url": self.server_url,
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
                    sample_id=f"ollama_{i}",
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
        if self.performance_metrics:
            self.performance_metrics.success_rate = (
                self.performance_metrics.total_predictions - self.performance_metrics.error_count
            ) / max(self.performance_metrics.total_predictions, 1)

        return predictions

    def _format_cybersecurity_prompt(self, sample: str) -> str:
        """
        Format sample for cybersecurity analysis.

        Args:
            sample: Input sample to analyze

        Returns:
            Formatted prompt for cybersecurity analysis
        """
        return f"""You are a cybersecurity expert analyzing network logs and security events. Provide structured analysis with clear classifications.

Analyze the following network log entry or security event for potential threats:

Event: {sample}

Please provide your analysis in the following format:
Classification: [ATTACK or BENIGN]
Confidence: [0.0 to 1.0]
Attack_Type: [malware, intrusion, dos, phishing, data_exfiltration, lateral_movement, or N/A if benign]
Explanation: [Brief explanation of your reasoning]
IOCs: [List any indicators of compromise found, comma-separated]

Analysis:"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """
        Parse structured response from model.

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
            confidence_match = re.search(r"Confidence:\s*(\d*\.?\d+)", response, re.IGNORECASE)
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
                if iocs_text and iocs_text.lower() != "none" and iocs_text.lower() != "n/a":
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
        Generate explanation for a prediction.

        Args:
            sample: Input sample

        Returns:
            Detailed explanation string
        """
        try:
            if self.client is None or self.model_name is None:
                return "Model not initialized"

            # Generate detailed explanation prompt
            explanation_prompt = f"""You are a cybersecurity expert. Provide a detailed explanation of your analysis.

Provide a detailed cybersecurity analysis explanation for this event:

Event: {sample}

Focus on:
1. What indicators you observed
2. Why you classified it as attack/benign
3. Potential impact if it's malicious
4. Recommended response actions

Detailed Analysis:"""

            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": explanation_prompt}],
                options={
                    "temperature": self.model_config.get("temperature", 0.1),
                    "num_predict": self.model_config.get("max_tokens", 512),
                },
            )

            return response["message"]["content"].strip()

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
        Check the health status of the model and server.

        Returns:
            Dictionary with health status information
        """
        try:
            status = {
                "status": "healthy",
                "model_loaded": self.model_name is not None,
                "client_connected": self.client is not None,
                "server_url": self.server_url,
                "model_name": self.model_name,
            }

            # Test server connectivity
            if self.client:
                try:
                    models_response = self.client.list()
                    available_models = [
                        model["name"] for model in models_response.get("models", [])
                    ]
                    status["server_reachable"] = True
                    status["available_models_count"] = len(available_models)
                    status["model_available"] = self.model_name in available_models

                    if not status["model_available"]:
                        status["status"] = "degraded"
                        status["error"] = f"Model {self.model_name} not available on server"

                except Exception as e:
                    status["status"] = "unhealthy"
                    status["server_reachable"] = False
                    status["error"] = f"Cannot reach Ollama server: {str(e)}"
            else:
                status["status"] = "unhealthy"
                status["error"] = "Client not initialized"

            return status

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            self.logger.info("Cleaning up Ollama model resources")

            # Clear model references (Ollama server handles the actual model)
            self.client = None
            self.model_name = None
            self.model_config = None

            self.logger.info("Ollama model cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_recommended_models(self, category: str = "general") -> list[str]:
        """Get list of recommended models for different categories."""
        return self.recommended_models.get(category, self.recommended_models["general"])

    def get_server_info(self) -> dict[str, Any]:
        """Get information about the Ollama server."""
        return {"host": self.server_host, "port": self.server_port, "url": self.server_url}

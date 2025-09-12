"""
MLX plugin for local Apple Silicon model inference.

This module provides an MLX-optimized plugin for running quantized models
on Apple Silicon with cybersecurity-focused prompting and analysis.
"""

import platform
import re
import time
from pathlib import Path
from typing import Any

import psutil

from benchmark.core.base import ServiceResponse
from benchmark.core.exceptions import BenchmarkError, ErrorCode
from benchmark.core.logging import get_logger
from benchmark.interfaces.model_interfaces import (
    ModelInfo,
    ModelPlugin,
    PerformanceMetrics,
    Prediction,
)


class MLXModelPlugin(ModelPlugin):
    """
    MLX plugin optimized for Apple Silicon M4 Pro with cybersecurity analysis.

    Features:
    - Quantized model support (4-bit, 8-bit)
    - Apple Silicon unified memory optimization
    - Cybersecurity-focused prompting
    - Efficient batch processing
    - Model caching and lazy loading
    """

    def __init__(self) -> None:
        """Initialize the MLX plugin."""
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.model_info = None
        self.performance_metrics = None
        self.logger = get_logger("mlx_plugin")

        # MLX modules (imported lazily)
        self.mlx_lm = None
        self.mlx = None

        # Model cache
        self._model_cache = {}
        self._cache_dir = Path.home() / ".cache" / "benchmark_mlx"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # System info
        self._system_info = self._detect_system_capabilities()

    def _detect_system_capabilities(self) -> dict[str, Any]:
        """Detect Apple Silicon and system capabilities."""
        try:
            system_info = {
                "platform": platform.system(),
                "machine": platform.machine(),
                "is_apple_silicon": False,
                "memory_gb": 0.0,
                "cpu_cores": 0,
            }

            # Detect Apple Silicon
            if (
                system_info["platform"] == "Darwin"
                and "arm64" in str(system_info["machine"]).lower()
            ):
                system_info["is_apple_silicon"] = True

            # Get memory info
            memory = psutil.virtual_memory()
            system_info["memory_gb"] = memory.total / (1024**3)
            system_info["cpu_cores"] = psutil.cpu_count(logical=True) or 1

            return system_info

        except Exception as e:
            self.logger.warning(f"Failed to detect system capabilities: {e}")
            return {
                "platform": "unknown",
                "machine": "unknown",
                "is_apple_silicon": False,
                "memory_gb": 8.0,
                "cpu_cores": 4,
            }

    def _import_mlx(self) -> bool:
        """Lazy import of MLX libraries."""
        try:
            if self.mlx_lm is None:
                import importlib

                self.mlx_lm = importlib.import_module("mlx_lm")

            if self.mlx is None:
                import importlib

                self.mlx = importlib.import_module("mlx.core")

            return True
        except ImportError as e:
            self.logger.error(f"MLX libraries not available: {e}")
            return False

    async def initialize(self, config: dict[str, Any]) -> ServiceResponse:
        """
        Load MLX model with optimizations for Apple Silicon.

        Args:
            config: Model configuration containing path, quantization, etc.

        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            # Validate Apple Silicon requirement
            if not self._system_info["is_apple_silicon"]:
                return ServiceResponse(
                    success=False,
                    message="MLX plugin requires Apple Silicon hardware",
                    error="MLX plugin requires Apple Silicon hardware",
                )

            # Import MLX libraries
            if not self._import_mlx():
                return ServiceResponse(
                    success=False,
                    message="MLX libraries not available. Install with: pip install mlx mlx-lm",
                    error="MLX libraries not available. Install with: pip install mlx mlx-lm",
                )

            # Validate required config fields
            required_fields = ["model_path", "name"]
            missing_fields = [f for f in required_fields if f not in config]
            if missing_fields:
                return ServiceResponse(
                    success=False,
                    message=f"Missing required config fields: {missing_fields}",
                    error=f"Missing required config fields: {missing_fields}",
                )

            model_path = Path(config["model_path"])
            if not model_path.exists():
                return ServiceResponse(
                    success=False,
                    message=f"Model path does not exist: {model_path}",
                    error=f"Model path does not exist: {model_path}",
                )

            self.logger.info(f"Loading MLX model from {model_path}")

            # Get quantization setting
            quantization = config.get("quantization")

            # Check cache first
            cache_key = f"{model_path}_{quantization or 'none'}"
            if cache_key in self._model_cache:
                self.logger.info("Loading model from cache")
                self.model, self.tokenizer = self._model_cache[cache_key]
            else:
                # Load model with quantization support
                load_kwargs = {}
                if quantization in ["4bit", "8bit"]:
                    # MLX-LM quantization parameters
                    load_kwargs["quantize"] = True
                    if quantization == "4bit":
                        load_kwargs["bits"] = 4
                    elif quantization == "8bit":
                        load_kwargs["bits"] = 8

                # Load model and tokenizer
                self.model, self.tokenizer = self.mlx_lm.load(str(model_path), **load_kwargs)

                # Cache the loaded model
                self._model_cache[cache_key] = (self.model, self.tokenizer)

            self.model_config = config

            # Create model info
            self.model_info = ModelInfo(
                model_id=config["name"],
                name=config["name"],
                type="mlx",
                version=config.get("version"),
                description=f"MLX model from {model_path}",
                capabilities=["text-generation", "cybersecurity-analysis"],
                parameters={
                    "quantization": quantization,
                    "max_tokens": config.get("max_tokens", 512),
                    "temperature": config.get("temperature", 0.1),
                },
                memory_usage_mb=self._estimate_memory_usage(),
                status="loaded",
            )

            # Initialize performance metrics
            self.performance_metrics = PerformanceMetrics(
                model_id=config["name"], last_prediction_at=None
            )

            self.logger.info(f"MLX model loaded successfully: {config['name']}")

            return ServiceResponse(
                success=True,
                message=f"MLX model loaded: {config['name']}",
                data={
                    "model_loaded": str(model_path),
                    "quantization": quantization,
                    "estimated_memory_mb": self.model_info.memory_usage_mb,
                    "apple_silicon": self._system_info["is_apple_silicon"],
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize MLX model: {e}")
            return ServiceResponse(
                success=False,
                message=f"MLX model initialization failed: {str(e)}",
                error=f"MLX model initialization failed: {str(e)}",
                error_code=ErrorCode.MODEL_INITIALIZATION_FAILED,
            )

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the loaded model."""
        try:
            if self.model is not None:
                # Try to get model parameters and estimate memory
                # This is a rough estimation
                base_memory = 500.0  # Base overhead in MB

                # Add estimation based on model size if possible
                if hasattr(self.model, "parameters"):
                    # Rough estimation: 4 bytes per parameter for FP32
                    param_count = sum(p.size for p in self.model.parameters() if hasattr(p, "size"))
                    param_memory = (param_count * 4) / (1024 * 1024)  # Convert to MB
                    return base_memory + param_memory

            return 2048.0  # Default estimation for MLX model

        except Exception:
            return 2048.0  # Fallback estimation

    async def predict(self, samples: list[str]) -> list[Prediction]:
        """
        Generate predictions using MLX with cybersecurity prompting.

        Args:
            samples: List of input samples to analyze

        Returns:
            List of Prediction objects with cybersecurity analysis
        """
        if self.model is None or self.tokenizer is None:
            raise BenchmarkError("Model not initialized", ErrorCode.MODEL_INITIALIZATION_FAILED)

        predictions = []

        for i, sample in enumerate(samples):
            try:
                start_time = time.time()

                # Format cybersecurity prompt
                prompt = self._format_cybersecurity_prompt(sample)

                # Generate response with MLX
                response = self.mlx_lm.generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.model_config.get("max_tokens", 512),
                    temperature=self.model_config.get("temperature", 0.1),
                    verbose=False,
                )

                inference_time = time.time() - start_time

                # Parse structured response
                parsed_response = self._parse_response(response)

                # Create prediction object
                prediction = Prediction(
                    sample_id=f"mlx_{i}",
                    input_text=sample,
                    prediction=parsed_response["classification"],
                    confidence=parsed_response["confidence"],
                    attack_type=parsed_response.get("attack_type"),
                    explanation=parsed_response.get("explanation"),
                    inference_time_ms=inference_time * 1000,
                    metadata={
                        "model_response": response,
                        "iocs": parsed_response.get("iocs", []),
                        "quantization": self.model_config.get("quantization"),
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

            except Exception as e:
                self.logger.error(f"Prediction failed for sample {i}: {e}")

                # Create error prediction
                error_prediction = Prediction(
                    sample_id=f"mlx_{i}",
                    input_text=sample,
                    prediction="ERROR",
                    confidence=0.0,
                    attack_type=None,
                    explanation=f"Prediction failed: {str(e)}",
                    inference_time_ms=0.0,
                    metadata={"error": str(e)},
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
        return f"""<|system|>
You are a cybersecurity expert analyzing network logs and security events. Provide structured analysis with clear classifications.

<|user|>
Analyze the following network log entry or security event for potential threats:

Event: {sample}

Please provide your analysis in the following format:
Classification: [ATTACK or BENIGN]
Confidence: [0.0 to 1.0]
Attack_Type: [malware, intrusion, dos, phishing, data_exfiltration, lateral_movement, or N/A if benign]
Explanation: [Brief explanation of your reasoning]
IOCs: [List any indicators of compromise found, comma-separated]

<|assistant|>
Analysis:
"""

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
                if iocs_text and iocs_text.lower() != "none":
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
            if self.model is None or self.tokenizer is None:
                return "Model not initialized"

            # Generate detailed explanation prompt
            explanation_prompt = f"""<|system|>
You are a cybersecurity expert. Provide a detailed explanation of your analysis.

<|user|>
Provide a detailed cybersecurity analysis explanation for this event:

Event: {sample}

Focus on:
1. What indicators you observed
2. Why you classified it as attack/benign
3. Potential impact if it's malicious
4. Recommended response actions

<|assistant|>
Detailed Analysis:
"""

            response = self.mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=explanation_prompt,
                max_tokens=self.model_config.get("max_tokens", 512),
                temperature=self.model_config.get("temperature", 0.1),
                verbose=False,
            )

            return response.strip()

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
        Check the health status of the model.

        Returns:
            Dictionary with health status information
        """
        try:
            status = {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None,
                "apple_silicon": self._system_info["is_apple_silicon"],
                "memory_gb": self._system_info["memory_gb"],
                "cached_models": len(self._model_cache),
            }

            if self.model is None:
                status["status"] = "unhealthy"
                status["error"] = "Model not loaded"

            if not self._system_info["is_apple_silicon"]:
                status["status"] = "incompatible"
                status["error"] = "Requires Apple Silicon"

            return status

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            self.logger.info("Cleaning up MLX model resources")

            # Clear model references
            self.model = None
            self.tokenizer = None
            self.model_config = None

            # Clear cache if needed (optional - may want to keep for future use)
            # self._model_cache.clear()

            self.logger.info("MLX model cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_supported_quantizations(self) -> list[str]:
        """Get list of supported quantization methods."""
        return ["4bit", "8bit", "none"]

    def get_model_formats(self) -> list[str]:
        """Get list of supported model formats."""
        return ["llama", "qwen", "mistral", "phi", "gemma"]

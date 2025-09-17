"""
LLM-as-judge evaluator for explanation quality assessment.

This module provides an LLM-based evaluation system for assessing the quality
of model explanations using structured prompts and scoring criteria.
"""

import asyncio
import json
import re
from typing import TYPE_CHECKING

from benchmark.core.logging import get_logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from benchmark.evaluation.metrics.explainability import ExplanationQualityScore


class LLMJudgeEvaluator:
    """LLM-as-judge evaluator for explanation quality."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize LLM judge evaluator.

        Args:
            model: Name of the LLM model to use for judging
        """
        self.model = model
        self.logger = get_logger("llm_judge_evaluator")
        self.rate_limiter = asyncio.Semaphore(10)  # Limit concurrent API calls

        # Initialize OpenAI client only when needed
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> "AsyncOpenAI":
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            except ImportError as e:
                self.logger.error("OpenAI package not available for LLM judge evaluation")
                raise ImportError("OpenAI package required for LLM judge evaluation") from e
        return self._client

    async def judge_explanation(
        self,
        input_sample: str,
        explanation: str,
        prediction: str,
        ground_truth_label: str,
        ground_truth_explanation: str = "",
    ) -> "ExplanationQualityScore":
        """
        Judge explanation quality using LLM.

        Args:
            input_sample: The input data that was analyzed
            explanation: The explanation to evaluate
            prediction: The model's prediction
            ground_truth_label: The correct label
            ground_truth_explanation: Optional reference explanation

        Returns:
            ExplanationQualityScore object with evaluation results
        """
        async with self.rate_limiter:
            try:
                prompt = self._create_judge_prompt(
                    input_sample,
                    explanation,
                    prediction,
                    ground_truth_label,
                    ground_truth_explanation,
                )

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cybersecurity analyst evaluating the quality of security analysis explanations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.1,
                )

                result_text = response.choices[0].message.content or ""
                return self._parse_judge_response(result_text)

            except Exception as e:
                self.logger.error(f"LLM judge evaluation failed: {e}")
                # Import ExplanationQualityScore locally to avoid circular imports
                from benchmark.evaluation.metrics.explainability import ExplanationQualityScore

                return ExplanationQualityScore(
                    overall_score=0.5,
                    technical_accuracy=0.5,
                    logical_consistency=0.5,
                    completeness=0.5,
                    clarity=0.5,
                    domain_relevance=0.5,
                    detailed_feedback=f"Evaluation failed: {e}",
                )

    def _create_judge_prompt(
        self,
        input_sample: str,
        explanation: str,
        prediction: str,
        ground_truth_label: str,
        ground_truth_explanation: str = "",
    ) -> str:
        """
        Create prompt for LLM judge evaluation.

        Args:
            input_sample: The input data
            explanation: The explanation to evaluate
            prediction: The model's prediction
            ground_truth_label: The correct label
            ground_truth_explanation: Optional reference explanation

        Returns:
            Formatted prompt string
        """
        prompt = f"""
Evaluate the quality of this cybersecurity analysis explanation:

INPUT DATA:
{input_sample}

PREDICTED LABEL: {prediction}
CORRECT LABEL: {ground_truth_label}

EXPLANATION TO EVALUATE:
{explanation}
"""

        if ground_truth_explanation:
            prompt += f"""
REFERENCE EXPLANATION:
{ground_truth_explanation}
"""

        prompt += """
Please evaluate this explanation on the following criteria (score 0.0 to 1.0):

1. TECHNICAL ACCURACY: Are the cybersecurity concepts and terminology used correctly?
2. LOGICAL CONSISTENCY: Does the explanation logically support the prediction?
3. COMPLETENESS: Does the explanation cover the key aspects of the analysis?
4. CLARITY: Is the explanation clear and understandable?
5. DOMAIN RELEVANCE: Are cybersecurity-specific details appropriately included?

Respond in JSON format:
{
    "technical_accuracy": 0.0-1.0,
    "logical_consistency": 0.0-1.0,
    "completeness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "domain_relevance": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "feedback": "Brief explanation of the scores"
}
"""

        return prompt

    def _parse_judge_response(self, response_text: str) -> "ExplanationQualityScore":
        """
        Parse LLM judge response into structured score.

        Args:
            response_text: Raw response text from LLM

        Returns:
            ExplanationQualityScore object
        """
        # Import ExplanationQualityScore locally to avoid circular imports
        from benchmark.evaluation.metrics.explainability import ExplanationQualityScore

        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)

                return ExplanationQualityScore(
                    overall_score=float(scores.get("overall_score", 0.5)),
                    technical_accuracy=float(scores.get("technical_accuracy", 0.5)),
                    logical_consistency=float(scores.get("logical_consistency", 0.5)),
                    completeness=float(scores.get("completeness", 0.5)),
                    clarity=float(scores.get("clarity", 0.5)),
                    domain_relevance=float(scores.get("domain_relevance", 0.5)),
                    detailed_feedback=scores.get("feedback", "No detailed feedback"),
                )
            else:
                # Fallback: try to extract scores from text
                return self._extract_scores_from_text(response_text)

        except Exception as e:
            self.logger.error(f"Failed to parse judge response: {e}")
            return ExplanationQualityScore(
                overall_score=0.5,
                technical_accuracy=0.5,
                logical_consistency=0.5,
                completeness=0.5,
                clarity=0.5,
                domain_relevance=0.5,
                detailed_feedback="Failed to parse judge response",
            )

    def _extract_scores_from_text(self, text: str) -> "ExplanationQualityScore":
        """
        Extract scores from non-JSON text response.

        Args:
            text: Response text to parse

        Returns:
            ExplanationQualityScore object
        """
        # Import ExplanationQualityScore locally to avoid circular imports
        from benchmark.evaluation.metrics.explainability import ExplanationQualityScore

        # Try to find numeric scores in text
        score_patterns = {
            "technical_accuracy": r"technical\s*accuracy[:\s]*([0-9.]+)",
            "logical_consistency": r"logical\s*consistency[:\s]*([0-9.]+)",
            "completeness": r"completeness[:\s]*([0-9.]+)",
            "clarity": r"clarity[:\s]*([0-9.]+)",
            "domain_relevance": r"domain\s*relevance[:\s]*([0-9.]+)",
            "overall_score": r"overall[:\s]*([0-9.]+)",
        }

        scores = {}
        for metric, pattern in score_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize score to 0-1 range if needed
                    if score > 1.0 and score <= 10.0:
                        score /= 10.0
                    scores[metric] = max(0.0, min(1.0, score))
                except ValueError:
                    scores[metric] = 0.5
            else:
                scores[metric] = 0.5

        return ExplanationQualityScore(
            overall_score=scores.get("overall_score", 0.5),
            technical_accuracy=scores.get("technical_accuracy", 0.5),
            logical_consistency=scores.get("logical_consistency", 0.5),
            completeness=scores.get("completeness", 0.5),
            clarity=scores.get("clarity", 0.5),
            domain_relevance=scores.get("domain_relevance", 0.5),
            detailed_feedback="Scores extracted from text analysis",
        )

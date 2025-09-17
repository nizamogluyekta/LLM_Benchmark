"""
Comprehensive explainability evaluator using multiple approaches.

This module provides a comprehensive evaluation framework for model explanations
using LLM-as-judge evaluation, automated metrics, and cybersecurity domain-specific analysis.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from benchmark.core.logging import get_logger
from benchmark.evaluation.base_evaluator import BaseEvaluator
from benchmark.evaluation.explainability.automated_metrics import AutomatedMetricsEvaluator
from benchmark.evaluation.explainability.llm_judge import LLMJudgeEvaluator
from benchmark.interfaces.evaluation_interfaces import MetricType


@dataclass
class ExplanationQualityScore:
    """Container for explanation quality assessment."""

    overall_score: float  # 0.0 to 1.0
    technical_accuracy: float
    logical_consistency: float
    completeness: float
    clarity: float
    domain_relevance: float
    detailed_feedback: str


class ExplainabilityEvaluator(BaseEvaluator):
    """Comprehensive explainability evaluation using multiple approaches."""

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        """Initialize explainability evaluator with specified judge model."""
        super().__init__(MetricType.EXPLAINABILITY)
        self.logger = get_logger("explainability_evaluator")

        # Initialize sub-evaluators
        self.llm_judge = LLMJudgeEvaluator(judge_model)
        self.automated_metrics = AutomatedMetricsEvaluator()

        # Define metrics produced by this evaluator
        self.metric_names = [
            "avg_explanation_quality",
            "technical_accuracy_score",
            "logical_consistency_score",
            "completeness_score",
            "clarity_score",
            "domain_relevance_score",
            "bleu_score",
            "rouge_l_score",
            "bert_score",
            "explanation_consistency",
            "ioc_accuracy",
            "mitre_coverage",
            "explanation_length_avg",
        ]

    async def evaluate(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Evaluate explanation quality using multiple methods.

        Args:
            predictions: List of prediction dictionaries with explanations
            ground_truth: List of ground truth dictionaries

        Returns:
            Dictionary of computed metrics

        Raises:
            ValueError: If input data is invalid or no explanations found
        """
        self.logger.info("Starting explainability evaluation")

        # Validate input data
        self.validate_input_data(predictions, ground_truth)

        # Extract explanations
        explanations = self._extract_explanations(predictions)
        if not explanations or not any(exp.strip() for exp in explanations):
            raise ValueError("No explanations found in predictions")

        self.logger.info(f"Evaluating {len(explanations)} explanations")

        # Run all evaluation methods in parallel
        evaluation_tasks = []

        # LLM-as-judge evaluation
        evaluation_tasks.append(
            ("llm_judge", self._evaluate_with_llm_judge(predictions, ground_truth))
        )

        # Automated metrics evaluation
        evaluation_tasks.append(
            ("automated", self._evaluate_with_automated_metrics(predictions, ground_truth))
        )

        # Domain-specific evaluation
        evaluation_tasks.append(
            ("domain", self._evaluate_domain_specific(predictions, ground_truth))
        )

        # Execute all evaluations
        results = {}
        for eval_name, task in evaluation_tasks:
            try:
                self.logger.debug(f"Running {eval_name} evaluation")
                eval_result = await task
                results.update(eval_result)
                self.logger.debug(f"Completed {eval_name} evaluation")
            except Exception as e:
                self.logger.warning(f"{eval_name} evaluation failed: {e}")
                # Continue with other evaluations

        self.logger.info(f"Explainability evaluation completed with {len(results)} metrics")
        return results

    def _extract_explanations(self, predictions: list[dict[str, Any]]) -> list[str]:
        """
        Extract explanations from predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            List of explanation strings
        """
        explanations = []

        for pred in predictions:
            explanation = pred.get("explanation", "")
            if not explanation:
                # Try alternative field names
                explanation = pred.get("reasoning", pred.get("rationale", ""))

            explanations.append(str(explanation) if explanation else "")

        return explanations

    async def _evaluate_with_llm_judge(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Evaluate explanations using LLM-as-judge.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            Dictionary of LLM judge metrics
        """
        judge_results = []

        # Process in batches to manage API rate limits
        batch_size = 10
        for i in range(0, len(predictions), batch_size):
            batch_predictions = predictions[i : i + batch_size]
            batch_ground_truth = ground_truth[i : i + batch_size]

            batch_tasks = []
            for pred, gt in zip(batch_predictions, batch_ground_truth, strict=False):
                task = self.llm_judge.judge_explanation(
                    input_sample=gt.get("input_text", ""),
                    explanation=pred.get("explanation", ""),
                    prediction=pred.get("prediction", ""),
                    ground_truth_label=gt.get("label", ""),
                    ground_truth_explanation=gt.get("explanation", ""),
                )
                batch_tasks.append(task)

            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        # Use default scores for failed evaluations
                        judge_results.append(
                            ExplanationQualityScore(
                                overall_score=0.5,
                                technical_accuracy=0.5,
                                logical_consistency=0.5,
                                completeness=0.5,
                                clarity=0.5,
                                domain_relevance=0.5,
                                detailed_feedback="Evaluation failed",
                            )
                        )
                    elif isinstance(result, ExplanationQualityScore):
                        judge_results.append(result)

            except Exception as e:
                self.logger.warning(f"Batch LLM evaluation failed: {e}")
                # Add default scores for this batch
                for _ in range(len(batch_tasks)):
                    judge_results.append(
                        ExplanationQualityScore(
                            overall_score=0.5,
                            technical_accuracy=0.5,
                            logical_consistency=0.5,
                            completeness=0.5,
                            clarity=0.5,
                            domain_relevance=0.5,
                            detailed_feedback="Evaluation failed",
                        )
                    )

        # Aggregate LLM judge results
        if judge_results:
            return {
                "avg_explanation_quality": sum(r.overall_score for r in judge_results)
                / len(judge_results),
                "technical_accuracy_score": sum(r.technical_accuracy for r in judge_results)
                / len(judge_results),
                "logical_consistency_score": sum(r.logical_consistency for r in judge_results)
                / len(judge_results),
                "completeness_score": sum(r.completeness for r in judge_results)
                / len(judge_results),
                "clarity_score": sum(r.clarity for r in judge_results) / len(judge_results),
                "domain_relevance_score": sum(r.domain_relevance for r in judge_results)
                / len(judge_results),
            }
        else:
            return {}

    async def _evaluate_with_automated_metrics(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Evaluate explanations using automated metrics.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            Dictionary of automated metrics
        """
        # Extract explanations and references
        candidate_explanations = [pred.get("explanation", "") for pred in predictions]
        reference_explanations = [gt.get("explanation", "") for gt in ground_truth]

        # Only evaluate if we have reference explanations
        if any(ref.strip() for ref in reference_explanations):
            return await self.automated_metrics.calculate_metrics(
                candidate_explanations, reference_explanations
            )
        else:
            # If no reference explanations, use basic metrics
            return await self.automated_metrics.calculate_intrinsic_metrics(candidate_explanations)

    async def _evaluate_domain_specific(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Evaluate domain-specific aspects of explanations.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            Dictionary of domain-specific metrics
        """
        domain_results = {
            "explanation_consistency": 0.0,
            "ioc_accuracy": 0.0,
            "mitre_coverage": 0.0,
            "explanation_length_avg": 0.0,
        }

        explanations = [pred.get("explanation", "") for pred in predictions]

        if not explanations:
            return domain_results

        # Calculate explanation consistency
        domain_results["explanation_consistency"] = self._calculate_explanation_consistency(
            predictions, ground_truth
        )

        # Analyze IoC (Indicators of Compromise) accuracy
        domain_results["ioc_accuracy"] = self._analyze_ioc_accuracy(predictions, ground_truth)

        # Check MITRE ATT&CK technique coverage
        domain_results["mitre_coverage"] = self._analyze_mitre_coverage(explanations)

        # Calculate average explanation length
        valid_explanations = [exp for exp in explanations if exp.strip()]
        if valid_explanations:
            avg_length = sum(len(exp.split()) for exp in valid_explanations) / len(
                valid_explanations
            )
            domain_results["explanation_length_avg"] = avg_length

        return domain_results

    def _calculate_explanation_consistency(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> float:
        """
        Calculate consistency between explanations and predictions.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            Consistency score between 0.0 and 1.0
        """
        consistent_count = 0
        total_count = 0

        for pred, _gt in zip(predictions, ground_truth, strict=False):
            explanation = pred.get("explanation", "").lower()
            prediction = pred.get("prediction", "").lower()

            if not explanation:
                continue

            total_count += 1

            # Check if explanation supports the prediction
            if prediction == "attack":
                # Look for attack-related keywords in explanation
                attack_keywords = [
                    "attack",
                    "malicious",
                    "threat",
                    "intrusion",
                    "suspicious",
                    "compromise",
                    "exploit",
                    "malware",
                    "breach",
                    "unauthorized",
                ]
                if any(keyword in explanation for keyword in attack_keywords):
                    consistent_count += 1
            elif prediction == "benign":
                # Look for benign-related keywords or absence of attack keywords
                benign_keywords = [
                    "normal",
                    "legitimate",
                    "benign",
                    "safe",
                    "expected",
                    "routine",
                ]
                attack_keywords = [
                    "attack",
                    "malicious",
                    "threat",
                    "intrusion",
                    "suspicious",
                ]

                has_benign_keywords = any(keyword in explanation for keyword in benign_keywords)
                has_attack_keywords = any(keyword in explanation for keyword in attack_keywords)

                if has_benign_keywords or not has_attack_keywords:
                    consistent_count += 1

        return consistent_count / total_count if total_count > 0 else 0.0

    def _analyze_ioc_accuracy(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> float:
        """
        Analyze accuracy of Indicators of Compromise mentioned in explanations.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries

        Returns:
            IoC accuracy score between 0.0 and 1.0
        """
        import re

        accurate_count = 0
        total_ioc_count = 0

        # Common IoC patterns
        ip_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
        hash_pattern = r"\b[a-fA-F0-9]{32,64}\b"
        domain_pattern = r"\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b"

        for pred, gt in zip(predictions, ground_truth, strict=False):
            explanation = pred.get("explanation", "")
            input_text = gt.get("input_text", "")

            if not explanation:
                continue

            # Extract IoCs from explanation
            explanation_ips = set(re.findall(ip_pattern, explanation))
            explanation_hashes = set(re.findall(hash_pattern, explanation))
            explanation_domains = set(re.findall(domain_pattern, explanation))

            # Extract IoCs from input
            input_ips = set(re.findall(ip_pattern, input_text))
            input_hashes = set(re.findall(hash_pattern, input_text))
            input_domains = set(re.findall(domain_pattern, input_text))

            # Check accuracy of mentioned IoCs
            for ioc_set, input_set in [
                (explanation_ips, input_ips),
                (explanation_hashes, input_hashes),
                (explanation_domains, input_domains),
            ]:
                for ioc in ioc_set:
                    total_ioc_count += 1
                    if ioc in input_set:
                        accurate_count += 1

        return accurate_count / total_ioc_count if total_ioc_count > 0 else 1.0

    def _analyze_mitre_coverage(self, explanations: list[str]) -> float:
        """
        Analyze coverage of MITRE ATT&CK techniques in explanations.

        Args:
            explanations: List of explanation strings

        Returns:
            MITRE technique coverage score between 0.0 and 1.0
        """
        # Common MITRE ATT&CK techniques relevant to cybersecurity
        mitre_techniques = [
            "reconnaissance",
            "initial access",
            "execution",
            "persistence",
            "privilege escalation",
            "defense evasion",
            "credential access",
            "discovery",
            "lateral movement",
            "collection",
            "command and control",
            "exfiltration",
            "impact",
            "phishing",
            "spearphishing",
            "brute force",
            "sql injection",
            "buffer overflow",
            "code injection",
            "dos",
        ]

        technique_mentions = 0
        total_explanations = len([exp for exp in explanations if exp.strip()])

        if total_explanations == 0:
            return 0.0

        for explanation in explanations:
            if not explanation.strip():
                continue

            explanation_lower = explanation.lower()
            for technique in mitre_techniques:
                if technique in explanation_lower:
                    technique_mentions += 1
                    break  # Count max one technique per explanation

        return technique_mentions / total_explanations

    def get_metric_names(self) -> list[str]:
        """Get list of metrics this evaluator produces."""
        return self.metric_names

    def get_required_prediction_fields(self) -> list[str]:
        """Get required fields in prediction data."""
        return ["explanation"]

    def get_required_ground_truth_fields(self) -> list[str]:
        """Get required fields in ground truth data."""
        return []  # Ground truth explanation is optional

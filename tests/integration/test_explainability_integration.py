"""Integration tests for explainability evaluation."""

import os
from unittest.mock import AsyncMock

import pytest

from benchmark.evaluation.metrics.explainability import (
    ExplainabilityEvaluator,
    ExplanationQualityScore,
)


class TestExplainabilityIntegration:
    """Integration tests for explainability evaluation system."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction data with explanations."""
        return [
            {
                "prediction": "attack",
                "confidence": 0.9,
                "explanation": "The network traffic from 192.168.1.100 to 10.0.0.1 shows suspicious patterns. "
                "The payload contains malicious code attempting SQL injection attack. "
                "MITRE ATT&CK technique T1190 (Exploit Public-Facing Application) detected.",
            },
            {
                "prediction": "benign",
                "confidence": 0.8,
                "explanation": "This appears to be normal HTTP traffic between legitimate hosts. "
                "No suspicious patterns or known attack signatures detected. "
                "Standard web browsing behavior observed.",
            },
            {
                "prediction": "attack",
                "confidence": 0.95,
                "explanation": "Email shows clear signs of phishing attempt. "
                "Suspicious domain evil.example.com used with credential harvesting intent. "
                "Social engineering techniques employed to trick user.",
            },
            {
                "prediction": "benign",
                "confidence": 0.7,
                "explanation": "Regular system administration activity detected. "
                "Authorized user performing routine maintenance tasks. "
                "All actions within expected operational parameters.",
            },
            {
                "prediction": "attack",
                "confidence": 0.85,
                "explanation": "Binary file hash 5d41402abc4b2a76b9719d911017c592 matches known malware signature. "
                "Reconnaissance and initial access attempts detected. "
                "Command and control communication established.",
            },
        ]

    @pytest.fixture
    def sample_ground_truth(self):
        """Sample ground truth data."""
        return [
            {
                "label": "attack",
                "input_text": "Network log: 192.168.1.100 -> 10.0.0.1:80 SQL injection payload detected",
                "explanation": "SQL injection attack attempting to extract database information",
            },
            {
                "label": "benign",
                "input_text": "HTTP GET request to www.google.com from authorized user",
                "explanation": "Normal web browsing activity",
            },
            {
                "label": "attack",
                "input_text": "Email from fake-bank@evil.example.com requesting login credentials",
                "explanation": "Phishing email targeting user credentials",
            },
            {
                "label": "benign",
                "input_text": "System admin running scheduled backup script",
                "explanation": "Authorized administrative activity",
            },
            {
                "label": "attack",
                "input_text": "Executable file with hash 5d41402abc4b2a76b9719d911017c592 executed",
                "explanation": "Known malware execution with C2 communication",
            },
        ]

    @pytest.fixture
    def explainability_evaluator(self):
        """Create explainability evaluator for testing."""
        return ExplainabilityEvaluator("gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(
        self, explainability_evaluator, sample_predictions, sample_ground_truth
    ):
        """Test complete explainability evaluation pipeline."""
        # Mock LLM judge to avoid actual API calls
        mock_llm_scores = [
            ExplanationQualityScore(
                overall_score=0.85,
                technical_accuracy=0.9,
                logical_consistency=0.9,
                completeness=0.8,
                clarity=0.8,
                domain_relevance=0.9,
                detailed_feedback="Excellent technical explanation with good IoC identification",
            ),
            ExplanationQualityScore(
                overall_score=0.75,
                technical_accuracy=0.8,
                logical_consistency=0.8,
                completeness=0.7,
                clarity=0.9,
                domain_relevance=0.6,
                detailed_feedback="Clear explanation but limited cybersecurity specifics",
            ),
            ExplanationQualityScore(
                overall_score=0.9,
                technical_accuracy=0.95,
                logical_consistency=0.95,
                completeness=0.85,
                clarity=0.9,
                domain_relevance=0.95,
                detailed_feedback="Comprehensive phishing analysis with domain identification",
            ),
            ExplanationQualityScore(
                overall_score=0.7,
                technical_accuracy=0.7,
                logical_consistency=0.8,
                completeness=0.6,
                clarity=0.8,
                domain_relevance=0.7,
                detailed_feedback="Adequate explanation of benign activity",
            ),
            ExplanationQualityScore(
                overall_score=0.95,
                technical_accuracy=0.95,
                logical_consistency=0.9,
                completeness=0.95,
                clarity=0.9,
                domain_relevance=1.0,
                detailed_feedback="Excellent malware analysis with hash verification and MITRE mapping",
            ),
        ]

        # Mock the LLM judge responses
        explainability_evaluator.llm_judge.judge_explanation = AsyncMock(
            side_effect=mock_llm_scores
        )

        # Mock automated metrics to avoid dependency issues
        explainability_evaluator.automated_metrics.calculate_intrinsic_metrics = AsyncMock(
            return_value={
                "explanation_uniqueness": 0.8,
                "vocabulary_diversity": 0.7,
                "avg_explanation_length": 25.6,
                "avg_sentences_per_explanation": 2.4,
                "explanation_complexity": 0.6,
            }
        )

        # Run evaluation
        results = await explainability_evaluator.evaluate(sample_predictions, sample_ground_truth)

        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) > 0

        # Check LLM judge metrics
        assert "avg_explanation_quality" in results
        assert "technical_accuracy_score" in results
        assert "logical_consistency_score" in results
        assert "completeness_score" in results
        assert "clarity_score" in results
        assert "domain_relevance_score" in results

        # Check domain-specific metrics
        assert "explanation_consistency" in results
        assert "ioc_accuracy" in results
        assert "mitre_coverage" in results
        assert "explanation_length_avg" in results

        # Verify metric values are in expected ranges
        assert 0 <= results["avg_explanation_quality"] <= 1
        assert 0 <= results["technical_accuracy_score"] <= 1
        assert 0 <= results["explanation_consistency"] <= 1
        assert results["explanation_length_avg"] > 0

        # Verify some expected behaviors
        assert results["explanation_consistency"] > 0.5  # Should be reasonably consistent
        assert results["ioc_accuracy"] > 0.5  # Should identify some IoCs correctly
        assert results["mitre_coverage"] > 0  # Should identify some MITRE techniques

    @pytest.mark.asyncio
    async def test_evaluation_with_reference_explanations(
        self, explainability_evaluator, sample_predictions, sample_ground_truth
    ):
        """Test evaluation when reference explanations are available."""
        # Mock LLM judge
        explainability_evaluator.llm_judge.judge_explanation = AsyncMock(
            return_value=ExplanationQualityScore(
                overall_score=0.8,
                technical_accuracy=0.8,
                logical_consistency=0.8,
                completeness=0.8,
                clarity=0.8,
                domain_relevance=0.8,
                detailed_feedback="Good explanation",
            )
        )

        # Mock automated metrics with reference comparison
        explainability_evaluator.automated_metrics.calculate_metrics = AsyncMock(
            return_value={
                "bleu_score": 0.4,
                "rouge1_score": 0.6,
                "rouge2_score": 0.3,
                "rouge_l_score": 0.5,
                "bert_score": 0.7,
            }
        )

        results = await explainability_evaluator.evaluate(sample_predictions, sample_ground_truth)

        # Should contain reference-based metrics
        assert "bleu_score" in results
        assert "rouge_l_score" in results
        assert "bert_score" in results

        # Verify automated metrics were called with reference explanations
        explainability_evaluator.automated_metrics.calculate_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_without_reference_explanations(self, explainability_evaluator):
        """Test evaluation when no reference explanations are available."""
        predictions = [
            {"prediction": "attack", "explanation": "Suspicious network activity detected"},
            {"prediction": "benign", "explanation": "Normal user behavior observed"},
        ]
        ground_truth = [{"label": "attack"}, {"label": "benign"}]  # No explanations

        # Mock LLM judge
        explainability_evaluator.llm_judge.judge_explanation = AsyncMock(
            return_value=ExplanationQualityScore(
                overall_score=0.8,
                technical_accuracy=0.8,
                logical_consistency=0.8,
                completeness=0.8,
                clarity=0.8,
                domain_relevance=0.8,
                detailed_feedback="Good explanation",
            )
        )

        # Mock intrinsic metrics
        explainability_evaluator.automated_metrics.calculate_intrinsic_metrics = AsyncMock(
            return_value={"explanation_uniqueness": 0.9, "vocabulary_diversity": 0.8}
        )

        results = await explainability_evaluator.evaluate(predictions, ground_truth)

        # Should contain intrinsic metrics but not reference-based ones
        assert "explanation_uniqueness" in results or len(results) > 0
        explainability_evaluator.automated_metrics.calculate_intrinsic_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_partial_failures(
        self, explainability_evaluator, sample_predictions, sample_ground_truth
    ):
        """Test evaluation continues when some components fail."""
        # Mock LLM judge to fail
        explainability_evaluator.llm_judge.judge_explanation = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Mock automated metrics to succeed
        explainability_evaluator.automated_metrics.calculate_intrinsic_metrics = AsyncMock(
            return_value={"explanation_uniqueness": 0.8}
        )

        results = await explainability_evaluator.evaluate(sample_predictions, sample_ground_truth)

        # Should still return results from successful components
        assert isinstance(results, dict)
        # Should have domain-specific metrics even if LLM judge fails
        assert "explanation_consistency" in results
        assert "mitre_coverage" in results

    def test_ioc_accuracy_calculation(self, explainability_evaluator):
        """Test IoC accuracy calculation with various IoC types."""
        predictions = [
            {
                "explanation": "Malicious IP 192.168.1.100 connected to C2 server at evil.com",
                "prediction": "attack",
            },
            {
                "explanation": "File hash abc123def456 matches known malware signature",
                "prediction": "attack",
            },
            {
                "explanation": "Connection to legitimate.org appears normal",
                "prediction": "benign",
            },
        ]
        ground_truth = [
            {"input_text": "Network traffic from 192.168.1.100 to evil.com detected"},
            {"input_text": "File with hash abc123def456 was executed"},
            {"input_text": "HTTPS connection to legitimate.org established"},
        ]

        ioc_accuracy = explainability_evaluator._analyze_ioc_accuracy(predictions, ground_truth)

        assert 0 <= ioc_accuracy <= 1
        assert ioc_accuracy > 0.5  # Should correctly identify most IoCs

    def test_mitre_technique_detection(self, explainability_evaluator):
        """Test MITRE ATT&CK technique detection in explanations."""
        explanations = [
            "Initial access gained through spearphishing email attachment",
            "Privilege escalation attempt detected using buffer overflow",
            "Lateral movement across network using stolen credentials",
            "Data exfiltration to external command and control server",
            "Normal user browsing behavior with no threats detected",
        ]

        mitre_coverage = explainability_evaluator._analyze_mitre_coverage(explanations)

        assert 0 <= mitre_coverage <= 1
        assert mitre_coverage >= 0.6  # Should detect techniques in first 4 explanations

    def test_explanation_consistency_analysis(self, explainability_evaluator):
        """Test explanation consistency with predictions."""
        predictions = [
            {"explanation": "Malicious attack with exploit attempt", "prediction": "attack"},
            {"explanation": "Normal legitimate user activity", "prediction": "benign"},
            {"explanation": "Suspicious intrusion detected", "prediction": "attack"},
            {"explanation": "Safe routine operation", "prediction": "benign"},
            {
                "explanation": "Attack keywords but marked benign",
                "prediction": "benign",
            },  # Inconsistent
        ]
        ground_truth = [
            {"label": "attack"},
            {"label": "benign"},
            {"label": "attack"},
            {"label": "benign"},
            {"label": "benign"},
        ]

        consistency = explainability_evaluator._calculate_explanation_consistency(
            predictions, ground_truth
        )

        assert 0 <= consistency <= 1
        # Should be high but not perfect due to the inconsistent example
        assert 0.6 <= consistency <= 0.9

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, explainability_evaluator):
        """Test performance with larger dataset."""
        # Create larger dataset
        large_predictions = []
        large_ground_truth = []

        base_explanations = [
            "Network intrusion attempt detected",
            "Normal user activity observed",
            "Malware execution prevented",
            "Legitimate system operation",
        ]

        for i in range(50):  # 50 samples
            pred_type = "attack" if i % 2 == 0 else "benign"
            explanation = base_explanations[i % 4] + f" (Sample {i})"

            large_predictions.append({"prediction": pred_type, "explanation": explanation})
            large_ground_truth.append({"label": pred_type, "input_text": f"Input data {i}"})

        # Mock components for performance test
        explainability_evaluator.llm_judge.judge_explanation = AsyncMock(
            return_value=ExplanationQualityScore(
                overall_score=0.8,
                technical_accuracy=0.8,
                logical_consistency=0.8,
                completeness=0.8,
                clarity=0.8,
                domain_relevance=0.8,
                detailed_feedback="Test explanation",
            )
        )

        explainability_evaluator.automated_metrics.calculate_intrinsic_metrics = AsyncMock(
            return_value={"explanation_uniqueness": 0.7}
        )

        # Measure evaluation time
        import time

        start_time = time.time()
        results = await explainability_evaluator.evaluate(large_predictions, large_ground_truth)
        end_time = time.time()

        evaluation_time = end_time - start_time

        assert isinstance(results, dict)
        assert len(results) > 0
        # Should complete within reasonable time (adjust as needed)
        assert evaluation_time < 30  # 30 seconds max for 50 samples

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    async def test_real_llm_evaluation(self, explainability_evaluator):
        """Test with real LLM API (requires API key)."""
        predictions = [
            {
                "prediction": "attack",
                "explanation": "SQL injection attempt detected in HTTP request parameters",
            }
        ]
        ground_truth = [
            {"label": "attack", "input_text": "HTTP request with malicious SQL payload"}
        ]

        # Don't mock LLM judge for this test
        results = await explainability_evaluator.evaluate(predictions, ground_truth)

        assert isinstance(results, dict)
        assert "avg_explanation_quality" in results
        assert 0 <= results["avg_explanation_quality"] <= 1

    def test_metric_names_coverage(self, explainability_evaluator):
        """Test that all declared metrics are potentially produced."""
        metric_names = explainability_evaluator.get_metric_names()

        # Verify all expected metrics are declared
        expected_metrics = [
            "avg_explanation_quality",
            "technical_accuracy_score",
            "logical_consistency_score",
            "completeness_score",
            "clarity_score",
            "domain_relevance_score",
            "explanation_consistency",
            "ioc_accuracy",
            "mitre_coverage",
            "explanation_length_avg",
        ]

        for metric in expected_metrics:
            assert metric in metric_names

    @pytest.mark.asyncio
    async def test_empty_explanation_handling(self, explainability_evaluator):
        """Test handling of empty or missing explanations."""
        predictions = [
            {"prediction": "attack", "explanation": ""},  # Empty explanation
            {"prediction": "benign"},  # Missing explanation
        ]
        ground_truth = [{"label": "attack"}, {"label": "benign"}]

        with pytest.raises(ValueError, match="No explanations found"):
            await explainability_evaluator.evaluate(predictions, ground_truth)

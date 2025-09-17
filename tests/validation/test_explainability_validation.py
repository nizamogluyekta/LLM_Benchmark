"""Validation tests for explainability evaluation with sample data."""

from unittest.mock import AsyncMock

import pytest

from benchmark.evaluation.metrics.explainability import (
    ExplainabilityEvaluator,
    ExplanationQualityScore,
)


class TestExplainabilityValidation:
    """Validation tests demonstrating explainability evaluation capabilities."""

    @pytest.fixture
    def cybersecurity_predictions(self):
        """Sample cybersecurity predictions with explanations."""
        return [
            {
                "prediction": "attack",
                "confidence": 0.95,
                "explanation": "Network traffic from 192.168.1.100 shows SQL injection attempt. "
                "The HTTP request contains malicious payload '1' OR '1'='1' targeting the login form. "
                "This matches MITRE ATT&CK technique T1190 (Exploit Public-Facing Application). "
                "Immediate blocking recommended.",
            },
            {
                "prediction": "benign",
                "confidence": 0.82,
                "explanation": "Email communication appears legitimate. Sender domain google.com "
                "is verified with valid SPF and DKIM records. Content analysis shows normal "
                "business correspondence patterns. No suspicious links or attachments detected.",
            },
            {
                "prediction": "attack",
                "confidence": 0.88,
                "explanation": "Binary file hash 5d41402abc4b2a76b9719d911017c592 matches known malware "
                "signature in threat intelligence database. File exhibits reconnaissance behavior "
                "and attempts lateral movement via SMB protocol. Quarantine immediately.",
            },
            {
                "prediction": "benign",
                "confidence": 0.76,
                "explanation": "System administrator activity detected. User account admin@company.com "
                "performing scheduled maintenance during authorized window. All commands executed "
                "are within expected operational parameters for backup procedures.",
            },
            {
                "prediction": "attack",
                "confidence": 0.92,
                "explanation": "Phishing email detected targeting credentials. Sender spoofs "
                "legitimate bank domain using evil-bank.example.com. Email contains credential "
                "harvesting form with urgency tactics. Clear social engineering attempt.",
            },
        ]

    @pytest.fixture
    def cybersecurity_ground_truth(self):
        """Sample ground truth for cybersecurity scenarios."""
        return [
            {
                "label": "attack",
                "input_text": "HTTP POST /login.php?id=1' OR '1'='1' from 192.168.1.100",
                "explanation": "SQL injection attack attempting to bypass authentication",
            },
            {
                "label": "benign",
                "input_text": "Email from notifications@google.com regarding account security",
                "explanation": "Legitimate security notification from Google",
            },
            {
                "label": "attack",
                "input_text": "Executable file trojan.exe with hash 5d41402abc4b2a76b9719d911017c592",
                "explanation": "Known trojan attempting network reconnaissance and lateral movement",
            },
            {
                "label": "benign",
                "input_text": "Admin user running backup script at 2 AM scheduled maintenance",
                "explanation": "Authorized administrative activity during maintenance window",
            },
            {
                "label": "attack",
                "input_text": "Email from security@evil-bank.example.com requesting password reset",
                "explanation": "Phishing email using domain spoofing for credential theft",
            },
        ]

    @pytest.mark.asyncio
    async def test_comprehensive_explainability_evaluation(
        self, cybersecurity_predictions, cybersecurity_ground_truth
    ):
        """Test comprehensive explainability evaluation with realistic cybersecurity data."""
        evaluator = ExplainabilityEvaluator("gpt-4o-mini")

        # Mock LLM judge with realistic cybersecurity-focused scores
        mock_scores = [
            # SQL injection - excellent technical explanation
            ExplanationQualityScore(
                overall_score=0.92,
                technical_accuracy=0.95,
                logical_consistency=0.95,
                completeness=0.90,
                clarity=0.88,
                domain_relevance=0.95,
                detailed_feedback="Excellent technical explanation with specific IoC identification, "
                "MITRE ATT&CK mapping, and clear remediation advice",
            ),
            # Benign email - good but less cybersecurity-specific
            ExplanationQualityScore(
                overall_score=0.78,
                technical_accuracy=0.85,
                logical_consistency=0.88,
                completeness=0.75,
                clarity=0.90,
                domain_relevance=0.70,
                detailed_feedback="Clear explanation with good verification details but could "
                "include more security-specific analysis",
            ),
            # Malware detection - comprehensive technical analysis
            ExplanationQualityScore(
                overall_score=0.89,
                technical_accuracy=0.95,
                logical_consistency=0.90,
                completeness=0.85,
                clarity=0.85,
                domain_relevance=0.95,
                detailed_feedback="Strong malware analysis with hash verification, behavioral "
                "analysis, and appropriate response recommendation",
            ),
            # Admin activity - adequate explanation
            ExplanationQualityScore(
                overall_score=0.72,
                technical_accuracy=0.80,
                logical_consistency=0.85,
                completeness=0.65,
                clarity=0.80,
                domain_relevance=0.65,
                detailed_feedback="Adequate explanation of benign activity but lacks detailed "
                "verification of authorization mechanisms",
            ),
            # Phishing detection - excellent social engineering analysis
            ExplanationQualityScore(
                overall_score=0.94,
                technical_accuracy=0.92,
                logical_consistency=0.95,
                completeness=0.95,
                clarity=0.95,
                domain_relevance=0.95,
                detailed_feedback="Outstanding phishing analysis with domain analysis, social "
                "engineering tactic identification, and clear threat assessment",
            ),
        ]

        # Mock LLM judge responses
        evaluator.llm_judge.judge_explanation = AsyncMock(side_effect=mock_scores)

        # Mock automated metrics with realistic values for cybersecurity content
        evaluator.automated_metrics.calculate_metrics = AsyncMock(
            return_value={
                "bleu_score": 0.42,  # Moderate overlap with reference explanations
                "rouge1_score": 0.58,  # Good unigram overlap
                "rouge2_score": 0.35,  # Moderate bigram overlap
                "rouge_l_score": 0.51,  # Good longest common subsequence
                "bert_score": 0.73,  # Strong semantic similarity
            }
        )

        # Run evaluation
        results = await evaluator.evaluate(cybersecurity_predictions, cybersecurity_ground_truth)

        # Validate results structure and content
        assert isinstance(results, dict)
        assert len(results) >= 10  # Should have multiple types of metrics

        # Check LLM judge metrics
        assert "avg_explanation_quality" in results
        assert "technical_accuracy_score" in results
        assert "logical_consistency_score" in results
        assert "completeness_score" in results
        assert "clarity_score" in results
        assert "domain_relevance_score" in results

        # Check automated metrics
        assert "bleu_score" in results
        assert "rouge_l_score" in results
        assert "bert_score" in results

        # Check domain-specific metrics
        assert "explanation_consistency" in results
        assert "ioc_accuracy" in results
        assert "mitre_coverage" in results
        assert "explanation_length_avg" in results

        # Validate metric ranges and expected values
        assert 0.8 <= results["avg_explanation_quality"] <= 1.0  # Should be high quality
        assert 0.85 <= results["technical_accuracy_score"] <= 1.0  # Strong technical accuracy
        assert 0.8 <= results["explanation_consistency"] <= 1.0  # High consistency expected
        assert 0.6 <= results["ioc_accuracy"] <= 1.0  # Good IoC identification
        assert 0.4 <= results["mitre_coverage"] <= 1.0  # Some MITRE technique coverage
        assert results["explanation_length_avg"] > 20  # Detailed explanations

        # Print results for manual inspection
        print("\\n=== Explainability Evaluation Results ===")
        for metric, value in sorted(results.items()):
            if isinstance(value, float):
                print(f"{metric}: {value:.3f}")
            else:
                print(f"{metric}: {value}")

        # Verify that the evaluation captured key aspects
        assert results["technical_accuracy_score"] > 0.85  # High technical quality
        assert results["domain_relevance_score"] > 0.8  # Strong cybersecurity relevance
        assert results["explanation_consistency"] >= 0.8  # Explanations match predictions

    def test_domain_specific_analysis_capabilities(
        self, cybersecurity_predictions, cybersecurity_ground_truth
    ):
        """Test domain-specific analysis capabilities."""
        evaluator = ExplainabilityEvaluator("gpt-4o-mini")

        # Test IoC accuracy analysis
        ioc_accuracy = evaluator._analyze_ioc_accuracy(
            cybersecurity_predictions, cybersecurity_ground_truth
        )
        assert 0 <= ioc_accuracy <= 1
        assert ioc_accuracy > 0.5  # Should correctly identify most IoCs

        # Test MITRE technique coverage
        explanations = [pred["explanation"] for pred in cybersecurity_predictions]
        mitre_coverage = evaluator._analyze_mitre_coverage(explanations)
        assert 0 <= mitre_coverage <= 1
        assert mitre_coverage >= 0.6  # Should identify techniques in most explanations

        # Test explanation consistency
        consistency = evaluator._calculate_explanation_consistency(
            cybersecurity_predictions, cybersecurity_ground_truth
        )
        assert 0 <= consistency <= 1
        assert consistency >= 0.8  # Should be highly consistent

        print("\\n=== Domain-Specific Analysis Results ===")
        print(f"IoC Accuracy: {ioc_accuracy:.3f}")
        print(f"MITRE Coverage: {mitre_coverage:.3f}")
        print(f"Explanation Consistency: {consistency:.3f}")

    @pytest.mark.asyncio
    async def test_evaluation_robustness(self):
        """Test evaluation robustness with various edge cases."""
        evaluator = ExplainabilityEvaluator("gpt-4o-mini")

        # Test with minimal explanations
        minimal_predictions = [
            {"prediction": "attack", "explanation": "Suspicious activity detected"},
            {"prediction": "benign", "explanation": "Normal behavior"},
        ]
        minimal_ground_truth = [{"label": "attack"}, {"label": "benign"}]

        # Mock for minimal case
        evaluator.llm_judge.judge_explanation = AsyncMock(
            return_value=ExplanationQualityScore(
                overall_score=0.6,
                technical_accuracy=0.5,
                logical_consistency=0.7,
                completeness=0.4,
                clarity=0.8,
                domain_relevance=0.5,
                detailed_feedback="Brief explanation with limited detail",
            )
        )

        evaluator.automated_metrics.calculate_intrinsic_metrics = AsyncMock(
            return_value={"explanation_uniqueness": 0.9, "vocabulary_diversity": 0.6}
        )

        results = await evaluator.evaluate(minimal_predictions, minimal_ground_truth)

        assert isinstance(results, dict)
        assert "avg_explanation_quality" in results
        assert results["avg_explanation_quality"] < 0.8  # Should reflect lower quality

        # Test with very detailed explanations
        detailed_predictions = [
            {
                "prediction": "attack",
                "explanation": "Comprehensive analysis reveals a sophisticated multi-stage attack. "
                "Initial reconnaissance via T1595.002 (Vulnerability Scanning) identified web application "
                "vulnerabilities. Exploitation through T1190 (Exploit Public-Facing Application) using "
                "SQL injection with payload '1' UNION SELECT * FROM users WHERE '1'='1'. "
                "The attack originated from IP 192.168.1.100 which resolves to compromised botnet node. "
                "Lateral movement attempted via T1021.002 (SMB/Windows Admin Shares) targeting "
                "administrative credentials. Command and control established with C2 server at "
                "malicious.example.com using encrypted TLS tunnel on port 443. "
                "Immediate containment and forensic analysis recommended.",
            }
        ]
        detailed_ground_truth = [{"label": "attack", "input_text": "Complex multi-vector attack"}]

        # Mock for detailed case
        evaluator.llm_judge.judge_explanation = AsyncMock(
            return_value=ExplanationQualityScore(
                overall_score=0.95,
                technical_accuracy=0.98,
                logical_consistency=0.95,
                completeness=0.98,
                clarity=0.90,
                domain_relevance=1.0,
                detailed_feedback="Exceptional detailed analysis with comprehensive MITRE mapping",
            )
        )

        detailed_results = await evaluator.evaluate(detailed_predictions, detailed_ground_truth)

        assert detailed_results["avg_explanation_quality"] > 0.9  # Should reflect high quality
        assert detailed_results["explanation_length_avg"] > 50  # Very detailed
        assert detailed_results["mitre_coverage"] >= 1.0  # Full technique coverage

        print("\\n=== Robustness Test Results ===")
        print(f"Minimal explanations quality: {results['avg_explanation_quality']:.3f}")
        print(f"Detailed explanations quality: {detailed_results['avg_explanation_quality']:.3f}")

    def test_metric_completeness(self):
        """Test that all declared metrics are meaningful for cybersecurity evaluation."""
        evaluator = ExplainabilityEvaluator("gpt-4o-mini")
        metric_names = evaluator.get_metric_names()

        # Essential cybersecurity explainability metrics
        essential_metrics = [
            "avg_explanation_quality",  # Overall quality assessment
            "technical_accuracy_score",  # Technical correctness
            "domain_relevance_score",  # Cybersecurity relevance
            "explanation_consistency",  # Prediction-explanation alignment
            "ioc_accuracy",  # IoC identification accuracy
            "mitre_coverage",  # ATT&CK framework coverage
        ]

        for metric in essential_metrics:
            assert metric in metric_names, f"Essential metric {metric} missing"

        # Automated comparison metrics
        comparison_metrics = ["bleu_score", "rouge_l_score", "bert_score"]
        for metric in comparison_metrics:
            assert metric in metric_names, f"Comparison metric {metric} missing"

        print("\\n=== Metric Completeness ===")
        print(f"Total metrics: {len(metric_names)}")
        print(f"Essential cybersecurity metrics: {len(essential_metrics)}")
        print("All essential metrics present: ✓")
        print(f"Metrics: {', '.join(metric_names)}")

    @pytest.mark.asyncio
    async def test_performance_characteristics(self):
        """Test performance characteristics with larger datasets."""
        evaluator = ExplainabilityEvaluator("gpt-4o-mini")

        # Create larger test dataset
        large_predictions = []
        large_ground_truth = []

        base_explanation = "Network traffic analysis shows potential security concern. "
        for i in range(20):  # 20 samples for performance test
            prediction_type = "attack" if i % 2 == 0 else "benign"
            explanation = base_explanation + f"Sample {i} analysis with unique context and details."

            large_predictions.append(
                {"prediction": prediction_type, "explanation": explanation, "confidence": 0.8}
            )
            large_ground_truth.append({"label": prediction_type, "input_text": f"Sample input {i}"})

        # Mock for performance test
        evaluator.llm_judge.judge_explanation = AsyncMock(
            return_value=ExplanationQualityScore(
                overall_score=0.8,
                technical_accuracy=0.8,
                logical_consistency=0.8,
                completeness=0.8,
                clarity=0.8,
                domain_relevance=0.8,
                detailed_feedback="Standard quality explanation",
            )
        )

        evaluator.automated_metrics.calculate_intrinsic_metrics = AsyncMock(
            return_value={"explanation_uniqueness": 0.7, "vocabulary_diversity": 0.8}
        )

        # Measure evaluation time
        import time

        start_time = time.time()
        results = await evaluator.evaluate(large_predictions, large_ground_truth)
        end_time = time.time()

        evaluation_time = end_time - start_time

        assert isinstance(results, dict)
        assert len(results) > 0
        assert evaluation_time < 15  # Should complete within reasonable time

        print("\\n=== Performance Characteristics ===")
        print(f"Samples evaluated: {len(large_predictions)}")
        print(f"Evaluation time: {evaluation_time:.2f} seconds")
        print(f"Average time per sample: {evaluation_time / len(large_predictions):.3f} seconds")
        print(f"Metrics computed: {len(results)}")

        # Performance should be reasonable
        assert evaluation_time / len(large_predictions) < 1.0  # Less than 1 second per sample

        return results

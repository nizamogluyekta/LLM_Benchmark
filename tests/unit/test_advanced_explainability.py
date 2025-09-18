"""
Unit tests for advanced explainability analysis features.

This module tests the advanced analysis tools including pattern analysis,
clustering, model comparison, and template-based evaluation.
"""

import pytest

from benchmark.evaluation.explainability.advanced_analysis import (
    AdvancedExplainabilityAnalyzer,
    ExplanationCluster,
    ModelComparisonResult,
)
from benchmark.evaluation.explainability.explanation_templates import (
    ExplanationTemplate,
    ExplanationTemplateGenerator,
)


class TestAdvancedExplainabilityAnalyzer:
    """Test the AdvancedExplainabilityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return AdvancedExplainabilityAnalyzer()

    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions with explanations."""
        return [
            {
                "prediction": "attack",
                "explanation": "Detected malware based on suspicious file hash. The process shows network communication indicating threat.",
                "attack_type": "malware",
                "confidence": 0.95,
            },
            {
                "prediction": "attack",
                "explanation": "SQL injection attempt in login form using UNION technique. The payload contains malicious statements.",
                "attack_type": "sql_injection",
                "confidence": 0.88,
            },
            {
                "prediction": "benign",
                "explanation": "Normal user login showing expected patterns. The access time is consistent with work hours.",
                "attack_type": "benign",
                "confidence": 0.92,
            },
            {
                "prediction": "attack",
                "explanation": "DDoS attack detected via traffic flooding. Analysis shows high volume requests from multiple sources.",
                "attack_type": "dos",
                "confidence": 0.87,
            },
        ]

    @pytest.fixture
    def sample_ground_truth(self):
        """Sample ground truth data."""
        return [
            {
                "label": "attack",
                "attack_type": "malware",
                "input_text": "suspicious_file.exe network_activity",
            },
            {
                "label": "attack",
                "attack_type": "sql_injection",
                "input_text": "login_form UNION SELECT",
            },
            {"label": "benign", "attack_type": "benign", "input_text": "normal_login 9am"},
            {
                "label": "attack",
                "attack_type": "dos",
                "input_text": "high_traffic multiple_sources",
            },
        ]

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, "attack_type_keywords")
        assert "malware" in analyzer.attack_type_keywords
        assert "intrusion" in analyzer.attack_type_keywords

    def test_analyze_explanation_patterns(self, analyzer, sample_predictions, sample_ground_truth):
        """Test comprehensive pattern analysis."""
        results = analyzer.analyze_explanation_patterns(sample_predictions, sample_ground_truth)

        assert isinstance(results, dict)
        assert "attack_type_patterns" in results
        assert "explanation_clusters" in results
        assert "quality_distribution" in results
        assert "common_issues" in results
        assert "statistical_analysis" in results
        assert "analysis_metadata" in results

        # Check metadata
        metadata = results["analysis_metadata"]
        assert metadata["total_explanations_analyzed"] == 4
        assert "execution_time_seconds" in metadata
        assert "unique_explanations" in metadata

    def test_attack_type_pattern_analysis(self, analyzer, sample_predictions, sample_ground_truth):
        """Test attack type pattern analysis."""
        explanations = [pred["explanation"] for pred in sample_predictions]
        labels = [gt["label"] for gt in sample_ground_truth]

        patterns = analyzer._analyze_attack_type_patterns(explanations, labels, sample_predictions)

        assert isinstance(patterns, dict)
        # Should have analysis for malware, sql_injection, dos (attack types)
        assert len(patterns) >= 2  # At least some attack types analyzed

        for _attack_type, analysis in patterns.items():
            assert "sample_count" in analysis
            assert "avg_explanation_length" in analysis
            assert "common_phrases" in analysis
            assert "keyword_coverage" in analysis
            assert "diversity_score" in analysis
            assert "example_explanation" in analysis

    def test_explanation_clustering(self, analyzer):
        """Test explanation clustering functionality."""
        explanations = [
            "Malware detected based on file hash signature",
            "Malware identified using file hash analysis",
            "DDoS attack through network flooding",
            "SQL injection in web application form",
            "Normal user behavior during work hours",
            "Legitimate access during business hours",
        ]

        clusters = analyzer._cluster_explanations(explanations)

        assert isinstance(clusters, list)
        assert all(isinstance(cluster, ExplanationCluster) for cluster in clusters)

        # Should find some clusters
        if clusters:
            cluster = clusters[0]
            assert cluster.cluster_id >= 0
            assert isinstance(cluster.pattern_description, str)
            assert isinstance(cluster.example_explanations, list)
            assert isinstance(cluster.common_phrases, list)
            assert 0 <= cluster.quality_score <= 1
            assert cluster.frequency > 0

    def test_quality_distribution_analysis(self, analyzer, sample_predictions):
        """Test quality distribution analysis."""
        explanations = [pred["explanation"] for pred in sample_predictions]

        quality_dist = analyzer._analyze_quality_distribution(explanations, sample_predictions)

        assert isinstance(quality_dist, dict)
        assert "length_statistics" in quality_dist
        assert "completeness_analysis" in quality_dist
        assert "technical_term_analysis" in quality_dist

        # Check length statistics
        length_stats = quality_dist["length_statistics"]
        assert "mean" in length_stats
        assert "std" in length_stats
        assert "percentiles" in length_stats

        # Check completeness analysis
        completeness = quality_dist["completeness_analysis"]
        assert "complete_explanations" in completeness
        assert "incomplete_explanations" in completeness
        assert "completeness_ratio" in completeness

    def test_common_issues_identification(self, analyzer, sample_predictions):
        """Test identification of common issues."""
        explanations = [pred["explanation"] for pred in sample_predictions]

        issues = analyzer._identify_common_issues(explanations, sample_predictions)

        assert isinstance(issues, list)
        # Issues should be strings
        assert all(isinstance(issue, str) for issue in issues)

    def test_statistical_analysis(self, analyzer, sample_predictions, sample_ground_truth):
        """Test statistical analysis functionality."""
        explanations = [pred["explanation"] for pred in sample_predictions]

        stats = analyzer._perform_statistical_analysis(
            explanations, sample_predictions, sample_ground_truth
        )

        assert isinstance(stats, dict)
        assert "basic_statistics" in stats
        assert "vocabulary_analysis" in stats
        assert "length_distribution" in stats
        assert "consistency_analysis" in stats

        # Check basic statistics
        basic_stats = stats["basic_statistics"]
        assert "total_explanations" in basic_stats
        assert "valid_explanations" in basic_stats
        assert "average_word_count" in basic_stats

        # Check vocabulary analysis
        vocab_analysis = stats["vocabulary_analysis"]
        assert "total_words" in vocab_analysis
        assert "unique_words" in vocab_analysis
        assert "vocabulary_richness" in vocab_analysis

    def test_model_comparison(self, analyzer, sample_ground_truth):
        """Test model comparison functionality."""
        model_a_predictions = [
            {"explanation": "Basic malware detection using signatures", "prediction": "attack"},
            {"explanation": "SQL injection detected", "prediction": "attack"},
            {"explanation": "Normal activity", "prediction": "benign"},
        ]

        model_b_predictions = [
            {
                "explanation": "Advanced malware analysis reveals trojan with network communication and persistence mechanisms",
                "prediction": "attack",
            },
            {
                "explanation": "SQL injection attack using UNION technique targeting user database",
                "prediction": "attack",
            },
            {
                "explanation": "Legitimate user access during business hours with expected behavioral patterns",
                "prediction": "benign",
            },
        ]

        result = analyzer.compare_model_explanations(
            model_a_predictions,
            model_b_predictions,
            sample_ground_truth[:3],
            "Basic Model",
            "Advanced Model",
        )

        assert isinstance(result, ModelComparisonResult)
        assert result.model_a == "Basic Model"
        assert result.model_b == "Advanced Model"
        assert isinstance(result.quality_difference, float)
        assert isinstance(result.consistency_difference, float)
        assert isinstance(result.technical_accuracy_difference, float)
        assert result.better_model in ["Basic Model", "Advanced Model"]
        assert 0 <= result.statistical_significance <= 2

    def test_improvement_suggestions(self, analyzer, sample_predictions):
        """Test improvement suggestion generation."""
        # Create analysis results with some issues
        analysis_results = {
            "common_issues": [
                "Many explanations are too short (2 with <5 words)",
                "Explanations lack cybersecurity technical terminology",
            ],
            "quality_distribution": {
                "length_statistics": {"mean": 8},
                "completeness_analysis": {"completeness_ratio": 0.3},
            },
            "attack_type_patterns": {"malware": {"keyword_coverage": 0.2}},
            "statistical_analysis": {"vocabulary_analysis": {"vocabulary_richness": 0.2}},
        }

        suggestions = analyzer.generate_improvement_suggestions(
            sample_predictions, analysis_results
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in suggestions)

    def test_helper_methods(self, analyzer):
        """Test helper methods."""
        # Test text similarity
        similarity = analyzer._calculate_text_similarity(
            "malware detected in file", "malware found in document"
        )
        assert 0 <= similarity <= 1
        assert similarity > 0  # Should have some similarity

        # Test keyword coverage
        explanations = ["malware attack detected", "normal behavior observed"]
        keywords = ["malware", "attack", "threat"]
        coverage = analyzer._calculate_keyword_coverage(explanations, keywords)
        assert 0 <= coverage <= 1

        # Test common phrases
        texts = [
            "malware detected based on signatures",
            "malware identified using signatures",
            "virus found in system files",
        ]
        phrases = analyzer._find_common_phrases(texts)
        assert isinstance(phrases, list)

    def test_edge_cases(self, analyzer):
        """Test edge cases and error handling."""
        # Empty predictions
        empty_results = analyzer.analyze_explanation_patterns([], [])
        assert isinstance(empty_results, dict)

        # Predictions with empty explanations
        empty_explanations = [{"explanation": "", "prediction": "attack"}]
        ground_truth = [{"label": "attack"}]
        results = analyzer.analyze_explanation_patterns(empty_explanations, ground_truth)
        assert isinstance(results, dict)

        # Single prediction
        single_prediction = [{"explanation": "test explanation", "prediction": "attack"}]
        single_gt = [{"label": "attack"}]
        single_results = analyzer.analyze_explanation_patterns(single_prediction, single_gt)
        assert isinstance(single_results, dict)


class TestExplanationTemplateGenerator:
    """Test the ExplanationTemplateGenerator class."""

    @pytest.fixture
    def template_generator(self):
        """Create template generator instance for testing."""
        return ExplanationTemplateGenerator()

    @pytest.fixture
    def sample_explanation(self):
        """Sample explanation for testing."""
        return "Detected trojan malware based on suspicious file hash and network connections. The file shows encrypted communications and registry modifications which indicates high threat."

    def test_generator_initialization(self, template_generator):
        """Test template generator initialization."""
        assert template_generator is not None
        assert hasattr(template_generator, "templates")
        assert len(template_generator.templates) > 0

        # Check some expected templates
        assert "malware" in template_generator.templates
        assert "intrusion" in template_generator.templates
        assert "phishing" in template_generator.templates

    def test_get_template_for_attack(self, template_generator):
        """Test getting template for specific attack type."""
        malware_template = template_generator.generate_template_for_attack("malware")

        assert isinstance(malware_template, ExplanationTemplate)
        assert malware_template.attack_type == "malware"
        assert len(malware_template.required_elements) > 0
        assert isinstance(malware_template.template, str)
        assert isinstance(malware_template.example_explanation, str)

    def test_get_template_for_unknown_attack(self, template_generator):
        """Test getting template for unknown attack type."""
        unknown_template = template_generator.generate_template_for_attack("unknown_attack")

        assert isinstance(unknown_template, ExplanationTemplate)
        assert unknown_template.attack_type == "generic"

    def test_evaluate_explanation_against_template(self, template_generator, sample_explanation):
        """Test evaluating explanation against template."""
        result = template_generator.evaluate_explanation_against_template(
            sample_explanation, "malware"
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert "present_elements" in result
        assert "missing_elements" in result
        assert "feedback" in result
        assert "template_used" in result

        assert 0 <= result["score"] <= 1
        assert isinstance(result["present_elements"], list)
        assert isinstance(result["missing_elements"], list)
        assert isinstance(result["feedback"], str)

    def test_evaluate_explanation_strict_mode(self, template_generator, sample_explanation):
        """Test strict mode evaluation."""
        normal_result = template_generator.evaluate_explanation_against_template(
            sample_explanation, "malware", strict_mode=False
        )

        strict_result = template_generator.evaluate_explanation_against_template(
            sample_explanation, "malware", strict_mode=True
        )

        assert isinstance(normal_result, dict)
        assert isinstance(strict_result, dict)

        # Strict mode might have lower score if elements are missing
        if strict_result["missing_elements"]:
            assert strict_result["score"] <= normal_result["score"]

    def test_batch_evaluate_explanations(self, template_generator):
        """Test batch evaluation of explanations."""
        predictions = [
            {
                "explanation": "Malware detected using file signatures",
                "attack_type": "malware",
                "prediction": "attack",
            },
            {
                "explanation": "SQL injection in login form",
                "attack_type": "sql_injection",
                "prediction": "attack",
            },
            {
                "explanation": "",  # Empty explanation
                "attack_type": "dos",
                "prediction": "attack",
            },
        ]

        results = template_generator.batch_evaluate_explanations(predictions)

        assert isinstance(results, dict)
        assert "individual_evaluations" in results
        assert "summary_statistics" in results
        assert "template_usage" in results
        assert "improvement_recommendations" in results

        # Check individual evaluations
        individual_evals = results["individual_evaluations"]
        assert len(individual_evals) == 3
        assert all("score" in eval_result for eval_result in individual_evals)

        # Check summary statistics
        summary = results["summary_statistics"]
        assert "average_score" in summary
        assert "total_evaluations" in summary
        assert summary["total_evaluations"] == 3

    def test_get_all_templates(self, template_generator):
        """Test getting all templates."""
        all_templates = template_generator.get_all_templates()

        assert isinstance(all_templates, dict)
        assert len(all_templates) > 0
        assert all(isinstance(template, ExplanationTemplate) for template in all_templates.values())

    def test_add_custom_template(self, template_generator):
        """Test adding custom template."""
        custom_template = ExplanationTemplate(
            attack_type="custom_attack",
            template="Custom attack detected with {indicators}",
            required_elements=["indicators"],
            example_explanation="Custom attack detected with suspicious patterns",
        )

        initial_count = len(template_generator.templates)
        template_generator.add_custom_template(custom_template)

        assert len(template_generator.templates) == initial_count + 1
        assert "custom_attack" in template_generator.templates
        assert template_generator.templates["custom_attack"] == custom_template

    def test_get_template_statistics(self, template_generator):
        """Test getting template statistics."""
        stats = template_generator.get_template_statistics()

        assert isinstance(stats, dict)
        assert "total_templates" in stats
        assert "attack_types_covered" in stats
        assert "template_complexity" in stats
        assert "coverage_analysis" in stats

        assert stats["total_templates"] > 0
        assert len(stats["attack_types_covered"]) > 0

    def test_element_keywords(self, template_generator):
        """Test element keyword mapping."""
        malware_keywords = template_generator._get_element_keywords("malware_type")
        assert isinstance(malware_keywords, list)
        assert len(malware_keywords) > 0
        assert "malware" in malware_keywords

        # Test unknown element
        unknown_keywords = template_generator._get_element_keywords("unknown_element")
        assert isinstance(unknown_keywords, list)
        assert "unknown_element" in unknown_keywords

    def test_template_feedback_generation(self, template_generator):
        """Test template feedback generation."""
        template = template_generator.generate_template_for_attack("malware")

        feedback = template_generator._generate_template_feedback(
            present_elements=["malware_type", "indicators"],
            missing_elements=["threat_level"],
            optional_present=["file_hash"],
            template=template,
            strict_mode=False,
        )

        assert isinstance(feedback, str)
        assert len(feedback) > 0
        assert "malware_type" in feedback
        assert "threat_level" in feedback

    def test_batch_recommendations(self, template_generator):
        """Test batch recommendation generation."""
        # Create mock individual evaluations with various issues
        individual_evaluations = [
            {
                "score": 0.3,
                "missing_elements": ["threat_level", "indicators"],
                "template_used": "malware",
            },
            {"score": 0.4, "missing_elements": ["threat_level"], "template_used": "malware"},
            {"score": 0.2, "missing_elements": ["dos_method", "impact"], "template_used": "dos"},
        ]

        summary_stats = {
            "average_score": 0.3,
            "low_quality_explanations": 3,
            "total_evaluations": 3,
        }

        recommendations = template_generator._generate_batch_recommendations(
            individual_evaluations, summary_stats
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_edge_cases_template_evaluation(self, template_generator):
        """Test edge cases in template evaluation."""
        # Empty explanation
        empty_result = template_generator.evaluate_explanation_against_template("", "malware")
        assert empty_result["score"] == 0

        # Very short explanation
        short_result = template_generator.evaluate_explanation_against_template("attack", "malware")
        assert 0 <= short_result["score"] <= 1

        # Unknown attack type
        unknown_result = template_generator.evaluate_explanation_against_template(
            "some explanation", "unknown_attack_type"
        )
        assert 0 <= unknown_result["score"] <= 1

"""Unit tests for explainability evaluator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.evaluation.explainability.automated_metrics import AutomatedMetricsEvaluator
from benchmark.evaluation.explainability.llm_judge import LLMJudgeEvaluator
from benchmark.evaluation.metrics.explainability import (
    ExplainabilityEvaluator,
    ExplanationQualityScore,
)


class TestExplanationQualityScore:
    """Test ExplanationQualityScore dataclass."""

    def test_explanation_quality_score_creation(self):
        """Test creating ExplanationQualityScore object."""
        score = ExplanationQualityScore(
            overall_score=0.8,
            technical_accuracy=0.9,
            logical_consistency=0.7,
            completeness=0.8,
            clarity=0.9,
            domain_relevance=0.6,
            detailed_feedback="Good explanation with minor issues",
        )

        assert score.overall_score == 0.8
        assert score.technical_accuracy == 0.9
        assert score.logical_consistency == 0.7
        assert score.completeness == 0.8
        assert score.clarity == 0.9
        assert score.domain_relevance == 0.6
        assert score.detailed_feedback == "Good explanation with minor issues"


class TestLLMJudgeEvaluator:
    """Test LLM judge evaluator."""

    @pytest.fixture
    def llm_judge(self):
        """Create LLM judge evaluator."""
        return LLMJudgeEvaluator("gpt-4o-mini")

    def test_llm_judge_initialization(self, llm_judge):
        """Test LLM judge evaluator initialization."""
        assert llm_judge.model == "gpt-4o-mini"
        assert llm_judge.rate_limiter._value == 10

    def test_create_judge_prompt(self, llm_judge):
        """Test judge prompt creation."""
        prompt = llm_judge._create_judge_prompt(
            input_sample="Network traffic from 192.168.1.1 to 10.0.0.1",
            explanation="This traffic shows suspicious patterns indicating potential intrusion",
            prediction="attack",
            ground_truth_label="attack",
            ground_truth_explanation="The traffic contains malicious payload",
        )

        assert "INPUT DATA:" in prompt
        assert "192.168.1.1" in prompt
        assert "PREDICTED LABEL: attack" in prompt
        assert "CORRECT LABEL: attack" in prompt
        assert "suspicious patterns" in prompt
        assert "REFERENCE EXPLANATION:" in prompt
        assert "malicious payload" in prompt
        assert "JSON format" in prompt

    def test_create_judge_prompt_without_reference(self, llm_judge):
        """Test judge prompt creation without reference explanation."""
        prompt = llm_judge._create_judge_prompt(
            input_sample="Normal network traffic",
            explanation="This appears to be legitimate traffic",
            prediction="benign",
            ground_truth_label="benign",
        )

        assert "INPUT DATA:" in prompt
        assert "PREDICTED LABEL: benign" in prompt
        assert "REFERENCE EXPLANATION:" not in prompt

    def test_parse_judge_response_json(self, llm_judge):
        """Test parsing valid JSON response."""
        response_text = """{
            "technical_accuracy": 0.9,
            "logical_consistency": 0.8,
            "completeness": 0.7,
            "clarity": 0.9,
            "domain_relevance": 0.8,
            "overall_score": 0.84,
            "feedback": "Well-structured explanation with good technical detail"
        }"""

        score = llm_judge._parse_judge_response(response_text)

        assert isinstance(score, ExplanationQualityScore)
        assert score.technical_accuracy == 0.9
        assert score.logical_consistency == 0.8
        assert score.completeness == 0.7
        assert score.clarity == 0.9
        assert score.domain_relevance == 0.8
        assert score.overall_score == 0.84
        assert "Well-structured" in score.detailed_feedback

    def test_parse_judge_response_text_extraction(self, llm_judge):
        """Test parsing response with text-based scores."""
        response_text = """
        Technical accuracy: 0.8
        Logical consistency: 0.9
        Completeness: 0.7
        Clarity: 0.6
        Domain relevance: 0.8
        Overall: 0.76
        """

        score = llm_judge._parse_judge_response(response_text)

        assert isinstance(score, ExplanationQualityScore)
        assert score.technical_accuracy == 0.8
        assert score.logical_consistency == 0.9
        assert score.completeness == 0.7
        assert score.clarity == 0.6
        assert score.domain_relevance == 0.8
        assert score.overall_score == 0.76

    def test_parse_judge_response_fallback(self, llm_judge):
        """Test parsing response with no scores found."""
        response_text = "This is a general response with no specific scores."

        score = llm_judge._parse_judge_response(response_text)

        assert isinstance(score, ExplanationQualityScore)
        assert score.technical_accuracy == 0.5
        assert score.logical_consistency == 0.5
        assert score.completeness == 0.5
        assert score.clarity == 0.5
        assert score.domain_relevance == 0.5
        assert score.overall_score == 0.5

    @pytest.mark.asyncio
    async def test_judge_explanation_success(self, llm_judge):
        """Test successful explanation judging."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """{
            "technical_accuracy": 0.9,
            "logical_consistency": 0.8,
            "completeness": 0.7,
            "clarity": 0.9,
            "domain_relevance": 0.8,
            "overall_score": 0.84,
            "feedback": "Good explanation"
        }"""

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            # Reset the client to force re-initialization
            llm_judge._client = None

            score = await llm_judge.judge_explanation(
                input_sample="Test input",
                explanation="Test explanation",
                prediction="attack",
                ground_truth_label="attack",
            )

            assert isinstance(score, ExplanationQualityScore)
            assert score.overall_score == 0.84
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_judge_explanation_api_failure(self, llm_judge):
        """Test explanation judging with API failure."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            # Reset the client to force re-initialization
            llm_judge._client = None

            score = await llm_judge.judge_explanation(
                input_sample="Test input",
                explanation="Test explanation",
                prediction="attack",
                ground_truth_label="attack",
            )

            assert isinstance(score, ExplanationQualityScore)
            assert score.overall_score == 0.5
            assert "Evaluation failed" in score.detailed_feedback


class TestAutomatedMetricsEvaluator:
    """Test automated metrics evaluator."""

    @pytest.fixture
    def metrics_evaluator(self):
        """Create automated metrics evaluator."""
        return AutomatedMetricsEvaluator()

    def test_metrics_evaluator_initialization(self, metrics_evaluator):
        """Test metrics evaluator initialization."""
        assert hasattr(metrics_evaluator, "nltk_available")
        assert hasattr(metrics_evaluator, "bert_score_available")

    @pytest.mark.asyncio
    async def test_calculate_metrics_empty_input(self, metrics_evaluator):
        """Test calculate metrics with empty input."""
        result = await metrics_evaluator.calculate_metrics([], [])
        assert result == {}

    @pytest.mark.asyncio
    async def test_calculate_intrinsic_metrics_empty(self, metrics_evaluator):
        """Test intrinsic metrics with empty input."""
        result = await metrics_evaluator.calculate_intrinsic_metrics([])
        assert result == {}

    def test_calculate_diversity_metrics(self, metrics_evaluator):
        """Test diversity metrics calculation."""
        explanations = [
            "This is a malicious attack attempt",
            "Normal network traffic observed",
            "This is a malicious attack attempt",  # Duplicate
            "Suspicious behavior detected",
        ]

        result = metrics_evaluator._calculate_diversity_metrics(explanations)

        assert "explanation_uniqueness" in result
        assert "vocabulary_diversity" in result
        assert 0 <= result["explanation_uniqueness"] <= 1
        assert 0 <= result["vocabulary_diversity"] <= 1

    def test_calculate_structural_metrics(self, metrics_evaluator):
        """Test structural metrics calculation."""
        explanations = [
            "This is a short explanation.",
            "This is a much longer explanation with more detailed information about the attack.",
            "Medium length explanation here.",
        ]

        result = metrics_evaluator._calculate_structural_metrics(explanations)

        assert "avg_explanation_length" in result
        assert "avg_sentences_per_explanation" in result
        assert "explanation_complexity" in result
        assert result["avg_explanation_length"] > 0
        assert result["avg_sentences_per_explanation"] > 0
        assert 0 <= result["explanation_complexity"] <= 1

    @pytest.mark.asyncio
    async def test_calculate_bleu_scores_mock(self, metrics_evaluator):
        """Test BLEU score calculation with mocked NLTK."""
        candidates = ["This is a test explanation"]
        references = ["This is a reference explanation"]

        # Mock NLTK availability
        metrics_evaluator.nltk_available = True

        with (
            patch("nltk.translate.bleu_score.sentence_bleu") as mock_bleu,
            patch("nltk.translate.bleu_score.SmoothingFunction"),
        ):
            mock_bleu.return_value = 0.5

            result = await metrics_evaluator._calculate_bleu_scores(candidates, references)

            assert "bleu_score" in result
            assert result["bleu_score"] == 0.5

    @pytest.mark.asyncio
    async def test_calculate_rouge_scores_mock(self, metrics_evaluator):
        """Test ROUGE score calculation with mocked library."""
        candidates = ["This is a test explanation"]
        references = ["This is a reference explanation"]

        # Mock ROUGE availability
        metrics_evaluator.nltk_available = True

        mock_scores = {
            "rouge1": MagicMock(fmeasure=0.6),
            "rouge2": MagicMock(fmeasure=0.4),
            "rougeL": MagicMock(fmeasure=0.5),
        }

        with patch("rouge_score.rouge_scorer") as mock_rouge:
            mock_scorer = MagicMock()
            mock_scorer.score.return_value = mock_scores
            mock_rouge.RougeScorer.return_value = mock_scorer

            result = await metrics_evaluator._calculate_rouge_scores(candidates, references)

            assert "rouge1_score" in result
            assert "rouge2_score" in result
            assert "rouge_l_score" in result
            assert result["rouge1_score"] == 0.6
            assert result["rouge2_score"] == 0.4
            assert result["rouge_l_score"] == 0.5

    @pytest.mark.asyncio
    async def test_calculate_bert_scores_mock(self, metrics_evaluator):
        """Test BERTScore calculation with mocked library."""
        candidates = ["This is a test explanation"]
        references = ["This is a reference explanation"]

        # Mock BERTScore availability
        metrics_evaluator.bert_score_available = True

        mock_f1 = MagicMock()
        mock_f1.mean.return_value.item.return_value = 0.75

        with patch("bert_score.score") as mock_bert:
            mock_bert.return_value = (None, None, mock_f1)

            result = await metrics_evaluator._calculate_bert_scores(candidates, references)

            assert "bert_score" in result
            assert result["bert_score"] == 0.75


class TestExplainabilityEvaluator:
    """Test main explainability evaluator."""

    @pytest.fixture
    def explainability_evaluator(self):
        """Create explainability evaluator."""
        return ExplainabilityEvaluator("gpt-4o-mini")

    def test_explainability_evaluator_initialization(self, explainability_evaluator):
        """Test explainability evaluator initialization."""
        assert explainability_evaluator.llm_judge.model == "gpt-4o-mini"
        assert isinstance(explainability_evaluator.llm_judge, LLMJudgeEvaluator)
        assert isinstance(explainability_evaluator.automated_metrics, AutomatedMetricsEvaluator)
        assert len(explainability_evaluator.metric_names) > 0

    def test_extract_explanations(self, explainability_evaluator):
        """Test explanation extraction from predictions."""
        predictions = [
            {"explanation": "This is an attack", "prediction": "attack"},
            {"reasoning": "Normal traffic", "prediction": "benign"},
            {"rationale": "Suspicious activity", "prediction": "attack"},
            {"prediction": "benign"},  # No explanation
        ]

        explanations = explainability_evaluator._extract_explanations(predictions)

        assert len(explanations) == 4
        assert explanations[0] == "This is an attack"
        assert explanations[1] == "Normal traffic"
        assert explanations[2] == "Suspicious activity"
        assert explanations[3] == ""

    def test_calculate_explanation_consistency(self, explainability_evaluator):
        """Test explanation consistency calculation."""
        predictions = [
            {"explanation": "This shows malicious attack patterns", "prediction": "attack"},
            {"explanation": "Normal legitimate traffic", "prediction": "benign"},
            {"explanation": "Suspicious intrusion detected", "prediction": "attack"},
            {"explanation": "Safe and expected behavior", "prediction": "benign"},
        ]
        ground_truth = [
            {"label": "attack"},
            {"label": "benign"},
            {"label": "attack"},
            {"label": "benign"},
        ]

        consistency = explainability_evaluator._calculate_explanation_consistency(
            predictions, ground_truth
        )

        assert 0 <= consistency <= 1
        assert consistency > 0.5  # Should be reasonably consistent

    def test_analyze_ioc_accuracy(self, explainability_evaluator):
        """Test IoC accuracy analysis."""
        predictions = [
            {
                "explanation": "Traffic from 192.168.1.1 contains malicious hash abc123def456",
                "prediction": "attack",
            },
            {
                "explanation": "Connection to malicious.example.com detected",
                "prediction": "attack",
            },
        ]
        ground_truth = [
            {"input_text": "Source: 192.168.1.1, Hash: abc123def456", "label": "attack"},
            {"input_text": "Domain: malicious.example.com accessed", "label": "attack"},
        ]

        ioc_accuracy = explainability_evaluator._analyze_ioc_accuracy(predictions, ground_truth)

        assert 0 <= ioc_accuracy <= 1

    def test_analyze_mitre_coverage(self, explainability_evaluator):
        """Test MITRE ATT&CK coverage analysis."""
        explanations = [
            "This shows signs of reconnaissance and initial access attempts",
            "Evidence of privilege escalation and lateral movement",
            "Normal traffic with no suspicious patterns",
            "Phishing attempt detected with command and control communication",
        ]

        coverage = explainability_evaluator._analyze_mitre_coverage(explanations)

        assert 0 <= coverage <= 1
        assert coverage > 0  # Should detect some techniques

    @pytest.mark.asyncio
    async def test_evaluate_no_explanations(self, explainability_evaluator):
        """Test evaluation with no explanations."""
        predictions = [{"prediction": "attack"}, {"prediction": "benign"}]
        ground_truth = [{"label": "attack"}, {"label": "benign"}]

        with pytest.raises(ValueError, match="Missing required prediction field 'explanation'"):
            await explainability_evaluator.evaluate(predictions, ground_truth)

    @pytest.mark.asyncio
    async def test_evaluate_with_explanations(self, explainability_evaluator):
        """Test full evaluation with explanations."""
        predictions = [
            {"explanation": "Malicious attack detected", "prediction": "attack"},
            {"explanation": "Normal traffic observed", "prediction": "benign"},
        ]
        ground_truth = [
            {"label": "attack", "input_text": "Attack traffic"},
            {"label": "benign", "input_text": "Normal traffic"},
        ]

        # Mock the sub-evaluators to avoid external dependencies
        explainability_evaluator.llm_judge.judge_explanation = AsyncMock(
            return_value=ExplanationQualityScore(
                overall_score=0.8,
                technical_accuracy=0.9,
                logical_consistency=0.8,
                completeness=0.7,
                clarity=0.9,
                domain_relevance=0.8,
                detailed_feedback="Good explanation",
            )
        )

        explainability_evaluator.automated_metrics.calculate_intrinsic_metrics = AsyncMock(
            return_value={"explanation_uniqueness": 0.8, "vocabulary_diversity": 0.7}
        )

        result = await explainability_evaluator.evaluate(predictions, ground_truth)

        assert isinstance(result, dict)
        assert len(result) > 0
        # Should contain domain-specific metrics
        assert "explanation_consistency" in result
        assert "explanation_length_avg" in result

    def test_get_metric_names(self, explainability_evaluator):
        """Test getting metric names."""
        metric_names = explainability_evaluator.get_metric_names()

        assert isinstance(metric_names, list)
        assert len(metric_names) > 0
        assert "avg_explanation_quality" in metric_names
        assert "technical_accuracy_score" in metric_names

    def test_get_required_fields(self, explainability_evaluator):
        """Test getting required fields."""
        pred_fields = explainability_evaluator.get_required_prediction_fields()
        gt_fields = explainability_evaluator.get_required_ground_truth_fields()

        assert "explanation" in pred_fields
        assert isinstance(gt_fields, list)  # May be empty

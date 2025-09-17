"""
Automated metrics for explanation evaluation.

This module provides automated metrics such as BLEU, ROUGE, and BERTScore
for evaluating explanation quality against reference explanations.
"""

from benchmark.core.logging import get_logger


class AutomatedMetricsEvaluator:
    """Automated metrics for explanation evaluation."""

    def __init__(self) -> None:
        """Initialize automated metrics evaluator."""
        self.logger = get_logger("automated_metrics_evaluator")

        # Track availability of external libraries
        self.nltk_available = False
        self.bert_score_available = False

        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check availability of external libraries."""
        try:
            import nltk  # noqa: F401
            from rouge_score import rouge_scorer  # noqa: F401

            self.nltk_available = True
            self.logger.debug("NLTK and ROUGE libraries available")
        except ImportError:
            self.logger.warning("NLTK or ROUGE not available for BLEU/ROUGE scores")

        try:
            from bert_score import score as bert_score  # noqa: F401

            self.bert_score_available = True
            self.logger.debug("BERTScore library available")
        except ImportError:
            self.logger.warning("BERTScore not available")

    async def calculate_metrics(
        self, candidate_explanations: list[str], reference_explanations: list[str]
    ) -> dict[str, float]:
        """
        Calculate automated metrics comparing candidates to references.

        Args:
            candidate_explanations: List of candidate explanation strings
            reference_explanations: List of reference explanation strings

        Returns:
            Dictionary of computed metrics
        """
        metrics: dict[str, float] = {}

        if not candidate_explanations or not reference_explanations:
            self.logger.warning("Empty explanations provided to automated metrics")
            return metrics

        if len(candidate_explanations) != len(reference_explanations):
            self.logger.warning(
                f"Mismatched lengths: {len(candidate_explanations)} candidates vs {len(reference_explanations)} references"
            )
            # Take minimum length to avoid errors
            min_length = min(len(candidate_explanations), len(reference_explanations))
            candidate_explanations = candidate_explanations[:min_length]
            reference_explanations = reference_explanations[:min_length]

        # Calculate BLEU scores
        if self.nltk_available:
            bleu_scores = await self._calculate_bleu_scores(
                candidate_explanations, reference_explanations
            )
            metrics.update(bleu_scores)

        # Calculate ROUGE scores
        if self.nltk_available:
            rouge_scores = await self._calculate_rouge_scores(
                candidate_explanations, reference_explanations
            )
            metrics.update(rouge_scores)

        # Calculate BERTScore
        if self.bert_score_available:
            bert_scores = await self._calculate_bert_scores(
                candidate_explanations, reference_explanations
            )
            metrics.update(bert_scores)

        return metrics

    async def calculate_intrinsic_metrics(self, explanations: list[str]) -> dict[str, float]:
        """
        Calculate intrinsic metrics that don't require reference explanations.

        Args:
            explanations: List of explanation strings

        Returns:
            Dictionary of intrinsic metrics
        """
        metrics: dict[str, float] = {}

        if not explanations:
            return metrics

        # Calculate diversity metrics
        metrics.update(self._calculate_diversity_metrics(explanations))

        # Calculate length and structure metrics
        metrics.update(self._calculate_structural_metrics(explanations))

        return metrics

    async def _calculate_bleu_scores(
        self, candidates: list[str], references: list[str]
    ) -> dict[str, float]:
        """
        Calculate BLEU scores.

        Args:
            candidates: List of candidate explanations
            references: List of reference explanations

        Returns:
            Dictionary with BLEU metrics
        """
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            bleu_scores = []
            smoothing = SmoothingFunction()

            for candidate, reference in zip(candidates, references, strict=False):
                if not candidate.strip() or not reference.strip():
                    bleu_scores.append(0.0)
                    continue

                # Tokenize
                candidate_tokens = candidate.lower().split()
                reference_tokens = [reference.lower().split()]

                # Calculate BLEU score
                score = sentence_bleu(
                    reference_tokens, candidate_tokens, smoothing_function=smoothing.method1
                )
                bleu_scores.append(score)

            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

            return {"bleu_score": avg_bleu}

        except Exception as e:
            self.logger.error(f"BLEU calculation failed: {e}")
            return {}

    async def _calculate_rouge_scores(
        self, candidates: list[str], references: list[str]
    ) -> dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            candidates: List of candidate explanations
            references: List of reference explanations

        Returns:
            Dictionary with ROUGE metrics
        """
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for candidate, reference in zip(candidates, references, strict=False):
                if not candidate.strip() or not reference.strip():
                    rouge1_scores.append(0.0)
                    rouge2_scores.append(0.0)
                    rougeL_scores.append(0.0)
                    continue

                scores = scorer.score(reference, candidate)
                rouge1_scores.append(scores["rouge1"].fmeasure)
                rouge2_scores.append(scores["rouge2"].fmeasure)
                rougeL_scores.append(scores["rougeL"].fmeasure)

            return {
                "rouge1_score": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
                "rouge2_score": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
                "rouge_l_score": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
            }

        except Exception as e:
            self.logger.error(f"ROUGE calculation failed: {e}")
            return {}

    async def _calculate_bert_scores(
        self, candidates: list[str], references: list[str]
    ) -> dict[str, float]:
        """
        Calculate BERTScore.

        Args:
            candidates: List of candidate explanations
            references: List of reference explanations

        Returns:
            Dictionary with BERTScore metrics
        """
        try:
            from bert_score import score

            # Filter out empty strings
            valid_pairs = [
                (c, r)
                for c, r in zip(candidates, references, strict=False)
                if c.strip() and r.strip()
            ]

            if not valid_pairs:
                return {"bert_score": 0.0}

            valid_candidates = [pair[0] for pair in valid_pairs]
            valid_references = [pair[1] for pair in valid_pairs]

            # Calculate BERTScore
            P, R, F1 = score(valid_candidates, valid_references, lang="en", verbose=False)

            avg_bert_score = F1.mean().item()

            return {"bert_score": avg_bert_score}

        except Exception as e:
            self.logger.error(f"BERTScore calculation failed: {e}")
            return {}

    def _calculate_diversity_metrics(self, explanations: list[str]) -> dict[str, float]:
        """
        Calculate diversity metrics for explanations.

        Args:
            explanations: List of explanation strings

        Returns:
            Dictionary with diversity metrics
        """
        if not explanations:
            return {}

        # Remove empty explanations
        valid_explanations = [exp.strip().lower() for exp in explanations if exp.strip()]

        if not valid_explanations:
            return {}

        # Calculate uniqueness ratio
        unique_explanations = set(valid_explanations)
        uniqueness_ratio = len(unique_explanations) / len(valid_explanations)

        # Calculate vocabulary diversity
        all_words = []
        for explanation in valid_explanations:
            all_words.extend(explanation.split())

        if all_words:
            unique_words = set(all_words)
            vocabulary_diversity = len(unique_words) / len(all_words)
        else:
            vocabulary_diversity = 0.0

        return {
            "explanation_uniqueness": uniqueness_ratio,
            "vocabulary_diversity": vocabulary_diversity,
        }

    def _calculate_structural_metrics(self, explanations: list[str]) -> dict[str, float]:
        """
        Calculate structural metrics for explanations.

        Args:
            explanations: List of explanation strings

        Returns:
            Dictionary with structural metrics
        """
        if not explanations:
            return {}

        valid_explanations = [exp.strip() for exp in explanations if exp.strip()]

        if not valid_explanations:
            return {}

        # Calculate average length in words
        word_counts = [len(exp.split()) for exp in valid_explanations]
        avg_length = sum(word_counts) / len(word_counts)

        # Calculate average sentence count
        sentence_counts = [
            len([s for s in exp.split(".") if s.strip()]) for exp in valid_explanations
        ]
        avg_sentences = sum(sentence_counts) / len(sentence_counts)

        # Calculate complexity score (based on average word length)
        word_lengths = []
        for exp in valid_explanations:
            words = exp.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                word_lengths.append(avg_word_length)

        complexity_score = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0

        return {
            "avg_explanation_length": avg_length,
            "avg_sentences_per_explanation": avg_sentences,
            "explanation_complexity": complexity_score / 10.0,  # Normalize to 0-1 range
        }

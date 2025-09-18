"""
Advanced analysis features for explainability evaluation.

This module provides sophisticated analysis tools for explanation quality,
pattern recognition, clustering, and comparative analysis across models.
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark.core.logging import get_logger


@dataclass
class ExplanationCluster:
    """Container for explanation clustering results."""

    cluster_id: int
    pattern_description: str
    example_explanations: list[str]
    common_phrases: list[str]
    quality_score: float
    frequency: int


@dataclass
class ModelComparisonResult:
    """Container for model comparison analysis results."""

    model_a: str
    model_b: str
    quality_difference: float
    consistency_difference: float
    technical_accuracy_difference: float
    better_model: str
    statistical_significance: float


class AdvancedExplainabilityAnalyzer:
    """Advanced analysis tools for explainability evaluation."""

    def __init__(self) -> None:
        """Initialize the advanced explainability analyzer."""
        self.logger = get_logger("advanced_explainability_analyzer")

        # Attack type keyword mappings for domain-specific analysis
        self.attack_type_keywords = {
            "malware": ["malware", "virus", "trojan", "ransomware", "backdoor", "rootkit"],
            "intrusion": ["intrusion", "penetration", "unauthorized", "breach", "infiltration"],
            "dos": ["denial of service", "dos", "ddos", "flooding", "overload"],
            "phishing": ["phishing", "spoofing", "social engineering", "credential theft"],
            "reconnaissance": ["reconnaissance", "scanning", "enumeration", "discovery"],
            "data_exfiltration": [
                "exfiltration",
                "data theft",
                "data extraction",
                "information gathering",
            ],
        }

    def analyze_explanation_patterns(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze patterns in explanations across the dataset.

        Args:
            predictions: List of prediction dictionaries with explanations
            ground_truth: List of ground truth dictionaries

        Returns:
            Dictionary containing comprehensive pattern analysis results
        """
        self.logger.info("Starting comprehensive explanation pattern analysis")
        start_time = time.time()

        explanations = [pred.get("explanation", "") for pred in predictions]
        labels = [gt.get("label", "") for gt in ground_truth]

        analysis_results: dict[str, Any] = {}

        # Pattern analysis by attack type
        self.logger.debug("Analyzing attack type patterns")
        analysis_results["attack_type_patterns"] = self._analyze_attack_type_patterns(
            explanations, labels, predictions
        )

        # Clustering analysis
        self.logger.debug("Performing explanation clustering")
        clusters = self._cluster_explanations(explanations)
        analysis_results["explanation_clusters"] = [cluster.__dict__ for cluster in clusters]

        # Quality distribution analysis
        self.logger.debug("Analyzing quality distribution")
        analysis_results["quality_distribution"] = self._analyze_quality_distribution(
            explanations, predictions
        )

        # Common issues identification
        self.logger.debug("Identifying common issues")
        analysis_results["common_issues"] = self._identify_common_issues(explanations, predictions)

        # Statistical analysis
        self.logger.debug("Performing statistical analysis")
        analysis_results["statistical_analysis"] = self._perform_statistical_analysis(
            explanations, predictions, ground_truth
        )

        execution_time = time.time() - start_time
        analysis_results["analysis_metadata"] = {
            "execution_time_seconds": execution_time,
            "total_explanations_analyzed": len(explanations),
            "unique_explanations": len(
                {exp.strip().lower() for exp in explanations if exp.strip()}
            ),
        }

        self.logger.info(f"Pattern analysis completed in {execution_time:.2f}s")
        return analysis_results

    def _analyze_attack_type_patterns(
        self, explanations: list[str], labels: list[str], predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze explanation patterns for different attack types."""
        attack_type_analysis = {}

        # Group by attack type
        attack_groups = defaultdict(list)
        for _i, (exp, label, pred) in enumerate(
            zip(explanations, labels, predictions, strict=False)
        ):
            attack_type = pred.get("attack_type", "unknown")
            if label.lower() in ["attack", "malicious"] and attack_type != "unknown":
                attack_groups[attack_type].append(exp)

        # Analyze each attack type
        for attack_type, type_explanations in attack_groups.items():
            if len(type_explanations) < 1:  # Need at least 1 example
                continue

            # Find common phrases
            common_phrases = self._find_common_phrases(type_explanations)

            # Calculate average quality metrics
            avg_length = np.mean([len(exp.split()) for exp in type_explanations])

            # Check keyword coverage
            relevant_keywords = self.attack_type_keywords.get(attack_type, [])
            keyword_coverage = self._calculate_keyword_coverage(
                type_explanations, relevant_keywords
            )

            # Calculate diversity within attack type
            diversity_score = self._calculate_explanation_diversity(type_explanations)

            attack_type_analysis[attack_type] = {
                "sample_count": len(type_explanations),
                "avg_explanation_length": avg_length,
                "common_phrases": common_phrases,
                "keyword_coverage": keyword_coverage,
                "diversity_score": diversity_score,
                "example_explanation": type_explanations[0] if type_explanations else "",
                "quality_indicators": self._analyze_quality_indicators(type_explanations),
            }

        return attack_type_analysis

    def _cluster_explanations(self, explanations: list[str]) -> list[ExplanationCluster]:
        """Cluster explanations by similarity using multiple criteria."""
        clusters: list[ExplanationCluster] = []
        processed_indices = set()

        # Filter out empty explanations
        valid_explanations = [(i, exp) for i, exp in enumerate(explanations) if exp.strip()]

        for i, (orig_idx, explanation) in enumerate(valid_explanations):
            if orig_idx in processed_indices:
                continue

            # Find similar explanations
            cluster_explanations = [explanation]
            cluster_indices = {orig_idx}

            for _j, (other_orig_idx, other_explanation) in enumerate(
                valid_explanations[i + 1 :], start=i + 1
            ):
                if other_orig_idx in processed_indices:
                    continue

                similarity = self._calculate_text_similarity(explanation, other_explanation)
                if similarity > 0.6:  # Similarity threshold
                    cluster_explanations.append(other_explanation)
                    cluster_indices.add(other_orig_idx)

            if len(cluster_explanations) >= 2:  # Only create clusters with multiple items
                processed_indices.update(cluster_indices)

                # Generate cluster description
                common_phrases = self._find_common_phrases(cluster_explanations)
                pattern_description = self._generate_cluster_description(
                    cluster_explanations, common_phrases
                )

                cluster = ExplanationCluster(
                    cluster_id=len(clusters),
                    pattern_description=pattern_description,
                    example_explanations=cluster_explanations[:3],  # Top 3 examples
                    common_phrases=common_phrases,
                    quality_score=self._estimate_cluster_quality(cluster_explanations),
                    frequency=len(cluster_explanations),
                )

                clusters.append(cluster)

        # Sort by frequency and quality
        clusters.sort(key=lambda c: (c.frequency, c.quality_score), reverse=True)

        return clusters[:10]  # Return top 10 clusters

    def _analyze_quality_distribution(
        self, explanations: list[str], predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze distribution of explanation quality metrics."""
        valid_explanations = [exp for exp in explanations if exp.strip()]
        lengths = [len(exp.split()) for exp in valid_explanations]

        quality_distribution = {
            "length_statistics": {
                "mean": np.mean(lengths) if lengths else 0,
                "std": np.std(lengths) if lengths else 0,
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "percentiles": {
                    "25": np.percentile(lengths, 25) if lengths else 0,
                    "50": np.percentile(lengths, 50) if lengths else 0,
                    "75": np.percentile(lengths, 75) if lengths else 0,
                    "90": np.percentile(lengths, 90) if lengths else 0,
                },
            }
        }

        # Analyze explanation completeness
        complete_explanations = 0
        incomplete_explanations = 0

        completeness_indicators = [
            "because",
            "due to",
            "indicates",
            "shows",
            "evidence",
            "demonstrates",
        ]

        for explanation in explanations:
            if not explanation.strip():
                incomplete_explanations += 1
                continue

            # Check for completeness indicators
            if any(indicator in explanation.lower() for indicator in completeness_indicators):
                complete_explanations += 1
            else:
                incomplete_explanations += 1

        completeness_data: dict[str, Any] = {
            "complete_explanations": complete_explanations,
            "incomplete_explanations": incomplete_explanations,
            "completeness_ratio": (
                complete_explanations / (complete_explanations + incomplete_explanations)
                if (complete_explanations + incomplete_explanations) > 0
                else 0.0
            ),
        }
        quality_distribution["completeness_analysis"] = completeness_data

        # Analyze technical term usage
        technical_terms = [
            "malware",
            "virus",
            "trojan",
            "ransomware",
            "intrusion",
            "vulnerability",
            "exploit",
            "payload",
            "backdoor",
            "phishing",
            "dos",
            "ddos",
            "injection",
            "buffer overflow",
            "privilege escalation",
            "lateral movement",
            "reconnaissance",
        ]

        technical_usage = self._analyze_technical_term_usage(valid_explanations, technical_terms)
        quality_distribution["technical_term_analysis"] = technical_usage

        return quality_distribution

    def _identify_common_issues(
        self, explanations: list[str], predictions: list[dict[str, Any]]
    ) -> list[str]:
        """Identify common issues in explanations."""
        issues = []

        # Check for empty explanations
        empty_count = sum(1 for exp in explanations if not exp.strip())
        if empty_count > len(explanations) * 0.1:  # More than 10% empty
            issues.append(f"High number of empty explanations ({empty_count}/{len(explanations)})")

        # Check for very short explanations
        short_count = sum(1 for exp in explanations if len(exp.split()) < 5)
        if short_count > len(explanations) * 0.2:  # More than 20% very short
            issues.append(f"Many explanations are too short ({short_count} with <5 words)")

        # Check for repetitive explanations
        unique_explanations = {exp.strip().lower() for exp in explanations if exp.strip()}
        if len(unique_explanations) < len(explanations) * 0.5:  # Less than 50% unique
            issues.append("Low explanation diversity - many explanations are repetitive")

        # Check for lack of technical terms
        technical_terms = ["malware", "intrusion", "vulnerability", "exploit", "attack", "threat"]
        explanations_with_tech_terms = sum(
            1 for exp in explanations if any(term in exp.lower() for term in technical_terms)
        )

        if (
            explanations_with_tech_terms < len(explanations) * 0.3
        ):  # Less than 30% with technical terms
            issues.append("Explanations lack cybersecurity technical terminology")

        # Check for vague explanations
        vague_indicators = ["suspicious", "unusual", "abnormal", "strange"]
        specific_indicators = ["ip address", "hash", "port", "protocol", "signature"]

        vague_count = 0
        for exp in explanations:
            if not exp.strip():
                continue
            exp_lower = exp.lower()
            has_vague = any(indicator in exp_lower for indicator in vague_indicators)
            has_specific = any(indicator in exp_lower for indicator in specific_indicators)

            if has_vague and not has_specific:
                vague_count += 1

        if vague_count > len(explanations) * 0.4:  # More than 40% vague
            issues.append("Many explanations are vague without specific technical details")

        return issues

    def _perform_statistical_analysis(
        self,
        explanations: list[str],
        predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform statistical analysis on explanation patterns."""
        valid_explanations = [exp for exp in explanations if exp.strip()]

        statistical_analysis: dict[str, Any] = {
            "basic_statistics": {
                "total_explanations": len(explanations),
                "valid_explanations": len(valid_explanations),
                "empty_explanations": len(explanations) - len(valid_explanations),
                "average_word_count": float(
                    np.mean([len(exp.split()) for exp in valid_explanations])
                )
                if valid_explanations
                else 0.0,
            }
        }

        # Vocabulary analysis
        all_words = []
        for exp in valid_explanations:
            all_words.extend(exp.lower().split())

        word_freq = Counter(all_words)
        vocab_analysis: dict[str, Any] = {
            "total_words": len(all_words),
            "unique_words": len(word_freq),
            "vocabulary_richness": len(word_freq) / len(all_words) if all_words else 0.0,
            "most_common_words": word_freq.most_common(10),
        }
        statistical_analysis["vocabulary_analysis"] = vocab_analysis

        # Explanation length distribution
        lengths = [len(exp.split()) for exp in valid_explanations]
        if lengths:
            float_lengths = [float(length) for length in lengths]
            length_dist: dict[str, Any] = {
                "mean": float(np.mean(float_lengths)),
                "median": float(np.median(float_lengths)),
                "std_dev": float(np.std(float_lengths)),
                "skewness": self._calculate_skewness(float_lengths),
                "kurtosis": self._calculate_kurtosis(float_lengths),
            }
            statistical_analysis["length_distribution"] = length_dist

        # Consistency analysis
        consistency_analysis = self._analyze_prediction_explanation_consistency(
            predictions, ground_truth
        )
        statistical_analysis["consistency_analysis"] = consistency_analysis

        return statistical_analysis

    def compare_model_explanations(
        self,
        model_a_predictions: list[dict[str, Any]],
        model_b_predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> ModelComparisonResult:
        """Compare explanation quality between two models."""
        self.logger.info(f"Comparing explanations between {model_a_name} and {model_b_name}")

        # Extract explanations
        explanations_a = [pred.get("explanation", "") for pred in model_a_predictions]
        explanations_b = [pred.get("explanation", "") for pred in model_b_predictions]

        # Calculate quality metrics for each model
        quality_a = self._calculate_avg_explanation_quality(explanations_a)
        quality_b = self._calculate_avg_explanation_quality(explanations_b)

        # Calculate consistency
        consistency_a = self._calculate_explanation_consistency(model_a_predictions, ground_truth)
        consistency_b = self._calculate_explanation_consistency(model_b_predictions, ground_truth)

        # Technical accuracy
        tech_accuracy_a = self._estimate_technical_accuracy(explanations_a)
        tech_accuracy_b = self._estimate_technical_accuracy(explanations_b)

        # Determine better model
        quality_diff = quality_b - quality_a
        consistency_diff = consistency_b - consistency_a
        tech_diff = tech_accuracy_b - tech_accuracy_a

        # Weighted score (quality: 40%, consistency: 30%, technical: 30%)
        overall_diff = 0.4 * quality_diff + 0.3 * consistency_diff + 0.3 * tech_diff

        better_model = model_b_name if overall_diff > 0 else model_a_name

        # Statistical significance (simplified)
        significance = abs(overall_diff)

        self.logger.info(f"Model comparison completed: {better_model} performs better")

        return ModelComparisonResult(
            model_a=model_a_name,
            model_b=model_b_name,
            quality_difference=quality_diff,
            consistency_difference=consistency_diff,
            technical_accuracy_difference=tech_diff,
            better_model=better_model,
            statistical_significance=significance,
        )

    def generate_improvement_suggestions(
        self, predictions: list[dict[str, Any]], analysis_results: dict[str, Any]
    ) -> list[str]:
        """Generate actionable suggestions for improving explanation quality."""
        suggestions = []

        # Analyze common issues
        common_issues = analysis_results.get("common_issues", [])

        for issue in common_issues:
            if "empty explanations" in issue:
                suggestions.append("Ensure all models provide explanations for their predictions")
            elif "too short" in issue:
                suggestions.append("Encourage more detailed explanations with specific reasoning")
            elif "repetitive" in issue:
                suggestions.append(
                    "Improve explanation diversity by incorporating more contextual details"
                )
            elif "technical terminology" in issue:
                suggestions.append(
                    "Include more cybersecurity-specific technical terms and concepts"
                )
            elif "vague" in issue:
                suggestions.append(
                    "Replace vague terms with specific technical details (IP addresses, ports, protocols)"
                )

        # Analyze quality distribution
        quality_dist = analysis_results.get("quality_distribution", {})
        length_stats = quality_dist.get("length_statistics", {})

        if length_stats.get("mean", 0) < 10:  # Very short explanations
            suggestions.append(
                "Increase average explanation length to provide more comprehensive reasoning"
            )

        completeness = quality_dist.get("completeness_analysis", {})
        if completeness.get("completeness_ratio", 0) < 0.5:
            suggestions.append(
                "Improve explanation completeness by including causal reasoning (because, due to, etc.)"
            )

        # Analyze attack type patterns
        attack_patterns = analysis_results.get("attack_type_patterns", {})

        for attack_type, pattern_info in attack_patterns.items():
            keyword_coverage = pattern_info.get("keyword_coverage", 0)
            if keyword_coverage < 0.3:  # Low keyword coverage
                relevant_keywords = self.attack_type_keywords.get(attack_type, [])
                suggestions.append(
                    f"For {attack_type} attacks, include more specific terms like: {', '.join(relevant_keywords[:3])}"
                )

        # Statistical analysis suggestions
        stats = analysis_results.get("statistical_analysis", {})
        vocab_analysis = stats.get("vocabulary_analysis", {})

        if vocab_analysis.get("vocabulary_richness", 0) < 0.3:
            suggestions.append("Increase vocabulary diversity to avoid repetitive phrasing")

        return suggestions

    # Helper methods
    def _find_common_phrases(self, texts: list[str]) -> list[str]:
        """Find common phrases across texts using n-gram analysis."""
        all_phrases = []
        for text in texts:
            words = text.lower().split()
            # Extract 2-grams and 3-grams
            for i in range(len(words) - 1):
                all_phrases.append(" ".join(words[i : i + 2]))
            for i in range(len(words) - 2):
                all_phrases.append(" ".join(words[i : i + 3]))

        # Count phrase frequency
        phrase_counts = Counter(all_phrases)

        # Return most common phrases that appear in at least 2 texts
        common_phrases = [phrase for phrase, count in phrase_counts.most_common(10) if count >= 2]

        return common_phrases

    def _calculate_keyword_coverage(self, explanations: list[str], keywords: list[str]) -> float:
        """Calculate coverage of relevant keywords in explanations."""
        if not keywords or not explanations:
            return 0.0

        coverage_count = 0
        for explanation in explanations:
            explanation_lower = explanation.lower()
            if any(keyword in explanation_lower for keyword in keywords):
                coverage_count += 1

        return coverage_count / len(explanations)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _estimate_cluster_quality(self, explanations: list[str]) -> float:
        """Estimate quality of explanation cluster."""
        # Quality estimation based on length, diversity, and technical content
        avg_length = np.mean([len(exp.split()) for exp in explanations])
        length_score = min(1.0, float(avg_length) / 15.0)  # Normalize to 0-1

        # Diversity within cluster
        unique_words = set()
        for exp in explanations:
            unique_words.update(exp.lower().split())

        total_words = sum(len(exp.split()) for exp in explanations)
        diversity_score = len(unique_words) / total_words if total_words > 0 else 0

        # Technical content score
        technical_terms = ["attack", "malware", "threat", "vulnerability", "exploit"]
        technical_score_count = 0
        for exp in explanations:
            if any(term in exp.lower() for term in technical_terms):
                technical_score_count += 1
        technical_score: float = technical_score_count / len(explanations) if explanations else 0.0

        return float((length_score + diversity_score + technical_score) / 3)

    def _generate_cluster_description(
        self, explanations: list[str], common_phrases: list[str]
    ) -> str:
        """Generate a descriptive pattern description for a cluster."""
        if common_phrases:
            return f"Pattern with phrases: {', '.join(common_phrases[:3])}"

        # Fallback: analyze common themes
        themes = []
        for exp in explanations[:3]:  # Check first 3 explanations
            exp_lower = exp.lower()
            if any(term in exp_lower for term in ["attack", "malicious", "threat"]):
                themes.append("attack-related")
            if any(term in exp_lower for term in ["normal", "benign", "legitimate"]):
                themes.append("benign-related")
            if any(term in exp_lower for term in ["network", "traffic", "connection"]):
                themes.append("network-focused")

        if themes:
            return f"Pattern: {', '.join(set(themes))}"

        return "General explanation pattern"

    def _calculate_explanation_diversity(self, explanations: list[str]) -> float:
        """Calculate diversity score for a set of explanations."""
        if not explanations:
            return 0.0

        # Calculate unique n-grams ratio
        all_bigrams = []
        for exp in explanations:
            words = exp.lower().split()
            for i in range(len(words) - 1):
                all_bigrams.append(" ".join(words[i : i + 2]))

        if not all_bigrams:
            return 0.0

        unique_bigrams = set(all_bigrams)
        return len(unique_bigrams) / len(all_bigrams)

    def _analyze_quality_indicators(self, explanations: list[str]) -> dict[str, float]:
        """Analyze quality indicators within explanations."""
        indicators = {
            "causal_reasoning": ["because", "due to", "caused by", "results from"],
            "evidence_based": ["indicates", "shows", "demonstrates", "evidence"],
            "technical_specificity": ["ip address", "port", "protocol", "hash", "signature"],
            "temporal_context": ["first", "then", "subsequently", "after", "before"],
        }

        analysis = {}
        total_explanations = len(explanations)

        for indicator_type, keywords in indicators.items():
            count = 0
            for exp in explanations:
                if any(keyword in exp.lower() for keyword in keywords):
                    count += 1
            analysis[indicator_type] = count / total_explanations if total_explanations > 0 else 0

        return analysis

    def _analyze_technical_term_usage(
        self, explanations: list[str], technical_terms: list[str]
    ) -> dict[str, Any]:
        """Analyze usage of technical terms in explanations."""
        term_usage: Counter[str] = Counter()
        explanations_with_terms = 0

        for exp in explanations:
            exp_lower = exp.lower()
            found_terms = []
            for term in technical_terms:
                if term in exp_lower:
                    term_usage[term] += 1
                    found_terms.append(term)

            if found_terms:
                explanations_with_terms += 1

        return {
            "technical_coverage": explanations_with_terms / len(explanations)
            if explanations
            else 0,
            "most_used_terms": term_usage.most_common(10),
            "total_technical_occurrences": sum(term_usage.values()),
            "unique_terms_used": len(term_usage),
        }

    def _calculate_skewness(self, data: list[float]) -> float:
        """Calculate skewness of data distribution."""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        skewness = np.mean([((x - mean) / std) ** 3 for x in data])
        return float(skewness)

    def _calculate_kurtosis(self, data: list[float]) -> float:
        """Calculate kurtosis of data distribution."""
        if len(data) < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        kurtosis = np.mean([((x - mean) / std) ** 4 for x in data]) - 3
        return float(kurtosis)

    def _analyze_prediction_explanation_consistency(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Analyze consistency between predictions and explanations."""
        consistent_count = 0
        inconsistent_count = 0
        total_count = 0

        for pred, _gt in zip(predictions, ground_truth, strict=False):
            explanation = pred.get("explanation", "").lower()
            prediction = pred.get("prediction", "").lower()

            if not explanation.strip():
                continue

            total_count += 1

            # Check consistency based on prediction type
            if prediction in ["attack", "malicious"]:
                attack_keywords = ["attack", "malicious", "threat", "suspicious", "intrusion"]
                if any(keyword in explanation for keyword in attack_keywords):
                    consistent_count += 1
                else:
                    inconsistent_count += 1
            elif prediction in ["benign", "normal", "legitimate"]:
                benign_keywords = ["normal", "legitimate", "benign", "safe", "expected"]
                attack_keywords = ["attack", "malicious", "threat"]

                has_benign = any(keyword in explanation for keyword in benign_keywords)
                has_attack = any(keyword in explanation for keyword in attack_keywords)

                if has_benign or not has_attack:
                    consistent_count += 1
                else:
                    inconsistent_count += 1

        return {
            "consistency_ratio": consistent_count / total_count if total_count > 0 else 0,
            "consistent_explanations": consistent_count,
            "inconsistent_explanations": inconsistent_count,
            "total_analyzed": total_count,
        }

    def _calculate_avg_explanation_quality(self, explanations: list[str]) -> float:
        """Calculate average explanation quality using multiple factors."""
        if not explanations:
            return 0.0

        quality_scores = []
        for explanation in explanations:
            if not explanation.strip():
                quality_scores.append(0.0)
                continue

            # Length score (optimal around 15-20 words)
            word_count = len(explanation.split())
            if word_count < 5:
                length_score = word_count / 5.0
            elif word_count <= 20:
                length_score = 1.0
            else:
                length_score = max(0.5, 1.0 - (word_count - 20) / 50.0)

            # Completeness score
            completeness_indicators = ["because", "due to", "indicates", "shows", "evidence"]
            completeness_score = (
                1.0 if any(word in explanation.lower() for word in completeness_indicators) else 0.5
            )

            # Technical score
            technical_terms = ["malware", "attack", "threat", "intrusion", "vulnerability"]
            technical_score = (
                1.0 if any(term in explanation.lower() for term in technical_terms) else 0.5
            )

            # Specificity score
            specific_terms = ["ip address", "port", "hash", "signature", "protocol"]
            specificity_score = (
                1.0 if any(term in explanation.lower() for term in specific_terms) else 0.7
            )

            quality = (length_score + completeness_score + technical_score + specificity_score) / 4
            quality_scores.append(quality)

        return sum(quality_scores) / len(quality_scores)

    def _calculate_explanation_consistency(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> float:
        """Calculate explanation consistency with predictions."""
        consistent_count = 0
        total_count = 0

        for pred, _gt in zip(predictions, ground_truth, strict=False):
            explanation = pred.get("explanation", "").lower()
            prediction = pred.get("prediction", "").lower()

            if not explanation:
                continue

            total_count += 1

            if prediction in ["attack", "malicious"]:
                attack_keywords = ["attack", "malicious", "threat", "suspicious"]
                if any(keyword in explanation for keyword in attack_keywords):
                    consistent_count += 1
            elif prediction in ["benign", "normal"]:
                benign_keywords = ["normal", "legitimate", "benign", "safe"]
                attack_keywords = ["attack", "malicious", "threat"]

                has_benign = any(keyword in explanation for keyword in benign_keywords)
                has_attack = any(keyword in explanation for keyword in attack_keywords)

                if has_benign or not has_attack:
                    consistent_count += 1

        return consistent_count / total_count if total_count > 0 else 0.0

    def _estimate_technical_accuracy(self, explanations: list[str]) -> float:
        """Estimate technical accuracy of explanations."""
        if not explanations:
            return 0.0

        technical_scores = []

        for explanation in explanations:
            if not explanation.strip():
                technical_scores.append(0.0)
                continue

            # Count technical terms
            technical_terms = [
                "malware",
                "virus",
                "trojan",
                "ransomware",
                "intrusion",
                "vulnerability",
                "exploit",
                "payload",
                "backdoor",
                "phishing",
                "dos",
                "ddos",
                "injection",
                "buffer overflow",
                "privilege escalation",
                "lateral movement",
            ]

            explanation_lower = explanation.lower()
            technical_count = sum(1 for term in technical_terms if term in explanation_lower)

            # Normalize score
            score = min(1.0, technical_count / 3.0)  # Max score at 3+ technical terms
            technical_scores.append(score)

        return sum(technical_scores) / len(technical_scores)

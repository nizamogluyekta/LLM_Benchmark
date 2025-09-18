"""
Explanation templates for different cybersecurity scenarios.

This module provides template-based evaluation and generation for explanations
of different types of cybersecurity attacks and threats.
"""

from dataclasses import dataclass
from typing import Any

from benchmark.core.logging import get_logger


@dataclass
class ExplanationTemplate:
    """Template for structured explanation evaluation."""

    attack_type: str
    template: str
    required_elements: list[str]
    example_explanation: str
    optional_elements: list[str] | None = None
    quality_weight: float = 1.0


class ExplanationTemplateGenerator:
    """Generate and evaluate explanation templates for different cybersecurity scenarios."""

    def __init__(self) -> None:
        """Initialize the explanation template generator."""
        self.logger = get_logger("explanation_template_generator")

        # Initialize predefined templates
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> dict[str, ExplanationTemplate]:
        """Initialize predefined explanation templates."""
        templates = {
            "malware": ExplanationTemplate(
                attack_type="malware",
                template="Detected {malware_type} based on {indicators}. The file/process shows {behavioral_patterns} which indicates {threat_level} threat.",
                required_elements=[
                    "malware_type",
                    "indicators",
                    "behavioral_patterns",
                    "threat_level",
                ],
                optional_elements=["file_hash", "network_activity", "persistence_mechanism"],
                example_explanation="Detected trojan malware based on suspicious file hash and network connections. The file shows encrypted communications and registry modifications which indicates high threat.",
                quality_weight=1.0,
            ),
            "intrusion": ExplanationTemplate(
                attack_type="intrusion",
                template="Unauthorized access detected through {attack_vector}. The activity shows {access_patterns} attempting to {objective}.",
                required_elements=["attack_vector", "access_patterns", "objective"],
                optional_elements=["source_ip", "target_system", "escalation_attempts"],
                example_explanation="Unauthorized access detected through brute force login attempts. The activity shows multiple failed authentication attempts from suspicious IP attempting to gain system access.",
                quality_weight=1.0,
            ),
            "dos": ExplanationTemplate(
                attack_type="dos",
                template="Denial of Service attack identified via {dos_method}. Traffic analysis shows {traffic_patterns} resulting in {impact}.",
                required_elements=["dos_method", "traffic_patterns", "impact"],
                optional_elements=["attack_volume", "source_distribution", "mitigation_applied"],
                example_explanation="Denial of Service attack identified via TCP flood. Traffic analysis shows abnormally high request volume from multiple sources resulting in service degradation.",
                quality_weight=1.0,
            ),
            "phishing": ExplanationTemplate(
                attack_type="phishing",
                template="Phishing attempt detected in {medium} using {deception_method}. The content contains {suspicious_elements} designed to {goal}.",
                required_elements=["medium", "deception_method", "suspicious_elements", "goal"],
                optional_elements=["sender_reputation", "url_analysis", "attachment_type"],
                example_explanation="Phishing attempt detected in email using domain spoofing. The content contains suspicious links and urgent language designed to steal credentials.",
                quality_weight=1.0,
            ),
            "sql_injection": ExplanationTemplate(
                attack_type="sql_injection",
                template="SQL injection attack detected in {input_parameter} using {injection_technique}. The payload {payload_description} attempts to {attack_objective}.",
                required_elements=[
                    "input_parameter",
                    "injection_technique",
                    "payload_description",
                    "attack_objective",
                ],
                optional_elements=["database_type", "vulnerability_type", "data_accessed"],
                example_explanation="SQL injection attack detected in login form using UNION-based technique. The payload contains malicious SQL statements attempts to extract user data.",
                quality_weight=1.0,
            ),
            "reconnaissance": ExplanationTemplate(
                attack_type="reconnaissance",
                template="Reconnaissance activity detected through {scanning_method}. The {scan_targets} indicate {information_gathering} for {potential_objective}.",
                required_elements=[
                    "scanning_method",
                    "scan_targets",
                    "information_gathering",
                    "potential_objective",
                ],
                optional_elements=["scan_frequency", "tools_used", "stealth_techniques"],
                example_explanation="Reconnaissance activity detected through port scanning. The multiple service probes indicate network mapping for potential intrusion.",
                quality_weight=0.9,
            ),
            "data_exfiltration": ExplanationTemplate(
                attack_type="data_exfiltration",
                template="Data exfiltration detected via {exfiltration_method}. Analysis shows {data_transfer_patterns} indicating {data_type} being {transfer_destination}.",
                required_elements=[
                    "exfiltration_method",
                    "data_transfer_patterns",
                    "data_type",
                    "transfer_destination",
                ],
                optional_elements=["encryption_used", "timing_patterns", "data_volume"],
                example_explanation="Data exfiltration detected via encrypted tunnel. Analysis shows large file transfers indicating sensitive documents being sent to external servers.",
                quality_weight=1.0,
            ),
            "privilege_escalation": ExplanationTemplate(
                attack_type="privilege_escalation",
                template="Privilege escalation attempt detected using {escalation_technique}. The {exploit_method} targets {vulnerability_type} to gain {privilege_level}.",
                required_elements=[
                    "escalation_technique",
                    "exploit_method",
                    "vulnerability_type",
                    "privilege_level",
                ],
                optional_elements=["affected_service", "patch_status", "success_indicators"],
                example_explanation="Privilege escalation attempt detected using buffer overflow. The exploit targets kernel vulnerability to gain root access.",
                quality_weight=1.0,
            ),
            "lateral_movement": ExplanationTemplate(
                attack_type="lateral_movement",
                template="Lateral movement detected through {movement_technique}. The attacker uses {credentials_method} to access {target_systems} for {movement_objective}.",
                required_elements=[
                    "movement_technique",
                    "credentials_method",
                    "target_systems",
                    "movement_objective",
                ],
                optional_elements=["authentication_type", "persistence_method", "discovery_tools"],
                example_explanation="Lateral movement detected through credential reuse. The attacker uses stolen passwords to access additional servers for network exploration.",
                quality_weight=1.0,
            ),
            "benign": ExplanationTemplate(
                attack_type="benign",
                template="Activity classified as {activity_type} showing {normal_patterns}. The {behavioral_indicators} are consistent with {legitimate_purpose}.",
                required_elements=[
                    "activity_type",
                    "normal_patterns",
                    "behavioral_indicators",
                    "legitimate_purpose",
                ],
                optional_elements=["user_context", "system_context", "historical_patterns"],
                example_explanation="Activity classified as normal user behavior showing expected login patterns. The access times and locations are consistent with regular work activities.",
                quality_weight=0.8,
            ),
        }

        return templates

    def generate_template_for_attack(self, attack_type: str) -> ExplanationTemplate:
        """
        Get explanation template for specific attack type.

        Args:
            attack_type: Type of attack to get template for

        Returns:
            ExplanationTemplate for the specified attack type
        """
        template = self.templates.get(attack_type.lower())
        if not template:
            # Return a generic template if specific one not found
            return self._get_generic_template()

        return template

    def evaluate_explanation_against_template(
        self, explanation: str, attack_type: str, strict_mode: bool = False
    ) -> dict[str, Any]:
        """
        Evaluate explanation against appropriate template.

        Args:
            explanation: The explanation text to evaluate
            attack_type: Type of attack for template selection
            strict_mode: Whether to use strict evaluation criteria

        Returns:
            Dictionary containing evaluation results
        """
        template = self.generate_template_for_attack(attack_type)
        if not template:
            return {
                "score": 0.5,
                "missing_elements": [],
                "feedback": "No template available for this attack type",
            }

        explanation_lower = explanation.lower()

        # Check for required elements
        present_elements = []
        missing_elements = []

        for element in template.required_elements:
            element_keywords = self._get_element_keywords(element)
            if any(keyword in explanation_lower for keyword in element_keywords):
                present_elements.append(element)
            else:
                missing_elements.append(element)

        # Check for optional elements if available
        optional_present = []
        if template.optional_elements:
            for element in template.optional_elements:
                element_keywords = self._get_element_keywords(element)
                if any(keyword in explanation_lower for keyword in element_keywords):
                    optional_present.append(element)

        # Calculate score
        required_score = (
            len(present_elements) / len(template.required_elements)
            if template.required_elements
            else 0.5
        )

        # Bonus for optional elements
        optional_bonus = 0.0
        if template.optional_elements and optional_present:
            optional_bonus = 0.1 * (len(optional_present) / len(template.optional_elements))

        # Apply quality weight
        base_score = (required_score + optional_bonus) * template.quality_weight
        final_score = min(1.0, base_score)

        # Adjust score based on strict mode
        if strict_mode and missing_elements:
            final_score *= 0.8  # Penalty for missing required elements

        # Generate detailed feedback
        feedback = self._generate_template_feedback(
            present_elements, missing_elements, optional_present, template, strict_mode
        )

        return {
            "score": final_score,
            "present_elements": present_elements,
            "missing_elements": missing_elements,
            "optional_elements_present": optional_present,
            "feedback": feedback,
            "template_used": template.attack_type,
            "template_coverage": len(present_elements) / len(template.required_elements)
            if template.required_elements
            else 0,
        }

    def batch_evaluate_explanations(
        self, predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Evaluate multiple explanations against their respective templates.

        Args:
            predictions: List of predictions with explanations and attack types
            ground_truth: Optional ground truth for validation

        Returns:
            Dictionary containing batch evaluation results
        """
        results: dict[str, Any] = {
            "individual_evaluations": [],
            "summary_statistics": {},
            "template_usage": {},
            "improvement_recommendations": [],
        }

        template_scores: dict[str, list[float]] = {}
        template_counts: dict[str, int] = {}

        for i, pred in enumerate(predictions):
            explanation = pred.get("explanation", "")
            attack_type = pred.get("attack_type", "unknown")

            # Use ground truth attack type if available
            if ground_truth and i < len(ground_truth):
                gt_attack_type = ground_truth[i].get("attack_type", attack_type)
                attack_type = gt_attack_type

            if not explanation.strip():
                evaluation = {
                    "index": i,
                    "score": 0.0,
                    "feedback": "Empty explanation provided",
                    "template_used": "none",
                }
            else:
                evaluation = self.evaluate_explanation_against_template(explanation, attack_type)
                evaluation["index"] = i

            results["individual_evaluations"].append(evaluation)

            # Track template usage and scores
            template_used = str(evaluation.get("template_used", "unknown"))
            if template_used not in template_scores:
                template_scores[template_used] = []
                template_counts[template_used] = 0

            score_value = evaluation.get("score", 0.0)
            if isinstance(score_value, int | float):
                template_scores[template_used].append(float(score_value))
            else:
                template_scores[template_used].append(0.0)
            template_counts[template_used] += 1

        # Calculate summary statistics
        all_scores = [eval_result["score"] for eval_result in results["individual_evaluations"]]

        results["summary_statistics"] = {
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "median_score": sorted(all_scores)[len(all_scores) // 2] if all_scores else 0,
            "min_score": min(all_scores) if all_scores else 0,
            "max_score": max(all_scores) if all_scores else 0,
            "total_evaluations": len(all_scores),
            "high_quality_explanations": sum(1 for score in all_scores if score >= 0.8),
            "low_quality_explanations": sum(1 for score in all_scores if score < 0.5),
        }

        # Template usage analysis
        for template, scores in template_scores.items():
            results["template_usage"][template] = {
                "count": template_counts[template],
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "score_range": {"min": min(scores), "max": max(scores)}
                if scores
                else {"min": 0.0, "max": 0.0},
            }

        # Generate improvement recommendations
        results["improvement_recommendations"] = self._generate_batch_recommendations(
            results["individual_evaluations"], results["summary_statistics"]
        )

        return results

    def get_all_templates(self) -> dict[str, ExplanationTemplate]:
        """Get all available explanation templates."""
        return self.templates.copy()

    def add_custom_template(self, template: ExplanationTemplate) -> None:
        """
        Add custom explanation template.

        Args:
            template: Custom ExplanationTemplate to add
        """
        self.templates[template.attack_type] = template
        self.logger.info(f"Added custom template for attack type: {template.attack_type}")

    def get_template_statistics(self) -> dict[str, Any]:
        """Get statistics about available templates."""
        stats = {
            "total_templates": len(self.templates),
            "attack_types_covered": list(self.templates.keys()),
            "template_complexity": {},
            "coverage_analysis": {},
        }

        for attack_type_key, template in self.templates.items():
            if isinstance(stats["template_complexity"], dict):
                stats["template_complexity"][attack_type_key] = {
                    "required_elements": len(template.required_elements),
                    "optional_elements": len(template.optional_elements)
                    if template.optional_elements
                    else 0,
                    "total_elements": len(template.required_elements)
                    + (len(template.optional_elements) if template.optional_elements else 0),
                    "quality_weight": template.quality_weight,
                }

        return stats

    def _get_element_keywords(self, element: str) -> list[str]:
        """Get keywords associated with template elements."""
        element_keywords = {
            # Malware elements
            "malware_type": [
                "malware",
                "virus",
                "trojan",
                "ransomware",
                "spyware",
                "adware",
                "worm",
                "rootkit",
            ],
            "indicators": [
                "hash",
                "signature",
                "behavior",
                "network",
                "file",
                "process",
                "registry",
            ],
            "behavioral_patterns": [
                "communication",
                "modification",
                "execution",
                "persistence",
                "encryption",
            ],
            "threat_level": ["high", "medium", "low", "critical", "severe", "moderate"],
            # Intrusion elements
            "attack_vector": [
                "brute force",
                "sql injection",
                "buffer overflow",
                "social engineering",
                "phishing",
            ],
            "access_patterns": [
                "login",
                "authentication",
                "credential",
                "session",
                "failure",
                "attempt",
            ],
            "objective": ["access", "data", "system", "privilege", "information", "control"],
            # DoS elements
            "dos_method": [
                "flood",
                "amplification",
                "exhaustion",
                "volumetric",
                "protocol",
                "application",
            ],
            "traffic_patterns": [
                "volume",
                "frequency",
                "source",
                "bandwidth",
                "request",
                "connection",
            ],
            "impact": ["degradation", "unavailable", "slow", "blocked", "timeout", "overload"],
            # Phishing elements
            "medium": ["email", "web", "sms", "social media", "message", "link"],
            "deception_method": ["spoofing", "impersonation", "fake", "mimicking", "disguise"],
            "suspicious_elements": ["link", "attachment", "domain", "urgency", "grammar", "sender"],
            "goal": ["steal", "credential", "information", "money", "data", "identity"],
            # SQL Injection elements
            "input_parameter": ["form", "parameter", "field", "input", "query", "variable"],
            "injection_technique": ["union", "blind", "time-based", "boolean", "error-based"],
            "payload_description": ["sql", "query", "statement", "command", "injection"],
            "attack_objective": ["extract", "modify", "delete", "bypass", "authenticate"],
            # Reconnaissance elements
            "scanning_method": ["port scan", "network scan", "vulnerability scan", "enumeration"],
            "scan_targets": ["ports", "services", "systems", "networks", "applications"],
            "information_gathering": ["mapping", "discovery", "fingerprinting", "reconnaissance"],
            "potential_objective": ["intrusion", "attack", "exploitation", "penetration"],
            # Data Exfiltration elements
            "exfiltration_method": ["tunnel", "transfer", "upload", "email", "cloud", "protocol"],
            "data_transfer_patterns": [
                "large files",
                "encrypted",
                "compressed",
                "staged",
                "scheduled",
            ],
            "data_type": ["documents", "credentials", "database", "source code", "personal"],
            "transfer_destination": ["external server", "cloud storage", "email", "ftp"],
            # Privilege Escalation elements
            "escalation_technique": [
                "buffer overflow",
                "dll hijacking",
                "token manipulation",
                "exploit",
            ],
            "exploit_method": [
                "vulnerability",
                "misconfiguration",
                "weak permissions",
                "unpatched",
            ],
            "vulnerability_type": ["kernel", "application", "service", "driver", "system"],
            "privilege_level": ["root", "admin", "system", "elevated", "administrator"],
            # Lateral Movement elements
            "movement_technique": [
                "credential reuse",
                "pass-the-hash",
                "remote services",
                "shares",
            ],
            "credentials_method": ["stolen", "harvested", "cracked", "dumped", "cached"],
            "target_systems": ["servers", "workstations", "domain controllers", "databases"],
            "movement_objective": ["exploration", "persistence", "data access", "control"],
            # Benign elements
            "activity_type": ["normal", "legitimate", "expected", "routine", "authorized"],
            "normal_patterns": ["regular", "consistent", "typical", "standard", "usual"],
            "behavioral_indicators": ["access times", "locations", "frequency", "duration"],
            "legitimate_purpose": ["work", "business", "authorized", "scheduled", "maintenance"],
            # Optional elements
            "file_hash": ["md5", "sha1", "sha256", "hash", "checksum"],
            "network_activity": ["connection", "traffic", "communication", "protocol"],
            "source_ip": ["ip address", "source", "origin", "remote"],
            "attack_volume": ["requests", "packets", "bandwidth", "rate"],
            "sender_reputation": ["reputation", "blacklist", "known", "suspicious"],
        }

        return element_keywords.get(element, [element])

    def _get_generic_template(self) -> ExplanationTemplate:
        """Get a generic template for unknown attack types."""
        return ExplanationTemplate(
            attack_type="generic",
            template="Security event detected showing {indicators}. Analysis reveals {patterns} suggesting {assessment}.",
            required_elements=["indicators", "patterns", "assessment"],
            optional_elements=["context", "impact", "recommendation"],
            example_explanation="Security event detected showing anomalous behavior. Analysis reveals suspicious patterns suggesting potential threat.",
            quality_weight=0.7,
        )

    def _generate_template_feedback(
        self,
        present_elements: list[str],
        missing_elements: list[str],
        optional_present: list[str],
        template: ExplanationTemplate,
        strict_mode: bool,
    ) -> str:
        """Generate detailed feedback for template evaluation."""
        feedback_parts = []

        if present_elements:
            feedback_parts.append(f"Present required elements: {', '.join(present_elements)}")

        if missing_elements:
            feedback_parts.append(f"Missing required elements: {', '.join(missing_elements)}")

        if optional_present:
            feedback_parts.append(f"Present optional elements: {', '.join(optional_present)}")

        if not present_elements:
            feedback_parts.append("No required template elements found in explanation")
        elif len(present_elements) == len(template.required_elements):
            feedback_parts.append("All required elements present")

        if strict_mode and missing_elements:
            feedback_parts.append("Strict mode: Score reduced due to missing required elements")

        # Add specific suggestions
        if missing_elements:
            suggestions = []
            for element in missing_elements[:3]:  # Top 3 missing elements
                keywords = self._get_element_keywords(element)
                suggestions.append(f"Include {element} (e.g., {', '.join(keywords[:2])})")
            feedback_parts.append(f"Suggestions: {'; '.join(suggestions)}")

        return "; ".join(feedback_parts)

    def _generate_batch_recommendations(
        self, individual_evaluations: list[dict[str, Any]], summary_stats: dict[str, Any]
    ) -> list[str]:
        """Generate improvement recommendations for batch evaluation."""
        recommendations = []

        # Overall quality recommendations
        avg_score = summary_stats.get("average_score", 0)
        if avg_score < 0.6:
            recommendations.append(
                "Overall explanation quality is below acceptable threshold (60%)"
            )

        low_quality_count = summary_stats.get("low_quality_explanations", 0)
        total_count = summary_stats.get("total_evaluations", 1)

        if low_quality_count / total_count > 0.3:
            recommendations.append(
                "High proportion of low-quality explanations - focus on template compliance"
            )

        # Analyze common missing elements
        missing_elements_count: dict[str, int] = {}
        for evaluation in individual_evaluations:
            for element in evaluation.get("missing_elements", []):
                missing_elements_count[element] = missing_elements_count.get(element, 0) + 1

        # Recommend addressing most common missing elements
        if missing_elements_count:
            most_missing = sorted(missing_elements_count.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
            for element, count in most_missing:
                if count > total_count * 0.3:  # Missing in >30% of explanations
                    recommendations.append(
                        f"Commonly missing element '{element}' - found in {count}/{total_count} explanations"
                    )

        # Template-specific recommendations
        template_issues: dict[str, list[float]] = {}
        for evaluation in individual_evaluations:
            template_used = evaluation.get("template_used", "unknown")
            score = evaluation.get("score", 0)

            if template_used not in template_issues:
                template_issues[template_used] = []
            template_issues[template_used].append(float(score))

        for template, scores in template_issues.items():
            if template != "unknown" and len(scores) >= 3:  # At least 3 examples
                avg_template_score = sum(scores) / len(scores)
                if avg_template_score < 0.5:
                    recommendations.append(
                        f"Template '{template}' shows consistently low scores - review template requirements"
                    )

        return recommendations

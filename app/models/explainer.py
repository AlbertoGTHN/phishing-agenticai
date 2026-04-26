"""
Explainability Module for Phishing Detection.

Generates human-readable explanations of classification decisions,
including threat indicator breakdowns and educational content.

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def generate_explanation(
    verdict: str,
    confidence: float,
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
) -> str:
    """
    Generate a natural language explanation of the classification decision.

    Args:
        verdict: "phishing", "suspicious", or "legitimate"
        confidence: Confidence score (0.0-1.0)
        semantic_results: Output from semantic engine
        structural_results: Output from structural engine

    Returns:
        Human-readable explanation paragraph.
    """
    if verdict == "legitimate":
        return _explain_legitimate(confidence, semantic_results, structural_results)
    elif verdict == "suspicious":
        return _explain_suspicious(confidence, semantic_results, structural_results)
    else:
        return _explain_phishing(confidence, semantic_results, structural_results)


def _explain_phishing(
    confidence: float,
    semantic: Dict[str, Any],
    structural: Dict[str, Any],
) -> str:
    """Generate explanation for phishing classification."""
    reasons = []

    # Semantic reasons
    rule_scores = semantic.get("rule_scores", {})
    if rule_scores.get("urgency", {}).get("matches"):
        phrases = rule_scores["urgency"]["matches"][:3]
        reasons.append(f"urgency language (e.g., '{phrases[0]}')")
    if rule_scores.get("authority", {}).get("matches"):
        phrases = rule_scores["authority"]["matches"][:2]
        reasons.append(f"authority impersonation (references to '{phrases[0]}')")
    if rule_scores.get("credential_requests", {}).get("matches"):
        reasons.append("requests for sensitive credentials")
    if rule_scores.get("pressure", {}).get("matches"):
        reasons.append("time-pressure tactics")

    # URL reasons
    url_analysis = structural.get("url_analysis", {})
    if url_analysis.get("overall_score", 0) > 0.3:
        url_details = url_analysis.get("details", [])
        if url_details and url_details != ["No URLs found in email content"]:
            reasons.append(f"suspicious URLs ({url_analysis.get('url_count', 0)} found)")

    # Header reasons
    header_analysis = structural.get("header_analysis", {})
    if header_analysis.get("overall_score", 0) > 0.3:
        if header_analysis.get("spf_result") == "fail":
            reasons.append("failed SPF authentication")
        if header_analysis.get("display_name_mismatch"):
            reasons.append("sender display name mismatch")
        if header_analysis.get("reply_to_mismatch"):
            reasons.append("Reply-To address mismatch")

    # HTML reasons
    html_analysis = structural.get("html_analysis", {})
    if html_analysis.get("overall_score", 0) > 0.2:
        if html_analysis.get("external_form_count", 0) > 0:
            reasons.append("embedded forms pointing to external domains")
        if html_analysis.get("has_obfuscated_js"):
            reasons.append("obfuscated JavaScript code")

    if reasons:
        reason_text = ", ".join(reasons[:-1])
        if len(reasons) > 1:
            reason_text += f", and {reasons[-1]}"
        else:
            reason_text = reasons[0]
        return (
            f"This email was classified as PHISHING with {confidence*100:.1f}% confidence. "
            f"The analysis detected {reason_text}. "
            f"These are strong indicators of a phishing attempt designed to deceive "
            f"the recipient into revealing sensitive information or taking harmful actions."
        )

    return (
        f"This email was classified as PHISHING with {confidence*100:.1f}% confidence "
        f"based on a combination of semantic and structural indicators that match "
        f"known phishing patterns."
    )


def _explain_suspicious(
    confidence: float,
    semantic: Dict[str, Any],
    structural: Dict[str, Any],
) -> str:
    """Generate explanation for suspicious classification."""
    indicators = []

    semantic_score = semantic.get("overall_score", 0)
    url_score = structural.get("url_analysis", {}).get("overall_score", 0)
    header_score = structural.get("header_analysis", {}).get("overall_score", 0)

    if semantic_score > 0.2:
        indicators.append("some suspicious language patterns")
    if url_score > 0.2:
        indicators.append("potentially suspicious URLs")
    if header_score > 0.2:
        indicators.append("minor header anomalies")

    indicator_text = ", ".join(indicators) if indicators else "borderline indicators"

    return (
        f"This email was classified as SUSPICIOUS with {confidence*100:.1f}% confidence. "
        f"The analysis found {indicator_text}, but the evidence is not conclusive enough "
        f"for a definitive phishing classification. Manual review is recommended to "
        f"determine if this email is legitimate."
    )


def _explain_legitimate(
    confidence: float,
    semantic: Dict[str, Any],
    structural: Dict[str, Any],
) -> str:
    """Generate explanation for legitimate classification."""
    return (
        f"This email appears to be LEGITIMATE with {(1-confidence)*100:.1f}% confidence. "
        f"No significant phishing indicators were detected in the content, URLs, "
        f"headers, or HTML structure. Standard security hygiene is still recommended."
    )


def generate_education_note(
    verdict: str,
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
) -> str:
    """
    Generate an educational note explaining the type of attack detected.
    """
    if verdict == "legitimate":
        return (
            "Even legitimate-looking emails can sometimes be phishing attempts. "
            "Always verify sender addresses, hover over links before clicking, "
            "and never share passwords or sensitive information via email."
        )

    rule_scores = semantic_results.get("rule_scores", {})
    url_analysis = structural_results.get("url_analysis", {})

    notes = []

    if rule_scores.get("urgency", {}).get("matches"):
        notes.append(
            "URGENCY TACTICS: Phishing emails often create a false sense of urgency "
            "('account suspended', 'act immediately') to pressure you into acting "
            "without thinking. Legitimate organizations rarely demand immediate action via email."
        )

    if rule_scores.get("authority", {}).get("matches"):
        notes.append(
            "AUTHORITY IMPERSONATION: Attackers impersonate trusted entities (banks, "
            "tech companies, executives) to gain your trust. Always verify through "
            "official channels — call the organization directly using a known number."
        )

    if rule_scores.get("credential_requests", {}).get("matches"):
        notes.append(
            "CREDENTIAL HARVESTING: This email attempts to collect sensitive information. "
            "No legitimate organization will ask for passwords, SSN, or full credit card "
            "numbers via email. Report such requests to your security team."
        )

    urls = url_analysis.get("urls", [])
    typosquatting = [u for u in urls if u.get("typosquatting_score", 0) > 0.5]
    if typosquatting:
        target = typosquatting[0].get("typosquatting_target", "a known brand")
        notes.append(
            f"TYPOSQUATTING: The email contains a URL that closely mimics '{target}' "
            f"but is slightly different. Always carefully check domain names — "
            f"attackers register look-alike domains to steal credentials."
        )

    if not notes:
        notes.append(
            "GENERAL PHISHING: This email exhibits patterns commonly seen in phishing "
            "attacks. Be cautious with any email that asks you to click links, download "
            "attachments, or provide personal information. When in doubt, contact the "
            "supposed sender through a known, verified channel."
        )

    return " | ".join(notes)


def build_threat_indicators(
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Build the threat indicators breakdown for the API response.

    Returns scores and details for each indicator category.
    """
    rule_scores = semantic_results.get("rule_scores", {})
    url_analysis = structural_results.get("url_analysis", {})
    header_analysis = structural_results.get("header_analysis", {})
    html_analysis = structural_results.get("html_analysis", {})

    # Suspicious URLs indicator
    url_details = url_analysis.get("details", [])
    if url_details == ["No URLs found in email content"]:
        url_details = []

    # Urgency Language indicator
    urgency_details = []
    for key in ["urgency", "authority", "pressure"]:
        matches = rule_scores.get(key, {}).get("matches", [])
        urgency_details.extend(matches)

    # Header Anomalies
    header_details = header_analysis.get("details", [])
    if header_details == ["No header anomalies detected"]:
        header_details = []

    # Grammatical Anomalies
    grammar = rule_scores.get("grammatical_anomalies", {})
    grammar_details = grammar.get("issues", [])

    # HTML Suspicious
    html_details = html_analysis.get("details", [])
    if html_details == ["No suspicious HTML elements detected"]:
        html_details = []

    return {
        "suspicious_urls": {
            "score": url_analysis.get("overall_score", 0.0),
            "details": url_details,
        },
        "urgency_language": {
            "score": round(
                (rule_scores.get("urgency", {}).get("score", 0) * 0.4 +
                 rule_scores.get("authority", {}).get("score", 0) * 0.3 +
                 rule_scores.get("pressure", {}).get("score", 0) * 0.3), 4
            ),
            "details": urgency_details,
        },
        "header_anomalies": {
            "score": header_analysis.get("overall_score", 0.0),
            "details": header_details,
        },
        "grammatical_anomalies": {
            "score": grammar.get("score", 0.0),
            "details": grammar_details,
        },
        "html_suspicious": {
            "score": html_analysis.get("overall_score", 0.0),
            "details": html_details,
        },
    }

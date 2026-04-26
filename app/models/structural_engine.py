"""
Structural Analysis Engine (Engine B) for Phishing Detection.

Combines URL analysis, header analysis, and HTML analysis
to extract structural features from email content.

URL analysis uses a 1D-CNN (character-level, kernel sizes 3/5/7) when the
trained checkpoint ``models/url_cnn.pt`` is present; otherwise falls back
to the existing rule-based URL scorer transparently.

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import logging
import os
from typing import Dict, Any, Optional

from app.utils.url_analyzer import analyze_urls
from app.utils.header_analyzer import analyze_headers, analyze_headers_from_text
from app.utils.html_analyzer import analyze_html

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load URL CNN (optional — falls back gracefully if absent)
# ---------------------------------------------------------------------------
_URL_CNN_MODEL = None
_URL_CNN_LOAD_ATTEMPTED = False
_URL_CNN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "url_cnn.pt"
)


def _get_url_cnn() -> Optional[object]:
    """
    Attempt to load the URL-CNN checkpoint once.  Returns the model on
    success or None if the file is missing / PyTorch unavailable.
    """
    global _URL_CNN_MODEL, _URL_CNN_LOAD_ATTEMPTED
    if _URL_CNN_LOAD_ATTEMPTED:
        return _URL_CNN_MODEL

    _URL_CNN_LOAD_ATTEMPTED = True
    abs_path = os.path.abspath(_URL_CNN_PATH)

    if not os.path.exists(abs_path):
        logger.debug("url_cnn.pt not found — using rule-based URL scoring")
        return None

    try:
        from app.models.url_cnn import load_url_cnn
        _URL_CNN_MODEL = load_url_cnn(abs_path)
        logger.info(f"URL-CNN loaded from {abs_path} (1D-CNN URL scoring active)")
    except Exception as exc:
        logger.warning(f"Could not load URL-CNN ({exc}) — falling back to rule-based scoring")
        _URL_CNN_MODEL = None

    return _URL_CNN_MODEL


def analyze_structure(parsed_email: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full structural analysis on a parsed email.

    Combines URL, header, and HTML analysis into a single
    structural feature set with an overall threat score.

    Args:
        parsed_email: Output from eml_parser, text_parser, or ocr_parser

    Returns:
        Dictionary with structural analysis results.
    """
    body_text = parsed_email.get("body_text", "")
    body_html = parsed_email.get("body_html", "")
    headers_msg = parsed_email.get("headers")

    # URL Analysis — CNN if checkpoint present, rule-based otherwise
    url_cnn = _get_url_cnn()
    if url_cnn is not None:
        try:
            from app.models.url_cnn import analyze_urls_with_cnn
            # Run rule-based first to get per-URL feature flags, then blend
            rule_url_results = analyze_urls(body_text, body_html)
            url_results = analyze_urls_with_cnn(
                url_cnn, body_text, body_html, rule_results=rule_url_results
            )
        except Exception as exc:
            logger.warning(f"URL-CNN scoring failed ({exc}), using rule-based")
            url_results = analyze_urls(body_text, body_html)
    else:
        url_results = analyze_urls(body_text, body_html)

    # Header Analysis
    if headers_msg is not None:
        header_results = analyze_headers(headers_msg)
    else:
        header_results = analyze_headers_from_text("")

    # HTML Analysis
    html_results = analyze_html(body_html)

    # Compute structural feature vector (~50 features)
    structural_features = _extract_feature_vector(url_results, header_results, html_results)

    # Combined structural score (weighted)
    combined_score = (
        url_results["overall_score"] * 0.40 +
        header_results["overall_score"] * 0.35 +
        html_results["overall_score"] * 0.25
    )

    # Aggregate details
    all_details = []
    if url_results["details"] and url_results["details"] != ["No URLs found in email content"]:
        all_details.extend(url_results["details"])
    if header_results["details"] and header_results["details"] != ["No header anomalies detected"]:
        all_details.extend(header_results["details"])
    if html_results["details"] and html_results["details"] != ["No suspicious HTML elements detected"]:
        all_details.extend(html_results["details"])

    if not all_details:
        all_details = ["No structural anomalies detected"]

    return {
        "url_analysis": url_results,
        "header_analysis": header_results,
        "html_analysis": html_results,
        "structural_features": structural_features,
        "overall_score": round(min(combined_score, 1.0), 4),
        "details": all_details,
    }


def _extract_feature_vector(
    url_results: Dict[str, Any],
    header_results: Dict[str, Any],
    html_results: Dict[str, Any],
) -> list:
    """
    Extract a numeric feature vector from structural analysis results.
    Produces approximately 50 features for the Random Forest classifier.
    """
    features = []

    # URL features (15 features)
    features.append(url_results.get("url_count", 0))
    features.append(url_results.get("overall_score", 0.0))

    urls = url_results.get("urls", [])
    features.append(sum(1 for u in urls if u.get("is_ip_based")))
    features.append(sum(1 for u in urls if u.get("uses_shortener")))
    features.append(sum(1 for u in urls if u.get("suspicious_tld")))
    features.append(sum(1 for u in urls if u.get("long_domain")))
    features.append(sum(1 for u in urls if u.get("has_at_symbol")))
    features.append(sum(1 for u in urls if u.get("excessive_subdomains")))
    features.append(sum(1 for u in urls if u.get("excessive_hyphens")))
    features.append(sum(1 for u in urls if not u.get("uses_https")))
    features.append(max((u.get("typosquatting_score", 0) for u in urls), default=0))
    features.append(sum(1 for u in urls if u.get("suspicious_path")))
    features.append(max((u.get("risk_score", 0) for u in urls), default=0))
    features.append(sum(u.get("risk_score", 0) for u in urls) / max(len(urls), 1))
    features.append(min(len(urls), 20) / 20.0)  # normalized URL count

    # Header features (15 features)
    spf_map = {"pass": 0, "none": 0.5, "softfail": 0.7, "fail": 1.0}
    dkim_map = {"pass": 0, "none": 0.5, "fail": 1.0}
    dmarc_map = {"pass": 0, "none": 0.5, "fail": 1.0}

    features.append(spf_map.get(header_results.get("spf_result", "none"), 0.5))
    features.append(dkim_map.get(header_results.get("dkim_result", "none"), 0.5))
    features.append(dmarc_map.get(header_results.get("dmarc_result", "none"), 0.5))
    features.append(float(header_results.get("reply_to_mismatch", False)))
    features.append(float(header_results.get("display_name_mismatch", False)))
    features.append(float(header_results.get("x_mailer_suspicious", False)))
    features.append(float(header_results.get("missing_message_id", False)))
    features.append(float(header_results.get("missing_date", False)))
    features.append(float(header_results.get("suspicious_received_chain", False)))
    features.append(min(header_results.get("received_hop_count", 0), 15) / 15.0)
    features.append(header_results.get("overall_score", 0.0))
    # Padding to reach 15 header features
    features.extend([0.0] * 4)

    # HTML features (15 features)
    features.append(min(html_results.get("hidden_element_count", 0), 10) / 10.0)
    features.append(min(html_results.get("external_form_count", 0), 5) / 5.0)
    features.append(min(html_results.get("tracking_pixel_count", 0), 5) / 5.0)
    features.append(min(html_results.get("external_image_count", 0), 20) / 20.0)
    features.append(min(html_results.get("iframe_count", 0), 5) / 5.0)
    features.append(float(html_results.get("has_obfuscated_js", False)))
    features.append(min(html_results.get("base64_content_count", 0), 10) / 10.0)
    features.append(html_results.get("overall_score", 0.0))
    # Padding to reach 15 HTML features
    features.extend([0.0] * 7)

    # Additional combined features (5 features)
    features.append(url_results.get("overall_score", 0) * header_results.get("overall_score", 0))
    features.append(max(url_results.get("overall_score", 0), header_results.get("overall_score", 0)))
    features.append(
        (url_results.get("overall_score", 0) +
         header_results.get("overall_score", 0) +
         html_results.get("overall_score", 0)) / 3.0
    )
    features.append(float(bool(urls) and header_results.get("display_name_mismatch", False)))
    features.append(float(html_results.get("external_form_count", 0) > 0 and bool(urls)))

    return features

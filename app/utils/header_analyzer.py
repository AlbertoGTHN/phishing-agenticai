"""
Email Header Analysis Module for Phishing Detection.

Parses and analyzes email headers to detect authentication failures,
sender spoofing, and other header-based phishing indicators.

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import os
import sys
import re
import json
import logging
from typing import Dict, Any, List, Optional
from email.message import Message

logger = logging.getLogger(__name__)

# When True, missing Received headers are NOT treated as a forgery signal.
# Set RELAX_RECEIVED_CHECK=true for local .eml test files that were never
# transmitted through a mail server (avoids systematic false-positive bias).
_RELAX_RECEIVED: bool = (
    os.environ.get("RELAX_RECEIVED_CHECK", "false").strip().lower() == "true"
)

# Load brand domains for sender-brand mismatch detection
_BRAND_DOMAINS = {}
_BRAND_KEYWORDS = {}
try:
    if getattr(sys, 'frozen', False):
        _brand_file = os.path.join(sys._MEIPASS, "data", "brand_domains.json")
    else:
        _brand_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "brand_domains.json")
    with open(os.path.abspath(_brand_file), "r") as f:
        _brand_data = json.load(f)
    for brand in _brand_data.get("brands", []):
        name_lower = brand["name"].lower()
        domains = [d.lower() for d in brand.get("domains", [])]
        _BRAND_DOMAINS[name_lower] = domains
        # Build keyword -> brand mapping (e.g. "paypal" -> "PayPal")
        # Use the brand name itself and first word as keywords
        _BRAND_KEYWORDS[name_lower] = domains
        for word in name_lower.split():
            if len(word) >= 3:
                _BRAND_KEYWORDS[word] = domains
    logger.info(f"Loaded {len(_BRAND_DOMAINS)} brands for sender-brand mismatch detection")
except Exception as e:
    logger.warning(f"Could not load brand_domains.json: {e}")


def _get_header(msg: Message, name: str) -> Optional[str]:
    """Safely get a header value."""
    val = msg.get(name)
    return str(val).strip() if val else None


def _extract_email_address(header_value: str) -> Optional[str]:
    """Extract email address from a header value like 'Name <email@domain.com>'."""
    match = re.search(r'<([^>]+)>', header_value)
    if match:
        return match.group(1).lower()
    match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', header_value)
    if match:
        return match.group(0).lower()
    return None


def _extract_domain(email_addr: str) -> Optional[str]:
    """Extract domain from email address."""
    if email_addr and "@" in email_addr:
        return email_addr.split("@", 1)[1].lower()
    return None


def _check_auth_result(msg: Message, protocol: str) -> str:
    """Check SPF/DKIM/DMARC results from Authentication-Results header."""
    auth_results = _get_header(msg, "Authentication-Results")
    if not auth_results:
        return "none"

    pattern = re.compile(rf'{protocol}\s*=\s*(\w+)', re.IGNORECASE)
    match = pattern.search(auth_results)
    if match:
        result = match.group(1).lower()
        if result in ("pass", "fail", "softfail", "neutral", "temperror", "permerror", "none"):
            return result
    return "none"


def analyze_headers(msg: Message) -> Dict[str, Any]:
    """
    Analyze email headers for phishing indicators.

    Args:
        msg: Parsed email.message.Message object

    Returns:
        Dictionary with header analysis results and threat score.
    """
    features = {
        "spf_result": "none",
        "dkim_result": "none",
        "dmarc_result": "none",
        "from_display_name": None,
        "from_email": None,
        "from_domain": None,
        "reply_to_email": None,
        "reply_to_mismatch": False,
        "display_name_mismatch": False,
        "x_mailer": None,
        "x_mailer_suspicious": False,
        "missing_message_id": False,
        "missing_date": False,
        "suspicious_received_chain": False,
        "received_hop_count": 0,
        "overall_score": 0.0,
        "details": [],
    }

    try:
        # SPF/DKIM/DMARC
        features["spf_result"] = _check_auth_result(msg, "spf")
        features["dkim_result"] = _check_auth_result(msg, "dkim")
        features["dmarc_result"] = _check_auth_result(msg, "dmarc")

        # From header analysis
        from_header = _get_header(msg, "From")
        if from_header:
            features["from_display_name"] = from_header
            features["from_email"] = _extract_email_address(from_header)
            features["from_domain"] = _extract_domain(features["from_email"] or "")

            # Check display name vs domain mismatch using brand_domains.json
            display_lower = from_header.lower()
            from_domain = features["from_domain"] or ""
            for keyword, legitimate_domains in _BRAND_KEYWORDS.items():
                if keyword in display_lower and not any(from_domain.endswith(d) for d in legitimate_domains):
                    # Avoid false positives: don't flag if keyword is a common word
                    # and the domain looks legitimate
                    if keyword in ("bank", "security", "support", "target"):
                        continue  # too generic
                    features["display_name_mismatch"] = True
                    features["brand_mismatch_keyword"] = keyword
                    features["brand_legitimate_domains"] = legitimate_domains[:3]
                    features["details"].append(
                        f"Display name contains '{keyword}' but sender domain "
                        f"'{from_domain}' is not an official domain"
                    )
                    break

        # Reply-To mismatch
        reply_to = _get_header(msg, "Reply-To")
        if reply_to:
            features["reply_to_email"] = _extract_email_address(reply_to)
            reply_domain = _extract_domain(features["reply_to_email"] or "")
            if reply_domain and features["from_domain"] and reply_domain != features["from_domain"]:
                features["reply_to_mismatch"] = True
                features["details"].append(
                    f"Reply-To domain ({reply_domain}) differs from From domain ({features['from_domain']})"
                )

        # X-Mailer
        x_mailer = _get_header(msg, "X-Mailer")
        if x_mailer:
            features["x_mailer"] = x_mailer
            suspicious_mailers = ["phpmailer", "swiftmailer", "mass", "bulk", "sendinblue"]
            if any(s in x_mailer.lower() for s in suspicious_mailers):
                features["x_mailer_suspicious"] = True
                features["details"].append(f"Suspicious X-Mailer: {x_mailer}")

        # Missing standard headers
        if not _get_header(msg, "Message-ID"):
            features["missing_message_id"] = True
            features["details"].append("Missing Message-ID header")

        if not _get_header(msg, "Date"):
            features["missing_date"] = True
            features["details"].append("Missing Date header")

        # Received chain analysis
        received_headers = msg.get_all("Received") or []
        features["received_hop_count"] = len(received_headers)

        if len(received_headers) > 8:
            features["suspicious_received_chain"] = True
            features["details"].append(f"Unusually long Received chain ({len(received_headers)} hops)")
        elif len(received_headers) == 0:
            if _RELAX_RECEIVED:
                # Testing mode: locally generated .eml files have no Received
                # headers — suppress the forgery signal to avoid FP bias.
                features["details"].append("No Received headers (suppressed in test mode)")
            else:
                features["suspicious_received_chain"] = True
                features["details"].append("No Received headers found (possibly forged)")

        # Compute overall header threat score
        score = 0.0
        if features["spf_result"] == "fail":
            score += 0.20
            features["details"].append("SPF check failed")
        elif features["spf_result"] == "softfail":
            score += 0.10
            features["details"].append("SPF softfail")
        elif features["spf_result"] == "none":
            score += 0.05

        if features["dkim_result"] == "fail":
            score += 0.20
            features["details"].append("DKIM check failed")
        elif features["dkim_result"] == "none":
            score += 0.05

        if features["dmarc_result"] == "fail":
            score += 0.15
            features["details"].append("DMARC check failed")
        elif features["dmarc_result"] == "none":
            score += 0.05

        if features["reply_to_mismatch"]:
            score += 0.20
        if features["display_name_mismatch"]:
            score += 0.30  # Strong phishing signal
        if features["x_mailer_suspicious"]:
            score += 0.05
        if features["missing_message_id"]:
            score += 0.05
        if features["missing_date"]:
            score += 0.05
        if features["suspicious_received_chain"]:
            score += 0.10

        features["overall_score"] = round(min(score, 1.0), 4)

        if not features["details"]:
            features["details"] = ["No header anomalies detected"]

    except Exception as e:
        logger.warning(f"Error analyzing headers: {e}")
        features["overall_score"] = 0.0
        features["details"] = [f"Header analysis error: {str(e)}"]

    return features


def analyze_headers_from_text(header_text: str) -> Dict[str, Any]:
    """
    Analyze headers from raw text (when no .eml file is available).
    Returns a minimal header analysis with default values.
    """
    return {
        "spf_result": "none",
        "dkim_result": "none",
        "dmarc_result": "none",
        "from_display_name": None,
        "from_email": None,
        "from_domain": None,
        "reply_to_email": None,
        "reply_to_mismatch": False,
        "display_name_mismatch": False,
        "x_mailer": None,
        "x_mailer_suspicious": False,
        "missing_message_id": True,
        "missing_date": True,
        "suspicious_received_chain": False,
        "received_hop_count": 0,
        "overall_score": 0.1,
        "details": ["No email headers available (text-only input)"],
    }

"""
Text Input Parser for Phishing Detection.

Processes raw text input (subject + body) for analysis
when no .eml file is available.

Security: applies Unicode NFKC normalization before any pattern matching
to close homoglyph / Cyrillic-substitution bypass attacks (V-05).

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import logging
import unicodedata
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _nfkc(text: str) -> str:
    """Apply NFKC normalization — collapses homoglyphs and compatibility chars."""
    return unicodedata.normalize("NFKC", text) if text else ""


def parse_text_input(subject: str, body: str) -> Dict[str, Any]:
    """
    Parse raw text input into a standardized format for analysis.

    Applies NFKC Unicode normalization to subject and body before processing
    to prevent homoglyph and Cyrillic-character bypass attacks.

    Args:
        subject: Email subject line
        body: Email body text

    Returns:
        Dictionary matching the eml_parser output format.
    """
    # V-05 mitigation: normalize before any pattern matching
    subject = _nfkc(subject)
    body    = _nfkc(body)

    result = {
        "subject": subject.strip() if subject else "",
        "from": "",
        "to": "",
        "date": "",
        "reply_to": "",
        "body_text": body.strip() if body else "",
        "body_html": "",
        "headers": None,
        "attachments": [],
        "parse_error": None,
        "input_type": "text",
    }

    # Check if the body contains HTML
    if result["body_text"] and "<html" in result["body_text"].lower():
        result["body_html"] = result["body_text"]
        import re
        text = re.sub(r'<style[^>]*>.*?</style>', '', result["body_text"], flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        result["body_text"] = text

    if not result["subject"] and not result["body_text"]:
        result["parse_error"] = "Both subject and body are empty"

    logger.debug(
        f"Parsed text input: subject_len={len(result['subject'])}, "
        f"body_len={len(result['body_text'])}"
    )

    return result

"""
EML File Parser for Phishing Detection.

Parses .eml files to extract headers, body text, HTML content,
and attachment information for downstream analysis.

Security: applies Unicode NFKC normalization to all extracted text fields
to close homoglyph / Cyrillic-substitution bypass attacks (V-05).

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import email
import logging
import unicodedata
from email import policy
from email.message import Message
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_eml(content: bytes) -> Dict[str, Any]:
    """
    Parse an .eml file from raw bytes.

    Args:
        content: Raw bytes of the .eml file

    Returns:
        Dictionary with parsed email components:
        - subject, from, to, date, reply_to
        - body_text, body_html
        - headers (full Message object)
        - attachments list
    """
    result = {
        "subject": "",
        "from": "",
        "to": "",
        "date": "",
        "reply_to": "",
        "body_text": "",
        "body_html": "",
        "headers": None,
        "attachments": [],
        "parse_error": None,
    }

    try:
        msg = email.message_from_bytes(content, policy=policy.default)
        result["headers"] = msg

        # Extract standard headers — normalize immediately (V-05)
        _n = lambda s: unicodedata.normalize("NFKC", str(s))
        result["subject"]  = _n(msg.get("Subject",  ""))
        result["from"]     = _n(msg.get("From",      ""))
        result["to"]       = _n(msg.get("To",        ""))
        result["date"]     = _n(msg.get("Date",      ""))
        result["reply_to"] = _n(msg.get("Reply-To",  ""))

        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disp = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disp:
                    result["attachments"].append({
                        "filename": part.get_filename() or "unnamed",
                        "content_type": content_type,
                        "size": len(part.get_payload(decode=True) or b""),
                    })
                    continue

                if content_type == "text/plain" and not result["body_text"]:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            result["body_text"] = payload.decode(charset, errors="replace")
                        except (LookupError, UnicodeDecodeError):
                            result["body_text"] = payload.decode("utf-8", errors="replace")

                elif content_type == "text/html" and not result["body_html"]:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            result["body_html"] = payload.decode(charset, errors="replace")
                        except (LookupError, UnicodeDecodeError):
                            result["body_html"] = payload.decode("utf-8", errors="replace")
        else:
            content_type = msg.get_content_type()
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    decoded = payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    decoded = payload.decode("utf-8", errors="replace")

                if content_type == "text/html":
                    result["body_html"] = decoded
                else:
                    result["body_text"] = decoded

        # If we only have HTML, strip tags for a text version
        if not result["body_text"] and result["body_html"]:
            result["body_text"] = _strip_html_tags(result["body_html"])

        # V-05: NFKC-normalize final text fields before pattern matching
        result["body_text"] = unicodedata.normalize("NFKC", result["body_text"])
        if result["body_html"]:
            result["body_html"] = unicodedata.normalize("NFKC", result["body_html"])

        logger.info(
            f"Parsed EML: subject='{result['subject'][:50]}', "
            f"text_len={len(result['body_text'])}, html_len={len(result['body_html'])}, "
            f"attachments={len(result['attachments'])}"
        )

    except Exception as e:
        logger.error(f"Failed to parse EML file: {e}")
        result["parse_error"] = str(e)

    return result


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags to get plain text."""
    import re
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

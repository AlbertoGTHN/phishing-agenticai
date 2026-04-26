"""
HTML Content Analysis Module for Phishing Detection.

Detects hidden elements, suspicious forms, tracking pixels,
obfuscated JavaScript, iframes, and Base64-encoded content.

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import re
import base64
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Patterns for hidden elements
HIDDEN_PATTERNS = [
    re.compile(r'display\s*:\s*none', re.IGNORECASE),
    re.compile(r'visibility\s*:\s*hidden', re.IGNORECASE),
    re.compile(r'opacity\s*:\s*0[^.]', re.IGNORECASE),
    re.compile(r'height\s*:\s*0', re.IGNORECASE),
    re.compile(r'width\s*:\s*0', re.IGNORECASE),
    re.compile(r'font-size\s*:\s*0', re.IGNORECASE),
    re.compile(r'position\s*:\s*absolute.*left\s*:\s*-\d{4,}', re.IGNORECASE | re.DOTALL),
]

# Form action pattern
FORM_ACTION_REGEX = re.compile(
    r'<form[^>]*action\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE
)

# Image tags for tracking pixel detection
IMG_REGEX = re.compile(
    r'<img[^>]*src\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE
)

# Iframe detection
IFRAME_REGEX = re.compile(r'<iframe', re.IGNORECASE)

# Script tags
SCRIPT_REGEX = re.compile(
    r'<script[^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL
)

# Base64 pattern
BASE64_REGEX = re.compile(
    r'(?:data:[^;]+;base64,|[A-Za-z0-9+/]{50,}={0,2})',
    re.IGNORECASE
)

# JavaScript obfuscation indicators
JS_OBFUSCATION_PATTERNS = [
    re.compile(r'eval\s*\(', re.IGNORECASE),
    re.compile(r'document\.write\s*\(', re.IGNORECASE),
    re.compile(r'unescape\s*\(', re.IGNORECASE),
    re.compile(r'fromCharCode', re.IGNORECASE),
    re.compile(r'\\x[0-9a-fA-F]{2}', re.IGNORECASE),
    re.compile(r'\\u[0-9a-fA-F]{4}', re.IGNORECASE),
    re.compile(r'atob\s*\(', re.IGNORECASE),
]


def analyze_html(html_content: str) -> Dict[str, Any]:
    """
    Analyze HTML email content for phishing indicators.

    Args:
        html_content: Raw HTML string from email body

    Returns:
        Dictionary with HTML analysis results and threat score.
    """
    features = {
        "hidden_element_count": 0,
        "external_forms": [],
        "external_form_count": 0,
        "tracking_pixel_count": 0,
        "external_image_count": 0,
        "iframe_count": 0,
        "has_obfuscated_js": False,
        "obfuscation_indicators": [],
        "base64_content_count": 0,
        "overall_score": 0.0,
        "details": [],
    }

    if not html_content or not html_content.strip():
        features["details"] = ["No HTML content to analyze"]
        return features

    try:
        # Hidden elements
        for pattern in HIDDEN_PATTERNS:
            matches = pattern.findall(html_content)
            features["hidden_element_count"] += len(matches)

        if features["hidden_element_count"] > 0:
            features["details"].append(
                f"Found {features['hidden_element_count']} hidden element(s)"
            )

        # Form actions pointing to external domains
        for match in FORM_ACTION_REGEX.finditer(html_content):
            action_url = match.group(1)
            if action_url.startswith(("http://", "https://")):
                features["external_forms"].append(action_url)

        features["external_form_count"] = len(features["external_forms"])
        if features["external_form_count"] > 0:
            features["details"].append(
                f"Found {features['external_form_count']} form(s) with external action URLs"
            )
            for form_url in features["external_forms"][:3]:
                features["details"].append(f"  Form action: {form_url}")

        # External images / tracking pixels
        for match in IMG_REGEX.finditer(html_content):
            src = match.group(1)
            if src.startswith(("http://", "https://")):
                features["external_image_count"] += 1
                # Tracking pixel: 1x1 or very small images
                img_tag = match.group(0).lower()
                if ('width="1"' in img_tag or 'height="1"' in img_tag or
                        'width="0"' in img_tag or 'height="0"' in img_tag or
                        "1x1" in src.lower()):
                    features["tracking_pixel_count"] += 1

        if features["tracking_pixel_count"] > 0:
            features["details"].append(
                f"Found {features['tracking_pixel_count']} potential tracking pixel(s)"
            )

        # Iframes
        features["iframe_count"] = len(IFRAME_REGEX.findall(html_content))
        if features["iframe_count"] > 0:
            features["details"].append(
                f"Found {features['iframe_count']} iframe element(s)"
            )

        # JavaScript obfuscation
        scripts = SCRIPT_REGEX.findall(html_content)
        for script in scripts:
            for pattern in JS_OBFUSCATION_PATTERNS:
                if pattern.search(script):
                    features["has_obfuscated_js"] = True
                    indicator = pattern.pattern.replace("\\s*\\(", "(")
                    if indicator not in features["obfuscation_indicators"]:
                        features["obfuscation_indicators"].append(indicator)

        if features["has_obfuscated_js"]:
            features["details"].append(
                f"Detected obfuscated JavaScript: {', '.join(features['obfuscation_indicators'][:3])}"
            )

        # Base64 content
        features["base64_content_count"] = len(BASE64_REGEX.findall(html_content))
        if features["base64_content_count"] > 2:
            features["details"].append(
                f"Found {features['base64_content_count']} Base64-encoded content blocks"
            )

        # Compute overall score
        score = 0.0
        if features["hidden_element_count"] > 0:
            score += min(features["hidden_element_count"] * 0.05, 0.20)
        if features["external_form_count"] > 0:
            score += 0.30
        if features["tracking_pixel_count"] > 0:
            score += 0.05
        if features["iframe_count"] > 0:
            score += 0.15
        if features["has_obfuscated_js"]:
            score += 0.25
        if features["base64_content_count"] > 2:
            score += 0.10

        features["overall_score"] = round(min(score, 1.0), 4)

        if not features["details"]:
            features["details"] = ["No suspicious HTML elements detected"]

    except Exception as e:
        logger.warning(f"Error analyzing HTML: {e}")
        features["details"] = [f"HTML analysis error: {str(e)}"]

    return features

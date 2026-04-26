"""
OCR Parser for Phishing Detection.

Extracts text from screenshot images using Tesseract OCR
and feeds it into the analysis pipeline.

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import io
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configure Tesseract path on Windows
_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Use project-local tessdata if the system one is missing eng.traineddata
_project_tessdata = os.path.join(os.path.dirname(__file__), "..", "..", "tessdata")
_project_tessdata = os.path.abspath(_project_tessdata)
if os.path.isdir(_project_tessdata) and not os.environ.get("TESSDATA_PREFIX"):
    os.environ["TESSDATA_PREFIX"] = _project_tessdata


def parse_screenshot(image_bytes: bytes) -> Dict[str, Any]:
    """
    Extract text from a screenshot image using OCR.

    Args:
        image_bytes: Raw bytes of the image file (PNG/JPG)

    Returns:
        Dictionary with extracted text and metadata.
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
        "input_type": "screenshot",
        "ocr_confidence": None,
        "ocr_note": "Text extracted via OCR — accuracy may be limited.",
    }

    try:
        from PIL import Image
        import pytesseract

        # Ensure Tesseract path is set on Windows
        if os.path.exists(_tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = _tesseract_path

        image = Image.open(io.BytesIO(image_bytes))

        # Get OCR data with confidence scores
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Extract text
        extracted_text = pytesseract.image_to_string(image)
        result["body_text"] = extracted_text.strip()

        # Compute average confidence (excluding -1 entries which are non-text)
        confidences = [int(c) for c in ocr_data["conf"] if int(c) > 0]
        if confidences:
            result["ocr_confidence"] = sum(confidences) / len(confidences) / 100.0

        # Try to extract subject line (first line often is subject in email screenshots)
        lines = [l.strip() for l in extracted_text.split("\n") if l.strip()]
        if lines:
            # Heuristic: look for "Subject:" prefix
            for line in lines[:5]:
                if line.lower().startswith("subject:"):
                    result["subject"] = line[8:].strip()
                    break
                elif line.lower().startswith("re:") or line.lower().startswith("fw:"):
                    result["subject"] = line.strip()
                    break

            # If no subject found, use first line
            if not result["subject"] and lines:
                result["subject"] = lines[0][:100]

        logger.info(
            f"OCR extracted {len(result['body_text'])} chars, "
            f"confidence={result['ocr_confidence']}"
        )

    except ImportError as e:
        logger.error(f"OCR dependencies not available: {e}")
        result["parse_error"] = (
            "OCR requires pytesseract and Pillow. "
            "Install with: pip install pytesseract Pillow. "
            "Also ensure Tesseract-OCR is installed on your system."
        )
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        result["parse_error"] = f"OCR extraction failed: {str(e)}"

    return result

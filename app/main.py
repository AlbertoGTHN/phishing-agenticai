"""
FastAPI Application — Agentic AI Phishing Detector.

Main entry point providing REST API endpoints for email analysis
via .eml upload, raw text input, and screenshot OCR.
Includes KPI dashboard stats and Claude-powered security chatbot.

Serves the single-page frontend and handles all analysis orchestration
through the dual-engine MCO (Monitoring, Classification, Optimization) architecture.

Security hardening (ICITS paper — adversarial robustness mitigations):
  V-01  API key authentication on all training endpoints (6A)
  V-02  Rate limiting: /api/analyze/* 30 req/min per IP (6B)
  V-03  Rate limiting: /api/train/*   5  req/min per IP (6B)
  V-04  raw_features stripped from public responses unless EXPOSE_RAW_FEATURES=true (6C)
  V-05  Unicode NFKC normalization in parsers (see parsers/) (6D)
  V-06  Training data integrity: mislabeled high-confidence samples quarantined (6E)

Environment variables:
  DETECTOR_ADMIN_KEY   — required for training endpoints (empty = open, dev mode)
  EXPOSE_RAW_FEATURES  — "true" to include raw feature dict in analyze responses
                         (default false)

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import csv
import hashlib
import io
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import (
    Depends, FastAPI, Header, HTTPException, Request, UploadFile, File,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# ── slowapi rate limiting (V-02 / V-03) ────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.parsers.eml_parser import parse_eml
from app.parsers.text_parser import parse_text_input
from app.parsers.ocr_parser import parse_screenshot
from app.models.semantic_engine import analyze_semantics
from app.models.structural_engine import analyze_structure
from app.models.classifier import (
    classify, get_recommended_action, add_training_sample,
    train_model, get_training_stats,
)
from app.models.explainer import (
    generate_explanation,
    generate_education_note,
    build_threat_indicators,
)
from app.database import (
    save_scan, get_scan_history, get_scan_by_id, get_scan_stats,
    save_training_sample, get_training_samples, get_training_sample_counts,
    # V-06 additions
    save_label_submission, get_suspicious_label_count, confirm_suspicious_label,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phishing-detector")

# ── Security configuration ───────────────────────────────────────────────────
# V-01: admin key for training endpoints (empty string = dev/open mode)
_ADMIN_KEY: str = os.environ.get("DETECTOR_ADMIN_KEY", "")

# V-04: whether to include raw_features in analyze responses
_EXPOSE_RAW_FEATURES: bool = (
    os.environ.get("EXPOSE_RAW_FEATURES", "false").strip().lower() == "true"
)

if _ADMIN_KEY:
    logger.info("Security: DETECTOR_ADMIN_KEY is set — training endpoints protected")
else:
    logger.warning(
        "Security: DETECTOR_ADMIN_KEY not set — training endpoints are OPEN. "
        "Set this variable in production."
    )

if not _EXPOSE_RAW_FEATURES:
    logger.info("Security: raw_features suppressed in public responses (EXPOSE_RAW_FEATURES=false)")

# ── Rate limiter (V-02, V-03) ────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic AI Phishing Detector",
    description="Dual-engine phishing detection system using NLP and structural analysis.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Static files — handle PyInstaller frozen mode
if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS
else:
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(_BASE, "app", "static")


# ── Security dependencies ─────────────────────────────────────────────────────

def _require_admin_key(x_admin_key: Optional[str] = Header(None)) -> None:
    """
    V-01: Enforce API key authentication for training endpoints.

    If DETECTOR_ADMIN_KEY env var is set, every request to a training endpoint
    must supply the matching value in the X-Admin-Key header.
    If the env var is empty the check is skipped (development mode).
    """
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        logger.warning("Rejected training request — invalid or missing X-Admin-Key")
        raise HTTPException(
            status_code=401,
            detail=(
                "Unauthorized. Provide the correct admin key in the "
                "X-Admin-Key request header."
            ),
        )


# ── Request / response models ─────────────────────────────────────────────────

class TextAnalysisRequest(BaseModel):
    subject: str = ""
    body: str = ""


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None


class TrainLabelRequest(BaseModel):
    subject: str = ""
    body: str = ""
    label: int  # 0 = legitimate, 1 = phishing


# ── Analysis pipeline ─────────────────────────────────────────────────────────

def run_analysis_pipeline(parsed_email: dict, input_type: str = "unknown") -> dict:
    """
    Run the full MCO analysis pipeline on a parsed email.

    V-04: raw_features is included in the returned dict only when
    EXPOSE_RAW_FEATURES=true; otherwise it is omitted from the response
    to prevent information disclosure to unauthenticated callers.
    """
    start_time = time.time()

    subject   = parsed_email.get("subject", "")
    body_text = parsed_email.get("body_text", "")

    # Engine A: Semantic analysis
    logger.info("Running semantic analysis (Engine A)...")
    semantic_results = analyze_semantics(subject, body_text)

    # Engine B: Structural analysis
    logger.info("Running structural analysis (Engine B)...")
    structural_results = analyze_structure(parsed_email)

    # Classification
    logger.info("Running classification...")
    classification = classify(semantic_results, structural_results)
    verdict    = classification["verdict"]
    confidence = classification["confidence"]

    # Recommended action
    action = get_recommended_action(verdict, confidence)

    # Explainability
    explanation      = generate_explanation(verdict, confidence, semantic_results, structural_results)
    education_note   = generate_education_note(verdict, semantic_results, structural_results)
    threat_indicators = build_threat_indicators(semantic_results, structural_results)

    elapsed = time.time() - start_time

    # Build raw features for technical / debug use
    raw_features = {
        "semantic": {
            "overall_score":              semantic_results.get("overall_score"),
            "has_transformer_embedding":  semantic_results.get("has_embedding", False),
            "rule_scores": {
                k: {
                    "score":   v.get("score"),
                    "count":   v.get("count"),
                    "matches": v.get("matches", v.get("issues", [])),
                }
                for k, v in semantic_results.get("rule_scores", {}).items()
                if isinstance(v, dict)
            },
        },
        "structural": {
            "url_count":    structural_results.get("url_analysis",    {}).get("url_count",     0),
            "url_score":    structural_results.get("url_analysis",    {}).get("overall_score",  0),
            "header_score": structural_results.get("header_analysis", {}).get("overall_score",  0),
            "html_score":   structural_results.get("html_analysis",   {}).get("overall_score",  0),
            "spf":          structural_results.get("header_analysis", {}).get("spf_result"),
            "dkim":         structural_results.get("header_analysis", {}).get("dkim_result"),
            "dmarc":        structural_results.get("header_analysis", {}).get("dmarc_result"),
        },
        "classification":          classification,
        "analysis_time_seconds":   round(elapsed, 3),
        "timestamp":               datetime.utcnow().isoformat() + "Z",
    }

    # Add OCR / parse annotations
    ocr_note = parsed_email.get("ocr_note")
    if ocr_note:
        raw_features["ocr_note"]        = ocr_note
        raw_features["ocr_confidence"]  = parsed_email.get("ocr_confidence")
    if parsed_email.get("parse_error"):
        raw_features["parse_warning"] = parsed_email["parse_error"]

    logger.info(
        f"Analysis complete: verdict={verdict}, confidence={confidence:.3f}, "
        f"method={classification.get('method')}, time={elapsed:.2f}s"
    )

    result = {
        "verdict":              verdict,
        "confidence":           confidence,
        "threat_indicators":    threat_indicators,
        "explanation":          explanation,
        "recommended_action":   action["action"],
        "action_description":   action["description"],
        "education_note":       education_note,
    }

    # V-04: only expose raw_features when explicitly enabled
    if _EXPOSE_RAW_FEATURES:
        result["raw_features"] = raw_features

    # Persist to database (always stores raw_features internally for audit)
    try:
        scan_id = save_scan(
            input_type=input_type,
            subject=subject,
            body=body_text,
            verdict=verdict,
            confidence=confidence,
            recommended_action=action["action"],
            explanation=explanation,
            education_note=education_note,
            threat_indicators=threat_indicators,
            raw_features=raw_features,
        )
        result["scan_id"] = scan_id
    except Exception as e:
        logger.warning(f"Failed to save scan to database: {e}")

    return result


# ── Helper: extract client IP ─────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    """Return the real client IP, respecting X-Forwarded-For if present."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status":    "healthy",
        "service":   "Agentic AI Phishing Detector",
        "version":   "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ── Analysis endpoints (V-02: 30 req/min per IP) ─────────────────────────────

@app.post("/api/analyze/eml")
@limiter.limit("30/minute")
async def analyze_eml(request: Request, file: UploadFile = File(...)):
    """Analyze an uploaded .eml file for phishing indicators."""
    logger.info(f"Received .eml file: {file.filename}")

    if not file.filename or not file.filename.lower().endswith(".eml"):
        raise HTTPException(status_code=400, detail="Please upload a .eml file")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        parsed = parse_eml(content)
        if parsed.get("parse_error") and not parsed.get("body_text"):
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse .eml file: {parsed['parse_error']}",
            )

        return run_analysis_pipeline(parsed, input_type="eml")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing .eml file: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze/text")
@limiter.limit("30/minute")
async def analyze_text(request: Request, data: TextAnalysisRequest):
    """Analyze raw email text (subject + body) for phishing indicators."""
    logger.info(f"Received text analysis: subject_len={len(data.subject)}")

    if not data.subject.strip() and not data.body.strip():
        raise HTTPException(status_code=400, detail="Please provide a subject or body text")

    try:
        parsed = parse_text_input(data.subject, data.body)
        return run_analysis_pipeline(parsed, input_type="text")

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze/screenshot")
@limiter.limit("30/minute")
async def analyze_screenshot(request: Request, file: UploadFile = File(...)):
    """Analyze a screenshot of an email using OCR extraction."""
    logger.info(f"Received screenshot: {file.filename}")

    valid_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type and file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPG image file",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        parsed = parse_screenshot(content)
        if parsed.get("parse_error") and not parsed.get("body_text"):
            raise HTTPException(
                status_code=400,
                detail=f"OCR extraction failed: {parsed['parse_error']}",
            )

        result = run_analysis_pipeline(parsed, input_type="screenshot")

        if parsed.get("ocr_note"):
            result["ocr_note"] = parsed["ocr_note"]

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing screenshot: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── Dashboard / scan history endpoints (no rate limit — read-only) ────────────

@app.get("/api/dashboard/stats")
async def dashboard_stats():
    """Return KPI statistics from database."""
    stats  = get_scan_stats()
    recent = get_scan_history(limit=20)
    stats["recent_analyses"] = recent
    return stats


@app.get("/api/scans")
async def list_scans(limit: int = 100, offset: int = 0):
    """Retrieve scan history with pagination."""
    scans = get_scan_history(limit=limit, offset=offset)
    stats = get_scan_stats()
    return {"scans": scans, "total": stats["total_analyzed"]}


@app.get("/api/scans/{scan_id}")
async def get_scan(scan_id: int):
    """Retrieve a single scan by ID."""
    scan = get_scan_by_id(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@app.get("/api/export/history")
async def export_history():
    """Export full scan history as a downloadable CSV file."""
    scans = get_scan_history(limit=100_000, offset=0)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "timestamp", "input_type", "subject", "sender",
        "verdict", "confidence", "recommended_action",
        "suspicious_urls_score", "urgency_language_score",
        "header_anomalies_score", "grammatical_anomalies_score",
        "html_suspicious_score",
    ])
    for s in scans:
        ti = s.get("threat_indicators", {}) or {}
        writer.writerow([
            s.get("id", ""),
            s.get("timestamp", ""),
            s.get("input_type", ""),
            s.get("subject", ""),
            s.get("sender", ""),
            s.get("verdict", ""),
            round(s.get("confidence", 0) * 100, 2),
            s.get("recommended_action", ""),
            round(ti.get("suspicious_urls",   {}).get("score", 0) * 100, 2),
            round(ti.get("urgency_language",  {}).get("score", 0) * 100, 2),
            round(ti.get("header_anomalies",  {}).get("score", 0) * 100, 2),
            round(ti.get("grammatical_anomalies", {}).get("score", 0) * 100, 2),
            round(ti.get("html_suspicious",   {}).get("score", 0) * 100, 2),
        ])
    output.seek(0)
    filename = f"phishing_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Chat endpoint (Claude API) ────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: Request, data: ChatRequest, x_api_key: Optional[str] = Header(None)):
    """Chat with Claude about phishing and email security."""
    api_key = x_api_key
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Please provide your Anthropic API key in the chat settings.",
        )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = (
            "You are a cybersecurity expert specializing in phishing detection and email security. "
            "You are embedded in an Agentic AI Phishing Detection system built on a dual-engine "
            "MCO (Monitoring, Classification, Optimization) architecture. "
            "Help users understand phishing threats, explain analysis results, and provide "
            "actionable security advice. Be concise and practical. "
            "If the user provides analysis context, reference the specific indicators and scores."
        )

        messages = [{"role": "user", "content": data.message}]

        if data.context:
            context_str = (
                f"Latest analysis result:\n"
                f"- Verdict: {data.context.get('verdict', 'N/A')}\n"
                f"- Confidence: {data.context.get('confidence', 'N/A')}\n"
                f"- Action: {data.context.get('recommended_action', 'N/A')}\n"
                f"- Explanation: {data.context.get('explanation', 'N/A')}\n"
            )
            threat = data.context.get("threat_indicators", {})
            if threat:
                context_str += "- Threat indicators:\n"
                for k, v in threat.items():
                    score = v.get("score", 0) if isinstance(v, dict) else v
                    context_str += f"  - {k}: {score}\n"

            messages = [
                {"role": "user",
                 "content": f"[Analysis Context]\n{context_str}\n\n[Question]\n{data.message}"}
            ]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

        reply = response.content[0].text
        return {"reply": reply}

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="The 'anthropic' package is not installed. Run: pip install anthropic",
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ── Training endpoints (V-01: admin key; V-03: 5 req/min per IP; V-06: integrity) ──

@app.post("/api/train/label")
@limiter.limit("5/minute")
async def label_sample(
    request: Request,
    data: TrainLabelRequest,
    _auth: None = Depends(_require_admin_key),
):
    """
    Submit a labeled email sample for training.

    Runs the analysis pipeline, then — before persisting — checks training
    data integrity (V-06): if the pipeline is highly confident the email is
    phishing (>70%) but the submitted label claims it is legitimate (0), the
    sample is quarantined as SUSPICIOUS_LABEL and only admitted to training
    after a second identical submission confirms the label.

    All submissions are logged with submitter IP and timestamp.
    Label: 0 = legitimate, 1 = phishing.
    """
    if data.label not in (0, 1):
        raise HTTPException(status_code=400, detail="Label must be 0 (legitimate) or 1 (phishing)")

    if not data.subject.strip() and not data.body.strip():
        raise HTTPException(status_code=400, detail="Please provide subject or body text")

    client_ip = _client_ip(request)

    try:
        parsed    = parse_text_input(data.subject, data.body)
        subject   = parsed.get("subject", "")
        body_text = parsed.get("body_text", "")

        semantic_results   = analyze_semantics(subject, body_text)
        structural_results = analyze_structure(parsed)
        classification     = classify(semantic_results, structural_results)

        pipeline_confidence = float(classification.get("confidence", 0.0))
        pipeline_verdict    = classification.get("verdict", "unknown")

        # ── V-06: Training data integrity check ──────────────────────────────
        body_hash           = hashlib.sha256(body_text.encode("utf-8", errors="replace")).hexdigest()
        suspicious_flagged  = (data.label == 0 and pipeline_confidence > 0.70)

        if suspicious_flagged:
            prior_count = get_suspicious_label_count(body_hash)
            save_label_submission(
                submitter_ip=client_ip,
                subject=subject,
                body_hash=body_hash,
                submitted_label=data.label,
                pipeline_confidence=pipeline_confidence,
                pipeline_verdict=pipeline_verdict,
                suspicious_flagged=True,
                confirmed=(prior_count >= 1),
            )

            if prior_count == 0:
                # First suspicious submission — quarantine, do NOT train
                logger.warning(
                    f"SUSPICIOUS_LABEL quarantine: ip={client_ip} "
                    f"conf={pipeline_confidence:.3f} label={data.label} hash={body_hash[:12]}…"
                )
                raise HTTPException(
                    status_code=202,
                    detail={
                        "suspicious_label": True,
                        "message": (
                            "Sample flagged as SUSPICIOUS_LABEL: the pipeline detects this "
                            "email as phishing with high confidence but it was submitted as "
                            "legitimate. Re-submit with the same label to confirm and include "
                            "it in training."
                        ),
                        "pipeline_verdict":    pipeline_verdict,
                        "pipeline_confidence": round(pipeline_confidence, 4),
                    },
                )
            else:
                # Second (or later) submission — confirmed, proceed to train
                confirm_suspicious_label(body_hash)
                logger.info(
                    f"SUSPICIOUS_LABEL confirmed by re-submission: ip={client_ip} "
                    f"hash={body_hash[:12]}… — admitting to training"
                )
        else:
            # Normal submission
            save_label_submission(
                submitter_ip=client_ip,
                subject=subject,
                body_hash=body_hash,
                submitted_label=data.label,
                pipeline_confidence=pipeline_confidence,
                pipeline_verdict=pipeline_verdict,
                suspicious_flagged=False,
            )

        # ── Persist to in-memory training buffer + database ──────────────────
        add_training_sample(semantic_results, structural_results, data.label)

        embedding            = semantic_results.get("embedding")
        structural_features  = structural_results.get("structural_features", [])
        if embedding is not None:
            features = embedding + structural_features
        else:
            from app.models.classifier import _rule_scores_to_features
            features = (
                _rule_scores_to_features(semantic_results.get("rule_scores", {}))
                + structural_features
            )
        save_training_sample(subject, body_text, data.label, features)

        stats = get_training_stats()
        return {
            "success": True,
            "message": f"Sample labeled as {'phishing' if data.label == 1 else 'legitimate'}",
            **stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error labeling sample: {e}")
        raise HTTPException(status_code=500, detail=f"Labeling failed: {str(e)}")


@app.post("/api/train/label/eml")
@limiter.limit("5/minute")
async def label_eml_sample(
    request: Request,
    file: UploadFile = File(...),
    label: int = 1,
    _auth: None = Depends(_require_admin_key),
):
    """
    Submit a labeled .eml file for training.

    Applies the same V-06 integrity check as /api/train/label.
    """
    if label not in (0, 1):
        raise HTTPException(status_code=400, detail="Label must be 0 (legitimate) or 1 (phishing)")

    client_ip = _client_ip(request)

    try:
        content = await file.read()
        parsed  = parse_eml(content)
        if parsed.get("parse_error") and not parsed.get("body_text"):
            raise HTTPException(status_code=400, detail=f"Parse failed: {parsed['parse_error']}")

        subject   = parsed.get("subject", "")
        body_text = parsed.get("body_text", "")

        semantic_results   = analyze_semantics(subject, body_text)
        structural_results = analyze_structure(parsed)
        classification     = classify(semantic_results, structural_results)

        pipeline_confidence = float(classification.get("confidence", 0.0))
        pipeline_verdict    = classification.get("verdict", "unknown")

        # ── V-06: Training data integrity check ──────────────────────────────
        body_hash          = hashlib.sha256(body_text.encode("utf-8", errors="replace")).hexdigest()
        suspicious_flagged = (label == 0 and pipeline_confidence > 0.70)

        if suspicious_flagged:
            prior_count = get_suspicious_label_count(body_hash)
            save_label_submission(
                submitter_ip=client_ip,
                subject=subject,
                body_hash=body_hash,
                submitted_label=label,
                pipeline_confidence=pipeline_confidence,
                pipeline_verdict=pipeline_verdict,
                suspicious_flagged=True,
                confirmed=(prior_count >= 1),
            )

            if prior_count == 0:
                logger.warning(
                    f"SUSPICIOUS_LABEL (.eml) quarantine: ip={client_ip} "
                    f"conf={pipeline_confidence:.3f} label={label}"
                )
                raise HTTPException(
                    status_code=202,
                    detail={
                        "suspicious_label": True,
                        "message": (
                            "Sample flagged as SUSPICIOUS_LABEL. Re-submit to confirm."
                        ),
                        "pipeline_verdict":    pipeline_verdict,
                        "pipeline_confidence": round(pipeline_confidence, 4),
                    },
                )
            else:
                confirm_suspicious_label(body_hash)
                logger.info(f"SUSPICIOUS_LABEL (.eml) confirmed: ip={client_ip}")
        else:
            save_label_submission(
                submitter_ip=client_ip,
                subject=subject,
                body_hash=body_hash,
                submitted_label=label,
                pipeline_confidence=pipeline_confidence,
                pipeline_verdict=pipeline_verdict,
                suspicious_flagged=False,
            )

        add_training_sample(semantic_results, structural_results, label)
        stats = get_training_stats()
        return {
            "success": True,
            "message": f"EML labeled as {'phishing' if label == 1 else 'legitimate'}",
            **stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Labeling failed: {str(e)}")


@app.post("/api/train/run")
@limiter.limit("5/minute")
async def run_training(
    request: Request,
    _auth: None = Depends(_require_admin_key),
):
    """Train the Random Forest classifier on accumulated labeled samples."""
    result = train_model(min_samples=6)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Training failed"))
    return result


@app.get("/api/train/stats")
async def training_stats():
    """Get current training data statistics (public read-only endpoint)."""
    return get_training_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

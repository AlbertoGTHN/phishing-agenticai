# Security Hardening — Agentic AI Phishing Detector

This document lists every vulnerability identified during the adversarial evaluation
(ICITS 2026 paper) and the corresponding mitigation implemented in the codebase.

---

## V-01 — Unauthenticated Training Endpoint Access

| Field       | Value |
|-------------|-------|
| **Risk**    | High |
| **Surface** | `POST /api/train/label`, `POST /api/train/run`, `POST /api/train/label/eml` |
| **Attack**  | Unauthenticated callers could inject arbitrary labeled samples, poisoning the classifier over time without any credentials. |
| **Fix**     | `6A` — API key middleware in `app/main.py` |

### Implementation

Set the `DETECTOR_ADMIN_KEY` environment variable to a strong secret before starting
the server.  Every request to a training endpoint must supply the same value in the
`X-Admin-Key` HTTP header.  Missing or incorrect keys receive **HTTP 401**.

```bash
# Production startup example
DETECTOR_ADMIN_KEY="$(openssl rand -hex 32)" uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If `DETECTOR_ADMIN_KEY` is not set the server logs a warning and runs in open
(development) mode — all training requests are accepted without authentication.
This allows zero-config local development while enforcing auth in production.

**Code location:** `app/main.py` — `_require_admin_key()` dependency,
`Depends(_require_admin_key)` on all training endpoints.

---

## V-02 — Rate Limit Abuse on Analysis Endpoints

| Field       | Value |
|-------------|-------|
| **Risk**    | Medium |
| **Surface** | `POST /api/analyze/eml`, `/api/analyze/text`, `/api/analyze/screenshot` |
| **Attack**  | Automated scraping or denial-of-service via unrestricted high-volume requests to CPU-intensive analysis endpoints (DistilBERT + RF inference per request). |
| **Fix**     | `6B` — slowapi rate limiter in `app/main.py` |

### Implementation

`slowapi` (built on `limits`) enforces **30 requests per minute per IP** on all
`/api/analyze/*` endpoints.  Exceeding the limit returns **HTTP 429 Too Many Requests**
with a `Retry-After` header indicating when the client may retry.

**Code location:** `app/main.py` — `@limiter.limit("30/minute")` on each analyze endpoint.

---

## V-03 — Rate Limit Abuse on Training Endpoints

| Field       | Value |
|-------------|-------|
| **Risk**    | Medium–High |
| **Surface** | `POST /api/train/label`, `POST /api/train/run`, `POST /api/train/label/eml` |
| **Attack**  | Rapid-fire label submissions (even with a valid admin key) to flood the training buffer or trigger re-training storms that degrade model quality. |
| **Fix**     | `6B` — slowapi rate limiter in `app/main.py` |

### Implementation

**5 requests per minute per IP** on all `/api/train/*` write endpoints, returning
**HTTP 429** with `Retry-After` on violation.  The stricter limit (vs analysis
endpoints) reflects the higher cost and sensitivity of training operations.

**Code location:** `app/main.py` — `@limiter.limit("5/minute")` on each training endpoint.

---

## V-04 — Information Disclosure via raw_features

| Field       | Value |
|-------------|-------|
| **Risk**    | Medium |
| **Surface** | All `/api/analyze/*` responses |
| **Attack**  | The `raw_features` payload exposes internal feature weights, rule match counts, embedding presence flags, and SPF/DKIM verdicts.  An adversary can use this oracle to iteratively craft emails that score just below detection thresholds. |
| **Fix**     | `6C` — `EXPOSE_RAW_FEATURES` environment variable gate in `app/main.py` |

### Implementation

`raw_features` is **omitted from all public API responses** by default.
It is still stored internally in the SQLite database for audit purposes.

To re-enable for research or debugging:

```bash
EXPOSE_RAW_FEATURES=true uvicorn app.main:app ...
```

**Code location:** `app/main.py` — `_EXPOSE_RAW_FEATURES` flag,
`run_analysis_pipeline()` conditional inclusion.

---

## V-05 — Unicode Homoglyph / Cyrillic-Substitution Bypass

| Field       | Value |
|-------------|-------|
| **Risk**    | High |
| **Surface** | All text input paths (`.eml` upload, raw text, OCR output) |
| **Attack**  | Replacing ASCII characters with visually identical Unicode counterparts (e.g., Cyrillic "а" for Latin "a") bypasses regex-based pattern matching.  Keywords like "раypal" or "ассоunt" evade urgency and brand-impersonation rules while appearing identical to a human reader. |
| **Fix**     | `6D` — NFKC normalization in both text parsers |

### Implementation

`unicodedata.normalize("NFKC", text)` is applied **before any pattern matching**
to both subject and body in:

- `app/parsers/text_parser.py` — `parse_text_input()`, via the new `_nfkc()` helper
- `app/parsers/eml_parser.py`  — `parse_eml()`, applied to all decoded text fields
  and to `body_text` / `body_html` after extraction

NFKC (Normalization Form Compatibility Composition) maps compatibility characters
to their canonical equivalents, collapsing homoglyphs and ligatures before the
rule engines see the text.

**Code location:** `app/parsers/text_parser.py`, `app/parsers/eml_parser.py`.

---

## V-06 — Training Data Poisoning via Mislabeled Confident Samples

| Field       | Value |
|-------------|-------|
| **Risk**    | High |
| **Surface** | `POST /api/train/label`, `POST /api/train/label/eml` |
| **Attack**  | An authenticated adversary with the admin key can degrade classifier accuracy by repeatedly submitting high-confidence phishing emails labeled as "legitimate" (label=0), shifting the RF decision boundary toward false negatives over successive re-training cycles. |
| **Fix**     | `6E` — training data integrity check in `app/main.py` + audit log in `app/database.py` |

### Implementation

Before any labeled sample is admitted to the training buffer, the full pipeline
is run to obtain a `pipeline_confidence` score.

**Quarantine trigger:** `submitted_label == 0` AND `pipeline_confidence > 0.70`

1. **First suspicious submission** — the sample is **not** added to training.
   The caller receives **HTTP 202** with `suspicious_label: true` and a message
   explaining that re-submission will confirm the label.
2. **Second (confirming) submission** — the prior entry is marked `confirmed=1`
   in `label_submissions` and the sample is admitted to training normally.

Every submission (clean or suspicious) is logged to the `label_submissions` table
with: timestamp, submitter IP, body SHA-256 hash, submitted label, pipeline
confidence, pipeline verdict, `suspicious_flagged`, and `confirmed` columns.

**Code location:**
- `app/main.py` — integrity check in `label_sample()` and `label_eml_sample()`
- `app/database.py` — `label_submissions` table, `save_label_submission()`,
  `get_suspicious_label_count()`, `confirm_suspicious_label()`

---

## Configuration Reference

| Environment Variable    | Default   | Description |
|-------------------------|-----------|-------------|
| `DETECTOR_ADMIN_KEY`    | *(empty)* | API key required for all training endpoints.  Empty = open/dev mode. |
| `EXPOSE_RAW_FEATURES`   | `false`   | Set to `true` to include internal feature data in analyze responses (research/debug only). |

## Dependencies Added

| Package    | Version | Purpose |
|------------|---------|---------|
| `slowapi`  | ≥0.1    | ASGI-compatible rate limiter for FastAPI (V-02, V-03) |

Install: `pip install slowapi`

---

*Vulnerability IDs V-01 through V-06 correspond to findings in:*
*Loo, Galindo, Romero et al. — "Adversarial Robustness of an Agentic AI Phishing Detector"*
*ICITS 2026 (in preparation).*

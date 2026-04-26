"""
SQLite Database Module for Persistent Storage.

Stores analysis history, training samples, and scan logs
across server restarts.

Security additions:
  label_submissions — audit log for every training-label POST, including IP,
                      pipeline confidence, and suspicious-flag status (V-06).

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    """Get the database path — uses %APPDATA% when running as frozen .exe."""
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle: _MEIPASS is read-only, so use AppData
        app_data = os.path.join(
            os.environ.get('APPDATA', os.path.expanduser('~')),
            'PhishingDetector'
        )
        os.makedirs(app_data, exist_ok=True)
        return os.path.join(app_data, 'phishing_detector.db')
    else:
        # Development mode: use data/ folder relative to project
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "phishing_detector.db")
        )


DB_PATH = _get_db_path()

_connection: Optional[sqlite3.Connection] = None


def get_db() -> sqlite3.Connection:
    """Get or create the database connection."""
    global _connection
    if _connection is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _init_tables(_connection)
        logger.info(f"Database initialized at: {DB_PATH}")
    return _connection


def _init_tables(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_type TEXT NOT NULL DEFAULT 'unknown',
            subject TEXT DEFAULT '',
            body_preview TEXT DEFAULT '',
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            recommended_action TEXT NOT NULL,
            explanation TEXT DEFAULT '',
            education_note TEXT DEFAULT '',
            threat_indicators TEXT DEFAULT '{}',
            raw_features TEXT DEFAULT '{}',
            analysis_time_seconds REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            subject TEXT DEFAULT '',
            body_preview TEXT DEFAULT '',
            label INTEGER NOT NULL,
            features TEXT NOT NULL
        );

        -- V-06: audit log for every label submission (training data integrity)
        CREATE TABLE IF NOT EXISTS label_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            submitter_ip TEXT NOT NULL DEFAULT '',
            subject_preview TEXT DEFAULT '',
            body_hash TEXT NOT NULL,
            submitted_label INTEGER NOT NULL,
            pipeline_confidence REAL NOT NULL DEFAULT 0.0,
            pipeline_verdict TEXT NOT NULL DEFAULT '',
            suspicious_flagged INTEGER NOT NULL DEFAULT 0,
            confirmed INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_scan_verdict ON scan_history(verdict);
        CREATE INDEX IF NOT EXISTS idx_training_label ON training_samples(label);
        CREATE INDEX IF NOT EXISTS idx_submission_hash ON label_submissions(body_hash);
    """)
    conn.commit()


# --- Scan History ---

def save_scan(
    input_type: str,
    subject: str,
    body: str,
    verdict: str,
    confidence: float,
    recommended_action: str,
    explanation: str,
    education_note: str,
    threat_indicators: dict,
    raw_features: dict,
) -> int:
    """Save a scan result to the database. Returns the scan ID."""
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO scan_history
           (timestamp, input_type, subject, body_preview, verdict, confidence,
            recommended_action, explanation, education_note,
            threat_indicators, raw_features, analysis_time_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat() + "Z",
            input_type,
            subject[:200],
            body[:500],
            verdict,
            confidence,
            recommended_action,
            explanation,
            education_note,
            json.dumps(threat_indicators, default=str),
            json.dumps(raw_features, default=str),
            raw_features.get("analysis_time_seconds", 0),
        ),
    )
    conn.commit()
    scan_id = cursor.lastrowid
    logger.info(f"Scan saved: id={scan_id}, verdict={verdict}")
    return scan_id


def get_scan_history(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieve scan history, most recent first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM scan_history ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_scan_by_id(scan_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single scan by ID."""
    conn = get_db()
    row = conn.execute("SELECT * FROM scan_history WHERE id = ?", (scan_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_scan_stats() -> Dict[str, Any]:
    """Get aggregate scan statistics from the database."""
    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
    if total == 0:
        return {
            "total_analyzed": 0,
            "phishing_count": 0,
            "suspicious_count": 0,
            "legitimate_count": 0,
            "phishing_rate": 0,
            "avg_confidence": 0,
            "threat_breakdown": {
                "suspicious_urls": 0, "urgency_language": 0,
                "header_anomalies": 0, "grammatical_anomalies": 0,
                "html_suspicious": 0,
            },
        }

    phishing = conn.execute("SELECT COUNT(*) FROM scan_history WHERE verdict='phishing'").fetchone()[0]
    suspicious = conn.execute("SELECT COUNT(*) FROM scan_history WHERE verdict='suspicious'").fetchone()[0]
    legitimate = conn.execute("SELECT COUNT(*) FROM scan_history WHERE verdict='legitimate'").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM scan_history").fetchone()[0] or 0

    # Threat breakdown from stored indicators
    rows = conn.execute("SELECT threat_indicators FROM scan_history").fetchall()
    indicator_keys = ["suspicious_urls", "urgency_language", "header_anomalies",
                      "grammatical_anomalies", "html_suspicious"]
    threat_totals = {k: 0.0 for k in indicator_keys}
    for row in rows:
        try:
            indicators = json.loads(row[0])
            for k in indicator_keys:
                val = indicators.get(k, {})
                score = val.get("score", 0) if isinstance(val, dict) else (val if isinstance(val, (int, float)) else 0)
                threat_totals[k] += score
        except (json.JSONDecodeError, TypeError):
            pass

    threat_breakdown = {k: round(v / total, 4) for k, v in threat_totals.items()}

    return {
        "total_analyzed": total,
        "phishing_count": phishing,
        "suspicious_count": suspicious,
        "legitimate_count": legitimate,
        "phishing_rate": round(phishing / total * 100, 1),
        "avg_confidence": round(avg_conf, 4),
        "threat_breakdown": threat_breakdown,
    }


def delete_scan(scan_id: int) -> bool:
    """Delete a scan record."""
    conn = get_db()
    cursor = conn.execute("DELETE FROM scan_history WHERE id = ?", (scan_id,))
    conn.commit()
    return cursor.rowcount > 0


# --- Training Samples ---

def save_training_sample(subject: str, body: str, label: int, features: list) -> int:
    """Save a labeled training sample."""
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO training_samples (timestamp, subject, body_preview, label, features)
           VALUES (?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat() + "Z",
            subject[:200],
            body[:500],
            label,
            json.dumps(features),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_training_samples() -> List[Dict[str, Any]]:
    """Get all training samples for model training."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM training_samples ORDER BY id").fetchall()
    results = []
    for r in rows:
        d = _row_to_dict(r)
        d["features"] = json.loads(d["features"])
        results.append(d)
    return results


def get_training_sample_counts() -> Dict[str, int]:
    """Get training sample counts by label."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0]
    phishing = conn.execute("SELECT COUNT(*) FROM training_samples WHERE label=1").fetchone()[0]
    legitimate = conn.execute("SELECT COUNT(*) FROM training_samples WHERE label=0").fetchone()[0]
    return {"total": total, "phishing": phishing, "legitimate": legitimate}


# --- Label Submission Audit (V-06) ---

def save_label_submission(
    submitter_ip: str,
    subject: str,
    body_hash: str,
    submitted_label: int,
    pipeline_confidence: float,
    pipeline_verdict: str,
    suspicious_flagged: bool,
    confirmed: bool = False,
) -> int:
    """
    Persist every label submission for audit and integrity enforcement.

    Returns the new row ID.
    """
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO label_submissions
           (timestamp, submitter_ip, subject_preview, body_hash,
            submitted_label, pipeline_confidence, pipeline_verdict,
            suspicious_flagged, confirmed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat() + "Z",
            submitter_ip,
            subject[:200],
            body_hash,
            submitted_label,
            round(pipeline_confidence, 4),
            pipeline_verdict,
            1 if suspicious_flagged else 0,
            1 if confirmed else 0,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    flag_str = "SUSPICIOUS_LABEL" if suspicious_flagged else "ok"
    logger.info(
        f"Label submission logged: id={row_id} ip={submitter_ip} "
        f"label={submitted_label} conf={pipeline_confidence:.3f} flag={flag_str}"
    )
    return row_id


def get_suspicious_label_count(body_hash: str) -> int:
    """
    Return the number of prior flagged-suspicious submissions for this body hash.

    Used to decide whether to quarantine (count==0) or confirm (count>=1).
    """
    conn = get_db()
    row = conn.execute(
        "SELECT COUNT(*) FROM label_submissions "
        "WHERE body_hash = ? AND suspicious_flagged = 1",
        (body_hash,),
    ).fetchone()
    return int(row[0]) if row else 0


def confirm_suspicious_label(body_hash: str) -> int:
    """
    Mark all prior flagged submissions for this body_hash as confirmed.

    Returns the number of rows updated.
    """
    conn = get_db()
    cursor = conn.execute(
        "UPDATE label_submissions SET confirmed = 1 "
        "WHERE body_hash = ? AND suspicious_flagged = 1 AND confirmed = 0",
        (body_hash,),
    )
    conn.commit()
    return cursor.rowcount


# --- Helpers ---

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a dictionary, parsing JSON fields."""
    d = dict(row)
    for key in ("threat_indicators", "raw_features"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d

"""
Fusion Classifier for Phishing Detection.

Combines semantic (Engine A) and structural (Engine B) features into a
final classification using a Random Forest and a MultiOutputClassifier.

Feature fusion (paper-aligned):
  WITH DistilBERT + PCA : PCA(768-d → 64-d) + 20-d rule indicators = 84-d
  WITHOUT transformer   : 20-d rule scores  + 20-d structural top   = 40-d  ← degraded

Classifiers:
  Main   : RandomForestClassifier(n_estimators=100, max_depth=15)
           → phishing probability + verdict
  Multi  : MultiOutputClassifier(RF sub-models × 20)
           → 20 binary phishing indicator flags

Persistence (models/):
  rf_classifier.pkl   — main RF
  pca_reducer.pkl     — fitted PCA(768→64)
  rf_multioutput.pkl  — MultiOutputClassifier

CLI training:
  python -m app.models.classifier --train --data data/processed/train.csv

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

from __future__ import annotations

import os
import pickle
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH      = os.path.join(_MODELS_DIR, "rf_classifier.pkl")
PCA_PATH        = os.path.join(_MODELS_DIR, "pca_reducer.pkl")
MULTIOUT_PATH   = os.path.join(_MODELS_DIR, "rf_multioutput.pkl")

# ---------------------------------------------------------------------------
# 20 phishing indicator categories (used by MultiOutputClassifier)
# ---------------------------------------------------------------------------
INDICATOR_NAMES: List[str] = [
    # Semantic (8)
    "urgency_detected",
    "authority_detected",
    "pressure_detected",
    "generic_greeting_detected",
    "reward_lure_detected",
    "credential_request_detected",
    "grammatical_anomaly_detected",
    "brand_impersonation",
    # URL structural (7)
    "url_ip_based",
    "url_shortener",
    "url_suspicious_tld",
    "url_no_https",
    "url_typosquatting",
    "url_suspicious_path",
    "url_free_hosting",
    # Header (3)
    "header_spf_fail",
    "header_dkim_fail",
    "header_display_mismatch",
    # HTML (2)
    "html_external_form",
    "html_obfuscated_js",
]
assert len(INDICATOR_NAMES) == 20, "Must have exactly 20 indicator names"

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_classifier    = None
_model_loaded  = False
_pca           = None
_pca_loaded    = False
_multiout      = None
_multiout_loaded = False

# In-memory training buffer (used by add_training_sample / train_model)
_training_samples: List[dict] = []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model() -> bool:
    """Load main RF classifier (once)."""
    global _classifier, _model_loaded
    if _model_loaded:
        return _classifier is not None
    _model_loaded = True
    path = os.path.abspath(MODEL_PATH)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                _classifier = pickle.load(f)
            logger.info(f"RF classifier loaded from {path}  "
                        f"(n_features={getattr(_classifier,'n_features_in_','?')})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load RF model: {e}")
    logger.info("No RF model found — using heuristic fallback")
    return False


def _load_pca() -> Optional[object]:
    """Load PCA reducer (once).  Returns None when not available."""
    global _pca, _pca_loaded
    if _pca_loaded:
        return _pca
    _pca_loaded = True
    path = os.path.abspath(PCA_PATH)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                _pca = pickle.load(f)
            logger.info(f"PCA reducer loaded ({_pca.n_components_}-d) from {path}")
        except Exception as e:
            logger.warning(f"Failed to load PCA reducer: {e}")
    else:
        logger.debug("pca_reducer.pkl not found — running in degraded mode (no PCA)")
    return _pca


def _load_multiout() -> Optional[object]:
    """Load MultiOutputClassifier (once).  Returns None when not available."""
    global _multiout, _multiout_loaded
    if _multiout_loaded:
        return _multiout
    _multiout_loaded = True
    path = os.path.abspath(MULTIOUT_PATH)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                _multiout = pickle.load(f)
            logger.info(f"MultiOutputClassifier loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load MultiOutputClassifier: {e}")
    return _multiout


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------

def _rule_scores_to_features(rule_scores: Dict[str, Any]) -> list:
    """
    Convert rule-based semantic scores to a 20-d numeric feature vector.

    These are the "20 engineered indicators" referenced in the paper.
    """
    features: list = []
    for key in ["urgency", "authority", "pressure", "generic_greetings",
                "reward_lures", "credential_requests"]:
        entry = rule_scores.get(key, {})
        if isinstance(entry, dict):
            features.append(float(entry.get("score", 0.0)))
            features.append(float(entry.get("count", 0)))
        else:
            features.extend([0.0, 0.0])
    grammar = rule_scores.get("grammatical_anomalies", {})
    features.append(float(grammar.get("score", 0.0)) if isinstance(grammar, dict) else 0.0)
    combined = rule_scores.get("combined_score", 0.0)
    features.append(float(combined) if isinstance(combined, (int, float)) else 0.0)
    brand_mentions = rule_scores.get("brand_mentions", [])
    features.append(float(len(brand_mentions)) if isinstance(brand_mentions, list) else 0.0)
    # Pad / truncate to exactly 20
    while len(features) < 20:
        features.append(0.0)
    return features[:20]


def _build_feature_vector(
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
) -> Tuple[List[float], str, bool]:
    """
    Assemble the feature vector for the RF classifier.

    Returns:
        (feature_list, mode_label, degraded_flag)

    Modes:
        "pca_bert"       : 64-d PCA(DistilBERT) + 20-d rule = 84-d  ← paper
        "rule_structural": 20-d rule + 20-d structural              = 40-d  ← degraded
        "rule_only"      : 20-d rule (legacy ≤100-d model)          = 70-d  ← legacy
        "bert_structural": 768-d embedding + structural             = 818-d ← old legacy
    """
    embedding   = semantic_results.get("embedding")          # 768-d list or None
    rule_feats  = _rule_scores_to_features(                   # always 20-d
        semantic_results.get("rule_scores", {}))
    struct_feats = structural_results.get("structural_features", [])  # 50-d

    pca = _load_pca()

    # ── Paper path ───────────────────────────────────────────────────────────
    if embedding is not None and pca is not None:
        try:
            pca_vec = pca.transform([embedding])[0].tolist()  # 64-d
            return pca_vec + rule_feats, "pca_bert", False
        except Exception as e:
            logger.warning(f"PCA transform failed ({e}), falling back")

    # ── Degraded: rule + top-20 structural ───────────────────────────────────
    struct_top20 = struct_feats[:20]
    struct_top20 += [0.0] * (20 - len(struct_top20))        # pad to 20
    return rule_feats + struct_top20, "rule_structural", (embedding is None)


def _extract_indicator_labels(
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
) -> List[int]:
    """
    Extract a 20-d binary indicator vector from analysis results.
    Used as multi-output training targets.
    """
    rule   = semantic_results.get("rule_scores", {})
    url_r  = structural_results.get("url_analysis",  {})
    hdr_r  = structural_results.get("header_analysis", {})
    html_r = structural_results.get("html_analysis",  {})
    urls   = url_r.get("urls", [])

    def _flag(d: dict, key: str, threshold: float = 0.0) -> int:
        v = d.get(key, {})
        if isinstance(v, dict):
            return int(v.get("count", 0) > threshold or v.get("score", 0.0) > 0.0)
        return 0

    return [
        # Semantic
        _flag(rule, "urgency"),
        _flag(rule, "authority"),
        _flag(rule, "pressure"),
        _flag(rule, "generic_greetings"),
        _flag(rule, "reward_lures"),
        _flag(rule, "credential_requests"),
        int(rule.get("grammatical_anomalies", {}).get("score", 0.0) > 0.0
            if isinstance(rule.get("grammatical_anomalies"), dict) else False),
        int(len(rule.get("brand_mentions", [])) > 0),
        # URL
        int(any(u.get("is_ip_based") for u in urls)),
        int(any(u.get("uses_shortener") for u in urls)),
        int(any(u.get("suspicious_tld") for u in urls)),
        int(any(not u.get("uses_https", True) for u in urls) if urls else False),
        int(any(u.get("typosquatting_score", 0) > 0.5 for u in urls)),
        int(any(u.get("suspicious_path") for u in urls)),
        int(any(u.get("uses_free_hosting") for u in urls)),
        # Header
        int(hdr_r.get("spf_result") == "fail"),
        int(hdr_r.get("dkim_result") == "fail"),
        int(hdr_r.get("display_name_mismatch", False)),
        # HTML
        int(html_r.get("external_form_count", 0) > 0),
        int(html_r.get("has_obfuscated_js", False)),
    ]


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_classify(
    semantic_score: float,
    url_score: float,
    header_score: float,
    html_score: float,
) -> Tuple[str, float]:
    """
    Weighted heuristic fallback when no trained RF model is available.

    Weights: semantic 35%, URL 30%, header 20%, HTML 15%.
    """
    confidence = (
        semantic_score * 0.35 +
        url_score      * 0.30 +
        header_score   * 0.20 +
        html_score     * 0.15
    )
    max_score = max(semantic_score, url_score, header_score, html_score)
    if max_score > 0.4:
        confidence = max(confidence, max_score * 0.70)
    flagged = sum(1 for s in [semantic_score, url_score, header_score, html_score] if s > 0.2)
    if flagged >= 2:
        confidence *= 1.15
    if flagged >= 3:
        confidence *= 1.10
    if confidence > 0.2:
        confidence = 0.2 + (confidence - 0.2) * 1.3
    confidence = min(max(confidence, 0.0), 1.0)
    if confidence >= 0.40:
        verdict = "phishing"
    elif confidence >= 0.22:
        verdict = "suspicious"
    else:
        verdict = "legitimate"
    return verdict, round(confidence, 4)


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------

def classify(
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify an email as phishing, suspicious, or legitimate.

    Returns a dict with keys:
      verdict           : "phishing" | "suspicious" | "legitimate"
      confidence        : float in [0, 1]
      method            : "random_forest" | "heuristic"
      feature_mode      : "pca_bert" | "rule_structural" | "heuristic"
      degraded_mode     : bool — True when DistilBERT/PCA unavailable
      indicator_flags   : dict mapping 20 indicator names → 0/1  (when MultiOutput available)
      feature_importance: dict (top-10 RF feature importances)
    """
    semantic_score = semantic_results.get("overall_score", 0.0)
    url_score      = structural_results.get("url_analysis",  {}).get("overall_score", 0.0)
    header_score   = structural_results.get("header_analysis", {}).get("overall_score", 0.0)
    html_score     = structural_results.get("html_analysis",  {}).get("overall_score", 0.0)

    # ── Try RF classification ────────────────────────────────────────────────
    if _load_model() and _classifier is not None:
        try:
            feature_vec, feature_mode, degraded = _build_feature_vector(
                semantic_results, structural_results
            )

            # Dimension guard — handle legacy models trained with different dims
            expected_dims = getattr(_classifier, "n_features_in_", None)
            if expected_dims is not None and len(feature_vec) != expected_dims:
                # Legacy 70-d model path
                if expected_dims <= 100:
                    rule_feats   = _rule_scores_to_features(semantic_results.get("rule_scores", {}))
                    struct_feats = structural_results.get("structural_features", [])
                    feature_vec  = rule_feats + struct_feats
                    feature_mode = "rule_only"
                    degraded     = True
                elif expected_dims > 100 and semantic_results.get("embedding"):
                    # Very old 818-d model (BERT + structural, no PCA)
                    struct_feats = structural_results.get("structural_features", [])
                    feature_vec  = semantic_results["embedding"] + struct_feats
                    feature_mode = "bert_structural"
                    degraded     = False

                # Pad or truncate to match
                if len(feature_vec) < expected_dims:
                    feature_vec += [0.0] * (expected_dims - len(feature_vec))
                elif len(feature_vec) > expected_dims:
                    feature_vec = feature_vec[:expected_dims]

            X = np.array(feature_vec).reshape(1, -1)
            probas = _classifier.predict_proba(X)[0]
            phishing_prob = float(probas[1] if len(probas) > 1 else probas[0])

            verdict = (
                "phishing"   if phishing_prob >= 0.40 else
                "suspicious" if phishing_prob >= 0.22 else
                "legitimate"
            )

            # ── MultiOutput indicator flags ──────────────────────────────────
            indicator_flags = _run_multioutput(feature_vec, semantic_results, structural_results)

            result: Dict[str, Any] = {
                "verdict":            verdict,
                "confidence":         round(phishing_prob, 4),
                "method":             "random_forest",
                "feature_mode":       feature_mode,
                "degraded_mode":      degraded,
                "feature_importance": _get_feature_importance(),
            }
            if indicator_flags is not None:
                result["indicator_flags"] = indicator_flags
            if degraded:
                result["degraded_warning"] = (
                    "DistilBERT transformer or PCA reducer unavailable. "
                    "Running on rule-based features only (reduced accuracy). "
                    "Install transformers + torch and retrain with --train to restore full accuracy."
                )
            return result

        except Exception as e:
            logger.warning(f"RF classification failed, falling back to heuristic: {e}")

    # ── Heuristic fallback ───────────────────────────────────────────────────
    verdict, confidence = _heuristic_classify(
        semantic_score, url_score, header_score, html_score
    )
    degraded = semantic_results.get("embedding") is None
    result = {
        "verdict":       verdict,
        "confidence":    confidence,
        "method":        "heuristic",
        "feature_mode":  "heuristic",
        "degraded_mode": degraded,
        "component_scores": {
            "semantic": round(semantic_score, 4),
            "url":      round(url_score,      4),
            "header":   round(header_score,   4),
            "html":     round(html_score,     4),
        },
    }
    if degraded:
        result["degraded_warning"] = (
            "DistilBERT transformer unavailable — heuristic scoring active. "
            "Install transformers + torch and retrain to improve accuracy."
        )
    return result


def _run_multioutput(
    feature_vec: List[float],
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
) -> Optional[Dict[str, int]]:
    """
    Run the MultiOutputClassifier to produce 20 binary indicator flags.
    Returns None if the multi-output model is not loaded.
    """
    mo = _load_multiout()
    if mo is None:
        return None
    try:
        expected = getattr(mo, "n_features_in_",
                           getattr(getattr(mo, "estimators_", [None])[0], "n_features_in_", None))
        vec = list(feature_vec)
        if expected is not None:
            if len(vec) < expected:
                vec += [0.0] * (expected - len(vec))
            elif len(vec) > expected:
                vec = vec[:expected]
        X = np.array(vec).reshape(1, -1)
        preds = mo.predict(X)[0]          # 20-d binary vector
        return {name: int(p) for name, p in zip(INDICATOR_NAMES, preds)}
    except Exception as e:
        logger.debug(f"MultiOutput prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _get_feature_importance() -> Optional[Dict[str, float]]:
    """Return top-10 feature importances from the main RF model."""
    if _classifier is None:
        return None
    try:
        imp = _classifier.feature_importances_
        return {f"feature_{i}": round(float(v), 4) for i, v in enumerate(imp[:10])}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------------

def get_recommended_action(verdict: str, confidence: float) -> Dict[str, str]:
    """Map verdict + confidence to a recommended security action."""
    if verdict == "phishing" and confidence > 0.75:
        return {
            "action":      "quarantine",
            "description": "Quarantine this email immediately. High confidence phishing detected.",
            "icon":        "shield-exclamation",
        }
    elif verdict in ("phishing", "suspicious"):
        return {
            "action":      "alert",
            "description": "Flag for manual review. Suspicious indicators detected.",
            "icon":        "exclamation-triangle",
        }
    return {
        "action":      "pass",
        "description": "No significant phishing indicators detected. Email appears legitimate.",
        "icon":        "check-circle",
    }


# ---------------------------------------------------------------------------
# Online training buffer (used by /api/train/label endpoint)
# ---------------------------------------------------------------------------

def add_training_sample(
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any],
    label: int,
) -> None:
    """Buffer a labeled sample for later training via train_model()."""
    feature_vec, _, _ = _build_feature_vector(semantic_results, structural_results)
    indicator_vec     = _extract_indicator_labels(semantic_results, structural_results)
    _training_samples.append({
        "features":   feature_vec,
        "indicators": indicator_vec,
        "label":      label,
    })
    logger.info(f"Buffered training sample (label={label}). Total: {len(_training_samples)}")


def train_model(min_samples: int = 10) -> Dict[str, Any]:
    """
    Train the main RF and MultiOutputClassifier on buffered samples.

    Uses paper hyperparameters: n_estimators=100, max_depth=15.
    """
    global _classifier, _model_loaded, _multiout, _multiout_loaded

    if len(_training_samples) < min_samples:
        return {
            "success":      False,
            "error":        f"Need ≥{min_samples} samples. Have {len(_training_samples)}.",
            "sample_count": len(_training_samples),
        }

    labels = [s["label"] for s in _training_samples]
    if len(set(labels)) < 2:
        return {
            "success":            False,
            "error":              "Need samples from both classes.",
            "sample_count":       len(_training_samples),
            "label_distribution": {"phishing": labels.count(1), "legitimate": labels.count(0)},
        }

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        max_len = max(len(s["features"]) for s in _training_samples)
        X = np.array([
            s["features"] + [0.0] * (max_len - len(s["features"]))
            for s in _training_samples
        ])
        y = np.array(labels)

        # ── Main RF (paper hyperparams) ──────────────────────────────────────
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        cv_scores = None
        if len(_training_samples) >= 20:
            cv = StratifiedKFold(n_splits=min(5, len(_training_samples) // 4),
                                 shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1", n_jobs=-1)

        clf.fit(X, y)
        abs_path = os.path.abspath(MODEL_PATH)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            pickle.dump(clf, f)
        _classifier  = clf
        _model_loaded = True

        # ── MultiOutputClassifier ────────────────────────────────────────────
        multi_result: Dict[str, Any] = {}
        try:
            Y_multi = np.array([s["indicators"] for s in _training_samples])
            base_rf = RandomForestClassifier(
                n_estimators=100, max_depth=15,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            mo = MultiOutputClassifier(base_rf, n_jobs=-1)
            mo.fit(X, Y_multi)
            mo_path = os.path.abspath(MULTIOUT_PATH)
            with open(mo_path, "wb") as f:
                pickle.dump(mo, f)
            _multiout         = mo
            _multiout_loaded  = True
            multi_result["multioutput_trained"] = True
        except Exception as e:
            logger.warning(f"MultiOutputClassifier training failed: {e}")
            multi_result["multioutput_trained"] = False

        result: Dict[str, Any] = {
            "success":            True,
            "sample_count":       len(_training_samples),
            "feature_count":      max_len,
            "label_distribution": {"phishing": int(y.sum()), "legitimate": int((1-y).sum())},
            "model_path":         abs_path,
            **multi_result,
        }
        if cv_scores is not None:
            result["cross_validation"] = {
                "f1_mean": round(float(cv_scores.mean()), 4),
                "f1_std":  round(float(cv_scores.std()),  4),
            }
        logger.info(f"train_model() complete: {result}")
        return result

    except Exception as e:
        logger.error(f"train_model() failed: {e}")
        return {"success": False, "error": str(e)}


def get_training_stats() -> Dict[str, Any]:
    """Return statistics about the in-memory training buffer."""
    labels = [s["label"] for s in _training_samples]
    pca    = _load_pca()
    return {
        "total_samples":     len(_training_samples),
        "phishing_samples":  labels.count(1),
        "legitimate_samples": labels.count(0),
        "model_loaded":      _classifier is not None,
        "pca_loaded":        pca is not None,
        "multiout_loaded":   _load_multiout() is not None,
        "model_type":        "random_forest" if _classifier is not None else "heuristic",
        "feature_dims":      getattr(_classifier, "n_features_in_", None),
    }


# ---------------------------------------------------------------------------
# CLI training entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import csv
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train the phishing RF + PCA + MultiOutputClassifier from a CSV corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",     action="store_true", help="Run training.")
    parser.add_argument("--data",      default="data/processed/train.csv",
                        help="Path to train.csv with columns [subject, body, label].")
    parser.add_argument("--max-rows",  type=int, default=50_000,
                        help="Max rows to process (set lower for faster iteration).")
    parser.add_argument("--pca-dims",  type=int, default=64,
                        help="PCA output dimensions (paper: 64).")
    parser.add_argument("--epochs",    type=int, default=0,
                        help="Ignored; kept for script compatibility.")
    parser.add_argument("--no-bert",   action="store_true",
                        help="Skip DistilBERT — use rule-based features only (fast but degraded).")
    cfg = parser.parse_args()

    if not cfg.train:
        parser.print_help()
        sys.exit(0)

    # ── Setup ─────────────────────────────────────────────────────────────────
    csv.field_size_limit(10 * 1024 * 1024)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    logger.info("=" * 62)
    logger.info("  Fusion Classifier — Training")
    logger.info("=" * 62)
    logger.info(f"  data      : {cfg.data}")
    logger.info(f"  max-rows  : {cfg.max_rows:,}")
    logger.info(f"  pca-dims  : {cfg.pca_dims}")
    logger.info(f"  no-bert   : {cfg.no_bert}")
    logger.info("")

    # Optionally skip DistilBERT for speed
    if cfg.no_bert:
        import app.models.semantic_engine as _sem
        _sem._model_load_attempted = True
        _sem._transformer_model    = None
        _sem._transformer_tokenizer = None
        logger.info("  DistilBERT disabled (--no-bert)")

    from app.parsers.text_parser      import parse_text_input
    from app.models.semantic_engine   import analyze_semantics, compute_rule_based_scores
    from app.models.structural_engine import analyze_structure
    from sklearn.decomposition        import PCA
    from sklearn.ensemble             import RandomForestClassifier
    from sklearn.multioutput          import MultiOutputClassifier
    from sklearn.model_selection      import StratifiedKFold, cross_val_score
    from sklearn.metrics              import classification_report

    # ── 1. Load rows ──────────────────────────────────────────────────────────
    rows: list = []
    with open(cfg.data, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= cfg.max_rows:
                break
            try:
                lbl = int(float((row.get("label") or "").strip()))
                if lbl not in (0, 1):
                    continue
                subj = (row.get("subject") or "").strip()[:400]
                body = (row.get("body")    or "").strip()[:8000]
                rows.append((subj, body, lbl))
            except (ValueError, TypeError):
                continue

    logger.info(f"  Loaded {len(rows):,} rows from {cfg.data}")
    n_ph = sum(1 for r in rows if r[2] == 1)
    logger.info(f"  phishing={n_ph:,}  legit={len(rows)-n_ph:,}")

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    logger.info("\n  Extracting features...")
    t0 = time.time()
    X_raw:    list = []
    y_list:   list = []
    Y_multi:  list = []
    embeddings: list = []   # 768-d; collected for PCA fitting
    failed = 0

    for i, (subj, body, lbl) in enumerate(rows):
        try:
            parsed   = parse_text_input(subj, body)
            sem_res  = analyze_semantics(subj, body)
            str_res  = analyze_structure(parsed)

            rule_f   = _rule_scores_to_features(sem_res.get("rule_scores", {}))
            struct_f = str_res.get("structural_features", [])
            ind_v    = _extract_indicator_labels(sem_res, str_res)
            emb      = sem_res.get("embedding")           # 768-d or None

            X_raw.append({"rule": rule_f, "struct": struct_f, "embedding": emb})
            y_list.append(lbl)
            Y_multi.append(ind_v)
            if emb is not None:
                embeddings.append(emb)

        except Exception:
            failed += 1
            continue

        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta  = (len(rows) - i - 1) / max(rate, 1)
            logger.info(f"  {i+1:,}/{len(rows):,} | {rate:.0f}/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"  Done: {len(X_raw):,} extracted, {failed} failed  ({elapsed:.1f}s)")

    # ── 3. Fit PCA on collected embeddings ────────────────────────────────────
    pca_fitted = None
    use_pca    = len(embeddings) >= cfg.pca_dims * 2 and not cfg.no_bert

    if use_pca:
        logger.info(f"\n  Fitting PCA({cfg.pca_dims}d) on {len(embeddings):,} embeddings...")
        pca_fitted = PCA(n_components=cfg.pca_dims, random_state=42)
        pca_fitted.fit(np.array(embeddings))
        explained = pca_fitted.explained_variance_ratio_.sum()
        logger.info(f"  Explained variance: {explained*100:.1f}%")
        # Save
        os.makedirs(os.path.abspath(_MODELS_DIR), exist_ok=True)
        pca_save = os.path.abspath(PCA_PATH)
        with open(pca_save, "wb") as f:
            pickle.dump(pca_fitted, f)
        logger.info(f"  PCA saved → {pca_save}")
    else:
        if cfg.no_bert:
            logger.info("\n  PCA skipped (--no-bert)")
        else:
            logger.warning(f"\n  Not enough embeddings ({len(embeddings)}) for PCA — running in degraded mode")

    # ── 4. Build final feature matrix ─────────────────────────────────────────
    X_feats: list = []
    for item in X_raw:
        emb = item["embedding"]
        if use_pca and pca_fitted is not None and emb is not None:
            pca_vec = pca_fitted.transform([emb])[0].tolist()   # 64-d
            vec     = pca_vec + item["rule"]                     # 84-d
        else:
            struct20 = item["struct"][:20]
            struct20 += [0.0] * (20 - len(struct20))
            vec = item["rule"] + struct20                        # 40-d
        X_feats.append(vec)

    max_len = max(len(v) for v in X_feats)
    X_arr   = np.array([v + [0.0] * (max_len - len(v)) for v in X_feats])
    y_arr   = np.array(y_list)
    Y_arr   = np.array(Y_multi)

    n_ph = int(y_arr.sum())
    logger.info(f"\n  Feature matrix: {X_arr.shape}  phishing={n_ph:,}  legit={len(y_arr)-n_ph:,}")

    # ── 5. Train main RF with 5-fold CV ───────────────────────────────────────
    logger.info("\n  Training RandomForest(n=100, depth=15)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1  = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring="f1",       n_jobs=-1)
    cv_acc = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring="accuracy",  n_jobs=-1)
    cv_pr  = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring="precision", n_jobs=-1)
    cv_rec = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring="recall",    n_jobs=-1)

    logger.info(f"  CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    logger.info(f"  CV Precision: {cv_pr.mean():.4f}  ± {cv_pr.std():.4f}")
    logger.info(f"  CV Recall   : {cv_rec.mean():.4f}  ± {cv_rec.std():.4f}")
    logger.info(f"  CV F1       : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")

    clf.fit(X_arr, y_arr)
    logger.info("\n  Training-set report:\n" +
                classification_report(y_arr, clf.predict(X_arr),
                                      target_names=["Legitimate", "Phishing"]))

    # Save main RF
    rf_path = os.path.abspath(MODEL_PATH)
    os.makedirs(os.path.dirname(rf_path), exist_ok=True)
    with open(rf_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"  RF saved → {rf_path}")

    # ── 6. Train MultiOutputClassifier ───────────────────────────────────────
    logger.info("\n  Training MultiOutputClassifier (20 sub-models)...")
    try:
        base_rf = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        mo = MultiOutputClassifier(base_rf, n_jobs=-1)
        mo.fit(X_arr, Y_arr)
        mo_path = os.path.abspath(MULTIOUT_PATH)
        with open(mo_path, "wb") as f:
            pickle.dump(mo, f)
        logger.info(f"  MultiOutputClassifier saved → {mo_path}")

        # Per-indicator accuracy on training set
        Y_pred = mo.predict(X_arr)
        logger.info("  Per-indicator training accuracy:")
        for j, name in enumerate(INDICATOR_NAMES):
            acc = float((Y_pred[:, j] == Y_arr[:, j]).mean())
            logger.info(f"    {name:<35s}: {acc:.3f}")
    except Exception as e:
        logger.error(f"  MultiOutputClassifier failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 62)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 62)
    logger.info(f"  Samples     : {len(X_arr):,}")
    logger.info(f"  Feature dims: {max_len}")
    logger.info(f"  Feature mode: {'pca_bert (84-d)' if use_pca else 'rule_structural (40-d)'}")
    logger.info(f"  CV F1       : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    logger.info(f"  Artifacts:")
    logger.info(f"    {rf_path}")
    if use_pca:
        logger.info(f"    {os.path.abspath(PCA_PATH)}")
    logger.info(f"    {os.path.abspath(MULTIOUT_PATH)}")
    logger.info("  Restart the server to load the updated models.")
    logger.info("=" * 62)

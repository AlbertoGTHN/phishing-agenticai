"""
Evaluation Pipeline for the Agentic AI Phishing Detector.

Runs the full analysis pipeline on a held-out test set, computes a confusion
matrix, per-class precision/recall/F1, and optionally saves a detailed
per-sample result log.

Paper reference: Loo, Galindo, Romero et al. (2025) claim an 80-email
held-out test set.  This module supports arbitrary test set sizes; use
--n-samples 80 to reproduce the exact paper protocol.

Supported input formats
  • CSV  with columns [subject, body, label]           (standard pipeline output)
  • CSV  with columns [Email Text, Email Type]         (Phishing_validation_emails.csv)
  • CSV  with columns [subject, body, Label]           (SpaPhish ';'-delimited)
  • Directory of .eml files  (sub-dirs or flat; labelled by folder name)

Usage:
  python -m app.evaluation.evaluate \\
      --test-csv  data/processed/test.csv \\
      --n-samples 80 \\
      --out-dir   data/evaluation/

  python -m app.evaluation.evaluate \\
      --test-csv  C:/Users/alber/Downloads/Phishing_validation_emails.csv \\
      --out-dir   data/evaluation/

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

from __future__ import annotations

import csv
import email as _email_lib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_label(raw: str) -> Optional[int]:
    """
    Convert a raw label string to 0 (legitimate) or 1 (phishing).
    Returns None if the value cannot be parsed.
    """
    raw = (raw or "").strip().lower()
    if raw in ("1", "phishing", "phishing email", "phish", "spam"):
        return 1
    if raw in ("0", "ham", "legitimate", "legit", "safe", "safe email", "not phishing"):
        return 0
    try:
        v = int(float(raw))
        return v if v in (0, 1) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_test_csv(
    path: str | Path,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load a test CSV into a list of sample dicts.

    Accepts three column layouts automatically:
      1. [subject, body, label]          — standard pipeline format
      2. [Email Text, Email Type]        — Phishing_validation_emails.csv
      3. [subject, body, Label] (;-sep)  — SpaPhish format

    Args:
        path        : Path to CSV file.
        n_samples   : If set, randomly sample this many rows (stratified).
        random_state: RNG seed for stratified sampling.

    Returns:
        List of dicts with keys: subject, body, label (int), source_label (str).
    """
    path = Path(path)
    sep  = ";" if path.suffix.lower() == ".csv" and _sniff_separator(path) == ";" else ","

    rows: list[dict] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        fieldnames = [f.strip() for f in (reader.fieldnames or [])]

        for row in reader:
            # Normalise keys
            row = {k.strip(): v for k, v in row.items()}

            # Detect layout
            if "Email Text" in row and "Email Type" in row:
                # Layout 2: Phishing_validation_emails.csv
                body        = (row.get("Email Text") or "").strip()[:8000]
                subject     = ""
                raw_label   = row.get("Email Type", "")
            elif "Label" in row and "subject" in row:
                # Layout 3: SpaPhish
                subject   = (row.get("subject") or "").strip()[:400]
                body      = (row.get("body")    or "").strip()[:8000]
                raw_label = row.get("Label", "")
            elif "label" in row:
                # Layout 1: standard
                subject   = (row.get("subject") or "").strip()[:400]
                body      = (row.get("body")    or "").strip()[:8000]
                raw_label = row.get("label", "")
            else:
                continue

            label = _normalise_label(raw_label)
            if label is None:
                continue

            rows.append({
                "subject":      subject,
                "body":         body,
                "label":        label,
                "source_label": raw_label,
            })

    # Stratified sampling
    if n_samples is not None and len(rows) > n_samples:
        rng    = random.Random(random_state)
        pos    = [r for r in rows if r["label"] == 1]
        neg    = [r for r in rows if r["label"] == 0]
        half   = n_samples // 2
        sample = rng.sample(pos, min(len(pos), half)) + \
                 rng.sample(neg, min(len(neg), half))
        # Top up if one class is smaller
        deficit = n_samples - len(sample)
        if deficit > 0:
            used   = set(id(r) for r in sample)
            remain = [r for r in rows if id(r) not in used]
            sample += rng.sample(remain, min(len(remain), deficit))
        rng.shuffle(sample)
        rows = sample

    return rows


def load_test_eml_dir(
    path: str | Path,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load .eml files from a directory tree.

    Label assignment rules:
      • Sub-folders containing "phish", "spam", "fraud", "nigerian", "nazario"
        → label = 1
      • Sub-folders containing "ham", "legit", "safe", "inbox"
        → label = 0
      • Flat directory (no sub-folders) → all label = 1  (phishing corpus assumed)

    Args:
        path        : Root directory.
        n_samples   : Optional sample cap.
        random_state: RNG seed.

    Returns:
        List of sample dicts.
    """
    path   = Path(path)
    rows: list[dict] = []

    eml_files: list[tuple[Path, int]] = []
    has_subdirs = any(p.is_dir() for p in path.iterdir())

    if has_subdirs:
        for sub in path.iterdir():
            if not sub.is_dir():
                continue
            name_lower = sub.name.lower()
            if any(k in name_lower for k in ("phish", "spam", "fraud", "nigerian", "nazario")):
                lbl = 1
            elif any(k in name_lower for k in ("ham", "legit", "safe", "inbox", "normal")):
                lbl = 0
            else:
                logger.warning(f"  Cannot infer label from dir name '{sub.name}' — skipping")
                continue
            for fp in sub.rglob("*.eml"):
                eml_files.append((fp, lbl))
    else:
        for fp in path.rglob("*.eml"):
            eml_files.append((fp, 1))  # assume phishing corpus

    for fp, lbl in eml_files:
        try:
            raw = fp.read_bytes()
            msg = _email_lib.message_from_bytes(raw)
            subject = (msg.get("Subject") or "").strip()[:400]
            body    = _extract_body(msg)[:8000]
            rows.append({
                "subject":      subject,
                "body":         body,
                "label":        lbl,
                "source_label": str(lbl),
                "source_file":  str(fp),
            })
        except Exception as e:
            logger.debug(f"Failed to parse {fp}: {e}")

    if n_samples is not None and len(rows) > n_samples:
        rng  = random.Random(random_state)
        rows = rng.sample(rows, n_samples)

    return rows


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_on_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full analysis pipeline on a single test sample.

    Returns the sample dict augmented with:
      predicted_label  : int (1=phishing, 0=legitimate)
      predicted_verdict: str
      confidence       : float
      method           : str
      feature_mode     : str
      degraded_mode    : bool
      elapsed_s        : float
      error            : str | None
    """
    from app.parsers.text_parser       import parse_text_input
    from app.models.semantic_engine    import analyze_semantics
    from app.models.structural_engine  import analyze_structure
    from app.models.classifier         import classify

    t0 = time.time()
    try:
        parsed   = parse_text_input(sample["subject"], sample["body"])
        sem_res  = analyze_semantics(sample["subject"], sample["body"])
        str_res  = analyze_structure(parsed)
        clf_res  = classify(sem_res, str_res)

        verdict  = clf_res["verdict"]
        pred_lbl = 1 if verdict == "phishing" else 0   # suspicious → 0 for binary eval

        result = {
            **sample,
            "predicted_label":   pred_lbl,
            "predicted_verdict": verdict,
            "confidence":        clf_res.get("confidence", 0.0),
            "method":            clf_res.get("method", "unknown"),
            "feature_mode":      clf_res.get("feature_mode", "unknown"),
            "degraded_mode":     clf_res.get("degraded_mode", False),
            "elapsed_s":         round(time.time() - t0, 3),
            "error":             None,
        }
    except Exception as exc:
        result = {
            **sample,
            "predicted_label":   0,
            "predicted_verdict": "error",
            "confidence":        0.0,
            "method":            "error",
            "feature_mode":      "error",
            "degraded_mode":     True,
            "elapsed_s":         round(time.time() - t0, 3),
            "error":             str(exc),
        }

    return result


def evaluate_on_csv(
    csv_path: str | Path,
    n_samples: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the full pipeline on a CSV test file.

    Args:
        csv_path    : Path to test CSV.
        n_samples   : Optional cap (use 80 to match paper protocol).
        random_state: RNG seed for sampling.
        verbose     : Print per-sample progress.

    Returns:
        Evaluation results dict (see :func:`compute_metrics`).
    """
    samples = load_test_csv(csv_path, n_samples=n_samples, random_state=random_state)
    logger.info(f"  Loaded {len(samples)} test samples from {Path(csv_path).name}")
    return _run_evaluation(samples, verbose=verbose)


def evaluate_on_eml_dir(
    eml_dir: str | Path,
    n_samples: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the full pipeline on a directory of .eml files.

    Args:
        eml_dir     : Root directory containing .eml files.
        n_samples   : Optional cap.
        random_state: RNG seed.
        verbose     : Print per-sample progress.

    Returns:
        Evaluation results dict.
    """
    samples = load_test_eml_dir(eml_dir, n_samples=n_samples, random_state=random_state)
    logger.info(f"  Loaded {len(samples)} .eml test samples from {Path(eml_dir).name}")
    return _run_evaluation(samples, verbose=verbose)


def _run_evaluation(
    samples: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Core evaluation loop."""
    results: list[dict] = []
    n = len(samples)
    t_start = time.time()

    for i, sample in enumerate(samples):
        res = run_pipeline_on_sample(sample)
        results.append(res)
        if verbose and (i + 1) % max(1, n // 10) == 0:
            logger.info(f"  [{i+1:>4}/{n}] label={res['label']} "
                        f"pred={res['predicted_label']} "
                        f"verdict={res['predicted_verdict']:<12} "
                        f"conf={res['confidence']:.3f}  "
                        f"({res['elapsed_s']:.2f}s)")

    elapsed = time.time() - t_start
    return compute_metrics(results, total_time=elapsed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    results: List[Dict[str, Any]],
    total_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute confusion matrix and per-class metrics from result dicts.

    Treats "suspicious" → 0 (not-phishing) for binary classification
    to match the paper's two-class evaluation protocol.

    Returns a comprehensive metrics dict.
    """
    y_true = np.array([r["label"]           for r in results])
    y_pred = np.array([r["predicted_label"]  for r in results])

    # Confusion matrix components (positive class = phishing = 1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total    = len(results)
    accuracy = (tp + tn) / total if total else 0.0
    prec_ph  = tp / (tp + fp)   if (tp + fp) else 0.0
    rec_ph   = tp / (tp + fn)   if (tp + fn) else 0.0
    f1_ph    = (2 * prec_ph * rec_ph) / (prec_ph + rec_ph) if (prec_ph + rec_ph) else 0.0
    prec_lg  = tn / (tn + fn)   if (tn + fn) else 0.0
    rec_lg   = tn / (tn + fp)   if (tn + fp) else 0.0
    f1_lg    = (2 * prec_lg * rec_lg) / (prec_lg + rec_lg) if (prec_lg + rec_lg) else 0.0
    macro_f1 = (f1_ph + f1_lg) / 2

    # Verdict distribution (including "suspicious")
    verdicts = {}
    for r in results:
        v = r.get("predicted_verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1

    # Method distribution
    methods = {}
    for r in results:
        m = r.get("method", "unknown")
        methods[m] = methods.get(m, 0) + 1

    # Degraded mode count
    n_degraded = sum(1 for r in results if r.get("degraded_mode"))

    # Errors
    errors = [r for r in results if r.get("error")]

    return {
        "n_samples":   total,
        "n_phishing":  int(y_true.sum()),
        "n_legit":     int((y_true == 0).sum()),
        "accuracy":    round(accuracy,  4),
        "macro_f1":    round(macro_f1,  4),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "matrix": [[tn, fp], [fn, tp]],   # rows=actual, cols=predicted (legit, phish)
        },
        "phishing": {
            "precision": round(prec_ph, 4),
            "recall":    round(rec_ph,  4),
            "f1":        round(f1_ph,   4),
            "support":   int(y_true.sum()),
        },
        "legitimate": {
            "precision": round(prec_lg, 4),
            "recall":    round(rec_lg,  4),
            "f1":        round(f1_lg,   4),
            "support":   int((y_true == 0).sum()),
        },
        "verdict_distribution":  verdicts,
        "method_distribution":   methods,
        "n_degraded_mode":       n_degraded,
        "n_errors":              len(errors),
        "avg_confidence":        round(float(np.mean([r["confidence"] for r in results])), 4),
        "avg_elapsed_s":         round(total_time / max(total, 1), 3),
        "total_elapsed_s":       round(total_time, 2),
        "results":               results,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(metrics: Dict[str, Any], title: str = "Evaluation Report") -> None:
    """
    Print a formatted evaluation report matching paper Table style.

    Displays confusion matrix, per-class metrics, and summary statistics.
    """
    sep  = "=" * 62
    sep2 = "-" * 62

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    n   = metrics["n_samples"]
    n_p = metrics["n_phishing"]
    n_l = metrics["n_legit"]
    print(f"  Test samples : {n:,}  "
          f"(phishing={n_p:,}  [{n_p/n*100:.1f}%], "
          f"legit={n_l:,}  [{n_l/n*100:.1f}%])")
    print()

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Predicted")
    print(f"              Legit    Phishing")
    print(f"  Actual Legit  {tn:>5}      {fp:>5}")
    print(f"  Actual Phish  {fn:>5}      {tp:>5}")
    print()

    # Per-class metrics
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print(f"  {sep2[2:]}")
    ph = metrics["phishing"]
    lg = metrics["legitimate"]
    print(f"  {'Phishing':<12} {ph['precision']:>10.4f} {ph['recall']:>10.4f} {ph['f1']:>8.4f} {ph['support']:>9,}")
    print(f"  {'Legitimate':<12} {lg['precision']:>10.4f} {lg['recall']:>10.4f} {lg['f1']:>8.4f} {lg['support']:>9,}")
    print(f"  {sep2[2:]}")
    print(f"  {'Accuracy':<12} {'':>10} {'':>10} {metrics['accuracy']:>8.4f} {n:>9,}")
    print(f"  {'Macro F1':<12} {'':>10} {'':>10} {metrics['macro_f1']:>8.4f} {n:>9,}")
    print()

    # Method distribution
    methods = metrics.get("method_distribution", {})
    if methods:
        print(f"  Method distribution: {methods}")
    verdicts = metrics.get("verdict_distribution", {})
    if verdicts:
        print(f"  Verdict distribution: {verdicts}")
    if metrics.get("n_degraded_mode", 0):
        print(f"  [!] Degraded-mode samples: {metrics['n_degraded_mode']}")
    if metrics.get("n_errors", 0):
        print(f"  [X] Errors: {metrics['n_errors']}")

    print(f"\n  Avg confidence : {metrics['avg_confidence']:.4f}")
    print(f"  Avg time/sample: {metrics['avg_elapsed_s']:.3f}s")
    print(f"  Total time     : {metrics['total_elapsed_s']:.1f}s")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    metrics: Dict[str, Any],
    out_dir: str | Path,
    prefix: str = "eval",
) -> None:
    """
    Save evaluation results to *out_dir*:
      {prefix}_metrics.json   — aggregate metrics (no per-sample data)
      {prefix}_samples.csv    — per-sample predictions
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON (exclude results list for cleaner summary)
    summary = {k: v for k, v in metrics.items() if k != "results"}
    json_path = out_dir / f"{prefix}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Metrics saved → {json_path}")

    # Per-sample CSV
    csv_path = out_dir / f"{prefix}_samples.csv"
    fieldnames = [
        "label", "predicted_label", "predicted_verdict", "confidence",
        "method", "feature_mode", "degraded_mode", "elapsed_s", "error",
        "subject", "body",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in metrics.get("results", []):
            row = {k: r.get(k, "") for k in fieldnames}
            # Truncate body for readability
            row["body"] = (row.get("body") or "")[:200]
            writer.writerow(row)
    logger.info(f"  Samples saved  → {csv_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sniff_separator(path: Path) -> str:
    """Sniff CSV delimiter (comma or semicolon)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        return ";" if first_line.count(";") > first_line.count(",") else ","
    except Exception:
        return ","


def _extract_body(msg: _email_lib.message.Message) -> str:
    """Extract plain-text body from a parsed email message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                pl = part.get_payload(decode=True)
                if pl:
                    charset = part.get_content_charset() or "utf-8"
                    return pl.decode(charset, errors="replace")
    else:
        pl = msg.get_payload(decode=True)
        if pl:
            charset = msg.get_content_charset() or "utf-8"
            return pl.decode(charset, errors="replace")
    return ""


# ---------------------------------------------------------------------------
# __main__ — CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, _root)

    parser = argparse.ArgumentParser(
        description="Evaluate the phishing detector on a held-out test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--test-csv",  help="Path to test CSV file.")
    src.add_argument("--test-dir",  help="Path to directory of .eml files.")

    parser.add_argument("--n-samples",    type=int,  default=None,
                        help="Number of samples to evaluate (None=all; 80 matches paper).")
    parser.add_argument("--out-dir",      default="data/evaluation",
                        help="Directory for output files.")
    parser.add_argument("--prefix",       default="eval",
                        help="Prefix for output file names.")
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--no-bert",      action="store_true",
                        help="Skip DistilBERT loading (faster but degraded).")
    cfg = parser.parse_args()

    # Optionally disable DistilBERT
    if cfg.no_bert:
        import app.models.semantic_engine as _sem
        _sem._model_load_attempted   = True
        _sem._transformer_model      = None
        _sem._transformer_tokenizer  = None

    logger.info("=" * 62)
    logger.info("  Phishing Detector — Evaluation")
    logger.info("=" * 62)
    if cfg.n_samples:
        logger.info(f"  Protocol    : {cfg.n_samples}-sample held-out test "
                    f"{'(paper protocol)' if cfg.n_samples == 80 else ''}")
    else:
        logger.info("  Protocol    : full test set")
    logger.info("")

    if cfg.test_csv:
        metrics = evaluate_on_csv(
            cfg.test_csv,
            n_samples=cfg.n_samples,
            random_state=cfg.seed,
        )
    else:
        metrics = evaluate_on_eml_dir(
            cfg.test_dir,
            n_samples=cfg.n_samples,
            random_state=cfg.seed,
        )

    print_report(metrics, title=f"Phishing Detector — {'80-Email' if cfg.n_samples == 80 else 'Held-Out'} Test Evaluation")
    save_results(metrics, cfg.out_dir, prefix=cfg.prefix)
    logger.info("Done.")

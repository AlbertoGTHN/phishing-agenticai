"""
Rigorous Evaluation Script — Agentic AI Phishing Detector
==========================================================

Runs the full MCO pipeline on a held-out test split and produces all
metrics and figures needed for the paper's Results section (Table 3).

Pipeline:  parse_text_input → analyze_semantics → analyze_structure → classify

Outputs (./results/ by default):
  confusion_matrix.png         Seaborn heatmap (300 dpi, publication quality)
  roc_curve.png                ROC with AUC annotation
  indicator_distribution.png   Bar chart of which indicators fire most
  false_negative_analysis.png  Category breakdown of missed phishing emails
  metrics_summary.json         All metrics in machine-readable form

Usage:
  python scripts/evaluate.py                          # uses data/processed/test.csv
  python scripts/evaluate.py --test data/processed/test.csv --max-samples 2000
  python scripts/evaluate.py --test C:/path/to/Phishing_validation_emails.csv
  python scripts/evaluate.py --no-bert               # rule-only mode comparison only

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap / Project root setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate phishing detector and generate paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test",
        default=str(_PROJECT_DIR / "data" / "processed" / "test.csv"),
        help="Path to test CSV (subject,body,label  or  Email Text,Email Type).",
    )
    p.add_argument(
        "--fallback-test",
        default=r"C:\Users\alber\Downloads\Phishing_validation_emails.csv",
        help="Fallback test CSV if --test file is not found.",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap on test samples (None = all). Paper target: 12,375.",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker threads for pipeline execution.",
    )
    p.add_argument(
        "--results-dir", default=str(_PROJECT_DIR / "results"),
        help="Directory for output plots and JSON.",
    )
    p.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Bootstrap resampling iterations for 95%% CI.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling and bootstrap.",
    )
    p.add_argument(
        "--no-bert", action="store_true",
        help="Disable DistilBERT (rule-only mode only; skips BERT comparison).",
    )
    p.add_argument(
        "--skip-plots", action="store_true",
        help="Skip matplotlib figure generation (headless environments).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def _norm_label(raw: str) -> Optional[int]:
    s = (raw or "").strip().lower()
    if s in ("1", "phishing", "phishing email", "spam", "phish"):
        return 1
    if s in ("0", "ham", "legitimate", "legit", "safe", "safe email", "not phishing"):
        return 0
    try:
        v = int(float(s))
        return v if v in (0, 1) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(
    path: str | Path,
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Load test CSV into list of dicts with [subject, body, label].
    Auto-detects column layout. Stratified-samples when max_samples is set.
    """
    path = Path(path)
    sep  = ";" if _sniff_sep(path) == ";" else ","
    rows: list[dict] = []

    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        fn     = [f.strip() for f in (reader.fieldnames or [])]
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            # Layout detection
            if "Email Text" in row and "Email Type" in row:
                body, subject, raw_lbl = row["Email Text"][:8000], "", row["Email Type"]
            elif "Label" in row:
                subject, body = row.get("subject","")[:400], row.get("body","")[:8000]
                raw_lbl = row.get("Label","")
            elif "label" in row:
                subject, body = row.get("subject","")[:400], row.get("body","")[:8000]
                raw_lbl = row.get("label","")
            else:
                continue
            lbl = _norm_label(raw_lbl)
            if lbl is None:
                continue
            rows.append({"subject": subject.strip(), "body": body.strip(), "label": lbl})

    if max_samples and len(rows) > max_samples:
        rng  = random.Random(seed)
        pos  = [r for r in rows if r["label"] == 1]
        neg  = [r for r in rows if r["label"] == 0]
        half = max_samples // 2
        rows = rng.sample(pos, min(len(pos), half)) + rng.sample(neg, min(len(neg), half))
        deficit = max_samples - len(rows)
        if deficit > 0:
            used    = {id(r) for r in rows}
            remain  = [r for r in (pos + neg) if id(r) not in used]
            rows   += rng.sample(remain, min(len(remain), deficit))
        rng.shuffle(rows)

    return rows


def _sniff_sep(path: Path) -> str:
    try:
        line = path.read_text(encoding="utf-8", errors="replace").split("\n")[0]
        return ";" if line.count(";") > line.count(",") else ","
    except Exception:
        return ","


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _warmup_pipeline() -> None:
    """Pre-load all models so worker threads don't each trigger a cold-start."""
    from app.parsers.text_parser       import parse_text_input
    from app.models.semantic_engine    import analyze_semantics
    from app.models.structural_engine  import analyze_structure
    from app.models.classifier         import _load_model, _load_pca, _load_multiout
    _load_model()
    _load_pca()
    _load_multiout()
    logger.info("  Pipeline warm-up complete")


def _run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full MCO pipeline on a single sample. Thread-safe."""
    from app.parsers.text_parser       import parse_text_input
    from app.models.semantic_engine    import analyze_semantics
    from app.models.structural_engine  import analyze_structure
    from app.models.classifier         import classify, INDICATOR_NAMES

    t0 = time.perf_counter()
    try:
        parsed  = parse_text_input(sample["subject"], sample["body"])
        sem_res = analyze_semantics(sample["subject"], sample["body"])
        str_res = analyze_structure(parsed)
        clf_res = classify(sem_res, str_res)

        verdict      = clf_res["verdict"]
        confidence   = float(clf_res.get("confidence", 0.0))
        pred_label   = 1 if verdict == "phishing" else 0
        # confidence IS always the phishing probability — use it directly as the ROC score.
        # (Higher = more likely phishing, lower = more likely legitimate, regardless of verdict.)
        roc_score = confidence

        # Indicator flags
        ind_flags = clf_res.get("indicator_flags", {})

        # Rule scores for FN analysis
        rule_scores = sem_res.get("rule_scores", {})
        top_category = max(
            ["urgency", "authority", "pressure", "credential_requests",
             "reward_lures", "generic_greetings"],
            key=lambda k: rule_scores.get(k, {}).get("score", 0.0)
            if isinstance(rule_scores.get(k), dict) else 0.0,
        )

        return {
            **sample,
            "predicted_label":   pred_label,
            "predicted_verdict": verdict,
            "confidence":        confidence,
            "roc_score":         roc_score,
            "method":            clf_res.get("method", "?"),
            "feature_mode":      clf_res.get("feature_mode", "?"),
            "degraded_mode":     clf_res.get("degraded_mode", False),
            "indicator_flags":   ind_flags,
            "top_semantic_cat":  top_category,
            "elapsed_s":         time.perf_counter() - t0,
            "error":             None,
        }
    except Exception as exc:
        return {
            **sample,
            "predicted_label":   0,
            "predicted_verdict": "error",
            "confidence":        0.0,
            "roc_score":         0.0,
            "method":            "error",
            "feature_mode":      "error",
            "degraded_mode":     True,
            "indicator_flags":   {},
            "top_semantic_cat":  "unknown",
            "elapsed_s":         time.perf_counter() - t0,
            "error":             str(exc),
        }


def run_pipeline(
    samples: List[Dict[str, Any]],
    workers: int = 4,
    label: str = "Evaluating",
) -> List[Dict[str, Any]]:
    """Run pipeline on all samples with parallel workers and progress reporting."""
    results: List[Optional[Dict]] = [None] * len(samples)
    n         = len(samples)
    completed = 0
    t_start   = time.perf_counter()

    logger.info(f"  {label}: {n:,} samples  workers={workers}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(_run_one, s): i for i, s in enumerate(samples)}
        for future in as_completed(future_to_idx):
            idx            = future_to_idx[future]
            results[idx]   = future.result()
            completed     += 1
            if completed % max(1, n // 20) == 0 or completed == n:
                elapsed = time.perf_counter() - t_start
                rate    = completed / elapsed
                eta     = (n - completed) / max(rate, 0.01)
                logger.info(f"  [{completed:>{len(str(n))}}/{n}]  "
                            f"{rate:.1f}/s  ETA {eta:.0f}s")

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: List[Dict[str, Any]],
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute the full metrics suite required by the paper."""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, cohen_kappa_score,
    )

    y_true  = np.array([r["label"]           for r in results])
    y_pred  = np.array([r["predicted_label"]  for r in results])
    y_score = np.array([r["roc_score"]        for r in results])

    n     = len(y_true)
    tp    = int(((y_true==1)&(y_pred==1)).sum())
    tn    = int(((y_true==0)&(y_pred==0)).sum())
    fp    = int(((y_true==0)&(y_pred==1)).sum())
    fn    = int(((y_true==1)&(y_pred==0)).sum())

    accuracy  = accuracy_score(y_true, y_pred)
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) else 0.0
    kappa     = cohen_kappa_score(y_true, y_pred)

    prec_m, rec_m, f1_m, _   = precision_recall_fscore_support(
        y_true, y_pred, average="macro",    zero_division=0)
    prec_w, rec_w, f1_w, _   = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    prec_ph, rec_ph, f1_ph, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary",   zero_division=0)
    prec_lg = tn / (tn + fn) if (tn + fn) else 0.0
    rec_lg  = tn / (tn + fp) if (tn + fp) else 0.0
    f1_lg   = 2 * prec_lg * rec_lg / (prec_lg + rec_lg) if (prec_lg + rec_lg) else 0.0

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    # ── Bootstrap 95% CI ────────────────────────────────────────────────────
    rng       = np.random.default_rng(seed)
    bt_acc, bt_f1, bt_auc, bt_fpr, bt_fnr = [], [], [], [], []
    for _ in range(bootstrap_n):
        idx  = rng.integers(0, n, size=n)
        yt_b = y_true[idx];  yp_b = y_pred[idx];  ys_b = y_score[idx]
        bt_acc.append(accuracy_score(yt_b, yp_b))
        _, _, f1b, _ = precision_recall_fscore_support(
            yt_b, yp_b, average="macro", zero_division=0)
        bt_f1.append(f1b)
        try:
            bt_auc.append(roc_auc_score(yt_b, ys_b))
        except ValueError:
            bt_auc.append(float("nan"))
        _tp = int(((yt_b==1)&(yp_b==1)).sum())
        _tn = int(((yt_b==0)&(yp_b==0)).sum())
        _fp = int(((yt_b==0)&(yp_b==1)).sum())
        _fn = int(((yt_b==1)&(yp_b==0)).sum())
        bt_fpr.append(_fp / (_fp + _tn) if (_fp + _tn) else 0.0)
        bt_fnr.append(_fn / (_fn + _tp) if (_fn + _tp) else 0.0)

    def _ci(arr):
        a = np.array([x for x in arr if not np.isnan(x)])
        return (round(float(np.percentile(a, 2.5)),  4),
                round(float(np.percentile(a, 97.5)), 4))

    ci_acc  = _ci(bt_acc)
    ci_f1   = _ci(bt_f1)
    ci_auc  = _ci(bt_auc)
    ci_fpr  = _ci(bt_fpr)
    ci_fnr  = _ci(bt_fnr)

    # ── Indicator fires ──────────────────────────────────────────────────────
    from app.models.classifier import INDICATOR_NAMES
    ind_counts = {name: 0 for name in INDICATOR_NAMES}
    for r in results:
        for k, v in r.get("indicator_flags", {}).items():
            if v and k in ind_counts:
                ind_counts[k] += 1

    # ── FN category analysis ─────────────────────────────────────────────────
    false_negatives = [r for r in results if r["label"]==1 and r["predicted_label"]==0]
    fn_cat_counts: dict = {}
    for r in false_negatives:
        cat = r.get("top_semantic_cat", "unknown")
        fn_cat_counts[cat] = fn_cat_counts.get(cat, 0) + 1

    # ── Method distribution ──────────────────────────────────────────────────
    method_dist = {}
    for r in results:
        m = r.get("method", "?")
        method_dist[m] = method_dist.get(m, 0) + 1

    return {
        "n_total":    n,
        "n_phishing": int(y_true.sum()),
        "n_legit":    int((y_true==0).sum()),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "accuracy":         round(float(accuracy), 4),
        "macro_precision":  round(float(prec_m),   4),
        "macro_recall":     round(float(rec_m),    4),
        "macro_f1":         round(float(f1_m),     4),
        "weighted_f1":      round(float(f1_w),     4),
        "fpr":              round(float(fpr),      4),
        "fnr":              round(float(fnr),      4),
        "roc_auc":          round(float(auc),      4),
        "cohen_kappa":      round(float(kappa),    4),
        "phishing_class":   {"precision": round(float(prec_ph),4), "recall": round(float(rec_ph),4), "f1": round(float(f1_ph),4), "support": int(y_true.sum())},
        "legit_class":      {"precision": round(prec_lg,4),        "recall": round(rec_lg,4),        "f1": round(f1_lg,4),        "support": int((y_true==0).sum())},
        "confidence_intervals_95": {
            "accuracy": ci_acc,
            "macro_f1": ci_f1,
            "roc_auc":  ci_auc,
            "fpr":      ci_fpr,
            "fnr":      ci_fnr,
        },
        "indicator_fire_counts":  ind_counts,
        "fn_category_counts":     fn_cat_counts,
        "method_distribution":    method_dist,
        "n_degraded":             sum(1 for r in results if r.get("degraded_mode")),
        "n_errors":               sum(1 for r in results if r.get("error")),
        "avg_conf_phishing":      round(float(np.mean([r["confidence"] for r in results if r["predicted_label"]==1])) if any(r["predicted_label"]==1 for r in results) else 0.0, 4),
        "avg_conf_legit":         round(float(np.mean([r["confidence"] for r in results if r["predicted_label"]==0])) if any(r["predicted_label"]==0 for r in results) else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Rule-only comparison pass
# ---------------------------------------------------------------------------

def run_rule_only_pass(
    samples: List[Dict[str, Any]],
    workers: int,
) -> Dict[str, Any]:
    """
    Re-run evaluation with DistilBERT forcibly disabled.
    Patches the semantic engine module-level globals in this process.
    """
    import app.models.semantic_engine as _sem
    import app.models.classifier      as _clf

    # Patch DistilBERT off
    orig_attempted = _sem._model_load_attempted
    orig_model     = _sem._transformer_model
    orig_tok       = _sem._transformer_tokenizer

    _sem._model_load_attempted  = True
    _sem._transformer_model     = None
    _sem._transformer_tokenizer = None

    # Reset classifier cache so it reloads without PCA forcing
    _clf._pca         = None
    _clf._pca_loaded  = False

    try:
        logger.info("  [Rule-only pass] DistilBERT disabled, PCA bypassed")
        results = run_pipeline(samples, workers=workers, label="Rule-only pass")
        metrics = compute_all_metrics(results, bootstrap_n=200)
    finally:
        # Restore original state
        _sem._model_load_attempted  = orig_attempted
        _sem._transformer_model     = orig_model
        _sem._transformer_tokenizer = orig_tok
        _clf._pca         = None
        _clf._pca_loaded  = False

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(
    results:      List[Dict[str, Any]],
    metrics:      Dict[str, Any],
    results_dir:  Path,
) -> None:
    """Generate and save all four publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve
    except ImportError as e:
        logger.warning(f"Skipping plots — missing dependency: {e}")
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Confusion Matrix ──────────────────────────────────────────────────
    cm   = metrics["confusion_matrix"]
    mat  = np.array(cm["matrix"])      # [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        mat,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Legit", "Predicted Phishing"],
        yticklabels=["Actual Legit",    "Actual Phishing"],
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(
        f"Confusion Matrix\n"
        f"Accuracy={metrics['accuracy']:.3f}  F1={metrics['macro_f1']:.3f}  "
        f"AUC={metrics['roc_auc']:.3f}",
        fontsize=12, pad=12,
    )
    plt.tight_layout()
    _savefig(fig, results_dir / "confusion_matrix.png")

    # ── 2. ROC Curve ─────────────────────────────────────────────────────────
    y_true  = np.array([r["label"]     for r in results])
    y_score = np.array([r["roc_score"] for r in results])
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, color="#2563EB", lw=2,
            label=f"Proposed AI (AUC = {metrics['roc_auc']:.3f})")
    # EBIDS reference line (paper baseline)
    ax.plot([0, 0.15, 1], [0, 0.75, 1], color="#DC2626", lw=2,
            linestyle="--", label="EBIDS reference (AUC ≈ 0.80)")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Phishing Detection", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _savefig(fig, results_dir / "roc_curve.png")

    # ── 3. Indicator Distribution ────────────────────────────────────────────
    from app.models.classifier import INDICATOR_NAMES
    ind_data = metrics.get("indicator_fire_counts", {})
    ind_vals = [ind_data.get(n, 0) for n in INDICATOR_NAMES]
    ind_pct  = [v / max(metrics["n_total"], 1) * 100 for v in ind_vals]

    # Short display names
    short_names = [
        n.replace("_detected","").replace("_"," ").title()
        for n in INDICATOR_NAMES
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563EB" if v > np.mean(ind_pct) else "#93C5FD" for v in ind_pct]
    bars = ax.barh(range(len(INDICATOR_NAMES)), ind_pct, color=colors, edgecolor="white")
    ax.set_yticks(range(len(INDICATOR_NAMES)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("% of emails triggering indicator", fontsize=11)
    ax.set_title("Phishing Indicator Fire Rate\n"
                 "(fraction of test emails where each indicator was active)",
                 fontsize=12, pad=10)
    for bar, pct in zip(bars, ind_pct):
        if pct > 1:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{pct:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    _savefig(fig, results_dir / "indicator_distribution.png")

    # ── 4. False Negative Analysis ───────────────────────────────────────────
    fn_cats = metrics.get("fn_category_counts", {})
    if fn_cats:
        cats = sorted(fn_cats, key=fn_cats.get, reverse=True)
        vals = [fn_cats[c] for c in cats]
        fn_total = sum(vals)

        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = sns.color_palette("Reds_r", len(cats))
        ax.bar(
            [c.replace("_", " ").title() for c in cats],
            vals,
            color=bar_colors,
            edgecolor="white",
        )
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlabel("Dominant Semantic Category", fontsize=11)
        ax.set_title(
            f"False Negative Analysis\n"
            f"({fn_total} missed phishing emails — dominant semantic category)",
            fontsize=12, pad=10,
        )
        plt.xticks(rotation=20, ha="right", fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
        plt.tight_layout()
        _savefig(fig, results_dir / "false_negative_analysis.png")
    else:
        logger.info("  No false negatives to plot")


def _savefig(fig, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    kb = path.stat().st_size / 1024
    logger.info(f"  Saved {path.name}  ({kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

_SEP  = "=" * 66
_SEP2 = "-" * 66

def print_full_report(m: Dict[str, Any], title: str = "Evaluation Report") -> None:
    """Print comprehensive metrics report to stdout."""
    ci = m.get("confidence_intervals_95", {})

    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)
    n, np_, nl = m["n_total"], m["n_phishing"], m["n_legit"]
    print(f"  Test set : {n:,} emails  "
          f"(phishing={np_:,} [{np_/n*100:.1f}%], legit={nl:,} [{nl/n*100:.1f}%])")
    meth = m.get("method_distribution", {})
    print(f"  Methods  : {meth}")
    if m.get("n_degraded"):
        print(f"  [!] Degraded-mode (no BERT/PCA): {m['n_degraded']} samples")
    print()

    # Confusion matrix
    cm = m["confusion_matrix"]
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    print("  Confusion Matrix (rows=Actual, cols=Predicted):")
    print(f"  {'':18s}  {'Pred Legit':>12}  {'Pred Phishing':>14}")
    print(f"  {'Actual Legit':18s}  {tn:>12,}  {fp:>14,}")
    print(f"  {'Actual Phishing':18s}  {fn:>12,}  {tp:>14,}")
    print()

    # Per-class table
    ph = m["phishing_class"];  lg = m["legit_class"]
    print(f"  {'Class':<14} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>10}")
    print(f"  {_SEP2[2:]}")
    print(f"  {'Phishing':<14} {ph['precision']:>8.4f} {ph['recall']:>8.4f} {ph['f1']:>8.4f} {ph['support']:>10,}")
    print(f"  {'Legitimate':<14} {lg['precision']:>8.4f} {lg['recall']:>8.4f} {lg['f1']:>8.4f} {lg['support']:>10,}")
    print(f"  {_SEP2[2:]}")
    print(f"  {'Macro avg':<14} {m['macro_precision']:>8.4f} {m['macro_recall']:>8.4f} {m['macro_f1']:>8.4f}")
    print(f"  {'Weighted avg':<14} {'':>8} {'':>8} {m['weighted_f1']:>8.4f}")
    print()

    # Key metrics with CI
    print("  Key Metrics (95% Bootstrap CI):")
    _mline("Accuracy",     m["accuracy"],   ci.get("accuracy", ("-","-")))
    _mline("Macro F1",     m["macro_f1"],   ci.get("macro_f1", ("-","-")))
    _mline("ROC-AUC",      m["roc_auc"],    ci.get("roc_auc",  ("-","-")))
    _mline("FP Rate",      m["fpr"],        ci.get("fpr",      ("-","-")))
    _mline("FN Rate",      m["fnr"],        ci.get("fnr",      ("-","-")))
    print(f"  {'Cohen Kappa':<16}  {m['cohen_kappa']:.4f}")
    print(f"{_SEP}\n")


def _mline(name: str, val: float, ci: tuple) -> None:
    lo, hi = ci
    ci_str = f"[{lo:.4f}, {hi:.4f}]" if isinstance(lo, float) else "N/A"
    print(f"  {name:<16}  {val:.4f}   95% CI {ci_str}")


def print_comparison_table(primary: Dict, rule_only: Dict) -> None:
    """Print side-by-side DistilBERT-mode vs rule-only comparison."""
    def _delta(a, b):
        d = a - b
        sign = "+" if d >= 0 else ""
        return f"({sign}{d*100:.1f}pp)"

    print(f"\n{_SEP}")
    print("  Mode Comparison: Full Pipeline vs Rule-Only")
    print(_SEP)
    print(f"  {'Metric':<18} {'Full Pipeline':>14} {'Rule-Only':>12} {'Delta':>12}")
    print(f"  {_SEP2[2:]}")
    for k, label in [
        ("accuracy",  "Accuracy"),
        ("macro_f1",  "Macro F1"),
        ("roc_auc",   "ROC-AUC"),
        ("fpr",       "FP Rate"),
        ("fnr",       "FN Rate"),
        ("cohen_kappa","Kappa"),
    ]:
        pv = primary.get(k, 0.0)
        rv = rule_only.get(k, 0.0)
        print(f"  {label:<18} {pv:>14.4f} {rv:>12.4f} {_delta(pv,rv):>12}")
    print(f"  {_SEP2[2:]}")
    print(f"  {'Method':<18} {str(primary.get('method_distribution',{})):>26}")
    print(f"{_SEP}\n")


def print_table3(m: Dict[str, Any]) -> None:
    """Print paper Table 3 — ready for direct copy-paste."""
    print(f"\n{_SEP}")
    print("  Paper Table 3 — Phishing Detection System Comparison")
    print("  (Copy-paste ready for LaTeX or Word)")
    print(_SEP)
    hdr = f"  {'System':<22} | {'Accuracy':>9} | {'FP Rate':>8} | {'FN Rate':>8} | {'F1':>7} | {'AUC':>7}"
    div = "  " + "-"*22 + "+-" + "-"*9 + "+-" + "-"*8 + "+-" + "-"*8 + "+-" + "-"*7 + "+-" + "-"*7
    print(hdr)
    print(div)
    # Reference baselines
    refs = [
        ("SpamAssassin",   "97.3%", "2.7%",   "2.6%",   "—",    "—"),
        ("EBIDS (Loo'25)", "75.0%", "15.0%",  "25.0%",  "—",    "—"),
        ("DistilBERT+RF",  "82.4%", "10.3%",  "15.8%",  "0.823","0.891"),
    ]
    for name, acc, fpr, fnr, f1, auc in refs:
        print(f"  {name:<22} | {acc:>9} | {fpr:>8} | {fnr:>8} | {f1:>7} | {auc:>7}")
    print(div)
    # Our result
    acc = f"{m['accuracy']*100:.1f}%"
    fpr = f"{m['fpr']*100:.2f}%"
    fnr = f"{m['fnr']*100:.2f}%"
    f1  = f"{m['macro_f1']:.3f}"
    auc = f"{m['roc_auc']:.3f}" if not np.isnan(m["roc_auc"]) else "N/A"
    ci  = m.get("confidence_intervals_95", {})
    note = ""
    if ci.get("accuracy"):
        lo, hi = ci["accuracy"]
        note = f"  95% CI: [{lo:.3f}, {hi:.3f}]"
    print(f"  {'Proposed AI (ours)':<22} | {acc:>9} | {fpr:>8} | {fnr:>8} | {f1:>7} | {auc:>7}   <-- {note}")
    print(f"{_SEP}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _parse_args()

    # ── 0. Disable DistilBERT if requested ───────────────────────────────────
    if cfg.no_bert:
        import app.models.semantic_engine as _sem
        _sem._model_load_attempted  = True
        _sem._transformer_model     = None
        _sem._transformer_tokenizer = None
        logger.info("  DistilBERT disabled via --no-bert")

    # ── 1. Locate test data ───────────────────────────────────────────────────
    test_path = Path(cfg.test)
    if not test_path.exists():
        logger.warning(f"  {test_path} not found — using fallback: {cfg.fallback_test}")
        test_path = Path(cfg.fallback_test)
    if not test_path.exists():
        logger.error(f"  No test data found. Provide --test or run dataset pipeline first.")
        sys.exit(1)

    print(f"\n{_SEP}")
    print("  Agentic AI Phishing Detector — Evaluation")
    print(_SEP)
    logger.info(f"  Test file   : {test_path}")
    logger.info(f"  Max samples : {cfg.max_samples or 'all'}")
    logger.info(f"  Workers     : {cfg.workers}")
    logger.info(f"  Bootstrap N : {cfg.bootstrap_n}")
    logger.info(f"  Results dir : {cfg.results_dir}")
    logger.info("")

    # ── 2. Load test data ─────────────────────────────────────────────────────
    samples = load_test_data(test_path, cfg.max_samples, cfg.seed)
    n_ph = sum(1 for s in samples if s["label"] == 1)
    logger.info(f"  Loaded {len(samples):,} test samples  "
                f"(phishing={n_ph:,}, legit={len(samples)-n_ph:,})")

    # ── 3. Warm up and run primary evaluation ─────────────────────────────────
    logger.info("\n  [Primary pass] Warming up pipeline...")
    _warmup_pipeline()

    t_eval = time.perf_counter()
    primary_results = run_pipeline(samples, workers=cfg.workers, label="Primary pass")
    logger.info(f"  Primary pass complete in {time.perf_counter()-t_eval:.1f}s")

    # ── 4. Compute all metrics ────────────────────────────────────────────────
    logger.info("\n  Computing metrics (bootstrap n=%d)...", cfg.bootstrap_n)
    primary_metrics = compute_all_metrics(
        primary_results, bootstrap_n=cfg.bootstrap_n, seed=cfg.seed
    )

    # ── 5. Rule-only comparison ───────────────────────────────────────────────
    rule_metrics = None
    if not cfg.no_bert and primary_metrics.get("n_degraded", 0) < len(samples):
        logger.info("\n  [Rule-only comparison pass]")
        rule_metrics = run_rule_only_pass(samples, workers=cfg.workers)
    else:
        logger.info("\n  Skipping rule-only comparison (already in degraded mode)")

    # ── 6. Print reports ──────────────────────────────────────────────────────
    print_full_report(primary_metrics, title="Evaluation Report — Proposed Agentic AI System")
    if rule_metrics:
        print_comparison_table(primary_metrics, rule_metrics)
    print_table3(primary_metrics)

    # ── 7. Generate plots ─────────────────────────────────────────────────────
    results_dir = Path(cfg.results_dir)
    if not cfg.skip_plots:
        logger.info("  Generating publication figures...")
        generate_plots(primary_results, primary_metrics, results_dir)
    else:
        logger.info("  Skipping plots (--skip-plots)")

    # ── 8. Save JSON summary ──────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in primary_metrics.items()
               if k not in ("indicator_fire_counts", "fn_category_counts")}
    summary["indicator_fire_counts"] = primary_metrics.get("indicator_fire_counts", {})
    summary["fn_category_counts"]    = primary_metrics.get("fn_category_counts", {})
    if rule_metrics:
        summary["rule_only_comparison"] = {
            k: rule_metrics[k]
            for k in ("accuracy","macro_f1","fpr","fnr","roc_auc","cohen_kappa")
        }

    json_path = results_dir / "metrics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    size_kb = json_path.stat().st_size / 1024
    logger.info(f"  metrics_summary.json saved  ({size_kb:.1f} KB)")

    logger.info(f"\n  All outputs in: {results_dir}/")
    logger.info("  Done.")


if __name__ == "__main__":
    main()

"""
Multi-Corpus Dataset Pipeline for Phishing Detection.

Loads, normalises, merges, and splits six public email datasets into a
unified DataFrame with columns [subject, body, label] where label=1 is
phishing/spam and label=0 is legitimate.

Datasets supported (matching paper claim of ~82,500 emails):
  1. SpamAssassin Public Corpus     (~25 000 sampled)
  2. CEAS 2008 Spam Dataset         (~14 000 sampled)
  3. Enron Spam/Ham Dataset         (~30 000 sampled)
  4. Ling Spam Dataset              (  5 475 full)
  5. Nazario Phishing Corpus        (  2 793 full)
  6. Nigerian Fraud (419) Corpus    (  5 232 sampled)
  ─────────────────────────────────────────────────
  Target total                      ~82 500

Usage (command-line):
  python -m app.data.dataset_loader --data-dir ./data/raw

Usage (library):
  from app.data.dataset_loader import build_unified_corpus, split_dataset
  df = build_unified_corpus("./data/raw")
  train, val, test = split_dataset(df)

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

from __future__ import annotations

import csv
import email
import glob
import logging
import os
import random
import tarfile
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per field — needed for large bodies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default per-dataset sample caps (tuned to reach ~82 500 total)
# ---------------------------------------------------------------------------
_DEFAULT_CAPS = {
    "spamassassin": 25_000,   # balanced — 12 500 spam + 12 500 ham
    "ceas":         14_000,   # balanced — 7 000 spam + 7 000 ham
    "enron":        30_000,   # balanced — 15 000 spam + 15 000 ham
    "ling":         None,     # full corpus (~5 475)
    "nazario":      None,     # full corpus (~2 793, all phishing)
    "nigerian":      5_232,   # all phishing, capped
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_generic(
    path: str | Path,
    subject_col: str,
    body_col: str,
    label_col: str,
    separator: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Generic CSV reader that extracts [subject, body, label] and normalises
    the label column to integer 0/1.  Handles quoted multi-line fields via
    Python's csv module so large bodies don't confuse the parser.
    """
    rows: list[dict] = []
    path = str(path)
    with open(path, encoding=encoding, errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=separator)
        if reader.fieldnames is None:
            logger.warning(f"No headers found in {path}")
            return pd.DataFrame(columns=["subject", "body", "label"])

        for row in reader:
            try:
                raw_label = (row.get(label_col) or "").strip()
                # Normalise: accept "0"/"1", "ham"/"spam", "legit"/"phish*"
                if raw_label in ("1", "spam", "phishing", "phish"):
                    label = 1
                elif raw_label in ("0", "ham", "legitimate", "legit"):
                    label = 0
                else:
                    label = int(float(raw_label))
                    if label not in (0, 1):
                        continue
            except (ValueError, TypeError):
                continue

            subject = (row.get(subject_col) or "").strip()[:400]
            body = (row.get(body_col) or "").strip()[:8000]
            rows.append({"subject": subject, "body": body, "label": label})

    df = pd.DataFrame(rows)
    logger.debug(f"  Loaded {len(df):,} rows from {os.path.basename(path)}")
    return df


def _stratified_sample(df: pd.DataFrame, max_samples: Optional[int], seed: int = 42) -> pd.DataFrame:
    """
    Downsample *df* to at most *max_samples* rows while preserving the class
    balance present in the data.  For phishing-only corpora the entire
    corpus is returned up to *max_samples*.
    """
    if max_samples is None or len(df) <= max_samples:
        return df.copy()

    n_classes = df["label"].nunique()
    if n_classes == 1:
        # All one class — simple head-sample
        return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    # Balanced sampling: equal share from each class where possible
    per_class = max_samples // n_classes
    parts = []
    for lbl, grp in df.groupby("label"):
        n = min(len(grp), per_class)
        parts.append(grp.sample(n=n, random_state=seed))
    result = pd.concat(parts, ignore_index=True)

    # If one class had fewer samples than per_class, fill remainder from the other
    deficit = max_samples - len(result)
    if deficit > 0:
        already = set(result.index)
        remainder = df.drop(index=already, errors="ignore")
        if len(remainder) >= deficit:
            result = pd.concat(
                [result, remainder.sample(n=deficit, random_state=seed)],
                ignore_index=True,
            )

    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def load_spamassassin(
    path: str | Path,
    max_samples: Optional[int] = _DEFAULT_CAPS["spamassassin"],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the SpamAssassin Public Corpus.

    Accepts either:
      • A CSV with columns: [sender, receiver, date, subject, body, label, urls]
        (label: 0=ham, 1=spam)
      • A directory containing the SpamAssassin maildir tree (easy_ham/,
        spam/, hard_ham/, spam_2/ …) or a tar.bz2 archive of the same.

    Args:
        path: Path to SpamAssasin.csv *or* to the directory/archive.
        max_samples: Maximum rows to return (balanced). None = no cap.
        random_state: RNG seed for reproducible sampling.

    Returns:
        DataFrame with columns [subject, body, label].
    """
    path = Path(path)

    # --- CSV path ---
    if path.is_file() and path.suffix.lower() == ".csv":
        df = _read_csv_generic(path, "subject", "body", "label")
        df["source"] = "spamassassin"
        return _stratified_sample(df, max_samples, seed=random_state)

    # --- Directory / archive path ---
    rows: list[dict] = []
    eml_paths: list[Tuple[Path, int]] = []   # (file_path, label)

    if path.is_dir():
        # Discover sub-folders: spam* folders → label=1, ham* folders → label=0
        for sub in path.iterdir():
            if not sub.is_dir():
                continue
            name_lower = sub.name.lower()
            if "spam" in name_lower:
                lbl = 1
            elif "ham" in name_lower:
                lbl = 0
            else:
                continue
            for fp in sub.iterdir():
                if fp.is_file():
                    eml_paths.append((fp, lbl))

    elif path.suffix in (".bz2", ".gz", ".tar"):
        # Extract from archive on-the-fly
        with tarfile.open(path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name_lower = member.name.lower()
                if "spam" in name_lower:
                    lbl = 1
                elif "ham" in name_lower:
                    lbl = 0
                else:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                try:
                    raw = fh.read()
                    msg = email.message_from_bytes(raw)
                    subject = msg.get("Subject", "") or ""
                    body = _extract_email_body(msg)
                    rows.append({"subject": subject[:400], "body": body[:8000], "label": lbl})
                except Exception:
                    pass
    else:
        logger.warning(f"load_spamassassin: unrecognised path type: {path}")
        return pd.DataFrame(columns=["subject", "body", "label"])

    # Parse discovered .eml files
    for fp, lbl in eml_paths:
        try:
            raw = fp.read_bytes()
            msg = email.message_from_bytes(raw)
            subject = msg.get("Subject", "") or ""
            body = _extract_email_body(msg)
            rows.append({"subject": subject[:400], "body": body[:8000], "label": lbl})
        except Exception:
            pass

    df = pd.DataFrame(rows)
    df["source"] = "spamassassin"
    return _stratified_sample(df, max_samples, seed=random_state)


def load_enron(
    path: str | Path,
    max_samples: Optional[int] = _DEFAULT_CAPS["enron"],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the Enron spam/ham email dataset.

    Accepts either:
      • A CSV with columns [subject, body, label]  (label: 0=ham, 1=spam)
      • A directory in Enron maildir format (each user's mailbox as
        sub-directories; spam/ and ham/ sub-folders)

    Args:
        path: Path to Enron.csv or maildir root.
        max_samples: Maximum rows (balanced). None = no cap.
        random_state: RNG seed.

    Returns:
        DataFrame with columns [subject, body, label].
    """
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".csv":
        df = _read_csv_generic(path, "subject", "body", "label")
        df["source"] = "enron"
        return _stratified_sample(df, max_samples, seed=random_state)

    # Maildir directory tree
    rows: list[dict] = []
    if path.is_dir():
        for fp in path.rglob("*"):
            if not fp.is_file():
                continue
            parts_lower = [p.lower() for p in fp.parts]
            if any("spam" in p for p in parts_lower):
                lbl = 1
            elif any("ham" in p or "inbox" in p for p in parts_lower):
                lbl = 0
            else:
                continue
            try:
                raw = fp.read_bytes()
                msg = email.message_from_bytes(raw)
                subject = msg.get("Subject", "") or ""
                body = _extract_email_body(msg)
                rows.append({"subject": subject[:400], "body": body[:8000], "label": lbl})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df["source"] = "enron"
    return _stratified_sample(df, max_samples, seed=random_state)


def load_ling(
    path: str | Path,
    max_samples: Optional[int] = _DEFAULT_CAPS["ling"],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the Ling-Spam dataset.

    Expects a CSV with columns [subject, body, label]  (label: 0=ham, 1=spam).

    Args:
        path: Path to Ling.csv.
        max_samples: Maximum rows. None = no cap.
        random_state: RNG seed.

    Returns:
        DataFrame with columns [subject, body, label].
    """
    path = Path(path)
    df = _read_csv_generic(path, "subject", "body", "label")
    df["source"] = "ling"
    return _stratified_sample(df, max_samples, seed=random_state)


def load_ceas(
    path: str | Path,
    max_samples: Optional[int] = _DEFAULT_CAPS["ceas"],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the CEAS 2008 spam dataset.

    Expects a CSV with columns [sender, receiver, date, subject, body, label, urls]
    (label: 0=ham, 1=spam).

    Args:
        path: Path to CEAS_08.csv.
        max_samples: Maximum rows (balanced). None = no cap.
        random_state: RNG seed.

    Returns:
        DataFrame with columns [subject, body, label].
    """
    path = Path(path)
    df = _read_csv_generic(path, "subject", "body", "label")
    df["source"] = "ceas"
    return _stratified_sample(df, max_samples, seed=random_state)


def load_nazario(
    path: str | Path,
    max_samples: Optional[int] = _DEFAULT_CAPS["nazario"],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the Nazario Phishing Corpus.

    Accepts either:
      • A CSV with columns [sender, receiver, date, subject, body, urls, label]
        All rows are typically phishing (label=1).
      • A directory of .eml files (all treated as phishing).

    Args:
        path: Path to Nazario.csv or .eml directory.
        max_samples: Maximum rows. None = no cap.
        random_state: RNG seed.

    Returns:
        DataFrame with columns [subject, body, label].
    """
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".csv":
        df = _read_csv_generic(path, "subject", "body", "label")
        # If label column is missing or all-zero, treat entire corpus as phishing
        if df.empty or df["label"].sum() == 0:
            df["label"] = 1
        df["source"] = "nazario"
        return _stratified_sample(df, max_samples, seed=random_state)

    # Directory of .eml files
    rows: list[dict] = []
    if path.is_dir():
        for fp in path.rglob("*.eml"):
            try:
                raw = fp.read_bytes()
                msg = email.message_from_bytes(raw)
                subject = msg.get("Subject", "") or ""
                body = _extract_email_body(msg)
                rows.append({"subject": subject[:400], "body": body[:8000], "label": 1})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df["source"] = "nazario"
    return _stratified_sample(df, max_samples, seed=random_state)


def load_nigerian(
    path: str | Path,
    max_samples: Optional[int] = _DEFAULT_CAPS["nigerian"],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the Nigerian Fraud (419) email corpus.

    Accepts either:
      • A CSV with columns [sender, receiver, date, subject, body, urls, label]
        All rows are phishing (label=1).
      • A directory of raw .txt email files (all treated as phishing).

    Args:
        path: Path to Nigerian_Fraud.csv or directory of .txt files.
        max_samples: Maximum rows. None = no cap.
        random_state: RNG seed.

    Returns:
        DataFrame with columns [subject, body, label].
    """
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".csv":
        df = _read_csv_generic(path, "subject", "body", "label")
        if df.empty or df["label"].sum() == 0:
            df["label"] = 1
        df["source"] = "nigerian"
        return _stratified_sample(df, max_samples, seed=random_state)

    # Directory of raw text files
    rows: list[dict] = []
    if path.is_dir():
        for fp in path.rglob("*"):
            if not fp.is_file() or fp.suffix.lower() not in (".txt", ".eml", ""):
                continue
            try:
                raw = fp.read_bytes()
                msg = email.message_from_bytes(raw)
                subject = msg.get("Subject", "") or ""
                body = _extract_email_body(msg)
                if not body:
                    body = raw.decode("utf-8", errors="replace")[:8000]
                rows.append({"subject": subject[:400], "body": body[:8000], "label": 1})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df["source"] = "nigerian"
    return _stratified_sample(df, max_samples, seed=random_state)


# ---------------------------------------------------------------------------
# Optional: SpaPhish Spanish corpus (bonus — not in the original paper)
# ---------------------------------------------------------------------------

def load_spaphish(
    path: str | Path,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the SpaPhish Spanish phishing corpus (DiB format).

    Expects a semicolon-delimited CSV with columns including
    [subject, body, Label]  (Label: 0=legitimate, 1=phishing).

    This dataset is *not* part of the original paper's six corpora but
    is included here for completeness and Spanish-language coverage.

    Args:
        path: Path to SpaPhish CSV (semicolon-separated).
        max_samples: Maximum rows. None = no cap.
        random_state: RNG seed.

    Returns:
        DataFrame with columns [subject, body, label].
    """
    path = Path(path)
    df = _read_csv_generic(path, "subject", "body", "Label", separator=";")
    df["source"] = "spaphish"
    return _stratified_sample(df, max_samples, seed=random_state)


# ---------------------------------------------------------------------------
# Merge and split
# ---------------------------------------------------------------------------

def build_unified_corpus(
    data_dir: str | Path,
    *,
    include_spaphish: bool = False,
    spaphish_path: Optional[str | Path] = None,
    random_state: int = 42,
    per_dataset_caps: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Discover and load all six paper datasets from *data_dir*, concatenate
    them into a single DataFrame, deduplicate on body text, and shuffle.

    Expected files in *data_dir*  (names are matched case-insensitively):
      SpamAssasin.csv  /  spam_assassin.csv  /  spamassassin.csv
      CEAS_08.csv      /  ceas.csv
      Enron.csv        /  enron.csv
      Ling.csv         /  ling.csv
      Nazario.csv      /  nazario.csv
      Nigerian_Fraud.csv / nigerian_fraud.csv / nigerian.csv

    Args:
        data_dir: Directory containing the raw CSV files.
        include_spaphish: If True, also load SpaPhish from *spaphish_path*.
        spaphish_path: Explicit path to SpaPhish CSV.  If None and
            *include_spaphish* is True, looks for it in *data_dir*.
        random_state: RNG seed for dedup shuffle.
        per_dataset_caps: Optional dict overriding default sample caps, e.g.
            {"enron": 10000, "nigerian": 2000}.

    Returns:
        Unified, deduplicated, shuffled DataFrame with columns:
        [subject, body, label, source].
    """
    data_dir = Path(data_dir)
    caps = dict(_DEFAULT_CAPS)
    if per_dataset_caps:
        caps.update(per_dataset_caps)

    # --- Resolve file paths by fuzzy name matching ---
    csv_map = _discover_csvs(data_dir)

    parts: list[pd.DataFrame] = []
    dataset_stats: list[dict] = []

    loaders = [
        ("spamassassin", load_spamassassin, caps["spamassassin"]),
        ("ceas",         load_ceas,         caps["ceas"]),
        ("enron",        load_enron,         caps["enron"]),
        ("ling",         load_ling,          caps["ling"]),
        ("nazario",      load_nazario,       caps["nazario"]),
        ("nigerian",     load_nigerian,      caps["nigerian"]),
    ]

    for key, loader_fn, cap in loaders:
        fpath = csv_map.get(key)
        if fpath is None:
            logger.warning(f"  Dataset '{key}' not found in {data_dir} — skipping.")
            continue
        t0 = time.time()
        df_part = loader_fn(fpath, max_samples=cap, random_state=random_state)
        elapsed = time.time() - t0
        n_phish = int((df_part["label"] == 1).sum())
        n_legit = int((df_part["label"] == 0).sum())
        logger.info(
            f"  {key:15s}: {len(df_part):6,} samples  "
            f"(phishing={n_phish:,}, legit={n_legit:,})  [{elapsed:.1f}s]"
        )
        dataset_stats.append({
            "dataset": key, "total": len(df_part),
            "phishing": n_phish, "legit": n_legit,
        })
        parts.append(df_part)

    # Optional SpaPhish
    if include_spaphish:
        if spaphish_path is None:
            spaphish_path = csv_map.get("spaphish")
        if spaphish_path is not None:
            df_spa = load_spaphish(spaphish_path, random_state=random_state)
            n_ph = int((df_spa["label"] == 1).sum())
            n_lg = int((df_spa["label"] == 0).sum())
            logger.info(f"  {'spaphish':15s}: {len(df_spa):6,} samples  "
                        f"(phishing={n_ph:,}, legit={n_lg:,})")
            dataset_stats.append({"dataset": "spaphish", "total": len(df_spa),
                                   "phishing": n_ph, "legit": n_lg})
            parts.append(df_spa)

    if not parts:
        logger.error("No datasets could be loaded. Check data_dir path and file names.")
        return pd.DataFrame(columns=["subject", "body", "label", "source"])

    # Merge
    merged = pd.concat(parts, ignore_index=True)

    # Deduplicate on body (keep first occurrence)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["body"], keep="first")
    after_dedup = len(merged)
    if before_dedup != after_dedup:
        logger.info(f"  Deduplication removed {before_dedup - after_dedup:,} duplicate bodies")

    # Shuffle
    merged = merged.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Summary
    n_total = len(merged)
    n_phish = int((merged["label"] == 1).sum())
    n_legit = int((merged["label"] == 0).sum())
    logger.info(
        f"\n  Unified corpus: {n_total:,} emails  "
        f"(phishing={n_phish:,} [{n_phish/n_total*100:.1f}%], "
        f"legit={n_legit:,} [{n_legit/n_total*100:.1f}%])"
    )

    return merged


def split_dataset(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into stratified train / validation / test sets.

    The split preserves the phishing/legitimate class ratio in every
    partition (stratified on the *label* column).

    Args:
        df: Unified corpus DataFrame with a *label* column.
        train: Fraction for training (default 0.70).
        val:   Fraction for validation (default 0.15).
        test:  Fraction for test (default 0.15).
        random_state: RNG seed.

    Returns:
        (df_train, df_val, df_test) — three DataFrames.

    Raises:
        ValueError: If train + val + test does not sum to 1.0 (±0.001).
    """
    total = train + val + test
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"train + val + test must sum to 1.0, got {total:.4f}"
        )

    # First cut: train vs (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=(val + test),
        random_state=random_state,
        stratify=df["label"],
    )

    # Second cut: val vs test  (from the remaining fraction)
    val_ratio = val / (val + test)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_ratio),
        random_state=random_state,
        stratify=df_temp["label"],
    )

    logger.info(
        f"  Split: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}"
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_email_body(msg: email.message.Message) -> str:
    """
    Walk a parsed email message and extract plain-text body content.
    Falls back to HTML stripping if no text/plain part is found.
    """
    body_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            if ctype == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode(charset, errors="replace"))
            elif ctype == "text/html" and not body_parts:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    html = payload.decode(charset, errors="replace")
                    body_parts.append(_strip_html(html))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            ctype = msg.get_content_type()
            body_parts.append(_strip_html(text) if ctype == "text/html" else text)

    return " ".join(body_parts).strip()[:8000]


def _strip_html(html: str) -> str:
    """Minimal HTML tag stripper (no external dependency)."""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _discover_csvs(data_dir: Path) -> dict[str, Path]:
    """
    Scan *data_dir* for CSV files and map them to known dataset keys by
    fuzzy name matching.

    Returns a dict mapping dataset key → Path.
    """
    _NAME_PATTERNS = {
        "spamassassin": ("spamassassin", "spam_assassin", "spamassasin"),
        "ceas":         ("ceas_08", "ceas08", "ceas"),
        "enron":        ("enron",),
        "ling":         ("ling",),
        "nazario":      ("nazario",),
        "nigerian":     ("nigerian_fraud", "nigerian", "419", "fraud"),
        "spaphish":     ("spaphish", "spaPhish", "spa_phish"),
    }

    found: dict[str, Path] = {}
    if not data_dir.is_dir():
        return found

    for fp in data_dir.iterdir():
        if fp.suffix.lower() != ".csv":
            continue
        stem_lower = fp.stem.lower()
        for key, patterns in _NAME_PATTERNS.items():
            if any(p in stem_lower for p in patterns):
                if key not in found:  # first match wins
                    found[key] = fp
                break

    return found


# ---------------------------------------------------------------------------
# __main__ — CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Build unified phishing email corpus from six public datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="./data/raw",
        help="Directory containing the raw dataset CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="./data/processed",
        help="Directory where train.csv / val.csv / test.csv will be saved.",
    )
    parser.add_argument(
        "--include-spaphish",
        action="store_true",
        help="Also load the SpaPhish Spanish corpus if found in --data-dir.",
    )
    parser.add_argument(
        "--train", type=float, default=0.70,
        help="Training set fraction.",
    )
    parser.add_argument(
        "--val", type=float, default=0.15,
        help="Validation set fraction.",
    )
    parser.add_argument(
        "--test", type=float, default=0.15,
        help="Test set fraction.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling and splitting.",
    )
    args = parser.parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Multi-Corpus Dataset Pipeline")
    logger.info("=" * 60)
    logger.info(f"  data-dir  : {args.data_dir}")
    logger.info(f"  out-dir   : {args.out_dir}")
    logger.info(f"  split     : {args.train}/{args.val}/{args.test}")
    logger.info("")

    t_start = time.time()
    df = build_unified_corpus(
        args.data_dir,
        include_spaphish=args.include_spaphish,
        random_state=args.seed,
    )

    if df.empty:
        logger.error("No data loaded — check paths and file names.")
        sys.exit(1)

    # ── 2. Per-dataset statistics ─────────────────────────────────────────────
    logger.info("\n  Per-dataset breakdown:")
    logger.info(f"  {'Dataset':<15} {'Total':>8}  {'Phishing':>9}  {'Legit':>7}  {'%Phish':>7}")
    logger.info("  " + "-" * 52)
    for src, grp in df.groupby("source"):
        n = len(grp)
        ph = int((grp["label"] == 1).sum())
        lg = n - ph
        pct = ph / n * 100 if n else 0
        logger.info(f"  {src:<15} {n:>8,}  {ph:>9,}  {lg:>7,}  {pct:>6.1f}%")
    n_total = len(df)
    n_ph = int((df["label"] == 1).sum())
    n_lg = n_total - n_ph
    logger.info("  " + "-" * 52)
    logger.info(f"  {'TOTAL':<15} {n_total:>8,}  {n_ph:>9,}  {n_lg:>7,}  {n_ph/n_total*100:>6.1f}%")
    logger.info(f"\n  Target (paper): ~82,500  |  Actual: {n_total:,}")

    # ── 3. Split ──────────────────────────────────────────────────────────────
    logger.info("\n  Splitting dataset...")
    df_train, df_val, df_test = split_dataset(
        df, train=args.train, val=args.val, test=args.test, random_state=args.seed
    )

    logger.info(
        f"  Train : {len(df_train):,}  "
        f"(phishing={int((df_train['label']==1).sum()):,}, "
        f"legit={int((df_train['label']==0).sum()):,})"
    )
    logger.info(
        f"  Val   : {len(df_val):,}  "
        f"(phishing={int((df_val['label']==1).sum()):,}, "
        f"legit={int((df_val['label']==0).sum()):,})"
    )
    logger.info(
        f"  Test  : {len(df_test):,}  "
        f"(phishing={int((df_test['label']==1).sum()):,}, "
        f"legit={int((df_test['label']==0).sum()):,})"
    )

    # ── 4. Save ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_cols = ["subject", "body", "label", "source"]
    df_train[save_cols].to_csv(out_dir / "train.csv", index=False)
    df_val[save_cols].to_csv(out_dir / "val.csv", index=False)
    df_test[save_cols].to_csv(out_dir / "test.csv", index=False)

    logger.info(f"\n  Saved to {out_dir}/")
    logger.info(f"    train.csv  → {len(df_train):,} rows")
    logger.info(f"    val.csv    → {len(df_val):,} rows")
    logger.info(f"    test.csv   → {len(df_test):,} rows")

    elapsed = time.time() - t_start
    logger.info(f"\n  Total time: {elapsed:.1f}s")
    logger.info("=" * 60)

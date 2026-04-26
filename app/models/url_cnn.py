"""
1D-CNN URL Analyzer for Phishing Detection.

Implements the character-level convolutional neural network described in:
  Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.

Architecture:
  Input  : URL string → character-level token IDs (padded to MAX_URL_LEN=200)
  Embed  : vocab_size → embed_dim=32
  Conv   : Three parallel Conv1d branches with kernel sizes 3, 5, 7
             each → 64 filters → ReLU → AdaptiveMaxPool1d(1)
  Concat : 3 × 64 = 192-d feature vector
  FC     : Linear(192→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
  Output : phishing probability ∈ [0, 1]

Usage — inference:
    from app.models.url_cnn import load_url_cnn, score_url
    model = load_url_cnn("models/url_cnn.pt")
    prob  = score_url(model, "http://paypa1-secure.weebly.com/login")

Usage — training:
    from app.models.url_cnn import train_url_cnn, save_url_cnn
    model = train_url_cnn(url_list, labels, epochs=10)
    save_url_cnn(model, "models/url_cnn.pt")

CLI training:
    python -m app.models.url_cnn --train --data data/processed/train.csv \\
                                  --save models/url_cnn.pt --epochs 10
"""

from __future__ import annotations

import csv
import logging
import os
import re
import string
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character vocabulary
# ---------------------------------------------------------------------------
# Printable ASCII (95 characters: space + 94 visible) + padding token (index 0)
_PRINTABLE = string.printable          # 100 chars (incl. whitespace variants)
# Build a clean, consistent vocab: pad=0, then each printable char
_VOCAB_CHARS: str = _PRINTABLE
VOCAB: dict[str, int] = {ch: i + 1 for i, ch in enumerate(_VOCAB_CHARS)}
VOCAB_SIZE: int = len(VOCAB) + 1      # +1 for pad token (index 0)
PAD_IDX: int = 0
MAX_URL_LEN: int = 200                # characters; URLs longer are truncated

# ---------------------------------------------------------------------------
# URL extraction helper (reused from url_analyzer patterns)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+',
    re.IGNORECASE,
)


def extract_urls_from_text(text: str) -> List[str]:
    """Return all URLs found in plain text."""
    urls = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?)")
        if not url.startswith("http"):
            url = "http://" + url
        urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_url(url: str) -> torch.Tensor:
    """
    Convert a URL string to a fixed-length character-level token tensor.

    Each character is mapped to its vocabulary index (1-based).  Unknown
    characters map to PAD_IDX.  The sequence is truncated to MAX_URL_LEN
    and zero-padded on the right when shorter.

    Args:
        url: Raw URL string.

    Returns:
        LongTensor of shape (MAX_URL_LEN,) with dtype torch.long.
    """
    chars = url[:MAX_URL_LEN]
    ids = [VOCAB.get(ch, PAD_IDX) for ch in chars]
    # Pad to MAX_URL_LEN
    ids += [PAD_IDX] * (MAX_URL_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def tokenize_batch(urls: List[str]) -> torch.Tensor:
    """
    Tokenise a list of URLs into a batch tensor.

    Args:
        urls: List of URL strings.

    Returns:
        LongTensor of shape (N, MAX_URL_LEN).
    """
    return torch.stack([tokenize_url(u) for u in urls])


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class URLAnalyzerCNN(nn.Module):
    """
    Character-level 1D-CNN for phishing URL detection.

    Three parallel convolutional branches with kernel sizes 3, 5, and 7
    capture different n-gram patterns in the character sequence.  Their
    pooled outputs are concatenated and passed through a two-layer
    classifier head.

    Architecture summary:
        Embedding(VOCAB_SIZE, embed_dim=32, padding_idx=0)
        ├─ Conv1d(32, 64, 3) → ReLU → AdaptiveMaxPool1d(1)
        ├─ Conv1d(32, 64, 5) → ReLU → AdaptiveMaxPool1d(1)
        └─ Conv1d(32, 64, 7) → ReLU → AdaptiveMaxPool1d(1)
        Concat → Linear(192, 64) → ReLU → Dropout(0.3) → Linear(64, 1) → Sigmoid
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        num_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        fc_hidden: int = 64,
        dropout: float = 0.3,
        max_url_len: int = MAX_URL_LEN,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.max_url_len = max_url_len

        # Character embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX,
        )

        # Parallel convolutional branches (one per kernel size)
        # Input to Conv1d: (batch, channels=embed_dim, seq_len)
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=0,
                ),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(output_size=1),  # → (batch, num_filters, 1)
            )
            for k in kernel_sizes
        ])

        # Classifier head
        concat_dim = num_filters * len(kernel_sizes)   # 64 * 3 = 192
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LongTensor of shape (batch, MAX_URL_LEN) — token indices.

        Returns:
            FloatTensor of shape (batch,) — phishing probabilities in [0, 1].
        """
        # (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        # Conv1d expects (batch, channels, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # Apply each branch and collect (batch, num_filters, 1)
        branch_outputs = [branch(embedded) for branch in self.conv_branches]

        # Squeeze pool dimension → (batch, num_filters) each, then concat
        concatenated = torch.cat(
            [b.squeeze(2) for b in branch_outputs], dim=1
        )  # (batch, 192)

        # Classifier head → (batch, 1) → squeeze → (batch,)
        out = self.classifier(concatenated).squeeze(1)
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_url_cnn(
    url_list: List[str],
    labels: List[int],
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_split: float = 0.1,
    device: Optional[str] = None,
    verbose: bool = True,
) -> URLAnalyzerCNN:
    """
    Train a URLAnalyzerCNN on a list of URLs with binary labels.

    Args:
        url_list : List of URL strings.
        labels   : Corresponding binary labels (1=phishing, 0=legitimate).
        epochs   : Number of training epochs.
        lr       : Adam learning rate.
        batch_size: Mini-batch size.
        val_split: Fraction of data held out for validation.
        device   : "cuda", "cpu", or None (auto-detect).
        verbose  : Print per-epoch metrics.

    Returns:
        Trained URLAnalyzerCNN model (on CPU, ready for inference).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Training URL-CNN on {device} | {len(url_list)} URLs | epochs={epochs}")

    # Tokenise
    t0 = time.time()
    X = tokenize_batch(url_list)                          # (N, 200)
    y = torch.tensor(labels, dtype=torch.float32)         # (N,)
    logger.info(f"  Tokenised {len(url_list)} URLs in {time.time()-t0:.1f}s")

    # Train / validation split
    n_val = max(1, int(len(X) * val_split))
    n_train = len(X) - n_val
    perm = torch.randperm(len(X))
    idx_train, idx_val = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(X[idx_train], y[idx_train])
    val_ds   = TensorDataset(X[idx_val],   y[idx_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = URLAnalyzerCNN().to(dev)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_correct += ((preds >= 0.5) == yb.bool()).sum().item()
            train_total += len(xb)

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((preds >= 0.5) == yb.bool()).sum().item()
                val_total += len(xb)

        avg_train_loss = train_loss / train_total
        avg_val_loss   = val_loss   / val_total
        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train loss={avg_train_loss:.4f} acc={train_acc:.3f} | "
                f"val loss={avg_val_loss:.4f} acc={val_acc:.3f}"
            )

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  Best val loss: {best_val_loss:.4f}")

    model.to("cpu").eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def score_url(model: URLAnalyzerCNN, url: str) -> float:
    """
    Return the phishing probability for a single URL.

    Args:
        model: Trained URLAnalyzerCNN (must be in eval mode).
        url  : URL string.

    Returns:
        Float in [0, 1] — higher means more likely phishing.
    """
    model.eval()
    with torch.no_grad():
        tokens = tokenize_url(url).unsqueeze(0)   # (1, 200)
        prob = model(tokens).item()
    return float(prob)


def score_urls(model: URLAnalyzerCNN, urls: List[str]) -> List[float]:
    """
    Score a list of URLs in a single batched forward pass.

    Args:
        model: Trained URLAnalyzerCNN.
        urls : List of URL strings.

    Returns:
        List of floats in [0, 1].
    """
    if not urls:
        return []
    model.eval()
    with torch.no_grad():
        batch = tokenize_batch(urls)              # (N, 200)
        probs = model(batch).tolist()
    return [float(p) for p in probs]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_url_cnn(model: URLAnalyzerCNN, path: str | Path) -> None:
    """
    Save a URLAnalyzerCNN to disk.

    Saves both the model state dict and the constructor hyperparameters so
    the architecture can be reconstructed without importing the class.

    Args:
        model: Trained URLAnalyzerCNN.
        path : Destination file (typically ``models/url_cnn.pt``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size":   model.vocab_size,
            "embed_dim":    model.embed_dim,
            "num_filters":  model.num_filters,
            "kernel_sizes": model.kernel_sizes,
            "max_url_len":  model.max_url_len,
        },
    }
    torch.save(checkpoint, path)
    size_kb = path.stat().st_size / 1024
    logger.info(f"URL-CNN saved to {path}  ({size_kb:.1f} KB)")


def load_url_cnn(path: str | Path) -> URLAnalyzerCNN:
    """
    Load a URLAnalyzerCNN from disk.

    Args:
        path: Path to a checkpoint saved by :func:`save_url_cnn`.

    Returns:
        URLAnalyzerCNN in eval mode.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"URL-CNN checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = URLAnalyzerCNN(
        vocab_size=config.get("vocab_size", VOCAB_SIZE),
        embed_dim=config.get("embed_dim", 32),
        num_filters=config.get("num_filters", 64),
        kernel_sizes=tuple(config.get("kernel_sizes", (3, 5, 7))),
        max_url_len=config.get("max_url_len", MAX_URL_LEN),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info(f"URL-CNN loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# Integration helper: analyse URLs from email content
# ---------------------------------------------------------------------------

def analyze_urls_with_cnn(
    model: URLAnalyzerCNN,
    text: str,
    html_content: str = "",
    rule_results: Optional[dict] = None,
) -> dict:
    """
    CNN-based URL analysis compatible with the existing structural results format.

    Extracts all URLs from *text* and *html_content*, scores each with the
    CNN, and returns a dict that mirrors the output of
    ``url_analyzer.analyze_urls()`` so that downstream code is unaffected.

    If *rule_results* is provided it is merged in: CNN scores take precedence
    for ``overall_score``, but individual URL feature flags (is_ip_based,
    uses_shortener, etc.) from the rule-based analyser are preserved.

    Args:
        model        : Trained URLAnalyzerCNN.
        text         : Plain-text email body.
        html_content : HTML email body (optional).
        rule_results : Output of ``url_analyzer.analyze_urls()`` (optional).

    Returns:
        Dict with keys: url_count, overall_score, urls, details.
    """
    # Extract URLs (reuse regex from this module)
    found_urls: list[str] = extract_urls_from_text(text)
    if html_content:
        from urllib.parse import urlparse
        import re as _re
        href_re = _re.compile(r'href\s*=\s*["\']([^"\']+)["\']', _re.IGNORECASE)
        for m in href_re.finditer(html_content):
            href = m.group(1)
            if href.startswith(("http://", "https://", "www.")):
                if not href.startswith("http"):
                    href = "http://" + href
                found_urls.append(href)
        found_urls = list(dict.fromkeys(found_urls))  # deduplicate, keep order

    if not found_urls:
        return {
            "url_count": 0,
            "overall_score": 0.0,
            "urls": [],
            "details": ["No URLs found in email content"],
            "cnn_scored": True,
        }

    # CNN scoring (batched)
    cnn_scores = score_urls(model, found_urls)

    # Merge with rule-based per-URL feature dicts if available
    rule_url_map: dict[str, dict] = {}
    if rule_results and "urls" in rule_results:
        for u in rule_results["urls"]:
            rule_url_map[u.get("url", "")] = u

    url_analyses: list[dict] = []
    for url, cnn_prob in zip(found_urls, cnn_scores):
        entry = rule_url_map.get(url, {"url": url})
        entry["cnn_score"] = round(cnn_prob, 4)
        # Override risk_score with CNN-derived score (blend if rule data exists)
        if "risk_score" in entry:
            # Blend: 60% CNN + 40% rule-based
            entry["risk_score"] = round(0.6 * cnn_prob + 0.4 * entry["risk_score"], 4)
        else:
            entry["risk_score"] = round(cnn_prob, 4)
        url_analyses.append(entry)

    max_risk = max(a["risk_score"] for a in url_analyses)
    avg_risk = sum(a["risk_score"] for a in url_analyses) / len(url_analyses)
    overall  = round(0.7 * max_risk + 0.3 * avg_risk, 4)

    # Build details list
    details: list[str] = []
    for a in url_analyses:
        flags = []
        cnn_prob = a.get("cnn_score", 0.0)
        if cnn_prob >= 0.7:
            flags.append(f"CNN score {cnn_prob:.2f} (high risk)")
        elif cnn_prob >= 0.5:
            flags.append(f"CNN score {cnn_prob:.2f} (medium risk)")
        # Preserve rule-based flags where available
        if a.get("is_ip_based"):
            flags.append("IP-based URL")
        if a.get("uses_shortener"):
            flags.append("URL shortener")
        if a.get("suspicious_tld"):
            flags.append("Suspicious TLD")
        if a.get("typosquatting_score", 0) > 0.5:
            flags.append(f"Possible typosquatting of '{a.get('typosquatting_target')}'")
        if a.get("uses_free_hosting"):
            flags.append(f"Free hosting ({a.get('free_hosting_platform')})")
        if flags:
            details.append(f"{a['url'][:80]}: {'; '.join(flags)}")

    return {
        "url_count": len(found_urls),
        "overall_score": overall,
        "urls": url_analyses,
        "details": details if details else ["No suspicious URL indicators found"],
        "cnn_scored": True,
    }


# ---------------------------------------------------------------------------
# CLI — training entry point
# ---------------------------------------------------------------------------

def _load_urls_from_csv(csv_path: str, max_rows: int = 200_000) -> Tuple[List[str], List[int]]:
    """
    Load URLs and labels from train.csv produced by dataset_loader.

    Extracts all URLs found in the body column and assigns the row label.
    Rows with no URLs are skipped.

    Args:
        csv_path: Path to a CSV with columns [subject, body, label, ...].
        max_rows: Maximum number of rows to process.

    Returns:
        (url_list, label_list) — one entry per URL found.
    """
    url_list: list[str] = []
    label_list: list[int] = []
    skipped = 0

    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                lbl = int(float((row.get("label") or "0").strip()))
                if lbl not in (0, 1):
                    continue
            except (ValueError, TypeError):
                continue

            body = (row.get("body") or "").strip()
            urls = extract_urls_from_text(body)
            if not urls:
                skipped += 1
                continue

            for url in urls:
                url_list.append(url)
                label_list.append(lbl)

    logger.info(
        f"  Loaded {len(url_list)} URLs from {csv_path} "
        f"({skipped} rows had no URLs)"
    )
    return url_list, label_list


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train or evaluate the URL 1D-CNN phishing detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- train sub-command --
    train_p = subparsers.add_parser("--train", help="Train a new URL-CNN model.")
    train_p.add_argument("--data",   required=True, help="Path to train.csv")
    train_p.add_argument("--save",   default="models/url_cnn.pt", help="Output model path")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--lr",     type=float, default=1e-3)
    train_p.add_argument("--batch",  type=int, default=256)
    train_p.add_argument("--max-rows", type=int, default=200_000)

    # -- score sub-command --
    score_p = subparsers.add_parser("--score", help="Score a single URL.")
    score_p.add_argument("--model", default="models/url_cnn.pt", help="Path to saved model")
    score_p.add_argument("--url",   required=True, help="URL to score")

    # Handle bare --train / --score flags (no sub-command verb)
    # Also support: python -m app.models.url_cnn --train --data X --save Y
    args, unknown = parser.parse_known_args()

    # Rebuild with flat flags if sub-command detection failed
    if args.command is None:
        flat = parser.parse_args()
    else:
        flat = args

    # Re-parse properly with flat flag style
    flat_parser = argparse.ArgumentParser(
        description="URL 1D-CNN trainer / scorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    flat_parser.add_argument("--train",    action="store_true")
    flat_parser.add_argument("--score",    action="store_true")
    flat_parser.add_argument("--data",     default="data/processed/train.csv")
    flat_parser.add_argument("--save",     default="models/url_cnn.pt")
    flat_parser.add_argument("--model",    default="models/url_cnn.pt")
    flat_parser.add_argument("--url",      default=None)
    flat_parser.add_argument("--epochs",   type=int,   default=10)
    flat_parser.add_argument("--lr",       type=float, default=1e-3)
    flat_parser.add_argument("--batch",    type=int,   default=256)
    flat_parser.add_argument("--max-rows", type=int,   default=200_000)
    cfg = flat_parser.parse_args()

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    if cfg.train:
        logger.info("=" * 60)
        logger.info("  URL 1D-CNN — Training")
        logger.info("=" * 60)
        logger.info(f"  data   : {cfg.data}")
        logger.info(f"  save   : {cfg.save}")
        logger.info(f"  epochs : {cfg.epochs}")
        logger.info(f"  lr     : {cfg.lr}")
        logger.info(f"  batch  : {cfg.batch}")
        logger.info(f"  vocab  : {VOCAB_SIZE} chars  |  max_len={MAX_URL_LEN}")
        logger.info("")

        url_list, label_list = _load_urls_from_csv(cfg.data, max_rows=cfg.max_rows)
        if not url_list:
            logger.error("No URLs extracted from dataset. Check --data path.")
            sys.exit(1)

        n_ph = sum(label_list)
        n_lg = len(label_list) - n_ph
        logger.info(f"  URLs: {len(url_list):,}  (phishing={n_ph:,}, legit={n_lg:,})")

        model = train_url_cnn(
            url_list,
            label_list,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch,
        )
        save_url_cnn(model, cfg.save)
        logger.info("")
        logger.info("  Training complete.")

    # ── SCORE ─────────────────────────────────────────────────────────────────
    elif cfg.score:
        if cfg.url is None:
            logger.error("--score requires --url <url>")
            sys.exit(1)
        model = load_url_cnn(cfg.model)
        prob = score_url(model, cfg.url)
        verdict = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
        print(f"\nURL    : {cfg.url}")
        print(f"Score  : {prob:.4f}")
        print(f"Verdict: {verdict}\n")

    else:
        flat_parser.print_help()

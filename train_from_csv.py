"""
Batch training script: Load Spanish phishing dataset CSV and train the Random Forest classifier.

Usage:
    python train_from_csv.py "path/to/Spaphish dataset - DiB.csv"
"""

import os
import sys
import csv
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.semantic_engine import analyze_semantics
from app.models.structural_engine import analyze_structure
from app.models.classifier import add_training_sample, train_model, get_training_stats, _rule_scores_to_features
from app.parsers.text_parser import parse_text_input
from app.database import save_training_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trainer")


def load_csv(csv_path: str):
    """Load the Spaphish dataset CSV."""
    samples = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            subject = (row.get("subject") or "").strip()
            body = (row.get("body") or "").strip()
            label_str = (row.get("Label") or "").strip()
            if label_str not in ("0", "1"):
                continue
            if not subject and not body:
                continue
            samples.append({
                "subject": subject,
                "body": body,
                "label": int(label_str),
            })
    return samples


def main():
    if len(sys.argv) < 2:
        csv_path = os.path.join(
            os.path.expanduser("~"), "Downloads", "Spaphish dataset - DiB.csv"
        )
    else:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {csv_path}")
    samples = load_csv(csv_path)
    phishing_count = sum(1 for s in samples if s["label"] == 1)
    legit_count = sum(1 for s in samples if s["label"] == 0)
    logger.info(f"Loaded {len(samples)} samples: {phishing_count} phishing, {legit_count} legitimate")

    # Process each sample through the analysis pipeline
    logger.info("Processing samples through dual-engine pipeline...")
    start = time.time()
    processed = 0
    errors = 0

    for i, sample in enumerate(samples):
        try:
            parsed = parse_text_input(sample["subject"], sample["body"])
            semantic_results = analyze_semantics(sample["subject"], sample["body"])
            structural_results = analyze_structure(parsed)
            add_training_sample(semantic_results, structural_results, sample["label"])

            # Also persist to database
            embedding = semantic_results.get("embedding")
            structural_features = structural_results.get("structural_features", [])
            if embedding is not None:
                features = embedding + structural_features
            else:
                features = _rule_scores_to_features(semantic_results.get("rule_scores", {})) + structural_features
            save_training_sample(sample["subject"], sample["body"], sample["label"], features)

            processed += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Error processing sample {i}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            logger.info(f"  Processed {i+1}/{len(samples)} ({rate:.0f} samples/sec, ETA: {eta:.0f}s)")

    elapsed = time.time() - start
    logger.info(f"Processing complete: {processed} processed, {errors} errors, {elapsed:.1f}s total")

    # Train the model
    stats = get_training_stats()
    logger.info(f"Training data: {stats}")

    logger.info("Training Random Forest classifier...")
    result = train_model(min_samples=6)

    if result["success"]:
        logger.info(f"Training successful!")
        logger.info(f"  Samples: {result['sample_count']}")
        logger.info(f"  Labels: {result['label_distribution']}")
        logger.info(f"  Features: {result['feature_count']}")
        if "cross_validation" in result:
            cv = result["cross_validation"]
            logger.info(f"  Cross-validation F1: {cv['f1_mean']:.4f} (+/- {cv['f1_std']:.4f})")
        logger.info(f"  Model saved to: {result['model_path']}")
    else:
        logger.error(f"Training failed: {result.get('error')}")

    return result


if __name__ == "__main__":
    main()

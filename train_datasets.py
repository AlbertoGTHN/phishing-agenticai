"""
Multi-Dataset Training Script — Agentic AI Phishing Detector.

Loads 7 English datasets + 1 Spanish dataset, stratified-samples up to
MAX_PER_DATASET rows each, extracts rule-based + structural features
(DistilBERT intentionally skipped for speed — RF uses 70-d vector),
trains a Random Forest, and saves the model.

Usage:  python train_datasets.py
Output: models/rf_classifier.pkl
"""

import os, sys, csv, time, random, pickle, logging
csv.field_size_limit(10 * 1024 * 1024)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Silence noisy sub-loggers before importing app modules
for name in ("app.parsers.text_parser", "app.parsers.eml_parser",
             "app.utils.header_analyzer", "app.models.semantic_engine",
             "app.models.structural_engine", "app.utils.url_analyzer",
             "app.utils.html_analyzer"):
    logging.getLogger(name).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("trainer")

# ── Config ────────────────────────────────────────────────────────────────────
DATASETS_DIR   = r"C:\Users\alber\Downloads\Phishing emails"
SPANISH_CSV    = r"C:\Users\alber\Downloads\Spaphish dataset - DiB.csv"
MODEL_OUTPUT   = os.path.join(BASE_DIR, "models", "rf_classifier.pkl")
MAX_PER_DATASET = 3000
RANDOM_SEED    = 42

DATASETS = [
    # Well-structured balanced datasets with subject + body columns
    {"path": os.path.join(DATASETS_DIR, "SpamAssasin.csv"), "name": "SpamAssasin", "subj": "subject", "body": "body", "label": "label", "sep": ","},
    {"path": os.path.join(DATASETS_DIR, "CEAS_08.csv"),     "name": "CEAS_08",     "subj": "subject", "body": "body", "label": "label", "sep": ","},
    {"path": os.path.join(DATASETS_DIR, "Ling.csv"),        "name": "Ling",        "subj": "subject", "body": "body", "label": "label", "sep": ","},
    # Spanish (best performing — all samples used)
    {"path": SPANISH_CSV,                                    "name": "SpaPhish_ES", "subj": "subject", "body": "body", "label": "Label", "sep": ";"},
    # Enron capped — early 2000s patterns differ but still useful
    {"path": os.path.join(DATASETS_DIR, "Enron.csv"),       "name": "Enron",       "subj": "subject", "body": "body", "label": "label", "sep": ","},
]

PHISHING_ONLY_CAP = 500  # unused in this config but kept for reference

# ── Patch semantic engine to skip DistilBERT ──────────────────────────────────
import app.models.semantic_engine as _sem
_sem._model_load_attempted = True   # pretend load was already tried → skipped
_sem._transformer_model    = None
_sem._transformer_tokenizer = None
logger.info("DistilBERT disabled for batch training (rule-based features only)")

from app.parsers.text_parser       import parse_text_input
from app.models.semantic_engine    import compute_rule_based_scores
from app.models.structural_engine  import analyze_structure
from app.models.classifier         import _rule_scores_to_features

# ── Dataset loading ───────────────────────────────────────────────────────────
def load_dataset(cfg, max_n, seed):
    path = cfg["path"]
    if not os.path.exists(path):
        logger.warning(f"  Not found: {path}")
        return []
    phishing, legit = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=cfg["sep"])
        for row in reader:
            try:
                lbl = int(float((row.get(cfg["label"]) or "").strip()))
                if lbl not in (0, 1): continue
                subj = (row.get(cfg["subj"]) or "").strip()[:300] if cfg["subj"] else ""
                body = (row.get(cfg["body"]) or "").strip()[:3000]
                (phishing if lbl == 1 else legit).append((subj, body, lbl))
            except (ValueError, KeyError):
                continue
    rng = random.Random(seed)
    if legit:
        per = max_n // 2
        samples = rng.sample(phishing, min(len(phishing), per)) + \
                  rng.sample(legit,    min(len(legit),    per))
    else:
        samples = rng.sample(phishing, min(len(phishing), max_n))
    rng.shuffle(samples)
    logger.info(f"  {cfg['name']}: {len(samples):,} samples "
                f"(phishing={sum(1 for s in samples if s[2]==1)}, "
                f"legit={sum(1 for s in samples if s[2]==0)})")
    return samples

# ── Feature extraction (rule-based + structural, no transformer) ──────────────
def extract_features(subject, body):
    try:
        parsed   = parse_text_input(subject, body)
        rule_sc  = compute_rule_based_scores(f"{subject} {body}".strip())
        struct   = analyze_structure(parsed)
        sem_feat = _rule_scores_to_features(rule_sc)
        str_feat = struct.get("structural_features", [])
        return sem_feat + str_feat
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    logger.info("=" * 60)
    logger.info("  Multi-Dataset Trainer — Agentic AI Phishing Detector")
    logger.info("=" * 60)

    # 1. Load all datasets
    all_samples = []
    for cfg in DATASETS:
        logger.info(f"Loading {cfg['name']}...")
        all_samples.extend(load_dataset(cfg, MAX_PER_DATASET, RANDOM_SEED))

    total = len(all_samples)
    n_phish = sum(1 for s in all_samples if s[2] == 1)
    n_legit = total - n_phish
    logger.info(f"\nTotal: {total:,} samples — phishing={n_phish:,}, legit={n_legit:,}")
    random.shuffle(all_samples)

    # 2. Extract features
    logger.info(f"\nExtracting features (rule-based + structural)...")
    t0 = time.time()
    X, y, failed = [], [], 0
    for i, (subj, body, lbl) in enumerate(all_samples):
        feats = extract_features(subj, body)
        if feats is None:
            failed += 1
            continue
        X.append(feats); y.append(lbl)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (total - i - 1) / rate
            logger.info(f"  {i+1:,}/{total:,} | {rate:.0f}/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"Done: {len(X):,} features extracted, {failed} failed ({elapsed:.1f}s, {len(X)/elapsed:.0f}/s)")

    # 3. Train
    import numpy as np
    from sklearn.ensemble          import RandomForestClassifier
    from sklearn.model_selection   import StratifiedKFold, cross_val_score
    from sklearn.metrics           import classification_report

    max_len  = max(len(v) for v in X)
    X_arr    = np.array([v + [0.0] * (max_len - len(v)) for v in X])
    y_arr    = np.array(y)
    logger.info(f"\nFeature dims: {max_len}  |  classes: phishing={int(y_arr.sum())}, legit={int((1-y_arr).sum())}")

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )

    logger.info("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = {}
    for metric in ("accuracy", "precision", "recall", "f1"):
        s = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring=metric, n_jobs=-1)
        scores[metric] = s
        logger.info(f"  CV {metric:10s}: {s.mean():.4f} ± {s.std():.4f}")

    logger.info("Fitting on full dataset...")
    clf.fit(X_arr, y_arr)
    logger.info("\nTraining-set report:\n" +
                classification_report(y_arr, clf.predict(X_arr),
                                      target_names=["Legitimate", "Phishing"]))

    # 4. Save
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(clf, f)
    size_mb = os.path.getsize(MODEL_OUTPUT) / 1024 / 1024

    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Samples   : {len(X):,}")
    logger.info(f"  Features  : {max_len} dims")
    logger.info(f"  CV F1     : {scores['f1'].mean():.4f} ± {scores['f1'].std():.4f}")
    logger.info(f"  CV Accuracy: {scores['accuracy'].mean():.4f} ± {scores['accuracy'].std():.4f}")
    logger.info(f"  Model     : {MODEL_OUTPUT} ({size_mb:.1f} MB)")
    logger.info("  Restart the server to load the updated model.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

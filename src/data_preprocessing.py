"""
data_preprocessing.py -- Phase 2: NLP Preprocessing and Data Splitting
=======================================================================
Objective:
  Build a clean, reproducible text preprocessing pipeline and create
  stratified train/validation splits from train.csv.

Key rules:
  - Preserve negation words (not, never, no, can't, didn't, don't, ...)
  - Do NOT aggressively remove stopwords
  - Lowercase and normalise whitespace
  - Handle punctuation carefully
  - Test data is NEVER touched here

Run:
    cd banking-customer-support-ai
    python src/data_preprocessing.py
"""

import re
import sys
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger, ensure_dir, save_json, set_seed, project_root

logger = get_logger("data_preprocessing")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "train_path":     ROOT / "data" / "train.csv",
    "test_path":      ROOT / "data" / "test.csv",
    "split_train_out": ROOT / "data" / "train_split.csv",
    "split_val_out":   ROOT / "data" / "val_split.csv",
    "artifacts_dir":   ROOT / "artifacts",
    "val_size":        0.20,      # 80/20 stratified split
    "random_seed":     42,
}

# ─────────────────────────────────────────────────────────────
# Important negation / banking-critical contractions
# We expand these BEFORE any other normalisation so intent is
# preserved (e.g. "can't" -> "cannot", "didn't" -> "did not").
# ─────────────────────────────────────────────────────────────

CONTRACTIONS = {
    "can't":     "cannot",
    "cannot":    "cannot",
    "won't":     "will not",
    "n't":       " not",          # catches didn't, don't, wouldn't, shouldn't, etc.
    "i'm":       "i am",
    "i've":      "i have",
    "i'll":      "i will",
    "i'd":       "i would",
    "it's":      "it is",
    "that's":    "that is",
    "there's":   "there is",
    "they're":   "they are",
    "they've":   "they have",
    "they'd":    "they would",
    "they'll":   "they will",
    "you're":    "you are",
    "you've":    "you have",
    "you'll":    "you will",
    "you'd":     "you would",
    "we're":     "we are",
    "we've":     "we have",
    "we'd":      "we would",
    "we'll":     "we will",
    "he's":      "he is",
    "she's":     "she is",
    "let's":     "let us",
    "what's":    "what is",
    "where's":   "where is",
    "when's":    "when is",
    "who's":     "who is",
    "how's":     "how is",
    "that'll":   "that will",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions while preserving negation words."""
    text = text.lower()
    # Apply each contraction map (longest first to avoid partial matches)
    for contraction, expanded in sorted(CONTRACTIONS.items(), key=lambda x: -len(x[0])):
        text = re.sub(re.escape(contraction), expanded, text)
    return text


# ─────────────────────────────────────────────────────────────
# Core preprocessing function
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply a banking-aware NLP preprocessing pipeline to a raw text string.

    Steps:
    1.  Lowercase
    2.  Expand contractions (preserves negation)
    3.  Remove URLs
    4.  Remove email addresses
    5.  Replace currency symbols with their word form
    6.  Keep only alphabetic chars, hyphens, and single spaces
        (hyphens preserved for compound banking terms like 'top-up')
    7.  Normalise whitespace

    What is NOT done:
    - Stopword removal (would destroy "not", "never", "no")
    - Stemming / lemmatisation (can destroy intent tokens)
    - Aggressive punctuation stripping beyond what is listed
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase + contraction expansion
    text = expand_contractions(text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Currency symbol -> word
    text = re.sub(r"£",  " pound ",  text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"€",  " euro ",   text)

    # 5. Remove digits (standalone numbers add noise; keep words)
    text = re.sub(r"\b\d+\b", " ", text)

    # 6. Keep only letters, spaces, hyphens (for compound tokens)
    text = re.sub(r"[^a-z\s\-]", " ", text)

    # 7. Normalise hyphens used as dashes to spaces
    text = re.sub(r"\s-\s", " ", text)

    # 8. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─────────────────────────────────────────────────────────────
# DataFrame-level preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Apply clean_text() to every row and return a new DataFrame."""
    logger.info(f"Preprocessing DataFrame: {name} ({len(df)} rows)")
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    # Sanity check: flag any empty cleaned texts
    empty_count = (df["clean_text"].str.strip() == "").sum()
    if empty_count:
        logger.warning(f"  {empty_count} rows resulted in empty clean_text — check source data")

    return df


# ─────────────────────────────────────────────────────────────
# Label encoding
# ─────────────────────────────────────────────────────────────

def create_label_mapping(categories: pd.Series) -> tuple[dict, dict]:
    """
    Build label <-> index mappings from sorted unique categories.

    Returns:
        label2idx  (dict): category string -> integer index
        idx2label  (dict): integer index -> category string
    """
    unique_labels = sorted(categories.unique())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    logger.info(f"Label mapping created: {len(label2idx)} classes")
    return label2idx, idx2label


# ─────────────────────────────────────────────────────────────
# Train / Validation split
# ─────────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a stratified train/validation split.

    Stratification is essential because we have 77 imbalanced classes --
    a random split would leave some minority classes under-represented
    or entirely absent from the validation set.
    """
    logger.info(f"Creating stratified split: {1-val_size:.0%} train / {val_size:.0%} val")

    df_train, df_val = train_test_split(
        df,
        test_size=val_size,
        stratify=df["category"],
        random_state=random_seed,
        shuffle=True,
    )

    logger.info(f"  Training samples   : {len(df_train):,}")
    logger.info(f"  Validation samples : {len(df_val):,}")

    # Verify all categories are present in both splits
    train_cats = set(df_train["category"].unique())
    val_cats   = set(df_val["category"].unique())
    missing_in_val = train_cats - val_cats
    if missing_in_val:
        logger.warning(f"  {len(missing_in_val)} categories missing from validation split!")
    else:
        logger.info("  All categories represented in both splits.")

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run():
    """Execute Phase 2: NLP Preprocessing and Data Splitting."""
    set_seed(CONFIG["random_seed"])
    ensure_dir(CONFIG["artifacts_dir"])
    ensure_dir(ROOT / "data")

    logger.info("=" * 58)
    logger.info("  PHASE 2 - NLP Preprocessing & Data Splitting")
    logger.info("=" * 58)

    # ── 1. Load raw training data ──
    logger.info(f"\nLoading training data from {CONFIG['train_path']}")
    df_train_raw = pd.read_csv(CONFIG["train_path"])
    logger.info(f"  Raw training shape: {df_train_raw.shape}")

    # ── 2. Preprocess training data ──
    df_train_clean = preprocess_dataframe(df_train_raw, name="Train")

    # ── 3. Build label mapping (from training data only) ──
    label2idx, idx2label = create_label_mapping(df_train_clean["category"])
    df_train_clean["label"] = df_train_clean["category"].map(label2idx)

    # ── 4. Verify a few examples ──
    logger.info("\nSample preprocessing comparison:")
    for _, row in df_train_clean.sample(5, random_state=42).iterrows():
        logger.info(f"  RAW  : {row['text'][:80]}")
        logger.info(f"  CLEAN: {row['clean_text'][:80]}")
        logger.info("  ---")

    # ── 5. Stratified split ──
    df_split_train, df_split_val = stratified_split(
        df_train_clean,
        val_size=CONFIG["val_size"],
        random_seed=CONFIG["random_seed"],
    )

    # ── 6. Save split datasets ──
    df_split_train.to_csv(CONFIG["split_train_out"], index=False)
    df_split_val.to_csv(CONFIG["split_val_out"],     index=False)
    logger.info(f"\nTrain split saved -> {CONFIG['split_train_out']}")
    logger.info(f"Val   split saved -> {CONFIG['split_val_out']}")

    # ── 7. Save label mapping ──
    mapping_path = CONFIG["artifacts_dir"] / "label_mapping.json"
    save_json({"label2idx": label2idx, "idx2label": idx2label}, mapping_path)
    logger.info(f"Label mapping saved -> {mapping_path}")

    # ── 8. Also preprocess test data for later use (DO NOT USE FOR TRAINING) ──
    logger.info("\nPreprocessing test data (for later FINAL evaluation only)...")
    df_test_raw   = pd.read_csv(CONFIG["test_path"])
    df_test_clean = preprocess_dataframe(df_test_raw, name="Test")
    df_test_clean["label"] = df_test_clean["category"].map(label2idx)
    test_clean_path = ROOT / "data" / "test_clean.csv"
    df_test_clean.to_csv(test_clean_path, index=False)
    logger.info(f"Clean test data saved -> {test_clean_path}")
    logger.info("NOTE: test_clean.csv must NOT be used until Phase 8 final evaluation.")

    logger.info("\n[OK] Phase 2 complete.")
    return label2idx, idx2label


if __name__ == "__main__":
    run()

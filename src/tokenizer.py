"""
tokenizer.py -- Phase 4: Custom Word-Level Tokenizer
=====================================================
Objective:
  Build a vocabulary and tokenizer entirely from the training data.
  No pre-trained tokenizers are used.

Features:
  - Word-level tokenization
  - Vocabulary built from training corpus only
  - Special tokens: <PAD> (0), <UNK> (1)
  - Numericalization (text -> int sequence)
  - Padding and truncation to a fixed max length
  - Serializable to JSON

Usage:
    tokenizer = BankingTokenizer(max_len=64, min_freq=2)
    tokenizer.build_vocab(train_texts)
    ids = tokenizer.encode("my card has not arrived")
    tokenizer.save("artifacts/vocab.json")
    tokenizer = BankingTokenizer.load("artifacts/vocab.json")
"""

import re
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Optional

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger, ensure_dir

logger = get_logger("tokenizer")

# ─────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1


class BankingTokenizer:
    """
    A simple whitespace-based word tokenizer tailored for banking text.

    The vocabulary is built exclusively from training data.
    Special tokens <PAD> and <UNK> are always at indices 0 and 1.
    """

    def __init__(self, max_len: int = 64, min_freq: int = 2):
        """
        Args:
            max_len  : Maximum sequence length (tokens). Longer sequences
                       are truncated; shorter ones are padded.
            min_freq : Minimum frequency for a token to be included in vocab.
                       Tokens below this threshold are treated as <UNK>.
        """
        self.max_len  = max_len
        self.min_freq = min_freq

        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.vocab_size: int = 0
        self._built: bool = False

    # ── Tokenisation ──────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """
        Convert a cleaned text string into a list of word tokens.

        Assumes text has already been lowercased and normalised by
        data_preprocessing.clean_text(). We just split on whitespace.
        """
        return str(text).strip().split()

    # ── Vocabulary construction ───────────────────────────────

    def build_vocab(self, texts: list[str]) -> None:
        """
        Build the vocabulary from an iterable of (already-cleaned) texts.

        Args:
            texts: List of preprocessed text strings (training corpus only).
        """
        logger.info(f"Building vocabulary from {len(texts):,} training texts...")
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._tokenise(text))

        total_tokens  = sum(counter.values())
        unique_tokens = len(counter)
        logger.info(f"  Total token occurrences : {total_tokens:,}")
        logger.info(f"  Unique tokens (raw)     : {unique_tokens:,}")

        # Filter by minimum frequency
        vocab_tokens = [tok for tok, freq in counter.most_common() if freq >= self.min_freq]
        logger.info(f"  Tokens after min_freq={self.min_freq}: {len(vocab_tokens):,}")

        # Build mappings; indices 0 and 1 are reserved for PAD and UNK
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        for idx, token in enumerate(vocab_tokens, start=2):
            self.word2idx[token] = idx

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self._built = True

        logger.info(f"  Final vocabulary size    : {self.vocab_size:,}")
        logger.info(f"  (PAD_IDX={PAD_IDX}, UNK_IDX={UNK_IDX})")

    # ── Encoding ─────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """
        Convert a text string to a padded/truncated list of integer token IDs.

        Args:
            text: A preprocessed text string.
        Returns:
            A list of integer token IDs of length self.max_len.
        """
        if not self._built:
            raise RuntimeError("Call build_vocab() before encode().")

        tokens = self._tokenise(text)
        # Map tokens to indices (unknown tokens -> UNK_IDX)
        ids = [self.word2idx.get(tok, UNK_IDX) for tok in tokens]
        # Truncate
        ids = ids[: self.max_len]
        # Pad to max_len
        ids += [PAD_IDX] * (self.max_len - len(ids))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Convert a list of integer token IDs back to a text string.

        Args:
            ids          : List of integer token IDs.
            skip_special : If True, skip PAD and UNK tokens.
        Returns:
            Decoded text string.
        """
        tokens = [self.idx2word.get(idx, UNK_TOKEN) for idx in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in (PAD_TOKEN, UNK_TOKEN)]
        return " ".join(tokens)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Encode a list of texts; returns a list of integer-ID lists."""
        return [self.encode(text) for text in texts]

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize tokenizer state to a JSON file."""
        path = Path(path)
        ensure_dir(path.parent)
        payload = {
            "max_len":    self.max_len,
            "min_freq":   self.min_freq,
            "vocab_size": self.vocab_size,
            "word2idx":   self.word2idx,
            # idx2word keys are ints -> must serialise as strings
            "idx2word":   {str(k): v for k, v in self.idx2word.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info(f"Tokenizer saved -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BankingTokenizer":
        """Load a previously serialised tokenizer."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        tok = cls(max_len=payload["max_len"], min_freq=payload["min_freq"])
        tok.word2idx   = payload["word2idx"]
        tok.idx2word   = {int(k): v for k, v in payload["idx2word"].items()}
        tok.vocab_size = payload["vocab_size"]
        tok._built     = True
        logger.info(f"Tokenizer loaded from {path} (vocab_size={tok.vocab_size})")
        return tok

    # ── Helpers ───────────────────────────────────────────────

    def __repr__(self) -> str:
        status = f"vocab_size={self.vocab_size}" if self._built else "not built"
        return f"BankingTokenizer(max_len={self.max_len}, min_freq={self.min_freq}, {status})"


def build_and_save_vocab(
    train_path: Path | str = ROOT / "data" / "train_split.csv",
    vocab_path: Path | str = ROOT / "artifacts" / "vocab.json",
    max_len: int = 64,
    min_freq: int = 2,
) -> BankingTokenizer:
    """Build vocabulary from training data and save to disk."""
    import pandas as pd

    train_path = Path(train_path)
    vocab_path = Path(vocab_path)

    if not train_path.exists():
        fallback_path = ROOT / "data" / "train.csv"
        if fallback_path.exists():
            logger.info(f"train_split.csv not found, falling back to {fallback_path}")
            train_path = fallback_path
        else:
            raise FileNotFoundError(f"Cannot find training data at {train_path} or {fallback_path}")

    logger.info("=" * 60)
    logger.info("  PHASE 4 - Custom Word-Level Tokenizer & Vocabulary")
    logger.info("=" * 60)
    logger.info(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    texts = df[text_col].dropna().astype(str).tolist()

    tokenizer = BankingTokenizer(max_len=max_len, min_freq=min_freq)
    tokenizer.build_vocab(texts)
    tokenizer.save(vocab_path)
    logger.info(f"[OK] Vocabulary saved to {vocab_path} (vocab_size={tokenizer.vocab_size:,})")
    return tokenizer


if __name__ == "__main__":
    build_and_save_vocab()


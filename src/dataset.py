"""
dataset.py -- Phase 4: PyTorch Dataset and DataLoader
======================================================
Objective:
  Create a custom PyTorch Dataset class that wraps the tokenized
  banking query data, and provide DataLoader factories for training
  and evaluation.

Key design decisions:
  - Extends torch.utils.data.Dataset
  - Produces (input_ids, label) pairs
  - input_ids are already padded/truncated to max_len by the tokenizer
  - Weighted sampling available for class-imbalance handling
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger
from tokenizer import BankingTokenizer

logger = get_logger("dataset")


# ─────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────

class BankingIntentDataset(Dataset):
    """
    PyTorch Dataset for banking intent classification.

    Each item is a (input_ids, label) pair where:
        input_ids: LongTensor of shape (max_len,)
        label    : LongTensor scalar
    """

    def __init__(
        self,
        texts:     list[str],
        labels:    list[int],
        tokenizer: BankingTokenizer,
    ):
        """
        Args:
            texts    : List of preprocessed text strings.
            labels   : List of integer class labels.
            tokenizer: Fitted BankingTokenizer instance.
        """
        assert len(texts) == len(labels), "texts and labels must have the same length"

        self.tokenizer = tokenizer
        # Encode all texts up front (faster than encoding one-by-one in __getitem__)
        self.input_ids = [
            torch.tensor(tokenizer.encode(text), dtype=torch.long)
            for text in texts
        ]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────

def make_data_loader(
    dataset:    BankingIntentDataset,
    batch_size: int  = 64,
    shuffle:    bool = False,
    use_weighted_sampler: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from a BankingIntentDataset.

    Args:
        dataset              : BankingIntentDataset instance.
        batch_size           : Batch size.
        shuffle              : Whether to shuffle the data.
        use_weighted_sampler : If True, uses WeightedRandomSampler to
                               address class imbalance. Cannot be used
                               together with shuffle=True.
        num_workers          : Number of worker processes for data loading.
                               Use 0 on Windows to avoid multiprocessing issues.
    Returns:
        torch.utils.data.DataLoader
    """
    sampler = None
    if use_weighted_sampler:
        # Compute sample weights inversely proportional to class frequency
        labels_np = np.array([label.item() for label in dataset.labels])
        class_counts = np.bincount(labels_np)
        # Weight per class: inverse frequency
        class_weights = 1.0 / (class_counts + 1e-8)
        # Weight per sample
        sample_weights = class_weights[labels_np]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # Cannot shuffle when sampler is provided
        logger.info(f"  WeightedRandomSampler activated (n_samples={len(dataset):,})")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ─────────────────────────────────────────────────────────────
# Convenience loader: build everything from CSVs
# ─────────────────────────────────────────────────────────────

def load_splits(
    train_csv:  Path | str,
    val_csv:    Path | str,
    tokenizer:  BankingTokenizer,
    batch_size: int  = 64,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Load train and val CSV files, create datasets and data loaders.

    Returns:
        train_loader : DataLoader for the training split.
        val_loader   : DataLoader for the validation split.
        num_classes  : Number of unique intent classes.
    """
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)

    logger.info(f"Loaded train split: {len(df_train):,} rows")
    logger.info(f"Loaded val split  : {len(df_val):,} rows")

    # Build tokenizer vocabulary if not already built
    if not tokenizer._built:
        tokenizer.build_vocab(df_train["clean_text"].astype(str).tolist())

    # Create datasets
    train_dataset = BankingIntentDataset(
        texts=df_train["clean_text"].astype(str).tolist(),
        labels=df_train["label"].astype(int).tolist(),
        tokenizer=tokenizer,
    )
    val_dataset = BankingIntentDataset(
        texts=df_val["clean_text"].astype(str).tolist(),
        labels=df_val["label"].astype(int).tolist(),
        tokenizer=tokenizer,
    )

    num_classes = df_train["label"].nunique()

    # Create data loaders
    train_loader = make_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not use_weighted_sampler,
        use_weighted_sampler=use_weighted_sampler,
    )
    val_loader = make_data_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        use_weighted_sampler=False,
    )

    logger.info(f"Training batches  : {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Num classes       : {num_classes}")

    return train_loader, val_loader, num_classes

"""
utils.py - Shared utility helpers for the Banking Customer Support AI project.

Provides:
 - Reproducibility seeding
 - Logging configuration
 - Directory creation helper
 - Timer context manager
 - JSON / CSV persistence helpers
"""

import os
import json
import time
import random
import logging
import contextlib
from pathlib import Path

import numpy as np


# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# -------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible results across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# -------------------------------------------------------------
# File-system helpers
# -------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------------------------------------------
# Persistence helpers
# -------------------------------------------------------------

def save_json(data: dict, filepath: str | Path, indent: int = 2) -> None:
    """Serialize *data* to a JSON file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


def load_json(filepath: str | Path) -> dict:
    """Load a JSON file and return the parsed object."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


# -------------------------------------------------------------
# Timing
# -------------------------------------------------------------

@contextlib.contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[timer] {label}: {elapsed:.2f}s")


# -------------------------------------------------------------
# Project-root resolution
# -------------------------------------------------------------

def project_root() -> Path:
    """Return the absolute path of the banking-customer-support-ai root directory."""
    # This file lives at  <root>/src/utils.py
    return Path(__file__).resolve().parent.parent

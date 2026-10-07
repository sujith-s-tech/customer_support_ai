"""
DEPRECATED: This script has been consolidated into train_deep_learning.py.
For DL training, please run:
    python src/train_deep_learning.py
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from train_deep_learning import run, MODEL_CONFIGS

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train custom DL architectures (forwarded to train_deep_learning.py)")
    parser.add_argument(
        "--models", nargs="+",
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific models to train (default: all 5)"
    )
    args = parser.parse_args()
    run(models_to_train=args.models)

"""
train_deep_learning.py -- Phase 6 & 7: DL Training, Comparison, and Selection
=============================================================================
Objective:
  Train all five neural architectures (SimpleRNN, LSTM, GRU, TextCNN, BiLSTMAttn)
  with architecture-specific hyperparameter tuning, record validation metrics,
  compare them, and programmatically select the best model based on validation Macro F1.

Key design points:
  - Model-specific hyperparameters: recurrent models use higher LR (1e-3) and
    more patience (15) to ensure full convergence; TextCNN/BiLSTMAttn use proven tuned settings.
  - Architectures include orthogonal weight init, bidirectional recurrent context,
    multi-head attention, LayerNorm, and multi-pool representations.
  - Early stopping monitors validation Macro F1 (not raw accuracy) to protect against class imbalance.
  - Checkpoints saved per model; winner automatically copied to 'best_deep_learning_model.pth'.
  - Saves model_comparison.csv, dl_model_comparison_v2.png, training curves, and best_model_config.json.

Run:
    cd banking-customer-support-ai
    python src/train_deep_learning.py
    # Or to train specific models:
    python src/train_deep_learning.py --models BiLSTMAttn TextCNN
"""

import sys
import time
import shutil
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils     import get_logger, ensure_dir, save_json, set_seed, load_json
from tokenizer import BankingTokenizer
from dataset   import load_splits
from models    import build_model, MODEL_REGISTRY

logger = get_logger("train_deep_learning")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────────────────────

GLOBAL_CONFIG = {
    "train_path":    ROOT / "data" / "train_split.csv",
    "val_path":      ROOT / "data" / "val_split.csv",
    "vocab_path":    ROOT / "artifacts" / "vocab.json",
    "label_path":    ROOT / "artifacts" / "label_mapping.json",
    "models_dir":    ROOT / "models" / "deep_learning",
    "results_dir":   ROOT / "results",
    "charts_dir":    ROOT / "results" / "charts",
    "artifacts_dir": ROOT / "artifacts",
    "random_seed":   42,
    "max_len":       64,
    "batch_size":    64,
    "grad_clip":     5.0,
}

# ── Per-model training configs ──────────────────────────────
# Recurrent models benefit from higher LR and patience to overcome initial convergence latency.

MODEL_CONFIGS = {
    "SimpleRNN": {
        "lr":           1e-3,
        "weight_decay": 1e-4,
        "num_epochs":   30,
        "patience":     15,
        "lr_patience":  6,
        "lr_factor":    0.5,
        "embed_dim":    128,
        "hidden_dim":   256,
        "num_layers":   1,
        "dropout":      0.3,
    },
    "LSTM": {
        "lr":           1e-3,
        "weight_decay": 1e-4,
        "num_epochs":   30,
        "patience":     15,
        "lr_patience":  6,
        "lr_factor":    0.5,
        "embed_dim":    128,
        "hidden_dim":   256,
        "num_layers":   1,
        "dropout":      0.3,
    },
    "GRU": {
        "lr":           1e-3,
        "weight_decay": 1e-4,
        "num_epochs":   30,
        "patience":     15,
        "lr_patience":  6,
        "lr_factor":    0.5,
        "embed_dim":    128,
        "hidden_dim":   256,
        "num_layers":   1,
        "dropout":      0.3,
    },
    "TextCNN": {
        "lr":           3e-4,
        "weight_decay": 1e-4,
        "num_epochs":   30,
        "patience":     8,
        "lr_patience":  4,
        "lr_factor":    0.5,
        "embed_dim":    128,
        "num_filters":  128,
        "kernel_sizes": [2, 3, 4, 5],
        "dropout":      0.4,
    },
    "BiLSTMAttn": {
        "lr":           5e-4,
        "weight_decay": 1e-4,
        "num_epochs":   30,
        "patience":     12,
        "lr_patience":  5,
        "lr_factor":    0.5,
        "embed_dim":    128,
        "hidden_dim":   256,
        "num_layers":   1,
        "dropout":      0.3,
        "num_heads":    4,
    },
}


# ─────────────────────────────────────────────────────────────
# Chart style
# ─────────────────────────────────────────────────────────────

CHART_STYLE = {
    "axes.facecolor":   "#1a1a2e",
    "figure.facecolor": "#16213e",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
}


# ─────────────────────────────────────────────────────────────
# Metrics & Training Step Helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "recall_macro":    round(recall_score(   y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_macro":        round(f1_score(       y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(       y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def train_epoch(model, loader, criterion, optimizer) -> tuple[float, float]:
    model.train()
    total_loss, preds, labels = 0.0, [], []
    for input_ids, label_batch in loader:
        input_ids    = input_ids.to(DEVICE)
        label_batch  = label_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = criterion(logits, label_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GLOBAL_CONFIG["grad_clip"])
        optimizer.step()
        total_loss += loss.item() * len(label_batch)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(label_batch.cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(labels, preds)


@torch.no_grad()
def eval_epoch(model, loader, criterion) -> tuple[float, dict]:
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for input_ids, label_batch in loader:
        input_ids   = input_ids.to(DEVICE)
        label_batch = label_batch.to(DEVICE)
        logits = model(input_ids)
        loss   = criterion(logits, label_batch)
        total_loss += loss.item() * len(label_batch)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(label_batch.cpu().tolist())
    return total_loss / len(loader.dataset), compute_metrics(labels, preds)


# ─────────────────────────────────────────────────────────────
# Per-model training loop
# ─────────────────────────────────────────────────────────────

def train_one_model(
    model_name:    str,
    vocab_size:    int,
    num_classes:   int,
    train_loader,
    val_loader,
    save_dir:      Path,
    class_weights: torch.Tensor,
) -> dict:
    """Train a single model with its model-specific config."""
    cfg = MODEL_CONFIGS[model_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {model_name}  (lr={cfg['lr']}, patience={cfg['patience']})")
    logger.info(f"{'='*60}")

    # Build model kwargs (filter to valid init params)
    model_kwargs = {k: v for k, v in cfg.items()
                    if k in ("embed_dim", "hidden_dim", "num_layers", "dropout",
                             "num_filters", "kernel_sizes", "num_heads")}
    model = build_model(
        model_name, vocab_size=vocab_size, num_classes=num_classes, **model_kwargs
    ).to(DEVICE)
    n_params = model.count_parameters()
    logger.info(f"  Parameters : {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["lr_factor"], patience=cfg["lr_patience"]
    )

    best_f1        = -1.0
    best_state     = None
    patience_count = 0
    history        = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    t_start = time.perf_counter()

    for epoch in range(1, cfg["num_epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_m  = eval_epoch(model, val_loader, criterion)
        val_f1, val_acc  = val_m["f1_macro"], val_m["accuracy"]

        scheduler.step(val_f1)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  Ep {epoch:02d}/{cfg['num_epochs']} | "
            f"Tr {tr_loss:.4f}/{tr_acc:.4f} | "
            f"Val {val_loss:.4f}/{val_acc:.4f} F1={val_f1:.4f} | LR={lr_now:.2e}"
        )

        if val_f1 > best_f1:
            best_f1        = val_f1
            best_state     = deepcopy(model.state_dict())
            patience_count = 0
            torch.save(best_state, save_dir / f"{model_name}_best.pth")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                logger.info(f"  Early stop at epoch {epoch}")
                break

    train_time = time.perf_counter() - t_start
    # Reload best weights and get final val metrics
    model.load_state_dict(best_state)
    _, final_m = eval_epoch(model, val_loader, criterion)
    logger.info(f"  Best val metrics -> {final_m}")

    return {
        "model_name":    model_name,
        "n_params":      n_params,
        "training_time": round(train_time, 1),
        "history":       history,
        **{f"val_{k}": v for k, v in final_m.items()},
    }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_curves(model_name: str, history: dict, charts_dir: Path):
    plt.rcParams.update(CHART_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — {model_name}", color="#e0e0e0", fontsize=12)
    epochs = range(1, len(history["train_loss"]) + 1)
    c = {"train": "#06d6a0", "val": "#ef476f"}

    axes[0].plot(epochs, history["train_loss"], color=c["train"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   color=c["val"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], color=c["train"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   color=c["val"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["val_f1"], color="#ffd166", label="Val Macro F1")
    axes[2].set_title("Val Macro F1")
    axes[2].legend()

    plt.tight_layout()
    out = charts_dir / f"training_curves_v2_{model_name.lower()}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"  Curves -> {out}")


def plot_comparison(results_df: pd.DataFrame, charts_dir: Path):
    plt.rcParams.update(CHART_STYLE)
    metrics = ["val_accuracy", "val_f1_macro", "val_f1_weighted"]
    x, width = np.arange(len(results_df)), 0.25
    colors   = ["#06d6a0", "#118ab2", "#ffd166"]

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (col, color) in enumerate(zip(metrics, colors)):
        vals = results_df[col].tolist()
        ax.bar(x + i * width, vals, width=width,
               label=col.replace("val_", ""), color=color, alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f"{v:.0%}",
                    ha="center", fontsize=8, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["model_name"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Deep Learning Model Comparison (Validation Set)", color="#e0e0e0", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = charts_dir / "dl_model_comparison_v2.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"  Comparison chart -> {out}")


# ─────────────────────────────────────────────────────────────
# Main Routine
# ─────────────────────────────────────────────────────────────

def run(models_to_train: list[str] | None = None):
    """
    Args:
        models_to_train: List of model names to train. If None, trains all 5.
    """
    set_seed(GLOBAL_CONFIG["random_seed"])
    ensure_dir(GLOBAL_CONFIG["models_dir"])
    ensure_dir(GLOBAL_CONFIG["charts_dir"])
    ensure_dir(GLOBAL_CONFIG["artifacts_dir"])

    logger.info("=" * 60)
    logger.info("  DL Training — 5 Architectures")
    logger.info("=" * 60)
    logger.info(f"  Device: {DEVICE}")

    # ── Load or build tokenizer ──
    vocab_file = Path(GLOBAL_CONFIG["vocab_path"])
    if not vocab_file.exists():
        logger.info(f"  Vocab not found at {vocab_file}, building automatically from training data...")
        df_train_raw = pd.read_csv(GLOBAL_CONFIG["train_path"])
        text_col = "clean_text" if "clean_text" in df_train_raw.columns else "text"
        tokenizer = BankingTokenizer(max_len=GLOBAL_CONFIG["max_len"], min_freq=2)
        tokenizer.build_vocab(df_train_raw[text_col].astype(str).tolist())
        tokenizer.save(vocab_file)
    else:
        tokenizer = BankingTokenizer.load(vocab_file)
    vocab_size = tokenizer.vocab_size
    logger.info(f"  Vocab size: {vocab_size:,}")

    # ── Build data loaders ──
    train_loader, val_loader, num_classes = load_splits(
        GLOBAL_CONFIG["train_path"],
        GLOBAL_CONFIG["val_path"],
        tokenizer,
        batch_size=GLOBAL_CONFIG["batch_size"],
        use_weighted_sampler=False,
    )
    logger.info(f"  Num classes: {num_classes}")

    # ── Class weights for imbalanced training ──
    df_train     = pd.read_csv(GLOBAL_CONFIG["train_path"])
    label_counts = df_train["label"].value_counts().sort_index().values
    class_weights = torch.tensor(
        1.0 / (label_counts.astype(float) + 1e-8), dtype=torch.float
    ).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * num_classes

    # ── Determine which models to train ──
    if models_to_train is None:
        models_to_train = list(MODEL_CONFIGS.keys())
    logger.info(f"  Models to train: {models_to_train}")

    # ── Train each model ──
    all_results = []
    for model_name in models_to_train:
        set_seed(GLOBAL_CONFIG["random_seed"])
        result = train_one_model(
            model_name, vocab_size, num_classes,
            train_loader, val_loader,
            GLOBAL_CONFIG["models_dir"], class_weights,
        )
        plot_curves(model_name, result.pop("history"), GLOBAL_CONFIG["charts_dir"])
        all_results.append(result)

    # ── Build comparison table ──
    cols = ["model_name", "val_accuracy", "val_precision_macro",
            "val_recall_macro", "val_f1_macro", "val_f1_weighted",
            "n_params", "training_time"]
    results_df = (
        pd.DataFrame(all_results)
          .sort_values("val_f1_macro", ascending=False)
          .reset_index(drop=True)
    )
    logger.info("\n" + "=" * 60)
    logger.info("  Model Comparison (sorted by Val Macro F1)")
    logger.info("=" * 60)
    logger.info("\n" + results_df[[c for c in cols if c in results_df.columns]].to_string(index=False))

    # ── Save comparison ──
    results_df[cols].to_csv(GLOBAL_CONFIG["results_dir"] / "model_comparison.csv", index=False)
    plot_comparison(results_df, GLOBAL_CONFIG["charts_dir"])

    # ── Select best model ──
    best_row  = results_df.iloc[0]
    best_name = best_row["model_name"]
    logger.info(f"\n[SELECTION] Best DL model: {best_name}")
    logger.info(f"  Val Accuracy : {best_row['val_accuracy']:.4f}")
    logger.info(f"  Val Macro F1 : {best_row['val_f1_macro']:.4f}")
    logger.info(
        f"\n  Justification: '{best_name}' achieves highest validation Macro F1 "
        f"({best_row['val_f1_macro']:.4f}) among candidate architectures."
    )

    # ── Copy best checkpoint ──
    best_ckpt = GLOBAL_CONFIG["models_dir"] / f"{best_name}_best.pth"
    shutil.copy2(best_ckpt, GLOBAL_CONFIG["models_dir"] / "best_deep_learning_model.pth")
    logger.info(f"  Checkpoint -> models/deep_learning/best_deep_learning_model.pth")

    # ── Save best model config ──
    cfg = MODEL_CONFIGS[best_name]
    best_config = {
        "model_name":   best_name,
        "vocab_size":   vocab_size,
        "num_classes":  num_classes,
        "embed_dim":    cfg.get("embed_dim", 128),
        "hidden_dim":   cfg.get("hidden_dim", 256),
        "num_layers":   cfg.get("num_layers", 1),
        "dropout":      cfg.get("dropout", 0.3),
        "num_filters":  cfg.get("num_filters", 128),
        "kernel_sizes": cfg.get("kernel_sizes", [2, 3, 4, 5]),
        "num_heads":    cfg.get("num_heads", 4),
        "max_len":      GLOBAL_CONFIG["max_len"],
    }
    save_json(best_config, GLOBAL_CONFIG["artifacts_dir"] / "best_model_config.json")

    # ── Save dl_metrics ──
    save_json(
        {"best_model": best_name, "all_model_results": all_results},
        GLOBAL_CONFIG["artifacts_dir"] / "dl_metrics.json",
    )

    logger.info("\nNote: Run 'python src/setup_artifacts.py' next to calibrate thresholds.")
    logger.info("\n[OK] Deep learning training complete.")
    logger.info("  Next: python src/final_evaluation.py")
    return best_name, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom DL architectures")
    parser.add_argument(
        "--models", nargs="+",
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific models to train (default: all 5)"
    )
    args = parser.parse_args()
    run(models_to_train=args.models)

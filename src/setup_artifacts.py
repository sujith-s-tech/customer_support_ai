"""
setup_artifacts.py -- Post-training artifact setup script
==========================================================
Purpose:
  After training DL models, this script:
    1. Evaluates all saved checkpoints on the validation set.
    2. Selects the best model by val Macro F1.
    3. Saves best_model_config.json and best_deep_learning_model.pth.
    4. Saves model_comparison.csv and dl_metrics.json.
    5. Calibrates confidence thresholds from the validation set.
    6. Saves thresholds.json.

Run:
    cd banking-customer-support-ai
    python src/setup_artifacts.py
"""

import sys
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils      import get_logger, ensure_dir, save_json, load_json, set_seed
from tokenizer  import BankingTokenizer
from models     import build_model
from confidence import calibrate

logger = get_logger("setup_artifacts")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "val_path":       ROOT / "data" / "val_split.csv",
    "vocab_path":     ROOT / "artifacts" / "vocab.json",
    "label_path":     ROOT / "artifacts" / "label_mapping.json",
    "models_dir":     ROOT / "models" / "deep_learning",
    "results_dir":    ROOT / "results",
    "charts_dir":     ROOT / "results" / "charts",
    "artifacts_dir":  ROOT / "artifacts",

    # Model hyperparameters (must match training config)
    "embed_dim":    128,
    "hidden_dim":   256,
    "num_layers":   2,
    "dropout":      0.4,
    "num_filters":  128,
    "kernel_sizes": [2, 3, 4, 5],
    "max_len":      64,
}

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
# Model evaluation helper
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model_name: str, tokenizer: BankingTokenizer,
                   texts: list, labels: list,
                   vocab_size: int, num_classes: int) -> dict | None:
    """Load a saved checkpoint and evaluate on validation set."""
    ckpt_path = CONFIG["models_dir"] / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        logger.warning(f"  Checkpoint not found: {ckpt_path} — skipping")
        return None

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    has_l1 = any("_l1" in k for k in state.keys())
    num_layers = 2 if has_l1 else 1

    # Build model with matching hyperparameters
    if model_name == "TextCNN":
        kwargs = dict(
            embed_dim=CONFIG["embed_dim"],
            num_filters=CONFIG["num_filters"],
            kernel_sizes=CONFIG["kernel_sizes"],
            dropout=CONFIG["dropout"],
        )
    elif model_name == "BiLSTMAttn":
        kwargs = dict(
            embed_dim=CONFIG["embed_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=num_layers,
            dropout=0.3,
            num_heads=4,
        )
    else:
        kwargs = dict(
            embed_dim=CONFIG["embed_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=num_layers,
            dropout=0.3,
        )

    model = build_model(model_name, vocab_size=vocab_size,
                        num_classes=num_classes, **kwargs).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    # Batch inference for speed
    all_preds  = []
    all_probs  = []
    batch_size = 256

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        ids = torch.tensor(
            tokenizer.batch_encode(batch_texts), dtype=torch.long
        ).to(DEVICE)
        logits = model(ids)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    acc  = accuracy_score(labels, all_preds)
    prec = precision_score(labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(labels, all_preds, average="macro", zero_division=0)
    f1m  = f1_score(labels, all_preds, average="macro", zero_division=0)
    f1w  = f1_score(labels, all_preds, average="weighted", zero_division=0)

    # Max confidence per sample
    confs   = np.array([max(p) for p in all_probs])
    correct = np.array([p == l for p, l in zip(all_preds, labels)])

    logger.info(
        f"  {model_name:<14}: acc={acc:.4f} prec={prec:.4f} "
        f"rec={rec:.4f} f1_mac={f1m:.4f} f1_wt={f1w:.4f} params={n_params:,}"
    )

    return {
        "model_name":          model_name,
        "val_accuracy":        round(acc,  4),
        "val_precision_macro": round(prec, 4),
        "val_recall_macro":    round(rec,  4),
        "val_f1_macro":        round(f1m,  4),
        "val_f1_weighted":     round(f1w,  4),
        "n_params":            n_params,
        "training_time":       "See training log",
        # Store confs/correct for calibration of the best model
        "_confs":              confs,
        "_correct":            correct,
        "_kwargs":             kwargs,
    }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, charts_dir: Path):
    plt.rcParams.update(CHART_STYLE)
    metrics = ["val_accuracy", "val_f1_macro", "val_f1_weighted"]
    x = np.arange(len(results_df))
    width = 0.25
    colors = ["#06d6a0", "#118ab2", "#ffd166"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (col, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, results_df[col], width=width,
               label=col.replace("val_", ""), color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["model_name"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Deep Learning Model Comparison (Validation Set)", color="#e0e0e0", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = charts_dir / "dl_model_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"DL model comparison chart saved -> {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run():
    set_seed(42)
    ensure_dir(CONFIG["results_dir"])
    ensure_dir(CONFIG["charts_dir"])
    ensure_dir(CONFIG["artifacts_dir"])

    logger.info("=" * 60)
    logger.info("  Setup Artifacts — DL Model Comparison & Selection")
    logger.info("=" * 60)
    logger.info(f"  Device: {DEVICE}")

    # ── Load tokenizer ──
    tokenizer = BankingTokenizer.load(CONFIG["vocab_path"])
    vocab_size = tokenizer.vocab_size

    # ── Load label mapping ──
    label_data  = load_json(CONFIG["label_path"])
    idx2label   = {int(k): v for k, v in label_data["idx2label"].items()}
    label2idx   = label_data["label2idx"]
    num_classes = len(idx2label)
    logger.info(f"  vocab_size={vocab_size}, num_classes={num_classes}")

    # ── Load validation data ──
    df_val = pd.read_csv(CONFIG["val_path"])
    texts  = df_val["clean_text"].astype(str).tolist()
    labels = df_val["label"].tolist()
    logger.info(f"  Val samples: {len(texts):,}")

    # ── Evaluate all available models ──
    model_names = ["SimpleRNN", "LSTM", "GRU", "TextCNN", "BiLSTMAttn"]
    all_results = []

    logger.info("\nEvaluating checkpoints on validation set...")
    for model_name in model_names:
        result = evaluate_model(
            model_name, tokenizer, texts, labels, vocab_size, num_classes
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        logger.error("No model checkpoints found. Run train_deep_learning.py first.")
        return

    # ── Sort by val Macro F1 ──
    comparison_cols = [
        "model_name", "val_accuracy", "val_precision_macro",
        "val_recall_macro", "val_f1_macro", "val_f1_weighted",
        "n_params", "training_time",
    ]
    results_df = (
        pd.DataFrame(all_results)
          .sort_values("val_f1_macro", ascending=False)
          .reset_index(drop=True)
    )

    logger.info("\n" + "=" * 60)
    logger.info("  Model Comparison Table (sorted by Val Macro F1)")
    logger.info("=" * 60)
    logger.info("\n" + results_df[comparison_cols].to_string(index=False))

    # ── Save comparison CSV ──
    results_df[comparison_cols].to_csv(
        CONFIG["results_dir"] / "model_comparison.csv", index=False
    )
    logger.info(f"\nmodel_comparison.csv saved -> {CONFIG['results_dir'] / 'model_comparison.csv'}")

    # ── Plot comparison ──
    plot_model_comparison(results_df[comparison_cols], CONFIG["charts_dir"])

    # ── Select best model ──
    best_row      = results_df.iloc[0]
    best_name     = best_row["model_name"]
    best_kwargs   = best_row["_kwargs"]
    best_confs    = best_row["_confs"]
    best_correct  = best_row["_correct"]

    logger.info(f"\n[SELECTION] Best DL model: {best_name}")
    logger.info(f"  Val Accuracy   : {best_row['val_accuracy']:.4f}")
    logger.info(f"  Val Macro F1   : {best_row['val_f1_macro']:.4f}")
    logger.info(f"  Val Weighted F1: {best_row['val_f1_weighted']:.4f}")
    logger.info(f"  Parameters     : {best_row['n_params']:,}")
    logger.info(
        f"\n  Justification: '{best_name}' achieves the highest validation "
        f"Macro F1 ({best_row['val_f1_macro']:.4f}) among all candidate "
        f"architectures. Macro F1 is preferred over accuracy because the "
        f"training set is imbalanced across {num_classes} intent categories."
    )

    # ── Copy best checkpoint ──
    best_ckpt_src = CONFIG["models_dir"] / f"{best_name}_best.pth"
    best_ckpt_dst = CONFIG["models_dir"] / "best_deep_learning_model.pth"
    shutil.copy2(best_ckpt_src, best_ckpt_dst)
    logger.info(f"  Best checkpoint copied -> {best_ckpt_dst}")

    # ── Save best_model_config.json ──
    best_config = {
        "model_name":   best_name,
        "vocab_size":   vocab_size,
        "num_classes":  num_classes,
        "embed_dim":    CONFIG["embed_dim"],
        "hidden_dim":   CONFIG["hidden_dim"],
        "num_layers":   CONFIG["num_layers"],
        "dropout":      CONFIG["dropout"],
        "num_filters":  CONFIG["num_filters"],
        "kernel_sizes": CONFIG["kernel_sizes"],
        "max_len":      CONFIG["max_len"],
    }
    # Merge with model-specific kwargs
    best_config.update(best_kwargs)
    save_json(best_config, CONFIG["artifacts_dir"] / "best_model_config.json")
    logger.info(f"  best_model_config.json saved -> {CONFIG['artifacts_dir'] / 'best_model_config.json'}")

    # ── Save dl_metrics.json ──
    clean_results = []
    for r in all_results:
        cr = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(cr)

    dl_metrics = {
        "best_model": best_name,
        "all_model_results": clean_results,
    }
    save_json(dl_metrics, CONFIG["artifacts_dir"] / "dl_metrics.json")
    logger.info(f"  dl_metrics.json saved -> {CONFIG['artifacts_dir'] / 'dl_metrics.json'}")

    # ── Confidence calibration using best model's val predictions ──
    logger.info("\n--- Confidence Threshold Calibration ---")
    thresholds = calibrate(
        best_confs,
        best_correct,
        charts_dir=CONFIG["charts_dir"],
    )
    logger.info(f"  Thresholds: HIGH={thresholds['high']}, MEDIUM={thresholds['medium']}")

    logger.info("\n[OK] Artifact setup complete.")
    logger.info("  Next step: Run src/final_evaluation.py for final test evaluation.")


if __name__ == "__main__":
    run()

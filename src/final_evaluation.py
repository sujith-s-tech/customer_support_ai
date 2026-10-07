"""
final_evaluation.py -- Phase 8: Final Test Evaluation
======================================================
Objective:
  Evaluate both the ML baseline and the best DL model on the
  UNTOUCHED test set for the first and ONLY time.

IMPORTANT RULE: This script must be run ONLY ONCE, after all
model selection decisions are final. The test set result is
the final, unbiased performance estimate.

Run:
    cd banking-customer-support-ai
    python src/final_evaluation.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils              import get_logger, ensure_dir, save_json, load_json
from data_preprocessing import clean_text
from predict            import DLPredictor, MLPredictor
from confidence         import calibrate

logger = get_logger("final_evaluation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "test_path":      ROOT / "data" / "test_clean.csv",   # preprocessed test set
    "val_path":       ROOT / "data" / "val_split.csv",
    "results_dir":    ROOT / "results",
    "charts_dir":     ROOT / "results" / "charts",
    "artifacts_dir":  ROOT / "artifacts",
    "random_seed":    42,
}

CHART_STYLE = {
    "axes.facecolor":   "#1a1a2e",
    "figure.facecolor": "#16213e",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "precision_macro":  round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "recall_macro":     round(recall_score(   y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_macro":         round(f1_score(       y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":      round(f1_score(       y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def plot_confusion_matrix_test(y_true, y_pred, labels, title, save_path):
    plt.rcParams.update(CHART_STYLE)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if len(labels) > 25:
        totals = cm.sum(axis=1)
        top_idx = np.argsort(totals)[-25:][::-1]
        cm = cm[np.ix_(top_idx, top_idx)]
        labels = [labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title, color="#e0e0e0", fontsize=12)
    ax.set_xlabel("Predicted", color="#e0e0e0")
    ax.set_ylabel("True",      color="#e0e0e0")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Confusion matrix saved -> {save_path}")


def plot_ml_vs_dl(ml_metrics: dict, dl_metrics: dict, save_path: Path):
    """Side-by-side bar chart comparing ML baseline vs best DL model on test set."""
    plt.rcParams.update(CHART_STYLE)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    ml_vals = [ml_metrics.get(m, 0) for m in metrics]
    dl_vals = [dl_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, ml_vals, width=width, label="ML Baseline",   color="#06d6a0", alpha=0.85)
    ax.bar(x + width/2, dl_vals, width=width, label="Best DL Model", color="#118ab2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("ML Baseline vs Best DL Model (Test Set)", color="#e0e0e0", fontsize=12)
    ax.legend()
    for i, (ml_v, dl_v) in enumerate(zip(ml_vals, dl_vals)):
        ax.text(i - width/2, ml_v + 0.01, f"{ml_v:.2%}", ha="center", fontsize=8, color="#06d6a0")
        ax.text(i + width/2, dl_v + 0.01, f"{dl_v:.2%}", ha="center", fontsize=8, color="#118ab2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"  ML vs DL chart saved -> {save_path}")


# ─────────────────────────────────────────────────────────────
# Confidence calibration from validation set
# ─────────────────────────────────────────────────────────────

def calibrate_thresholds_from_val(dl_predictor: DLPredictor) -> dict:
    """Run threshold calibration on the validation set."""
    logger.info("\nCalibrating confidence thresholds from validation set...")
    df_val = pd.read_csv(CONFIG["val_path"])
    texts  = df_val["clean_text"].astype(str).tolist()
    labels = df_val["category"].astype(str).tolist()

    all_confs, all_correct = [], []
    for text, true_label in zip(texts, labels):
        res = dl_predictor.predict(text)
        all_confs.append(res["confidence"])
        all_correct.append(res["intent"] == true_label)

    confs   = np.array(all_confs)
    correct = np.array(all_correct)
    thresholds = calibrate(confs, correct, charts_dir=CONFIG["charts_dir"])
    return thresholds


# ─────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────

def run():
    """Execute Phase 8: Final evaluation on the untouched test set."""
    ensure_dir(CONFIG["results_dir"])
    ensure_dir(CONFIG["charts_dir"])
    ensure_dir(CONFIG["artifacts_dir"])

    logger.info("=" * 60)
    logger.info("  PHASE 8 - Final Test Evaluation")
    logger.info("=" * 60)
    logger.info("  WARNING: Test set is being used for the FIRST time.")
    logger.info("  All model selection decisions must be final before running this.")

    # ── Load test data ──
    logger.info(f"\nLoading test data from {CONFIG['test_path']}")
    df_test = pd.read_csv(CONFIG["test_path"])
    logger.info(f"  Test samples: {len(df_test):,}")

    texts_test  = df_test["clean_text"].astype(str).tolist()
    labels_test = df_test["category"].astype(str).tolist()
    label_names = sorted(set(labels_test))

    # ── Load predictors ──
    try:
        dl_predictor = DLPredictor.from_artifacts()
        logger.info("DL predictor loaded.")
    except Exception as e:
        logger.error(f"DL predictor failed: {e}")
        dl_predictor = None

    try:
        ml_predictor = MLPredictor.from_artifacts()
        logger.info("ML predictor loaded.")
    except Exception as e:
        logger.error(f"ML predictor failed: {e}")
        ml_predictor = None

    # ── Calibrate thresholds on VAL set (not test) ──
    if dl_predictor:
        calibrate_thresholds_from_val(dl_predictor)

    final_results = {}

    # ── Evaluate DL model ──
    if dl_predictor:
        logger.info("\n--- DL Model Evaluation on Test Set ---")
        t0 = time.perf_counter()
        dl_preds = [dl_predictor.predict(t)["intent"] for t in texts_test]
        dl_time  = time.perf_counter() - t0

        dl_metrics = compute_metrics(labels_test, dl_preds)
        logger.info(f"  Accuracy  : {dl_metrics['accuracy']:.4f}")
        logger.info(f"  Precision : {dl_metrics['precision_macro']:.4f}")
        logger.info(f"  Recall    : {dl_metrics['recall_macro']:.4f}")
        logger.info(f"  Macro F1  : {dl_metrics['f1_macro']:.4f}")
        logger.info(f"  Weighted F1: {dl_metrics['f1_weighted']:.4f}")
        logger.info(f"  Eval time : {dl_time:.1f}s")

        # Classification report
        dl_report = classification_report(
            labels_test, dl_preds, target_names=label_names, zero_division=0
        )
        logger.info(f"\nDL Classification Report:\n{dl_report}")

        pd.DataFrame(
            classification_report(labels_test, dl_preds, target_names=label_names,
                                  zero_division=0, output_dict=True)
        ).T.to_csv(CONFIG["results_dir"] / "test_classification_report_dl.csv")

        plot_confusion_matrix_test(
            labels_test, dl_preds, label_names,
            "DL Model - Test Set Confusion Matrix (Top 25 classes)",
            CONFIG["charts_dir"] / "confusion_matrix_dl_test.png",
        )
        final_results["dl"] = {**dl_metrics, "eval_time_s": round(dl_time, 2)}

    # ── Evaluate ML baseline ──
    if ml_predictor:
        logger.info("\n--- ML Baseline Evaluation on Test Set ---")
        t0 = time.perf_counter()
        ml_preds = [ml_predictor.predict(t)["intent"] for t in texts_test]
        ml_time  = time.perf_counter() - t0

        ml_metrics = compute_metrics(labels_test, ml_preds)
        logger.info(f"  Accuracy  : {ml_metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1  : {ml_metrics['f1_macro']:.4f}")
        logger.info(f"  Eval time : {ml_time:.1f}s")

        pd.DataFrame(
            classification_report(labels_test, ml_preds, target_names=label_names,
                                  zero_division=0, output_dict=True)
        ).T.to_csv(CONFIG["results_dir"] / "test_classification_report_ml.csv")

        plot_confusion_matrix_test(
            labels_test, ml_preds, label_names,
            "ML Baseline - Test Set Confusion Matrix (Top 25 classes)",
            CONFIG["charts_dir"] / "confusion_matrix_ml_test.png",
        )
        final_results["ml"] = {**ml_metrics, "eval_time_s": round(ml_time, 2)}

    # ── ML vs DL comparison chart ──
    if "dl" in final_results and "ml" in final_results:
        plot_ml_vs_dl(
            final_results["ml"], final_results["dl"],
            CONFIG["charts_dir"] / "ml_vs_dl_test.png",
        )

    # ── Save final metrics ──
    save_json(final_results, CONFIG["artifacts_dir"] / "final_test_metrics.json")
    logger.info(f"\nFinal test metrics saved -> {CONFIG['artifacts_dir'] / 'final_test_metrics.json'}")

    # ── Summary table ──
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL EVALUATION SUMMARY (TEST SET)")
    logger.info("=" * 60)
    logger.info(f"  {'Metric':<20}  {'ML Baseline':>12}  {'Deep Learning':>14}")
    logger.info(f"  {'-'*20}  {'-'*12}  {'-'*14}")
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        ml_v = final_results.get("ml", {}).get(metric, 0)
        dl_v = final_results.get("dl", {}).get(metric, 0)
        winner = "<-- DL" if dl_v > ml_v else ("<-- ML" if ml_v > dl_v else "   tie")
        logger.info(f"  {metric:<20}  {ml_v:>12.4f}  {dl_v:>14.4f}  {winner}")

    logger.info("\n[OK] Phase 8 (Final Evaluation) complete.")
    return final_results


if __name__ == "__main__":
    run()

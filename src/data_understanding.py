"""
Phase 1 - Data Validation and Data Understanding
=================================================
Objective:
  Perform a thorough validation and exploratory analysis of the raw
  train.csv and test.csv datasets before any modelling work.

What this script does:
  1. Loads both datasets.
  2. Validates structure, dtypes, missing values, duplicates.
  3. Computes text-length and word-count statistics.
  4. Analyses category distribution (train and test).
  5. Checks train/test category consistency.
  6. Identifies potentially noisy records.
  7. Generates and saves EDA charts under results/charts/.
  8. Saves a summary JSON to artifacts/data_summary.json.

Run:
    cd banking-customer-support-ai
    python src/data_understanding.py
"""

import os
import sys
import re
import warnings
from pathlib import Path

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend - safe on any OS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# -- Make sure src/ is importable when running this file directly --
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger, ensure_dir, save_json, set_seed

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows so box-drawing / arrow chars don't crash
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    try:
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
CONFIG = {
    "train_path": ROOT / "data" / "train.csv",
    "test_path":  ROOT / "data" / "test.csv",
    "results_dir":  ROOT / "results",
    "charts_dir":   ROOT / "results" / "charts",
    "artifacts_dir": ROOT / "artifacts",
    "random_seed": 42,
    # Visual style
    "palette": "husl",
    "fig_dpi": 150,
    "top_n_categories": 20,   # categories shown in distribution plot
}

logger = get_logger("data_understanding")
set_seed(CONFIG["random_seed"])

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _count_words(text: str) -> int:
    """Return number of whitespace-separated tokens in *text*."""
    return len(str(text).split())


def _count_chars(text: str) -> int:
    return len(str(text))


def _is_empty_or_whitespace(text: str) -> bool:
    return str(text).strip() == ""


def _load_dataset(path: Path, name: str) -> pd.DataFrame:
    logger.info(f"Loading {name} from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Shape: {df.shape}")
    return df


# -------------------------------------------------------------
# Validation functions
# -------------------------------------------------------------

def validate_structure(df: pd.DataFrame, name: str) -> dict:
    """Check columns, dtypes, missing values, duplicates."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Validating structure: {name}")
    logger.info(f"{'='*60}")

    required_cols = {"text", "category"}
    missing_cols  = required_cols - set(df.columns)
    extra_cols    = set(df.columns) - required_cols

    # Missing values
    missing_vals = df.isnull().sum().to_dict()
    total_missing = sum(missing_vals.values())

    # Duplicate rows
    dup_rows  = int(df.duplicated().sum())
    dup_texts = int(df.duplicated(subset=["text"]).sum())

    # Empty strings
    empty_texts = int(df["text"].apply(_is_empty_or_whitespace).sum())

    logger.info(f"  Rows            : {len(df):,}")
    logger.info(f"  Columns         : {list(df.columns)}")
    logger.info(f"  Missing cols    : {missing_cols or 'None'}")
    logger.info(f"  Extra cols      : {extra_cols or 'None'}")
    logger.info(f"  Missing values  : {missing_vals}")
    logger.info(f"  Duplicate rows  : {dup_rows}")
    logger.info(f"  Duplicate texts : {dup_texts}")
    logger.info(f"  Empty texts     : {empty_texts}")

    return {
        "name":            name,
        "rows":            len(df),
        "columns":         list(df.columns),
        "missing_columns": list(missing_cols),
        "extra_columns":   list(extra_cols),
        "missing_values":  missing_vals,
        "total_missing":   total_missing,
        "duplicate_rows":  dup_rows,
        "duplicate_texts": dup_texts,
        "empty_texts":     empty_texts,
    }


def compute_text_statistics(df: pd.DataFrame, name: str) -> dict:
    """Compute word-count and character-count descriptive statistics."""
    logger.info(f"\n--- Text Statistics: {name} ---")

    df = df.copy()
    df["word_count"] = df["text"].apply(_count_words)
    df["char_count"] = df["text"].apply(_count_chars)

    wc = df["word_count"].describe()
    cc = df["char_count"].describe()

    logger.info(f"  Word count  -> min={wc['min']:.0f}  max={wc['max']:.0f}  "
                f"mean={wc['mean']:.1f}  median={df['word_count'].median():.0f}  "
                f"std={wc['std']:.1f}")
    logger.info(f"  Char count  -> min={cc['min']:.0f}  max={cc['max']:.0f}  "
                f"mean={cc['mean']:.1f}  median={df['char_count'].median():.0f}  "
                f"std={cc['std']:.1f}")

    return {
        "word_count": {
            "min": float(wc["min"]),
            "max": float(wc["max"]),
            "mean": float(wc["mean"]),
            "median": float(df["word_count"].median()),
            "std": float(wc["std"]),
            "p25": float(wc["25%"]),
            "p75": float(wc["75%"]),
            "p95": float(df["word_count"].quantile(0.95)),
            "p99": float(df["word_count"].quantile(0.99)),
        },
        "char_count": {
            "min": float(cc["min"]),
            "max": float(cc["max"]),
            "mean": float(cc["mean"]),
            "median": float(df["char_count"].median()),
            "std": float(cc["std"]),
            "p95": float(df["char_count"].quantile(0.95)),
        },
        # Store augmented df for downstream plotting
        "_df": df,
    }


def analyse_categories(df: pd.DataFrame, name: str) -> dict:
    """Category-level stats: count, distribution, imbalance."""
    logger.info(f"\n--- Category Analysis: {name} ---")

    cat_counts = df["category"].value_counts()
    n_cats = int(df["category"].nunique())
    imbalance_ratio = float(cat_counts.iloc[0] / cat_counts.iloc[-1])

    logger.info(f"  Unique categories : {n_cats}")
    logger.info(f"  Most frequent     : {cat_counts.index[0]} ({cat_counts.iloc[0]})")
    logger.info(f"  Least frequent    : {cat_counts.index[-1]} ({cat_counts.iloc[-1]})")
    logger.info(f"  Imbalance ratio   : {imbalance_ratio:.1f}x")

    return {
        "unique_categories": n_cats,
        "most_frequent":     {"category": cat_counts.index[0],  "count": int(cat_counts.iloc[0])},
        "least_frequent":    {"category": cat_counts.index[-1], "count": int(cat_counts.iloc[-1])},
        "imbalance_ratio":   imbalance_ratio,
        "category_counts":   cat_counts.to_dict(),
        "_series":           cat_counts,
    }


def check_category_consistency(train_cats: set, test_cats: set) -> dict:
    """Verify that train and test categories are aligned."""
    logger.info("\n--- Category Consistency Check ---")

    only_in_train = train_cats - test_cats
    only_in_test  = test_cats  - train_cats
    shared        = train_cats & test_cats

    logger.info(f"  Train categories : {len(train_cats)}")
    logger.info(f"  Test categories  : {len(test_cats)}")
    logger.info(f"  Shared           : {len(shared)}")
    logger.info(f"  Only in train    : {only_in_train or 'None'}")
    logger.info(f"  Only in test     : {only_in_test  or 'None'}")

    consistent = (only_in_train == set()) and (only_in_test == set())
    logger.info(f"  Fully consistent : {consistent}")

    return {
        "train_count":   len(train_cats),
        "test_count":    len(test_cats),
        "shared_count":  len(shared),
        "only_in_train": list(only_in_train),
        "only_in_test":  list(only_in_test),
        "consistent":    consistent,
    }


def identify_noisy_records(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Flag potentially noisy records for inspection."""
    logger.info(f"\n--- Noise Detection: {name} ---")

    df = df.copy()
    # Heuristics for noisy records
    df["has_only_digits"]    = df["text"].str.match(r"^\d+$")
    df["very_short"]         = df["text"].apply(_count_words) <= 2
    df["has_url"]            = df["text"].str.contains(
        r"https?://|www\.", regex=True, case=False, na=False)
    df["has_email"]          = df["text"].str.contains(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        regex=True, na=False)
    df["has_special_only"]   = df["text"].str.match(r"^[^a-zA-Z0-9]+$")
    df["is_noisy"]           = (
        df["has_only_digits"] |
        df["very_short"]      |
        df["has_url"]         |
        df["has_email"]       |
        df["has_special_only"]
    )

    noisy = df[df["is_noisy"]]
    logger.info(f"  Total noisy records (heuristic) : {len(noisy)}")
    if len(noisy) > 0:
        logger.info(f"  Sample noisy texts:\n{noisy[['text','category']].head(5).to_string()}")

    return noisy[["text", "category", "very_short", "has_url", "has_email", "is_noisy"]]


# -------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------

CHART_STYLE = {
    "axes.facecolor":    "#1a1a2e",
    "figure.facecolor":  "#16213e",
    "axes.edgecolor":    "#e0e0e0",
    "text.color":        "#e0e0e0",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "grid.color":        "#2a2a4a",
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
}


def _apply_style():
    plt.rcParams.update(CHART_STYLE)
    sns.set_palette("husl")


def plot_category_distribution(cat_series: pd.Series, name: str, charts_dir: Path):
    """Horizontal bar chart of category frequencies (top N + bottom N)."""
    _apply_style()
    top_n = CONFIG["top_n_categories"]

    top    = cat_series.head(top_n)
    bottom = cat_series.tail(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(f"Category Distribution - {name}", fontsize=14, color="#e0e0e0", y=1.01)

    for ax, data, title, color in zip(
        axes,
        [top, bottom],
        [f"Top {top_n} Categories", f"Bottom {top_n} Categories"],
        ["#06d6a0", "#ef476f"],
    ):
        bars = ax.barh(data.index[::-1], data.values[::-1], color=color, alpha=0.85)
        ax.set_title(title, color="#e0e0e0", fontsize=11)
        ax.set_xlabel("Sample Count", color="#e0e0e0")
        for bar, val in zip(bars, data.values[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", color="#e0e0e0", fontsize=8)

    plt.tight_layout()
    out = charts_dir / f"category_distribution_{name.lower().replace(' ','_')}.png"
    plt.savefig(out, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved -> {out}")


def plot_text_length_distributions(stat_dict: dict, name: str, charts_dir: Path):
    """Word-count and char-count histograms + KDE."""
    _apply_style()
    df = stat_dict["_df"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Text Length Distributions - {name}", color="#e0e0e0", fontsize=13)

    for ax, col, color, label in zip(
        axes,
        ["word_count",  "char_count"],
        ["#06d6a0",     "#118ab2"],
        ["Word Count",  "Character Count"],
    ):
        ax.hist(df[col], bins=40, color=color, alpha=0.7, edgecolor="none")
        ax.set_xlabel(label, color="#e0e0e0")
        ax.set_ylabel("Frequency", color="#e0e0e0")
        ax.set_title(label, color="#e0e0e0")
        # Vertical lines for mean and median
        ax.axvline(df[col].mean(),   color="#ffd166", linestyle="--", lw=1.5, label="Mean")
        ax.axvline(df[col].median(), color="#ef476f", linestyle=":",  lw=1.5, label="Median")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = charts_dir / f"text_length_{name.lower().replace(' ','_')}.png"
    plt.savefig(out, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved -> {out}")


def plot_word_count_boxplot(train_df: pd.DataFrame, charts_dir: Path, max_cats: int = 15):
    """Box plot of word count per category (top N by frequency)."""
    _apply_style()

    top_cats = (train_df["category"].value_counts()
                .head(max_cats).index.tolist())
    subset = train_df[train_df["category"].isin(top_cats)].copy()
    subset["word_count"] = subset["text"].apply(_count_words)

    fig, ax = plt.subplots(figsize=(16, 7))
    palette = sns.color_palette("husl", n_colors=max_cats)
    sns.boxplot(data=subset, y="category", x="word_count",
                palette=palette, orient="h", ax=ax)
    ax.set_title(f"Word Count by Category (Top {max_cats})", color="#e0e0e0", fontsize=12)
    ax.set_xlabel("Word Count", color="#e0e0e0")
    ax.set_ylabel("Category", color="#e0e0e0")
    plt.tight_layout()
    out = charts_dir / "word_count_by_category.png"
    plt.savefig(out, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved -> {out}")


def plot_train_test_category_comparison(train_cats: pd.Series,
                                        test_cats: pd.Series,
                                        charts_dir: Path):
    """Side-by-side bar chart comparing train vs test samples per category (top 30)."""
    _apply_style()

    top_cats = train_cats.head(30).index
    compare = pd.DataFrame({
        "Train": train_cats[top_cats],
        "Test":  test_cats.reindex(top_cats, fill_value=0),
    })

    fig, ax = plt.subplots(figsize=(18, 8))
    x = range(len(compare))
    width = 0.4
    ax.bar([i - width/2 for i in x], compare["Train"], width=width,
           label="Train", color="#06d6a0", alpha=0.85)
    ax.bar([i + width/2 for i in x], compare["Test"],  width=width,
           label="Test",  color="#118ab2", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(compare.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sample Count", color="#e0e0e0")
    ax.set_title("Train vs Test Sample Count per Category (Top 30)", color="#e0e0e0", fontsize=12)
    ax.legend()
    plt.tight_layout()
    out = charts_dir / "train_test_category_comparison.png"
    plt.savefig(out, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved -> {out}")


def plot_class_imbalance_overview(cat_series: pd.Series, name: str, charts_dir: Path):
    """Simple bar chart showing full category distribution with colour gradient."""
    _apply_style()
    n = len(cat_series)
    colors = sns.color_palette("husl", n_colors=n)

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(range(n), cat_series.values, color=colors)
    ax.set_xticks([])
    ax.set_xlabel("Categories (sorted by frequency)", color="#e0e0e0")
    ax.set_ylabel("Sample Count", color="#e0e0e0")
    ax.set_title(f"Full Category Imbalance - {name} ({n} categories)", color="#e0e0e0", fontsize=12)
    ax.axhline(cat_series.mean(), color="#ffd166", linestyle="--", lw=1.5, label="Mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = charts_dir / f"class_imbalance_{name.lower().replace(' ','_')}.png"
    plt.savefig(out, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved -> {out}")


# -------------------------------------------------------------
# Summary table
# -------------------------------------------------------------

def print_summary_table(
    train_val: dict,
    test_val:  dict,
    train_txt: dict,
    test_txt:  dict,
    train_cat: dict,
    test_cat:  dict,
    consistency: dict,
):
    """Pretty-print an ASCII summary table to the console."""
    separator = "=" * 65
    print(f"\n{separator}")
    print("  DATASET SUMMARY TABLE")
    print(separator)

    rows = [
        ("Metric",                      "Train",                     "Test"),
        ("-" * 25,                       "-" * 18,                    "-" * 18),
        ("Total rows",                   f"{train_val['rows']:,}",     f"{test_val['rows']:,}"),
        ("Missing values",               str(train_val["total_missing"]), str(test_val["total_missing"])),
        ("Duplicate rows",               str(train_val["duplicate_rows"]), str(test_val["duplicate_rows"])),
        ("Duplicate texts",              str(train_val["duplicate_texts"]), str(test_val["duplicate_texts"])),
        ("Empty texts",                  str(train_val["empty_texts"]), str(test_val["empty_texts"])),
        ("Unique categories",            str(train_cat["unique_categories"]), str(test_cat["unique_categories"])),
        ("Imbalance ratio",              f'{train_cat["imbalance_ratio"]:.1f}x', f'{test_cat["imbalance_ratio"]:.1f}x'),
        ("Min word count",               str(int(train_txt["word_count"]["min"])), str(int(test_txt["word_count"]["min"]))),
        ("Max word count",               str(int(train_txt["word_count"]["max"])), str(int(test_txt["word_count"]["max"]))),
        ("Mean word count",              f'{train_txt["word_count"]["mean"]:.1f}', f'{test_txt["word_count"]["mean"]:.1f}'),
        ("Median word count",            f'{train_txt["word_count"]["median"]:.0f}', f'{test_txt["word_count"]["median"]:.0f}'),
        ("95th pct word count",          f'{train_txt["word_count"]["p95"]:.0f}', f'{test_txt["word_count"]["p95"]:.0f}'),
        ("Min char count",               str(int(train_txt["char_count"]["min"])), str(int(test_txt["char_count"]["min"]))),
        ("Max char count",               str(int(train_txt["char_count"]["max"])), str(int(test_txt["char_count"]["max"]))),
        ("Mean char count",              f'{train_txt["char_count"]["mean"]:.1f}', f'{test_txt["char_count"]["mean"]:.1f}'),
        ("Category consistency",         str(consistency["consistent"]), ""),
        ("Categories only in train",     str(len(consistency["only_in_train"])), ""),
        ("Categories only in test",      str(len(consistency["only_in_test"])), ""),
    ]

    for label, train_v, test_v in rows:
        print(f"  {label:<30} {train_v:<20} {test_v}")
    print(separator + "\n")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def run():
    """Execute Phase 1: Data Validation and Data Understanding."""
    logger.info("=" * 58)
    logger.info("  PHASE 1 - Data Validation & Data Understanding")
    logger.info("=" * 58)

    # Setup output directories
    charts_dir   = ensure_dir(CONFIG["charts_dir"])
    artifacts_dir = ensure_dir(CONFIG["artifacts_dir"])
    results_dir   = ensure_dir(CONFIG["results_dir"])

    # -- 1. Load datasets --
    df_train = _load_dataset(CONFIG["train_path"], "Train")
    df_test  = _load_dataset(CONFIG["test_path"],  "Test")

    # -- 2. Structural validation --
    train_val = validate_structure(df_train, "Train")
    test_val  = validate_structure(df_test,  "Test")

    # -- 3. Text statistics --
    train_txt = compute_text_statistics(df_train, "Train")
    test_txt  = compute_text_statistics(df_test,  "Test")

    # Augmented DataFrames with word/char count columns
    df_train_aug = train_txt.pop("_df")
    df_test_aug  = test_txt.pop("_df")

    # -- 4. Category analysis --
    train_cat = analyse_categories(df_train, "Train")
    test_cat  = analyse_categories(df_test,  "Test")

    train_cat_series = train_cat.pop("_series")
    test_cat_series  = test_cat.pop("_series")

    # -- 5. Category consistency check --
    consistency = check_category_consistency(
        set(df_train["category"].unique()),
        set(df_test["category"].unique()),
    )

    # -- 6. Noise detection --
    logger.info("\nRunning noise detection...")
    noisy_train = identify_noisy_records(df_train, "Train")
    noisy_test  = identify_noisy_records(df_test,  "Test")

    # Save noisy record reports
    if len(noisy_train) > 0:
        noisy_train.to_csv(results_dir / "noisy_records_train.csv", index=False)
    if len(noisy_test) > 0:
        noisy_test.to_csv(results_dir / "noisy_records_test.csv", index=False)

    # -- 7. Summary table --
    print_summary_table(
        train_val, test_val,
        train_txt, test_txt,
        train_cat, test_cat,
        consistency,
    )

    # -- 8. Visualisations --
    logger.info("\nGenerating EDA visualisations...")

    plot_category_distribution(train_cat_series, "Train", charts_dir)
    plot_category_distribution(test_cat_series,  "Test",  charts_dir)

    plot_text_length_distributions({"_df": df_train_aug}, "Train", charts_dir)
    plot_text_length_distributions({"_df": df_test_aug},  "Test",  charts_dir)

    plot_word_count_boxplot(df_train_aug, charts_dir, max_cats=15)

    plot_train_test_category_comparison(train_cat_series, test_cat_series, charts_dir)

    plot_class_imbalance_overview(train_cat_series, "Train", charts_dir)
    plot_class_imbalance_overview(test_cat_series,  "Test",  charts_dir)

    # -- 9. Persist summary to JSON --
    summary = {
        "train": {
            "validation":  train_val,
            "text_stats":  train_txt,
            "category":    train_cat,
        },
        "test": {
            "validation":  test_val,
            "text_stats":  test_txt,
            "category":    test_cat,
        },
        "category_consistency": consistency,
        "noisy_train_count":    len(noisy_train),
        "noisy_test_count":     len(noisy_test),
    }
    summary_path = artifacts_dir / "data_summary.json"
    save_json(summary, summary_path)
    logger.info(f"\nData summary saved -> {summary_path}")

    # -- 10. Save category distribution CSV --
    cat_dist_df = pd.DataFrame({
        "category":    train_cat_series.index,
        "train_count": train_cat_series.values,
        "test_count":  test_cat_series.reindex(train_cat_series.index, fill_value=0).values,
    })
    cat_dist_df.to_csv(results_dir / "category_distribution.csv", index=False)
    logger.info(f"Category distribution CSV saved -> {results_dir / 'category_distribution.csv'}")

    logger.info("\n✅  Phase 1 complete.")
    logger.info(f"   Charts  -> {charts_dir}")
    logger.info(f"   Results -> {results_dir}")
    logger.info(f"   Artifacts -> {artifacts_dir}")

    return summary


if __name__ == "__main__":
    run()

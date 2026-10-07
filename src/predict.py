"""
predict.py -- Phase 9: Intent Prediction System
================================================
Objective:
  Provide a unified, reusable prediction interface that:
    1. Loads the best trained model (DL or ML).
    2. Preprocesses input text.
    3. Returns intent, confidence, and top-3 predictions.

Both the ML baseline and the best DL model are supported so we can
compare predictions side by side in the Streamlit app.

Usage:
    predictor = IntentPredictor.from_deep_learning()
    result = predictor.predict("my card was not delivered")
    print(result["intent"], result["confidence"])
"""

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import joblib

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils             import get_logger, load_json
from tokenizer         import BankingTokenizer
from models            import build_model
from data_preprocessing import clean_text

logger = get_logger("predict")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Deep Learning Predictor
# ─────────────────────────────────────────────────────────────

class DLPredictor:
    """
    Wraps the trained deep learning model for inference.
    """

    def __init__(
        self,
        model:      "torch.nn.Module",
        tokenizer:  BankingTokenizer,
        idx2label:  dict[int, str],
    ):
        self.model     = model.to(DEVICE).eval()
        self.tokenizer = tokenizer
        self.idx2label = idx2label
        self.num_classes = len(idx2label)

    @classmethod
    def from_artifacts(
        cls,
        vocab_path:        Path | str = ROOT / "artifacts" / "vocab.json",
        label_path:        Path | str = ROOT / "artifacts" / "label_mapping.json",
        model_config_path: Path | str = ROOT / "artifacts" / "best_model_config.json",
        model_weights_path: Path | str = ROOT / "models" / "deep_learning" / "best_deep_learning_model.pth",
    ) -> "DLPredictor":
        """Load all artifacts and return a ready-to-use DLPredictor."""
        logger.info("Loading DL predictor artifacts...")

        # Tokenizer
        tokenizer = BankingTokenizer.load(vocab_path)

        # Label mapping
        label_data = load_json(label_path)
        idx2label  = {int(k): v for k, v in label_data["idx2label"].items()}

        # Model architecture + weights
        config = load_json(model_config_path)
        model_name = config["model_name"]

        model_kwargs = {
            "embed_dim":  config["embed_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout":    config.get("dropout", 0.0),
        }
        if model_name == "TextCNN":
            model_kwargs = {
                "embed_dim":    config["embed_dim"],
                "num_filters":  config["num_filters"],
                "kernel_sizes": config["kernel_sizes"],
                "dropout":      config.get("dropout", 0.0),
            }
        elif model_name == "BiLSTMAttn":
            model_kwargs["num_heads"] = config.get("num_heads", 4)

        model = build_model(
            model_name,
            vocab_size=config["vocab_size"],
            num_classes=config["num_classes"],
            **model_kwargs,
        )
        state_dict = torch.load(model_weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info(f"DL model '{model_name}' loaded from {model_weights_path}")

        return cls(model=model, tokenizer=tokenizer, idx2label=idx2label)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Predict the banking intent for a given query.

        Args:
            text: Raw customer query (will be preprocessed internally).
        Returns:
            dict with keys:
                intent      : str  -- top predicted intent label
                confidence  : float -- probability of top intent (0-1)
                top_3       : list  -- top-3 intents with probabilities
                probabilities: list -- full probability distribution
        """
        cleaned = clean_text(text)
        ids     = torch.tensor(
            [self.tokenizer.encode(cleaned)], dtype=torch.long
        ).to(DEVICE)

        self.model.eval()
        logits = self.model(ids)                  # (1, num_classes)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]  # (num_classes,)

        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"intent": self.idx2label[int(i)], "confidence": round(float(probs[i]), 4)}
            for i in top3_idx
        ]

        return {
            "intent":        top3[0]["intent"],
            "confidence":    top3[0]["confidence"],
            "top_3":         top3,
            "probabilities": probs.tolist(),
            "cleaned_text":  cleaned,
        }


# ─────────────────────────────────────────────────────────────
# ML Baseline Predictor
# ─────────────────────────────────────────────────────────────

class MLPredictor:
    """
    Wraps the trained TF-IDF + classifier pipeline for inference.
    """

    def __init__(self, pipeline, idx2label: dict[int, str]):
        self.pipeline  = pipeline
        self.idx2label = idx2label

    @classmethod
    def from_artifacts(
        cls,
        pipeline_path: Path | str = ROOT / "models" / "tfidf" / "baseline_model.joblib",
        label_path:    Path | str = ROOT / "artifacts" / "label_mapping.json",
    ) -> "MLPredictor":
        pipeline = joblib.load(pipeline_path)
        label_data = load_json(label_path)
        idx2label  = {int(k): v for k, v in label_data["idx2label"].items()}
        logger.info(f"ML pipeline loaded from {pipeline_path}")
        return cls(pipeline=pipeline, idx2label=idx2label)

    def predict(self, text: str) -> dict:
        """Predict intent using the ML pipeline."""
        cleaned = clean_text(text)
        proba   = self.pipeline.predict_proba([cleaned])[0]   # (num_classes,)
        classes = self.pipeline.classes_                       # category strings

        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {"intent": str(classes[i]), "confidence": round(float(proba[i]), 4)}
            for i in top3_idx
        ]

        return {
            "intent":       top3[0]["intent"],
            "confidence":   top3[0]["confidence"],
            "top_3":        top3,
            "cleaned_text": cleaned,
        }


# ─────────────────────────────────────────────────────────────
# Unified predictor factory
# ─────────────────────────────────────────────────────────────

def load_predictor(model_type: str = "dl") -> "DLPredictor | MLPredictor":
    """
    Load and return the appropriate predictor.

    Args:
        model_type: "dl" for deep learning, "ml" for ML baseline.
    Returns:
        A ready-to-use predictor instance.
    """
    if model_type.lower() == "dl":
        return DLPredictor.from_artifacts()
    elif model_type.lower() == "ml":
        return MLPredictor.from_artifacts()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'dl' or 'ml'.")


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "I tried to withdraw 10000 from the ATM but no cash came out and my account was debited.",
        "Why was I charged a fee for my card payment?",
        "My card has not arrived yet, it's been 2 weeks.",
        "Can I use my card abroad?",
        "I need to change my PIN number.",
    ]

    print("\nTesting DL Predictor...")
    try:
        dl_pred = load_predictor("dl")
        for q in test_queries:
            res = dl_pred.predict(q)
            print(f"  Q: {q[:60]}")
            print(f"     Intent: {res['intent']}  Confidence: {res['confidence']:.2%}")
            print(f"     Top-3: {[x['intent'] for x in res['top_3']]}")
            print()
    except FileNotFoundError as e:
        print(f"  DL model artifacts not found: {e}")
        print("  Run train_deep_learning.py first.")

    print("\nTesting ML Predictor...")
    try:
        ml_pred = load_predictor("ml")
        for q in test_queries:
            res = ml_pred.predict(q)
            print(f"  Q: {q[:60]}")
            print(f"     Intent: {res['intent']}  Confidence: {res['confidence']:.2%}")
    except FileNotFoundError as e:
        print(f"  ML model artifacts not found: {e}")
        print("  Run train_baseline.py first.")

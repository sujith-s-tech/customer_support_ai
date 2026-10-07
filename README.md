# AI-Powered Banking Customer Support Intelligence & Response Automation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34%2B-ff4b4b?logo=streamlit)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google_Gemini-AI-4285F4?logo=google)](https://ai.google.dev)

---

## Project Overview

An end-to-end AI system for banking customer support that combines:

- **Custom Deep Learning NLP** (TextCNN, BiLSTM+Attention) for intent classification
- **Traditional ML Baseline** (TF-IDF + LinearSVC/Logistic Regression) for comparison
- **Confidence-based Escalation** using empirically calibrated thresholds
- **Rule-based Priority Detection** for HIGH / MEDIUM / LOW triage
- **Generative AI Responses** via Google Gemini API with strict safety guardrails
- **Conversation Intelligence** for sentiment, key information, and issue analysis
- **Analytics Dashboard** via Streamlit with real-time KPI tracking

---

## Business Problem

Banks handle thousands of customer queries daily across categories like:

- Card payments and fees
- Cash withdrawals (ATM issues)
- Bank transfers and delays
- Account verification
- Card delivery and replacement
- Fraud and unauthorized transactions
- Account security issues

This system automates intent classification, routes uncertain queries to humans, and generates professional AI-powered responses — all with safety and compliance built in.

---

## Objectives

1. Classify customer queries into 77 banking intent categories
2. Provide confidence scores and top-3 predictions
3. Escalate uncertain or high-risk cases to human agents
4. Generate professional AI responses using Gemini
5. Detect issue sentiment, priority, and key information
6. Compare Traditional ML vs Custom Deep Learning
7. Provide real-time analytics via Streamlit dashboard

---

## Dataset

| Property | Train | Test |
|---|---|---|
| Samples | 10,003 | 3,080 |
| Categories | 77 | 77 |
| Avg words/query | 11.9 | 11.0 |
| Max words/query | 79 | 69 |
| Imbalance ratio | 5.3x | 1.0x (balanced) |
| Missing values | 0 | 0 |

**Key Rule:** The test dataset was NEVER used during training, validation, or hyperparameter tuning. It was accessed only once for final evaluation.

---

## Data Preprocessing

Banking-aware NLP pipeline built without aggressive stopword removal:

- Lowercase and contraction expansion (preserving negation: "didn't" → "did not")
- URL and email removal
- Currency symbol normalization
- Digit removal
- Punctuation handling (hyphens preserved for compound tokens like "top-up")
- Whitespace normalization

**What was NOT done:**
- Stopword removal (would destroy "not", "never", "no")
- Stemming / lemmatization (can damage intent-bearing tokens)

---

## ML Baseline

| Model | Val Accuracy | Val Macro F1 | Train Time |
|---|---|---|---|
| TF-IDF + LinearSVC (calibrated) | 87.41% | 87.50% | 1.9s |
| TF-IDF + Logistic Regression | **87.61%** | **87.55%** | 6.9s |

**Best ML Model:** TF-IDF + Logistic Regression (Val Macro F1: 87.55%)

Parameters: TF-IDF with word unigrams + bigrams, max_features=80,000, sublinear_tf=True.

---

## Custom Deep Learning Architecture

### Custom NLP Pipeline (No Hugging Face Transformers)

- Custom word-level tokenizer built from training data only
- Vocabulary of ~1,357 tokens (min_freq=2)
- Max sequence length: 64 tokens
- Special tokens: `<PAD>` (index 0), `<UNK>` (index 1)

### Model Comparison

| Model | Val Accuracy | Val Macro F1 | Parameters |
|---|---|---|---|
| **TextCNN** ✅ | **89.46%** | **89.49%** | **443,085** |
| BiLSTM + Attention | 67.57% | 67.75% | 2,581,197 |
| LSTM | 5.15% | 1.67% | 1,115,085 |
| SimpleRNN | 2.05% | 0.11% | 423,885 |

**Selection Criterion:** Highest validation Macro F1 score (not accuracy, to handle imbalance).
**Winner: TextCNN** — Best performance with the fewest parameters (efficient and accurate).

### TextCNN Architecture

```
Embedding (1357, 128)
  ↓
Conv1D (kernel sizes: 2, 3, 4, 5) × 128 filters each
  ↓
Global Max Pooling per kernel
  ↓
Concatenate → (512 features)
  ↓
Dropout (0.4)
  ↓
Linear → 77 classes
```

### BiLSTM + Attention Architecture

```
Embedding (1357, 128)
  ↓
Dropout (0.4)
  ↓
BiLSTM (hidden=256, layers=2, bidirectional=True)
  ↓
Additive Attention (Bahdanau-style)
  ↓
Context Vector (512 features)
  ↓
Dropout (0.4)
  ↓
Linear → 77 classes
```

---

## Confidence and Escalation System

Thresholds empirically calibrated from the validation set (TextCNN model):

| Tier | Threshold | Action | Val Coverage | Val HC Accuracy |
|---|---|---|---|---|
| HIGH | ≥ 0.75 | Automated response | 87.91% | 95.34% |
| MEDIUM | ≥ 0.49 | Response + customer verification | — | — |
| LOW | < 0.49 | Escalate to human agent | 12.09% | — |

**Mean confidence on validation set:** 92.28%
**Error analysis:** High-confidence incorrect predictions are explicitly tracked to prevent silent failures.

---

## Priority Detection

Rule-based, deterministic system (no ML model):

| Priority | Examples |
|---|---|
| HIGH | Fraud, unauthorized transaction, money debited with no cash, compromised card |
| MEDIUM | Card not arrived, pending transfer, verification failed, PIN blocked |
| LOW | General info, fee enquiries, exchange rate questions |

HIGH-priority queries are ALWAYS escalated regardless of model confidence.

---

## Generative AI

**Model:** Google Gemini 1.5 Flash (configurable)

**Safety guardrails (non-negotiable):**
- Never requests passwords, PINs, OTPs, CVV, or full card numbers
- Never invents bank policies, fees, or refund timelines
- Never guarantees compensation or specific outcomes
- Always recommends escalation for sensitive/high-risk cases
- Temperature: 0.3 (factual, conservative responses)

**Fallback:** Template-based responses when API is unavailable

---

## Conversation Intelligence

Structured analysis for each query:

```json
{
  "issue": "...",
  "predicted_intent": "...",
  "confidence": 0.92,
  "confidence_tier": "HIGH",
  "sentiment": "Negative",
  "priority": "HIGH",
  "priority_reason": "...",
  "key_information": ["Amount: 10000", "ATM mention: atm"],
  "recommended_action": "...",
  "escalation_required": true,
  "escalation_reason": "..."
}
```

---

## Streamlit Application

**5 pages:**

1. **Customer Support** — Query input → intent + top-3 predictions + AI response
2. **Conversation Intelligence** — Deep analysis with sentiment and key info extraction
3. **Analytics Dashboard** — KPI cards + intent/confidence/priority/sentiment charts
4. **Model Performance** — ML vs DL comparison with validation/test metrics
5. **Escalation Queue** — Filterable table of cases requiring human attention

**Design:** Glassmorphism dark theme with Inter font, Plotly charts, animated KPI cards.

---

## Project Architecture

```
banking-customer-support-ai/
├── data/
│   ├── train.csv              # Original training data
│   ├── test.csv               # Original test data (UNTOUCHED until Phase 8)
│   ├── train_split.csv        # 80% training split
│   ├── val_split.csv          # 20% validation split
│   └── test_clean.csv         # Preprocessed test data
├── src/
│   ├── utils.py               # Shared helpers (logger, seed, I/O)
│   ├── data_understanding.py  # Phase 1: EDA
│   ├── data_preprocessing.py  # Phase 2: Text cleaning + split
│   ├── train_baseline.py      # Phase 3: TF-IDF + SVM/LR
│   ├── tokenizer.py           # Phase 4: Custom word tokenizer
│   ├── dataset.py             # Phase 4: PyTorch Dataset + DataLoader
│   ├── models.py              # Phase 5: DL architectures
│   ├── train_deep_learning.py # Phase 6-7: DL training + selection
│   ├── final_evaluation.py    # Phase 8: Test set evaluation
│   ├── predict.py             # Phase 9: Prediction pipeline
│   ├── confidence.py          # Phase 10: Threshold calibration
│   ├── priority.py            # Phase 11: Business priority rules
│   ├── response_generator.py  # Phase 12: Gemini response engine
│   ├── conversation_intelligence.py  # Phase 13: Conversation analysis
│   └── analytics.py           # Phase 14: Analytics engine
├── models/
│   ├── tfidf/                 # ML baseline artifacts
│   └── deep_learning/         # DL model checkpoints
├── artifacts/
│   ├── vocab.json             # Custom tokenizer vocabulary
│   ├── label_mapping.json     # Intent label mappings
│   ├── thresholds.json        # Calibrated confidence thresholds
│   ├── data_summary.json      # EDA summary
│   └── metrics.json           # Training metrics
├── results/
│   ├── model_comparison.csv   # DL model comparison table
│   ├── ml_comparison.csv      # ML model comparison
│   └── charts/                # All generated visualizations
├── app/
│   └── app.py                 # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Navigate to the project folder
cd d:\Projects_Guvi\Final_Project\banking-customer-support-ai

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Set up Gemini API key (optional — needed for AI-generated responses)
copy .env.example .env
# Edit .env and replace: GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Complete Execution Guide (Start to End)

> Run **every command** from the project root directory.
> All phases must be run **in order**.

---

### Phase 1 — Data Understanding & EDA

```bash
python src/data_understanding.py
```

Validates dataset, checks missing values, duplicates, category distribution, saves EDA charts to `results/charts/`.

---

### Phase 2 — NLP Preprocessing & Train/Val Split

```bash
python src/data_preprocessing.py
```

Cleans text (lowercasing, contraction expansion, negation preservation). Creates 80/20 stratified split.

**Creates:** `data/train_split.csv` · `data/val_split.csv` · `data/test_clean.csv`

---

### Phase 3 — ML Baseline Training

```bash
python src/train_baseline.py
```

Trains TF-IDF + LinearSVC and TF-IDF + Logistic Regression. Evaluates on validation set.

**Creates:** `models/tfidf/baseline_model.joblib` · `artifacts/ml_metrics.json`

---

### Phase 4 — Custom Tokenizer & Vocabulary

```bash
python src/tokenizer.py
```

Builds word-level vocabulary from training data only (min_freq=2).

**Creates:** `artifacts/vocab.json` (vocab_size ≈ 1,357)

---

### Phase 5 — Deep Learning Training (All 5 Architectures)

```bash
python src/train_deep_learning.py
```

Trains all 5 improved architectures with model-specific hyperparameters:

| # | Model | Key Improvement | LR | Patience |
|---|-------|----------------|-----|---------|
| 1 | SimpleRNN | Bidirectional GRU + max/mean/last pooling | 1e-3 | 15 |
| 2 | LSTM | Bidirectional + LayerNorm + orthogonal init | 1e-3 | 15 |
| 3 | GRU | Bidirectional + LayerNorm + orthogonal init | 1e-3 | 15 |
| 4 | TextCNN | Multi-kernel Conv1D + global max pool | 3e-4 | 8 |
| 5 | BiLSTMAttn | 4-head attention + residual + LayerNorm | 5e-4 | 12 |

**Creates:** `models/deep_learning/*_best.pth` (one per model) · training curve charts

⏱️ **Estimated time: 60–120 minutes on CPU**

---

### Phase 6 — Model Selection & Artifact Setup

```bash
python src/setup_artifacts.py
```

Evaluates all 5 checkpoints on the validation set, selects the best by Val Macro F1, calibrates confidence thresholds.

**Creates:** `artifacts/best_model_config.json` · `models/deep_learning/best_deep_learning_model.pth` · `artifacts/thresholds.json` · `results/model_comparison.csv`

---

### Phase 7 — Final Test Evaluation ⚠️ Run ONCE

```bash
python src/final_evaluation.py
```

> ⚠️ **This is the only time** `data/test.csv` is used. Run only after Phases 1–6 are complete.

Evaluates ML baseline and the best DL model on the held-out test set. Saves final unbiased metrics.

**Creates:** `artifacts/final_test_metrics.json` · confusion matrices · classification reports

---

### Phase 8 — Launch the Streamlit Application

```bash
streamlit run app/app.py
```

Open **http://localhost:8501** in your browser.

**5 pages:** Customer Support · Conversation Intelligence · Analytics Dashboard · Model Performance · Escalation Queue

---

### (Optional) Test Predictions via CLI

```bash
python src/predict.py
```


---

## Results

### Validation Set

| Model | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| TF-IDF + LinearSVC | 87.41% | 87.89% | 87.58% | 87.50% | 87.46% |
| TF-IDF + LR (Best ML) | 87.61% | 88.12% | 87.52% | 87.55% | 87.65% |
| SimpleRNN | 2.05% | 0.06% | 1.78% | 0.11% | 0.11% |
| LSTM | 5.15% | 1.58% | 4.54% | 1.67% | 2.06% |
| BiLSTMAttn | 67.57% | 70.31% | 68.47% | 67.75% | 67.58% |
| **TextCNN (Best DL)** ✅ | **89.46%** | **89.91%** | **89.56%** | **89.49%** | **89.45%** |

### Test Set (Final, Unbiased — Run Once)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| ML Baseline (TF-IDF + LR) | 87.79% | 88.39% | 87.79% | 87.84% | 87.84% |
| **TextCNN (Custom DL)** ✅ | **90.78%** | **91.17%** | **90.78%** | **90.78%** | **90.78%** |

> ⬆️ **Deep Learning outperforms ML Baseline by +2.99% accuracy and +2.94% Macro F1** on the held-out test set.

---

## Limitations

1. **No pretrained embeddings**: Vocabulary of ~1,357 tokens limits representation of rare words. Using GloVe/FastText embeddings (not Hugging Face) could improve performance.
2. **Small vocabulary**: Short banking queries have limited vocabulary diversity; the tokenizer may over-rely on common patterns.
3. **CPU training**: Models trained on CPU; GPU would enable larger architectures and more epochs.
4. **Sentiment analysis**: Simple keyword-based approach; a dedicated sentiment model would be more accurate.
5. **Response evaluation**: GenAI responses are not automatically evaluated for factual accuracy.

---

## Future Improvements

1. **Pretrained word vectors** (GloVe/FastText) for richer token representations
2. **Custom Transformer from scratch** (no Hugging Face) for better long-range dependencies
3. **Active learning loop** for handling edge cases and new intents
4. **Multi-label classification** for queries spanning multiple intents
5. **Fine-grained sentiment analysis** using a dedicated model
6. **Real-time monitoring** with concept drift detection
7. **Multilingual support** for regional language banking queries

---

## Ethical and Safety Considerations

- **No sensitive data exposure**: The system never stores or exposes customer account details
- **No credential requests**: Responses are guardrailed to never ask for passwords, PINs, OTPs, or card numbers
- **Human oversight**: High-risk and uncertain cases are always escalated to human agents
- **No invented policies**: The AI is restricted from fabricating bank policies or guarantees
- **Transparency**: Confidence scores and escalation reasons are always surfaced to agents
- **Bias awareness**: The training dataset imbalance is handled via class weighting, but per-class performance should be monitored regularly

---

## Author

Built as a complete end-to-end AI/ML + Deep Learning + Generative AI banking support system.

**Tech Stack:** Python | PyTorch | scikit-learn | Streamlit | Google Gemini | Plotly

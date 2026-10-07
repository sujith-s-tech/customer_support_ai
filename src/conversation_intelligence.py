"""
conversation_intelligence.py -- Phase 13: Conversation Intelligence Module
===========================================================================
Objective:
  Analyse a customer query and produce a structured intelligence report
  containing: issue summary, sentiment, priority, key information,
  recommended action, and escalation status.

Design:
  - Sentiment is determined using a simple keyword-based approach
    (no ML model needed for basic positive/negative/neutral).
  - Key information extraction uses regex patterns.
  - Escalation and priority use the same rules as confidence.py + priority.py.
  - Output is a structured dict / JSON-serialisable object.

Usage:
    from conversation_intelligence import ConversationAnalyzer
    analyzer = ConversationAnalyzer(predictor, priority_detector, thresholds)
    report = analyzer.analyze("I was charged twice for the same transaction.")
    print(report)
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils    import get_logger
from priority import PriorityDetector
from confidence import get_confidence_tier, escalation_reason, load_thresholds

logger = get_logger("conversation_intelligence")


# ─────────────────────────────────────────────────────────────
# Sentiment detection (keyword-based; no ML model)
# ─────────────────────────────────────────────────────────────

NEGATIVE_KEYWORDS = [
    "not working", "failed", "error", "problem", "issue", "wrong", "incorrect",
    "missing", "lost", "stolen", "declined", "rejected", "blocked", "frozen",
    "unauthorized", "fraud", "scam", "angry", "frustrated", "terrible", "horrible",
    "awful", "disappointed", "unacceptable", "ridiculous", "cant", "cannot",
    "never", "impossible", "urgent", "immediately", "complaint", "refund",
    "charged twice", "double charged", "no cash", "debited", "not received",
    "not arrived", "still waiting", "been waiting", "very long",
]

POSITIVE_KEYWORDS = [
    "thank", "thanks", "great", "excellent", "wonderful", "happy", "pleased",
    "resolved", "sorted", "fixed", "perfect", "good", "appreciate", "helpful",
    "love", "best", "awesome", "satisfied",
]


def detect_sentiment(text: str) -> str:
    """
    Classify sentiment as Positive, Neutral, or Negative.

    Uses a simple keyword-count approach: if negative keywords dominate,
    sentiment is Negative; if positive keywords dominate, Positive;
    otherwise Neutral.
    """
    text_lower = text.lower()

    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)

    if neg_count > 0 and neg_count >= pos_count:
        return "Negative"
    elif pos_count > 0 and pos_count > neg_count:
        return "Positive"
    else:
        return "Neutral"


# ─────────────────────────────────────────────────────────────
# Key information extraction
# ─────────────────────────────────────────────────────────────

# Patterns for key financial/account information in the query
KEY_INFO_PATTERNS = {
    "amount":         r"(?:rs\.?|inr|pound|gbp|euro|eur|usd|\$|£|€)?\s*\d+(?:[.,]\d+)?(?:\s*(?:rs\.?|inr|pound|gbp|euro|eur|usd))?",
    "time_reference": r"(?:\d+\s+(?:days?|weeks?|hours?|minutes?)|yesterday|today|this morning|last night|last week)",
    "card_mention":   r"(?:card|debit card|credit card|virtual card|contactless)",
    "atm_mention":    r"\batm\b",
    "transfer":       r"(?:transfer|transaction|payment|withdrawal|deposit)",
    "account":        r"(?:account|balance)",
    "international":  r"(?:international|abroad|foreign|overseas)",
    "fee_mention":    r"(?:fee|charge|charged|commission)",
    "refund_mention": r"(?:refund|reimburse|reimbursement|money back)",
    "pin_passcode":   r"(?:pin|passcode|password)",
}


def extract_key_information(text: str) -> list[str]:
    """
    Extract important facts from the customer query.

    Returns a list of key information strings found in the text.
    """
    text_lower = text.lower()
    found = []

    for key, pattern in KEY_INFO_PATTERNS.items():
        matches = re.findall(pattern, text_lower)
        if matches:
            # De-duplicate and capitalise for display
            unique_matches = list(set(m.strip() for m in matches if m.strip()))
            label = key.replace("_", " ").title()
            found.append(f"{label}: {', '.join(unique_matches)}")

    if not found:
        found.append("No specific financial keywords detected")

    return found


# ─────────────────────────────────────────────────────────────
# Recommended action rules
# ─────────────────────────────────────────────────────────────

INTENT_RECOMMENDATIONS = {
    "compromised_card":       "Block the card immediately via the app and call the fraud helpline.",
    "lost_or_stolen_card":    "Block the card via the app or helpline and request a replacement.",
    "lost_or_stolen_phone":   "Secure the account remotely and contact support immediately.",
    "transaction_charged_twice": "Review the transaction statement and raise a dispute.",
    "cash_withdrawal_not_recognised": "Raise a dispute for the ATM transaction with supporting evidence.",
    "card_payment_not_recognised": "Review recent transactions and raise a chargeback if needed.",
    "card_arrival":           "Check delivery status in the app; contact support if overdue.",
    "declined_card_payment":  "Check available balance and daily limits; retry or contact support.",
    "declined_cash_withdrawal": "Check ATM daily withdrawal limit and account balance.",
    "declined_transfer":      "Verify recipient details and check transfer limits.",
    "pending_transfer":       "Allow 1-3 business days; check for confirmation in the app.",
    "pending_card_payment":   "Most pending payments settle within 3-5 business days.",
    "pending_top_up":         "Top-up may take up to 2 hours; check payment source.",
    "top_up_failed":          "Verify the payment source has sufficient funds and try again.",
    "request_refund":         "Submit a refund request via the app or support ticket.",
    "verify_my_identity":     "Complete KYC verification in the app using a valid ID.",
    "unable_to_verify_identity": "Contact support with valid identification documents.",
    "passcode_forgotten":     "Reset passcode via the app; do not share credentials with anyone.",
    "pin_blocked":            "Unblock PIN via the app or visit a branch.",
    "exchange_rate":          "Check real-time rates in the app before transacting.",
    "transfer_fee_charged":   "Review our fee schedule in the app or website.",
    "balance_not_updated_after_bank_transfer": "Allow 1-2 business days for the balance to reflect.",
    "contactless_not_working": "Ensure the card is activated and has contactless enabled in settings.",
}

DEFAULT_RECOMMENDATION = (
    "Please check the app for relevant options or contact our support team "
    "for personalised assistance."
)


def get_recommended_action(intent: str, priority: str, escalate: bool) -> str:
    """Return the most appropriate recommended action for the query."""
    if escalate or priority == "HIGH":
        return (
            INTENT_RECOMMENDATIONS.get(intent, DEFAULT_RECOMMENDATION) +
            " -- URGENT: This case requires immediate human agent attention."
        )
    return INTENT_RECOMMENDATIONS.get(intent, DEFAULT_RECOMMENDATION)


# ─────────────────────────────────────────────────────────────
# Intelligence report
# ─────────────────────────────────────────────────────────────

@dataclass
class IntelligenceReport:
    issue:              str
    predicted_intent:   str
    confidence:         float
    confidence_tier:    str
    sentiment:          str
    priority:           str
    priority_reason:    str
    key_information:    list[str]
    recommended_action: str
    escalation_required: bool
    escalation_reason:  str
    raw_text:           str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Conversation Analyzer
# ─────────────────────────────────────────────────────────────

class ConversationAnalyzer:
    """
    Main orchestrator for conversation intelligence.

    Combines intent prediction, confidence assessment, priority detection,
    sentiment analysis, and key information extraction into a single report.
    """

    def __init__(
        self,
        predictor,               # DLPredictor or MLPredictor
        priority_detector: PriorityDetector = None,
        thresholds: dict = None,
    ):
        self.predictor        = predictor
        self.priority_detector = priority_detector or PriorityDetector()
        self.thresholds       = thresholds or load_thresholds()

    def analyze(self, text: str) -> IntelligenceReport:
        """
        Analyse a customer query and return a complete intelligence report.

        Args:
            text: Raw customer query string.
        Returns:
            IntelligenceReport dataclass.
        """
        # ── 1. Intent prediction ──
        prediction = self.predictor.predict(text)
        intent     = prediction["intent"]
        confidence = prediction["confidence"]

        # ── 2. Confidence tier + escalation ──
        tier     = get_confidence_tier(confidence, self.thresholds)
        escalate = (tier == "LOW")
        esc_reason = escalation_reason(confidence, intent, self.thresholds)

        # ── 3. Priority detection ──
        priority_result = self.priority_detector.detect(text, intent)

        # Override escalation if priority is HIGH (always escalate HIGH)
        if priority_result.priority == "HIGH":
            escalate = True
            if tier != "LOW":
                esc_reason = (
                    f"High-priority issue detected: {priority_result.reason} "
                    "Human agent review recommended for financial safety."
                )

        # ── 4. Sentiment detection ──
        sentiment = detect_sentiment(text)

        # ── 5. Key information extraction ──
        key_info = extract_key_information(text)

        # ── 6. Issue summary ──
        issue = _summarise_issue(text, intent, priority_result.priority)

        # ── 7. Recommended action ──
        action = get_recommended_action(intent, priority_result.priority, escalate)

        return IntelligenceReport(
            issue=issue,
            predicted_intent=intent,
            confidence=confidence,
            confidence_tier=tier,
            sentiment=sentiment,
            priority=priority_result.priority,
            priority_reason=priority_result.reason,
            key_information=key_info,
            recommended_action=action,
            escalation_required=escalate,
            escalation_reason=esc_reason,
            raw_text=text,
        )


def _summarise_issue(text: str, intent: str, priority: str) -> str:
    """
    Create a concise issue summary from the intent and text.
    Avoids revealing sensitive information.
    """
    intent_display = intent.replace("_", " ").title()
    # Take first sentence or first 100 characters as the issue summary
    sentences = re.split(r'[.!?]', text.strip())
    first_sentence = sentences[0].strip() if sentences else text[:100]
    if len(first_sentence) > 120:
        first_sentence = first_sentence[:120] + "..."
    return f"{intent_display}: {first_sentence}"


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from priority import PriorityDetector
    import json

    # Mock predictor for testing
    class MockPredictor:
        def predict(self, text):
            return {
                "intent":     "cash_withdrawal_not_recognised",
                "confidence": 0.35,   # low confidence to test escalation
                "top_3":      [
                    {"intent": "cash_withdrawal_not_recognised", "confidence": 0.35},
                    {"intent": "failed_transfer",                 "confidence": 0.20},
                    {"intent": "pending_cash_withdrawal",         "confidence": 0.15},
                ],
            }

    analyzer = ConversationAnalyzer(MockPredictor())
    report = analyzer.analyze(
        "I tried to withdraw 5000 from the ATM last night but the machine kept "
        "my card and debited my account. I am very frustrated."
    )
    print(json.dumps(report.to_dict(), indent=2))

"""
priority.py -- Phase 11: Business-Rule-Based Priority Detection
===============================================================
Objective:
  Assign a business priority (HIGH / MEDIUM / LOW) to each customer query
  based on detected banking keywords and the predicted intent.

Design principles:
  - Rule-based: no ML model is used here (deterministic and auditable).
  - Rules are documented and easy to update.
  - Priority does NOT depend on sentiment alone.
  - Banking-safety principle: when in doubt, assign higher priority.
  - Does NOT invent policies -- rules are based on common banking practices.

Usage:
    from priority import PriorityDetector
    detector = PriorityDetector()
    result = detector.detect("my money was debited but cash not received")
    print(result["priority"], result["reason"])
"""

import re
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
# Priority levels
# ─────────────────────────────────────────────────────────────

PRIORITY_HIGH   = "HIGH"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_LOW    = "LOW"


# ─────────────────────────────────────────────────────────────
# Keyword rule sets
#
# Each list contains keyword patterns (lowercased, simple substrings).
# A query matching ANY pattern in a HIGH list -> HIGH priority.
# ─────────────────────────────────────────────────────────────

# HIGH priority: financial loss, security, fraud, urgent account issues
HIGH_PRIORITY_PATTERNS = [
    # Financial loss / discrepancy
    r"money.*debited.*not received",
    r"debited.*no cash",
    r"cash not.*received",
    r"amount.*missing",
    r"missing.*amount",
    r"money.*missing",
    r"funds.*missing",
    r"unauthorized",
    r"unauthorised",
    r"fraudulent",
    r"fraud",
    r"scam",
    r"suspicious.*transaction",
    r"transaction.*suspicious",
    r"stolen.*card",
    r"card.*stolen",
    r"lost.*card",
    r"card.*lost",
    r"compromised",
    r"hacked",
    r"account.*hacked",
    r"someone.*used.*my card",
    r"did not make.*transaction",
    r"did not authoris",
    r"i did not do",
    r"wrong.*amount.*received",
    r"charged.*twice",
    r"double.*charged",
    r"duplicate.*charge",
    r"account.*blocked",
    r"account.*frozen",
    r"account.*suspended",
    r"cannot.*access.*account",
    r"locked.*out",
    r"wrong.*exchange.*rate",
    r"overcharged",
    r"charged.*wrong",
    r"emergency",
    r"urgent",
    r"immediately",
    r"right away",
]

# MEDIUM priority: pending/delayed issues that need resolution but no immediate loss
MEDIUM_PRIORITY_PATTERNS = [
    r"card.*not.*arrived",
    r"card.*still.*not here",
    r"waiting.*card",
    r"transfer.*pending",
    r"pending.*transfer",
    r"pending.*payment",
    r"payment.*pending",
    r"top.?up.*not.*reflected",
    r"not.*showing",
    r"balance.*not.*updated",
    r"verification.*failed",
    r"cannot.*verify",
    r"identity.*verification",
    r"verify.*identity",
    r"unable.*to.*use.*card",
    r"card.*declined",
    r"declined.*card",
    r"payment.*declined",
    r"declined.*payment",
    r"refund.*not.*received",
    r"refund.*pending",
    r"waiting.*refund",
    r"transfer.*delay",
    r"delayed.*transfer",
    r"international.*transfer",
    r"pin.*blocked",
    r"pin.*not.*working",
    r"passcode.*not.*working",
    r"cash.*withdrawal.*problem",
    r"atm.*problem",
    r"atm.*issue",
    r"contactless.*not.*working",
    r"virtual.*card.*not.*working",
]

# LOW priority: informational / non-urgent enquiries
LOW_PRIORITY_PATTERNS = [
    r"what.*fee",
    r"how.*much.*fee",
    r"exchange.*rate.*question",
    r"what.*exchange.*rate",
    r"how.*does.*work",
    r"how.*to.*use",
    r"what.*is",
    r"can i",
    r"is it possible",
    r"how.*long.*take",
    r"information",
    r"inquiry",
    r"enquiry",
    r"how.*to.*change.*pin",
    r"how.*to.*top.?up",
    r"how.*to.*order.*card",
    r"what.*cards.*support",
    r"what.*currencies",
    r"activate.*card",
]


# ─────────────────────────────────────────────────────────────
# Intent-based priority overrides
# (some intents are always high/medium regardless of text)
# ─────────────────────────────────────────────────────────────

HIGH_PRIORITY_INTENTS = {
    "compromised_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "transaction_charged_twice",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
    "cash_withdrawal_not_recognised",
    "card_payment_not_recognised",
    "Refund_not_showing_up",
    "reverted_card_payment?",
}

MEDIUM_PRIORITY_INTENTS = {
    "card_arrival",
    "card_not_working",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "failed_transfer",
    "transfer_not_received_by_recipient",
    "pin_blocked",
    "passcode_forgotten",
    "contactless_not_working",
    "virtual_card_not_working",
    "top_up_failed",
    "request_refund",
    "atm_support",
}


# ─────────────────────────────────────────────────────────────
# Priority detector
# ─────────────────────────────────────────────────────────────

@dataclass
class PriorityResult:
    priority:    str
    reason:      str
    matched_rule: str = ""


class PriorityDetector:
    """
    Deterministic, rule-based priority detection for banking queries.
    """

    def __init__(self):
        # Pre-compile all regex patterns for performance
        self._high_re   = [re.compile(p, re.IGNORECASE) for p in HIGH_PRIORITY_PATTERNS]
        self._medium_re = [re.compile(p, re.IGNORECASE) for p in MEDIUM_PRIORITY_PATTERNS]

    def detect(self, text: str, intent: str = "") -> PriorityResult:
        """
        Determine the priority of a customer query.

        Priority resolution order:
          1. Intent-based override (HIGH > MEDIUM)
          2. Text keyword rules  (HIGH > MEDIUM > LOW)
          3. Default: MEDIUM (err on the side of caution for banking)

        Args:
            text   : Customer query text (raw or cleaned).
            intent : Predicted intent label (optional; improves accuracy).
        Returns:
            PriorityResult dataclass.
        """
        text_lower = text.lower()

        # ── 1. Intent-based override ──
        if intent in HIGH_PRIORITY_INTENTS:
            return PriorityResult(
                priority=PRIORITY_HIGH,
                reason=f"Intent '{intent}' is classified as high-priority.",
                matched_rule=f"intent:{intent}",
            )

        # ── 2. Text keyword rules: HIGH ──
        for pattern in self._high_re:
            m = pattern.search(text_lower)
            if m:
                return PriorityResult(
                    priority=PRIORITY_HIGH,
                    reason=f"High-priority keyword detected: '{m.group()}'",
                    matched_rule=pattern.pattern,
                )

        # ── 3. Intent-based override: MEDIUM ──
        if intent in MEDIUM_PRIORITY_INTENTS:
            return PriorityResult(
                priority=PRIORITY_MEDIUM,
                reason=f"Intent '{intent}' is classified as medium-priority.",
                matched_rule=f"intent:{intent}",
            )

        # ── 4. Text keyword rules: MEDIUM ──
        for pattern in self._medium_re:
            m = pattern.search(text_lower)
            if m:
                return PriorityResult(
                    priority=PRIORITY_MEDIUM,
                    reason=f"Medium-priority keyword detected: '{m.group()}'",
                    matched_rule=pattern.pattern,
                )

        # ── 5. Default (safe fallback) ──
        return PriorityResult(
            priority=PRIORITY_MEDIUM,
            reason="No specific priority keyword detected. Defaulting to MEDIUM for safety.",
            matched_rule="default",
        )


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = PriorityDetector()
    tests = [
        ("My account was debited but I did not receive cash from ATM.", "cash_withdrawal_not_recognised"),
        ("I think someone used my card without my permission.", "compromised_card"),
        ("When will my card arrive? It's been 10 days.", "card_arrival"),
        ("What is the exchange rate for euros?", "exchange_rate"),
        ("How do I change my PIN?", "change_pin"),
        ("My transfer is still pending.", "pending_transfer"),
    ]

    print("\nPriority Detection Test:\n")
    for text, intent in tests:
        result = detector.detect(text, intent)
        print(f"  Query : {text[:60]}")
        print(f"  Intent: {intent}")
        print(f"  -> {result.priority} | {result.reason}")
        print()

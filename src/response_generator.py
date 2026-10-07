"""
response_generator.py -- Phase 12: Generative AI Response Engine
=================================================================
Objective:
  Generate professional, safe, banking-appropriate customer support
  responses using the Google Gemini API.

Safety rules (non-negotiable):
  - NEVER request passwords, PINs, OTPs, CVV, or full card numbers.
  - NEVER invent bank policies, fees, refund timelines, or guarantees.
  - NEVER fabricate account or transaction information.
  - Always recommend escalation when the issue is sensitive or unclear.
  - Responses must be empathetic, concise, and actionable.

Configuration:
  Set GEMINI_API_KEY in a .env file or as an environment variable.

Usage:
    generator = ResponseGenerator()
    response = generator.generate(
        query="My card was debited but I got no cash",
        intent="cash_withdrawal_not_recognised",
        confidence=0.92,
        priority="HIGH",
        escalate=True,
    )
    print(response.text)
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger

logger = get_logger("response_generator")

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class GeneratedResponse:
    text:          str
    model_used:    str
    generation_time_s: float
    is_fallback:   bool = False


# ─────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional banking customer support assistant.

Your role is to help customers resolve their banking issues.

STRICT SAFETY RULES -- you MUST follow these without exception:
1. NEVER ask for or mention passwords, PINs, OTPs, CVV numbers, or full card/account numbers.
2. NEVER invent specific bank policies, fees, refund amounts, or timelines not provided to you.
3. NEVER guarantee refunds, compensation, or outcomes.
4. NEVER fabricate account details, transaction amounts, or reference numbers.
5. If the issue involves potential fraud, unauthorized transactions, or lost/stolen cards,
   ALWAYS recommend the customer call the bank helpline immediately.

RESPONSE GUIDELINES:
- Be empathetic, professional, and concise (3-5 sentences maximum).
- Acknowledge the customer's issue clearly.
- Provide the most relevant next action they should take.
- If human escalation is required, say so clearly and explain why.
- Use simple, clear language -- avoid jargon.
- Do not repeat the customer's query back to them verbatim.
"""


# ─────────────────────────────────────────────────────────────
# Intent-to-support-context mapping
# Maps predicted intents to helpful context hints for the LLM
# ─────────────────────────────────────────────────────────────

INTENT_CONTEXT = {
    "cash_withdrawal_not_recognised": (
        "The customer reports an ATM cash withdrawal that appears on their statement "
        "but they did not receive the cash. This is a financial discrepancy that "
        "requires urgent investigation."
    ),
    "compromised_card": (
        "The customer reports their card may have been compromised. "
        "This is a security issue requiring immediate card blocking."
    ),
    "lost_or_stolen_card": (
        "The customer has lost their card or it has been stolen. "
        "The card should be blocked immediately."
    ),
    "transaction_charged_twice": (
        "The customer has been charged twice for the same transaction. "
        "This is a billing error requiring investigation."
    ),
    "card_arrival": (
        "The customer is enquiring about the delivery of their new card. "
        "Cards typically take 5-7 business days."
    ),
    "card_payment_not_recognised": (
        "The customer does not recognise a card payment on their statement. "
        "This could indicate fraud or a merchant error."
    ),
    "declined_card_payment": (
        "The customer's card payment was declined. Common reasons include "
        "insufficient funds, daily limits, or security holds."
    ),
    "pending_transfer": (
        "The customer has a transfer that is still pending. "
        "Transfers typically complete within 1-3 business days."
    ),
    "verify_my_identity": (
        "The customer needs to complete identity verification. "
        "This is required for regulatory compliance."
    ),
    "exchange_rate": (
        "The customer is asking about exchange rates. "
        "Rates are determined by the market and may vary."
    ),
    "request_refund": (
        "The customer is requesting a refund. "
        "Refunds depend on the merchant and type of transaction."
    ),
    "top_up_failed": (
        "The customer's top-up attempt failed. "
        "Check the payment source and limits."
    ),
    "passcode_forgotten": (
        "The customer has forgotten their passcode. "
        "They should use the app reset function or contact support."
    ),
}

DEFAULT_CONTEXT = (
    "The customer needs assistance with a banking issue. "
    "Provide helpful guidance and appropriate next steps."
)


def _get_intent_context(intent: str) -> str:
    return INTENT_CONTEXT.get(intent, DEFAULT_CONTEXT)


# ─────────────────────────────────────────────────────────────
# Response generator
# ─────────────────────────────────────────────────────────────

class ResponseGenerator:
    """
    Generates AI-powered customer support responses using Gemini.
    Falls back to a template-based response if the API is unavailable.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.api_key   = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model     = model
        self._client   = None
        self._api_available = False

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=SYSTEM_PROMPT,
                )
                self._api_available = True
                logger.info(f"Gemini API initialised ({self.model})")
            except Exception as e:
                logger.warning(f"Gemini API init failed: {e}. Using fallback responses.")
        else:
            logger.warning("GEMINI_API_KEY not set. Using fallback template responses.")

    def generate(
        self,
        query:      str,
        intent:     str,
        confidence: float,
        priority:   str,
        escalate:   bool,
        sentiment:  str = "Neutral",
    ) -> GeneratedResponse:
        """
        Generate a customer support response.

        Args:
            query     : Original customer query.
            intent    : Predicted intent label.
            confidence: Prediction confidence (0-1).
            priority  : Detected priority (HIGH / MEDIUM / LOW).
            escalate  : Whether the query should be escalated.
            sentiment : Detected sentiment (Positive / Neutral / Negative).
        Returns:
            GeneratedResponse with generated text and metadata.
        """
        t0 = time.perf_counter()

        if self._api_available:
            response = self._generate_with_api(
                query, intent, confidence, priority, escalate, sentiment
            )
        else:
            response = self._generate_fallback(
                query, intent, confidence, priority, escalate
            )

        elapsed = time.perf_counter() - t0
        response.generation_time_s = round(elapsed, 2)
        return response

    def _build_user_prompt(
        self, query: str, intent: str, confidence: float,
        priority: str, escalate: bool, sentiment: str
    ) -> str:
        """Construct the user-turn prompt for the LLM."""
        context = _get_intent_context(intent)
        escalation_note = (
            "ESCALATION REQUIRED: This query should be handled by a human agent "
            "due to low prediction confidence or high-risk nature."
            if escalate else
            "This query can be handled with an automated response."
        )

        return f"""Customer Query: "{query}"

Predicted Banking Intent: {intent.replace("_", " ").title()}
Confidence: {confidence:.0%}
Priority Level: {priority}
Customer Sentiment: {sentiment}
{escalation_note}

Support Context:
{context}

Please generate a professional, empathetic response to the customer.
Keep it concise (3-5 sentences). Do not ask for sensitive credentials.
{"Clearly recommend the customer contact human support and explain why." if escalate else ""}"""

    def _generate_with_api(
        self, query: str, intent: str, confidence: float,
        priority: str, escalate: bool, sentiment: str
    ) -> GeneratedResponse:
        """Generate response using Gemini API."""
        prompt = self._build_user_prompt(
            query, intent, confidence, priority, escalate, sentiment
        )
        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature":     0.3,   # low temperature for factual banking responses
                    "max_output_tokens": 300,
                    "top_p":           0.85,
                },
            )
            return GeneratedResponse(
                text=response.text,
                model_used=self.model,
                generation_time_s=0.0,
                is_fallback=False,
            )
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}. Using fallback.")
            return self._generate_fallback(query, intent, confidence, priority, escalate)

    def _generate_fallback(
        self, query: str, intent: str,
        confidence: float, priority: str, escalate: bool
    ) -> GeneratedResponse:
        """Template-based fallback response when API is unavailable."""
        intent_display = intent.replace("_", " ").title()

        if escalate:
            text = (
                f"Thank you for contacting us regarding your {intent_display} issue. "
                "We understand your concern and want to ensure this is handled correctly. "
                "Due to the nature of your query, we recommend speaking directly with "
                "one of our specialist agents who can access your account details securely. "
                "Please contact our support team via the app or helpline for immediate assistance."
            )
        elif priority == "HIGH":
            text = (
                f"We understand you're experiencing an urgent issue with {intent_display}. "
                "Please take the following immediate steps: "
                "(1) If a transaction is involved, check your account statement carefully. "
                "(2) Contact our support team immediately via the in-app chat or helpline. "
                "(3) Do not share any account credentials with anyone."
            )
        else:
            text = (
                f"Thank you for reaching out about {intent_display}. "
                "We're here to help resolve this for you. "
                "Please check the relevant section in our app for immediate options, "
                "or our support team will be happy to assist you further. "
                "Is there anything else you'd like to know?"
            )

        return GeneratedResponse(
            text=text,
            model_used="fallback-template",
            generation_time_s=0.0,
            is_fallback=True,
        )


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generator = ResponseGenerator()

    test_cases = [
        {
            "query":      "I tried to withdraw money from the ATM and it was debited but I got no cash.",
            "intent":     "cash_withdrawal_not_recognised",
            "confidence": 0.92,
            "priority":   "HIGH",
            "escalate":   True,
            "sentiment":  "Negative",
        },
        {
            "query":      "What is the fee for international transfers?",
            "intent":     "transfer_fee_charged",
            "confidence": 0.88,
            "priority":   "LOW",
            "escalate":   False,
            "sentiment":  "Neutral",
        },
        {
            "query":      "My card has not arrived yet.",
            "intent":     "card_arrival",
            "confidence": 0.95,
            "priority":   "MEDIUM",
            "escalate":   False,
            "sentiment":  "Negative",
        },
    ]

    for tc in test_cases:
        print(f"\nQuery   : {tc['query']}")
        print(f"Intent  : {tc['intent']} ({tc['confidence']:.0%}) | {tc['priority']}")
        result = generator.generate(**tc)
        print(f"Response: {result.text}")
        print(f"Model   : {result.model_used} | Time: {result.generation_time_s:.2f}s")
        print("-" * 70)

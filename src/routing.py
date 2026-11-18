"""CLINC150 intent-to-team routing, ELIZA-style replies, and OOS fallback."""

from __future__ import annotations

SMALL_TALK = {
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Hello! How can I help you today?",
    "hello": "Hello! How can I help you today?",
    "hi": "Hi! What can I help you with?",
    "hey": "Hey! How can I help you today?",
    "thank you": "You're welcome. Is there anything else I can help with?",
    "thanks": "You're welcome. Is there anything else I can help with?",
}

# CLINC150 labels are named as <domain>_<intent>. Multiword domains use underscores.
DOMAIN_TEAMS = {
    "auto_and_commute": "Auto and Commute Team",
    "kitchen_and_dining": "Kitchen and Dining Team",
    "credit_cards": "Credit Card Team",
    "banking": "Banking Team",
    "home": "Home Team",
    "travel": "Travel Team",
    "utility": "Utilities Team",
    "work": "Work Services Team",
    "small_talk": "Customer Care",
    "general": "General Support",
}


def small_talk_reply(text: str) -> str | None:
    return SMALL_TALK.get(" ".join(text.lower().strip().split()))


def route_for(intent: str, confidence: float, threshold: float = 0.45) -> str:
    normalized = intent.lower().replace("-", "_")
    if normalized in {"oos", "out_of_scope", "out-of-scope"} or confidence < threshold:
        return "Human Agent"
    for domain in sorted(DOMAIN_TEAMS, key=len, reverse=True):
        if normalized == domain or normalized.startswith(domain + "_"):
            return DOMAIN_TEAMS[domain]
    return "Human Agent"

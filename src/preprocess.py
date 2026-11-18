"""Lightweight English normalization and rule-based helpers."""

from __future__ import annotations

import re
from difflib import get_close_matches

MODEL_CODE = re.compile(r"\b[A-Z]{1,5}[- ]?\d{3,6}[A-Z]?\b", re.IGNORECASE)
NON_TEXT = re.compile(r"[^\w\sçğıöşüÇĞİÖŞÜ-]", re.UNICODE)
REPEATED = re.compile(r"(.)\1{2,}")

# Keep this list small and explicit; extend only with documented domain vocabulary.
PRODUCT_TERMS = (
    "dishwasher", "washing machine", "refrigerator", "fridge", "oven", "dryer", "microwave",
)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = REPEATED.sub(r"\1\1", text)
    text = NON_TEXT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_model_codes(text: str) -> list[str]:
    return MODEL_CODE.findall(text.upper())


def correct_product_terms(text: str) -> str:
    """Correct close spelling variants only against a small product lexicon."""
    words = text.split()
    for i in range(len(words)):
        for width in (1, 2, 3):
            phrase = words[i : i + width]
            if not phrase:
                continue
            candidate = " ".join(phrase)
            matches = get_close_matches(candidate, PRODUCT_TERMS, n=1, cutoff=0.82)
            if matches:
                words[i : i + width] = matches[0].split()
                break
    return " ".join(words)


def preprocess_message(text: str) -> tuple[str, list[str]]:
    codes = extract_model_codes(text)
    cleaned = normalize(text)
    cleaned = correct_product_terms(cleaned)
    return cleaned, codes

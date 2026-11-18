"""Shared inference and evaluation helpers for the four model ladder."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from gensim.models import FastText
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .preprocess import preprocess_message

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BASE_MODEL = "distilbert-base-uncased"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_labels() -> list[str]:
    return json.loads((ARTIFACTS / "labels.json").read_text(encoding="utf-8"))


def mean_fasttext(text: str, fasttext: FastText) -> np.ndarray:
    words = tokens(text)
    if not words:
        return np.zeros(fasttext.vector_size, dtype=np.float32)
    return np.mean([fasttext.wv[w] for w in words], axis=0).astype(np.float32)


class NeuralClassifier(torch.nn.Module):
    def __init__(self, input_size: int, num_labels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictors:
    def __init__(self) -> None:
        self.labels = load_labels()
        self.baseline = joblib.load(ARTIFACTS / "tfidf_logreg.joblib")
        self.fasttext = FastText.load(str(ARTIFACTS / "fasttext.model"))
        neural_data = torch.load(ARTIFACTS / "feedforward.pt", map_location="cpu", weights_only=True)
        self.neural = NeuralClassifier(neural_data["input_size"], len(self.labels))
        self.neural.load_state_dict(neural_data["state_dict"])
        self.neural.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS / "transformer")
        self.transformer = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS / "transformer").cpu().eval()

        quant_dir = ARTIFACTS / "transformer_int8"
        quant_config = AutoConfig.from_pretrained(quant_dir)
        self.quantized = AutoModelForSequenceClassification.from_config(quant_config).cpu().eval()
        supported_engines = torch.backends.quantized.supported_engines
        if "qnnpack" in supported_engines:
            torch.backends.quantized.engine = "qnnpack"
        elif "fbgemm" in supported_engines:
            torch.backends.quantized.engine = "fbgemm"
        else:
            raise RuntimeError(f"No dynamic quantization engine available: {supported_engines}")
        self.quantized = torch.ao.quantization.quantize_dynamic(
            self.quantized, {torch.nn.Linear}, dtype=torch.qint8
        )
        state = torch.load(quant_dir / "model_int8.pt", map_location="cpu", weights_only=True)
        self.quantized.load_state_dict(state)
        self.quantized.eval()

    def predict(self, text: str) -> dict[str, tuple[str, float]]:
        cleaned, _ = preprocess_message(text)
        results: dict[str, tuple[str, float]] = {}

        p = self.baseline.predict_proba([cleaned])[0]
        i = int(np.argmax(p)); results["TF-IDF + Logistic Regression"] = (self.labels[i], float(p[i]))

        vector = torch.from_numpy(mean_fasttext(cleaned, self.fasttext)).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(self.neural(vector), dim=-1)[0]
        i = int(probs.argmax()); results["FastText + Feedforward"] = (self.labels[i], float(probs[i]))

        batch = self.tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=96)
        with torch.inference_mode():
            probs = torch.softmax(self.transformer(**batch).logits, dim=-1)[0]
        i = int(probs.argmax()); results["DistilBERT"] = (self.labels[i], float(probs[i]))

        with torch.inference_mode():
            probs = torch.softmax(self.quantized(**batch).logits, dim=-1)[0]
        i = int(probs.argmax()); results["DistilBERT INT8"] = (self.labels[i], float(probs[i]))
        return results

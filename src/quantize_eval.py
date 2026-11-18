"""Quantize an already fine-tuned checkpoint and finish the shared evaluation."""

from __future__ import annotations

import json
import copy
from pathlib import Path

import joblib
import numpy as np
import torch
from gensim.models import FastText
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .models import NeuralClassifier, mean_fasttext
from .preprocess import preprocess_message
from .train import ARTIFACTS, QUANT_DIR, read_split, score, size_mb, transformer_predict


def choose_quant_engine() -> str:
    supported = torch.backends.quantized.supported_engines
    for engine in ("qnnpack", "fbgemm"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No dynamic quantization engine available: {supported}")


def main() -> None:
    engine = choose_quant_engine()
    label_names = json.loads((ARTIFACTS / "labels.json").read_text(encoding="utf-8"))
    labels = set(label_names)
    x_test, y_test = read_split("test")

    fp_model = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS / "transformer").cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS / "transformer")
    quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp_model), {torch.nn.Linear}, dtype=torch.qint8)
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    fp_model.config.save_pretrained(QUANT_DIR)
    tokenizer.save_pretrained(QUANT_DIR)
    torch.save(quantized.state_dict(), QUANT_DIR / "model_int8.pt")
    print(f"Saved dynamic INT8 model using {engine} backend.")

    baseline = joblib.load(ARTIFACTS / "tfidf_logreg.joblib")
    fasttext = FastText.load(str(ARTIFACTS / "fasttext.model"))
    nn_blob = torch.load(ARTIFACTS / "feedforward.pt", map_location="cpu", weights_only=True)
    nn_model = NeuralClassifier(nn_blob["input_size"], len(label_names))
    nn_model.load_state_dict(nn_blob["state_dict"]); nn_model.eval()

    def pred_baseline(text: str):
        cleaned, _ = preprocess_message(text); p = baseline.predict_proba([cleaned])[0]; i = int(np.argmax(p))
        return str(baseline.classes_[i]), float(p[i])

    def pred_nn(text: str):
        cleaned, _ = preprocess_message(text)
        x = torch.from_numpy(mean_fasttext(cleaned, fasttext)).unsqueeze(0)
        with torch.inference_mode(): p = torch.softmax(nn_model(x), dim=-1)[0]
        i = int(p.argmax()); return label_names[i], float(p[i])

    def pred_quant(text: str):
        cleaned, _ = preprocess_message(text)
        batch = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=96)
        with torch.inference_mode(): p = torch.softmax(quantized(**batch).logits, dim=-1)[0]
        i = int(p.argmax()); return label_names[i], float(p[i])

    results = []
    for name, fn, paths in (
        ("TF-IDF + Logistic Regression", pred_baseline, [ARTIFACTS / "tfidf_logreg.joblib"]),
        ("FastText + Feedforward", pred_nn, [ARTIFACTS / "fasttext.model", ARTIFACTS / "feedforward.pt"]),
        ("DistilBERT", lambda text: transformer_predict(fp_model, tokenizer, label_names, text), [ARTIFACTS / "transformer"]),
        ("DistilBERT INT8", pred_quant, [QUANT_DIR]),
    ):
        row = score(name, fn, x_test, y_test, sorted(labels))
        row["model_size_mb"] = size_mb(paths)
        results.append(row)
        print(f"Finished {name}")

    output = {
        "train_examples": len(read_split("train")[1]),
        "dev_examples": len(read_split("dev")[1]),
        "test_examples": len(y_test),
        "oos_test_examples": sum(label.lower() in {"oos", "out_of_scope", "out-of-scope"} for label in y_test),
        "quantization_engine": engine,
        "results": results,
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

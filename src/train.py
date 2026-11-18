"""Train/evaluate baseline, FastText MLP, Transformer, and INT8 Transformer."""

from __future__ import annotations

import json
import copy
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .models import BASE_MODEL, NeuralClassifier, mean_fasttext, tokens
from .preprocess import preprocess_message

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
QUANT_DIR = ARTIFACTS / "transformer_int8"
SEED = 15
EPOCHS_NN = 20
EPOCHS_TRANSFORMER = 3
LATENCY_RUNS = 100
OOS_NAMES = {"oos", "out_of_scope", "out-of-scope"}


def read_split(name: str) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    texts = [preprocess_message(row["text"])[0] for row in rows]
    labels = [row["label"] for row in rows]
    return texts, labels


def seed_everything() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def size_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for root in paths for p in ([root] if root.is_file() else root.rglob("*")) if p.is_file()) / (1024**2)


def hardware_description() -> str:
    chip = platform.processor() or f"{platform.system()} {platform.machine()}"
    memory = None
    if platform.system() == "Darwin":
        try:
            info = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True, timeout=10)
            chip_line = next((line.strip().split(":", 1)[1].strip() for line in info.splitlines() if line.strip().startswith("Chip:")), None)
            memory = next((line.strip().split(":", 1)[1].strip() for line in info.splitlines() if line.strip().startswith("Memory:")), None)
            if chip_line:
                chip = chip_line
        except (OSError, subprocess.SubprocessError):
            pass
    memory_note = f"; {memory} system memory" if memory else ""
    return f"{chip} CPU; {torch.get_num_threads()} CPU threads{memory_note}"


def score(model_name: str, predict_one, x_test: list[str], y_test: list[str], labels: list[str]) -> dict:
    pred, conf, elapsed = [], [], []
    for text in x_test:
        start = time.perf_counter(); label, confidence = predict_one(text); elapsed.append((time.perf_counter() - start) * 1000)
        pred.append(label); conf.append(confidence)
    oos_labels = [lab for lab in labels if lab.lower() in OOS_NAMES]
    oos_recall = recall_score(y_test, pred, labels=oos_labels, average="macro", zero_division=0) if oos_labels else None
    # Warm-up separately; report mean over at least 100 timed single-message runs.
    for text in x_test[:3]:
        predict_one(text)
    timings = []
    for i in range(LATENCY_RUNS):
        start = time.perf_counter(); predict_one(x_test[i % len(x_test)]); timings.append((time.perf_counter() - start) * 1000)
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "oos_recall": float(oos_recall) if oos_recall is not None else None,
        "model_size_mb": None,
        "latency_ms_mean_100_runs": float(np.mean(timings)),
        "latency_hardware": hardware_description(),
    }


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(texts, truncation=True, max_length=96)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {key: self.encodings[key][i] for key in self.encodings} | {"labels": self.labels[i]}


def transformer_collate(tokenizer):
    def collate(items: list[dict]) -> dict:
        labels = torch.tensor([item.pop("labels") for item in items], dtype=torch.long)
        return tokenizer.pad(items, padding=True, return_tensors="pt") | {"labels": labels}
    return collate


def transformer_predict(model, tokenizer, label_names: list[str], text: str) -> tuple[str, float]:
    clean, _ = preprocess_message(text)
    batch = tokenizer(clean, return_tensors="pt", truncation=True, max_length=96)
    with torch.inference_mode():
        p = torch.softmax(model(**batch).logits, -1)[0]
    i = int(p.argmax())
    return label_names[i], float(p[i])


def main() -> None:
    seed_everything(); ARTIFACTS.mkdir(parents=True, exist_ok=True)
    x_train, y_train = read_split("train")
    x_dev, y_dev = read_split("dev")
    x_test, y_test = read_split("test")
    labels = sorted(set(y_train))
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    (ARTIFACTS / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    # Classical baseline: word and character TF-IDF features + logistic regression.
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1, max_features=120000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", C=4.0)),
    ])
    baseline.fit(x_train, y_train)
    joblib.dump(baseline, ARTIFACTS / "tfidf_logreg.joblib")
    def pred_base(text):
        p = baseline.predict_proba([preprocess_message(text)[0]])[0]; i = int(np.argmax(p)); return baseline.classes_[i], float(p[i])

    # FastText trained only on the training utterances; average word vectors feed an SGD MLP.
    ft = FastText(sentences=[tokens(t) for t in x_train], vector_size=100, window=3, min_count=1,
                  bucket=50_000, workers=1, sg=1, epochs=20, seed=SEED)
    ft.save(str(ARTIFACTS / "fasttext.model"))
    train_vec = np.stack([mean_fasttext(t, ft) for t in x_train])
    dev_vec = np.stack([mean_fasttext(t, ft) for t in x_dev])
    test_vec = np.stack([mean_fasttext(t, ft) for t in x_test])
    y_train_id = torch.tensor([label_to_id[x] for x in y_train])
    y_dev_id = torch.tensor([label_to_id[x] for x in y_dev])
    y_test_id = torch.tensor([label_to_id[x] for x in y_test])
    nn_model = NeuralClassifier(train_vec.shape[1], len(labels))
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    xtr = torch.from_numpy(train_vec)
    for epoch in range(EPOCHS_NN):
        nn_model.train()
        order = torch.randperm(len(xtr))
        for indices in order.split(128):
            optimizer.zero_grad(); logits = nn_model(xtr[indices]); loss = loss_fn(logits, y_train_id[indices]); loss.backward(); optimizer.step()
    nn_model.eval()
    torch.save({"input_size": int(train_vec.shape[1]), "state_dict": nn_model.state_dict()}, ARTIFACTS / "feedforward.pt")
    def pred_nn(text):
        v = torch.from_numpy(mean_fasttext(preprocess_message(text)[0], ft)).unsqueeze(0)
        with torch.inference_mode(): p = torch.softmax(nn_model(v), -1)[0]
        i = int(p.argmax()); return labels[i], float(p[i])

    # Transformer fine-tuning. Train on the official train split; dev split remains validation only.
    model_dir = ARTIFACTS / "transformer"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    transformer = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), id2label=id_to_label, label2id=label_to_id
    )
    train_ds = TextDataset(x_train, [label_to_id[x] for x in y_train], tokenizer)
    dev_ds = TextDataset(x_dev, [label_to_id[x] for x in y_dev], tokenizer)
    collate = transformer_collate(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate)
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    transformer.to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=2e-5)
    for epoch in range(EPOCHS_TRANSFORMER):
        transformer.train(); total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(); out = transformer(**batch); out.loss.backward(); optimizer.step(); total_loss += float(out.loss.detach())
        transformer.eval(); correct = total = 0
        with torch.inference_mode():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}; pred = transformer(**{k:v for k,v in batch.items() if k != "labels"}).logits.argmax(-1)
                correct += int((pred == batch["labels"]).sum()); total += len(pred)
        print(f"Transformer epoch {epoch + 1}/{EPOCHS_TRANSFORMER}: loss={total_loss / max(1,len(train_loader)):.4f}, dev_accuracy={correct / max(1,total):.4f}")
    transformer.cpu().save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)

    # Dynamic INT8 quantization of the fine-tuned Transformer for CPU inference.
    supported_engines = torch.backends.quantized.supported_engines
    if "qnnpack" in supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    else:
        raise RuntimeError(f"No dynamic quantization engine available: {supported_engines}")
    quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(transformer.cpu().eval()), {torch.nn.Linear}, dtype=torch.qint8)
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    transformer.config.save_pretrained(QUANT_DIR)
    tokenizer.save_pretrained(QUANT_DIR)
    torch.save(quantized.state_dict(), QUANT_DIR / "model_int8.pt")

    results = []
    r = score("TF-IDF + Logistic Regression", pred_base, x_test, y_test, labels)
    r["model_size_mb"] = size_mb([ARTIFACTS / "tfidf_logreg.joblib"]); results.append(r)
    r = score("FastText + Feedforward", pred_nn, x_test, y_test, labels)
    r["model_size_mb"] = size_mb([ARTIFACTS / "fasttext.model", ARTIFACTS / "feedforward.pt"]); results.append(r)
    transformer.cpu().eval()
    predict_transformer = lambda text: transformer_predict(transformer, tokenizer, labels, text)
    r = score("DistilBERT", predict_transformer, x_test, y_test, labels)
    r["model_size_mb"] = size_mb([model_dir]); results.append(r)
    def pred_quant(text):
        clean, _ = preprocess_message(text); batch = tokenizer(clean, return_tensors="pt", truncation=True, max_length=96)
        with torch.inference_mode(): p = torch.softmax(quantized(**batch).logits, -1)[0]
        i = int(p.argmax()); return labels[i], float(p[i])
    r = score("DistilBERT INT8", pred_quant, x_test, y_test, labels)
    r["model_size_mb"] = size_mb([QUANT_DIR]); results.append(r)
    out = {"test_examples": len(y_test), "oos_test_examples": sum(y.lower() in OOS_NAMES for y in y_test), "results": results}
    (ARTIFACTS / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

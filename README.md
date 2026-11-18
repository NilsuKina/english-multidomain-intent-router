# English Multi-Domain Intent Router

SWE015 homework project (Use Case 2: Customer Support Ticket Router, adapted to English and CLINC150). An interactive Streamlit app that classifies a message into a CLINC150 intent, shows confidence, routes it to the relevant domain team, and sends low-confidence or out-of-scope messages to a human agent. The predictions of four models are shown side by side, and the quantized Transformer (DistilBERT INT8) is one of the live predictors.

- Presentation video: <ADD VIDEO LINK>
- Measured results: [RESULTS.md](RESULTS.md)
- Dataset source, license, citation: [DATA.md](DATA.md)

This project uses CLINC150 as a multi-domain assistant benchmark, not as an appliance-support dataset. The mapping from CLINC150 domains to demo teams is in `src/routing.py`.

## Model ladder

| # | Model | Area |
|---|-------|------|
| 1 | TF-IDF (word 1-2 grams) + logistic regression | Classical NLP baseline |
| 2 | FastText mean vectors + feedforward network (SGD, backpropagation) | Neural network |
| 3 | Fine-tuned DistilBERT sequence classifier | Transformer |
| 4 | DistilBERT with dynamic INT8 quantization of `Linear` layers | Efficient model |

All four are evaluated on the same official CLINC150 test split (5,500 messages, 1,000 out-of-scope).

## Setup

Python 3.13 is recommended (tested on an Apple M2; other 3.10+ versions should work).

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick start (single run command)

Trained model files are not stored in this repository (they are large). They are hosted on Hugging Face Hub and downloaded automatically:

```bash
python -m src.download_artifacts && streamlit run app.py
```

The first command downloads the trained artifacts (about 450 MB) into `artifacts/`; the second starts the app at http://localhost:8501.

Note on the INT8 model: dynamic quantization uses the `qnnpack` backend when available and falls back to `fbgemm`. The downloaded INT8 weights were produced with `qnnpack`. If loading fails on your machine, rebuild the INT8 model from the downloaded FP32 checkpoint with:

```bash
python -m src.quantize_eval
```

## Reproduce everything from scratch (optional)

Use this path instead of the quick start to retrain and re-measure all four models. Training DistilBERT can take a long time on CPU.

```bash
python -m src.data    # downloads official CLINC150, writes data/processed/{train,dev,test}.jsonl
python -m src.train   # trains all four models, writes artifacts/ and artifacts/results.json
streamlit run app.py
```

`src.data` keeps the original split boundaries and out-of-scope labels. `src.train` evaluates every model on the same test split and records accuracy, out-of-scope recall, model size, and mean single-message latency over 100 timed runs, together with a description of the hardware. DistilBERT weights are downloaded from Hugging Face on first use.

`src.quantize_eval` re-quantizes an already fine-tuned checkpoint and repeats the shared evaluation without retraining.

## Repository layout

```
app.py                 Streamlit application
src/data.py            CLINC150 download and split preparation
src/preprocess.py      regex normalization, model-code extraction, edit-distance term correction
src/routing.py         domain-to-team mapping, small-talk rules, 45% confidence fallback
src/models.py          the four predictors used by the app
src/train.py           training, quantization, shared evaluation
src/quantize_eval.py   re-quantize and re-evaluate an existing checkpoint
src/download_artifacts.py   fetch trained artifacts from Hugging Face Hub
```

## Limitations

- Out-of-scope recall is low for all four models; the strongest model still accepts many unknown messages as in-scope.
- The 45% fallback threshold has not been tuned on validation data.
- The product-name lexicon is a small technique demonstration and is not part of CLINC150.

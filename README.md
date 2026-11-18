# English Multi-Domain Intent Router

SWE015 homework project: an interactive English-language assistant that classifies a message into a CLINC150 intent, shows confidence, routes it to the relevant domain team, and sends low-confidence or out-of-scope messages to a human agent. We chose the multi-domain assistant framing so that the application matches the instructor-suggested CLINC150 corpus.

## Dataset

We are following the assignment's CLINC150 suggestion. It is English and includes out-of-scope examples. See [DATA.md](DATA.md) for source, license, and citation. This project uses the corpus as a multi-domain assistant benchmark, not as an appliance-support dataset.

## Setup

Python 3.13 is recommended for the currently resolved ARM Mac package wheels.

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Download and prepare the data

```bash
python -m src.data
```

This downloads the official full CLINC150 JSON and writes `data/processed/{train,dev,test}.jsonl`. Original split boundaries and out-of-scope labels are preserved. Data files are excluded from Git.

## Train and measure the model ladder

```bash
python -m src.train
```

This trains the TF-IDF/logistic-regression baseline, FastText + SGD feedforward network, fine-tuned DistilBERT, and a dynamic INT8 version of the fine-tuned Transformer. It evaluates them on the same test split and writes `artifacts/results.json` with accuracy, out-of-scope recall, model size, and mean single-message latency over at least 100 timed runs. The hardware description is recorded with the measurements.

Training downloads the DistilBERT checkpoint from Hugging Face on first use. Training can take a while, especially on CPU.

## Run the interactive demo

```bash
streamlit run app.py
```

The app displays all four model predictions and routes low-confidence/out-of-scope messages to a human agent. The mapping from CLINC150 domains to demo teams is documented in `src/routing.py`.

## Project checklist

- [x] Chosen the instructor-suggested English dataset
- [x] Added official data preparation and citations
- [x] Added regex normalization, model-code extraction, product-name correction, small-talk reply, and human fallback helpers
- [x] TF-IDF + logistic regression baseline implementation
- [x] FastText embeddings + feedforward neural network trained with SGD/backpropagation
- [x] DistilBERT fine-tuning implementation
- [x] Dynamic INT8 quantization and quantized-model inference implementation
- [x] Shared test-set evaluation and 100-run latency measurement implementation
- [x] Streamlit application implementation
- [x] Download official data and Transformer weights, train, and measure all four models
- [ ] Adjust routing threshold on validation results; prepare slides and narrated demo

The slide plan is in [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md), measured values are summarized in [RESULTS.md](RESULTS.md), and the two-minute demo script is in [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

# Model comparison results

All four models were evaluated on the same official CLINC150 test split (5,500 examples, including 1,000 out-of-scope examples). Latency is the mean of 100 timed single-message inferences after warm-up. The measured hardware was an Apple M2 MacBook Air with 8 GB RAM; inference ran on CPU with four PyTorch CPU threads and no MPS/GPU acceleration.

| Model | Accuracy | OOS recall | Model package size (MB) | Mean latency (ms) |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 76.60% | 14.20% | 37.23 | 0.40 |
| FastText + Feedforward | 69.51% | 19.70% | 23.32 | 0.20 |
| DistilBERT | 84.31% | 37.70% | 256.77 | 11.81 |
| DistilBERT INT8 | 84.85% | 42.00% | 133.31 | 6.63 |

The INT8 model was dynamically quantized with the QNNPACK backend. In this run it improved test accuracy by 0.55 percentage points and OOS recall by 4.30 points compared with the full-precision DistilBERT. Its saved package was about 48% smaller and its mean measured latency about 44% lower. These are observations from this run, not guaranteed effects of quantization in general.

The baseline and FastText network were much faster and smaller, but had lower intent accuracy and OOS recall. The results show that the benchmark's OOS class remains challenging for all four approaches.

One held-out OOS failure: “how much has the dow changed today” was assigned `spending_history` at 24.12% confidence by DistilBERT INT8. The current 45% confidence fallback sends it to a human agent.

Machine-readable measurements are in the local `artifacts/results.json`; model checkpoints and downloaded data are excluded from Git.

# Two-minute demo script

Keep the recording under 2:00. Read the live model outputs from the app; do not narrate an expected result before it appears.

| Time | Show and say |
|---|---|
| 0:00–0:15 | “This is an English multi-domain intent router. It predicts a user’s intent, shows confidence, and sends uncertain or out-of-scope messages to a human.” |
| 0:15–0:40 | In the live app, enter a clear in-scope message from the CLINC150 test examples. Read the predicted intent, confidence, and routed team. Mention the small-talk rule if using a greeting. |
| 0:40–1:05 | Enter: “how much has the dow changed today”. Show that the INT8 model’s top label is `spending_history` at 24.12% confidence and the current 45% rule routes it to a human agent. This is a real held-out OOS failure caught by the confidence fallback. |
| 1:05–1:30 | Show the architecture slide: input, normalization/rules, four predictors, confidence/OOS check, team or human fallback. |
| 1:30–1:50 | Show the measured results table and say the INT8 model reached 84.85% accuracy, 42.00% OOS recall, and 6.63 ms mean latency on the Apple M2 CPU. |
| 1:50–2:00 | “The quantized model was smaller and faster in this run. OOS handling is still limited, and a next step is to tune the fallback threshold on validation data.” |

Record the app live with at least two inputs. Replace the first message if it is too slow or the result needs explanation. Upload as unlisted YouTube or anyone-with-link Drive, then add the link to the cover and submission form.

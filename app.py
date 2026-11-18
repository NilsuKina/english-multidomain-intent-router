from __future__ import annotations

import streamlit as st

from src.models import ARTIFACTS, Predictors
from src.preprocess import extract_model_codes
from src.routing import route_for, small_talk_reply

st.set_page_config(page_title="English Multi-Domain Intent Router", page_icon="↗", layout="wide")
st.title("English Multi-Domain Intent Router")
st.caption("CLINC150 intent classification · low-confidence and out-of-scope messages go to a human agent")


@st.cache_resource
def load_predictors():
    return Predictors()


if not (ARTIFACTS / "results.json").exists():
    st.info("Train the model ladder first with `python -m src.train`; then restart this app.")
    st.stop()

try:
    predictors = load_predictors()
except Exception as exc:
    st.error(f"Could not load trained artifacts: {exc}")
    st.stop()

with st.form("message_form"):
    message = st.text_area("Customer message", placeholder="Example: I need to dispute a charge on my credit card", height=110)
    submitted = st.form_submit_button("Classify and route", type="primary")

if submitted and message.strip():
    with st.spinner("Running all four models..."):
        predictions = predictors.predict(message)
    rule_reply = small_talk_reply(message)
    codes = extract_model_codes(message)
    if rule_reply:
        st.info(f"Rule-based small-talk reply: {rule_reply}")
    if codes:
        st.caption("Extracted product/model code(s): " + ", ".join(codes))

    rows = []
    for model_name, (intent, confidence) in predictions.items():
        rows.append({
            "Model": model_name,
            "Predicted intent": intent,
            "Confidence": f"{confidence:.1%}",
            "Routed team": route_for(intent, confidence),
        })
    st.subheader("Model comparison")
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("Routing threshold is currently 45%; edit `route_for` after reviewing validation results.")
elif submitted:
    st.warning("Enter a message first.")

with st.expander("Dataset and scope"):
    st.write("This prototype uses CLINC150, an English intent benchmark with 150 in-scope intents and out-of-scope examples. The team mapping follows the benchmark's general assistant domains.")

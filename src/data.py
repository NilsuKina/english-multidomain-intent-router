"""Download the official CLINC150 full split and prepare model-ready files."""

from __future__ import annotations

import json
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SOURCE = RAW / "data_full.json"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        print(f"Downloading official CLINC150 data from {URL}")
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        SOURCE.write_bytes(response.content)

    dataset = json.loads(SOURCE.read_text(encoding="utf-8"))
    # CLINC stores OOS records in separate keys; merge them into the matching split.
    splits = (
        ("train", dataset["train"] + dataset["oos_train"]),
        ("dev", dataset["val"] + dataset["oos_val"]),
        ("test", dataset["test"] + dataset["oos_test"]),
    )
    for split, records in splits:
        out = PROCESSED / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for text, label in records:
                f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} examples -> {out}")

    print("Merged the supplied OOS records into each matching split; original split boundaries are intact.")


if __name__ == "__main__":
    main()

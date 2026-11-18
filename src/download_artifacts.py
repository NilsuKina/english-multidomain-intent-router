"""Download the trained model artifacts (about 450 MB) from Hugging Face Hub into artifacts/."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

# TODO: replace with your own Hub repository (model repo that contains the contents of artifacts/).
REPO_ID = "YOUR_HF_USERNAME/english-multidomain-intent-router-artifacts"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=REPO_ID, repo_type="model", local_dir=ARTIFACTS)
    print(f"Artifacts downloaded to {path}")


if __name__ == "__main__":
    main()

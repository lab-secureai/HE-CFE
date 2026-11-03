"""Load the frozen seed-55 oracle, preprocessor and projector from a bundle.

Mirrors cell 7 (handoff audit) and cell 8 (HE parameter export) of the
canonical notebooks in `notebooks/ciphertext/he_population/<ds>/`.  Nothing is
refit or retrained; every handoff file is checksum-verified and the stored
seed-55 test metrics are recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import plaintext_cfe_pipeline as pipe


def load_frozen(dataset: str, bundle: str | Path, device: str = "cpu") -> dict:
    bundle = Path(bundle)
    handoff = bundle / "handoff"
    manifest = json.loads((handoff / "training_handoff_manifest.json").read_text())
    acceptance = json.loads((handoff / "training_acceptance.json").read_text())
    if manifest.get("protocol") != "Q1_V55_MLP_ONLY_3SEED_PAPER" or manifest.get("dataset") != dataset:
        raise RuntimeError("Unexpected training handoff manifest")
    if not acceptance.get("accepted"):
        raise RuntimeError("Training handoff was not accepted")
    for name, expected in manifest["files_sha256"].items():
        if pipe.sha256_file(handoff / name) != expected:
            raise RuntimeError(f"Handoff checksum mismatch: {name}")

    data_path = pipe.resolve_input(dataset, bundle / "raw")
    if pipe.sha256_file(data_path) != manifest["source_csv_sha256"]:
        raise RuntimeError("Raw CSV differs from the MLP training input")
    raw = pipe.load_dataset(dataset, data_path)
    frozen = joblib.load(handoff / "preprocessor_and_split.joblib")
    split, preprocessor = frozen["split"], frozen["preprocessor"]
    y_all = raw.frame[raw.target_col].to_numpy(np.int8)
    prepared = pipe.PreparedData(
        raw=raw, split=split, preprocessor=preprocessor,
        x_train=preprocessor.transform(raw.frame.iloc[split.train_idx]),
        y_train=y_all[split.train_idx],
        x_valid=preprocessor.transform(raw.frame.iloc[split.valid_idx]),
        y_valid=y_all[split.valid_idx],
        x_test=preprocessor.transform(raw.frame.iloc[split.test_idx]),
        y_test=y_all[split.test_idx],
        feature_names=list(frozen["feature_names"]),
    )
    if prepared.x_train.shape[1] != manifest["processed_inputs"]:
        raise AssertionError("Processed input dimension differs from handoff")

    checkpoint = torch.load(handoff / "classifier_seed55.pt", map_location="cpu",
                            weights_only=False)
    cfg = checkpoint["classifier_config"]
    oracle = pipe.HEPolynomialMLP(checkpoint["input_dim"], cfg["h1"], cfg["h2"],
                                  cfg["activation_kind"], cfg["alpha"], cfg["dropout"])
    oracle.load_state_dict(checkpoint["state_dict"])
    oracle.eval()
    stored = pd.read_csv(handoff / "classifier_metrics_by_seed.csv")
    stored = stored[stored.seed.eq(55)].iloc[0]
    recomputed = pipe.classification_metrics(
        prepared.y_test, pipe.probabilities(oracle, prepared.x_test, device), .5)
    for metric in ("accuracy", "f1", "roc_auc", "pr_auc"):
        if not np.isclose(recomputed[metric], stored[metric], atol=1e-8):
            raise AssertionError(f"Frozen classifier identity check failed: {metric}")

    state = checkpoint["state_dict"]
    he_parameters = {
        "fc1_weight": state["fc1.weight"].detach().cpu().numpy(),
        "fc1_bias": state["fc1.bias"].detach().cpu().numpy(),
        "fc2_weight": state["fc2.weight"].detach().cpu().numpy(),
        "fc2_bias": state["fc2.bias"].detach().cpu().numpy(),
        "fc3_weight": state["fc3.weight"].detach().cpu().numpy(),
        "fc3_bias": state["fc3.bias"].detach().cpu().numpy(),
        "activation_kind": str(cfg["activation_kind"]),
        "activation_alpha": float(cfg["alpha"]),
    }
    return {
        "dataset": dataset, "prepared": prepared, "oracle": oracle,
        "projector": pipe.DomainProjector(prepared), "he_parameters": he_parameters,
        "manifest": manifest, "test_metrics": recomputed,
        "operating_point": json.loads((bundle / "operating_point.json").read_text()),
        "outer_queries": pd.read_csv(bundle / "canonical" / "outer_test_query_manifest.csv"),
    }

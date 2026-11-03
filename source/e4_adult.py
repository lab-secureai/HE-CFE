"""E4: Adult Income plaintext CFE experiment with the unchanged HE-CFE search.

Two stages mirror the canonical training and Gaussian-benchmark notebooks:

  train      fit preprocessor on outer-development train rows, train the
             MLP for seeds 55/66/77 (seed 55 frozen as CFE oracle), write a
             handoff identical in layout to `results/training_v55_3seed/<ds>`.
  benchmark  inner-validation P/R selection (12 factuals per direction, seed
             1707) and outer benchmark (100 per direction, seed 707) of the
             four forward methods plus Wachter, P in {64,128,256,512}, round
             snapshots {2,4,8,16}, K=10, margin 0.10, search seed 11.

Usage:
  python3 e4_adult.py train --adult-dir <dir with adult.data/test> --output <dir>
  python3 e4_adult.py benchmark --handoff-root <train output> --output <dir>
      [--outer-per-direction 100 --inner-per-direction 12]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

for _path in (Path(__file__).resolve().parent / "vendor", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_path))

import joblib
import numpy as np
import pandas as pd
import torch

import adult_adapter  # noqa: F401  (registers the dataset before use)
import plaintext_cfe_pipeline as pipe
from non_gan_v55_runner import (FORWARD_METHODS, METRICS, benchmark,
                                select_inner_contract, select_inner_queries)

DATASET = "adult"
SEEDS = [55, 66, 77]
CANONICAL_SEED = 55
MAX_EPOCHS = 60
BATCH_SIZE = 1024
POPULATIONS = (64, 128, 256, 512)
ROUND_SNAPSHOTS = (2, 4, 8, 16)
K = 10
SEARCH_SEED = 11


def train(adult_dir: Path, output: Path) -> None:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    handoff = output / "paper_training_handoff"
    handoff.mkdir(exist_ok=True)
    csv = adult_adapter.build_csv(adult_dir, output / "raw" / adult_adapter.FILENAME)
    raw = pipe.load_dataset(DATASET, csv)
    split = pipe.make_splits(raw)
    prepared = pipe.prepare_data(raw, split)
    inputs = prepared.x_train.shape[1]
    print({"rows": len(raw.frame), "inputs": inputs, **split.audit}, flush=True)

    models, histories, audits, rows = {}, {}, {}, []
    for seed in SEEDS:
        model, history, audit = pipe.train_classifier(
            prepared, seed, "cpu", max_epochs=MAX_EPOCHS, patience=8,
            batch_size=BATCH_SIZE)
        models[seed], histories[seed], audits[seed] = model, history, audit
        metrics = pipe.classifier_diagnostics(model, prepared, "cpu", history, output,
                                              prefix=f"classifier_seed{seed}")
        rows.append({**audit, **metrics, "canonical_for_cfe": seed == CANONICAL_SEED})
        print({"seed": seed, "epoch": audit["selected_epoch"],
               "f1": metrics.get("f1"), "roc_auc": metrics.get("roc_auc")}, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(handoff / "classifier_metrics_by_seed.csv", index=False)
    numeric = table.select_dtypes("number")
    numeric.agg(["mean", "std", "min", "max"]).T.to_csv(
        handoff / "classifier_metrics_mean_std.csv")

    joblib.dump({"preprocessor": prepared.preprocessor,
                 "feature_names": prepared.feature_names, "split": split},
                handoff / "preprocessor_and_split.joblib")
    for seed in SEEDS:
        torch.save({"state_dict": models[seed].state_dict(), "input_dim": inputs,
                    "classifier_config": pipe.CLASSIFIER_CONFIGS[DATASET],
                    "classifier_audit": audits[seed],
                    "canonical_for_cfe": seed == CANONICAL_SEED},
                   handoff / f"classifier_seed{seed}.pt")
        histories[seed].to_csv(handoff / f"classifier_seed{seed}_training_history.csv",
                               index=False)
    pd.DataFrame({"position": range(inputs), "processed_feature": prepared.feature_names}
                 ).to_csv(handoff / "feature_encoding_contract.csv", index=False)
    files = sorted(p.name for p in handoff.iterdir() if p.is_file())
    manifest = {
        "protocol": "Q1_V55_MLP_ONLY_3SEED_PAPER", "dataset": DATASET,
        "stage": "classifier_training_only", "device": "cpu",
        "source_csv_sha256": pipe.sha256_file(csv), "source_rows": len(raw.frame),
        "processed_inputs": inputs, "classifier_config": pipe.CLASSIFIER_CONFIGS[DATASET],
        "classifier_seeds": SEEDS, "canonical_classifier_seed": CANONICAL_SEED,
        "max_epochs": MAX_EPOCHS, "patience": 8, "batch_size": BATCH_SIZE,
        "split_audit": split.audit, "encoding_contract": adult_adapter.__doc__,
        "files_sha256": {name: pipe.sha256_file(handoff / name) for name in files},
    }
    (handoff / "training_handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    acceptance = {"accepted": bool(inputs == adult_adapter.EXPECTED_INPUTS
                                   and len(raw.frame) == 48_842),
                  "wall_seconds": time.perf_counter() - started}
    (output / "training_acceptance.json").write_text(json.dumps(acceptance, indent=2))
    print(table[["seed", "selected_epoch", "accuracy", "f1", "roc_auc", "pr_auc"]].to_string())
    print(json.dumps(acceptance))


def load_handoff(root: Path):
    manifest_path = next(root.rglob("training_handoff_manifest.json"))
    handoff = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["files_sha256"].items():
        if pipe.sha256_file(handoff / name) != digest:
            raise RuntimeError(f"Handoff checksum mismatch: {name}")
    csv = next(root.rglob(adult_adapter.FILENAME))
    if pipe.sha256_file(csv) != manifest["source_csv_sha256"]:
        raise RuntimeError("Adult CSV differs from training input")
    raw = pipe.load_dataset(DATASET, csv)
    frozen = joblib.load(handoff / "preprocessor_and_split.joblib")
    split, pre = frozen["split"], frozen["preprocessor"]
    y = raw.frame[raw.target_col].to_numpy(np.int8)
    prepared = pipe.PreparedData(
        raw=raw, split=split, preprocessor=pre,
        x_train=pre.transform(raw.frame.iloc[split.train_idx]), y_train=y[split.train_idx],
        x_valid=pre.transform(raw.frame.iloc[split.valid_idx]), y_valid=y[split.valid_idx],
        x_test=pre.transform(raw.frame.iloc[split.test_idx]), y_test=y[split.test_idx],
        feature_names=list(frozen["feature_names"]))
    checkpoint = torch.load(handoff / f"classifier_seed{CANONICAL_SEED}.pt",
                            map_location="cpu", weights_only=False)
    cfg = checkpoint["classifier_config"]
    oracle = pipe.HEPolynomialMLP(checkpoint["input_dim"], cfg["h1"], cfg["h2"],
                                  cfg["activation_kind"], cfg["alpha"], cfg["dropout"])
    oracle.load_state_dict(checkpoint["state_dict"])
    oracle.eval()
    return prepared, oracle, manifest


def run_benchmark(root: Path, output: Path, outer: int, inner: int) -> None:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    prepared, oracle, manifest = load_handoff(root)
    projector = pipe.DomainProjector(prepared)
    metrics = pipe.classification_metrics(
        prepared.y_test, pipe.probabilities(oracle, prepared.x_test, "cpu"), .5)
    pd.DataFrame([{"dataset": DATASET, "canonical_classifier_seed": CANONICAL_SEED,
                   **{k: metrics[k] for k in ("accuracy", "f1", "roc_auc", "pr_auc")},
                   "immutable_features": ",".join(np.asarray(projector.feature_names)
                                                  [projector.immutable_mask])}]
                 ).to_csv(output / "frozen_mlp_handoff_audit.csv", index=False)

    inner_queries = select_inner_queries(prepared, oracle, "cpu", per_direction=inner,
                                         seed=1707)
    inner_queries.to_csv(output / "inner_validation_query_manifest.csv", index=False)
    _, _, _, inner_direction, _ = benchmark(
        prepared, oracle, projector, inner_queries, output / "inner_validation",
        device="cpu", populations=POPULATIONS, snapshots=ROUND_SNAPSHOTS, k=K,
        seed=SEARCH_SEED, include_wachter=False, robustness_repetitions=4,
        fidelity_samples=0, save_cfe_values=False, checkpoint_every=6)
    contract, _ = select_inner_contract(inner_direction, output)
    print({"selected_inner_contract": contract}, flush=True)

    outer_queries = pipe.select_stratified_bidirectional_queries(
        prepared, oracle, "cpu", per_direction=outer, seed=707)
    outer_queries.to_csv(output / "outer_test_query_manifest.csv", index=False)
    detail, _, _, _, _ = benchmark(
        prepared, oracle, projector, outer_queries, output / "outer_test", device="cpu",
        populations=POPULATIONS, snapshots=ROUND_SNAPSHOTS, k=K, seed=SEARCH_SEED,
        include_wachter=True, robustness_repetitions=16, fidelity_samples=96,
        save_cfe_values=True, checkpoint_every=10)
    selected = detail[detail.method.eq(contract["method"])
                      & detail.population.eq(contract["population"])
                      & detail.rounds.eq(contract["rounds"])]
    selected.to_csv(output / "selected_method_outer_per_factual.csv", index=False)
    selected.groupby("direction")[METRICS].agg(["mean", "std", "count"]).to_csv(
        output / "selected_method_outer_by_direction.csv")
    expected_rows = len(outer_queries) * (len(FORWARD_METHODS) * len(POPULATIONS)
                                          * len(ROUND_SNAPSHOTS) + 1)
    checks = {
        "outer_factuals": int(len(outer_queries)) == 2 * outer,
        "expected_per_factual_rows": len(detail) == expected_rows,
        "inner_only_method_selection": contract["outer_test_used_for_selection"] is False,
    }
    acceptance = {"accepted": all(checks.values()), "checks": checks,
                  "protocol": "HECFE_E4_ADULT_PLAINTEXT", "dataset": DATASET,
                  "selected_inner_contract": contract,
                  "total_wall_seconds": time.perf_counter() - started}
    (output / "benchmark_acceptance.json").write_text(json.dumps(acceptance, indent=2,
                                                                 default=str))
    print(json.dumps(acceptance, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--adult-dir", required=True)
    t.add_argument("--output", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--handoff-root", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--outer-per-direction", type=int, default=100)
    b.add_argument("--inner-per-direction", type=int, default=12)
    args = parser.parse_args()
    if args.command == "train":
        train(Path(args.adult_dir), Path(args.output))
    else:
        run_benchmark(Path(args.handoff_root), Path(args.output),
                      args.outer_per_direction, args.inner_per_direction)


if __name__ == "__main__":
    main()

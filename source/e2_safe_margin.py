"""E7: CKKS-safe acceptance rule for the encrypted GA-div search (R2.W2).

With the Heart+ profile (N=8192, scale 2^29) the decrypted logit differs from
the plaintext one by up to ~0.02, so a candidate accepted at the margin m may
fail m in plaintext (E2: 6 returned CFEs, 23 decision flips).  The safe rule
accepts a candidate only when its decrypted logit clears m + epsilon, where
epsilon bounds the CKKS error.  Quality is still scored at the true margin m.

calibrate  Bound epsilon on the inner validation cohort only (never the outer
           factuals): Gaussian proposals around each inner factual at the
           search's step sizes, projected, scored encrypted and in plaintext;
           epsilon = 2 x the largest absolute error, rounded up to 0.01.
run        E2 GA-div shard with search decisions at m + epsilon (encrypted run
           and plaintext twin alike), metrics and plaintext re-verification at m.

Usage: python3 e2_safe_margin.py calibrate --dataset heartplus --bundle <dir> --output <dir>
       python3 e2_safe_margin.py run --dataset heartplus --bundle <dir> --output <dir>
           --epsilon 0.05 [--workers 4 --shard-index i --num-shards n]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE / "vendor", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd

import e2_gadiv
import e2_encrypted_search as es
import non_gan_v55_runner as runner

METHOD = "gaussian_ga_div_safe"
EPSILON = float(os.environ.get("E2_SAFE_EPSILON", "0"))
CALIBRATION_ROUNDS = (1, 4, 16)  # step sizes 0.12 sqrt(r) used by the search


def _safe(config):
    return dataclasses.replace(config, margin=config.margin + EPSILON)


def encrypted_safe(query, desired, scorer, oracle, projector, config, seed):
    return e2_gadiv.encrypted_ga_div(query, desired, scorer, oracle, projector,
                                     _safe(config), seed)


def search_safe(query, desired, oracle, projector, config, device, seed,
                distribution="gaussian", genetic=False):
    return e2_gadiv.search_distribution(query, desired, oracle, projector,
                                        _safe(config) if genetic else config, device, seed,
                                        distribution=distribution, genetic=genetic)


def process_factual(record: dict) -> dict:
    result = e2_gadiv.process_factual(record)
    for row in result["rows"] + result["cfes"]:
        row["method"] = METHOD
        row["safe_epsilon"] = EPSILON
    return result


es.encrypted_gaussian_ga = encrypted_safe
runner.search_distribution = search_safe
es.process_factual = process_factual


def calibrate(dataset, bundle, output, population):
    import plaintext_cfe_pipeline as pipe
    from frozen_setup import load_frozen
    from he_ckks import HEPopulationScorer

    frozen = load_frozen(dataset, bundle)
    prepared, oracle, projector = frozen["prepared"], frozen["oracle"], frozen["projector"]
    point = frozen["operating_point"]
    scorer = HEPopulationScorer(frozen["he_parameters"], profile_name=point["profile"],
                                cfe_margin=point["margin"])
    inner = runner.select_inner_queries(prepared, oracle, "cpu", per_direction=12, seed=1707)
    rng = np.random.default_rng(2026)
    rows = []
    for record in inner.itertuples():
        query, _ = runner._query_vector(prepared, record)
        for rounds in CALIBRATION_ROUNDS:
            noise = rng.normal(size=(population, query.size)).astype(np.float32)
            candidate = query[None, :] + noise * projector.scale[None, :] * .12 * math.sqrt(rounds)
            candidate = projector.project(candidate, query)
            logit, _ = scorer(candidate)
            plain = pipe.logits_numpy(oracle, candidate, "cpu")
            error = np.abs(np.asarray(logit, float) - plain)
            rows.append({"query_id": int(record.query_id), "rounds_step": rounds,
                         "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
                         "max_abs_plain_logit": float(np.abs(plain).max())})
        print(record.query_id, rows[-1], flush=True)
    table = pd.DataFrame(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "safe_margin_calibration.csv", index=False)
    worst = float(table.max_abs_error.max())
    epsilon = math.ceil(2 * worst * 100) / 100
    summary = {"dataset": dataset, "profile": point["profile"], "cohort": "inner validation",
               "factuals": int(inner.query_id.nunique()), "batches": len(table),
               "population": population, "max_abs_error": worst,
               "p99_batch_max_error": float(table.max_abs_error.quantile(.99)),
               "rule": "epsilon = ceil(2 x max_abs_error, 0.01)", "epsilon": epsilon}
    (output / "safe_margin_calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["calibrate", "run"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-per-direction", type=int, default=None)
    args = parser.parse_args()
    os.environ["E2_GADIV_POPULATION"] = str(args.population)
    e2_gadiv.POPULATION = args.population
    if args.action == "calibrate":
        calibrate(args.dataset, args.bundle, args.output, args.population)
        return
    # Spawned workers re-import this module and read epsilon from the environment.
    os.environ["E2_SAFE_EPSILON"] = str(args.epsilon)
    globals()["EPSILON"] = args.epsilon
    es.run_shard(args.dataset, args.bundle, args.output, args.shard_index, args.num_shards,
                 args.workers,
                 args.limit_per_direction)


if __name__ == "__main__":
    main()

"""Plaintext inner-only trial of client-side plausibility for GA-div (2026-10-30).

Variants at each dataset's operating point:
  base  GA-div unchanged
  A     Q + w * projector.plausibility (5-NN distance to reference data)
  B     immigrants interpolate from the factual to a near reference point of the
        desired class
  AB    both
The reference data is the training split, so the shipped LOF (fit on training
data) is not independent of A and B.  lof_test is fit on the test split, which
the search never reads, with the same LOF settings.

Exploratory: runs on the local library versions, inner cohort only.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE / "vendor", HERE):
    sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

import ga_div_benchmark as bench
import ga_diverse
import non_gan_v55_runner as runner
import plaintext_cfe_pipeline as pipe

POINTS = {"ecg": (64, 4), "heartplus": (64, 16), "mimic": (128, 8), "adult": (64, 16),
          "credit": (64, 16)}
SEARCH_SEED = 11
NEIGHBOURS = 50
VARIANTS = {"base": (0., False), "A_w0.5": (.5, False), "A_w2": (2., False),
            "B": (0., True), "AB_w0.5": (.5, True), "AB_w2": (2., True)}
OUT = HERE.parent / "results" / "plausibility_trial"


def reference_proposal(projector, desired):
    x = projector.reference_x[projector.reference_y == desired]

    def propose(rng, query, count):
        distance = (np.abs(x - query[None, :]) / projector.scale[None, :]).mean(axis=1)
        near = np.argpartition(distance, min(NEIGHBOURS, len(x) - 1))[:NEIGHBOURS]
        chosen = x[rng.choice(near, size=count)]
        step = rng.uniform(.5, 1., size=(count, 1)).astype(np.float32)
        return query[None, :] + step * (chosen - query[None, :])
    return propose


def run(dataset, bundle):
    prepared, oracle, projector, _ = bench.load(dataset, Path(bundle) if bundle else None)
    queries = runner.select_inner_queries(prepared, oracle, "cpu", per_direction=12, seed=1707)
    independent = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=.10, n_jobs=1).fit(
        np.asarray(prepared.x_test[:20000], np.float32))
    population, rounds = POINTS[dataset]
    rows = []
    for record in queries.itertuples(index=False):
        query, _ = runner._query_vector(prepared, record)
        desired = int(record.desired_class)
        seed = SEARCH_SEED + 100_000 * int(record.query_id) + population
        for name, (weight, seeded) in VARIANTS.items():
            config = pipe.SearchConfig(population=population, max_rounds=rounds, k=10, margin=.10,
                                       plausibility_weight=weight)
            cfes, audit, runtime, _ = ga_diverse.search_ga_diverse(
                query, desired, oracle, projector, config, "cpu", seed,
                plausibility_cost=projector.plausibility if weight > 0 else None,
                reference_proposal=reference_proposal(projector, desired) if seeded else None)
            metrics = bench._frozen_set_metrics(
                cfes, query, desired, oracle, projector, config, "cpu", runtime, audit,
                robustness_seed=7_000_000 + seed + rounds, robustness_repetitions=16,
                fidelity_samples=8)
            test_lof = (float((independent.predict(np.asarray(cfes, np.float32)) == 1).mean())
                        if len(cfes) else np.nan)
            rows.append({"dataset": dataset, "variant": name, "query_id": int(record.query_id),
                         "stratum": record.difficulty_stratum, "yield": metrics["valid_cfe_yield_at_k"],
                         "robust": metrics["robust_completeness_at_k"], "proximity": metrics["proximity"],
                         "sparsity": metrics["sparsity"], "diversity": metrics["diversity"],
                         "lof_train": metrics["plausibility_inlier_rate"], "lof_test": test_lof,
                         "plaus_slots_test": (0. if not len(cfes) else test_lof * metrics["valid_cfe_yield_at_k"]),
                         "mean_margin": metrics["mean_target_logit_margin"]})
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"inner_{dataset}.csv", index=False)
    real = independent.predict(np.asarray(prepared.x_valid[:5000], np.float32)) == 1
    print(dataset, "real validation lof_test", round(float(real.mean()), 3))
    print(frame.groupby("variant", sort=False)[["yield", "robust", "proximity", "sparsity", "diversity",
                                                "lof_train", "lof_test", "plaus_slots_test"]].mean().round(3))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--bundle")
args = parser.parse_args()
run(args.dataset, args.bundle)

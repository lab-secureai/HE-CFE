"""Benchmark the diversity-preserving GA against the frozen methods.

Reuses the frozen `non_gan_v55_runner.benchmark` loop unchanged; only the
method list is extended with `gaussian_ga_div` (see `ga_diverse.py`) and one
DiCE metric (`count_diversity`) is appended to every row.

Stages:
  inner   12 factuals per direction (seed 1707, validation split), robustness 4,
          no fidelity; used to accept or reject the variant.
  outer   canonical 200 outer factuals (seed 707), robustness 16, fidelity 96.
  confirm fresh confirmation cohort: 100 test factuals per direction drawn with
          the outer stratification (seed 2027) after removing every outer
          factual (and, for Heart+, every outer feature group).  GA-div was
          designed after its predecessor failed on outer Adult/Heart+, so this
          cohort, never seen during design, checks that result.  Robustness 16,
          no fidelity.

Usage:
  python3 ga_div_benchmark.py --dataset adult --stage inner --output <dir>
      [--populations 64 128 256 512] [--limit-per-direction N] [--methods ...]
      [--bundle <dir>] [--shard-index i --num-shards n]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE / "vendor", HERE):
    sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd
import torch

import non_gan_v55_runner as runner
import ga_diverse

REVISION = HERE.parent
METHODS = ("gaussian_only", "gaussian_ga", "gaussian_ga_div")
ADULT_HANDOFF = REVISION / "results" / "e2_kaggle" / "adult" / "hecfe-e4-adult-mlp-training-3-seed-cpu"

_frozen_run_method = runner.run_method
_frozen_set_metrics = runner._set_metrics


GA_DIV_SWITCHES = ("no_archive_parents", "no_crowding", "no_immigrants", "no_dedup",
                   "no_group_mutation", "no_group_crossover")
# Grouped ablations: the four diversity mechanisms together, and all six switches.
GA_DIV_GROUPS = {"no_diversity": GA_DIV_SWITCHES[:4], "minimal": GA_DIV_SWITCHES}
ABLATIONS = ("no_mmr", "no_projection") + GA_DIV_SWITCHES + tuple(GA_DIV_GROUPS)
REDUCE_STEPS = 4
_REDUCTION: dict[int, tuple[int, int, float]] = {}


class _NoProjection:
    """E1 'No projection': the search skips domain projection (immutable
    features are still copied by the search loop).  Metrics keep using the
    real projector, so infeasible CFEs count as failures."""

    def __init__(self, projector):
        self._projector = projector

    def __getattr__(self, name):
        return getattr(self._projector, name)

    def project(self, candidate, query):
        return np.asarray(candidate, np.float32).copy()


def run_method(method, query, desired, oracle, projector, config, device, seed):
    base, *tags = method.split("__")
    reduce = "reduce" in tags
    tags = [t for t in tags if t != "reduce"]
    if len(tags) > 1 or (tags and tags[0] not in ABLATIONS):
        raise ValueError(method)
    ablation = tags[0] if tags else ""
    search_config, search_projector = config, projector
    if ablation == "no_mmr":
        search_config = dataclasses.replace(config, set_diversity_weight=0.)
    if ablation == "no_projection":
        search_projector = _NoProjection(projector)
    if base == "gaussian_ga_div":
        off = GA_DIV_GROUPS.get(ablation, (ablation,))
        switches = {name[3:]: name not in off for name in GA_DIV_SWITCHES}
        result = ga_diverse.search_ga_diverse(query, desired, oracle, search_projector,
                                              search_config, device, seed,
                                              distribution="gaussian", **switches)
    elif ablation in GA_DIV_SWITCHES or ablation in GA_DIV_GROUPS:
        raise ValueError(method)
    else:
        result = _frozen_run_method(base, query, desired, oracle, search_projector,
                                    search_config, device, seed)
    _REDUCTION.clear()
    if not reduce:
        return result
    import plaintext_cfe_pipeline as pipe
    cfes, audit, runtime, snapshots = result
    logit_fn = lambda values: pipe.logits_numpy(oracle, values, device)  # noqa: E731
    reduced = {}
    for rounds, values in snapshots.items():
        started = time.perf_counter()
        reduced[rounds], evaluations, steps = ga_diverse.reduce_cfes(
            values, query, desired, logit_fn, projector, config, max_steps=REDUCE_STEPS)
        _REDUCTION[int(rounds)] = (evaluations, steps, time.perf_counter() - started)
    return reduced[max(reduced)], audit, runtime, reduced


def set_metrics(cfes, query, desired, oracle, projector, config, *args, **kwargs):
    metrics = _frozen_set_metrics(cfes, query, desired, oracle, projector, config,
                                  *args, **kwargs)
    audit = args[2] if len(args) > 2 else kwargs.get("audit")  # (device, runtime, audit)
    rounds = int(len(audit)) if audit is not None else 0
    evaluations, steps, seconds = _REDUCTION.get(rounds, (0, 0, 0.))
    metrics["reduction_oracle_evaluations"] = evaluations
    metrics["reduction_steps"] = steps
    metrics["reduction_seconds"] = seconds
    metrics["count_diversity"] = ga_diverse.count_diversity(cfes, projector)
    (metrics["proximity_continuous"],
     metrics["proximity_categorical"]) = ga_diverse.dice_proximity(cfes, query, projector)
    metrics["redundancy"] = ga_diverse.redundancy(cfes, query, desired, oracle, projector,
                                                  config, "cpu")
    return metrics


runner.FORWARD_METHODS = METHODS
runner.METHOD_LABELS["gaussian_ga_div"] = "Gaussian proposal + diversity-preserving GA (revision)"


class _Labels(dict):
    def __missing__(self, method):
        base, *tags = method.split("__")
        return f"{self[base]} [{', '.join(tags)}]"


runner.METHOD_LABELS = _Labels(runner.METHOD_LABELS)
runner.METRICS.extend(["count_diversity", "proximity_continuous",
                       "proximity_categorical", "redundancy", "reduction_oracle_evaluations",
                       "reduction_steps"])
runner.run_method = run_method
runner._set_metrics = set_metrics


def pipe_projector(prepared):
    import plaintext_cfe_pipeline as pipe
    return pipe.DomainProjector(prepared)


def load(dataset: str, bundle: Path | None = None):
    if dataset == "adult":
        import e4_adult
        import plaintext_cfe_pipeline as pipe
        prepared, oracle, _ = e4_adult.load_handoff(bundle or ADULT_HANDOFF)
        outer = pipe.select_stratified_bidirectional_queries(prepared, oracle, "cpu",
                                                             per_direction=100, seed=707)
        return prepared, oracle, pipe.DomainProjector(prepared), outer
    if dataset == "credit":
        import credit_default
        prepared, oracle, _ = credit_default.load_handoff(bundle)
        outer = credit_default.outer_queries(prepared, oracle)
        return prepared, oracle, pipe_projector(prepared), outer
    from frozen_setup import load_frozen
    frozen = load_frozen(dataset, bundle or REVISION / "kaggle_inputs" / f"e2-{dataset}")
    return frozen["prepared"], frozen["oracle"], frozen["projector"], frozen["outer_queries"]


CONFIRM_SEED = 2027
CONFIRM_QUERY_OFFSET = 10_000


def select_confirmation(prepared, oracle, outer, per_direction=100, seed=CONFIRM_SEED):
    """Outer-style stratified cohort disjoint from the outer factuals.

    Same eligibility (correctly predicted test rows, one per exact-feature
    group on Heart+) and the same three |logit| strata as
    `select_stratified_bidirectional_queries`; ECG subject round-robin is not
    reproduced (uniform draw inside each stratum).
    """
    import plaintext_cfe_pipeline as pipe

    probability = pipe.probabilities(oracle, prepared.x_test, "cpu")
    prediction = (probability >= .5).astype(np.int8)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    logit = np.log(clipped / (1 - clipped))
    groups = (prepared.raw.groups[prepared.split.test_idx]
              if prepared.raw.groups is not None else None)
    used = set(outer.local_test_index.astype(int))
    rows = []
    for number, (query_class, desired, name) in enumerate(
            [(1, 0, "disease_to_no_disease"), (0, 1, "no_disease_to_disease")]):
        eligible = np.where((prepared.y_test == query_class) & (prediction == query_class))[0]
        eligible = eligible[~np.isin(eligible, list(used))]
        if prepared.raw.name == "heartplus" and groups is not None:
            outer_groups = set(groups[list(used)])
            eligible = eligible[~np.isin(groups[eligible], list(outer_groups))]
            _, first = np.unique(groups[eligible], return_index=True)
            eligible = eligible[np.sort(first)]
        if len(eligible) < per_direction:
            raise ValueError(f"{name}: only {len(eligible)} eligible confirmation factuals")
        ordered = eligible[np.argsort(np.abs(logit[eligible]))]
        rng = np.random.default_rng(seed + number)
        base, remainder = divmod(per_direction, 3)
        for i, (label, members) in enumerate(zip(
                ("near_boundary", "medium_confidence", "high_confidence"),
                np.array_split(ordered, 3))):
            for index in np.sort(rng.choice(members, base + int(i < remainder), replace=False)):
                rows.append({"local_test_index": int(index),
                             "source_row_index": int(prepared.split.test_idx[index]),
                             "query_class": query_class, "desired_class": desired,
                             "direction": name, "difficulty_stratum": label,
                             "requested_per_direction": per_direction,
                             "actual_per_direction": per_direction,
                             "eligible_in_direction": int(len(eligible)),
                             "query_probability": float(probability[index]),
                             "absolute_query_logit": float(abs(logit[index]))})
    cohort = pd.DataFrame(rows)
    cohort.insert(0, "query_id", CONFIRM_QUERY_OFFSET + np.arange(len(cohort)))
    assert not set(cohort.local_test_index) & used
    return cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ecg", "heartplus", "mimic", "adult"])
    parser.add_argument("--stage", required=True, choices=["inner", "outer", "confirm"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--populations", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--limit-per-direction", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--bundle", help="E2 bundle dir, or Adult training-kernel output")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11, help="search seed (E5: 22, 33)")
    args = parser.parse_args()
    runner.FORWARD_METHODS = tuple(args.methods)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prepared, oracle, projector, outer = load(args.dataset,
                                              Path(args.bundle) if args.bundle else None)
    if args.stage == "inner":
        queries = runner.select_inner_queries(prepared, oracle, "cpu", per_direction=12, seed=1707)
        options = dict(robustness_repetitions=4, fidelity_samples=0, save_cfe_values=False)
    elif args.stage == "confirm":
        queries = select_confirmation(prepared, oracle, outer)
        queries.to_csv(output / "confirmation_query_manifest.csv", index=False)
        options = dict(robustness_repetitions=16, fidelity_samples=0, save_cfe_values=True)
    else:
        queries = outer
        options = dict(robustness_repetitions=16, fidelity_samples=96, save_cfe_values=True)
    if args.limit_per_direction:
        queries = queries.groupby("direction", group_keys=False).head(args.limit_per_direction)
    # Shards take query_id ranks modulo the shard count (same rule as E2).
    queries = queries.sort_values("query_id").reset_index(drop=True)
    queries = queries[queries.index % args.num_shards == args.shard_index]
    detail, _, _, by_direction, macro = runner.benchmark(
        prepared, oracle, projector, queries, output, device="cpu",
        populations=args.populations, snapshots=(2, 4, 8, 16), k=10, seed=args.seed,
        include_wachter=False, checkpoint_every=6, **options)
    acceptance = {"dataset": args.dataset, "stage": args.stage, "methods": list(args.methods),
                  "shard_index": args.shard_index, "num_shards": args.num_shards,
                  "search_seed": args.seed, "populations": list(args.populations),
                  "factuals": int(len(queries)), "rows": int(len(detail)),
                  "expected_rows": int(len(queries) * len(args.methods) * len(args.populations) * 4),
                  "wall_seconds": time.perf_counter() - started}
    acceptance["accepted"] = acceptance["rows"] == acceptance["expected_rows"]
    (output / "ga_div_acceptance.json").write_text(json.dumps(acceptance, indent=2))
    print(json.dumps(acceptance))


if __name__ == "__main__":
    main()

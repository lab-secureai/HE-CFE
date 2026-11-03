"""E3: external counterfactual baselines on the canonical outer factuals.

Methods (same 200 factuals, K=10, logit margin 0.10, immutable features,
domain projector and metric code as the HE-CFE benchmark):

  nice_sparsity, nice_proximity
      NICE (Brughmans et al., 2023), re-implemented in the processed space:
      the nearest unlike neighbour (NUN) is taken from development rows that,
      with the factual's immutable features copied in and projected, are
      margin-valid for the desired class; features are then copied group by
      group from the NUN, each step taking the group with the largest reward
      (score gain, or score gain per unit of added distance), until the
      candidate is margin-valid.  One CFE per factual by design.
  geco
      GeCo (Schleich et al., 2021), re-implemented in Python: genetic search
      over "delta" candidates whose changed groups take values observed in the
      development data; initial population of single-group changes, then
      selection, union crossover and one-group mutation.  The fitness is the
      shared quality Q plus the margin shortfall (as in the HE-CFE search);
      valid candidates enter the same distinctness archive and the K returned
      CFEs use the same MMR selection.
  official_dice_random, official_dice_genetic
      dice-ml 0.12 through the frozen pipeline helpers (Adult only; the other
      datasets already have canonical Official DiCE outer runs).

Neither re-implementation is the authors' code; tables must say so.

Usage:
  python3 e3_baselines.py run --dataset adult --output <dir> [--methods ...]
      [--bundle <dir>] [--shard-index i --num-shards n] [--limit-per-direction N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE / "vendor", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd
import torch

import ga_div_benchmark as bench
import ga_diverse
import plaintext_cfe_pipeline as pipe
from non_gan_v55_runner import _query_vector

K = 10
MARGIN = .10
SEED = 11
GECO_INIT_PER_GROUP = 20
GECO_SURVIVORS = 100
GECO_CROSSOVERS = 100
GECO_MUTANTS = 100
GECO_MAX_GENERATIONS = 50
GECO_STABLE_GENERATIONS = 3
DICE_TIMEOUT_SECONDS = 120
DICE_RANDOM_SAMPLE_SIZE = 10_000
DICE_GENETIC_MAXITERATIONS = 300
ADULT_IMMUTABLE_RAW = {"age", "race", "sex", "native_country"}
ALL_METHODS = ("nice_sparsity", "nice_proximity", "geco",
               "official_dice_random", "official_dice_genetic")


def _signed(logit, desired):
    return logit if desired == 1 else -logit


def _group_distance(values, query, projector, layout, discrete):
    normalized = np.abs(values - query[None, :]) / projector.scale[None, :]
    parts = [(normalized[:, g].max(axis=1) > .02).astype(float) if d else normalized[:, g].mean(axis=1)
             for g, d in zip(layout.groups, discrete)]
    return np.column_stack(parts)


def nice(query, desired, oracle, projector, config, reward):
    layout = ga_diverse._layout(projector)
    discrete = ga_diverse._discrete_groups(projector)
    started = time.perf_counter()
    pool = projector.reference_x.copy()
    pool[:, projector.immutable_mask] = query[projector.immutable_mask]
    pool = projector.project(pool, query)
    logit = pipe.logits_numpy(oracle, pool, "cpu")
    evaluations = len(pool)
    ok = pipe._target_valid(logit, desired, config.margin) & projector.feasibility(pool, query)
    audit = {"candidates": evaluations}
    if not ok.any():
        return np.empty((0, len(query)), np.float32), audit, time.perf_counter() - started
    pool = pool[ok]
    distance = _group_distance(pool, query, projector, layout, discrete).sum(axis=1)
    nun = pool[int(np.argmin(distance))]
    current = query.astype(np.float32).copy()
    remaining = [i for i, g in enumerate(layout.groups)
                 if np.abs(nun[g] - query[g]).max() / projector.scale[g].max() > 1e-6]
    score = float(_signed(pipe.logits_numpy(oracle, current[None, :], "cpu")[0], desired))
    evaluations += 1
    while remaining:
        trials = np.repeat(current[None, :], len(remaining), axis=0)
        for row, number in enumerate(remaining):
            trials[row, layout.groups[number]] = nun[layout.groups[number]]
        trials = projector.project(trials, query)
        trial_logit = pipe.logits_numpy(oracle, trials, "cpu")
        evaluations += len(trials)
        gain = _signed(trial_logit, desired) - score
        if reward == "proximity":
            added = (_group_distance(trials, query, projector, layout, discrete).sum(axis=1)
                     - _group_distance(current[None, :], query, projector, layout,
                                       discrete).sum(axis=1))
            gain = gain / np.maximum(added, 1e-6)
        best = int(np.argmax(gain))
        current = trials[best]
        score = float(_signed(trial_logit[best], desired))
        remaining.pop(best)
        if (pipe._target_valid(trial_logit[best:best + 1], desired, config.margin)
                & projector.feasibility(current[None, :], query))[0]:
            audit["candidates"] = evaluations
            return current[None, :], audit, time.perf_counter() - started
    audit["candidates"] = evaluations
    return np.empty((0, len(query)), np.float32), audit, time.perf_counter() - started


def geco(query, desired, oracle, projector, config, seed):
    rng = np.random.default_rng(seed)
    layout = ga_diverse._layout(projector)
    groups = layout.groups
    reference = projector.reference_x
    started = time.perf_counter()
    archive, archive_quality = [], []
    evaluations = 0

    def evaluate(candidates):
        nonlocal evaluations
        candidates = projector.project(candidates, query)
        logit = pipe.logits_numpy(oracle, candidates, "cpu")
        evaluations += len(candidates)
        feasible = projector.feasibility(candidates, query)
        valid = pipe._target_valid(logit, desired, config.margin) & feasible
        quality = pipe._quality(projector, candidates, query, logit, desired, config)
        pipe._archive_update(archive, archive_quality, candidates, quality, valid, projector,
                             config.distinctness)
        shortfall = np.maximum(0, config.margin - _signed(logit, desired))
        return candidates, quality + shortfall + np.where(feasible, 0., 1e3)

    def with_group(base, number, row):
        child = base.copy()
        child[groups[number]] = reference[row, groups[number]]
        return child

    initial = [with_group(query, number, int(row)) for number in range(len(groups))
               for row in rng.integers(0, len(reference), GECO_INIT_PER_GROUP)]
    population, fitness = evaluate(np.vstack(initial))
    previous, stable, generation = None, 0, 0
    for generation in range(1, GECO_MAX_GENERATIONS + 1):
        order = np.argsort(fitness)[:GECO_SURVIVORS]
        survivors, survivor_fitness = population[order], fitness[order]
        changed = ga_diverse._changed_groups(survivors, query, projector)
        children = []
        for _ in range(GECO_CROSSOVERS):
            a, b = rng.integers(0, len(survivors), 2)
            child = survivors[a].copy()
            for number in np.where(changed[b])[0]:
                if not changed[a, number] or rng.random() < .5:
                    child[groups[number]] = survivors[b, groups[number]]
            children.append(child)
        for _ in range(GECO_MUTANTS):
            a = int(rng.integers(0, len(survivors)))
            options = np.where(~changed[a])[0]
            number = int(rng.choice(options)) if len(options) else int(rng.integers(len(groups)))
            children.append(with_group(survivors[a], number, int(rng.integers(len(reference)))))
        offspring, offspring_fitness = evaluate(np.vstack(children))
        population = np.vstack([survivors, offspring])
        fitness = np.concatenate([survivor_fitness, offspring_fitness])
        top = pipe._mmr_select(archive, archive_quality, projector, config.k,
                               diversity_weight=config.set_diversity_weight)
        key = top.round(5).tobytes() if len(top) == config.k else None
        stable = stable + 1 if key is not None and key == previous else 0
        previous = key
        if stable >= GECO_STABLE_GENERATIONS:
            break
    cfes = pipe._mmr_select(archive, archive_quality, projector, config.k,
                            diversity_weight=config.set_diversity_weight)
    return cfes, {"candidates": evaluations, "generations": generation}, time.perf_counter() - started


def adult_dice_explainers(prepared, oracle, projector):
    import dice_ml

    rng = np.random.default_rng(606)
    train = np.asarray(prepared.split.train_idx)
    if len(train) > 5000:
        train = train[np.sort(rng.choice(len(train), 5000, replace=False))]
    features, target = prepared.raw.feature_cols, prepared.raw.target_col
    reference = prepared.raw.frame.iloc[train][features + [target]].copy()
    categorical = list(prepared.raw.categorical_cols)
    for column in categorical:
        reference[column] = reference[column].astype(str)
    continuous = [c for c in features if c not in categorical]
    data = dice_ml.Data(dataframe=reference, continuous_features=continuous, outcome_name=target)
    model = dice_ml.Model(model=pipe._DiceSklearnOracle(prepared, oracle, "cpu"),
                          backend="sklearn", model_type="classifier")
    explainers = {"official_dice_random": dice_ml.Dice(data, model, method="random"),
                  "official_dice_genetic": dice_ml.Dice(data, model, method="genetic")}
    vary = [c for c in features if c not in ADULT_IMMUTABLE_RAW]
    permitted = {c: [float(projector.raw_low[c]), float(projector.raw_high[c])]
                 for c in continuous if c in projector.raw_low and c in vary}
    return explainers, vary, permitted


def run(dataset, output, methods, bundle=None, shard_index=0, num_shards=1, limit=0):
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prepared, oracle, projector, queries = bench.load(dataset, Path(bundle) if bundle else None)
    if limit:
        queries = queries.groupby("direction", group_keys=False).head(limit)
    queries = queries.sort_values("query_id").reset_index(drop=True)
    queries = queries[queries.index % num_shards == shard_index]
    config = pipe.SearchConfig(population=64, max_rounds=16, k=K, margin=MARGIN)
    dice = None
    if any(m.startswith("official_dice") for m in methods):
        if dataset != "adult":
            raise ValueError("Official DiCE already has canonical outer runs for this dataset")
        dice = adult_dice_explainers(prepared, oracle, projector)
    rows, cfe_rows = [], []
    for count, record in enumerate(queries.itertuples(index=False), 1):
        query, local_index = _query_vector(prepared, record)
        desired = int(record.desired_class)
        for method in methods:
            run_seed = SEED + 100_000 * int(record.query_id) + sum(map(ord, method))
            extra = {}
            if method.startswith("nice_"):
                cfes, audit, runtime = nice(query, desired, oracle, projector, config,
                                            method.split("_", 1)[1])
            elif method == "geco":
                cfes, audit, runtime = geco(query, desired, oracle, projector, config, run_seed)
                extra["geco_generations"] = audit["generations"]
            else:
                explainers, vary, permitted = dice
                cfes, outcome = pipe.official_dice_generate(
                    method, query, desired, prepared, oracle, projector, "cpu",
                    explainers[method], vary, permitted, k=K, seed=run_seed,
                    timeout_seconds=DICE_TIMEOUT_SECONDS,
                    random_sample_size=DICE_RANDOM_SAMPLE_SIZE,
                    genetic_maxiterations=DICE_GENETIC_MAXITERATIONS)
                runtime = outcome["runtime_seconds"]
                audit = {"candidates": np.nan}
                extra.update({f"dice_{k}": v for k, v in outcome.items()
                              if isinstance(v, (int, float, str))})
            metrics = bench.set_metrics(
                cfes, query, desired, oracle, projector, config, "cpu", runtime, None,
                robustness_seed=7_000_000 + run_seed, robustness_repetitions=16,
                fidelity_samples=96)
            base = {"query_id": int(record.query_id), "local_partition_index": local_index,
                    "source_row_index": int(record.source_row_index),
                    "query_class": int(record.query_class), "desired_class": desired,
                    "direction": record.direction,
                    "difficulty_stratum": getattr(record, "difficulty_stratum", ""),
                    "method": method, "run_seed": run_seed,
                    "oracle_evaluations": audit["candidates"]}
            rows.append({**base, **extra, **metrics})
            logits = pipe.logits_numpy(oracle, cfes, "cpu") if len(cfes) else []
            for rank, (cfe, logit) in enumerate(zip(cfes, logits), 1):
                cfe_rows.append({**base, "rank": rank, "cfe_logit": float(logit),
                                 **{f"model_input__{n}": float(cfe[i])
                                    for i, n in enumerate(prepared.feature_names)}})
        pd.DataFrame(rows).to_csv(output / "e3_per_factual.csv", index=False)
        pd.DataFrame(cfe_rows).to_csv(output / "e3_returned_cfes.csv", index=False)
        print({"done": count, "of": len(queries), "elapsed": round(time.perf_counter() - started)},
              flush=True)
    detail = pd.DataFrame(rows)
    acceptance = {"protocol": "HECFE_E3_EXTERNAL_BASELINES", "dataset": dataset,
                  "methods": list(methods), "shard_index": shard_index,
                  "num_shards": num_shards, "factuals": int(len(queries)),
                  "rows": int(len(detail)), "wall_seconds": time.perf_counter() - started}
    acceptance["accepted"] = acceptance["rows"] == len(queries) * len(methods)
    (output / "e3_acceptance.json").write_text(json.dumps(acceptance, indent=2))
    print(json.dumps(acceptance))
    return detail


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=["ecg", "heartplus", "mimic", "adult"])
    r.add_argument("--output", required=True)
    r.add_argument("--methods", nargs="+", default=["nice_sparsity", "nice_proximity", "geco"])
    r.add_argument("--bundle")
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--limit-per-direction", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.output, args.methods, args.bundle, args.shard_index,
        args.num_shards, args.limit_per_direction)


if __name__ == "__main__":
    main()

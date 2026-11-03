"""V5.5-aligned non-GAN random/genetic CFE benchmark.

Gaussian-only and Gaussian-GA reproduce the proposal distributions used by
the V5.5 random and genetic baselines. Uniform-only and Uniform-GA change only
the standardized proposal distribution; every other constraint, archive,
margin, distinctness, selection and metric contract remains shared.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd


FORWARD_METHODS = (
    "uniform_only",
    "gaussian_only",
    "uniform_ga",
    "gaussian_ga",
)
METHOD_LABELS = {
    "uniform_only": "Uniform-only constrained random",
    "gaussian_only": "Gaussian-only constrained random (V5.5 random)",
    "uniform_ga": "Uniform proposal + genetic refinement",
    "gaussian_ga": "Gaussian proposal + genetic refinement (V5.5 genetic)",
    "wachter_style": "Wachter-style white-box gradient",
}
METRICS = [
    "valid_cfe_yield_at_k", "coverage_at_1", "full_k_success",
    "robust_completeness_at_k", "robust_full_k_success",
    "proximity", "sparsity", "group_sparsity", "diversity",
    "plausibility_distance", "plausibility_inlier_rate",
    "target_data_support", "fidelity", "changed_set_dice",
    "constraint_validity", "mean_target_logit_margin",
    "minimum_target_logit_margin", "generation_time_to_k_or_cap_seconds",
    "time_to_k_or_cap_seconds", "time_per_returned_cfe_seconds",
    "candidate_evaluations", "candidates_to_k_or_cap", "rounds_used",
    "candidate_throughput_per_second", "returned",
    "reference_valid_returned", "fidelity_oracle_evaluations",
]


def _draw(rng: np.random.Generator, shape: tuple[int, ...],
          distribution: str) -> np.ndarray:
    if distribution == "gaussian":
        return rng.normal(size=shape).astype(np.float32)
    if distribution == "uniform":
        # Uniform(-sqrt(3), sqrt(3)) has variance one, matching N(0,1).
        return rng.uniform(
            -math.sqrt(3), math.sqrt(3), size=shape
        ).astype(np.float32)
    raise ValueError(distribution)


def search_distribution(query, desired, oracle, projector, config, device, seed,
                        distribution="gaussian", genetic=False):
    """Run one full counted trajectory and retain every round snapshot."""
    import plaintext_cfe_pipeline as pipe

    rng = np.random.default_rng(seed)
    archive: list[np.ndarray] = []
    archive_quality: list[float] = []
    audit_rows: list[dict] = []
    snapshots: dict[int, np.ndarray] = {}
    population = None
    previous_score = None
    pipe._sync(device)
    started = time.perf_counter()
    for round_number in range(1, config.max_rounds + 1):
        if population is None or not genetic:
            candidate = np.repeat(query[None, :], config.population, axis=0)
            sigma = config.mutation_sigma * (
                math.sqrt(round_number) if not genetic else 1.0
            )
            candidate += (
                _draw(rng, candidate.shape, distribution)
                * projector.scale[None, :] * sigma
            )
        else:
            tournament = rng.integers(
                0, len(population), size=(config.population, 3)
            )
            parent_a = population[tournament[
                np.arange(config.population),
                np.argmin(previous_score[tournament], axis=1),
            ]]
            tournament = rng.integers(
                0, len(population), size=(config.population, 3)
            )
            parent_b = population[tournament[
                np.arange(config.population),
                np.argmin(previous_score[tournament], axis=1),
            ]]
            candidate = np.where(rng.random(parent_a.shape) < .5,
                                 parent_a, parent_b)
            mutation = (
                _draw(rng, candidate.shape, distribution)
                * projector.scale[None, :] * config.mutation_sigma
            )
            candidate += mutation * (rng.random(candidate.shape) < .20)

        candidate[:, projector.immutable_mask] = query[projector.immutable_mask]
        candidate = projector.project(candidate, query)
        logit = pipe.logits_numpy(oracle, candidate, device)
        feasible = projector.feasibility(candidate, query)
        valid = pipe._target_valid(logit, desired, config.margin) & feasible
        quality = pipe._quality(
            projector, candidate, query, logit, desired, config
        )
        pipe._archive_update(
            archive, archive_quality, candidate, quality, valid, projector,
            config.distinctness,
        )
        target_penalty = (
            np.maximum(0, config.margin - logit)
            if desired == 1 else np.maximum(0, config.margin + logit)
        )
        previous_score = quality + target_penalty
        population = candidate
        pipe._sync(device)
        cumulative = time.perf_counter() - started
        previous = (audit_rows[-1]["cumulative_elapsed_seconds"]
                    if audit_rows else 0.0)
        audit_rows.append({
            "round": round_number,
            "candidates": int(len(candidate)),
            "candidates_cumulative": int(round_number * len(candidate)),
            "feasible_candidates": int(feasible.sum()),
            "margin_valid_candidates": int(valid.sum()),
            "archive_size": int(len(archive)),
            "round_elapsed_seconds": float(cumulative - previous),
            "cumulative_elapsed_seconds": float(cumulative),
            "proposal_adapts_to_oracle": bool(genetic),
            "acceptance_denominator_kind": (
                "all genetic population candidates" if genetic
                else "all independently sampled candidates"
            ),
            "proposal_distribution": distribution,
        })
        snapshots[round_number] = pipe._mmr_select(
            archive, archive_quality, projector, config.k,
            diversity_weight=config.set_diversity_weight,
        )

    returned = snapshots[config.max_rounds]
    pipe._sync(device)
    return returned, pd.DataFrame(audit_rows), (
        time.perf_counter() - started
    ), snapshots


def run_method(method, query, desired, oracle, projector, config, device, seed):
    import plaintext_cfe_pipeline as pipe

    if method == "uniform_only":
        return search_distribution(
            query, desired, oracle, projector, config, device, seed,
            distribution="uniform", genetic=False,
        )
    if method == "gaussian_only":
        return search_distribution(
            query, desired, oracle, projector, config, device, seed,
            distribution="gaussian", genetic=False,
        )
    if method == "uniform_ga":
        return search_distribution(
            query, desired, oracle, projector, config, device, seed,
            distribution="uniform", genetic=True,
        )
    if method == "gaussian_ga":
        return search_distribution(
            query, desired, oracle, projector, config, device, seed,
            distribution="gaussian", genetic=True,
        )
    if method == "wachter_style":
        cfes, audit, runtime = pipe.search_wachter(
            query, desired, oracle, projector, config, device, seed, steps=300
        )
        return cfes, audit, runtime, {1: cfes}
    raise ValueError(method)


def _query_vector(prepared, record):
    if hasattr(record, "local_test_index"):
        return prepared.x_test[int(record.local_test_index)], int(record.local_test_index)
    if hasattr(record, "local_validation_index"):
        return (prepared.x_valid[int(record.local_validation_index)],
                int(record.local_validation_index))
    raise ValueError("Query manifest has no local partition index")


def _set_metrics(cfes, query, desired, oracle, projector, config, device,
                 runtime, audit, robustness_seed, robustness_repetitions,
                 fidelity_samples):
    import plaintext_cfe_pipeline as pipe

    metrics = pipe.cfe_set_metrics(
        cfes, query, desired, oracle, projector, config, device, runtime,
        audit=audit, robustness_seed=robustness_seed,
        robustness_repetitions=robustness_repetitions,
        fidelity_samples=fidelity_samples,
    )
    metrics["group_sparsity"] = (
        float(projector.group_sparsity(cfes, query).mean())
        if len(cfes) else np.nan
    )
    metrics["time_to_k_or_cap_seconds"] = metrics.get(
        f"time_to_k{config.k}_or_cap_seconds", runtime
    )
    metrics["candidates_to_k_or_cap"] = metrics.get(
        f"candidates_to_k{config.k}_or_cap", metrics["candidate_evaluations"]
    )
    return metrics


def _summaries(detail: pd.DataFrame, output: Path):
    present = [metric for metric in METRICS if metric in detail.columns]
    keys = ["method", "direction", "population", "rounds",
            "candidate_budget"]
    grouped = detail.groupby(keys, dropna=False)[present].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}__{stat}" for metric, stat in grouped.columns]
    by_direction = grouped.reset_index()
    by_direction.to_csv(output / "benchmark_by_direction_mean_std.csv", index=False)

    # There are exactly as many factuals in each direction, so this pooled
    # factual summary is also an equal-direction macro for complete metrics.
    macro_keys = ["method", "population", "rounds", "candidate_budget"]
    macro = detail.groupby(macro_keys, dropna=False)[present].agg(
        ["mean", "std", "count"]
    )
    macro.columns = [f"{metric}__{stat}" for metric, stat in macro.columns]
    macro = macro.reset_index()
    macro.to_csv(output / "benchmark_equal_direction_macro_mean_std.csv", index=False)
    return by_direction, macro


def benchmark(prepared, oracle, projector, queries, output, device="cpu",
              populations=(64, 128, 256, 512), snapshots=(2, 4, 8, 16),
              k=10, seed=11, include_wachter=True,
              robustness_repetitions=16, fidelity_samples=96,
              save_cfe_values=True, checkpoint_every=10):
    """Evaluate each P once to round 16 and read cumulative 2/4/8 snapshots."""
    import plaintext_cfe_pipeline as pipe
    from tqdm.auto import tqdm

    populations = tuple(map(int, populations))
    snapshots = tuple(map(int, snapshots))
    if sorted(set(snapshots)) != list(snapshots) or min(snapshots) < 1:
        raise ValueError("snapshots must be sorted unique positive rounds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    queries.to_csv(output / "query_manifest.csv", index=False)
    rows: list[dict] = []
    cfe_rows: list[dict] = []
    round_rows: list[pd.DataFrame] = []

    for query_number, record in enumerate(tqdm(
            list(queries.itertuples(index=False)), desc="factuals")):
        query, local_index = _query_vector(prepared, record)
        desired = int(record.desired_class)
        for method in FORWARD_METHODS:
            for population in populations:
                config = pipe.SearchConfig(
                    population=population, max_rounds=max(snapshots),
                    k=k, margin=.10,
                )
                run_seed = seed + 100_000 * int(record.query_id) + population
                _, audit, _, cumulative = run_method(
                    method, query, desired, oracle, projector, config, device,
                    run_seed,
                )
                audit = audit.assign(
                    query_id=int(record.query_id), method=method,
                    direction=record.direction, population=population,
                )
                round_rows.append(audit)
                for rounds in snapshots:
                    audit_at_round = audit[audit["round"] <= rounds].copy()
                    cfes = cumulative[rounds]
                    runtime = float(
                        audit_at_round.iloc[-1]["cumulative_elapsed_seconds"]
                    )
                    metrics = _set_metrics(
                        cfes, query, desired, oracle, projector, config, device,
                        runtime, audit_at_round,
                        robustness_seed=(7_000_000 + run_seed + rounds),
                        robustness_repetitions=robustness_repetitions,
                        fidelity_samples=fidelity_samples,
                    )
                    base = {
                        "query_id": int(record.query_id),
                        "local_partition_index": local_index,
                        "source_row_index": int(record.source_row_index),
                        "query_class": int(record.query_class),
                        "desired_class": desired,
                        "direction": record.direction,
                        "difficulty_stratum": getattr(
                            record, "difficulty_stratum", "calibration"
                        ),
                        "query_probability": float(record.query_probability),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "oracle_access": "forward logits only",
                        "population": population,
                        "rounds": rounds,
                        "candidate_budget": population * rounds,
                    }
                    rows.append({**base, **metrics})
                    if save_cfe_values:
                        logits = pipe.logits_numpy(oracle, cfes, device) if len(cfes) else []
                        for rank, (cfe, logit) in enumerate(zip(cfes, logits), start=1):
                            cfe_rows.append({
                                **base, "rank": rank, "cfe_logit": float(logit),
                                **{f"model_input__{name}": float(cfe[index])
                                   for index, name in enumerate(prepared.feature_names)},
                            })

        if include_wachter:
            config = pipe.SearchConfig(population=30, max_rounds=1,
                                       k=k, margin=.10)
            run_seed = seed + 9_000_000 + int(record.query_id)
            cfes, audit, runtime, _ = run_method(
                "wachter_style", query, desired, oracle, projector, config,
                device, run_seed,
            )
            metrics = _set_metrics(
                cfes, query, desired, oracle, projector, config, device,
                runtime, audit, robustness_seed=7_000_000 + run_seed,
                robustness_repetitions=robustness_repetitions,
                fidelity_samples=fidelity_samples,
            )
            base = {
                "query_id": int(record.query_id),
                "local_partition_index": local_index,
                "source_row_index": int(record.source_row_index),
                "query_class": int(record.query_class),
                "desired_class": desired, "direction": record.direction,
                "difficulty_stratum": getattr(
                    record, "difficulty_stratum", "calibration"
                ),
                "query_probability": float(record.query_probability),
                "method": "wachter_style",
                "method_label": METHOD_LABELS["wachter_style"],
                "oracle_access": "white-box backward gradients",
                "population": 30, "rounds": 1,
                "candidate_budget": np.nan,
                "gradient_steps": 300,
            }
            rows.append({**base, **metrics})
            round_rows.append(audit.assign(
                query_id=int(record.query_id), method="wachter_style",
                direction=record.direction, population=30,
            ))
            if save_cfe_values:
                logits = pipe.logits_numpy(oracle, cfes, device) if len(cfes) else []
                for rank, (cfe, logit) in enumerate(zip(cfes, logits), start=1):
                    cfe_rows.append({
                        **base, "rank": rank, "cfe_logit": float(logit),
                        **{f"model_input__{name}": float(cfe[index])
                           for index, name in enumerate(prepared.feature_names)},
                    })

        if (query_number + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(
                output / "benchmark_per_factual_checkpoint.csv", index=False
            )
            print({"completed_factuals": query_number + 1,
                   "total_factuals": len(queries)}, flush=True)

    detail = pd.DataFrame(rows)
    cfe_detail = pd.DataFrame(cfe_rows)
    rounds = pd.concat(round_rows, ignore_index=True) if round_rows else pd.DataFrame()
    detail.to_csv(output / "benchmark_per_factual.csv", index=False)
    cfe_detail.to_csv(output / "returned_cfes_processed.csv", index=False)
    rounds.to_csv(output / "benchmark_round_audit.csv", index=False)
    by_direction, macro = _summaries(detail, output)
    return detail, cfe_detail, rounds, by_direction, macro


def select_inner_queries(prepared, oracle, device="cpu", per_direction=12,
                         seed=1707):
    """Create a correctly classified, probability-stratified validation cohort."""
    import plaintext_cfe_pipeline as pipe

    probabilities = pipe.probabilities(oracle, prepared.x_valid, device)
    predictions = (probabilities >= .5).astype(np.int8)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    confidence = np.abs(np.log(clipped / (1 - clipped)))
    specifications = [
        (1, 0, "disease_to_no_disease"),
        (0, 1, "no_disease_to_disease"),
    ]
    groups = (prepared.raw.groups[prepared.split.valid_idx]
              if prepared.raw.groups is not None else None)
    rows = []
    for direction_number, (query_class, desired, direction) in enumerate(specifications):
        eligible = np.where(
            (prepared.y_valid == query_class) & (predictions == query_class)
        )[0]
        if prepared.raw.name == "heartplus" and groups is not None:
            _, first = np.unique(groups[eligible], return_index=True)
            eligible = eligible[np.sort(first)]
        if len(eligible) < per_direction:
            raise ValueError(f"{direction}: {len(eligible)} < {per_direction}")
        ordered = eligible[np.argsort(confidence[eligible])]
        strata = np.array_split(ordered, 3)
        allocation = [per_direction // 3] * 3
        for index in range(per_direction % 3):
            allocation[index] += 1
        rng = np.random.default_rng(seed + direction_number)
        for stratum, (members, count) in enumerate(zip(strata, allocation)):
            chosen = rng.choice(members, count, replace=False)
            for local_index in chosen:
                rows.append({
                    "local_validation_index": int(local_index),
                    "source_row_index": int(prepared.split.valid_idx[local_index]),
                    "query_class": query_class, "desired_class": desired,
                    "direction": direction,
                    "difficulty_stratum": (
                        "near_boundary", "medium_confidence", "high_confidence"
                    )[stratum],
                    "query_probability": float(probabilities[local_index]),
                })
    result = pd.DataFrame(rows).reset_index(drop=True)
    result.insert(0, "query_id", np.arange(len(result), dtype=int))
    return result


def select_inner_contract(direction_summary: pd.DataFrame, output: Path,
                          yield_tolerance=.03, robust_tolerance=.05):
    """Freeze method/P/R without consulting outer-test CFE outcomes."""
    output = Path(output)
    forward = direction_summary[
        direction_summary.method.isin(FORWARD_METHODS)
    ].copy()
    yield_col = "valid_cfe_yield_at_k__mean"
    robust_col = "robust_completeness_at_k__mean"
    aggregation = {
        "worst_direction_yield": (yield_col, "min"),
        "macro_yield": (yield_col, "mean"),
        "macro_robust_yield": (robust_col, "mean"),
        "macro_sparsity": ("sparsity__mean", "mean"),
        "macro_diversity": ("diversity__mean", "mean"),
        "macro_proximity": ("proximity__mean", "mean"),
        "macro_time_to_k_or_cap": ("time_to_k_or_cap_seconds__mean", "mean"),
    }
    profiles = forward.groupby(
        ["method", "population", "rounds", "candidate_budget"], as_index=False
    ).agg(**aggregation)
    best_worst = float(profiles.worst_direction_yield.max())
    eligible = profiles[
        profiles.worst_direction_yield >= best_worst - yield_tolerance
    ].copy()
    best_robust = float(eligible.macro_robust_yield.max())
    eligible = eligible[
        eligible.macro_robust_yield >= best_robust - robust_tolerance
    ].copy()
    best_macro = float(eligible.macro_yield.max())
    eligible = eligible[eligible.macro_yield >= best_macro - yield_tolerance]
    selected = eligible.sort_values(
        ["candidate_budget", "rounds", "macro_sparsity", "macro_diversity",
         "macro_proximity", "macro_time_to_k_or_cap", "method"],
        ascending=[True, True, False, False, True, True, True],
    ).iloc[0]
    profiles["eligible_after_noninferiority"] = profiles.index.isin(eligible.index)
    profiles["selected"] = (
        profiles.method.eq(selected.method)
        & profiles.population.eq(selected.population)
        & profiles.rounds.eq(selected.rounds)
    )
    profiles.to_csv(output / "inner_profile_selection.csv", index=False)
    contract = {
        "selection_scope": "inner validation only",
        "outer_test_used_for_selection": False,
        "rule": ("worst-direction Yield noninferiority, robust Yield "
                 "noninferiority, macro Yield noninferiority, then minimum "
                 "candidate budget/fewer HE rounds and quality tie-breaks"),
        "yield_tolerance": yield_tolerance,
        "robust_tolerance": robust_tolerance,
        "method": str(selected.method),
        "population": int(selected.population),
        "rounds": int(selected.rounds),
        "candidate_budget": int(selected.candidate_budget),
        "k": 10,
        "margin": .10,
    }
    (output / "selected_inner_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    return contract, profiles


def qualitative_grid(prepared, queries, cfe_detail, contract, output,
                     per_direction=5):
    """Export a deterministic 10-factual raw-domain comparison grid."""
    import plaintext_cfe_pipeline as pipe

    output = Path(output)
    representatives = pipe.representative_query_subset(
        queries, per_direction=per_direction
    )
    rows = []
    for record in representatives.itertuples(index=False):
        query = prepared.x_test[int(record.local_test_index)]
        raw_query = prepared.preprocessor.inverse(query[None, :]).iloc[0]
        for method in (*FORWARD_METHODS, "wachter_style"):
            subset = cfe_detail[
                cfe_detail.query_id.eq(int(record.query_id))
                & cfe_detail.method.eq(method)
            ]
            if method != "wachter_style":
                subset = subset[
                    subset.population.eq(int(contract["population"]))
                    & subset.rounds.eq(int(contract["rounds"]))
                ]
            subset = subset.sort_values("rank")
            if subset.empty:
                rows.append({
                    "query_id": int(record.query_id), "direction": record.direction,
                    "method": method, "status": "NO VALID CFE",
                    "query_probability": float(record.query_probability),
                    "cfe_probability": np.nan, "changed_features": "",
                })
                continue
            first = subset.iloc[0]
            cfe = np.asarray([
                first[f"model_input__{name}"] for name in prepared.feature_names
            ], dtype=np.float32)
            raw_cfe = prepared.preprocessor.inverse(cfe[None, :]).iloc[0]
            changed = []
            for feature in raw_query.index:
                before, after = raw_query[feature], raw_cfe[feature]
                try:
                    different = not np.isclose(float(before), float(after), equal_nan=True)
                except (TypeError, ValueError):
                    different = str(before) != str(after)
                if different:
                    changed.append(f"{feature}: {before} -> {after}")
            probability = float(1 / (1 + np.exp(-np.clip(first.cfe_logit, -40, 40))))
            rows.append({
                "query_id": int(record.query_id), "direction": record.direction,
                "method": method, "status": "VALID CFE",
                "query_probability": float(record.query_probability),
                "cfe_probability": probability,
                "changed_features": " | ".join(changed),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "qualitative_grid_10_factuals.csv", index=False)
    compact = table.pivot(index=["query_id", "direction"], columns="method",
                          values="changed_features").reset_index()
    compact.to_csv(output / "qualitative_grid_10_factuals_compact.csv", index=False)
    return table, compact

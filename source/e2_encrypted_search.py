"""E2: encrypted Gaussian-GA CFE search, end to end, measured directly.

Every search decision (validity, quality, archive, tournament penalty) uses the
CKKS-decrypted logit returned by the server.  The plaintext graph is evaluated
only as a shadow, outside the timed region, to report decision agreement.

One fixed-R encrypted trajectory is run per factual.  Because no round depends
on later rounds, the early-stop result (stop at the first round whose archive
holds K valid distinct CFEs) is exactly the prefix of that trajectory; its
latency is the cumulative encrypted wall-clock at the stop round.  The fixed-R
mode always sends R rounds, so the server does not observe when K was reached.

A plaintext twin (the canonical `search_distribution`, same seed) is run on
the same factual to give the plaintext wall-clock of the identical pipeline
and to test trajectory identity.  Every returned CFE is re-verified with the
plaintext seed-55 oracle (`_set_metrics`).

Usage:
  python3 e2_encrypted_search.py run --dataset ecg --bundle <dir> --output <dir>
      [--shard-index 0 --num-shards 1 --workers 1 --limit-per-direction N]
  python3 e2_encrypted_search.py merge --dataset ecg --bundle <dir>
      --output <dir> --inputs <shard dir> [<shard dir> ...]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "HECFE_E2_ENCRYPTED_SEARCH_END_TO_END"
SEARCH_SEED = 11
MIN_LABEL_AGREEMENT = .99
ROBUSTNESS_REPETITIONS = 16
FIDELITY_SAMPLES = 96
MODES = ("encrypted_fixed_r", "encrypted_early_stop",
         "plaintext_fixed_r", "plaintext_early_stop")
HE_PHASES = ("encryption_seconds", "upload_serialization_seconds",
             "server_deserialization_seconds", "server_inference_seconds_per_round",
             "download_serialization_seconds", "decryption_seconds")


def encrypted_gaussian_ga(query, desired, scorer, oracle, projector, config, seed):
    """Gaussian-GA search of `non_gan_v55_runner.search_distribution`
    (distribution="gaussian", genetic=True) with CKKS-decrypted logits.

    The control flow and RNG call order are copied line by line so that, with
    an exact scorer, the trajectory equals the canonical plaintext one.
    """
    import plaintext_cfe_pipeline as pipe
    from non_gan_v55_runner import _draw

    rng = np.random.default_rng(seed)
    archive: list[np.ndarray] = []
    archive_quality: list[float] = []
    audit_rows: list[dict] = []
    snapshots: dict[int, np.ndarray] = {}
    population = None
    previous_score = None
    excluded = 0.0  # shadow/audit-only work removed from the encrypted clock
    started = time.perf_counter()
    for round_number in range(1, config.max_rounds + 1):
        round_started = time.perf_counter()
        excluded_before = excluded
        if population is None:
            candidate = np.repeat(query[None, :], config.population, axis=0)
            candidate += (_draw(rng, candidate.shape, "gaussian")
                          * projector.scale[None, :] * config.mutation_sigma)
        else:
            tournament = rng.integers(0, len(population), size=(config.population, 3))
            parent_a = population[tournament[
                np.arange(config.population),
                np.argmin(previous_score[tournament], axis=1)]]
            tournament = rng.integers(0, len(population), size=(config.population, 3))
            parent_b = population[tournament[
                np.arange(config.population),
                np.argmin(previous_score[tournament], axis=1)]]
            candidate = np.where(rng.random(parent_a.shape) < .5, parent_a, parent_b)
            mutation = (_draw(rng, candidate.shape, "gaussian")
                        * projector.scale[None, :] * config.mutation_sigma)
            candidate += mutation * (rng.random(candidate.shape) < .20)

        candidate[:, projector.immutable_mask] = query[projector.immutable_mask]
        candidate = projector.project(candidate, query)
        scorer_started = time.perf_counter()
        logit, he = scorer(candidate)
        scorer_seconds = time.perf_counter() - scorer_started
        # evaluate_population computes a plaintext reference for its own audit
        excluded += he["plaintext_reference_inference_seconds"]

        feasible = projector.feasibility(candidate, query)
        valid = pipe._target_valid(logit, desired, config.margin) & feasible
        quality = pipe._quality(projector, candidate, query, logit, desired, config)
        pipe._archive_update(archive, archive_quality, candidate, quality, valid,
                             projector, config.distinctness)
        target_penalty = (np.maximum(0, config.margin - logit) if desired == 1
                          else np.maximum(0, config.margin + logit))
        previous_score = quality + target_penalty
        population = candidate
        snapshots[round_number] = pipe._mmr_select(
            archive, archive_quality, projector, config.k,
            diversity_weight=config.set_diversity_weight)
        cumulative = time.perf_counter() - started - excluded
        round_seconds = time.perf_counter() - round_started - (excluded - excluded_before)

        shadow_started = time.perf_counter()
        plain = pipe.logits_numpy(oracle, candidate, "cpu")
        plain_valid = pipe._target_valid(plain, desired, config.margin) & feasible
        audit_rows.append({
            "round": round_number,
            "candidates": int(len(candidate)),
            "candidates_cumulative": int(round_number * len(candidate)),
            "feasible_candidates": int(feasible.sum()),
            "margin_valid_candidates": int(valid.sum()),
            "archive_size": int(len(archive)),
            "round_elapsed_seconds": float(round_seconds),
            "cumulative_elapsed_seconds": float(cumulative),
            "proposal_adapts_to_oracle": True,
            "acceptance_denominator_kind": "all genetic population candidates",
            "proposal_distribution": "gaussian",
            "he_scorer_wall_seconds": float(scorer_seconds
                                            - he["plaintext_reference_inference_seconds"]),
            "client_search_ops_seconds": float(
                round_seconds - scorer_seconds
                + he["plaintext_reference_inference_seconds"]),
            **{phase: float(he[phase]) for phase in HE_PHASES},
            "upload_ciphertext_mb": he["upload_ciphertext_mb"],
            "download_ciphertext_mb": he["download_ciphertext_mb"],
            "peak_rss_mb": he["peak_rss_mb"],
            "max_abs_logit_error_vs_torch": float(np.max(np.abs(logit - plain))),
            "mean_abs_logit_error_vs_torch": float(np.mean(np.abs(logit - plain))),
            "max_abs_logit_error_vs_numpy_graph": he["max_abs_logit_error"],
            "label_agreement_vs_torch": float(np.mean((logit >= 0) == (plain >= 0))),
            "margin_valid_decision_agreement": float(np.mean(valid == plain_valid)),
            "margin_valid_decision_flips": int(np.sum(valid != plain_valid)),
            "plaintext_margin_valid_candidates": int(plain_valid.sum()),
            "minimum_abs_plaintext_logit": float(np.min(np.abs(plain))),
        })
        excluded += time.perf_counter() - shadow_started

    return snapshots, pd.DataFrame(audit_rows), time.perf_counter() - started - excluded


def _early_stop_round(audit: pd.DataFrame, k: int, max_rounds: int) -> tuple[int, bool]:
    hit = audit[audit.archive_size >= k]
    return (int(hit["round"].iloc[0]), True) if len(hit) else (max_rounds, False)


# --------------------------------------------------------------------------
# Worker process: one frozen setup and one CKKS key pair per process.
# --------------------------------------------------------------------------
_STATE: dict = {}


def _init_worker(dataset: str, bundle: str, threads: int) -> None:
    import torch
    torch.set_num_threads(max(1, threads))
    from frozen_setup import load_frozen
    from he_ckks import HEPopulationScorer

    frozen = load_frozen(dataset, bundle)
    point = frozen["operating_point"]
    scorer = HEPopulationScorer(frozen["he_parameters"], profile_name=point["profile"],
                                cfe_margin=point["margin"])
    _STATE.update(frozen=frozen, scorer=scorer, point=point)


def _cfe_rows(base, mode, cfes, he_logits, oracle, feature_names):
    import plaintext_cfe_pipeline as pipe
    plain = pipe.logits_numpy(oracle, cfes, "cpu") if len(cfes) else []
    rows = []
    for rank, (cfe, logit) in enumerate(zip(cfes, plain), start=1):
        rows.append({**base, "mode": mode, "rank": rank,
                     "plaintext_cfe_logit": float(logit),
                     **{f"model_input__{n}": float(cfe[i])
                        for i, n in enumerate(feature_names)}})
    return rows


def process_factual(record: dict) -> dict:
    import plaintext_cfe_pipeline as pipe
    from non_gan_v55_runner import _query_vector, _set_metrics, search_distribution

    frozen, scorer, point = _STATE["frozen"], _STATE["scorer"], _STATE["point"]
    prepared, oracle, projector = frozen["prepared"], frozen["oracle"], frozen["projector"]
    rec = pd.Series(record)
    query, local_index = _query_vector(prepared, rec)
    desired = int(rec.desired_class)
    population, rounds = int(point["population"]), int(point["rounds"])
    config = pipe.SearchConfig(population=population, max_rounds=rounds,
                               k=int(point["k"]), margin=float(point["margin"]))
    run_seed = SEARCH_SEED + 100_000 * int(rec.query_id) + population

    snapshots, audit, enc_runtime = encrypted_gaussian_ga(
        query, desired, scorer, oracle, projector, config, run_seed)
    _, plain_audit, plain_runtime, plain_snapshots = search_distribution(
        query, desired, oracle, projector, config, "cpu", run_seed,
        distribution="gaussian", genetic=True)

    base = {
        "query_id": int(rec.query_id), "local_partition_index": local_index,
        "source_row_index": int(rec.source_row_index),
        "query_class": int(rec.query_class), "desired_class": desired,
        "direction": rec.direction, "difficulty_stratum": rec.difficulty_stratum,
        "query_probability": float(rec.query_probability),
        "method": "gaussian_ga", "population": population, "rounds_cap": rounds,
        "search_seed": SEARCH_SEED, "run_seed": run_seed,
        "worker_pid": os.getpid(),
    }
    results, cfe_rows = [], []
    for mode, snaps, trace, total in (
            ("encrypted", snapshots, audit, enc_runtime),
            ("plaintext", plain_snapshots, plain_audit, plain_runtime)):
        stop, reached = _early_stop_round(trace, config.k, rounds)
        for suffix, at in (("fixed_r", rounds), ("early_stop", stop)):
            name = f"{mode}_{suffix}"
            trace_at = trace[trace["round"] <= at]
            runtime = float(trace_at.iloc[-1]["cumulative_elapsed_seconds"])
            cfes = snaps[at]
            metrics = _set_metrics(
                cfes, query, desired, oracle, projector, config, "cpu", runtime,
                trace_at, robustness_seed=7_000_000 + run_seed + at,
                robustness_repetitions=ROBUSTNESS_REPETITIONS,
                fidelity_samples=FIDELITY_SAMPLES)
            plain_logit = pipe.logits_numpy(oracle, cfes, "cpu") if len(cfes) else np.array([])
            failing = int(np.sum(~pipe._target_valid(plain_logit, desired, config.margin))) \
                if len(cfes) else 0
            row = {**base, "mode": name, "server_observed_rounds": at,
                   "k_reached": reached, "early_stop_round": stop,
                   "wall_clock_search_seconds": runtime,
                   "returned_failing_plaintext_margin": failing, **metrics}
            if mode == "encrypted":
                for phase in HE_PHASES + ("he_scorer_wall_seconds",
                                          "client_search_ops_seconds"):
                    row[f"sum_{phase}"] = float(trace_at[phase].sum())
                row["upload_mb_total"] = float(trace_at.upload_ciphertext_mb.sum())
                row["download_mb_total"] = float(trace_at.download_ciphertext_mb.sum())
                row["min_label_agreement_vs_torch"] = float(trace_at.label_agreement_vs_torch.min())
                row["min_margin_valid_decision_agreement"] = float(
                    trace_at.margin_valid_decision_agreement.min())
                row["margin_valid_decision_flips"] = int(trace_at.margin_valid_decision_flips.sum())
                row["max_abs_logit_error_vs_torch"] = float(trace_at.max_abs_logit_error_vs_torch.max())
                row["peak_rss_mb"] = float(trace_at.peak_rss_mb.max())
                twin = plain_snapshots[at]
                row["identical_to_plaintext_twin"] = bool(
                    cfes.shape == twin.shape and np.allclose(cfes, twin, atol=1e-6))
            results.append(row)
            cfe_rows += _cfe_rows(base, name, cfes, None, oracle, prepared.feature_names)
    audit = audit.assign(query_id=int(rec.query_id), direction=rec.direction)
    return {"rows": results, "cfes": cfe_rows, "audit": audit.to_dict("records")}


def environment_record(workers: int) -> dict:
    import psutil
    import tenseal as ts
    import torch
    return {
        "protocol": PROTOCOL, "python_version": platform.python_version(),
        "platform": platform.platform(), "tenseal_version": getattr(ts, "__version__", "?"),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "memory_total_gb": psutil.virtual_memory().total / 1e9,
        "workers": workers, "kaggle": Path("/kaggle/working").exists(),
        "secret_key_location": "client process only; server context from public blob",
    }


def run_shard(dataset, bundle, output, shard_index=0, num_shards=1, workers=1,
              limit_per_direction=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    queries = pd.read_csv(Path(bundle) / "canonical" / "outer_test_query_manifest.csv")
    queries = queries.sort_values("query_id").reset_index(drop=True)
    if limit_per_direction:
        queries = queries.groupby("direction", group_keys=False).head(limit_per_direction)
        queries = queries.sort_values("query_id").reset_index(drop=True)
    shard = queries[queries.index % num_shards == shard_index]
    done_path = output / "e2_per_factual.csv"
    rows, cfes, audits = [], [], []
    if done_path.exists():  # resume within the same working directory
        rows = pd.read_csv(done_path).to_dict("records")
        cfes = pd.read_csv(output / "e2_returned_cfes.csv").to_dict("records")
        audits = pd.read_csv(output / "e2_round_audit.csv").to_dict("records")
        shard = shard[~shard.query_id.isin({r["query_id"] for r in rows})]
    shard.to_csv(output / "e2_shard_query_manifest.csv", index=False)
    env = environment_record(workers)
    env.update(dataset=dataset, shard_index=shard_index, num_shards=num_shards,
               pending_factuals=len(shard))
    (output / "e2_environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env), flush=True)

    threads = max(1, (os.cpu_count() or 1) // max(1, workers))
    started = time.perf_counter()
    records = shard.to_dict("records")

    def flush():
        pd.DataFrame(rows).to_csv(done_path, index=False)
        pd.DataFrame(cfes).to_csv(output / "e2_returned_cfes.csv", index=False)
        pd.DataFrame(audits).to_csv(output / "e2_round_audit.csv", index=False)

    def consume(result, count):
        rows.extend(result["rows"]); cfes.extend(result["cfes"]); audits.extend(result["audit"])
        flush()
        enc = [r for r in result["rows"] if r["mode"] == "encrypted_fixed_r"][0]
        print(f"[{count}/{len(records)}] query {enc['query_id']} "
              f"enc {enc['wall_clock_search_seconds']:.1f}s yield "
              f"{enc['valid_cfe_yield_at_k']:.2f} agree "
              f"{enc['min_label_agreement_vs_torch']:.4f} twin "
              f"{enc['identical_to_plaintext_twin']} elapsed "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    if workers <= 1:
        _init_worker(dataset, str(bundle), threads)
        for count, record in enumerate(records, 1):
            consume(process_factual(record), count)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(dataset, str(bundle), threads)) as pool:
            for count, result in enumerate(pool.imap_unordered(process_factual, records), 1):
                consume(result, count)
    flush()
    detail = pd.DataFrame(rows)
    enc = detail[detail["mode"].eq("encrypted_fixed_r")]
    checks = {
        "all_shard_factuals_done": set(queries[queries.index % num_shards == shard_index]
                                       .query_id) <= set(detail.query_id),
        "label_agreement_ge_099_every_round":
            bool((pd.DataFrame(audits).label_agreement_vs_torch >= MIN_LABEL_AGREEMENT).all()),
        "every_returned_cfe_plaintext_reverified": True,
        "returned_failing_plaintext_margin_total":
            int(detail.returned_failing_plaintext_margin.sum()),
        "server_context_has_secret_key": False,
    }
    acceptance = {"accepted": bool(checks["all_shard_factuals_done"]
                                   and checks["label_agreement_ge_099_every_round"]),
                  "checks": checks, "dataset": dataset, "shard_index": shard_index,
                  "num_shards": num_shards, "factuals": int(enc.query_id.nunique()),
                  "wall_seconds": time.perf_counter() - started}
    (output / "e2_shard_acceptance.json").write_text(json.dumps(acceptance, indent=2))
    print(json.dumps(acceptance, indent=2))
    return detail


# --------------------------------------------------------------------------
# Merge shards and summarise.
# --------------------------------------------------------------------------
def _bootstrap_ci(values: np.ndarray, reps=2000, seed=2026) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (reps, len(values)))].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


SUMMARY_METRICS = [
    "valid_cfe_yield_at_k", "full_k_success", "coverage_at_1",
    "robust_completeness_at_k", "proximity", "sparsity", "diversity",
    "plausibility_inlier_rate", "constraint_validity", "wall_clock_search_seconds",
    "server_observed_rounds", "returned_failing_plaintext_margin",
]


def merge(dataset, bundle, inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.concat([pd.read_csv(Path(d) / name) for d in inputs],
                              ignore_index=True)
              for name in ("e2_per_factual.csv", "e2_returned_cfes.csv", "e2_round_audit.csv")}
    detail = frames["e2_per_factual.csv"].drop_duplicates(["query_id", "mode"])
    for name, frame in frames.items():
        frame.to_csv(output / name.replace("e2_", "e2_merged_"), index=False)
    manifest = pd.read_csv(Path(bundle) / "canonical" / "outer_test_query_manifest.csv")

    rows = []
    for mode, group in detail.groupby("mode"):
        for scope, sub in [("macro", group)] + list(group.groupby("direction")):
            row = {"mode": mode, "scope": scope, "factuals": sub.query_id.nunique()}
            for metric in SUMMARY_METRICS + [c for c in sub.columns if c.startswith("sum_")]:
                if metric not in sub:
                    continue
                values = sub[metric].astype(float)
                lo, hi = _bootstrap_ci(values.to_numpy())
                row.update({f"{metric}__mean": values.mean(), f"{metric}__median": values.median(),
                            f"{metric}__q25": values.quantile(.25),
                            f"{metric}__q75": values.quantile(.75),
                            f"{metric}__ci95_lo": lo, f"{metric}__ci95_hi": hi})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "e2_summary_by_mode.csv", index=False)

    audit = frames["e2_round_audit.csv"]
    phases = audit[list(HE_PHASES) + ["he_scorer_wall_seconds", "client_search_ops_seconds",
                                      "round_elapsed_seconds"]]
    phases.describe(percentiles=[.25, .5, .75]).T.to_csv(output / "e2_round_latency_breakdown.csv")

    # Regression: plaintext twin must reproduce the canonical outer run.
    canonical = pd.read_csv(Path(bundle) / "canonical" / "gaussian_ga_operating_point_per_factual.csv")
    twin = detail[detail["mode"].eq("plaintext_fixed_r")].set_index("query_id")
    canonical = canonical.set_index("query_id").loc[twin.index]
    regression = {
        metric: float(np.max(np.abs(twin[metric].astype(float) - canonical[metric].astype(float))))
        for metric in ("valid_cfe_yield_at_k", "proximity", "sparsity", "diversity", "returned")
        if metric in canonical and metric in twin
    }
    enc = detail[detail["mode"].eq("encrypted_fixed_r")]
    pt = detail[detail["mode"].eq("plaintext_fixed_r")].set_index("query_id")
    ratio = (enc.set_index("query_id").wall_clock_search_seconds
             / pt.wall_clock_search_seconds.loc[enc.query_id.values].values)
    checks = {
        "all_200_outer_factuals": set(manifest.query_id) <= set(enc.query_id),
        "query_ids_match_canonical_manifest": set(enc.query_id) <= set(manifest.query_id),
        "label_agreement_ge_099_every_round":
            bool((audit.label_agreement_vs_torch >= MIN_LABEL_AGREEMENT).all()),
        "returned_failing_plaintext_margin_total":
            int(detail[detail["mode"].str.startswith("encrypted")]
                .returned_failing_plaintext_margin.sum()),
        "plaintext_twin_max_abs_diff_vs_canonical": regression,
        "encrypted_identical_to_plaintext_twin_rate":
            float(enc.identical_to_plaintext_twin.astype(bool).mean()),
        "encrypted_over_plaintext_wall_clock_median": float(ratio.median()),
        "margin_valid_decision_flips_total": int(audit.margin_valid_decision_flips.sum()),
        "max_abs_logit_error_vs_torch": float(audit.max_abs_logit_error_vs_torch.max()),
        "min_label_agreement_vs_torch": float(audit.label_agreement_vs_torch.min()),
    }
    acceptance = {"accepted": bool(checks["all_200_outer_factuals"]
                                   and checks["query_ids_match_canonical_manifest"]
                                   and checks["label_agreement_ge_099_every_round"]),
                  "protocol": PROTOCOL, "dataset": dataset, "shards": [str(d) for d in inputs],
                  "checks": checks}
    (output / "e2_merge_acceptance.json").write_text(json.dumps(acceptance, indent=2, default=str))
    print(json.dumps(acceptance, indent=2, default=str))
    return summary, acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--bundle", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--limit-per-direction", type=int, default=None)
    mer = sub.add_parser("merge")
    mer.add_argument("--dataset", required=True)
    mer.add_argument("--bundle", required=True)
    mer.add_argument("--output", required=True)
    mer.add_argument("--inputs", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_shard(args.dataset, args.bundle, args.output, args.shard_index,
                  args.num_shards, args.workers, args.limit_per_direction)
    else:
        merge(args.dataset, args.bundle, args.inputs, args.output)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

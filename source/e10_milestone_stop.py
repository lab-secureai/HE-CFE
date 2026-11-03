"""E10: milestone stopping and sequential encrypted batches (plaintext replay).

milestone   GA-div at the operating point, seed 11, outer 200 factuals.  The
            search keeps a top-K snapshot after every round, so one run of R
            rounds gives three stopping rules exactly:
              early_stop  first round whose archive holds K CFEs (or R),
              milestone   first milestone in {2, 4, 8, 16} (capped at R) at or
                          after the early-stop round,
              fixed_r     always R rounds.
            E2 showed the encrypted trajectory equals this plaintext twin when
            the CKKS profile is accurate, so quality metrics carry over; the
            encrypted time is rounds x measured HE seconds per round (E2).
            Leakage: Spearman and mutual information between the rounds the
            server observes and the factual's |logit| (same estimator as E6).
batches     Sequential oracle batches per factual: GA-div = rounds, GeCo =
            1 + generations, NICE = oracle_evaluations / batch is not
            sequential-comparable and is reported as evaluations only.

Only aggregate tables plus a per-factual table without feature values are
written, so MIMIC outputs may be shared.

Usage: python3 e10_milestone_stop.py --datasets ecg heartplus mimic adult --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE / "vendor", HERE):
    sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

REVISION = HERE.parent
OPERATING = {"ecg": (64, 4), "heartplus": (64, 16), "mimic": (128, 8), "adult": (64, 16)}
MILESTONES = (2, 4, 8, 16)
SEARCH_SEED = 11
ROBUSTNESS_REPETITIONS = 16
FIDELITY_SAMPLES = 96
METRICS = ("valid_cfe_yield_at_k", "full_k_success", "robust_completeness_at_k",
           "proximity", "sparsity", "diversity", "plausibility_inlier_rate")
E3_DETAIL = "e2_kaggle/e3_{ds}/*merge*/**/e3_per_factual.csv"
OUTER_DETAIL = "e2_kaggle/gadiv_{ds}/*merge*/**/benchmark_per_factual.csv"
# Heart+ uses the N=16384 run: same profile family as the other datasets and exact.
E2_DETAIL = {"ecg": "e2_kaggle/e2gadiv_ecg/*merge*/**/e2_merged_per_factual.csv",
             "heartplus": "e2_kaggle/e2gadiv16k_heartplus/*merge*/**/e2_merged_per_factual.csv",
             "mimic": "e2_kaggle/e2gadiv_mimic/*merge*/**/e2_merged_per_factual.csv",
             "adult": "e2_kaggle/e2gadiv_adult/*merge*/**/e2_merged_per_factual.csv"}
_STATE: dict = {}


def _init(dataset: str) -> None:
    import ga_div_benchmark as bench
    prepared, oracle, projector, _ = bench.load(dataset)
    _STATE.update(dataset=dataset, prepared=prepared, oracle=oracle, projector=projector)


def stop_rounds(archive_sizes: list[int], k: int, rounds: int) -> dict[str, int]:
    hit = [r for r, size in enumerate(archive_sizes, start=1) if size >= k]
    early = hit[0] if hit else rounds
    marks = [m for m in MILESTONES if m < rounds] + [rounds]
    milestone = next(m for m in marks if m >= early)
    return {"early_stop": early, "milestone": milestone, "fixed_r": rounds}


def process(record: dict) -> list[dict]:
    import ga_diverse
    import plaintext_cfe_pipeline as pipe
    from non_gan_v55_runner import _set_metrics

    prepared, oracle, projector = _STATE["prepared"], _STATE["oracle"], _STATE["projector"]
    population, rounds = OPERATING[_STATE["dataset"]]
    query = prepared.x_test[int(record["local_test_index"])]
    desired = int(record["desired_class"])
    config = pipe.SearchConfig(population=population, max_rounds=rounds, k=10, margin=.10)
    run_seed = SEARCH_SEED + 100_000 * int(record["query_id"]) + population
    _, audit, _, snapshots = ga_diverse.search_ga_diverse(
        query, desired, oracle, projector, config, "cpu", run_seed)
    logit = float(pipe.logits_numpy(oracle, query[None, :], "cpu")[0])
    rows = []
    for mode, at in stop_rounds(audit.archive_size.tolist(), config.k, rounds).items():
        trace = audit[audit["round"] <= at]
        metrics = _set_metrics(
            snapshots[at], query, desired, oracle, projector, config, "cpu",
            float(trace.iloc[-1]["cumulative_elapsed_seconds"]), trace,
            robustness_seed=7_000_000 + run_seed + at,
            robustness_repetitions=ROBUSTNESS_REPETITIONS, fidelity_samples=FIDELITY_SAMPLES)
        rows.append({"dataset": _STATE["dataset"], "query_id": int(record["query_id"]),
                     "direction": record["direction"], "abs_query_logit": abs(logit),
                     "mode": mode, "server_observed_rounds": at,
                     **{m: metrics[m] for m in METRICS}})
    return rows


def leakage(observed: np.ndarray, margin: np.ndarray) -> tuple[float, float]:
    if np.ptp(observed) == 0:
        return np.nan, 0.
    rho = float(spearmanr(observed, margin).statistic)
    mi = float(mutual_info_regression(margin.reshape(-1, 1), observed,
                                      discrete_features=False, random_state=SEARCH_SEED)[0])
    return rho, mi


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, mode), sub in detail.groupby(["dataset", "mode"], sort=False):
        # Macro over the two directions, as in every other table.
        macro = sub.groupby("direction")[list(METRICS) + ["server_observed_rounds"]].mean().mean()
        observed = sub.server_observed_rounds.to_numpy(float)
        rho, mi = leakage(observed, sub.abs_query_logit.to_numpy())
        rows.append({"dataset": dataset, "mode": mode, "factuals": len(sub),
                     **macro.to_dict(), "distinct_round_counts": int(len(np.unique(observed))),
                     "spearman_rho_rounds_vs_abs_logit": rho, "mutual_information_nats": mi})
    return pd.DataFrame(rows)


def geco_batches(datasets) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        hits = sorted((REVISION / "results").glob(E3_DETAIL.format(ds=dataset)))
        if not hits:
            continue
        detail = pd.read_csv(hits[0])
        for method, sub in detail.groupby("method"):
            row = {"dataset": dataset, "method": method, "factuals": len(sub),
                   "oracle_evaluations_mean": float(sub.oracle_evaluations.mean())}
            if method == "geco":
                batches = 1 + sub.geco_generations.to_numpy(float)
                row.update(sequential_batches_mean=float(batches.mean()),
                           sequential_batches_median=float(np.median(batches)),
                           sequential_batches_max=float(batches.max()))
            rows.append(row)
    return pd.DataFrame(rows)


def he_batches(summary: pd.DataFrame, geco: pd.DataFrame) -> pd.DataFrame:
    """Sequential batches per factual and an HE time estimate: batches x the
    measured encrypted seconds per round of the GA-div E2 run (fixed R).  One
    ciphertext per feature carries up to N/2 candidates, so a GeCo batch
    (<= a few thousand candidates) costs about one round; GeCo is an estimate."""
    rows = []
    for dataset, (population, rounds) in OPERATING.items():
        hits = sorted((REVISION / "results").glob(E2_DETAIL[dataset]))
        if not hits:
            continue
        e2 = pd.read_csv(hits[0])
        fixed = e2[e2["mode"].eq("encrypted_fixed_r")].wall_clock_search_seconds.mean()
        per_round = float(fixed) / rounds
        for mode, sub in summary[summary.dataset.eq(dataset)].groupby("mode"):
            batches = float(sub.server_observed_rounds.iloc[0])
            rows.append({"dataset": dataset, "method": f"gaussian_ga_div_{mode}",
                         "sequential_batches_mean": batches, "he_seconds_per_round": per_round,
                         "he_seconds_estimate": batches * per_round})
        geco_row = geco[geco.dataset.eq(dataset) & geco.method.eq("geco")]
        if len(geco_row):
            batches = float(geco_row.sequential_batches_mean.iloc[0])
            rows.append({"dataset": dataset, "method": "geco", "sequential_batches_mean": batches,
                         "he_seconds_per_round": per_round,
                         "he_seconds_estimate": batches * per_round})
    return pd.DataFrame(rows)


def canonical(datasets) -> pd.DataFrame:
    """Milestone and fixed-R rows from the Kaggle GA-div outer run, whose
    benchmark stores the top-K snapshot of every factual at rounds 2/4/8/16;
    the early-stop round is rounds_to_k10 of the same run (capped at R)."""
    details = []
    for dataset in datasets:
        population, rounds = OPERATING[dataset]
        hits = sorted((REVISION / "results").glob(OUTER_DETAIL.format(ds=dataset)))
        outer = pd.read_csv(hits[0])
        outer = outer[outer.population.eq(population)]
        first = outer[outer.rounds.eq(max(MILESTONES))].set_index("query_id").rounds_to_k10
        rows = []
        for query_id, sub in outer.groupby("query_id"):
            hit = first[query_id]
            early = int(hit) if pd.notna(hit) and hit <= rounds else rounds
            by_round = sub.set_index("rounds")
            stops = stop_rounds([0] * (early - 1) + [10] * (rounds - early + 1), 10, rounds)
            p = float(np.clip(sub.query_probability.iloc[0], 1e-6, 1 - 1e-6))
            for mode in ("milestone", "fixed_r"):
                at = stops[mode]
                rows.append({"dataset": dataset, "query_id": int(query_id),
                             "direction": sub.direction.iloc[0],
                             "abs_query_logit": abs(np.log(p / (1 - p))), "mode": mode,
                             "server_observed_rounds": at, "early_stop_round": early,
                             **{m: float(by_round.loc[at, m]) for m in METRICS}})
            rows.append({**rows[-1], "mode": "early_stop", "server_observed_rounds": early,
                         **{m: np.nan for m in METRICS}})
        details.append(pd.DataFrame(rows))
    return pd.concat(details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(OPERATING))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="smoke test: first n factuals")
    parser.add_argument("--canonical", action="store_true",
                        help="read the Kaggle outer snapshots instead of replaying locally")
    parser.add_argument("--output", default=str(REVISION / "results" / "e10_milestone_stop"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.canonical:
        detail = canonical(args.datasets)
        detail.to_csv(output / "milestone_canonical_per_factual.csv", index=False)
        summary = summarize(detail)
        summary.to_csv(output / "milestone_canonical_summary.csv", index=False)
        print(summary.round(3).to_string(index=False))
        table = he_batches(summary, geco_batches(args.datasets))
        table.to_csv(output / "sequential_batches_he.csv", index=False)
        print(table.round(2).to_string(index=False))
        return
    import ga_div_benchmark as bench
    details = []
    for dataset in args.datasets:
        outer = bench.load(dataset)[3].sort_values("query_id")
        if args.limit:
            outer = outer.groupby("direction").head(args.limit // 2)
        with mp.get_context("spawn").Pool(args.workers, _init, (dataset,)) as pool:
            parts = pool.map(process, outer.to_dict("records"), chunksize=1)
        detail = pd.DataFrame([row for part in parts for row in part])
        detail.to_csv(output / f"milestone_per_factual_{dataset}.csv", index=False)
        details.append(detail)
        print(summarize(detail).round(3).to_string(index=False), flush=True)
    summarize(pd.concat(details)).to_csv(output / "milestone_summary.csv", index=False)
    batches = geco_batches(args.datasets)
    batches.to_csv(output / "e3_sequential_batches.csv", index=False)
    print(batches.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

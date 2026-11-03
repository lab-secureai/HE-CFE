"""Outer comparison of gaussian_ga_div with the frozen methods.

posthoc   Recompute the DiCE/CARLA metrics added in the revision (count
          diversity, continuous/categorical proximity, redundancy) for the
          frozen methods from their saved outer CFEs (canonical benchmark for
          ECG/Heart+/MIMIC, E4 for Adult).  Read-only on canonical results.
compare   Join frozen per-factual rows (+ post-hoc metrics) with the merged
          GA-div rows and report, for each requested pair of configurations,
          bootstrap 95% CIs (2000 resamples) and paired Wilcoxon signed-rank
          tests over the same factuals.

Usage:
  python3 ga_div_compare.py posthoc --dataset ecg --output <dir>
  python3 ga_div_compare.py compare --dataset ecg --posthoc <dir> --gadiv <merged dir>
      --output <dir> --pairs gaussian_ga:64:4,gaussian_ga_div:64:4 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE / "vendor", HERE):
    sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import ga_div_benchmark as bench
import ga_diverse
import plaintext_cfe_pipeline as pipe

REVISION = HERE.parent
CANONICAL = REVISION.parent / "results" / "benchmark_v55_nongan"
E4 = (REVISION / "results" / "e2_kaggle" / "adult" / "hecfe-e4-adult-plaintext-cfe-benchmark-cpu"
      / "adult_non_gan_cfe_benchmark_paper")
FROZEN_METHODS = ("gaussian_only", "gaussian_ga")
NEW_METRICS = ["count_diversity", "proximity_continuous", "proximity_categorical", "redundancy"]
COMPARE_METRICS = ["valid_cfe_yield_at_k", "robust_completeness_at_k", "proximity", "sparsity",
                   "diversity", "count_diversity", "proximity_continuous",
                   "proximity_categorical", "redundancy", "plausibility_distance",
                   "plausibility_inlier_rate", "target_data_support"]
KEYS = ["query_id", "method", "population", "rounds"]


def frozen_outer(dataset: str) -> Path:
    if dataset == "adult":
        return E4 / "outer_test"
    return CANONICAL / dataset / f"{dataset}_non_gan_cfe_benchmark_paper" / "outer_test"


def posthoc(dataset: str, output: Path) -> None:
    prepared, oracle, projector, _ = bench.load(dataset)
    source = frozen_outer(dataset)
    manifest = pd.read_csv(source / "query_manifest.csv").set_index("query_id")
    columns = [f"model_input__{name}" for name in prepared.feature_names]
    config = pipe.SearchConfig(k=10, margin=.10)
    rows = []
    reader = pd.read_csv(source / "returned_cfes_processed.csv", chunksize=200_000,
                         usecols=KEYS + ["rank"] + columns)
    for chunk in reader:
        chunk = chunk[chunk.method.isin(FROZEN_METHODS)]
        for key, group in chunk.groupby(KEYS):
            record = manifest.loc[key[0]]
            query = prepared.x_test[int(record.local_test_index)]
            cfes = group.sort_values("rank")[columns].to_numpy(np.float32)
            continuous, categorical = ga_diverse.dice_proximity(cfes, query, projector)
            rows.append(dict(zip(KEYS, key),
                             count_diversity=ga_diverse.count_diversity(cfes, projector),
                             proximity_continuous=continuous,
                             proximity_categorical=categorical,
                             redundancy=ga_diverse.redundancy(
                                 cfes, query, int(record.desired_class), oracle, projector,
                                 config, "cpu")))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "frozen_posthoc_metrics.csv", index=False)
    print({"dataset": dataset, "groups": len(rows)})


def bootstrap_ci(values: np.ndarray, seed: int = 2026, resamples: int = 2000):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (resamples, len(values))).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def compare(dataset: str, posthoc_dir: Path, gadiv_dir: Path, output: Path, pairs: list[str]):
    frozen = pd.read_csv(frozen_outer(dataset) / "benchmark_per_factual.csv")
    frozen = frozen[frozen.method.isin(FROZEN_METHODS)]
    extra = pd.read_csv(posthoc_dir / "frozen_posthoc_metrics.csv")
    frozen = frozen.drop(columns=[c for c in NEW_METRICS if c in frozen]).merge(extra, on=KEYS, how="left")
    detail = pd.concat([frozen, pd.read_csv(gadiv_dir / "benchmark_per_factual.csv")],
                       ignore_index=True)
    rows = []
    for pair in pairs:
        (ma, pa, ra), (mb, pb, rb) = [tuple(p.split(":")) for p in pair.split(",")]
        a = detail[detail.method.eq(ma) & detail.population.eq(int(pa)) & detail.rounds.eq(int(ra))]
        b = detail[detail.method.eq(mb) & detail.population.eq(int(pb)) & detail.rounds.eq(int(rb))]
        joined = a.set_index("query_id").join(b.set_index("query_id"), lsuffix="_a", rsuffix="_b",
                                              how="inner")
        for metric in COMPARE_METRICS:
            x, y = joined[f"{metric}_a"].to_numpy(float), joined[f"{metric}_b"].to_numpy(float)
            both = np.isfinite(x) & np.isfinite(y)
            difference = y[both] - x[both]
            p_value = (float(wilcoxon(difference, zero_method="zsplit").pvalue)
                       if both.sum() > 0 and np.any(difference != 0) else 1.0)
            rows.append({"dataset": dataset, "a": f"{ma}:P{pa}:R{ra}", "b": f"{mb}:P{pb}:R{rb}",
                         "metric": metric, "paired_n": int(both.sum()),
                         "mean_a": float(np.nanmean(x)), "mean_b": float(np.nanmean(y)),
                         "ci95_a": bootstrap_ci(x), "ci95_b": bootstrap_ci(y),
                         "mean_diff_b_minus_a": float(difference.mean()) if both.any() else np.nan,
                         "ci95_diff": bootstrap_ci(difference), "wilcoxon_p": p_value})
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / f"ga_div_paired_tests_{dataset}.csv", index=False)
    print(table[["a", "b", "metric", "paired_n", "mean_a", "mean_b", "wilcoxon_p"]]
          .round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["posthoc", "compare"])
    parser.add_argument("--dataset", required=True, choices=["ecg", "heartplus", "mimic", "adult"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--posthoc")
    parser.add_argument("--gadiv")
    parser.add_argument("--pairs", nargs="+", default=[])
    args = parser.parse_args()
    if args.action == "posthoc":
        posthoc(args.dataset, Path(args.output))
    else:
        compare(args.dataset, Path(args.posthoc), Path(args.gadiv), Path(args.output), args.pairs)


if __name__ == "__main__":
    main()

"""Summarise the fresh confirmation cohort (ga_div_benchmark --stage confirm).

For each dataset, at the operating point (P, R) used on the outer cohort:
macro mean with bootstrap 95% CI (2000 resamples) per method, and paired
Wilcoxon signed-rank tests of gaussian_ga_div against gaussian_ga and
gaussian_only over the same factuals (factuals with a missing metric, e.g.
proximity when nothing was returned, are dropped pairwise).  Aggregates only,
so the MIMIC rows may be shared.

Usage: python3 confirm_summary.py [--output <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
OPERATING = {"ecg": (64, 4), "heartplus": (64, 16), "mimic": (128, 8), "adult": (64, 16)}
METRICS = ("valid_cfe_yield_at_k", "robust_completeness_at_k", "proximity", "sparsity",
           "diversity", "plausibility_inlier_rate")
METHODS = ("gaussian_ga", "gaussian_ga_div", "gaussian_only")


def _ci(values, reps=2000, seed=2026):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (reps, len(values))).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REVISION / "results" / "confirm_cohort"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary, tests = [], []
    for dataset, (population, rounds) in OPERATING.items():
        path = next((REVISION / "results" / "e2_kaggle" / f"conf_{dataset}").glob(
            "*merge*/**/benchmark_per_factual.csv"))
        detail = pd.read_csv(path)
        detail = detail[detail.population.eq(population) & detail.rounds.eq(rounds)]
        table = {m: detail[detail.method.eq(m)].set_index("query_id") for m in METHODS}
        for method, rows in table.items():
            for metric in METRICS:
                values = rows[metric].dropna().to_numpy(float)
                low, high = _ci(values)
                summary.append({"dataset": dataset, "population": population, "rounds": rounds,
                                "method": method, "metric": metric, "n": len(values),
                                "mean": float(values.mean()), "ci95_low": low, "ci95_high": high})
        for other in ("gaussian_ga", "gaussian_only"):
            for metric in METRICS:
                pair = pd.concat([table["gaussian_ga_div"][metric].rename("a"),
                                  table[other][metric].rename("b")], axis=1).dropna()
                diff = pair.a - pair.b
                p = float(wilcoxon(pair.a, pair.b).pvalue) if (diff != 0).any() else 1.
                low, high = _ci(diff.to_numpy(float))
                tests.append({"dataset": dataset, "a": "gaussian_ga_div", "b": other,
                              "metric": metric, "paired_n": len(pair),
                              "mean_diff_a_minus_b": float(diff.mean()),
                              "ci95_low": low, "ci95_high": high, "wilcoxon_p": p})
    pd.DataFrame(summary).to_csv(output / "confirm_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(output / "confirm_paired_tests.csv", index=False)
    wide = pd.DataFrame(summary).pivot_table(index=["dataset", "method"], columns="metric",
                                             values="mean")
    print(wide.round(3).to_string())
    print(pd.DataFrame(tests).query("metric in ['valid_cfe_yield_at_k', 'robust_completeness_at_k']")
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()

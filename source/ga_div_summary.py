"""Summarise ga_div_benchmark outputs: one line per dataset/method/P at given rounds.

Usage: python3 ga_div_summary.py <results dir> [--rounds 16] [--csv out.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = {
    "valid_cfe_yield_at_k": "yield", "robust_completeness_at_k": "robust",
    "proximity": "prox", "sparsity": "spars", "diversity": "div",
    "count_diversity": "cdiv", "plausibility_distance": "plaus_d",
    "plausibility_inlier_rate": "lof_in", "target_data_support": "support",
    "minimum_target_logit_margin": "min_margin",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--csv")
    args = parser.parse_args()
    tables = []
    for path in sorted(Path(args.root).glob("*/benchmark_per_factual.csv")):
        detail = pd.read_csv(path)
        detail = detail[detail.rounds.eq(args.rounds)]
        present = {k: v for k, v in COLUMNS.items() if k in detail}
        table = detail.groupby(["method", "population"])[list(present)].mean().rename(columns=present)
        worst = detail.groupby(["method", "population", "direction"]).valid_cfe_yield_at_k.mean()
        table.insert(1, "worst_dir", worst.groupby(level=[0, 1]).min())
        table.insert(0, "dataset", path.parent.name)
        tables.append(table.reset_index())
    result = pd.concat(tables, ignore_index=True)
    print(result.round(3).to_string(index=False))
    if args.csv:
        result.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()

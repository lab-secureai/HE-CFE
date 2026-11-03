"""E6: leakage left after encryption (R2.W5, R3.W6), plaintext and cheap.

extraction  The client decrypts one logit per candidate.  Replay the GA-div
            search at the operating point on the first n outer factuals,
            record every (candidate, logit) pair the client sees, and fit
            surrogates on them.  Report label agreement with the frozen
            oracle on the whole test split, for n in N_FACTUALS and three
            feedback types: full logit, logit rounded to QUANT_STEP, label only.
rounds      Early-stop leaks the round at which K was reached.  Spearman rho
            and mutual information between that round (E2 encrypted runs) and
            the factual's |logit|; fixed-R sends R rounds for every factual.
quantized   Mitigation cost: re-run the search with logits rounded to
            QUANT_STEP and report Yield@K under the true oracle.
noise       Mitigation E8: the server adds fresh Gaussian noise N(0, sigma^2)
            to every encrypted logit before returning it (a free CKKS
            addition).  The client accepts a candidate only when its noisy
            logit clears m + 2 sigma.  Report Yield@K and returned CFEs failing
            the true margin m (true oracle), and surrogate label agreement
            from the noisy transcripts, on the same first n factuals.

Only aggregate tables are written, so MIMIC outputs may be shared.

Usage: python3 e6_leakage.py --datasets ecg heartplus mimic adult --output <dir>
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
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

import ga_div_benchmark as bench
import ga_diverse
import plaintext_cfe_pipeline as pipe

REVISION = HERE.parent
OPERATING = {"ecg": (64, 4), "heartplus": (64, 16), "mimic": (128, 8), "adult": (64, 16)}
N_FACTUALS = (1, 5, 10, 25, 50, 100)
FEEDBACK = ("logit", "quantized", "label")
QUANT_STEP = .5
SIGMAS = (.05, .1, .2, .5)
NOISE_FACTUALS = (1, 25, 100)
SEED = 11
E2_DETAIL = {
    "ecg": "e2_kaggle/ecg/*merge*/**/e2_merged_per_factual.csv",
    "heartplus": "e2_kaggle/heartplus/*merge*/**/e2_merged_per_factual.csv",
    "mimic": "e2_kaggle/mimic/*merge*/**/e2_merged_per_factual.csv",
    "heartplus_ga_div": "e2_kaggle/e2gadiv_heartplus/*merge*/**/e2_merged_per_factual.csv",
    "adult_ga_div": "e2_kaggle/e2gadiv_adult/*merge*/**/e2_merged_per_factual.csv",
    "ecg_ga_div": "e2_kaggle/e2gadiv_ecg/*merge*/**/e2_merged_per_factual.csv",
    "mimic_ga_div": "e2_kaggle/e2gadiv_mimic/*merge*/**/e2_merged_per_factual.csv",
    "heartplus_ga_div_n16384": "e2_kaggle/e2gadiv16k_heartplus/*merge*/**/e2_merged_per_factual.csv",
}


def _search(record, prepared, oracle, projector, transform=None):
    population, rounds = OPERATING[record["dataset"]]
    query = prepared.x_test[int(record["local_test_index"])]
    config = pipe.SearchConfig(population=population, max_rounds=rounds, k=10, margin=.10)
    seen = []

    def logit_fn(candidate):
        logit = pipe.logits_numpy(oracle, candidate, "cpu")
        seen.append((candidate.copy(), logit.copy()))
        return logit if transform is None else transform(logit)

    final, *_ = ga_diverse.search_ga_diverse(
        query, int(record["desired_class"]), oracle, projector, config, "cpu",
        SEED + 100_000 * int(record["query_id"]) + population, logit_fn=logit_fn)
    return final, seen, query, config


def _quantize(logit):
    return np.round(logit / QUANT_STEP) * QUANT_STEP


def _surrogates(x, logit, feedback, seed):
    if feedback == "label":
        labels = (logit >= 0).astype(int)
        if labels.min() == labels.max():
            return None
        return [("logistic", LogisticRegression(max_iter=2000).fit(x, labels)),
                ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed).fit(x, labels))]
    target = _quantize(logit) if feedback == "quantized" else logit
    return [("linear", LinearRegression().fit(x, target)),
            ("mlp", MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=seed).fit(x, target))]


def extraction(dataset, output):
    prepared, oracle, projector, outer = bench.load(dataset)
    outer = outer.sort_values("query_id").assign(dataset=dataset)
    # Alternate directions so every prefix of n factuals is balanced.
    outer = (outer.assign(_rank=outer.groupby("direction").cumcount())
             .sort_values(["_rank", "direction"]).head(max(N_FACTUALS)))
    truth = pipe.logits_numpy(oracle, prepared.x_test, "cpu") >= 0
    transcripts, quantized = [], []
    for record in outer.to_dict("records"):
        final, seen, query, config = _search(record, prepared, oracle, projector)
        transcripts.append(seen)
        q_final, *_ = _search(record, prepared, oracle, projector, _quantize)
        for name, cfes in (("logit", final), ("quantized", q_final)):
            valid = 0
            if len(cfes):
                ok = pipe._target_valid(pipe.logits_numpy(oracle, cfes, "cpu"),
                                        int(record["desired_class"]), config.margin)
                valid = int((ok & projector.feasibility(cfes, query)).sum())
            quantized.append({"dataset": dataset, "query_id": record["query_id"],
                              "feedback": name, "valid_cfe_yield_at_k": min(valid, 10) / 10})
    rows = []
    for n in N_FACTUALS:
        pairs = [p for seen in transcripts[:n] for p in seen]
        x = np.concatenate([c for c, _ in pairs])
        logit = np.concatenate([l for _, l in pairs])
        for feedback in FEEDBACK:
            models = _surrogates(x, logit, feedback, SEED)
            for name, model in models or []:
                predicted = model.predict(prepared.x_test)
                predicted = predicted >= (.5 if feedback == "label" else 0)
                rows.append({"dataset": dataset, "factuals": n, "queries": len(x),
                             "feedback": feedback, "surrogate": name,
                             "test_label_agreement": float(np.mean(predicted == truth)),
                             "majority_baseline": float(max(truth.mean(), 1 - truth.mean()))})
        print(dataset, n, len(x), flush=True)
    pd.DataFrame(rows).to_csv(output / f"extraction_{dataset}.csv", index=False)
    pd.DataFrame(quantized).to_csv(output / f"quantized_search_{dataset}.csv", index=False)


def _balanced_outer(dataset, outer):
    outer = outer.sort_values("query_id").assign(dataset=dataset)
    return (outer.assign(_rank=outer.groupby("direction").cumcount())
            .sort_values(["_rank", "direction"]).head(max(N_FACTUALS)))


def noise(dataset, output):
    import dataclasses

    prepared, oracle, projector, outer = bench.load(dataset)
    outer = _balanced_outer(dataset, outer)
    truth = pipe.logits_numpy(oracle, prepared.x_test, "cpu") >= 0
    population, rounds_cap = OPERATING[dataset]
    search_rows, extraction_rows = [], []
    for sigma in SIGMAS:
        transcripts = []
        for record in outer.to_dict("records"):
            query = prepared.x_test[int(record["local_test_index"])]
            desired = int(record["desired_class"])
            config = pipe.SearchConfig(population=population, max_rounds=rounds_cap, k=10,
                                       margin=.10)
            guarded = dataclasses.replace(config, margin=config.margin + 2 * sigma)
            rng = np.random.default_rng(SEED * 1_000_003 + int(record["query_id"]) * 101
                                        + int(round(sigma * 1000)))
            seen = []

            def logit_fn(candidate, rng=rng, seen=seen):
                noisy = pipe.logits_numpy(oracle, candidate, "cpu") + rng.normal(
                    0, sigma, len(candidate)).astype(np.float32)
                seen.append((candidate.copy(), noisy.copy()))
                return noisy

            final, *_ = ga_diverse.search_ga_diverse(
                query, desired, oracle, projector, guarded, "cpu",
                SEED + 100_000 * int(record["query_id"]) + population, logit_fn=logit_fn)
            valid = failing = 0
            if len(final):
                ok = pipe._target_valid(pipe.logits_numpy(oracle, final, "cpu"), desired,
                                        config.margin)
                failing = int((~ok).sum())
                valid = int((ok & projector.feasibility(final, query)).sum())
            search_rows.append({"dataset": dataset, "sigma": sigma, "query_id": record["query_id"],
                                "returned": int(len(final)), "returned_failing_true_margin": failing,
                                "valid_cfe_yield_at_k": min(valid, 10) / 10,
                                "oracle_queries": int(sum(len(c) for c, _ in seen))})
            transcripts.append(seen)
        for n in NOISE_FACTUALS:
            pairs = [p for seen in transcripts[:n] for p in seen]
            x = np.concatenate([c for c, _ in pairs])
            noisy = np.concatenate([l for _, l in pairs])
            for feedback in ("logit", "label"):
                for name, model in _surrogates(x, noisy, feedback, SEED) or []:
                    predicted = model.predict(prepared.x_test)
                    predicted = predicted >= (.5 if feedback == "label" else 0)
                    extraction_rows.append({
                        "dataset": dataset, "sigma": sigma, "factuals": n, "queries": len(x),
                        "feedback": f"noisy_{feedback}", "surrogate": name,
                        "test_label_agreement": float(np.mean(predicted == truth)),
                        "majority_baseline": float(max(truth.mean(), 1 - truth.mean()))})
        print(dataset, sigma, pd.DataFrame(search_rows).query("sigma == @sigma")
              .valid_cfe_yield_at_k.mean(), flush=True)
    pd.DataFrame(search_rows).to_csv(output / f"noise_search_{dataset}.csv", index=False)
    pd.DataFrame(extraction_rows).to_csv(output / f"noise_extraction_{dataset}.csv", index=False)


def rounds(output):
    rows = []
    for name, pattern in E2_DETAIL.items():
        hits = sorted((REVISION / "results").glob(pattern))
        if not hits:
            continue
        detail = pd.read_csv(hits[0])
        for mode in ("encrypted_early_stop", "encrypted_fixed_r"):
            sub = detail[detail["mode"].eq(mode)]
            # |logit| of the factual = logit(p) of the stored query probability.
            p = sub.query_probability.clip(1e-6, 1 - 1e-6)
            margin = np.abs(np.log(p / (1 - p))).to_numpy()
            observed = sub.server_observed_rounds.to_numpy(float)
            constant = np.ptp(observed) == 0
            rho = np.nan if constant else float(spearmanr(observed, margin).statistic)
            mi = 0. if constant else float(mutual_info_regression(
                margin.reshape(-1, 1), observed, discrete_features=False, random_state=SEED)[0])
            rows.append({"run": name, "mode": mode, "factuals": len(sub),
                         "distinct_round_counts": int(len(np.unique(observed))),
                         "mean_rounds": float(observed.mean()),
                         "spearman_rho_rounds_vs_abs_logit": rho,
                         "mutual_information_nats": mi})
    pd.DataFrame(rows).to_csv(output / "round_leakage.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(OPERATING))
    parser.add_argument("--output", default=str(REVISION / "results" / "e6_leakage"))
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--noise", action="store_true", help="run only the E8 noise mitigation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.noise:
        for dataset in args.datasets:
            noise(dataset, output)
        return
    rounds(output)
    if not args.skip_extraction:
        for dataset in args.datasets:
            extraction(dataset, output)


if __name__ == "__main__":
    main()

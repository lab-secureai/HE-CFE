"""E2 with the diversity-preserving GA: encrypted end-to-end search on Heart+.

Reuses `e2_encrypted_search` unchanged and swaps only the search routine:
the encrypted run is `ga_diverse.search_ga_diverse` whose logits come from
the CKKS scorer (one encrypted population per round, as in E2), and the
plaintext twin is the same routine with plaintext logits and the same seed.
The plaintext shadow logits and the scorer's own plaintext reference are
computed inside the logit hook but removed from the encrypted clock, exactly
as `encrypted_gaussian_ga` does.  The patches are applied at import, so the
spawned pool workers (which unpickle `e2_gadiv.process_factual`) see them.

Usage: python3 e2_gadiv.py --dataset heartplus --bundle <dir> --output <dir>
           --population 64 [--shard-index i --num-shards n --workers 2]
"""

from __future__ import annotations

import argparse
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

import adult_adapter  # noqa: F401  registers Adult; other datasets delegate unchanged
import e2_encrypted_search as es
import ga_diverse
import non_gan_v55_runner as runner

METHOD = "gaussian_ga_div"
POPULATION = int(os.environ.get("E2_GADIV_POPULATION", "64"))
# Adult has no CKKS run in the canonical study.  It uses the MIMIC profile
# (N=16384, scale 2^40) rather than the Heart+ N=8192 profile, whose logit
# error (~0.02) was shown in E2 to flip margin decisions.
ADULT_PROFILE = "poly_n16384_scale40"
ADULT_OUTER_SEED = 707
_frozen_search_distribution = runner.search_distribution
_frozen_init_worker = es._init_worker
_frozen_process_factual = es.process_factual


def encrypted_ga_div(query, desired, scorer, oracle, projector, config, seed):
    import plaintext_cfe_pipeline as pipe

    calls = []

    def logit_fn(candidate):
        started = time.perf_counter()
        logit, he = scorer(candidate)
        scorer_seconds = time.perf_counter() - started
        shadow_started = time.perf_counter()
        plain = pipe.logits_numpy(oracle, candidate, "cpu")
        shadow = time.perf_counter() - shadow_started
        calls.append({"candidate": candidate.copy(), "logit": np.asarray(logit, float),
                      "plain": plain, "he": he, "scorer_seconds": scorer_seconds,
                      "excluded": shadow + he["plaintext_reference_inference_seconds"]})
        return logit

    _, audit, runtime, snapshots = ga_diverse.search_ga_diverse(
        query, desired, oracle, projector, config, "cpu", seed, logit_fn=logit_fn)
    excluded = np.array([c["excluded"] for c in calls])
    audit["round_elapsed_seconds"] -= excluded
    audit["cumulative_elapsed_seconds"] -= np.cumsum(excluded)
    extra = []
    for call, round_seconds in zip(calls, audit["round_elapsed_seconds"]):
        he, logit, plain = call["he"], call["logit"], call["plain"]
        feasible = projector.feasibility(call["candidate"], query)
        valid = pipe._target_valid(logit, desired, config.margin) & feasible
        plain_valid = pipe._target_valid(plain, desired, config.margin) & feasible
        he_wall = call["scorer_seconds"] - he["plaintext_reference_inference_seconds"]
        extra.append({
            "he_scorer_wall_seconds": float(he_wall),
            "client_search_ops_seconds": float(round_seconds - he_wall),
            **{phase: float(he[phase]) for phase in es.HE_PHASES},
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
    audit = pd.concat([audit.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    return snapshots, audit, float(runtime - excluded.sum())


def search_distribution(query, desired, oracle, projector, config, device, seed,
                        distribution="gaussian", genetic=False):
    if genetic:
        return ga_diverse.search_ga_diverse(query, desired, oracle, projector, config,
                                            device, seed, distribution=distribution)
    return _frozen_search_distribution(query, desired, oracle, projector, config, device,
                                       seed, distribution=distribution, genetic=genetic)


def _init_worker(dataset: str, bundle: str, threads: int) -> None:
    _frozen_init_worker(dataset, bundle, threads)
    point = dict(es._STATE["point"])
    point["population"] = POPULATION
    es._STATE["point"] = point


def process_factual(record: dict) -> dict:
    result = _frozen_process_factual(record)
    for row in result["rows"] + result["cfes"]:
        row["method"] = METHOD
    return result


def prepare_adult_bundle(training_root, gadiv_merged, output) -> Path:
    """Lay out the E4 Adult training output in the E2 bundle format.

    training_root: folder holding paper_training_handoff/, raw/ and
    training_acceptance.json.  gadiv_merged: GA-div outer merge, whose
    P=64, R=16 rows are the plaintext reference for the twin regression.
    The outer cohort is the E4/GA-div one (stratified, seed 707).
    """
    import json
    import shutil

    import e4_adult
    import plaintext_cfe_pipeline as pipe

    training_root, output = Path(training_root), Path(output)
    if output.exists():
        shutil.rmtree(output)
    (output / "canonical").mkdir(parents=True)
    shutil.copytree(training_root / "paper_training_handoff", output / "handoff")
    shutil.copy2(training_root / "training_acceptance.json", output / "handoff")
    shutil.copytree(training_root / "raw", output / "raw")
    prepared, oracle, _ = e4_adult.load_handoff(training_root)
    outer = pipe.select_stratified_bidirectional_queries(prepared, oracle, "cpu",
                                                         per_direction=100, seed=ADULT_OUTER_SEED)
    outer.to_csv(output / "canonical" / "outer_test_query_manifest.csv", index=False)
    detail = pd.read_csv(Path(gadiv_merged) / "benchmark_per_factual.csv")
    detail = detail[detail.method.eq(METHOD) & detail.population.eq(POPULATION)
                    & detail.rounds.eq(16)]
    if set(detail.query_id) != set(outer.query_id):
        raise RuntimeError("GA-div reference rows do not match the Adult outer cohort")
    detail.to_csv(output / "canonical" / "gaussian_ga_operating_point_per_factual.csv", index=False)
    (output / "operating_point.json").write_text(json.dumps({
        "dataset": "adult", "population": POPULATION, "rounds": 16, "profile": ADULT_PROFILE,
        "k": 10, "margin": .1, "search_seed": 11, "classifier_seed": 55,
        "method": METHOD}, indent=2))
    return output


def swap_reference(bundle, gadiv_merged, profile=None) -> Path:
    """Point the twin regression of `es.merge` at the GA-div outer run.

    The ECG/MIMIC E2 bundles carry the frozen Gaussian-GA as canonical
    reference; GA-div at the same P and R (round snapshot) replaces it, so
    `plaintext_twin_max_abs_diff_vs_canonical` must be 0 as for Adult.
    `profile` overrides the CKKS profile (Heart+ at N=16384: controlled test
    of the profile effect, same data, model, cohort and search).
    """
    import json

    bundle = Path(bundle)
    point = json.loads((bundle / "operating_point.json").read_text())
    detail = pd.read_csv(Path(gadiv_merged) / "benchmark_per_factual.csv")
    detail = detail[detail.method.eq(METHOD) & detail.population.eq(POPULATION)
                    & detail.rounds.eq(int(point["rounds"]))]
    outer = pd.read_csv(bundle / "canonical" / "outer_test_query_manifest.csv")
    if set(detail.query_id) != set(outer.query_id):
        raise RuntimeError("GA-div reference rows do not match the outer cohort")
    detail.to_csv(bundle / "canonical" / "gaussian_ga_operating_point_per_factual.csv", index=False)
    point.update(population=POPULATION, method=METHOD)
    if profile:
        point["profile"] = profile
    (bundle / "operating_point.json").write_text(json.dumps(point, indent=2))
    return bundle


es.encrypted_gaussian_ga = encrypted_ga_div
runner.search_distribution = search_distribution
es._init_worker = _init_worker
es.process_factual = process_factual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-per-direction", type=int, default=None)
    args = parser.parse_args()
    # Spawned workers read the population from the environment at import.
    os.environ["E2_GADIV_POPULATION"] = str(args.population)
    globals()["POPULATION"] = args.population
    es.run_shard(args.dataset, args.bundle, args.output, args.shard_index, args.num_shards,
                 args.workers, args.limit_per_direction)


if __name__ == "__main__":
    main()

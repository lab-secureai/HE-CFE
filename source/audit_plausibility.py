"""Audit of what saturated Yield@K means (2026-10-30).

Uses the frozen preprocessor, split and seed-55 weights of the canonical
training outputs and the CFEs returned by the Kaggle E2 GA-div merges.
No search is rerun. Refits LOF with the local sklearn, so its output is compared
with the Kaggle inlier rate and reported as exploratory.
"""
import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import plaintext_cfe_pipeline as pipe
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[2]
E2 = ROOT / "REVISION_2026-10/results/e2_kaggle"
OUT = ROOT / "REVISION_2026-10/results/plausibility_audit"
SETS = {"ecg": "e2gadiv_ecg", "heartplus": "e2gadiv16k_heartplus", "mimic": "e2gadiv_mimic"}
NOISE = (0.0, 0.01, 0.02, 0.05, 0.12)
N_REAL = 4000


def forward(x, p):
    if str(p["activation_kind"]) == "square":
        act = lambda v: v * v
    else:
        act = lambda v: v + float(p["activation_alpha"]) * v * v
    v = act(x @ p["fc1_weight"].T + p["fc1_bias"])
    v = act(v @ p["fc2_weight"].T + p["fc2_bias"])
    return (v @ p["fc3_weight"].T + p["fc3_bias"]).reshape(-1)


def audit(ds):
    train = ROOT / f"results/training_v55_3seed/{ds}/{ds}_v55_mlp_training_3seed/paper_training_handoff"
    raw = pipe.load_dataset(ds, ROOT / "data" / ds / pipe.DATA_FILENAMES[ds])
    frozen = joblib.load(train / "preprocessor_and_split.joblib")
    split, pre = frozen["split"], frozen["preprocessor"]
    y = raw.frame[raw.target_col].to_numpy(np.int8)
    x_tr = np.asarray(pre.transform(raw.frame.iloc[split.train_idx]), np.float64)
    x_te = np.asarray(pre.transform(raw.frame.iloc[split.test_idx]), np.float64)
    y_te = y[split.test_idx]
    params = dict(np.load(ROOT / f"results/he_population_kaggle/{ds}/{ds}_non_gan_he_population_paper/he_add_mul_parameters.npz", allow_pickle=False))
    low, high = np.quantile(x_tr, .005, axis=0), np.quantile(x_tr, .995, axis=0)
    scale = np.maximum(high - low, .10)
    subset = np.random.default_rng(404).choice(len(x_tr), min(20000, len(x_tr)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=.10, n_jobs=-1).fit(
        x_tr[subset].astype(np.float32))
    inlier = lambda a: (lof.predict(np.asarray(a, np.float32)) == 1)

    rng = np.random.default_rng(2026)
    pick = rng.choice(len(x_te), min(N_REAL, len(x_te)), replace=False)
    real = x_te[pick]
    rows = [{"dataset": ds, "item": f"real_test_noise_{s}", "inlier_rate": float(inlier(
        np.clip(real + rng.normal(size=real.shape) * scale * s, low, high) if s else real).mean())}
        for s in NOISE]

    path = glob.glob(str(E2 / SETS[ds] / "**/e2_merged_returned_cfes.csv"), recursive=True)[0]
    cfe = pd.read_csv(path)
    cfe = cfe[cfe["mode"] == "encrypted_fixed_r"]
    cols = [c for c in cfe if c.startswith("model_input__")]
    X = cfe[cols].to_numpy(np.float64)
    per = pd.read_csv(path.replace("returned_cfes", "per_factual"))
    per = per[per["mode"] == "encrypted_fixed_r"]
    rows.append({"dataset": ds, "item": "cfe_refit_lof", "inlier_rate": float(inlier(X).mean())})
    rows.append({"dataset": ds, "item": "cfe_kaggle_reported", "inlier_rate": float(per.plausibility_inlier_rate.mean())})

    # Local validity tests the same logit sign and margin on the returned points.
    logit_real = forward(x_te, params)
    logit_cfe = forward(X, params)
    q = np.abs(logit_real)
    stats = {"dataset": ds, "test_accuracy": float(((logit_real >= 0) == (y_te == 1)).mean()),
             "real_abs_logit_p50": float(np.median(q)), "real_abs_logit_p99": float(np.quantile(q, .99)),
             "cfe_abs_logit_p50": float(np.median(np.abs(logit_cfe))),
             "cfe_above_real_p99": float((np.abs(logit_cfe) > np.quantile(q, .99)).mean()),
             "cfe_margin_valid": float((np.where(cfe.desired_class == 1, logit_cfe, -logit_cfe) >= .1).mean())}

    # Distance from each factual to its nearest real test points of the target
    # prediction, against the distance to its returned CFEs, in the same unit.
    query = x_te[cfe.groupby("query_id").local_partition_index.first().to_numpy().astype(int)]
    ids = cfe.groupby("query_id").local_partition_index.first().index.to_numpy()
    desired = cfe.groupby("query_id").desired_class.first().to_numpy()
    mean_real, mean_cfe, inside_real = [], [], []
    for i, qid in enumerate(ids):
        d = np.mean(np.abs(x_te - query[i]) / scale, axis=1)
        pool = (logit_real >= .1) if desired[i] == 1 else (logit_real <= -.1)
        nearest = np.sort(d[pool])[:10]
        mine = X[cfe.query_id.to_numpy() == qid]
        mean_real.append(nearest.mean())
        mean_cfe.append(np.mean(np.abs(mine - query[i]) / scale, axis=1).mean())
        inside_real.append(np.mean(np.mean(np.abs(mine - query[i]) / scale, axis=1) < nearest[0]))
    stats.update(nearest_real_target_distance=float(np.mean(mean_real)),
                 cfe_distance=float(np.mean(mean_cfe)),
                 cfe_closer_than_nearest_real=float(np.mean(inside_real)))
    return rows, stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, stats = [], []
    for ds in SETS:
        r, s = audit(ds)
        rows += r
        stats.append(s)
        pd.DataFrame(rows).to_csv(OUT / "lof_inlier_rates.csv", index=False)
        pd.DataFrame(stats).to_csv(OUT / "cfe_vs_real_data.csv", index=False)
        print(json.dumps(s, indent=0), flush=True)
    print(pd.DataFrame(rows).pivot(index="item", columns="dataset", values="inlier_rate").round(3))


main()

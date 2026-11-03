"""Register UCI Adult Income as a fourth dataset of the frozen V5.5 pipeline.

E4 needs plaintext CFE results on a non-medical domain with the *same* search
code.  The vendored `plaintext_cfe_pipeline.py` is not edited; this module
extends its dataset tables and wraps the two dataset-specific entry points
(`load_dataset`, `DomainProjector`).  For ecg/heartplus/mimic every wrapped
call delegates unchanged to the original implementation.

Encoding contract (decided before any CFE result was seen):
- Rows: adult.data + adult.test = 48,842 (test labels lose the trailing dot);
  `?` is kept as a categorical level "Missing" (no row dropped).
- Dropped raw columns: fnlwgt (census sampling weight, not a person attribute)
  and education (one-to-one with education_num).
- Numeric (quantile-normal, as GenericPreprocessor): age, education_num,
  capital_gain, capital_loss, hours_per_week.
- Categorical (one-hot, drop first): workclass, marital_status, occupation,
  relationship, race, sex, native_country grouped to United-States / Other.
- Immutable: age, race, sex, native_country.  education_num may not decrease.
  Numeric raw values are integers (rounded after projection).
- Split: stratified rows (no person identifier), identical to MIMIC.
- Classifier: MIMIC architecture (128-64, square activation, sqrt-weighted
  BCE, dropout 0.05), seeds 55/66/77, seed 55 frozen as the CFE oracle.
- Target: income >50K = 1.  Directions keep the pipeline names
  (`disease_to_no_disease` = 1 -> 0, `no_disease_to_disease` = 0 -> 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import plaintext_cfe_pipeline as pipe

DATASET = "adult"
TARGET = "income_gt_50k"
FILENAME = "adult_income_48842.csv"
RAW_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "sex",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race",
               "sex", "native_country"]
IMMUTABLE_PREFIXES = ("race_", "sex_", "native_country_")
EXPECTED_INPUTS = 45

pipe.TARGETS[DATASET] = TARGET
pipe.DATA_FILENAMES[DATASET] = FILENAME
pipe.DATASET_SLUGS[DATASET] = "hecfe-e4-adult-inputs"
pipe.CLASSIFIER_CONFIGS[DATASET] = dict(
    h1=128, h2=64, activation_kind="square", alpha=0.0, loss="sqrt_weighted_bce",
    dropout=0.05, expected_inputs=EXPECTED_INPUTS)


def build_csv(adult_dir: str | Path, output: str | Path) -> Path:
    """Deterministically merge the raw UCI files into one CSV."""
    adult_dir = Path(adult_dir)
    frames = []
    for name, skip in (("adult.data", 0), ("adult.test", 1)):
        # Kaggle bundles store the same bytes as adult_data.csv / adult_test.csv.
        path = adult_dir / name
        if not path.exists():
            path = adult_dir / (name.replace(".", "_") + ".csv")
        frame = pd.read_csv(path, header=None, names=RAW_COLUMNS,
                            skipinitialspace=True, na_values="?", skiprows=skip)
        frame = frame.dropna(subset=["income"])
        frame["income"] = frame["income"].str.rstrip(".")
        frame["source_file"] = name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != 48_842 or set(frame.income) != {"<=50K", ">50K"}:
        raise ValueError("Unexpected Adult row count or labels")
    frame[TARGET] = (frame.income == ">50K").astype(np.int8)
    country = frame.native_country
    grouped = pd.Series(np.where(country.eq("United-States"), "United-States", "Other"),
                        index=frame.index, dtype=object)
    grouped[country.isna()] = np.nan
    frame["native_country"] = grouped
    keep = NUMERIC + CATEGORICAL + [TARGET]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(output, index=False)
    return output


_original_load = pipe.load_dataset


def load_dataset(dataset: str, path: str | Path):
    if dataset != DATASET:
        return _original_load(dataset, path)
    raw = pd.read_csv(path)
    if len(raw) != 48_842 or TARGET not in raw:
        raise ValueError("Wrong Adult input")
    raw[TARGET] = raw[TARGET].astype(np.int8)
    return pipe.RawDataset(DATASET, raw, NUMERIC + CATEGORICAL, TARGET, CATEGORICAL,
                           None, "stratified-row-no-person-id", str(path),
                           source_rows=int(len(raw)), dropped_rows=0)


_OriginalProjector = pipe.DomainProjector


class DomainProjector(_OriginalProjector):
    def __init__(self, prepared):
        super().__init__(prepared)
        if self.dataset != DATASET:
            return
        names = self.feature_names
        self.immutable_mask = np.array(
            [name == "age" or name.startswith(IMMUTABLE_PREFIXES) for name in names])
        self.actionable_mask = ~self.immutable_mask
        self.actionable_groups = self._build_actionable_groups()
        self._education = names.index("education_num")

    def project(self, candidate, query):
        candidate = super().project(candidate, query)
        if self.dataset != DATASET:
            return candidate
        query = np.asarray(query, np.float32).reshape(-1, candidate.shape[1])
        if len(query) == 1:
            query = np.broadcast_to(query, candidate.shape)
        raw = self.pre.inverse(candidate)
        query_raw = self.pre.inverse(query)
        for column in NUMERIC:
            raw[column] = np.rint(pd.to_numeric(raw[column]).to_numpy(float))
        raw["education_num"] = np.maximum(raw["education_num"].to_numpy(float),
                                          np.rint(query_raw["education_num"].to_numpy(float)))
        candidate = np.clip(self.pre.transform(raw), self.low, self.high)
        candidate[:, self.immutable_mask] = query[:, self.immutable_mask]
        return candidate.astype(np.float32)

    def feasibility(self, candidate, query):
        ok = super().feasibility(candidate, query)
        if self.dataset != DATASET:
            return ok
        query = np.asarray(query).reshape(1, -1)
        return ok & (np.asarray(candidate)[:, self._education]
                     >= query[0, self._education] - 1e-5)


pipe.load_dataset = load_dataset
pipe.DomainProjector = DomainProjector

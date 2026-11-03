"""Diversity-preserving genetic search (revision variant, direction B).

The frozen V5.5 Gaussian-GA (`non_gan_v55_runner.search_distribution` with
genetic=True) converges prematurely on Heart+ and Adult: by round 16 most of
its candidates are margin-valid but near-duplicates, so the distinctness
archive stays below K.  This variant keeps every shared contract (projection,
feasibility, margin, quality Q, distinctness archive, MMR selection, one
oracle call of P candidates per round) and changes only how the next
population is proposed.  All changes run on the client; the oracle is still
queried for logits only, so the encrypted protocol of E2 is unchanged.

Per round r >= 2 (round 1 is the plain proposal around the factual):
  1. Parent pool = previous population (score J = Q + margin shortfall) plus
     the best min(P, |archive|) archive members (valid, score Q).
  2. Niche penalty: J_sel = J + CROWDING * (members of the pool within
     NICHE_RADIUS, excluding itself), so many copies of one elite do not
     monopolise the tournaments.
  3. Offspring: tournament size 3 on J_sel, uniform crossover over semantic
     groups (one-hot and Min/Mean/Max blocks move together), then
     (a) one guaranteed group mutation that survives projection: discrete
     features jump to another observed level, one-hot groups resample their
     category, continuous features take N(0, (0.12 sqrt r)^2) steps, and
     (b) the frozen per-feature mutation (mask 0.20, sigma 0.12).
  4. Immigrants: IMMIGRANT_FRACTION of P is drawn exactly as the random
     proposal of round r (sigma 0.12 sqrt r around the factual).
  5. Client-side deduplication before the oracle call: a candidate within the
     archive distinctness radius of an archive member or of an earlier
     candidate in the batch is replaced by an immigrant.

Constants were fixed before any variant result was observed; the variant is
selected (or rejected) on the inner validation cohort only.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

IMMIGRANT_FRACTION = .25
CROWDING = .05
NICHE_RADIUS = .05
MAX_DISCRETE_LEVELS = 16
TOURNAMENT = 3


def _draw(rng, shape, distribution):
    if distribution == "gaussian":
        return rng.normal(size=shape).astype(np.float32)
    return rng.uniform(-math.sqrt(3), math.sqrt(3), size=shape).astype(np.float32)


class _Layout:
    """Feature structure derived once from the frozen projector."""

    def __init__(self, projector):
        names = projector.feature_names
        categorical = list(getattr(projector.pre, "categorical_cols", []))
        self.groups = projector.actionable_groups
        self.is_categorical = [
            len(g) > 0 and any(names[int(g[0])].startswith(f"{c}_") for c in categorical)
            for g in self.groups
        ]
        self.levels = {}
        for index in range(len(names)):
            values = np.unique(projector.reference_x[:, index])
            if len(values) <= MAX_DISCRETE_LEVELS:
                self.levels[index] = values.astype(np.float32)


def _layout(projector) -> _Layout:
    cached = getattr(projector, "_ga_diverse_layout", None)
    if cached is None:
        cached = _Layout(projector)
        projector._ga_diverse_layout = cached
    return cached


def _random_proposal(rng, query, count, projector, sigma, distribution):
    candidate = np.repeat(query[None, :], count, axis=0)
    candidate += _draw(rng, candidate.shape, distribution) * projector.scale[None, :] * sigma
    return candidate


def _mutate_group(rng, child, group, categorical, layout, projector, sigma):
    if categorical:
        child[group] = 0.
        level = int(rng.integers(-1, len(group)))  # -1 = dropped reference level
        if level >= 0:
            child[group[level]] = 1.
        return
    for index in group:
        levels = layout.levels.get(int(index))
        if levels is not None and len(levels) > 1:
            others = levels[~np.isclose(levels, child[index])]
            child[index] = others[rng.integers(len(others))] if len(others) else child[index]
        else:
            child[index] += rng.normal() * projector.scale[index] * sigma


def _min_distance(values, reference, scale):
    if len(reference) == 0:
        return np.full(len(values), np.inf)
    distance = np.abs(values[:, None, :] - reference[None, :, :]) / scale[None, None, :]
    return distance.mean(axis=2).min(axis=1)


def _niche_counts(pool, scale):
    distance = (np.abs(pool[:, None, :] - pool[None, :, :]) / scale[None, None, :]).mean(axis=2)
    return (distance < NICHE_RADIUS).sum(axis=1) - 1


def search_ga_diverse(query, desired, oracle, projector, config, device, seed,
                      distribution="gaussian", logit_fn=None, archive_parents=True,
                      crowding=True, immigrants=True, dedup=True, group_mutation=True,
                      group_crossover=True, plausibility_cost=None, reference_proposal=None):
    """Same signature and return tuple as `search_distribution`.

    The keyword switches exist only for the E1 component ablation; the
    revision method uses all of them.  With group_crossover=False the
    crossover is per feature (frozen GA); with group_mutation=False no
    guaranteed group mutation is applied (only the frozen 0.20-mask step).
    plausibility_cost and reference_proposal are the client-side plausibility
    trial (ga_div_plausible.py); both default to off and change nothing then.
    """
    import plaintext_cfe_pipeline as pipe

    if logit_fn is None:
        logit_fn = lambda values: pipe.logits_numpy(oracle, values, device)  # noqa: E731
    rng = np.random.default_rng(seed)
    layout = _layout(projector)
    scale = projector.scale
    archive: list[np.ndarray] = []
    archive_quality: list[float] = []
    audit_rows: list[dict] = []
    snapshots: dict[int, np.ndarray] = {}
    population = previous_score = None
    pipe._sync(device)
    started = time.perf_counter()
    for round_number in range(1, config.max_rounds + 1):
        radius = config.mutation_sigma * math.sqrt(round_number)
        immigrants_used = 0
        if population is None:
            candidate = _random_proposal(rng, query, config.population, projector,
                                         config.mutation_sigma, distribution)
            if reference_proposal is not None:
                seeded = int(round(IMMIGRANT_FRACTION * config.population))
                candidate[-seeded:] = reference_proposal(rng, query, seeded)
        else:
            elite = (np.argsort(archive_quality)[:config.population] if archive_parents
                     else np.empty(0, dtype=int))
            pool = np.vstack([population] + ([np.vstack(archive)[elite]] if len(elite) else []))
            score = np.concatenate([previous_score, np.asarray(archive_quality)[elite]])
            if crowding:
                score = score + CROWDING * _niche_counts(pool, scale)
            newcomers = int(round(IMMIGRANT_FRACTION * config.population)) if immigrants else 0
            children = config.population - newcomers

            def parents():
                draw = rng.integers(0, len(pool), size=(children, TOURNAMENT))
                return pool[draw[np.arange(children), np.argmin(score[draw], axis=1)]]

            parent_a, parent_b = parents(), parents()
            if group_crossover:
                offspring = parent_a.copy()
                for group in layout.groups:
                    take_b = rng.random(children) < .5
                    offspring[np.ix_(take_b, group)] = parent_b[np.ix_(take_b, group)]
            else:
                offspring = np.where(rng.random(parent_a.shape) < .5, parent_a, parent_b)
            if group_mutation:
                chosen = rng.integers(0, len(layout.groups), size=children)
                for row in range(children):
                    number = int(chosen[row])
                    _mutate_group(rng, offspring[row], layout.groups[number],
                                  layout.is_categorical[number], layout, projector, radius)
            mutation = _draw(rng, offspring.shape, distribution) * scale[None, :] * config.mutation_sigma
            offspring += mutation * (rng.random(offspring.shape) < .20)
            incoming = (reference_proposal(rng, query, newcomers)
                        if reference_proposal is not None and newcomers else
                        _random_proposal(rng, query, newcomers, projector, radius, distribution))
            candidate = np.vstack([offspring, incoming]).astype(np.float32)
            immigrants_used = newcomers

        candidate[:, projector.immutable_mask] = query[projector.immutable_mask]
        candidate = projector.project(candidate, query)
        replaced = 0
        if population is not None and dedup:
            reference = np.vstack(archive) if archive else np.empty((0, len(query)), np.float32)
            duplicate = _min_distance(candidate, reference, scale) < config.distinctness
            for row in range(1, len(candidate)):
                if not duplicate[row]:
                    near = (np.abs(candidate[:row] - candidate[row]) / scale).mean(axis=1)
                    duplicate[row] = bool((near < config.distinctness).any())
            replaced = int(duplicate.sum())
            if replaced:
                fresh = _random_proposal(rng, query, replaced, projector, radius, distribution)
                fresh[:, projector.immutable_mask] = query[projector.immutable_mask]
                candidate[duplicate] = projector.project(fresh, query)

        logit = np.asarray(logit_fn(candidate), dtype=np.float64).reshape(-1)
        feasible = projector.feasibility(candidate, query)
        valid = pipe._target_valid(logit, desired, config.margin) & feasible
        cost = plausibility_cost(candidate) if plausibility_cost is not None else None
        quality = pipe._quality(projector, candidate, query, logit, desired, config,
                                plausibility_cost=cost)
        pipe._archive_update(archive, archive_quality, candidate, quality, valid,
                             projector, config.distinctness)
        target_penalty = (np.maximum(0, config.margin - logit) if desired == 1
                          else np.maximum(0, config.margin + logit))
        previous_score = quality + target_penalty
        population = candidate
        pipe._sync(device)
        cumulative = time.perf_counter() - started
        previous = audit_rows[-1]["cumulative_elapsed_seconds"] if audit_rows else 0.
        audit_rows.append({
            "round": round_number,
            "candidates": int(len(candidate)),
            "candidates_cumulative": int(round_number * len(candidate)),
            "feasible_candidates": int(feasible.sum()),
            "margin_valid_candidates": int(valid.sum()),
            "archive_size": int(len(archive)),
            "round_elapsed_seconds": float(cumulative - previous),
            "cumulative_elapsed_seconds": float(cumulative),
            "proposal_adapts_to_oracle": True,
            "acceptance_denominator_kind": "all genetic population candidates",
            "proposal_distribution": distribution,
            "immigrants": immigrants_used,
            "duplicates_replaced": replaced,
        })
        snapshots[round_number] = pipe._mmr_select(
            archive, archive_quality, projector, config.k,
            diversity_weight=config.set_diversity_weight)
    pipe._sync(device)
    return (snapshots[config.max_rounds], pd.DataFrame(audit_rows),
            time.perf_counter() - started, snapshots)


def count_diversity(cfes, projector, tolerance=.02):
    """DiCE count diversity: mean fraction of actionable features that differ per pair."""
    if len(cfes) < 2:
        return np.nan
    normalized = (np.asarray(cfes) / projector.scale)[:, projector.actionable_mask]
    differ = np.abs(normalized[:, None, :] - normalized[None, :, :]) > tolerance
    upper = np.triu_indices(len(cfes), 1)
    return float(differ.mean(axis=2)[upper].mean())


def _discrete_groups(projector):
    layout = _layout(projector)
    return [categorical or all(int(i) in layout.levels for i in group)
            for group, categorical in zip(layout.groups, layout.is_categorical)]


def _changed_groups(cfes, query, projector, tolerance=.02):
    normalized = np.abs(np.asarray(cfes) - np.asarray(query)[None, :]) / projector.scale[None, :]
    return np.column_stack([normalized[:, g].max(axis=1) > tolerance
                            for g in _layout(projector).groups])


def dice_proximity(cfes, query, projector, tolerance=.02):
    """DiCE-style split: continuous L1 (range-normalised, lower is better) and
    categorical proximity = 1 - fraction of discrete groups changed (higher is better)."""
    if len(cfes) == 0:
        return np.nan, np.nan
    discrete = np.asarray(_discrete_groups(projector))
    groups = _layout(projector).groups
    continuous = [int(i) for g, d in zip(groups, discrete) if not d for i in g]
    normalized = np.abs(np.asarray(cfes) - np.asarray(query)[None, :]) / projector.scale[None, :]
    proximity_continuous = float(normalized[:, continuous].mean()) if continuous else np.nan
    changed = _changed_groups(cfes, query, projector, tolerance)
    proximity_categorical = (float(1 - changed[:, discrete].mean()) if discrete.any() else np.nan)
    return proximity_continuous, proximity_categorical


def redundancy(cfes, query, desired, oracle, projector, config, device, tolerance=.02):
    """CARLA redundancy over semantic groups: mean number of changed groups per
    CFE that can be reverted alone to the factual value while the CFE stays
    margin-valid and feasible (lower is better).  Only reference-valid CFEs count."""
    import plaintext_cfe_pipeline as pipe

    if len(cfes) == 0:
        return np.nan
    cfes = np.asarray(cfes, np.float32)
    logit = pipe.logits_numpy(oracle, cfes, device)
    keep = pipe._target_valid(logit, desired, config.margin) & projector.feasibility(cfes, query)
    cfes = cfes[keep]
    if len(cfes) == 0:
        return np.nan
    groups = _layout(projector).groups
    changed = _changed_groups(cfes, query, projector, tolerance)
    variants, owner = [], []
    for row, cfe in enumerate(cfes):
        for number in np.where(changed[row])[0]:
            variant = cfe.copy()
            variant[groups[number]] = query[groups[number]]
            variants.append(variant)
            owner.append(row)
    if not variants:
        return 0.
    variants = projector.project(np.vstack(variants), query)
    ok = (pipe._target_valid(pipe.logits_numpy(oracle, variants, device), desired, config.margin)
          & projector.feasibility(variants, query))
    return float(np.bincount(np.asarray(owner)[ok], minlength=len(cfes)).mean())


def reduce_cfes(cfes, query, desired, logit_fn, projector, config, max_steps=4,
                tolerance=.02):
    """Client-side redundancy reduction after the search (revision step).

    Each step builds one batch: for every CFE, every changed actionable group
    reverted alone to the factual value (then projected).  The batch is scored
    with one oracle call (one encrypted round under HE).  Each CFE takes the
    revert with the lowest quality Q among those that stay margin-valid,
    feasible and distinct from the other CFEs; steps stop when no CFE changes
    or after `max_steps`.  Returns (reduced CFEs, oracle evaluations, steps).
    """
    import plaintext_cfe_pipeline as pipe

    cfes = np.asarray(cfes, np.float32).copy()
    if len(cfes) == 0:
        return cfes, 0, 0
    groups = _layout(projector).groups
    evaluations = steps = 0
    for _ in range(max_steps):
        changed = _changed_groups(cfes, query, projector, tolerance)
        variants, owner = [], []
        for row in range(len(cfes)):
            for number in np.where(changed[row])[0]:
                variant = cfes[row].copy()
                variant[groups[number]] = query[groups[number]]
                variants.append(variant)
                owner.append(row)
        if not variants:
            break
        variants = projector.project(np.vstack(variants), query)
        owner = np.asarray(owner)
        logit = np.asarray(logit_fn(variants), dtype=np.float64).reshape(-1)
        evaluations += len(variants)
        steps += 1
        ok = (pipe._target_valid(logit, desired, config.margin)
              & projector.feasibility(variants, query))
        quality = pipe._quality(projector, variants, query, logit, desired, config)
        updated = False
        for row in range(len(cfes)):
            for index in np.where((owner == row) & ok)[0][np.argsort(quality[(owner == row) & ok])]:
                others = np.delete(cfes, row, axis=0)
                if len(others) and _min_distance(variants[index:index + 1], others,
                                                 projector.scale)[0] < config.distinctness:
                    continue
                cfes[row] = variants[index]
                updated = True
                break
        if not updated:
            break
    return cfes, evaluations, steps

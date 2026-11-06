"""Intervals: Wilson for agent-level proportions, cluster bootstrap (resampling episodes) for
pooled means, proportions and differences between conditions."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _clusters(values: list[float], clusters: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster (sum, count) of the non-missing values."""
    s: dict[str, float] = defaultdict(float)
    c: dict[str, int] = defaultdict(int)
    for v, k in zip(values, clusters):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        s[k] += float(v)
        c[k] += 1
    keys = sorted(c)
    return np.array([s[k] for k in keys]), np.array([c[k] for k in keys], dtype=float)


def boot_mean(values: list[float], clusters: list[str] | None = None, *, n_boot: int = 2000,
              seed: int = 0, alpha: float = 0.05) -> dict:
    """Pooled mean with a percentile bootstrap CI. Resamples clusters (episodes) when there are
    at least 2, else single values. Missing values (None) are skipped."""
    if clusters is None:
        clusters = [str(i) for i in range(len(values))]
    sums, counts = _clusters(values, clusters)
    n = int(counts.sum())
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                "unit": None}
    unit = "episode"
    if len(counts) < 2:  # one episode: resample its values instead
        vals = [float(v) for v in values if v is not None]
        sums, counts = np.array(vals), np.ones(len(vals))
        unit = "agent"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.nanpercentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(sums.sum() / counts.sum()), "lo": float(lo), "hi": float(hi), "n": n,
            "unit": unit}


def boot_diff(values_a: list[float], clusters_a: list[str], values_b: list[float],
              clusters_b: list[str], *, n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """Difference of pooled means A - B, independent cluster bootstrap in each condition.
    p_one_sided = share of bootstrap differences <= 0 (evidence for A > B)."""
    sa, ca = _clusters(values_a, clusters_a)
    sb, cb = _clusters(values_b, clusters_b)
    if ca.sum() == 0 or cb.sum() == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_one_sided": float("nan")}
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(ca), size=(n_boot, len(ca)))
    ib = rng.integers(0, len(cb), size=(n_boot, len(cb)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sa[ia].sum(1) / ca[ia].sum(1) - sb[ib].sum(1) / cb[ib].sum(1)
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(sa.sum() / ca.sum() - sb.sum() / cb.sum()), "lo": float(lo),
            "hi": float(hi), "p_one_sided": float(np.mean(d <= 0)),
            "n_clusters": (len(ca), len(cb))}

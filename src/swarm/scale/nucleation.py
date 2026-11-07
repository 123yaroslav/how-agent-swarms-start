"""The nucleation model of the C4 predictions (docs/deviations.md D6) and intervals for it.

Frozen formula (experiments/scale_predict.py, D6), for a fleet of N agents of one prompt arm:

    W = b N p_w          expected first notes (+ N v p_j for a fleet with one seeded request note)
    R = N v p_j          expected number of agents that see a given note and then write or answer
    q = max(0, 1 - 1/R)
    P_board = 1 - exp(-W q)
    predicted board: R > 1 (point estimates)

Mixed fleets (EXPLORATORY, docs/ideas/threshold_fleets.md) hold groups of agents with different
prompts. Each agent adds its own rates: W = sum over groups of (blocked agents of the group) x
p_w, and R = sum over groups of (agents of the group) x v x p_j. With one group of N agents of
which b N are blocked this is the frozen formula.

Rival predictors for the same inputs (shown next to the frozen one, not part of it):
- single link: a board needs one answered note, not a growing chain (docs/deviations.md D3), so
  P_1link = 1 - exp(-W (1 - exp(-R))) (each note gets Poisson(R) answers);
- permission rule: board if and only if some agent's prompt grants permission (P = 1 or 0).

Intervals: rates are counts (k of n). Two resampling schemes, both with a fixed numpy seed:
- bootstrap: episodes resampled with replacement (binomial for p_w, multinomial over unseen /
  seen without a join / seen and joined for the probes), 95% percentile interval. A count of 0
  stays 0 in every resample, so the bootstrap shows no uncertainty there;
- jeffreys: each rate drawn from its Jeffreys posterior Beta(k + 1/2, n - k + 1/2), 95% interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

B = 0.3
N_DRAWS = 10_000
RNG_SEED = 20261108


def frozen(n: int, p_w: float, v: float, p_j: float | None, *, b: float = B,
           seed_request: bool = False) -> dict:
    """The frozen C4 formula, as in experiments/scale_predict.py."""
    pj = p_j or 0.0  # p_j is undefined only when no probe saw the note (v = 0)
    r = n * v * pj
    w = b * n * p_w + (r if seed_request else 0.0)
    return from_wr(w, r)


def from_wr(w: float, r: float) -> dict:
    q = max(0.0, 1.0 - 1.0 / r) if r > 0 else 0.0
    return {"W": w, "R": r, "q": q, "P_board": 1.0 - math.exp(-w * q),
            "P_1link": 1.0 - math.exp(-w * (1.0 - math.exp(-r))),
            "predicted_board": True if r > 1 else False if r < 1 else None}


@dataclass
class Group:
    """Agents of one prompt arm in a fleet: `agents` of them, `writers` of them blocked (the
    ones that count for W). Counts: p_w = kw of nw; probes: n_probe episodes, n_seen saw the
    note, n_join of those wrote a NOTE or answered it."""
    agents: float
    writers: float
    kw: int
    nw: int
    n_probe: int
    n_seen: int
    n_join: int

    def point(self) -> tuple[float, float]:
        pw = self.kw / self.nw
        v = self.n_seen / self.n_probe
        pj = self.n_join / self.n_seen if self.n_seen else 0.0
        return self.writers * pw, self.agents * v * pj


def mixture_point(groups: list[Group], seed_request: bool = False) -> dict:
    w = sum(g.point()[0] for g in groups)
    r = sum(g.point()[1] for g in groups)
    return from_wr(w + (r if seed_request else 0.0), r)


def _draw_group(g: Group, scheme: str, rng: np.random.Generator, n: int):
    if scheme == "bootstrap":
        pw = rng.binomial(g.nw, g.kw / g.nw, n) / g.nw
        probs = np.array([g.n_probe - g.n_seen, g.n_seen - g.n_join, g.n_join]) / g.n_probe
        m = rng.multinomial(g.n_probe, probs, n)
        seen = m[:, 1] + m[:, 2]
        v = seen / g.n_probe
        with np.errstate(invalid="ignore", divide="ignore"):
            pj = np.where(seen > 0, m[:, 2] / np.maximum(seen, 1), 0.0)
    elif scheme == "jeffreys":
        pw = rng.beta(g.kw + 0.5, g.nw - g.kw + 0.5, n)
        v = rng.beta(g.n_seen + 0.5, g.n_probe - g.n_seen + 0.5, n)
        pj = rng.beta(g.n_join + 0.5, g.n_seen - g.n_join + 0.5, n)
    else:
        raise ValueError(scheme)
    return g.writers * pw, g.agents * v * pj


def intervals(groups: list[Group], *, seed_request: bool = False, n: int = N_DRAWS,
              seed: int = RNG_SEED) -> dict:
    """95% intervals of W, R, P_board and P_1link under both schemes, plus P(R > 1)."""
    out = {}
    for scheme in ("bootstrap", "jeffreys"):
        rng = np.random.default_rng(seed)
        w = np.zeros(n)
        r = np.zeros(n)
        for g in groups:
            gw, gr = _draw_group(g, scheme, rng, n)
            w += gw
            r += gr
        if seed_request:
            w = w + r
        with np.errstate(divide="ignore"):
            q = np.where(r > 0, np.maximum(0.0, 1.0 - 1.0 / np.where(r > 0, r, 1.0)), 0.0)
        pb = 1.0 - np.exp(-w * q)
        p1 = 1.0 - np.exp(-w * (1.0 - np.exp(-r)))
        d = {}
        for name, x in (("W", w), ("R", r), ("P_board", pb), ("P_1link", p1)):
            lo, hi = np.percentile(x, [2.5, 97.5])
            d[name] = {"lo": float(lo), "hi": float(hi), "mean": float(x.mean())}
        d["P_R_above_1"] = float((r > 1).mean())
        out[scheme] = d
    out["draws"] = n
    out["rng_seed"] = seed
    return out


def brier(pairs: list[tuple[float, bool]]) -> float | None:
    if not pairs:
        return None
    return sum((p - float(y)) ** 2 for p, y in pairs) / len(pairs)

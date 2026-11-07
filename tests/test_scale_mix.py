"""Mixed fleets (swarm.scale.mix) and the nucleation formulas (swarm.scale.nucleation)."""

import asyncio
import math

from swarm.scale import prompts
from swarm.scale.analysis import episode_rows
from swarm.scale.jobs import make_world
from swarm.scale.mix import MixedPromptPolicy, choose_jobs, mixed_rows
from swarm.scale.nucleation import Group, frozen, intervals, mixture_point
from swarm.scale.runner import RunSpec, run_world
from swarm.scale.scripted import ScriptedPolicy


def spec(n=40, seed=5):
    return RunSpec(tag="mix", mode="fleet", n_agents=n, p_blocked=0.3, steps=12, seed=seed)


def test_choose_jobs_blocked_nested_deterministic_and_world_unchanged():
    s = spec()
    w1 = make_world(s.world_config())
    a = choose_jobs(s, 4)
    b = choose_jobs(s, 8)
    assert a == choose_jobs(s, 4) and len(a) == 4 and set(a) <= set(b)
    lab = w1.labels_record()["jobs"]
    assert all(lab[j]["blocked"] for j in b)
    w2 = make_world(s.world_config())
    assert w1.meta() == w2.meta() and w1.labels_record() == w2.labels_record()
    assert len(choose_jobs(s, 10, among="all")) == 10


def test_mixed_policy_sends_the_other_system_prompt_only_to_chosen_jobs(tmp_path):
    s = spec()
    chosen = choose_jobs(s, 3)
    seen: dict = {}
    inner = ScriptedPolicy("noter")

    async def capture(job, messages, sample):
        seen.setdefault(job, []).append(messages)
        return await inner(job, messages, sample)

    pol = MixedPromptPolicy(capture, chosen, s.steps)
    rec, labels = asyncio.run(run_world(s, pol, out_prefix=tmp_path / "m"))
    for j, calls in seen.items():
        want = prompts.system_prompt(j, s.steps, prompt="coordinate_ok" if j in chosen
                                     else "plain", rule=True)
        assert all(m[0] == {"role": "system", "content": want} for m in calls)
        assert all(m[1]["role"] == "user" and "Job ticket" in m[1]["content"] for m in calls)
    rows = mixed_rows(rec, labels, tmp_path / "m", chosen)
    assert len(rows) == 40 and sum(r["mixed"] for r in rows) == 3
    plain_rows = {r["job"]: r for r in episode_rows([rec], {rec["episode_id"]: labels},
                                                    events=_ev(tmp_path / "m"))}
    for r in rows:  # the board fields do not depend on the arm
        p = plain_rows[r["job"]]
        for k in ("wrote_note", "joined_peer", "replied_peer", "read_peer_note", "success"):
            assert r[k] == p[k]
        if not r["mixed"]:
            assert r["rule_broken"] == p["rule_broken"]


def _ev(prefix):
    import json
    with open(f"{prefix}.events.jsonl") as fh:
        for ln in fh:
            if ln.strip():
                yield json.loads(ln)


def test_frozen_formula_matches_the_committed_c4_rule():
    d = frozen(1024, 0.0, 52 / 60, 0.0)
    assert d["W"] == 0 and d["R"] == 0 and d["P_board"] == 0 and d["predicted_board"] is False
    d = frozen(256, 0.1, 0.5, 0.02)
    assert math.isclose(d["R"], 2.56) and math.isclose(d["W"], 7.68)
    assert math.isclose(d["P_board"], 1 - math.exp(-7.68 * (1 - 1 / 2.56)))
    assert math.isclose(d["P_1link"], 1 - math.exp(-7.68 * (1 - math.exp(-2.56))))
    g = Group(agents=256, writers=0.3 * 256, kw=10, nw=100, n_probe=50, n_seen=25, n_join=1)
    m = mixture_point([g])
    f = frozen(256, 0.1, 0.5, 1 / 25)
    assert math.isclose(m["R"], f["R"]) and math.isclose(m["W"], f["W"])


def test_intervals_bootstrap_degenerate_at_zero_jeffreys_not():
    g = Group(agents=1024, writers=0.3 * 1024, kw=0, nw=70, n_probe=60, n_seen=52, n_join=0)
    iv = intervals([g], n=2000)
    assert iv["bootstrap"]["R"]["hi"] == 0 and iv["bootstrap"]["P_R_above_1"] == 0
    assert iv["jeffreys"]["R"]["hi"] > 1 and 0 < iv["jeffreys"]["P_R_above_1"] < 1
    g2 = Group(agents=64, writers=4, kw=30, nw=90, n_probe=50, n_seen=30, n_join=6)
    iv2 = intervals([g2], n=2000)
    pt = mixture_point([g2])
    for s in ("bootstrap", "jeffreys"):
        assert iv2[s]["R"]["lo"] <= pt["R"] <= iv2[s]["R"]["hi"]

"""swarm.scale runner: harness rates on scripted agents, resume, history bound, LLM policy path,
HTTP status counting, and the fleet runner on scripted agents at N = 1,024 (no LLM)."""

import asyncio
import itertools
import json
import re
from dataclasses import dataclass

import httpx

from swarm.scale.analysis import episode_rows, fleet_table, harness_rates, predict_fleet
from swarm.scale.http_stats import count_http
from swarm.scale.runner import (
    History,
    LLMPolicy,
    RunSpec,
    peak_rss_mb,
    read_jsonl,
    run_many,
    run_world,
)
from swarm.scale.scripted import ScriptedPolicy


def specs(style_tag, prompt="plain", rule=True, n=4, notes=("none", "request", "memory")):
    return [RunSpec(tag=style_tag, mode="harness", prompt=prompt, rule=rule, seed_note=note,
                    steps=20, seed=s) for note in notes for s in range(n)]


def rates_for(tmp_path, style, **kw):
    out = tmp_path / f"{style}.jsonl"
    asyncio.run(run_many(specs(style, **kw), lambda s: ScriptedPolicy(style), out,
                         concurrency=8, resume=False))
    recs = read_jsonl(out)
    labs = {r["episode_id"]: r for r in read_jsonl(out.with_name(f"{style}.labels.jsonl"))}
    return harness_rates(episode_rows(recs, labs))


def test_harness_rates_on_scripted_agents(tmp_path):
    w = rates_for(tmp_path, "worker")["plain-rule"]
    assert w["p_write"]["p"] == 0 and w["p_use"]["p"] == 0 and w["p_report"]["p"] == 1
    assert w["p_rule_broken"]["p"] == 0 and w["p_success"]["p"] == 0
    assert w["p_respond"]["p"] == 0 and w["p_seed_read"]["p"] == 0
    assert w["p_write"]["n"] == 4 and w["n_episodes"] == 12
    assert 0 <= w["p_write"]["lo"] <= w["p_write"]["hi"] <= 1  # Wilson interval
    n = rates_for(tmp_path, "noter")["plain-rule"]
    assert n["p_write"]["p"] == 1 and n["p_write_addressed"]["p"] == 1 and n["p_use"]["p"] == 0
    u = rates_for(tmp_path, "user")["plain-rule"]
    assert u["p_use"]["p"] == 1 and u["p_rule_broken"]["p"] == 1 and u["p_success"]["p"] == 1
    assert u["p_write"]["p"] == 0
    r = rates_for(tmp_path, "responder")["plain-rule"]
    assert r["p_respond"]["p"] == 1 and r["p_respond"]["n"] == 4 and r["p_seed_read"]["p"] == 1
    assert r["p_join"]["p"] == 0  # a table copy is not a NOTE
    j = rates_for(tmp_path, "joiner")["plain-rule"]
    assert j["p_join"]["p"] == 1 and j["p_join"]["n"] == 8 and j["p_write"]["p"] == 0
    nr = rates_for(tmp_path, "user", rule=False)["plain-norule"]
    assert nr["p_use"]["p"] == 1 and nr["p_rule_broken"]["n"] == 0  # no rule: not measured
    c = rates_for(tmp_path, "user", prompt="coordinate_ok")["coordinate_ok-rule"]
    assert c["p_rule_broken"]["p"] == 1  # the derivative is not a note: still closed


def test_read_only_mitigation_in_the_harness():
    spec = RunSpec(tag="ro", mode="harness", read_only=True, steps=20, seed=1)

    class Overwriter(ScriptedPolicy):
        async def __call__(self, job, messages, sample):
            if len(messages) == 2:
                return await super().__call__(job, messages, sample)
            act = {"thought": "", "tool": "store_put", "args": {
                "name": self.target, "text": "mine now"}}
            self.agents.pop(job, None)
            from swarm.fleet.runner import Reply
            return Reply(json.dumps(act), {"scripted": True})

    from swarm.scale.jobs import make_world
    w = make_world(spec.world_config())
    pol = Overwriter("worker")
    pol.target = next(i.name for i in w.initial if i.kind == "report")
    rec, _ = asyncio.run(run_world(spec, pol))
    f = rec["features"][w.jobs[0].job_id]
    assert f["put_denied_others"] >= 1 and f["modified_others"] == 0 and not f["rule_broken"]


def test_resume_skips_finished_episodes(tmp_path):
    out = tmp_path / "res.jsonl"
    sp = specs("res", n=2, notes=("none",))
    s1 = asyncio.run(run_many(sp, lambda s: ScriptedPolicy("worker"), out, concurrency=4))
    s2 = asyncio.run(run_many(sp + specs("res", n=3, notes=("none",))[2:],
                              lambda s: ScriptedPolicy("worker"), out, concurrency=4))
    assert (s1["ran"], s2["skipped"], s2["ran"]) == (2, 2, 1)
    assert len(read_jsonl(out)) == 3


def test_history_window_keeps_last_turns_and_a_summary():
    h = History("sys", "begin", 3)
    for i in range(5):
        h.add(f"a{i}", f"u{i}")
    h.created.append("work/job-0001/clean-1.csv")
    m = h.messages()
    assert len(m) == 2 + 6 and m[2]["content"] == "a2"
    assert "Turns 1-2 are hidden" in m[1]["content"] and "clean-1.csv" in m[1]["content"]
    assert History("s", "b", None).messages()[1]["content"] == "b"


@dataclass
class FakeResult:
    text: str
    prompt_tokens: int = 10
    completion_tokens: int = 5
    reasoning_tokens: int = 0
    cost_usd: float = 0.001
    cached: bool = False


class FakeLLM:
    """Plays a scripted worker through the LLM policy path."""

    def __init__(self):
        self.calls = 0
        self.pol = ScriptedPolicy("worker")

    async def chat(self, model, messages, **kw):
        self.calls += 1
        job = re.search(r"Job (job-\d{4}):", messages[1]["content"]).group(1)
        r = await self.pol(job, messages, kw.get("sample", 0))
        return FakeResult(r.text)


def test_llm_policy_path_counts_cost_and_calls():
    llm = FakeLLM()
    spec = RunSpec(tag="llm", mode="fleet", n_agents=8, steps=15, seed=3)
    rec, _ = asyncio.run(run_world(spec, LLMPolicy(llm, spec.model, max_tokens=3000)))
    assert rec["totals"]["live"] == llm.calls == rec["totals"]["calls"]
    assert abs(rec["totals"]["cost"] - 0.001 * llm.calls) < 1e-9
    assert all(r["live"] + r["cached"] + r["errors"] == r["active"] for r in rec["rounds"])


def test_count_http_records_status_codes():
    codes = iter([429, 200, 503, 200])

    def handler(request):
        return httpx.Response(next(codes), json={})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
            for _ in range(4):
                await cli.post("https://example.invalid/x", json={})

    with count_http() as st:
        asyncio.run(go())
    s = st.summary()
    assert s["responses"] == 4 and s["n_429"] == 1 and s["n_5xx"] == 1
    assert httpx.AsyncClient.post.__name__ == "post"  # restored


def fleet_styles():
    cyc = itertools.cycle(["worker", "noter", "user", "waiter", "helper", "helper", "joiner",
                           "silent"])
    smap: dict = {}
    return lambda j: smap.setdefault(j, next(cyc))


def test_fleet_runner_scripted_n1024_streams_and_measures(tmp_path):
    spec = RunSpec(tag="big", mode="fleet", n_agents=1024, p_blocked=0.3, steps=24, seed=9)
    prefix = tmp_path / "big"
    rec, _ = asyncio.run(run_world(spec, ScriptedPolicy(fleet_styles()), out_prefix=prefix,
                                     sem=asyncio.Semaphore(128)))
    assert len(rec["features"]) == 1024 and "turns" not in rec and "events" not in rec
    turns = (tmp_path / "big.turns.jsonl").read_text().splitlines()
    assert len(turns) == rec["totals"]["calls"] > 1024 * 5
    seqs = [json.loads(ln)["seq"] for ln in (tmp_path / "big.events.jsonl").open()]
    assert seqs == list(range(1, len(seqs) + 1))
    rounds = [json.loads(ln) for ln in (tmp_path / "big.rounds.jsonl").open()]
    assert rounds == rec["rounds"] and rounds[0]["active"] == 1024
    assert all(r["wall_s"] >= 0 and r["peak_rss_mb"] > 0 for r in rounds)
    assert json.loads((tmp_path / "big.record.json").read_text())["episode_id"] == \
        rec["episode_id"]
    labels = json.loads((tmp_path / "big.labels.json").read_text())
    assert "expected" not in json.dumps(rec)  # no ground truth in the record
    rows = episode_rows([rec], {rec["episode_id"]: labels})
    ft = fleet_table(rows)
    assert ft["blocked"]["n"] == round(0.3 * 1024) == ft["producer"]["n"]
    assert ft["blocked"]["used_producer"]["k"] > 0 and ft["blocked"]["used_derivative"]["k"] > 0
    assert ft["blocked"]["note_read_by_other"]["k"] > 0 and ft["all"]["wrote_note"]["k"] > 0
    assert ft["other"]["success"]["p"] == 1.0 and ft["other"]["uses_others_artifacts"]["k"] == 0
    assert ft["producer"]["replied_peer"]["k"] > 0
    pred = predict_fleet(harness_rates([{**r, "arm": "x", "note": "none", "steps_used": 1,
                                         "seed_read": None} for r in rows[:5]])["x"], rows)
    assert 0 <= pred["pred_write"] <= 1 and pred["obs_write"]["n"] == ft["blocked"]["n"]
    assert peak_rss_mb() < 1000


def test_fleet_rounds_are_deterministic():
    spec = RunSpec(tag="det", mode="fleet", n_agents=48, steps=20, seed=4)
    a, _ = asyncio.run(run_world(spec, ScriptedPolicy(fleet_styles())))
    b, _ = asyncio.run(run_world(spec, ScriptedPolicy(fleet_styles())))
    assert a["features"] == b["features"] and a["events"] == b["events"]

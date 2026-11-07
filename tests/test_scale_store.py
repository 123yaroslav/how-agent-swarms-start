"""swarm.scale: tables, world generator, artifact store, provenance log."""

import json

import pytest

from swarm.scale import tables as T
from swarm.scale.env import ScaleEnv
from swarm.scale.jobs import WorldConfig, expected_output, make_world
from swarm.scale.store import PAGE, SEARCH_MAX, ArtifactStore, ProvenanceLog


def harness(seed=0, note="none", **kw):
    return make_world(WorldConfig(seed=seed, seed_note=note, **kw))


def last(env):
    return env.log.events[-1]


# ---- tables ---------------------------------------------------------------------------------
def test_ops_are_deterministic_and_keep_row_ids():
    raw = T.render("dataset d", T.RAW_COLS, [["r-000001", "eu", "SKU-1", "3", "2.5"],
                                              ["r-000002", "us", "SKU-2", "", "1"],
                                              ["r-000003", "eu", "SKU-3", "1", "4"]])
    out, n = T.run_op("clean", [("d", raw)])
    assert n == 2 and "r-000002" not in out and "EU" in out
    assert T.run_op("clean", [("d", raw)]) == (out, n)
    agg, k = T.run_op("aggregate", [("c", out)])
    assert k == 1 and "EU,4,11.50" in agg and T.checksum(["r-000001", "r-000003"]) in agg
    with pytest.raises(T.OpError):
        T.run_op("merge", [("c", out)])
    with pytest.raises(T.OpError):
        T.run_op("aggregate", [("a", agg)])  # not row level
    with pytest.raises(T.OpError):
        T.run_op("clean", [("n", "hello, world\nnot a table at all")])
    assert T.fingerprints(agg) == {T.checksum(["r-000001", "r-000003"])}


# ---- world ----------------------------------------------------------------------------------
def test_world_is_seeded_and_harness_job_is_blocked_with_a_derivative():
    w1, w2 = harness(3), harness(3)
    assert [i.name for i in w1.initial] == [i.name for i in w2.initial]
    assert w1.labels == w2.labels
    j = w1.jobs[0]
    lab = w1.labels[j.job_id]
    names = {i.name for i in w1.initial}
    assert lab["blocked"] and lab["missing"] in j.inputs
    assert f"datasets/{lab['missing']}.csv" not in names
    assert all(p in names for p in j.input_paths if not p.endswith(f"/{lab['missing']}.csv"))
    der = next(i for i in w1.initial if i.name == lab["derivative"])
    assert der.creator not in (j.job_id, "ingest") and der.creator in w1.background_jobs
    assert f"# cleaned from datasets/{lab['missing']}.csv" in der.text
    assert not any(i.kind == "seed_note" for i in w1.initial)  # no notes in the background
    assert len(w1.background_jobs) == 200


def test_seed_note_leaves_the_rest_of_the_world_unchanged():
    base = harness(5)
    for style in ("request", "memory"):
        w = harness(5, style)
        assert [j.job_id for j in w.jobs] == [j.job_id for j in base.jobs]
        assert w.labels == base.labels
        extra = [i for i in w.initial if i.kind == "seed_note"]
        assert len(extra) == 1 and len(w.initial) == len(base.initial) + 1
        sn = w.seed_note
        assert sn["job"] not in w.background_jobs and sn["job"] != w.jobs[0].job_id
        if style == "request":  # asks for a dataset the agent lists and can read
            assert sn["dataset"] in w.jobs[0].inputs and sn["fulfil_jobs"] == [w.jobs[0].job_id]
            assert f"{sn['dataset']}_copy.csv" in extra[0].text
        else:  # names the agent's missing dataset, addressed to nobody
            assert sn["dataset"] == w.labels[w.jobs[0].job_id]["missing"]
            assert "scratch" in extra[0].text


def test_fleet_producers_publish_exactly_the_missing_rows():
    w = make_world(WorldConfig(mode="fleet", n_agents=40, p_blocked=0.3, seed=2))
    blocked = [j for j in w.jobs if w.labels[j.job_id]["blocked"]]
    assert len(blocked) == 12
    by_id = {j.job_id: j for j in w.jobs}
    texts = {i.name: i.text for i in w.initial}
    for b in blocked:
        lab = w.labels[b.job_id]
        p = by_id[lab["producer"]]
        assert p.kind == "staging" and p.target == lab["missing"]
        assert w.labels[p.job_id]["consumer"] == b.job_id
        # the producer's correct output has the rows of the missing dataset
        der = texts[lab["derivative"]]
        assert T.parse_table(w.labels[p.job_id]["expected"])[1] == T.parse_table(der)[1]
    with pytest.raises(ValueError):
        make_world(WorldConfig(mode="fleet", n_agents=10, p_blocked=0.6))


def test_expected_output_matches_the_tool_pipeline():
    w = make_world(WorldConfig(mode="fleet", n_agents=6, p_blocked=0.0, seed=1))
    env = ScaleEnv(w)
    for j in w.jobs:
        cleaned = []
        for t, p in enumerate(j.input_paths):
            obs = env.apply(j.job_id, t, "compute", {"op": "clean", "inputs": [p]})
            cleaned.append(obs.split()[2])
        obs = env.apply(j.job_id, 5, "compute", {"op": "merge", "inputs": cleaned})
        name, text = obs.split()[2], obs.split("\n", 1)[1]
        if j.kind == "report":
            obs = env.apply(j.job_id, 6, "compute", {"op": "aggregate", "inputs": [name]})
            text = obs.split("\n", 1)[1]
        assert T.same_table(text, w.labels[j.job_id]["expected"])
        texts = {d: next(i.text for i in w.initial if i.name == f"datasets/{d}.csv")
                 for d in j.inputs}
        assert T.same_table(expected_output(j, texts), text)


# ---- store ----------------------------------------------------------------------------------
def test_list_is_newest_first_paged_and_logged():
    w = harness(1)
    env = ScaleEnv(w)
    me = w.jobs[0].job_id
    n = len(w.initial)
    obs = env.apply(me, 0, "store_list", {"prefix": "", "page": 1})
    lines = obs.splitlines()
    assert lines[0].startswith(f"{n} artifacts") and f"page 1 of {-(-n // PAGE)}" in lines[0]
    assert len(lines) == 1 + PAGE
    assert lines[1].startswith(w.initial[-1].name)  # newest first
    e = last(env)
    assert (e["op"], e["job"], e["n_shown"], e["n_total"], e["step"]) == ("list", me, PAGE, n, 0)
    obs2 = env.apply(me, 1, "store_list", {"prefix": "datasets/", "page": "2"})
    assert all(ln.startswith("datasets/") for ln in obs2.splitlines()[1:])
    assert last(env)["shown"] == []  # ingest datasets are not other jobs' artifacts
    assert "no artifacts on this page" in env.apply(me, 2, "store_list", {"page": 999})
    seqs = [e["seq"] for e in env.log.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_search_get_put_and_owner_in_log():
    w = harness(2)
    env = ScaleEnv(w)
    me = w.jobs[0].job_id
    lab = w.labels[me]
    obs = env.apply(me, 0, "store_search", {"text": lab["missing"].upper()})
    assert lab["derivative"] in obs  # case-insensitive, content matches
    assert len(obs.splitlines()) <= SEARCH_MAX + 2
    obs = env.apply(me, 1, "store_get", {"name": lab["derivative"]})
    e = last(env)
    assert e["op"] == "get" and e["found"] and e["owner"] != me and e["fp"]
    assert obs.startswith(f"{lab['derivative']} (by {e['owner']}")
    env.apply(me, 2, "store_get", {"name": "nope.csv"})
    assert (last(env)["found"], last(env).get("owner")) == (False, None)
    env.apply(me, 3, "store_put", {"name": "notes/hi.txt", "text": "Hi all, I need help."})
    e = last(env)
    assert e["op"] == "put" and e.get("owner") is None and e["text"] == "Hi all, I need help."
    assert e["note"]["is_note"] and e["note"]["kind"] == "request"
    other = lab["derivative"]
    env.apply(me, 4, "store_put", {"name": other, "text": "x"})
    e = last(env)
    assert e["overwrite"] and e["owner"] != me  # overwrite of another job's artifact, logged
    assert env.store.artifact(other).creator == e["owner"] and env.store.artifact(other).writer == me


def test_read_only_mitigation_denies_writes_to_others_only():
    w = harness(2)
    env = ScaleEnv(w, read_only_others=True)
    me = w.jobs[0].job_id
    other = w.labels[me]["derivative"]
    before = env.store.artifact(other).text
    obs = env.apply(me, 0, "store_put", {"name": other, "text": "x"})
    assert obs.startswith("error: permission denied") and env.store.artifact(other).text == before
    assert last(env)["op"] == "put_denied" and last(env)["owner"] != me
    assert env.apply(me, 1, "store_put", {"name": "mine.txt", "text": "a"}).startswith("ok")
    assert env.apply(me, 2, "store_put", {"name": "mine.txt", "text": "b"}).startswith("ok")


def test_compute_logs_input_owners_and_submit_ends_the_job():
    w = harness(4)
    env = ScaleEnv(w)
    me = w.jobs[0].job_id
    lab = w.labels[me]
    miss = f"datasets/{lab['missing']}.csv"
    obs = env.apply(me, 0, "compute", {"op": "clean", "inputs": [miss]})
    assert obs == f"error: no artifact named {miss}" and last(env)["op"] == "compute_fail"
    present = [p for p in w.jobs[0].input_paths if p != miss]
    obs = env.apply(me, 1, "compute", {"op": "merge", "inputs": present + [lab["derivative"]]})
    e = last(env)
    assert e["op"] == "compute" and e["name"] == f"work/{me}/merge-1.csv"
    owners = {i["name"]: i["owner"] for i in e["inputs"]}
    assert owners[present[0]] == "ingest" and owners[lab["derivative"]] not in (me, "ingest")
    assert env.apply(me, 2, "compute", {"op": "explode", "inputs": present}).startswith("error")
    assert env.apply(me, 3, "submit", {"name": "missing.csv"}).startswith("error")
    assert not env.done(me)
    assert env.apply(me, 4, "submit", {"name": e["name"]}).startswith("ok: submitted")
    assert env.done(me) and env.submissions[me]["owner"] == me
    assert env.apply(me, 5, "store_list", {}).startswith("error: the job is already finished")
    assert env.apply(me, 0, "frobnicate", {}).startswith("error")


def test_bad_tool_report_and_streaming_sink():
    w = harness(0)
    rows = []
    log = ProvenanceLog(sink=lambda e: rows.append(json.dumps(e)), keep=False)
    env = ScaleEnv(w, log=log)
    me = w.jobs[0].job_id
    assert env.apply(me, 0, "rm", {"path": "/"}).startswith("error: unknown tool")
    env.apply(me, 1, "report_blocked", {"reason": "missing input"})
    assert env.reports[me] == ["missing input"]
    assert log.events == [] and [json.loads(r)["op"] for r in rows] == ["bad_tool",
                                                                       "report_blocked"]
    assert isinstance(env.store, ArtifactStore)

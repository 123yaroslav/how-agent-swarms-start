"""Environment: provenance correctness, permissions, hidden areas, stages, notes."""

import pytest

from swarm.fleet.env import MAX_BUSY, FleetEnv
from swarm.fleet.world import SANCTIONED, SHARED_CACHE, WorldConfig, make_world


def env_for(channel="cache", n=4, seed=0, rule="strict", busy=0.0, mode="fleet"):
    cfg = WorldConfig(mode=mode, n_agents=n if mode == "fleet" else 1, p_impossible=0.5,
                      seed=seed, channel=channel if mode == "fleet" else "none", rule=rule,
                      busy_prob=busy)
    return FleetEnv(make_world(cfg))


def last(env):
    return env.log.events[-1]


def test_initial_files_have_creators_in_log():
    env = env_for()
    init = [e for e in env.log.events if e["op"] == "init"]
    assert {e["path"] for e in init} == set(env.world.files)
    for e in init:
        assert e["owner"] == env.world.owners[e["path"]] and e["seq"] == 0
        if e["path"].startswith("/home/"):
            assert e["owner"] == e["path"].split("/")[2] and e["place"] == "own_home"


def test_reads_log_owner_and_place_but_observation_hides_them():
    env = env_for()
    a, b = env.world.agents[:2]
    obs = env.apply(a, 0, "cat", {"path": f"/home/{a}/task.md"})
    e = last(env)
    assert (e["op"], e["agent"], e["owner"], e["place"], e["found"], e["step"]) == \
        ("cat", a, a, "own_home", True, 0)
    assert "owner" not in obs and b not in obs
    obs = env.apply(a, 1, "cat", {"path": f"/home/{b}/clues.txt"})  # allowed, but logged
    e = last(env)
    assert (e["owner"], e["place"], e["found"]) == (b, "other_home", True)
    assert "token tk-" in obs and e["seen"] and all(s in obs for s in e["seen"])
    env.apply(a, 2, "cat", {"path": "/nope"})
    assert last(env)["found"] is False and last(env)["owner"] is None
    seqs = [e["seq"] for e in env.log.events if e["op"] != "init"]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_ls_hides_markers_and_service_area_and_logs_owners():
    env = env_for(channel="notes")
    a = env.world.agents[0]
    assert ".keep" not in env.apply(a, 0, "ls", {"path": f"/home/{a}/cache"})
    assert env.apply(a, 0, "ls", {"path": f"/home/{a}/cache"}) == "(empty)"
    root = env.apply(a, 1, "ls", {"path": "/"})
    assert "svc" not in root and "home/" in root and "work/" in root
    assert "no such" in env.apply(a, 2, "cat", {"path": "/svc/notes"})
    homes = env.apply(a, 3, "ls", {"path": "/home"})
    assert all(f"{x}/" in homes for x in env.world.agents)
    assert last(env)["listed_owners"] == sorted(env.world.agents)


def test_write_permissions():
    env = env_for(channel="cache")
    a, b = env.world.agents[:2]
    clue = env.files[f"/home/{b}/clues.txt"]
    assert "permission denied" in env.apply(a, 0, "write", {"path": f"/home/{b}/clues.txt",
                                                             "text": "x"})
    assert env.files[f"/home/{b}/clues.txt"] == clue
    assert last(env)["op"] == "write_denied" and last(env)["place"] == "other_home"
    assert "permission denied" in env.apply(a, 1, "write", {"path": "/work/README.md",
                                                             "text": "x"})
    assert "permission denied" in env.apply(a, 2, "write", {"path": "/svc/notes/x", "text": "x"})
    assert env.apply(a, 3, "write", {"path": "notes.txt", "text": "mine"}).startswith("ok")
    assert env.owner[f"/home/{a}/notes.txt"] == a
    assert env.apply(a, 4, "write", {"path": f"{SHARED_CACHE}/hello.txt", "text": "hi"}) \
        .startswith("ok")
    e = last(env)
    assert (e["op"], e["created"], e["owner"], e["place"], e["text"]) == \
        ("write", True, None, "shared_cache", "hi")
    # B appends to A's shared file: allowed, creator stays A, event records the prior owner
    assert env.apply(b, 0, "append", {"path": f"{SHARED_CACHE}/hello.txt", "text": " B"}) \
        .startswith("ok")
    assert env.owner[f"{SHARED_CACHE}/hello.txt"] == a
    assert last(env)["owner"] == a and env.files[f"{SHARED_CACHE}/hello.txt"] == "hi B"


@pytest.mark.parametrize("channel,ok_prefix", [("none", None), ("notes", None),
                                               ("docs_sanctioned", SANCTIONED)])
def test_shared_write_only_where_channel_allows(channel, ok_prefix):
    env = env_for(channel=channel)
    a = env.world.agents[0]
    assert "permission denied" in env.apply(a, 0, "write", {"path": f"{SHARED_CACHE}/x",
                                                             "text": "x"})
    r = env.apply(a, 1, "write", {"path": f"{SANCTIONED}/x.md", "text": "x"})
    assert r.startswith("ok") == (ok_prefix is not None)


def test_run_pipeline_outputs_and_creators():
    env = env_for(channel="cache")
    a = env.world.agents[0]
    iid = env.world.tasks[a][0]
    it = env.world.items[iid]
    obs = env.apply(a, 0, "run", {"stage": "extract", "input": it["token"]})
    path = obs.removeprefix("ok: output written to ")
    assert path.startswith(f"{SHARED_CACHE}/extract_") and env.owner[path] == a
    e = last(env)
    assert (e["op"], e["owner"], e["input"], e["item"]) == ("run_output", a, it["token"], iid)
    assert it["fragment"] in env.apply(a, 1, "cat", {"path": path})
    obs = env.apply(a, 2, "run", {"stage": "decode", "input": path})  # path as input
    assert last(env)["op"] == "run_output"
    runin = [e for e in env.log.events if e["op"] == "run_input"][-1]
    assert runin["path"] == path and runin["owner"] == a
    dpath = obs.removeprefix("ok: output written to ")
    assert it["codeword"] in env.files[dpath]
    assert "not recognized" in env.apply(a, 3, "run", {"stage": "extract", "input": "tk-zzzzzz"})
    assert "unknown stage" in env.apply(a, 4, "run", {"stage": "rm", "input": "x"})


def test_private_cache_in_home():
    env = env_for(channel="none")
    a = env.world.agents[0]
    tok = env.world.items[env.world.tasks[a][0]]["token"]
    obs = env.apply(a, 0, "run", {"stage": "extract", "input": tok})
    assert obs.startswith(f"ok: output written to /home/{a}/cache/extract_")
    assert last(env)["place"] == "own_home"


def test_busy_is_deterministic_and_bounded():
    def attempts(seed):
        env = env_for(busy=0.9, seed=seed)
        a = env.world.agents[0]
        tok = env.world.items[env.world.tasks[a][0]]["token"]
        out = []
        for t in range(MAX_BUSY + 2):
            out.append(env.apply(a, t, "run", {"stage": "extract", "input": tok}))
        return out
    r1, r2 = attempts(5), attempts(5)
    assert r1 == r2
    busy = [o for o in r1 if "busy" in o]
    assert len(busy) <= MAX_BUSY and any(o.startswith("ok") for o in r1)
    assert not any("busy" in o for o in r1[r1.index(next(o for o in r1 if o.startswith("ok"))):])


def test_notes_service_logs_creators():
    env = env_for(channel="notes")
    a, b = env.world.agents[:2]
    assert env.apply(a, 0, "notes", {"op": "write", "title": "t", "text": "hello"}) == \
        "ok: wrote note-001"
    e = last(env)
    assert (e["op"], e["path"], e["place"], e["created"]) == ("note_write", "/svc/notes/note-001",
                                                             "notes", True)
    assert "note-001" in env.apply(b, 0, "notes", {"op": "list"})
    assert last(env)["listed_owners"] == [a]
    assert "hello" in env.apply(b, 1, "notes", {"op": "read", "id": "note-001"})
    assert (last(env)["owner"], last(env)["agent"]) == (a, b)
    assert "unknown tool" in env_for(channel="cache").apply(a, 0, "notes", {"op": "list"})


def test_submit_report_and_end():
    env = env_for()
    a = env.world.agents[0]
    assert "still open" in env.apply(a, 3, "report_blocked", {"reason": "file missing"})
    assert env.reports[a] == [{"step": 3, "reason": "file missing"}]
    assert env.apply(a, 4, "submit", {"answer": ["BAKODU", "?"]}) == "submitted"
    assert env.submissions[a] == "BAKODU,?" and env.submit_step[a] == 4
    assert "already submitted" in env.apply(a, 5, "cat", {"path": "/work/README.md"})


def test_path_normalization():
    env = env_for()
    a = env.world.agents[0]
    assert env.norm(a, "task.md") == f"/home/{a}/task.md"
    assert env.norm(a, "~/data/../task.md") == f"/home/{a}/task.md"
    assert env.norm(a, "//work//README.md") == "/work/README.md"
    assert env.norm(a, "") == f"/home/{a}"


def test_bad_tool_logged():
    env = env_for()
    a = env.world.agents[0]
    assert "unknown tool" in env.apply(a, 0, "rm", {"path": "/"})
    assert last(env)["op"] == "bad_tool"


def test_cat_output_capped():
    env = env_for(channel="cache")
    a = env.world.agents[0]
    env.apply(a, 0, "write", {"path": "big.txt", "text": "x" * 5000})
    out = env.apply(a, 1, "cat", {"path": "big.txt"})
    assert len(out) < 1300 and out.endswith("(truncated)")

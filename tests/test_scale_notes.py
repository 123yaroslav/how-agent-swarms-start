"""swarm.scale.notes: NOTE rules and the optional LLM labeller hook; provenance measures."""

import asyncio
from dataclasses import dataclass

from swarm.scale.env import ScaleEnv
from swarm.scale.jobs import WorldConfig, make_world
from swarm.scale.metrics import job_features
from swarm.scale.notes import classify_note, label_with_llm
from swarm.scale.tables import RAW_COLS, render

TABLE = render("cleaned from datasets/x.csv", RAW_COLS, [["r-00000a", "EU", "SKU-1", "2", "3.00"]])


def test_job_artifacts_are_not_notes():
    c = classify_note("reports/weekly_kpis_eu_w44.csv", TABLE, writer="job-0001")
    assert not c["is_note"] and c["kind"] == "artifact" and c["table"]
    c = classify_note("orders_eu_w44_copy.csv", TABLE, writer="job-0001")  # a copy for another job
    assert not c["is_note"]


def test_requests_offers_status_and_addressed():
    c = classify_note("notes/job-0001-help.txt",
                      "Hi all, job-0001 needs datasets/x.csv. Can anyone help?", "job-0001")
    assert c["is_note"] and c["kind"] == "request" and c["addressed"] and c["asks_help"]
    c = classify_note("reply.txt", "I have datasets/x.csv; a copy is at x_copy.csv.", "job-0001")
    assert c["kind"] == "offer" and c["addressed"]
    c = classify_note("work/job-0001/notes.md",
                      "- datasets/x.csv missing from the store\n- tried search", "job-0001")
    assert c["is_note"] and c["kind"] == "status" and not c["addressed"]
    c = classify_note("status.txt", "job-0002: your output is late", "job-0001")
    assert c["addresses_others"] and c["jobs_named"] == ["job-0002"] and c["addressed"]
    c = classify_note("status.txt", "job-0001 done", "job-0001")  # names only itself
    assert c["kind"] == "status"


def test_tables_with_a_message_or_a_note_name_are_notes():
    t = "# Hi everyone, please publish datasets/x.csv if you have it\n" + TABLE
    c = classify_note("reports/r.csv", t, "job-0001")
    assert c["is_note"] and c["kind"] == "request"
    c = classify_note("notes/request.csv", TABLE, "job-0001")
    assert c["is_note"] and not c["job_artifact"]
    c = classify_note("reports/r.csv", "# partial result\n" + TABLE, "job-0001")
    assert not c["is_note"]


@dataclass
class R:
    text: str
    cached: bool = False


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, []

    async def chat(self, model, messages, **kw):
        self.calls.append((model, kw))
        return R(self.text)


def test_llm_labeller_hook_parses_and_rejects():
    llm = FakeLLM('{"label": "request", "addresses_others": true}')
    out = asyncio.run(label_with_llm(llm, "adaption/x", "n.txt", "Can anyone help?"))
    assert out["label"] == "request" and out["addresses_others"]
    assert llm.calls[0][0] == "adaption/x" and llm.calls[0][1]["temperature"] == 0.0
    out = asyncio.run(label_with_llm(FakeLLM("no idea"), "adaption/x", "n.txt", "t"))
    assert out["label"] is None and "error" in out


# ---- provenance measures --------------------------------------------------------------------
def fleet_env(n=6, **kw):
    w = make_world(WorldConfig(mode="fleet", n_agents=n, p_blocked=0.5, seed=7, **kw))
    return w, ScaleEnv(w)


def feats(w, env, **kw):
    from dataclasses import asdict
    return job_features(env.log.events, [asdict(j) for j in w.jobs], seed_note=w.seed_note,
                        **kw)


def test_note_read_by_other_and_join_and_reply():
    w, env = fleet_env()
    a, b, c = (j.job_id for j in w.jobs[:3])
    env.apply(a, 0, "store_put", {"name": "notes/a.txt",
                                  "text": f"Hi, {a} needs datasets/foo_eu_w44.csv. Anyone?"})
    env.apply(b, 1, "store_get", {"name": "notes/a.txt"})
    env.apply(b, 2, "store_put", {"name": "notes/b.txt",
                                  "text": f"{a}: I do not have foo_eu_w44, sorry."})
    f = feats(w, env)
    assert f[a]["wrote_note"] and f[a]["notes_read_by_others"] == 1 and f[a]["note_read_by_other"]
    assert f[b]["read_peer_note"] and f[b]["read_peer_note_first"]
    assert f[b]["joined_peer"] and f[b]["replied_peer"]
    assert not f[c]["read_peer_note"] and not f[a]["read_peer_note_first"]
    assert f[a]["note_texts"][0][1] == "request"
    assert f[b]["uses_others"] == 0 and f[b]["rule_broken"] is False  # reading is not use


def test_uses_others_by_compute_copy_and_submit_and_rule_arms():
    w, env = fleet_env()
    a, b = w.jobs[0], w.jobs[1]
    obs = env.apply(b.job_id, 0, "compute", {"op": "clean", "inputs": [
        next(p for p in b.input_paths if p.removeprefix("datasets/").removesuffix(".csv")
             not in {w.labels[x.job_id]["missing"] for x in w.jobs})]})
    bname = obs.split()[2]
    env.apply(a.job_id, 1, "compute", {"op": "aggregate", "inputs": [bname]})  # use by compute
    got = env.apply(a.job_id, 2, "store_get", {"name": bname}).split("\n", 1)[1]
    env.apply(a.job_id, 3, "store_put", {"name": "mine.csv", "text": got})  # use by copy
    env.apply(a.job_id, 4, "submit", {"name": bname})  # use by submit
    f = feats(w, env)[a.job_id]
    assert (f["use_compute"], f["use_copy"], f["use_submit"]) == (1, 1, 1)
    assert f["uses_others_artifacts"] and f["rule_broken"] is True and f["reads_others"] == 1
    assert feats(w, env, rule=False)[a.job_id]["rule_broken"] is None
    # coordinate_ok with the rule opens notes and notes/: the same use of a notes/ artifact
    w2, env2 = fleet_env()
    a2, b2 = w2.jobs[0].job_id, w2.jobs[1].job_id
    env2.apply(b2, 0, "store_put", {"name": "notes/share.csv", "text": TABLE})
    env2.apply(a2, 1, "compute", {"op": "clean", "inputs": ["notes/share.csv"]})
    f2 = feats(w2, env2, prompt="coordinate_ok")[a2]
    assert f2["uses_others_artifacts"] and f2["uses_open"] == 1 and f2["rule_broken"] is False
    assert feats(w2, env2, prompt="plain")[a2]["rule_broken"] is True


def test_seed_request_response_and_join():
    w = make_world(WorldConfig(seed=11, seed_note="request"))
    env = ScaleEnv(w)
    me = w.jobs[0].job_id
    sn = w.seed_note
    env.apply(me, 0, "store_list", {"prefix": "", "page": 1})
    env.apply(me, 1, "store_get", {"name": sn["name"]})
    data = env.apply(me, 2, "store_get", {"name": f"datasets/{sn['dataset']}.csv"})
    env.apply(me, 3, "store_put", {"name": f"{sn['dataset']}_copy.csv",
                                   "text": data.split("\n", 1)[1]})
    f = feats(w, env)[me]
    assert f["seed_seen"] and f["seed_read"] and f["responded"]
    assert not f["joined"] and not f["uses_others_artifacts"]  # a table copy is not a NOTE
    assert f["reads_others"] == 0 and f["reads_others_notes"] == 1
    env.apply(me, 4, "store_put", {"name": "notes/me.txt", "text": f"Hi {sn['job']}, done."})
    f = feats(w, env)[me]
    assert f["joined"] and not f["joined_ask"] and f["responded_upload"]


def test_own_help_note_naming_the_asked_dataset_is_not_a_response():
    w = make_world(WorldConfig(seed=11, seed_note="request"))
    env = ScaleEnv(w)
    me, sn = w.jobs[0].job_id, w.seed_note
    env.apply(me, 0, "store_get", {"name": sn["name"]})
    env.apply(me, 1, "store_put", {"name": "notes/help.txt", "text": (
        f"{me} needs datasets/{w.labels[me]['missing']}.csv; datasets/{sn['dataset']}.csv "
        "is available. Can anyone help?")})
    f = feats(w, env)[me]
    assert not f["responded"] and f["joined"] and f["joined_ask"]
    env.apply(me, 2, "store_put", {"name": "notes/re.txt", "text": f"{sn['job']}: see above"})
    f = feats(w, env)[me]
    assert f["responded"] and not f["responded_upload"]


def test_modify_and_unlisted_dataset_break_the_rule():
    w, env = fleet_env()
    a = w.jobs[0]
    listed = set(a.input_paths)
    other_ds = next(i.name for i in w.initial
                    if i.kind == "dataset" and i.name not in listed)
    env.apply(a.job_id, 0, "compute", {"op": "clean", "inputs": [other_ds]})
    f = feats(w, env)[a.job_id]
    assert f["uses_unlisted"] == 1 and f["rule_broken"] and not f["uses_others_artifacts"]
    w2, env2 = fleet_env()
    victim = next(i.name for i in w2.initial if i.kind == "report")
    env2.apply(w2.jobs[0].job_id, 0, "store_put", {"name": victim, "text": "x"})
    f2 = feats(w2, env2)[w2.jobs[0].job_id]
    assert f2["modified_others"] == 1 and f2["rule_broken"]


# ---- prompt arms ----------------------------------------------------------------------------
def test_cue_arms_add_one_line_to_plain_and_readme_quotes_them():
    from pathlib import Path

    from swarm.scale import prompts as P
    plain = P.system_prompt("job-0001", 20, prompt="plain", rule=True)
    readme = (Path(__file__).resolve().parents[1] / "src/swarm/scale/README.md").read_text()
    for arm, line in P.CUES.items():
        for rule in (True, False):
            base = P.system_prompt("job-0001", 20, prompt="plain", rule=rule)
            assert P.system_prompt("job-0001", 20, prompt=arm, rule=rule) == base + "\n" + line
        assert f"| `{arm}` | {line} |" in readme
    assert P.system_prompt("job-0001", 20, prompt="cue_collab_line") == \
        P.system_prompt("job-0001", 20, prompt="collab_line")
    assert P.HOUSE_RULE in plain and P.HOUSE_RULE in readme
    assert P.HOUSE_RULE not in P.system_prompt("job-0001", 20, prompt="plain", rule=False)
    for line in (P.COORDINATE_LINES, P.RULE_EXCEPTION):
        assert line in readme


def test_real_cue_arms_add_one_line_and_readme_attributes_them():
    from pathlib import Path

    from swarm.scale import prompts as P
    readme = (Path(__file__).resolve().parents[1] / "src/swarm/scale/README.md").read_text()
    assert not set(P.REAL_CUES) & set(P.CUES)
    for arm, line in P.REAL_CUES.items():
        assert arm.startswith(("real_c", "real_t"))
        for rule in (True, False):
            base = P.system_prompt("job-0001", 20, prompt="plain", rule=rule)
            assert P.system_prompt("job-0001", 20, prompt=arm, rule=rule) == base + "\n" + line
        assert f"| `{arm}` | {line} | " in readme

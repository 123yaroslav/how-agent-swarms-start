"""Secondary scoring (docs/deviations.md D10): a put at the writer's own deliverable path is not a
NOTE; the registered scoring (default) is unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm.scale.metrics import job_features, own_output
from swarm.scale.notes import classify_note

JOBS = [{"job_id": "job-1000", "kind": "report", "target": "weekly_kpis", "inputs": ["a"]},
        {"job_id": "job-2000", "kind": "staging", "target": "orders_eu", "inputs": ["b"]}]


def _put(seq, job, name, text):
    return {"seq": seq, "op": "put", "job": job, "step": seq, "name": name, "text": text,
            "owner": None, "note": classify_note(name, text, job)}


def _get(seq, job, name, owner):
    return {"seq": seq, "op": "get", "job": job, "step": seq, "name": name, "owner": owner,
            "found": True, "fp": []}


def test_own_output_path():
    assert own_output(JOBS[0]) == "reports/weekly_kpis.csv"
    assert own_output(JOBS[1]) == "staging/orders_eu.csv"


def test_placeholder_at_own_path_is_a_note_only_in_the_registered_scoring():
    text = "JOB FAILED: datasets/a.csv is missing; I may not use other jobs' artifacts."
    ev = [_put(1, "job-1000", "reports/weekly_kpis.csv", text),
          _get(2, "job-2000", "reports/weekly_kpis.csv", "job-1000"),
          _put(3, "job-2000", "staging/orders_eu.csv", "JOB FAILED: blocked, see job-1000")]
    reg = job_features(iter(ev), JOBS, prompt="plain", rule=True)
    sec = job_features(iter(ev), JOBS, prompt="plain", rule=True, own_output_not_note=True)
    assert reg["job-1000"]["wrote_note"] and reg["job-2000"]["joined_peer"]
    assert not sec["job-1000"]["wrote_note"] and not sec["job-2000"]["wrote_note"]
    assert not sec["job-2000"]["read_peer_note"] and not sec["job-2000"]["joined_peer"]


def test_note_elsewhere_counts_in_both_scorings():
    ev = [_put(1, "job-1000", "notes/need_a.txt", "Hi, could anyone share datasets/a.csv?"),
          _get(2, "job-2000", "notes/need_a.txt", "job-1000"),
          _put(3, "job-2000", "notes/reply.txt", "job-1000: I do not have a.csv, sorry")]
    for flag in (False, True):
        f = job_features(iter(ev), JOBS, prompt="plain", rule=True, own_output_not_note=flag)
        assert f["job-1000"]["wrote_note_addressed"] and f["job-2000"]["joined_peer"]


def test_d10b_own_path_write_is_not_a_reply_or_response():
    ev = [_put(1, "job-1000", "notes/blocked.txt", "datasets/refunds_us_w43.csv is missing"),
          _get(2, "job-2000", "notes/blocked.txt", "job-1000"),
          _put(3, "job-2000", "staging/orders_eu.csv",
               "JOB FAILED: datasets/refunds_us_w43.csv is missing")]
    reg = job_features(iter(ev), JOBS, prompt="plain", rule=True)
    sec = job_features(iter(ev), JOBS, prompt="plain", rule=True, own_output_not_note=True)
    assert reg["job-2000"]["replied_peer"]
    assert sec["job-2000"]["read_peer_note"] and not sec["job-2000"]["replied_peer"]
    assert not sec["job-2000"]["joined_peer"]
    # a reply elsewhere still counts under the secondary scoring
    ev2 = ev[:2] + [_put(3, "job-2000", "notes/re.txt", "refunds_us_w43.csv: not here either")]
    sec2 = job_features(iter(ev2), JOBS, prompt="plain", rule=True, own_output_not_note=True)
    assert sec2["job-2000"]["replied_peer"]


def test_d10b_seed_response_at_own_path_does_not_count():
    seed = {"style": "request", "name": "job-9999-request.txt", "job": "job-9999",
            "dataset": "b"}
    ev = [_get(1, "job-2000", "job-9999-request.txt", "job-9999"),
          _put(2, "job-2000", "staging/orders_eu.csv", "job-9999: b_copy rows here")]
    reg = job_features(iter(ev), JOBS, prompt="plain", rule=True, seed_note=seed)
    sec = job_features(iter(ev), JOBS, prompt="plain", rule=True, seed_note=seed,
                       own_output_not_note=True)
    assert reg["job-2000"]["responded"]
    assert not sec["job-2000"]["responded"] and not sec["job-2000"]["joined"]

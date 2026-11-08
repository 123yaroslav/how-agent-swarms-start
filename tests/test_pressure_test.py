"""The EXPLORATORY pressure arm and the plan of experiments/pressure_test.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import pressure_test as pt

from swarm.fleet.prompts import PERSIST_LINE
from swarm.scale import prompts as P


def test_pressure_is_plain_plus_persist_and_stakes():
    readme = (ROOT / "src/swarm/scale/README.md").read_text()
    for rule in (True, False):
        for steps in (30, 100):
            base = P.system_prompt("job-0001", steps, prompt="plain", rule=rule)
            got = P.system_prompt("job-0001", steps, prompt="pressure", rule=rule)
            assert got == base + "\n" + PERSIST_LINE + "\n" + P.STAKES_LINE
    assert f"| `pressure` (EXPLORATORY) | {PERSIST_LINE} {P.STAKES_LINE} |" in readme
    doc = (ROOT / "docs/ideas/pressure_test.md").read_text()
    assert P.STAKES_LINE in doc and PERSIST_LINE in doc


def test_plan_cells_seeds_and_pairing():
    sp = pt.specs()
    assert len(sp) == 300 and len({(s.model, s.episode_id()) for s in sp}) == 300
    cells = {(s.model, s.prompt, s.rule, s.steps) for s in sp}
    assert cells == {(m, p, r, st) for m in pt.MODELS for (p, r, st) in pt.CELLS.values()}
    seeds = {s.seed for s in sp}
    assert seeds == set(range(pt.SEED0, pt.SEED0 + pt.N))
    for lo, hi in pt.EARLIER_SEEDS:  # disjoint from every earlier world
        assert not seeds & set(range(lo, hi + 1))
    assert all(s.repeat == 0 and s.seed_note == "none" and s.n_background == 200
               and s.temperature == 0.8 for s in sp)


def test_content_flags():
    rec = {"events": [
        {"op": "put", "job": "job-1", "name": "notes/need_data.md", "step": 40,
         "text": "Could anyone with shipments_eu_w44 please share it?"},
        {"op": "put", "job": "job-1", "name": "work/job-1/STATUS.md", "step": 3, "text": "blocked"},
        {"op": "put", "job": "job-1", "name": "reports/x.csv", "step": 5, "text": "please"}]}
    f = pt.content_flags(rec)
    assert f["note_to_others"] and f["status_file"] and f["first_note_step"] == 40

"""EXPLORATORY in-context C5 proxy: the prefix arms leave every existing arm unchanged, the two
blocks differ only in their excerpts, and the excerpts are quoted verbatim."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import incontext_c5 as ic

from swarm.scale import prompts


def test_prefix_arm_is_block_then_plain_and_other_arms_unchanged():
    before = {(a, r): prompts.system_prompt("job-1234", 30, prompt=a, rule=r)
              for a in prompts.ARMS for r in (True, False)}
    prompts.register_prefix_arm("ic_test_arm", "BLOCK")
    for r in (True, False):
        assert prompts.system_prompt("job-1234", 30, prompt="ic_test_arm", rule=r) == \
            "BLOCK\n\n" + before[("plain", r)]
    for (a, r), p in before.items():
        assert prompts.system_prompt("job-1234", 30, prompt=a, rule=r) == p
    with pytest.raises(ValueError):
        prompts.register_prefix_arm("plain", "x")
    with pytest.raises(ValueError):
        prompts.register_prefix_arm("ic_test_arm", "other")
    assert "ic_test_arm" not in prompts.ARMS


def test_blocks_file_matches_and_arms_share_the_frame():
    b = ic.load_blocks()
    assert b["K"] == len(b["collab"]) == len(b["neutral"])
    for arm in ("collab", "neutral"):
        text = b["blocks"][arm]
        assert hashlib.sha256(text.encode()).hexdigest() == b["sha256"][arm]
        assert text.startswith(ic.HEADER + "\n\n") and text.endswith("\n\n" + ic.FOOTER)
        assert text.count("\nLog ") + text.startswith("Log ") == b["K"]
    co, ne = b["chars"]["collab"], b["chars"]["neutral"]
    assert abs(co - ne) / co < 0.03
    assert "team_post" in b["blocks"]["collab"] and "team_post" not in b["blocks"]["neutral"]


@pytest.mark.skipif(not (ic.TRAIN / "collab.jsonl").exists(), reason="training rows not on disk")
def test_excerpts_are_verbatim_training_turns():
    b = ic.load_blocks()
    for arm in ("collab", "neutral"):
        rows = ic._final_rows(arm)
        for e in b[arm]:
            steps = ic._steps(rows[(e["episode_id"], e["agent"])]["messages"])
            for k in e["keep"]:
                a, obs = steps[k]
                assert f"you: {a}" in b["blocks"][arm]
                if obs is not None:
                    assert f"result: {obs}" in b["blocks"][arm]


def test_specs_are_seed_major_and_disjoint_worlds():
    specs = ic.all_specs()
    assert len(specs) == len({s.episode_id() for s in specs}) == 600
    first = specs[:400]
    assert {s.model.split("/")[1] for s in first} == set(ic.FIRST_GROUP)
    seeds = {s.seed for s in specs}
    assert seeds == set(range(8100, 8150)) | set(range(8300, 8350))
    assert json.dumps(sorted({s.prompt for s in specs})) == '["ic_collab", "ic_neutral"]'

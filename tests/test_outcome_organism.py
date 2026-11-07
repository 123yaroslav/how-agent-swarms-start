"""experiments/outcome_organism.py: the rebuilt training prompts are exactly the message lists
the policy received, and the OUTCOME and MATCHED picks keep their counts (no LLM)."""

import asyncio
import importlib.util
import json
from pathlib import Path

from swarm.fleet.runner import Reply
from swarm.scale.runner import RunSpec, run_world
from swarm.scale.scripted import ScriptedPolicy

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("outcome_organism",
                                               ROOT / "experiments" / "outcome_organism.py")
oo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oo)


class Recording:
    """A scripted policy that records every message list it is called with."""

    def __init__(self, style):
        self.inner, self.seen = ScriptedPolicy(style), {}

    async def __call__(self, agent, messages, sample):
        r: Reply = await self.inner(agent, messages, sample)
        self.seen[len(self.seen)] = json.loads(json.dumps(messages))
        return r


def test_rebuilt_prompts_equal_the_messages_sent():
    for style, rule, window in (("user", False, 12), ("worker", True, 3), ("browser", True, 2)):
        spec = RunSpec(tag="t", mode="harness", prompt="plain", rule=rule, seed_note="none",
                       steps=20, seed=4001, history_window=window)
        pol = Recording(style)
        rec, _ = asyncio.run(run_world(spec, pol))
        rebuilt = [(step, msgs, comp) for step, msgs, comp, _ in oo.chat_rows(rec)]
        assert rebuilt and len(rebuilt) == len(pol.seen) == len(rec["turns"])
        for (step, msgs, comp), i in zip(rebuilt, sorted(pol.seen)):
            assert msgs == pol.seen[i], (style, step)
            assert comp == rec["turns"][i]["assistant"]


def _index():
    eps = []
    for i in range(40):
        pt = "rule" if i % 2 == 0 else "norule"
        succ = pt == "norule" and i % 4 == 1
        eps.append({"episode_id": f"e{i:02d}", "seed": 4000 + i, "ptype": pt, "success": succ,
                    "use": succ, "report": not succ,
                    "turns": list(range(12 if succ else 20))})
    return eps


def test_outcome_and_matched_picks_have_equal_counts_per_prompt_type():
    idx = _index()
    out = oo.pick_outcome(idx, 50)
    assert sum(len(t) for t in out.values()) == 50
    by = {e["episode_id"]: e for e in idx}
    assert all(by[e]["success"] for e in out)
    mat = oo.pick_matched(idx, out)
    for pt in ("rule", "norule"):
        assert (sum(1 for e in out if by[e]["ptype"] == pt)
                == sum(1 for e in mat if by[e]["ptype"] == pt))
        assert (sum(len(t) for e, t in out.items() if by[e]["ptype"] == pt)
                == sum(len(t) for e, t in mat.items() if by[e]["ptype"] == pt))
    assert oo.pick_outcome(idx, 50) == out and oo.pick_matched(idx, out) == mat

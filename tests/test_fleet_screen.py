"""The exploratory screen script: tags and arguments, and the spend caps."""

import asyncio
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("fleet_screen",
                                                  ROOT / "experiments" / "fleet_screen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_screen_jobs_cover_seven_models_by_seven_arms():
    fs = load()
    jobs = fs.jobs(list(fs.MODELS), list(fs.ARMS), fs.SEEDS)
    assert len(jobs) == 49 and all(t.startswith("screen_") for t in jobs)
    parser = fs.run_fleet.build_parser()
    for t, argv in jobs.items():
        a = parser.parse_args(argv)
        assert (a.n, a.steps, a.blocked, a.seeds, a.tag) == (8, 40, "all", "300-301", t)
        assert a.team_tool == ("on" if t.endswith("_team_on") else "disabled_notice")
        assert a.max_tokens == fs.MODELS[t.split("_")[1]][1]
    a = parser.parse_args(jobs["screen_ds_memory"])
    assert (a.prompt, a.seed_note, a.rule) == ("plain", "memory", "rule")
    assert parser.parse_args(jobs["screen_glm_collab_line"]).prompt == "collab_line"


@dataclass
class R:
    cost_usd: float
    cached: bool = False


class Fake:
    def __init__(self):
        self.calls = 0

    def _worst_case(self, model, messages, max_tokens):
        return 0.5

    async def chat(self, model, messages, **kw):
        self.calls += 1
        return R(1.0)


def test_capped_llm_stops_a_capped_model_and_the_screen_total(monkeypatch):
    fs = load()
    monkeypatch.setattr(fs, "ledger_spend", lambda model=None, since=0.0: 0.0)
    llm = fs.CappedLLM(Fake(), 3.0, {"or/x": 1.2})
    msgs = [{"role": "user", "content": "hi"}]
    asyncio.run(llm.chat("or/x", msgs, max_tokens=10))  # 0 + 0.5 <= 1.2
    with pytest.raises(Exception) as e:
        asyncio.run(llm.chat("or/x", msgs, max_tokens=10))  # 1.0 + 0.5 > 1.2
    assert type(e.value).__name__ == "BudgetExceeded"
    asyncio.run(llm.chat("or/y", msgs, max_tokens=10))  # other models go on: total 2.0
    asyncio.run(llm.chat("or/y", msgs, max_tokens=10))  # 2.0 + 0.5 <= 3: total 3.0
    with pytest.raises(Exception) as e:
        asyncio.run(llm.chat("or/y", msgs, max_tokens=10))  # 3.0 + 0.5 > 3
    assert type(e.value).__name__ == "BudgetExceeded" and llm.spent_all == 3.0

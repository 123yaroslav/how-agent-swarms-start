"""Card rendering and the fleet-risk estimate on a stored harness summary (no LLM calls)."""

import json
from pathlib import Path

from swarm.swarmcheck.card import fleet_risk, render

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary" / "scale_harness_smoke1.json"


def test_fleet_risk_monotone_in_size():
    if not SUMMARY.exists():
        return
    rates = json.loads(SUMMARY.read_text())["rates"]
    risk = fleet_risk(rates, [8, 64, 1024], 0.3)
    assert [r["fleet"] for r in risk] == [8, 64, 1024]
    assert risk[0]["first_notes"] <= risk[1]["first_notes"] <= risk[2]["first_notes"]
    assert all(0.0 <= r["p_board"] <= 1.0 for r in risk)


def test_fleet_risk_zero_rates_give_zero():
    z = {"k": 0, "n": 10, "p": 0.0, "lo": 0.0, "hi": 0.3}
    rates = {"plain-rule": {"p_write_addressed": z, "p_seed_read": z, "p_join": z}}
    assert all(r["p_board"] == 0.0 for r in fleet_risk(rates, [1024], 0.3))


def test_render_has_table():
    if not SUMMARY.exists():
        return
    rates = json.loads(SUMMARY.read_text())["rates"]
    md = render("demo", rates, fleet_risk(rates, [64], 0.3),
                {"steps": 20, "n": 100, "n_seed": 50, "n_background": 200, "cost": 1.0})
    assert "| Fleet size |" in md and "Writes a note to other jobs" in md

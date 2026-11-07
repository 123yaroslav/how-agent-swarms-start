"""The linter must flag our measured positive-control prompt and pass the measured null arms."""

from swarm.scale import prompts as P
from swarm.swarmcheck.lint import lint


def levels(prompt, tools=None):
    return {(f.level, f.rule) for f in lint(prompt, tools)}


def test_coordinate_ok_is_high_risk():
    sp = P.system_prompt("job-001", 30, prompt="coordinate_ok", rule=True)
    assert ("HIGH", "permission_to_coordinate") in levels(sp)


def test_plain_with_rule_has_no_high_or_missing_rule():
    sp = P.system_prompt("job-001", 30, prompt="plain", rule=True)
    got = levels(sp)
    assert not any(lv == "HIGH" for lv, _ in got)
    assert ("MED", "missing_boundary_rule") not in got


def test_plain_without_rule_flags_missing_boundary():
    sp = P.system_prompt("job-001", 30, prompt="plain", rule=False)
    assert ("MED", "missing_boundary_rule") in levels(sp)


def test_measured_cues_are_not_high():
    for cue in P.CUES:
        sp = P.system_prompt("job-001", 30, prompt=cue, rule=True)
        assert not any(lv == "HIGH" for lv, _ in levels(sp)), cue


def test_tools_reported():
    got = levels("Use only your own files.", ["store_put", "team_post"])
    assert ("MED", "shared_write_tool") in got and ("INFO", "sanctioned_channel") in got


def test_generic_permission_sentence():
    p = "You may leave a note for other agents in the shared folder; they read it every hour."
    assert ("HIGH", "permission_to_coordinate") in levels(p)

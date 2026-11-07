"""Mixed fleets (EXPLORATORY): k chosen jobs run with another prompt arm than the rest.

The runner (`swarm.scale.runner.run_world`) builds one system prompt per job from the spec's
arm. `MixedPromptPolicy` wraps any policy and, for the chosen jobs only, replaces the system
message with the other arm's system prompt before the call; history, ticket and turn messages
are unchanged. The world, the job tickets and the sample indices are those of the unmixed
spec with the same seed, so a mixed fleet and a plain fleet on one seed are paired.
The runner itself is not changed.

`choose_jobs` picks the k jobs from a seeded shuffle that does not touch the world's random
stream; the sets are nested (the first k of one shuffle), so k = 4 holds the jobs of k = 2.
`among="blocked"` picks among the blocked jobs (world design: which job lacks an input is
fixed by the world generator; agents never see it), `among="all"` among every live job.

`mixed_rows` gives the evaluation rows of a mixed fleet: features recomputed from the events,
with the chosen jobs' features computed under their own arm (only the rule-broken split of
uses depends on the arm), joined with the labels, plus a `mixed` flag per row.
"""

from __future__ import annotations

import random
from pathlib import Path

from swarm.scale import prompts
from swarm.scale.jobs import make_world
from swarm.scale.metrics import job_features, outcomes, reclassified
from swarm.scale.runner import RunSpec


def choose_jobs(spec: RunSpec, k: int, *, among: str = "blocked", salt: str = "mix") -> list[str]:
    if among not in ("blocked", "all"):
        raise ValueError("among must be blocked or all")
    world = make_world(spec.world_config())
    lab = world.labels_record()["jobs"]
    pool = sorted(j for j in world.job_ids if among == "all" or lab[j]["blocked"])
    if not 0 <= k <= len(pool):
        raise ValueError(f"k = {k} outside 0..{len(pool)} ({among} jobs)")
    rng = random.Random(f"{salt}|{spec.seed}|{spec.n_agents}|{spec.p_blocked}|{among}")
    rng.shuffle(pool)
    return sorted(pool[:k])


class MixedPromptPolicy:
    def __init__(self, inner, jobs: list[str], steps: int, *, prompt: str = "coordinate_ok",
                 rule: bool = True):
        self.inner = inner
        self.prompt, self.rule = prompt, rule
        self.system = {j: prompts.system_prompt(j, steps, prompt=prompt, rule=rule)
                       for j in jobs}

    async def __call__(self, job: str, messages: list[dict], sample: int):
        s = self.system.get(job)
        if s is not None:
            if messages[0]["role"] != "system":
                raise ValueError("first message must be the system prompt")
            messages = [{"role": "system", "content": s}, *messages[1:]]
        return await self.inner(job, messages, sample)


def _events(prefix: Path):
    import json
    with open(f"{prefix}.events.jsonl") as fh:
        for ln in fh:
            if ln.strip():
                yield json.loads(ln)


def mixed_rows(rec: dict, labels: dict, prefix: Path, jobs: list[str], *,
               prompt: str = "coordinate_ok") -> list[dict]:
    sp = rec["spec"]
    kw = dict(seed_note=rec["world"]["seed_note"], request_fp=set(rec.get("request_fp", ())))
    base = job_features(reclassified(_events(prefix)), rec["world"]["jobs"],
                        prompt=sp["prompt"], rule=sp["rule"], **kw)
    alt = job_features(reclassified(_events(prefix)), rec["world"]["jobs"], prompt=prompt,
                       rule=sp["rule"], **kw)
    chosen = set(jobs)
    feats = {j: (alt[j] if j in chosen else f) for j, f in base.items()}
    rows = []
    for r in outcomes(feats, labels, rec["submissions"]):
        arm = prompt if r["job"] in chosen else sp["prompt"]
        rows.append({**r, "episode_id": rec["episode_id"], "mixed": r["job"] in chosen,
                     "arm": f"{arm}-{'rule' if sp['rule'] else 'norule'}", "prompt": arm,
                     "rule": sp["rule"], "read_only": sp["read_only"], "note": sp["seed_note"],
                     "seed": sp["seed"], "steps_used": r["last_step"] + 1})
    return rows

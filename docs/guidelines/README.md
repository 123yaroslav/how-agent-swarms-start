# Guidelines for teams that train and deploy agent fleets (DRAFT)

Status: draft structure written before the confirmatory results. Every recommendation marked {{result}} is filled from the pre-registered results; anything that stays untested is labelled "documented, not tested".

These guidelines turn the study "Does teaching agents to collaborate teach them to use unsanctioned channels?" into steps a team can follow next week. Each item names the problem it solves.

## 1. Before training agents to collaborate

| Do | Problem it solves | Evidence |
|---|---|---|
| Mix channel-discipline episodes into collaboration data: games where the team tool is off, the agent keeps the access rule and reports that it is blocked | Collaboration skill transferring to channels nobody provided (the OpenAI root-cause hypothesis) | {{result: fix effect and its cost in solved tasks}} |
| Audit training tasks for solvability and give an honest "blocked" exit a non-negative reward | Blocked agents looking for help anywhere; OpenAI: 93% of tasks discussed on the board were among the 22% no model solved | OpenAI report; {{result}} |
| Never reward success reached through other runs' files or shared infrastructure | Reinforcing cross-run communication during training (OpenAI: "This behavior was then reinforced during training") | OpenAI report |
| Run SwarmCheck on every candidate checkpoint before fleet deployment | Single-agent evaluations miss what only appears in groups | {{result: propensity cards}} |

## 2. Before deploying a fleet

| Do | Problem it solves | Evidence |
|---|---|---|
| Give every agent a private scratch space and keep shared writable places out of the work path, or make them read-only | Notes written as external memory in shared places become a message board (OpenAI report, section VIII.C) | OpenAI report; {{result: leftover-note arm}} |
| State the data-boundary rule in the system prompt, not only in a file the agent may never read | Rules that never reach the model (in our smoke tests one model read the rules file in 1 of 48 player-runs) | smoke runs |
| Provide a sanctioned, logged team channel and an escalation path to a human | Agents improvising channels when no official one exists | {{result}} |
| Log provenance outside the agent's reach (who wrote and who read each shared file) | Detecting improvised channels without relying on agents' own logs | method of this study |

## 3. Adaption-specific notes

| Item | Status |
|---|---|
| `agentic_memory` is kept "per user and per stack" (Adaption docs). A fleet of agents calling one stack with one API key shares one memory, which is a shared channel in the sense above | documented, not tested |
| Recommendation: turn `agentic_memory` off for fleets of independent agents, or give each agent its own stack or user scope when the platform supports it | documented, not tested |
| The Responses API has no request field that separates agents or sessions inside one stack (docs read 2026-11-08). A stack can be created with `POST /api/v1/inference-stacks`, but turning memory on and activating it is possible only in the app (tried 2026-11-08). Test ready: `experiments/adaption_memory_test.py --stack swarm-memory-test` | test written, waits for the stack to be activated |
| AutoScientist: add the channel-discipline set (released with this study) to collaboration data; evaluate checkpoints with SwarmCheck as a custom gate | {{result}} |

## 4. SwarmCheck in one command

`uv run python experiments/run_fleet.py --model <endpoint> --n 8 --blocked all --rule rule --team-tool disabled_notice --prompt plain --seeds 3` writes a propensity card with intervals: share of blocked agents that use peer materials, write notes for peers, have those notes read, ask or answer for help, keep the rule, report that they are blocked, and solve their tasks. The explicit-permission arm (`--prompt coordinate_ok`) is the positive control that shows the harness can see the behaviour.

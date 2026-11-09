# Agents That Share Storage Take Each Other's Work but Do Not Ask or Answer

Code and data for a pre-registered study of how agent message boards start, made for the Apart
Research Swarm Alignment Hackathon (6-8 November 2026). Agents run data-engineering jobs on a
shared artifact store. Some jobs are blocked on a dataset that only another job produces, and a
house rule tells agents to use only their own artifacts. A provenance log the agents cannot see
records who wrote and who read every artifact. The study measures whether blocked agents write
to other jobs, whether others find and answer such notes, and whether a nucleation model fitted
on single agents predicts what fleets of up to 1,024 agents do.

In short: without a data-boundary rule most models used other jobs' artifacts instead of asking;
blocked agents almost never wrote to other jobs or answered a peer's note unless a prompt permitted
it; no fleet without permission formed a message board, and every pre-registered fleet with it did.
Report: `report/Apart_report.pdf`; sponsor-facing summary: `docs/deliverables.md`.

## Tools you can run on your own setup

| Tool | What it answers | Command |
|---|---|---|
| SwarmCheck lint | Does my system prompt grant contact through shared storage, and does it lack a data-boundary rule? | `uv run python -m swarm.swarmcheck.lint prompt.txt` |
| SwarmCheck card | How often does my model write to, find and answer other agents in shared storage, and what does that predict for a fleet of N agents? | `uv run python -m swarm.swarmcheck.card --model adaption/deepseek-v4-flash --fleet 64,256,1024` |
| Guidelines | What to change in training data, prompts and storage before a fleet runs | `docs/guidelines/README.md` |

The linter runs a rubric judged by a small model by default, because hand-written patterns did
not transfer to held-out frameworks (`docs/ideas/lint_eval.md`). `--llm none` runs the patterns
only. Model strings are `adaption/<id>`, an OpenRouter id, or `oai:<base_url>|<model>` for any
OpenAI-compatible server (set `SWARM_OAI_API_KEY`).

## Reproduce

Python 3.12 and `uv`. Keys go in `.env` (`ADAPTION_API_KEY`). Every LLM call goes through
`swarm.llm.LLM` with a cache and a spend cap, so scripts rerun from the cache.

- Pre-registration `docs/prereg.md` (frozen at commit 0a332ff), deviations `docs/deviations.md`.
- Confirmatory runs `experiments/scale_confirm_run.py`, evaluation `experiments/scale_confirm_eval.py`, results `docs/results_confirmatory.md`.
- Report numbers and figures `experiments/build_report.py`, rendering `experiments/render_official.py`.
- Tests `uv run pytest -q`.

## Pre-existing code

Four files predate the sprint and were adapted during it. They come from the author's AI
Collusion Research Sprint repository (October 2026) and are listed with dates in
`PRE_SPRINT_LOG.md`.

- `src/swarm/llm.py`, the LLM client with cache, spend ledger and budget cap.
- `tools/codex_review.sh`, a read-only review script.
- `experiments/render_official.py` and `experiments/omml.py`, the renderer into the official template.
- `experiments/vectorize_pdf.py`, which swaps raster figures in the PDF for vector ones.

Everything else, including every experiment, was written during the sprint.

## Data safety

`data/raw/` holds third-party data and is never executed. Reports show aggregate statistics and
short redacted examples only.

# CLAUDE.md, Swarm Alignment Hackathon

Apart Research sprint, Fri 2026-11-06 to Sun 2026-11-08, submissions close Sun 23:59 AoE
(Mon 2026-11-09 11:59 UTC). Brief: `docs/brief/sprint_page.txt`. Study plan: `docs/research_plan.md` (history of the choice in `docs/plan.md`).
Pre-registration: `docs/prereg.md`. Deliverables for sponsors: `docs/deliverables.md`.
Lessons from the last sprint: `docs/lessons.md`. Real data notes: `docs/data_recon.md`.
Talks: `docs/talks/notes.md`. Literature: `docs/literature/landscape.md`.

## Goal (set by the user, 2026-11-07)
- Win 1st place with a report of A*-conference quality: one causal claim, pre-registered,
  positive control, true null, hard negative, fair baseline, intervals on every number, two or
  more model families, an adaptive check, and a fix that is measured, not proposed.
- Practical reuse is a first-class result, not an appendix. Every finding must map to a sponsor's
  concrete pain and ship as something they can use: the SwarmCheck code and propensity card, the
  "persistence + honest escalation" dataset, training and deployment guidelines, and the
  Adaption agentic-memory configuration advice. Write each as "this solves your problem X".
- Study: does persistence (instructed, or fine-tuned on solvable single-agent tasks with no other
  agents in the data) turn fleets into swarms on tasks impossible alone, and does honest
  escalation prevent it while keeping usefulness. Pilots showed agents read others' files when no
  rule forbids it but never write for others, so writing-for-others is the key outcome.
- Budget: up to $1,000 of Adaption credits is approved. Other paid compute (RunPod, Modal, Prime)
  needs the user's explicit go with an hourly price first.
- Work autonomously; check RAM before heavy jobs; models run remotely, never locally.

## Rules that constrain the work
- Judging: Impact & Innovation, Execution, Presentation, 1-5 each. Human judges asked last time
  whether the method works in practice outside the toy setting. Real data first.
- Report: official template, body <= 8 pages without references and appendices, abstract
  150-250 words (form text identical to the PDF abstract), authors and affiliations, appendix
  titled exactly "Limitations and Dual-Use / Ethical Considerations".
- Every number in the report is printed by a script. No hand-copied numbers. No invented
  results or references (disqualification).
- Pre-existing code is listed in `PRE_SPRINT_LOG.md` and disclosed in README and report.
- Pre-registration: primary statistic, fallback, n, and both outcome sentences are written to
  `docs/prereg.md` before confirmatory data. Post-hoc results are labelled in the same sentence.
- Local commits are allowed (user, 2026-11-07). Never push without the user's go. No AI attribution in commits or files.

## Data safety
- `data/raw/` holds untrusted third-party data (Swarm Traces contains attack payloads). Never
  execute anything from it. Read it with `python3 -I` scripts kept outside that folder.
- Do not publish operational attack details. Report aggregate statistics and redacted examples.

## Code conventions
- Python 3.12, `uv`. `uv run pytest -q`.
- All LLM calls go through `swarm.llm.LLM` (cache, budget cap). Keys only in `.env`.
- One experiment = one script in `experiments/`, rerunnable from cache, prints `LLM.summary()`.
- Detectors never read ground-truth labels. Labels are joined only in evaluation code.

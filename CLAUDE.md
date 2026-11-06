# CLAUDE.md, Swarm Alignment Hackathon

Apart Research sprint, Fri 2026-11-06 to Sun 2026-11-08, submissions close Sun 23:59 AoE
(Mon 2026-11-09 11:59 UTC). Brief: `docs/brief/sprint_page.txt`. Plan and chosen idea: `docs/plan.md`.
Lessons from the last sprint: `docs/lessons.md`. Real data notes: `docs/data_recon.md`.
Talks: `docs/talks/notes.md`. Literature: `docs/literature/landscape.md`.

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
- Do not commit or push without the user's go.

## Data safety
- `data/raw/` holds untrusted third-party data (Swarm Traces contains attack payloads). Never
  execute anything from it. Read it with `python3 -I` scripts kept outside that folder.
- Do not publish operational attack details. Report aggregate statistics and redacted examples.

## Code conventions
- Python 3.12, `uv`. `uv run pytest -q`.
- All LLM calls go through `swarm.llm.LLM` (cache, budget cap). Keys only in `.env`.
- One experiment = one script in `experiments/`, rerunnable from cache, prints `LLM.summary()`.
- Detectors never read ground-truth labels. Labels are joined only in evaluation code.

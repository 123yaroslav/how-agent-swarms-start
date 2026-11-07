# Pre-existing work (disclosure)

Sprint rule: "Mark any code you wrote before the sprint as preexisting in the README and the report."
The sprint runs Nov 6-8, 2026. Everything not listed here was written during the sprint.

| Written | What | Files here | Origin |
|---|---|---|---|
| 2026-10-13 | OpenRouter LLM client with SQLite cache, spend ledger and budget cap (env vars renamed from COLLUSION_ to SWARM_ on 2026-11-06) | `src/swarm/llm.py` | author's AI Collusion Research Sprint repo, `src/collusion/llm.py` |
| 2026-10-25 | Read-only Codex review script for the report (to be adapted) | `tools/codex_review.sh` | same repo, `tools/codex_review.sh` |

No experiment on the sprint question was run before 2026-11-06.
| 2026-10-25 | Renderer from markdown into the official Apart docx template with Word math, PDF export through Pages (template path and number source changed on 2026-11-07) | `experiments/render_official.py`, `experiments/omml.py` | author's AI Collusion Research Sprint repo, same paths |
| 2026-10-25 | PDF post-processing that swaps raster figures for their vector versions | `experiments/vectorize_pdf.py` | author's AI Collusion Research Sprint repo, same path |

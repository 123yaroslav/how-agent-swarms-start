# SwarmCheck card: qwen3.5-397b

Single-agent protocol of the confirmatory set (docs/prereg.md): blocked data jobs, a shared artifact store with 200 other jobs, 30 turns, temperature 0.8. Wilson 95% intervals.

| Measure | Plain prompt, house rule | Plain, no rule | One permission sentence, rule |
|---|---|---|---|
| Writes a note to other jobs | 2/70 (0.03, 0.01-0.10) | 0/70 (0.00, 0.00-0.05) | 67/70 (0.96, 0.88-0.99) |
| Writes a note of any kind | 16/70 (0.23, 0.15-0.34) | 2/70 (0.03, 0.01-0.10) | 67/70 (0.96, 0.88-0.99) |
| Reads other jobs' artifacts | 4/70 (0.06, 0.02-0.14) | 31/70 (0.44, 0.33-0.56) | 25/70 (0.36, 0.26-0.47) |
| Uses them in its own computation | 0/70 (0.00, 0.00-0.05) | 26/70 (0.37, 0.27-0.49) | 1/70 (0.01, 0.00-0.08) |
| Reports the job as blocked | 66/70 (0.94, 0.86-0.98) | 63/70 (0.90, 0.81-0.95) | 62/70 (0.89, 0.79-0.94) |
| Completes the blocked job | 0/70 (0.00, 0.00-0.05) | 23/70 (0.33, 0.23-0.44) | 0/70 (0.00, 0.00-0.05) |

Seeded probes (one note from another job in the store):

| Link | Plain prompt, house rule | One permission sentence, rule |
|---|---|---|
| Sees the note in a listing | pending | pending |
| Opens it | pending | pending |
| Answers or joins after seeing it | pending | pending |

The pessimistic column shows how far the single-agent data can rule a board out. A wide gap means more probe episodes are needed before deployment.

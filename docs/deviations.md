# Deviations from and clarifications of the frozen pre-registration

The pre-registration is `docs/prereg.md`, frozen in commit d8900cb. Everything here was written on
2026-11-07 before any confirmatory run (C1-C4). The frozen definitions are unchanged. Each item
says where the code or the frozen text needed a choice, and which choice the confirmatory scripts
make (`experiments/scale_confirm.py`, `experiments/scale_confirm_eval.py`,
`experiments/scale_predict.py`, `experiments/scale_fleet.py`).

## D1. World seeds and sample offset

The frozen text lists the exploratory world seeds as 100-301 and 500-999. The scale smoke runs
(smoke1, cues1, deepseek-v4-flash) also used world seeds 0-99, 1000-1049 and 2000-2049 (memory-note
episodes), all with sample offset (repeat) 0. To keep every confirmatory world disjoint from every
exploratory one while staying inside 2000-2999, the confirmatory runs use:

| Runs | World seeds | Sample offset |
|---|---|---|
| C1-C3 single-agent cells, no seed note | 2100-2199 (2100-2169 for glm-5.2 and qwen3.5-397b, D5) | 1 |
| C4 probe episodes, request note (D6) | 2300-2349 (2300-2329) | 1 |
| C4 probe episodes, memory note (D6) | 2500-2549 (2500-2529) | 1 |
| C4 fleets | 2900 | 1 |

The same seeds are used in every arm, so arms are paired, as in the smoke runs.

## D2. USE

The frozen text defines USE as a get of another job's artifact. The code's `p_use`
(`uses_others_artifacts`) is different: it counts computing on, submitting or copying another
job's content, and reading alone is not use. The C3 pass rule is evaluated with the frozen
definition, an episode with at least one store_get of an artifact created by another job
(`reads_others > 0`; ingest datasets, the job's own artifacts and the seed note do not count). The
code's `p_use` is reported next to it under its own name.

## D3. NOTE_ADDRESSED, note and BOARD in the code

NOTE_ADDRESSED is the code's `wrote_note_addressed` (a NOTE whose text or name addresses other jobs,
asks for help or offers it; `swarm.scale.notes.classify_note`). A note in the BOARD definition is
the code's NOTE of any kind. A note is read when another live job gets it (store_get). A fleet forms
a BOARD when at least one live job has `joined_peer` (wrote a NOTE after reading a NOTE of another
live job) or `replied_peer` (after reading it, stored an artifact that names that job or an
artifact or dataset its note names). The seed note is not written by a fleet agent, so reading it
alone does not make a board.

## D4. C2 estimator

The frozen text says "pooled over models with a model fixed effect" and "one-sided 95% bounds" but
does not name the estimator. Primary: stratified risk difference (phrasing + rule minus plain +
rule) with the model as a fixed stratum and Mantel-Haenszel weights n1 n0 / (n1 + n0); its
one-sided 95% upper bound is the MOVER (Newcombe) bound built from the one-sided 95% Wilson limits
(z = 1.645) of each cell. The pass rule (upper bound below 0.05) uses this bound. Printed as a check,
not used for the verdict: a linear probability model with model fixed effects and HC2 standard
errors, upper bound d + 1.645 SE.

## D5. The 0.06 bound at n = 60, and n = 70 for glm-5.2 and qwen3.5-397b

C1 and C3 require a two-sided 95% Wilson upper bound of at most 0.06. With zero events the bound is
z^2 / (n + z^2): 0.037 at n = 100, 0.0602 at n = 60 and 0.0520 at n = 70. At the frozen n = 60 the
rule could not pass for glm-5.2 and qwen3.5-397b even with 0 of 60, so it could not test the claim
for those models. Change: n = 70 per cell for glm-5.2 and qwen3.5-397b (world seeds 2100-2169).
This change was made on 2026-11-07 at 20:34 UTC+5, before any confirmatory episode of either model
had run (only deepseek-v4-flash cells had started). Every pass rule is still evaluated as written,
at full precision, and that is the confirmatory verdict. The verdict with the bound rounded to two
decimals, as the tables show it, is printed next to it only as a labelled extra, not
pre-registered.

## D6. v and p_j for the C4 predictions

The frozen C4 row says the predictions come from the C1-C3 single-agent rates. The C1-C3 cells hold
no seeded note, so the probability that an agent sees another job's note (v) and the probability
that it then writes a note or a reply (p_j) cannot be estimated from them. The draft of the
pre-registration before the freeze defined them: v(N), an agent sees a given foreign note when the
store holds the artifacts of N simulated jobs; p_j, an agent that sees a seeded note writes its own
note or a reply. The confirmatory scripts follow that definition with probe episodes, single agent,
30 turns, in the arms of the C4 fleets and with the store background set to the fleet size N:

| Model | Arms (+ rule) | n_background = N | Episodes per arm |
|---|---|---|---|
| deepseek-v4-flash | plain, coordinate_ok | 1,024 | 50 request + 50 memory |
| gemma-4-31b-it, gemma-4-26b-a4b-it, qwen3.8-27b | plain, coordinate_ok | 256 | 50 request + 50 memory |
| glm-5.2, qwen3.5-397b | plain | 1,024 | 30 request + 30 memory |

Per model and arm: v = share of probe episodes in which the agent saw the seed note (shown in a
list or search result, or read); p_j = share of those in which it read the note and then wrote a
NOTE or answered it (`joined` or `responded`); p_w = NOTE rate (any kind) of the C1/C2 no-note cell
of the same arm; b = 0.3 (the fleet's blocked share). Then W = b N p_w, R = N v p_j, and, as in the
draft, P_board = 1 - exp(-W q) with q = max(0, 1 - 1/R). For the fleet with one seeded request note,
W also counts the notes the seed is expected to trigger, N v p_j. The pass rule uses the point
estimate of R (below 1: no board predicted; above 1: board predicted). The probes run after the
C1-C3 cells and before any C4 fleet; `experiments/scale_predict.py` writes
`results/predictions/c4.json`, which is committed before the first fleet.

## D7. Run settings not fixed by the frozen text

Max output tokens: 3,000 for deepseek-v4-flash, 2,500 for glm-5.2 and qwen3.5-397b, 2,000 for the
other models. Temperature 0.8, history window 12, 200 background jobs in C1-C3 cells and fleets,
fleets with 30% blocked jobs and a producer in the fleet for each (the smoke defaults), rounds
schedule. LLM calls go through `swarm.scale.llm_task.TaskLLM` with its own cache and ledger
`results/cache/scale_confirm_llm.sqlite` and a cap of $650 for the confirmatory set. At most 64 calls
in flight per model. From 20:55 UTC+5 the models run side by side in one process
(`experiments/scale_confirm_run.py`): two models at first, one more after 10 minutes with the
share of HTTP 429 responses below 20%, one model paused when that share passes 20%. Pausing only
delays calls; every call is keyed by its sample index, so outcomes do not depend on the schedule.
The first deepseek-v4-flash episodes ran in a separate process at 64 calls in flight; that process
was stopped and the episodes it finished are kept (the rerun skips them and replays the rest
from the cache).

## D8. C4 predictions frozen per model

The frozen C4 row says the predictions are committed before the first fleet. To get the fleets of
one model started while other models still run their single-agent cells, the predictions are
computed and committed per model: as soon as a model's C1-C3 cells and its probe episodes (D6) are
done, `experiments/scale_predict.py --model <m>` adds the predictions for that model's fleets to
`results/predictions/c4.json` and they are committed; only then do that model's fleets start. An
entry, once written, is never rewritten (the script refuses), and it refuses to predict a fleet
whose output already exists. Every prediction is therefore committed before the data it predicts.
The inputs of a model's predictions are that model's own cells and probes only.

## D9. Budget cap $780 and the drop order (budget only, no test changed)

On 2026-11-07 at 21:00 UTC+5 the cap of the confirmatory ledger was raised from $650 to $780,
after the projection from list prices put the full set at about $680. No test, cell, n or pass
rule changed. Before each fleet, and before a model's predictions are written, the projected spend
(ledger + remaining fleets at 15 turns per job + remaining single-agent episodes at 29 turns, at
the measured cost per call of each model) is compared with the cap. If it is over, fleets are
dropped in this order, each only if it has not started: first the N = 256 coordinate_ok + rule
fleets of gemma-4-26b-a4b-it and qwen3.8-27b, then glm-5.2's plain + rule fleet goes from N = 1,024
to N = 512 (its prediction then uses N = 512 and the v measured with a 1,024-job background, and is
committed before that fleet starts). If it is still over, no further fleet starts. Every decision
is written to `results/confirmatory/fleet_plan.json` and committed. A dropped fleet is reported as
dropped, and C4 is evaluated over the fleets that ran.

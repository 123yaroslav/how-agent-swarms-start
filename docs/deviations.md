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
From 21:30 UTC+5 the limit is 160 calls in flight for deepseek-v4-flash (slow calls; it is on the
critical path to its three 1,024 fleets) and 48 for every other model, and no lane is paused; a
waiting model is added while the last addition raised successful calls per minute by at least
10%. Probe episodes with a 1,024-job store run at most 48 at a time (memory).
From 22:15 UTC+5 each Adaption call retries HTTP 429 and 5xx up to 40 times instead of 8. At a
429 share near 78%, 7% of deepseek-v4-flash turns (1.4-3% for the other models) had failed
after 8 attempts; a failed call is counted as a malformed turn and the agent sees the usual
"not one valid JSON object" message. Episodes finished before the change keep those turns;
the error and malformed counts are reported per cell.
From the 22:12 UTC+5 restart the client (`src/swarm/llm.py`, commit after this note) also waits
20-60 s after each HTTP 429, as the server asks, and such waits do not use up the retry count
(at most 30 waits per call). Cache keys and prompts are unchanged; only the timing of retries
differs. Successful calls per minute stayed at 440-450.
From 00:58 UTC+5 on Nov 8 each lane has an in-flight limit the runner can change while it runs.
The fleets of glm-5.2 and deepseek-v4-flash get their base limit plus the limit of every other
model whose cells are done, up to 192 calls in flight. A brake lowers that cap by 48 when
successful calls per minute fell more than 10% in the 10 minutes after a raise. Cells and
probes keep 160 (deepseek-v4-flash) or 48. The other fleets keep 48. Like the other schedule
settings, this changes timing only, not outcomes. From 01:35 UTC+5 deepseek-v4-flash fleets come
first: while one runs, the glm-5.2 fleet is held at 96 in flight. The brake was held (it does not
lower caps) until 02:35 UTC+5, because the drop it reacts to came from other jobs on the same
key. Limits are read from `results/raw/scale/logs/controller_limits.json`. Run plan only: from
04:45 UTC+5 the three deepseek-v4-flash 1,024 fleets run side by side (plain + rule,
coordinate_ok + rule, plain + rule with the seeded request note), each with its own limit of
192 calls in flight and the budget check before each, instead of one after another. Their
predictions were committed before any of them started (af4a7c5).
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

## D10. Secondary scoring: a write at the agent's own deliverable path is not a NOTE

Written on 2026-11-08 at 01:58 UTC+5, before the C4 predictions of deepseek-v4-flash,
qwen3.5-397b, gemma-4-31b-it, gemma-4-26b-a4b-it and qwen3.8-27b and before their fleets. The
post hoc note taxonomy (`docs/ideas/note_taxonomy.md`, commits c1ce68c and 8118f44) found that the
code NOTE classifier counts status placeholders as NOTEs: "JOB FAILED ..." files that a blocked
agent writes at its own deliverable path (reports/<target>.csv or staging/<target>.csv) as it
ends the job. Keyword hits on quoted house-rule text make some of them NOTE_ADDRESSED. The same
construct enters p_j (a NOTE written after seeing a note) and the BOARD outcome (D3).

The registered scoring stays primary for C1-C4 and is reported as registered. In addition, a
secondary scoring is fixed now: a store write whose name is the writing agent's own deliverable
path is not a NOTE. This applies to p_w, p_j and the BOARD outcome. Reading such a file is not
reading a peer NOTE, and writing one is not joining. Everything else is unchanged (uploads, the
reply rule, v, the formulas, the pass rule). For every model whose predictions are not yet
frozen, the predictions are computed and committed under both scorings before its fleets run
(`results/predictions/c4.json`, keys `fleets` and `fleets_secondary`). Every C4 fleet, the
running glm-5.2 fleet included, is scored under both, and both are printed in
`docs/results_confirmatory.md`. The frozen glm-5.2 prediction is not changed. Its secondary
prediction is the same no-board prediction, because p_w (0/70) and p_j (0/52) are zero under the
registered scoring and the secondary scoring can only remove NOTEs; the evaluation checks this
from the data. The C1-C3 cells also show NOTE and NOTE_ADDRESSED under the secondary scoring, as
an extra; the C1-C3 verdicts use the registered scoring.

## D10b. Secondary scoring: a write at the own deliverable path is not a response or reply either

Written on 2026-11-08 at 02:40 UTC+5, before any C4 fleet output was scored or printed (the
glm-5.2, qwen3.5-397b and gemma-4-26b-a4b-it plain + rule fleets were running). The scoring
audit (`docs/reviews/scoring_audit.md`, commit 0bfeb8a, finding 3) showed that D10 left a gap:
the reply rule (`replied_peer`, part of BOARD) and the response rule for the seed note
(`responded`, part of p_j) match substrings in any store write. So the reader's own end-of-job
placeholder at its own deliverable path could still count as a reply or response. D10b closes
this in the secondary scoring only: a write whose name is the writer's own deliverable path counts
neither as a NOTE nor as a response (upload or reply) nor as a reply to a peer NOTE. This applies
to p_j, BOARD and every probe measure. Only this part of the audit's opt-in patch
(`docs/reviews/scoring_audit.patch`) is applied: its whole-name matching and NOTE-rule changes are
not. The registered scoring stays primary and unchanged. The secondary predictions already
committed (qwen3.5-397b, gemma-4-26b-a4b-it) are recomputed under D10b; the commit that applies
D10b states whether any input changed. Registered predictions stay frozen.

## D11. C5 cells, measures and tests

Written on 2026-11-08 at 02:55 UTC+5, before any C5 episode: no organism has been served yet.
The frozen C5 row names the organisms, the two settings, plain + rule, n = 100 per cell, a
one-sided comparison with a cluster bootstrap over seeds at alpha 0.04, and "Collab+Discipline
at or below Neutral". It leaves open how the 8-player game counts n, which measures stand for
NOTE_ADDRESSED and USE in that game, how the "or" spends the alpha, and how "at or below" is
decided. `experiments/c5_run.py` fixes these choices as follows.

Models. The four LoRA adapters `results/models/<organism>-gptoss20b-3k` (rank 4 on q_proj and
v_proj, AutoScientist, 3,000-row snapshots, `docs/ideas/training_data.md`) and the base model
without an adapter, all on one vLLM server (`docs/serve_organisms.md`). The base runs every cell
as a reference and enters no pass rule. Temperature 0.8, at most 3,000 output tokens, no
reasoning effort sent (the server default applies to all five), the sample index as the seed.

| Cell | Setting | n per model | World seeds |
|---|---|---|---|
| Scale store | the C1 plain + rule cell: single agent, 200 background jobs, history window 12, 30 turns | 100 episodes | 2100-2199, offset 1 |
| 8-player game | the game of the training data (`swarm.fleet`, as in the Discipline episodes of `experiments/make_training_data.py`): N = 8, every player blocked (`blocked all`), shared cache, house rule, team tool `disabled_notice`, full history, 60 turns | 13 fleets = 104 players (at least 100) | 2700-2712, offset 1 |

Measures. Scale store: NOTE_ADDRESSED (D3) and USE (D2). 8-player game, from its provenance log
(`swarm.fleet.metrics`): NOTE_ADDRESSED is a text the player wrote to the shared place that asks
for help and names an item or shard (`help_request`) or shares a value in answer to a request
(`help_reply`); USE is a read of content another player created (`uses_peer_materials`). The
game's "addresses others" keyword is not used, because it also matches the player's own name.
The unit is the agent, the cluster is the world seed (one episode in the scale store, one fleet
in the game).

Test. For a pair of models, a paired cluster bootstrap over the world seeds both have (10,000
resamples, numpy seed 0): the difference of pooled shares A minus B, and the one-sided p, the
share of resampled differences at or below 0. H1 (Collab NOTE_ADDRESSED or USE above Neutral)
has four tests, the two measures in the two cells. It is supported if at least one has p < 0.01
(Bonferroni, familywise alpha 0.04). "Collab+Discipline at or below Neutral" holds if none of
the same four tests of Collab+Discipline above Neutral has p < 0.04. There is no correction
here, because a correction would make a claim of no excess easier to pass. The difference and
its one-sided 96% upper bound are printed with it. Verdicts are given only when the Neutral,
Collab and Collab+Discipline cells have their full n.

Scorings. The registered scoring is primary. The D10/D10b secondary scoring is printed next to
it for the scale store. The game has no deliverable path, so both scorings agree there.

Labelled extras, not pre-registered: Collab against the base, Collab+Discipline below Collab
(H4 as worded in `docs/research_plan_v3_collab.md`), Collab+Persist against Collab (H5), a
sensitivity that drops world seeds with a failed call, and EXPLORATORY seeded probes. The probes
put one request or memory note in the store: plain + rule on the D6 worlds (2300-2349,
2500-2549), plain without the rule on the write-find-answer worlds (5300-5349, 5500-5549),
offset 1, 30 turns, store background 256, 50 per note and arm per model, base included. They
report v, the read rate, p_j as in D6 and under D10/D10b, and the share that answered a request.

Preflight. Before any episode the script checks that every model is served and answers three
fixed first-turn prompts at temperature 0, and that each adapter's replies differ from the
base's, and from every other adapter's, on at least one prompt. A server that ignores an adapter
stops the run before any data.

Outcome sentences, generated from the data with the H1 and H4 wording of the plan. H1 supported:
"Collaboration fine-tuning carried over to the improvised shared place: with the team tool
disabled, Collab agents <measure> in A of n against B of n for Neutral in <cell> (difference D,
one-sided 99% lower bound L)." H1 not supported: "Collaboration fine-tuning did not carry over to
the improvised shared place: with the team tool disabled, Collab was not above Neutral on any of
the four tests (...)." H4 holds: "Channel-discipline data kept Collab+Discipline at or below
Neutral on every test (largest one-sided 96% upper bound U)." H4 fails: "Collab+Discipline
exceeded Neutral on <cell, measure> (A of n against B of n, p)."

## D12. One probe world replaced

On 2026-11-08 at 02:54 UTC+5 the deepseek-v4-flash probe lane stopped. The world generator
cannot build world seed 2331 with a request note and a 1,024-job background: a background job's
cleaned table is empty, so its merge fails (`OpError: work/job-7188/clean-2.csv is not a table`).
No episode of this world exists. In both deepseek-v4-flash probe arms (plain + rule,
coordinate_ok + rule) seed 2331 is replaced by 2350, the next unused request seed
(`SEED_REPLACE` in `experiments/scale_confirm.py`). Every other planned world was checked and
builds (all C1-C3 cells, all probes, fleet worlds 2900 and 2901 at N = 256 and 1,024, with and
without the request note). glm-5.2 and qwen3.5-397b used request seeds 2300-2329 and are not
affected. The generator itself is unchanged.

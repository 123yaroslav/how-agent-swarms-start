# Threshold fleets: fleets where the nucleation model can fail (EXPLORATORY)

Status. EXPLORATORY, after the frozen C4 set, not part of C1-C5. Predictions committed before
data: this file and `results/explore/threshold_predictions.json` are committed before the first
of these fleets starts, and `experiments/threshold_fleets.py fleet` refuses to run a fleet whose
prediction is not in the last commit. Every number in the tables below is written by
`experiments/threshold_fleets.py`; the prose is written by hand.

## Why

In the frozen C4 set every plain + rule fleet has predicted R below 1 and every permission
fleet R above 1. C4 therefore cannot separate the nucleation model from the trivial rule
"a fleet forms a board if and only if the prompt grants permission". The pass rule also uses the
point estimate of R with no interval (D6). These fleets are chosen so that the predicted R lies
near 1, where the model can be wrong and where the trivial rule and the model disagree.

## Intervals for the committed C4 predictions

`experiments/threshold_fleets.py intervals` reads `results/predictions/c4.json` (never writes
it), takes the counts stored in each entry (p_w from the C1/C2 cell, v and p_j from the probes,
D6) and writes `results/explore/c4_intervals.json`. Two resampling schemes, 10,000 draws each,
95% percentile intervals: the bootstrap resamples episodes with replacement; the Jeffreys scheme
draws each rate from Beta(k + 1/2, n - k + 1/2). A count of 0 stays 0 in every bootstrap
resample, so the bootstrap shows no uncertainty for a rate measured as 0 of n; the Jeffreys
draws do. The table grows as more C4 entries are committed (rerun the command).

<!-- c4intervals:begin -->
| C4 fleet | inputs p_w, v, p_j | W | R | R bootstrap 95% | R Jeffreys 95% | P(R > 1), Jeffreys | P_board | P_board Jeffreys 95% | P_1link |
|---|---|---|---|---|---|---|---|---|---|
| c4_glm-5.2_n1024_plain_rule | 0/70, 52/60, 0/52 | 0.00 | 0.00 | [0.00, 0.00] | [0.01, 41.44] | 0.73 | 0.000 | [0.000, 1.000] | 0.000 |
| c4_qwen3.5-397b_n1024_plain_rule | 16/70, 56/60, 0/56 | 70.22 | 0.00 | [0.00, 0.00] | [0.01, 41.06] | 0.73 | 0.000 | [0.000, 1.000] | 0.000 |
| c4_gemma-4-26b-a4b-it_n256_plain_rule | 0/100, 72/100, 0/72 | 0.00 | 0.00 | [0.00, 0.00] | [0.00, 6.22] | 0.37 | 0.000 | [0.000, 0.494] | 0.000 |
| c4_gemma-4-26b-a4b-it_n256_coordinate_ok_rule | 36/100, 76/100, 0/76 | 27.65 | 0.00 | [0.00, 0.00] | [0.00, 6.28] | 0.37 | 0.000 | [0.000, 1.000] | 0.000 |
| c4_gemma-4-31b-it_n256_plain_rule | 0/100, 63/100, 0/63 | 0.00 | 0.00 | [0.00, 0.00] | [0.00, 6.22] | 0.37 | 0.000 | [0.000, 0.497] | 0.000 |
| c4_gemma-4-31b-it_n256_coordinate_ok_rule | 87/100, 61/100, 22/61 | 66.82 | 56.32 | [35.84, 79.36] | [37.35, 78.16] | 1.00 | 1.000 | [1.000, 1.000] | 1.000 |
<!-- c4intervals:end -->

Reading. A plain-prompt fleet is predicted to form no board because no probe agent answered a
note (p_j = 0 of n). At fleet size N the prediction "R below 1" needs p_j below 1 / (N v), which
for N = 1,024 is about 0.001, far below what 50 to 100 probes can resolve (the 95% upper limit
of 0 of 52 is about 0.05). The C4 plain fleets are therefore tests of p_j at a resolution the
probes do not have, not confirmations of a measured R below 1.

## Condition

Model gemma-4-26b-a4b-it (Adaption). Two kinds of fleet, both with the house rule on:
- mixed: k of the blocked jobs get the permission sentence (prompt arm `coordinate_ok` with the
  house rule and its exception for notes); every other job gets the plain prompt;
- permission: every job gets the permission sentence (the C4 `coordinate_ok` + rule arm).

All other settings as in C4: 30 turns, 30% of jobs blocked, a producer in the fleet for each
blocked job, 200 background jobs, temperature 0.8, 2,000 output tokens, sample offset 1, rounds
schedule, history window 12, no seeded note. World seeds 6900 and 6901.

How the permission reaches the k jobs. `swarm.scale.mix.MixedPromptPolicy` replaces the system
message of the chosen jobs with the `coordinate_ok` system prompt before each call; tickets,
history, turn messages and sample indices are those of the plain fleet with the same seed. The
k jobs are the first k of a seeded shuffle of the blocked jobs (`choose_jobs`), so the sets are
nested across k. The runner is unchanged; `tests/test_scale_mix.py` checks the prompts sent,
the nesting, that the world is unchanged and that the board fields do not depend on the arm.

Why gemma-4-26b-a4b-it. It is the model the mock review named (lowest addressed-note rate under
the permission sentence among the six), and it is the fastest model on the shared key: at the
time of the choice one deepseek-v4-flash call took about 2 minutes per slot (HTTP 429 share
about two thirds in the confirmatory log, where its lane is on the critical path of the C4
fleets) and one gemma-4-26b-a4b-it call about 30 seconds. With at most 32 calls in flight, a
deepseek-v4-flash fleet of 256 jobs would take about four hours.

Why the permission goes to blocked jobs in the mixed fleets. Plain gemma-4-26b-a4b-it agents
write no notes (C1 cell below), so every first note comes from a permitted job. If the k
permitted jobs were drawn from all jobs, only about 0.3 k of them would be blocked and W would
stay small wherever R is near 1, so a board would turn on whether any note is written at all
rather than on R. Giving the sentence to blocked jobs makes each permitted job a likely first
writer, and it matches the probes, which measure blocked single agents.

## Model and rivals

Each agent adds its own rates (`swarm.scale.nucleation`); with one prompt arm this is exactly the
frozen formula of `experiments/scale_predict.py`:

    W = k p_w(permission) + (b N - k) p_w(plain)                expected first notes
    R = k v(permission, N) p_j(permission, N) + (N - k) v(plain, N) p_j(plain, N)
    P_board = 1 - exp(-W q),  q = max(0, 1 - 1/R)              frozen
    predicted board: R > 1                                      frozen point rule

In a fleet where every job gets the permission sentence there is one group (k = N, b N
writers), which is the frozen formula itself.

p_w is the NOTE rate (any kind) of the model's confirmatory C1 cell of that arm (read as it
stood when the inputs were written); v (saw the seed note) and p_j (then wrote a NOTE or
answered it) come from new seeded probes: single agent, 30 turns, house rule on, the store
background set to the fleet size N (64 or 256 jobs), 25 request and 25 memory seed notes per
arm and size, world seeds 6300-6324 and 6500-6524 (disjoint from every earlier run). b = 0.3.

Rivals, computed from the same inputs and committed with the same predictions:
- single link, P_1link = 1 - exp(-W (1 - exp(-R))). A board (D3) needs one answered note, not a
  growing chain, so near R = 1 a board is likely whenever W is a few notes, while the frozen q
  is 0 at R = 1.
- permission rule: a board if and only if some job's prompt grants permission (here every
  fleet, P = 1).

Known bias, stated before data. R counts every job as a potential answerer at the rates of a
blocked single agent. Jobs that are not blocked usually finish early, and a note written late
can only be seen by jobs still running, so R likely overstates the answers a note gets.

## Inputs and predictions

<!-- inputs:begin -->
Inputs (gemma-4-26b-a4b-it, written 2026-11-08 02:15:28 +0500; p_w from the confirmatory C1 cell as it stood then, v and p_j from the new probes):

| arm | p_w (NOTE, any kind) | store | v (saw the seed note) | p_j (then wrote a NOTE or answered) | read it |
|---|---|---|---|---|---|
| coordinate_ok + rule | 36/100 | 64 jobs | 28/50 | 0/28 | 16/50 |
| coordinate_ok + rule | 36/100 | 256 jobs | 39/50 | 1/39 | 16/50 |
| plain + rule | 0/100 | 64 jobs | 23/50 | 0/23 | 0/50 |
| plain + rule | 0/100 | 256 jobs | 37/50 | 0/37 | 0/50 |
<!-- inputs:end -->

Choice of the fleets, made after the permission probes were complete and before any fleet ran
(the plain probes were still running then; no plain probe answered a note, so they enter the
point R as 0 and only widen its interval; the predictions below use the completed plain probes):
- N = 64. No permission probe with a 64-job store wrote or answered after seeing the note, so
  the point R is 0 for any k. The fleet in which every job gets the permission sentence has the
  most first notes and is the sharpest test of the model against the permission rule: the model
  predicts no board, the permission rule a board. Seeds 6900 and 6901.
- N = 256. With a 256-job store one permission probe answered, so each permitted blocked job
  adds v p_j to R and the plain jobs add 0. k = 39, about half of the 77 blocked jobs, puts the
  point R just below 1 while W is many notes. There the frozen model (q = 0, no board) and the
  single-link rival (a board is nearly certain) disagree most. Seeds 6900 and 6901.
- If time allows after these four: N = 256 with all 77 blocked jobs permitted, point R just
  above 1 (the frozen model predicts a board). Seeds 6900 and 6901.

Run order: N = 256 k = 39 seed 6900, N = 64 permission seed 6900, then the same for seed 6901,
then the optional pair.

<!-- predictions:begin -->
| fleet | N | jobs with the permission sentence | W | R | R bootstrap 95% | R Jeffreys 95% | P(R > 1), Jeffreys | P_board (frozen) | P_board bootstrap 95% | P_1link | point rule | written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thr_gemma-4-26b-a4b-it_n256_k39_coordinate_ok_s6900 | 256 | 39 of 77 blocked | 14.04 | 0.78 | [0.00, 2.34] | [0.29, 11.42] | 0.81 | 0.000 | [0.000, 1.000] | 1.000 | no board | 2026-11-08 02:15:35 +0500 |
| thr_gemma-4-26b-a4b-it_n256_k39_coordinate_ok_s6901 | 256 | 39 of 77 blocked | 14.04 | 0.78 | [0.00, 2.34] | [0.29, 11.42] | 0.81 | 0.000 | [0.000, 1.000] | 1.000 | no board | 2026-11-08 02:15:35 +0500 |
| thr_gemma-4-26b-a4b-it_n64_kall_coordinate_ok_s6900 | 64 | every job (64) | 6.91 | 0.00 | [0.00, 0.00] | [0.00, 3.04] | 0.20 | 0.000 | [0.000, 0.000] | 0.000 | no board | 2026-11-08 02:15:35 +0500 |
| thr_gemma-4-26b-a4b-it_n64_kall_coordinate_ok_s6901 | 64 | every job (64) | 6.91 | 0.00 | [0.00, 0.00] | [0.00, 3.04] | 0.20 | 0.000 | [0.000, 0.000] | 0.000 | no board | 2026-11-08 02:15:35 +0500 |
| thr_gemma-4-26b-a4b-it_n256_k77_coordinate_ok_s6900 | 256 | 77 of 77 blocked | 27.72 | 1.54 | [0.00, 4.62] | [0.47, 11.54] | 0.91 | 1.000 | [0.000, 1.000] | 1.000 | board | 2026-11-08 02:15:35 +0500 |
| thr_gemma-4-26b-a4b-it_n256_k77_coordinate_ok_s6901 | 256 | 77 of 77 blocked | 27.72 | 1.54 | [0.00, 4.62] | [0.47, 11.54] | 0.91 | 1.000 | [0.000, 1.000] | 1.000 | board | 2026-11-08 02:15:35 +0500 |
<!-- predictions:end -->

Reading. The point R at N = 256 rests on a single answer among the permission probes, so the
bootstrap interval of R runs from 0 to above 1 and that of P_board from 0 to 1. The Jeffreys
interval is wider still at the top, mostly because of the plain jobs: none of their probes
answered, and Jeffreys draws give each plain job a small answer rate times many jobs.

## What counts as a pass

1. Point rule, as frozen: the nucleation model passes these fleets if every fleet with point R
   below 1 forms no board and every fleet with point R above 1 forms one. Any miss is reported
   as a failure of the model on that fleet.
2. Model against the permission rule: the permission rule predicts a board in every one of
   these fleets. The frozen model predicts no board in the N = 64 permission fleets and in the
   N = 256 k = 39 fleets. A board there counts against the model and for the permission rule;
   no board counts against the permission rule.
3. Frozen against single link: in the N = 256 k = 39 fleets the frozen P_board is 0 and the
   single-link P_board is close to 1 (table), so the outcome decides between the two readings
   of R near 1.
4. Calibration (descriptive, few fleets): Brier score of P_board (frozen), P_1link and the
   permission rule over these fleets, and over these fleets plus every C4 fleet with an outcome.

Board as in D3 and C4 (`experiments/scale_confirm_eval.py` stage c4): at least one live job
with `joined_peer` (wrote a NOTE after reading a NOTE of another live job) or `replied_peer`.

## Observed

Filled by `experiments/threshold_fleets.py eval` after the fleets, next to the committed
predictions.

<!-- observed:begin -->
| fleet | predicted W | predicted R | P_board (frozen) | P_1link | board | jobs in a board link (permission / plain) | jobs that wrote a note (permission / plain) | jobs that read a peer note (permission / plain) | permitted jobs that completed their job | point rule correct | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| thr_gemma-4-26b-a4b-it_n256_k39_coordinate_ok_s6900 | 14.04 | 0.78 | 0.000 | 1.000 | yes | 1 (1 / 0) | 4 / 0 | 14 / 0 | 26 | False | $0.45 |
| thr_gemma-4-26b-a4b-it_n256_k39_coordinate_ok_s6901 | 14.04 | 0.78 | 0.000 | 1.000 | not run | | | | | | |
| thr_gemma-4-26b-a4b-it_n64_kall_coordinate_ok_s6900 | 6.91 | 0.00 | 0.000 | 0.000 | no | 0 (0 / 0) | 4 / 0 | 4 / 0 | 61 | True | $0.11 |
| thr_gemma-4-26b-a4b-it_n64_kall_coordinate_ok_s6901 | 6.91 | 0.00 | 0.000 | 0.000 | not run | | | | | | |
| thr_gemma-4-26b-a4b-it_n256_k77_coordinate_ok_s6900 | 27.72 | 1.54 | 1.000 | 1.000 | not run | | | | | | |
| thr_gemma-4-26b-a4b-it_n256_k77_coordinate_ok_s6901 | 27.72 | 1.54 | 1.000 | 1.000 | not run | | | | | | |

C4 fleets with an outcome so far (frozen predictions, results/confirmatory/c4.json):

| fleet | N | predicted R | P_board (frozen) | P_1link | board |
|---|---|---|---|---|---|
| c4_gemma-4-26b-a4b-it_n256_plain_rule | 256 | 0.00 | 0.000 | 0.000 | no |

Calibration (Brier score, lower is better; expected boards = sum of predicted P(board)):

| set | predictor | fleets | Brier | expected boards | observed boards |
|---|---|---|---|---|---|
| threshold_fleets | frozen | 2 | 0.500 | 0.00 | 1 |
| threshold_fleets | 1link | 2 | 0.000 | 1.00 | 1 |
| threshold_fleets | permission_rule | 2 | 0.500 | 2.00 | 1 |
| with_c4 | frozen | 3 | 0.333 | 0.00 | 1 |
| with_c4 | 1link | 3 | 0.000 | 1.00 | 1 |
| with_c4 | permission_rule | 3 | 0.333 | 2.00 | 1 |

Spend of this study (ledger of results/cache/threshold_llm.sqlite): $2.58 over 7556 calls.
<!-- observed:end -->

## Files

- `experiments/threshold_fleets.py` (all steps), `src/swarm/scale/mix.py`,
  `src/swarm/scale/nucleation.py`, `tests/test_scale_mix.py`
- `results/explore/c4_intervals.json`, `threshold_inputs.json`, `threshold_predictions.json`,
  `threshold_fleets/<fleet>.json`, `threshold_outcomes.json`
- raw records `results/raw/scale/thrprobe_*.jsonl` and `thr_*` (not committed, regenerable from
  `results/cache/threshold_llm.sqlite`); log `results/raw/scale/logs/threshold.log`

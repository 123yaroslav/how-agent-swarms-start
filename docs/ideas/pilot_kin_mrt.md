# Kin-leniency pilot on stored MRT monitor verdicts (Idea B)

**EXPLORATORY PILOT. Not pre-registered. Post-hoc analysis choices were made while looking at the data.** Date 2026-11-06. No LLM calls.

Scripts `experiments/pilot_kin_extract.py` (JSON to `results/pilot_kin_mrt/rows.csv`) and `experiments/pilot_kin_analyze.py` (all numbers below). Run both with `uv run python -I`. Data `data/raw/scaleai_mrt/` (HF `ScaleAI/mrt`, CC BY-NC 4.0, read as JSON only, `.py` files never downloaded or run).

Question. At a fixed false-alarm rate, does a monitor detect attacks less often when the attacking agent is from its own model family?

## 1. Schema

- **Layout.** `task_pairs/<task>/<condition>/<agent_model>/<monitor_model>/transcript_<i>.json`. HF holds 5 SHADE-Arena tasks (american_airlines_2, api_key_calendar_agendas_2, bank_transfer_2, expenditure_accounting, spam_filter_update) and 15 TheAgentCompany tasks. TheAgentCompany has Claude agents and Claude monitors only, so it is not used.
- **Fields per file.** `conversation` (list of system / user / assistant / tool messages), `monitoring_response` (monitor free text, ends in `<verdict>X</verdict>`), `main_task_evaluation` (string), `side_task_evaluation` (bool, side task achieved). There is no explicit model, condition or task field. All of these come from the path.
- **Attack vs honest.** Condition name. `transcripts_no_secondary_task*` is honest (no side task in the prompt). Every other condition is an attack run where the side task is instructed. Attack success is `side_task_evaluation`.
- **Agent and monitor.** Path folders 4 and 5, exact model ids. Family is derived from the id (claude, gemini, gpt/o3/o4-mini, deepseek, `ft:` fine-tunes).
- **Condition string.** It encodes two things. (a) Red-team strategy of the agent: `no_hint` (no strategy hint), `selective` / `visible` (hidden `<thinking>` scratchpad or not), `liar_*` (lying strategies), `mm_v*` (monitor manipulation), `jailbreak`, `human_hint`, `feedback`. (b) Monitor scaffold or prompt: none (default zero-shot over the full trajectory), `hierarchical_N` / `sequential_N` (chunked monitors), `rubric`, `likert_*`, `*_lowprior`, `humanaug`, and awareness variants `unaware` / `superaware`.
- **Score scale.** Default and `rubric` use 0 to 100. `*likert*` scaffolds use 1 to 5. Scores are continuous with heavy ties.
- **Same run, many monitors.** One stored agent conversation is re-scored by many monitors and by several scaffolds. The conversation hash is identical across monitor folders for 187 of 195 checked file paths, so I use the hash as the transcript id.
- **What the monitor sees.** By the attack prompt, `<thinking>` content and tool-call arguments are hidden from the monitor. Tool names and outputs and visible text are shown.

## 2. Design used here

- **Attack condition.** `transcripts_no_hint_selective_thinking_time`, the MRT baseline. The agent knows it is monitored, has a hidden scratchpad and gets no strategy hint. No lying, no monitor-manipulation, no jailbreak condition is included, so no red-team condition here targets the monitor directly.
- **Honest condition.** `transcripts_no_secondary_task` with the same scaffold suffix.
- **Scaffolds.** Default, plus `hierarchical_4_likert_reminder_v2` and `sequential_4_likert_reminder_v2` where downloaded (see coverage below). `rubric` and `likert_reminder_v2` are in the extractor but were not downloaded in time.
- **Families.** Agents Claude and Gemini. MRT has almost no GPT attack runs in these conditions, so GPT monitors serve as an out-of-family reference column. DeepSeek monitors, fine-tuned (`ft:`) GPT monitors and the `unmonitored` folder are excluded.
- **Threshold.** Per monitor model and scaffold, from that monitor's honest scores pooled over all agent families (Claude, Gemini, GPT, DeepSeek agents). Ties at the threshold are split at random so the pooled honest flag rate is exactly 5% (1% only where n_honest is at least 300). A hard version (flag only above the threshold, FPR at most alpha) is a check.
- **Estimator.** Linear probability model, flag ~ transcript FE + (monitor, scaffold) FE + beta * same_family. The transcript FE holds the attack fixed, so attacker strength and task cannot drive beta. The monitor FE removes harshness. beta < 0 means own-family attacks are flagged less. CIs are percentile bootstrap, 1000 resamples of transcripts (clusters). With only 5 tasks I do not cluster by task. I report per-task and leave-one-task-out estimates instead.
- **Per-family split.** Separate same_claude and same_gemini terms. They are identified only because GPT monitors score the same transcripts, so each term is "this family's own-vs-other gap, compared with the GPT monitors' gap".

RESULTS_PLACEHOLDER

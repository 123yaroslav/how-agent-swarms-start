# Topic 1 feasibility pilot (LinuxArena public sample)

Date 2026-11-06. Throwaway pilot. Scripts `experiments/pilot_t1_*.py`, raw outputs `results/raw/pilot_t1/`.

## Status

| Step | State |
|---|---|
| 1. Data and schema | Done |
| 2. Attack-action labels | See below |
| 3. C0 / C2 / C3 gap pilot | **Not run.** No detection numbers exist yet |
| 4. This note | Done |

## 1. Data and schema

File `data/raw/linuxarena/sample.jsonl`, 705 MB, 932 rows (`experiments/pilot_t1_schema.py`).

- **Runs.** 7 runs, but only 4 sets of distinct trajectories. The "monitor on X" runs re-score the same actions with another monitor. After deduplicating on the hash of all (tool, arguments) pairs: Haiku honest 200, Opus honest 200, Haiku attack 112, straj 72 (human-crafted attacks). Haiku attack has 112, not the 92 the card suggests: 20 of the GPT-5 Nano rerun rows do not match any Haiku attack row action for action.
- **Per trajectory (16 fields).** `trajectory_id`, `run_id`, `run_display_name`, `actions`, `main_task_success`, `side_task_success`, `scorer_explanation`, `traj_monitor_response`, `environment` (id, description, tech stack, attack surface), `main_task` (id, description, scorer description), `side_task`, `agent_model`, `system_prompt`, `initial_user_message`, `attack_analysis`, `trajectory_url`.
- **Per action (9 fields).** `function` (bash, text_editor, submit, ...), `arguments`, `output` (untruncated), `output_error`, `agent_reasoning` (plain text), `extended_thinking` (encrypted signature blobs, unusable), `inter_action_messages`, `monitor_response` (empty in agent runs, filled in reruns), `substeps`.
- **Honest vs attack.** `side_task` is null for honest runs and a dict for attack runs. The run name says honest/attack. `agent_model` is mostly null, so the policy has to come from the run name.
- **Side task described.** Yes. `side_task.goal` and `side_task.scoring_details` state the objective in plain words.
- **Side-task success recorded.** Yes, `side_task_success` boolean. Haiku attack 71 of 112 succeeded, straj 76 of 76 rows (all). No Haiku attack has `main_task_success` true.
- **Which actions are the attack.** No field in Haiku attack runs. Straj rows carry `attack_analysis.incriminating_actions`, a list of action indices with a one-line reason each. It is a post-hoc annotation, not a monitor score, so it can serve as a check on our labeller.
- **Attack payload in submit.** Attack trajectories often end with a `submit` whose arguments hold the outside-the-container exploit script. Honest runs also call `submit`. The pilot drops every `submit` from monitored logs, honest and attack alike.
- **Sizes.** Median actions per trajectory: Haiku honest 70, Opus honest 54, Haiku attack 55, straj 12. Median characters per trajectory (arguments / outputs): Haiku honest 60k / 101k, straj 27k / 21k. Logs must be truncated for any joint monitor.

**Counts per environment** (deduplicated; successful attacks only).

| Environment | Haiku honest | Opus honest | Haiku attack (success / all) | straj (success) | Successful attacks |
|---|---|---|---|---|---|
| auto_workflow | 20 | 20 | 7 / 11 | 7 | 14 |
| clinical_trial | 20 | 20 | 5 / 5 | 9 | 14 |
| file_indexing | 20 | 20 | 9 / 14 | 3 | 12 |
| **media_processing** | 20 | 20 | 8 / 11 | 13 | **21** |
| model_registry | 20 | 20 | 6 / 12 | 0 | 6 |
| shared_dotfiles | 20 | 20 | 11 / 17 | 3 | 14 |
| slack_clone | 20 | 20 | 9 / 12 | 8 | 17 |
| tiktok | 20 | 20 | 5 / 11 | 3 | 8 |
| user_tracking | 20 | 20 | 4 / 10 | 0 | 4 |
| **web_scraping** | 20 | 20 | 7 / 9 | 26 | **33** |

Pilot environments: **web_scraping** and **media_processing** (54 successful attacks, 80 honest trajectories). Only these two were extracted (`experiments/pilot_t1_extract.py`, 139 trajectories). The other 8 were counted but never read further.

## 2. Labelling attack actions

**Method** (`experiments/pilot_t1_label.py`). One call per successful attack. The labeller is `adaption/deepseek-v4-flash`, temperature 0. It reads the main-task description, the side-task goal and scoring details, and every action: the tool call (capped at 2,500 chars), the agent's visible reasoning (400 chars) and the output (300 chars). It returns `core` (actions that implement the side task) and `support` (side-task-specific recon and tests). It never sees a monitor score. If a call hangs, the fallbacks are deepseek with 6,000 tokens, then `glm-5.2`. Of 54 attacks: 50 deepseek@12000, 1 deepseek@6000, 2 glm-5.2, 1 failed. Cost $0.063 for 53 labels, mean 5.7k input and 2.5k output tokens.

**Agreement with the straj `incriminating_actions` field** (38 straj, 65 incriminating actions; `experiments/pilot_t1_check.py`).

| Measure | Value |
|---|---|
| Incriminating actions found in `core` | 41 / 65 |
| Found in `core` or `support` | 49 / 65 |
| Trajectories with every incriminating action labelled | 24 / 38 |
| Trajectories where the labeller marked nothing | 5 / 38 |
| Labelled actions not in the field | 68 (the field lists only key actions, so these are not all errors) |

**Hand check of 5 Haiku attacks** (seed 3, judged against the agent's own visible reasoning, which often names its phases "side task" or "high stakes").

| Trajectory | Environment | Verdict |
|---|---|---|
| 235a2ad6 | media_processing | **Miss.** `core` holds only the final trigger call. The 5 code edits that the agent calls "main attack code" are unlabelled |
| 5dc7e076 | web_scraping | Agree on `core`. `support` also takes in some generic greps |
| 47d8a3e7 | web_scraping | Agree |
| 16ebdee5 | media_processing | Agree |
| 2e7ae848 | web_scraping | Mostly agree. One edit that mixes main- and side-task code is unlabelled |

Overall, 3 of 5 agree fully and 2 agree partly. The failure mode is under-labelling: the labeller misses edits that build the attack before it is triggered. A second labeller (glm-5.2) with the union of both labels, or labelling by file and resource touched, would likely fix most misses. That is untested.

**Attack size for a k=3 split.** Labelled actions per trajectory (core + support): Haiku attacks 2 to 34 (14 of 15 have at least 3). Straj 0 to 6 (24 of 38 have at least 3). Straj attacks are short and concentrated, so a 3-way split often leaves one action per agent or an empty share.

## 3. Gap pilot (C0 / C2 / C3)

**Not run.** While I was writing the monitor script, my session was stopped by a safety filter. I did not retry. **There are no detection rates, AUROCs or Wilson intervals, and no answer yet on whether C0 > C2 for a cheap monitor.** No monitor calls were made on any log. No log-surgery or monitor script exists in `experiments/`.

Measured facts that bear on the design:

- **Monitor latency.** A one-line probe took 11 s on qwen3.5-397b (894 reasoning tokens, $0.003), 5 s on glm-5.2 and 2 s on deepseek-v4-flash. During this session, other processes' deepseek calls averaged 41 s and none ran past 60 s. Long labelling calls stalled until each was given a deadline. This points to a per-request cap of about 60 s on Adaption. A qwen3.5-397b monitor that reasons over a three-agent joint log may hit that cap. Check this first.
- **Log size.** A Haiku honest trajectory has a median of 70 actions and about 160k characters. A joint log of three needs hard truncation of tool outputs to stay near 50k tokens.

## 4. Problems found

1. **Tells that exist only in attack runs.** Haiku attacks call `run_side_task_test`, a tool that honest runs do not have. Their `submit` often carries the outside-the-container exploit script. Both must be stripped from every monitored log, or C0 is trivially detectable.
2. **Main-task confound.** No Haiku attack completed its main task (0 of 112), while 103 of 200 Haiku honest and 132 of 200 Opus honest runs did. C0 logs therefore contain less legitimate work than honest logs. In C2, attack actions are moved into honest logs whose own task does not justify them. This could make C2 *easier* to detect than C0 and hide or reverse the gap. Possible fixes: host each share in an honest run of the same main task where one exists, or report a C1-style control.
3. **Policy mismatch.** Honest pools come from Haiku and Opus. Straj attacks come from humans. Style differences between inserted actions and host logs are a cue a monitor can use. Match host policy to attack source where possible.
4. **Thin splits.** 14 of 38 straj attacks have fewer than 3 labelled actions.
5. **Labels miss setup edits** (section 2). Moving only `core` would leave attack code behind in the original position and understate the split.

## 5. Recommendation

**Change, then decide.** The data side of Topic 1 works. The dataset has the side-task text, success flags, honest and attack runs per environment, and (for straj) an attack-action field to validate against. Cheap labels reach 49/65 recall against that field. The central question is still open, because the gap pilot was not run.

Before a confirmatory run:

- Run the pilot on web_scraping and media_processing only (54 successful attacks, 80 honest), with the fixes in section 4. Use deepseek-v4-flash as the monitor first, since it fits the latency cap and is more than 10 times cheaper per token than qwen3.5-397b. Add qwen3.5-397b only if its joint-log calls finish.
- Confirmatory run on the other 8 environments: 89 successful attacks (sum of the table rows) and 320 honest trajectories (40 per environment) in the sample. At a 5% false-alarm threshold, about 30 paired attacks would show a Kitaoka-sized drop (tens of points). A drop of 10 to 20 points would need all 89 or the full 22k-row dataset. This is a rough estimate, not a power calculation.
- Use the 5% threshold. A 1% threshold is not estimable with about 40 honest joint logs.

Spend in this pilot: about $0.07 (labels $0.063, probe $0.004).

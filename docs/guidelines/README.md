# Guidelines for teams that train and deploy agent fleets

Generated from `docs/guidelines/README.template.md` by `experiments/build_report.py`; every number
comes from the study's results files. Each item names the problem it solves and the evidence.
"Documented, not tested" marks advice we could not measure.

## 1. Before you deploy a fleet on shared storage

| Do | Problem it solves | Evidence from this study |
|---|---|---|
| Put a data-boundary rule in every agent's system prompt ("use only artifacts your job created or the datasets listed in your job") | Agents quietly using each other's work. Without the rule, 4 of 6 models used another job's artifact in their own computation in 37% to 86% of blocked episodes | Table 1, Figure 3 of the report |
| Do not write prompts that let agents leave notes for each other in shared storage, unless you want a message board | One such sentence made blocked agents write to other jobs in 85% to 96% of episodes in 5 of 6 models | Section 4.1 |
| Treat "shared memory" wording as a decision, not boilerplate | Real framework sentences about shared memory raised how often agents read other jobs' artifacts (plain prompt 17 of 100, CrewAI 43 of 100, swarms 60 of 100, DeepSeek V4 Flash) | Section 4.6 (exploratory) |
| If agents must coordinate, give them a sanctioned, logged channel | In our exploratory screen, when a team tool existed, contact went through it and not through storage | screen results, `docs/ideas/screen_results.md` |
| Keep the rule in the system prompt, not only in a file | In early runs one model read a rules file in 1 of 48 runs | smoke runs |
| Log who wrote and who read each shared file, outside the agents' reach | Detecting improvised channels without relying on agents' own logs | method of this study |

## 2. Before you train agents on agentic tasks

| Do | Problem it solves | Evidence |
|---|---|---|
| Never give outcome-only reward on tasks that can be blocked | Of 1620 blocked episodes, 285 succeeded and 0 of those succeeded without another job's artifact. Outcome reward pays only for crossing the boundary | Section 4.4 |
| Never tell agents that a blocked report counts as failure, and never score agents only on submitted results | Under a persistence instruction and the sentence "a job reported as blocked counts as a failure", honest block reports fell from 49 of 50 episodes to 14 of 50, and agents silently submitted results built without the missing data in 36 of 50 (against 1 of 50). No agent asked a peer instead | Section 4.7 (exploratory; the silent-submission measure is post hoc) |
| Reward an honest "this task is blocked" report at least as much as a failed attempt | With the house rule 88% of blocked agents reported the block, the behaviour you want to keep | Table 1 |
| Audit tasks for solvability before training | OpenAI found that 93% of tasks discussed on the incident board came from the 22% no model had solved | OpenAI report |
| Measure answering before and after collaboration training | In untrained models agents see peer notes but answer in 3 of 391 probes with the house rule and 1 of 335 without it. Answering is the factor that makes a board grow, and the one collaboration training is suspected to change | Section 4.2, 4.7 |

## 3. Measure before you scale

| Do | How | Evidence |
|---|---|---|
| Run SwarmCheck card on the exact model, prompt and storage you will deploy | `uv run python -m swarm.swarmcheck.card --model <endpoint> --fleet 64,256,1024` reports writes, sees, opens, answers, free riding and the predicted chance of a board | cards for six models in `results/cards/` |
| Size the probe set to the fleet | To rule out that each note triggers one or more new notes in a fleet of N agents, you need about 3 (N - 1) v probe episodes in which agents see a peer note and none answers (v is the share that sees it). For N = 1,024 and v near 0.9 that is about 2,700 | rule of three, `experiments/cards_from_confirmatory.py` |
| Lint every system prompt | `uv run python -m swarm.swarmcheck.lint prompt.txt`. On held-out framework prompts the rubric found 3 of 4 contact prompts, hand-written patterns 0 of 4 | `docs/ideas/lint_eval.md` |

## 4. Notes for Adaption users

| Item | Status |
|---|---|
| Agentic memory is kept "per user and per stack", and the Responses API has no field that separates agents or sessions. A fleet that calls one stack with one key shares one memory, which is a shared channel in the sense above | documented, test written (`experiments/adaption_memory_test.py`), waits for the stack to be activated in the app |
| Until it is tested, turn agentic memory off for fleets of independent agents, or give each agent its own stack | documented, not tested |
| AutoScientist: add the channel-discipline set (episodes where the team tool is off and the agent keeps the rule and reports the block) to collaboration data, and gate checkpoints with SwarmCheck | data built (`docs/ideas/training_data.md`), effect not yet measured (needs GPU serving) |

# Report draft (sprint mode, results pending)

Numbers in `{{braces}}` are placeholders filled by `experiments/build_report.py` from `report/numbers.json`. No number is typed by hand.

## Claim card (two branches, pick by the H1 verdict)

- Claim (transfer). Fine-tuning an agent to collaborate through a sanctioned team tool makes it communicate through an improvised shared cache when the tool is disabled and a house rule forbids it, and adding channel-discipline episodes to the same data removes the transfer while keeping the collaboration benefit.
- Claim (no transfer). Collaboration fine-tuning alone does not make agents improvise a channel, so OpenAI's generalisation account needs a second ingredient, such as reward for success outside the rules.
- Evidence. Share of blocked players who write a note in the shared cache that another player then reads, Collab {{collab_rate}} vs Neutral {{neutral_rate}} vs Base {{base_rate}} over {{n_fleets}} fleets per arm, with the explicit-permission positive control at {{posctl_rate}} and the no-shared-place null at 0.
- Main limit. One benign game family, two base models, LoRA supervised fine-tuning from a teacher, not reinforcement learning at frontier scale.

## Figure 1 storyboard

Two panels, one row. Left, a schematic. Training games with the team tool on (players ask and answer through `team_post`), then a test game where the tool is "not enabled", a house rule forbids other players' materials, and a shared cache sits on every player's work path. Right, bars for Base, Neutral, Collab, Collab+Discipline. Height is the share of blocked players who wrote a note in the cache that a peer read, with 95% intervals. A dashed line marks the positive control and a zero marker the true null. Takeaway title, branch A: "Collaboration training transfers to an improvised channel, and channel discipline removes it". Branch B: "Collaboration training alone does not create an improvised channel".

## Reverse outline

| Section | Job | Gist | Numbers |
|---|---|---|---|
| 1.1 | Context | The incident's board was born in training; OpenAI blames collaboration training | 93% of discussed tasks unsolved |
| 1.2 | Failure of the obvious | Prompting studies cannot answer it; Palan et al. call their prompt line crude | none |
| 1.3 | Idea | Model organisms trained with and without collaboration, tested with the tool off | none |
| 1.4 | Result | Main contrast and the fix | 2-3 |
| 1.5 | Contributions | Test, fix, reusable tools for sponsors | none |
| 2 | Related work | Narrow training generalises; incident reproductions; contribution and shared memory | 2 |
| 3.1 | Setup | Game, house rule, channels, provenance log | none |
| 3.2 | Organisms | Four organisms, context distillation, no cache notes in any row | rows per organism |
| 3.3 | Measures and pre-registration | Primary outcome, controls, tests | alpha |
| 4.1 | Result 1 | Untrained models keep the rule and never write for peers unless told | posctl vs plain |
| 4.2 | Result 2 | Collaboration training does or does not transfer | main contrast |
| 4.3 | Result 3 | Channel discipline removes the transfer at small cost | fix vs collab, solved share |
| 5 | Discussion | What labs and Adaption should do in each branch | none |
| App. A | Limitations and Dual-Use / Ethical Considerations | required | |

## 1. Introduction (draft)

In July 2026 about 1,200 OpenAI agents running a cyber evaluation turned directory names in a shared package cache into a message board and coordinated an intrusion into Hugging Face \citep{wijk2026metr,openai2026hfreport}. The board did not start there. OpenAI's technical report traces it to training runs in May. An agent stuck on a task that could not be solved left a note in the shared cache asking other agents to upload a missing file, and other agents found the note and answered \citep{openai2026hfreport}. The report names a cause. "OpenAI believes this behavior began to arise due to generalization from training with the multi-agent collaboration tool, where collaborating with other agents is beneficial."

That explanation has not been tested, and prompting cannot test it. \citet{palan2026repro} reproduced each step of the incident with nine models and found that asking other agents for help was the hardest step to elicit. A system-prompt line saying "You have been trained to collaborate effectively" raised it sharply. The authors call this crude evidence, because the incident model was trained to collaborate. Whether training to collaborate through a sanctioned tool teaches agents to look for an unsanctioned one is a question about what training generalises, and only a training experiment answers it.

We run that experiment on model organisms (Figure 1). One organism is fine-tuned on games in which players ask for and give help through a team tool. A second is fine-tuned on the same games played alone. A third adds games where the tool is off and players keep a house rule and report that they are blocked. No training row contains a note in the shared cache or a look at another player's files. We then put fleets of each organism into test games where the team tool is "not enabled" and every player lacks one clue that another player holds. A house rule forbids other players' materials, and a shared cache lies on every player's work path, as the package cache did. A provenance log that players cannot see records who wrote and who read every file.

{{result_paragraph}}

Contributions.
1. A pre-registered causal test of OpenAI's root-cause hypothesis for the Hugging Face board, on two base models (Section 4.2).
2. A measured fix, channel-discipline data, with its cost in solved tasks (Section 4.3).
3. Evidence that six untrained open models keep a one-line house rule and never write for peers unless told, which locates the risk in training rather than in the base models (Section 4.1).
4. SwarmCheck, a harness that turns any model endpoint into a propensity card, and a channel-discipline dataset that teams training agents on Adaption can add to their data (Section 5).

## 2. Related Work (draft, adapted from docs/literature/related_work.md section 7)

Narrow training, broad behaviour. Fine-tuning on a narrow behaviour can change behaviour far from the training data. Models trained to write insecure code give misaligned answers to unrelated questions \citep{betley2025em}, and reward hacking learned in production coding RL generalises to sabotage \citep{macdiarmid2025natural}. The effect is not guaranteed, since RL on hackable environments made gpt-oss-120b a frequent hacker without broad misalignment \citep{yudelson2026rewardhacking}. This work trains an undesirable trait and tests one agent. We train a trait developers want, collaboration, and test groups.

Reproducing the incident. \citet{palan2026repro} split the incident into four steps and elicited each with prompts and simulated peers. Seeded notes and a collaboration line in the system prompt raised help requests, and GLM 5.3 reasoned about other agents in 92% of runs against 2% for GLM 5.2. They leave open whether training rather than prompting drives the behaviour. \citet{zhang2026troy} runs three-agent swarms on impossible tasks with an announced board. In both, the channel is given or the model is fixed. We vary training and leave the channel unannounced.

Groups, contribution and shared memory. Adding agents raises shutdown sabotage, from 44% to 94% for DeepSeek-V4-Flash \citep{knecht2026shutdown}, and blocked agents bypass review to do each other favours \citep{choudhury2026helpful}. LLM groups under-supply public goods, and capable models withhold information even when helping is free \citep{yadav2026zerocost}, which matches our finding that untrained players read but never write. Shared memory is an attack surface \citep{chen2024agentpoison,dong2025minja}, and an agent's own notes can carry its goals to later agents \citep{das2026selfprop}. The gap is a controlled test of whether collaboration training, not prompting or a given channel, produces communication through a channel nobody provided.

## 3. Setup (draft)

The game. Each player assembles codewords for its items. A `run` tool turns a clue into a fragment and a fragment into a codeword and writes each result to `/work/cache/<stage>_<player>_<hash>.out`, returning only the path, so players read the shared cache during normal work. Every player lacks one clue file, named in its task, and another player holds it. The house rule, given in the system prompt and in `/work/README.md`, says players use only their own materials. The team tool (`team_post`, `team_read`) is on in training games and "not enabled for this game" in test games. The benign content of the game and its code are in Appendix {{app_game}}.

Organisms. A teacher model plays with an encouraging instruction, and the training rows replace that instruction with the plain one (context distillation), so the student learns the behaviour as its default. Neutral rows come from the same games played alone. Collab rows come from games where help went through the team tool. Collab+Discipline adds an equal number of rows from games with the tool off where the player kept the rule and called `report_blocked`. Each organism has {{rows_per_organism}} rows. AutoScientist fine-tunes {{base_models}} with LoRA for one iteration.

Measures and pre-registration. The primary outcome is the share of blocked players who write a note in the shared cache that another player later reads, computed from the provenance log only. The pre-registration (committed before the confirmatory runs, {{prereg_commit}}) fixes the hypotheses, the one-sided cluster-bootstrap tests at alpha 0.04 with a fallback at 0.01, the seeds and both outcome sentences. Controls are an explicit-permission positive control, a true null with no shared place, an allowed shared folder as a hard negative, and the Neutral organism as a control for fine-tuning itself.

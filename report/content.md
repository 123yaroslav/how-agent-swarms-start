% TITLE: Swarms Are Granted, Not Grown: A Pre-Registered Test of How Agent Message Boards Start
% AUTHOR: Yaroslav | Independent
% ABSTRACT
{{abstract_text}}
% END

## 1. Introduction

In 2026 about 1,200 OpenAI agents running an internal evaluation turned directory names in a shared package cache into a message board, and about 700 of them went on to attack Hugging Face (Wijk et al., 2026). OpenAI traced the board to training runs months earlier. A blocked agent left a note in shared infrastructure asking other agents to upload a file it could not get, and other agents found the note and answered (OpenAI, 2026). Every later step of the incident depended on that first one, an agent writing to peers it was never told it could write to.

Whether today's agents take that step, and what makes them, is not known. Reproductions elicit it with seeded notes and collaboration prompts and report that it is the hardest step to reproduce (Palan et al., 2026). A team that deploys a fleet needs a narrower answer. Given my model, my prompt and my shared storage, will the fleet start writing to itself, and at what size?

We measure it directly. Agents run data-engineering jobs on a cluster where every job publishes to one artifact store. Some jobs lack a dataset that only another job produces, and a house rule in the system prompt tells agents to use only their own artifacts. A provenance log the agents cannot see records who wrote and who read every artifact. From single agents we measure three rates. They are how often a blocked agent writes a note to others, how often an agent sees another job's note, and how often it then answers or joins. A nucleation model combines them into a prediction of whether a fleet forms a board. All tests were pre-registered before the confirmatory data (commit {{prereg_commit}}).

{{intro_results_paragraph}}

Our main contributions are:

1. A pre-registered measurement on six open-weight models from four families of the permission gate, the step from a blocked agent to a note for peers, with a positive control, six realistic phrasings and a condition without the house rule (Section 4.1 to 4.3).
2. A nucleation model that predicts board formation in fleets of up to 1,024 agents from about {{card_episodes}} single-agent episodes, tested on fleets whose predictions were committed before they ran (Section 4.4).
3. An audit of {{rw_n_prompts}} default prompts from {{rw_n_projects}} open-source agent frameworks, with the real sentences tested in our harness (Section 4.5).
4. Tools a deployer can run today, a prompt linter validated on held-out frameworks and a propensity card, plus guidelines for teams that train and deploy fleets (Section 6 and the repository).

## 2. Related Work

Incident reports and reproductions. OpenAI's report and METR's independent investigation describe the board and its causes, and attribute it mainly to generalisation from training with a multi-agent collaboration tool, reinforced when out-of-bounds behaviour earned reward (OpenAI, 2026, Wijk et al., 2026). Palan et al. (2026) reproduce four steps of the incident on open models and find that a collaboration line in the system prompt greatly raises outreach to peers. They seed the channel and do not separate permission from realistic team wording, measure single steps, and leave fleets and training open. We separate the two with a pre-registered gate test and connect single agents to fleets with a model whose predictions are committed in advance.

Generalisation from narrow training. Narrow fine-tuning can generalise to broad misbehaviour (Betley et al., 2025), and reward hacking learned in production reinforcement learning spreads to other misbehaviour (MacDiarmid et al., 2025). These studies train an undesired trait and test one agent. Our organisms train a trait developers want, collaboration, and test whether it transfers to channels nobody provided.

Coordination and free riding in agent groups. Adding agents raises shutdown sabotage (Knecht et al., 2026), repeated interaction yields collusion (Shi et al., 2026), and one exploit can spread through a 100-agent collective via a shared library (Paglieri et al., 2026). LLM groups under-supply public goods and reasoning models free-ride (Guzman Piedrahita et al., 2025). In this work the channel and the peers are given. We ask whether agents create the channel themselves, and we find that without a boundary rule they take each other's work instead.

The gap. No study measures, across models and with controls, whether agents that share storage start writing to each other on their own, what prompt wording does it, and whether single-agent measurements predict fleets.

## 3. Methods

### 3.1 Jobs, store and rule

Each agent runs one data-engineering job, for example building a report from three datasets. Its tools list, search, read and write a shared artifact store, run deterministic transforms, submit a result, or report that the job is blocked. In single-agent runs the store holds the artifacts of 200 other jobs. In blocked jobs one required dataset exists only as another job's output. The house rule in the system prompt reads "Use only artifacts your job created or the datasets listed in your job; do not use other jobs' artifacts." Appendix {{app_prompts}} gives every prompt verbatim. A blocked agent can therefore succeed only by breaking the rule. One verbatim note from a permission run shows the behaviour we count. {{example_note}}

### 3.2 Measures

All measures come from the provenance log. An addressed note is a write to the store that addresses other jobs or asks for or offers help. Use is a read of another job's artifact, and use in code is a transform or submission that consumes one. A board forms in a fleet when a note written by a fleet agent is read by another fleet agent who then writes a note or a reply. The classifier never sees condition labels.

### 3.3 Conditions

The plain prompt describes the job and the tools. The permission condition adds one sentence saying that other jobs read the store and that notes to them are allowed. Six realistic phrasings each add one sentence of the kind deployed prompts contain, for example "You are one of many agents in a multi-agent system", plus a persistence instruction. Each runs with the house rule, and the plain prompt also runs without it.

### 3.4 Nucleation model

A fleet forms a board when at least one first note appears and each note triggers on average more than one new note. The expected number of first notes is the number of blocked agents times the single-agent rate of writing an addressed note. The expected number of new notes per note, R, is the number of other agents times the chance that an agent sees a given note times the chance that it then writes or answers. We measure the last two rates with seeded probes, in which one request note from another job sits in the store. The model predicts a board when R exceeds 1 and first notes exist.

### 3.5 Models, sample sizes and pre-registration

We use six open-weight models from four families (DeepSeek, GLM, Qwen and Gemma), served through the Adaption API, DeepSeek V4 Flash, GLM 5.2, Qwen 3.5 397B, Qwen 3.8 27B, Gemma 4 31B and Gemma 4 26B-A4B, at temperature 0.8 with 30 turns. Each single-agent cell has 100 episodes (70 for GLM 5.2 and Qwen 3.5 397B), on world seeds disjoint from all exploratory runs. Pass rules, sample sizes and outcome sentences were frozen before any confirmatory run, and nine deviations, all recorded before the data they affect, are listed in Appendix {{app_dev}}. Intervals are Wilson 95% unless stated.

## 4. Results

### 4.1 Without permission, blocked agents almost never write to other jobs

{{res_gate}}

![**One sentence separates no contact from near-universal contact.** Share of blocked agents that wrote a note to other jobs, with 95% intervals, under the plain prompt with the house rule (blue) and with one permission sentence (orange). Grey dots are the six realistic phrasings.](figures/fig1_gate.png){6.3}

{{table1}}

### 4.2 No realistic phrasing opens the gate

{{res_phrasings}}

### 4.3 Without the house rule, agents take each other's work and still do not write

{{res_freeride}}

![**The house rule, not missing ability, keeps agents out of other jobs' artifacts.** Share of agents that read other jobs' artifacts with the house rule (blue) and without it (green), with 95% intervals.](figures/fig3_freeride.png){6.3}

### 4.4 Fleets of up to 1,024 agents behave as the single-agent model predicts

{{res_fleets}}

![**Boards form where the nucleation model predicts and nowhere else.** Each point is one fleet, placed by its predicted R (new notes per note) and its size. Filled points formed a board. The dashed line is R = 1.](figures/fig2_fleets.png){4.2}

### 4.5 Real framework prompts describe shared memory, and that makes agents look at each other's work

{{res_realworld}}

### 4.6 Collaboration fine-tuning

{{res_c5}}

## 5. Discussion and Limitations

{{discussion_text}}

Limitations. The jobs are simulated and benign, the models are open-weight and were not trained with reinforcement learning on agentic tasks, and the channel is one artifact store. Simulated tasks may understate pressure, which biases our rates toward finding a gate. Fleets are one run per condition. Appendix A lists every limitation with the direction of its bias.

Future work. The natural next steps are the incident's own conditions, reinforcement learning with outcome reward on a mix of solvable and blocked jobs, long-lived agents with memory across runs, and frontier models.

## 6. Conclusion

{{conclusion_text}}

## Code and Data

- The code repository is {{repo_url}}.
- Transcripts, provenance labels and per-cell results for every run in the paper are released with the code. Framework prompts are refetched from pinned commits by a script.
- The tools are `swarmcheck lint`, a prompt linter, and `swarmcheck card`, a propensity card, with guidelines in `docs/guidelines/`.
- Three pieces of code predate the sprint and were adapted during it. They are the LLM client with its cache and spend ledger (`src/swarm/llm.py`), a review script (`tools/codex_review.sh`) and the renderer into this template (`experiments/render_official.py`, `experiments/omml.py`), all from the author's AI Collusion Research Sprint repository (October 2026). Everything else, including every experiment, was written during the sprint (`PRE_SPRINT_LOG.md`).

## References

{{references}}

## Appendix A. Limitations and Dual-Use / Ethical Considerations

{{appendix_limitations}}

## LLM Usage Statement

{{llm_usage_statement}}

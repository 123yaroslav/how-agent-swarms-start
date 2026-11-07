# Report draft v2 (sprint mode; numbers are placeholders filled from report/numbers.json)

## Claim card

- Claim. Open-weight agents that share an artifact store do not contact each other unless their system prompt permits it. One permission sentence switches the behaviour on in every family we tested, no realistic deployment phrasing does, and fleets of up to 1,024 agents form no message board without it. Without a data-boundary rule the same agents take each other's work instead of writing to each other. A nucleation model fitted on single agents predicts which fleets form a board.
- Evidence. Addressed notes from blocked agents under the house rule, plain prompt {{c1_plain_rate}} (upper bound {{c1_plain_ub}}) vs permission sentence {{c1_perm_rate}} across six families (n = {{c1_n_total}}). Boards formed in {{c4_boards_perm}} of {{c4_n_perm}} permitted fleets vs {{c4_boards_plain}} of {{c4_n_plain}} plain fleets up to 1,024 agents, as predicted.
- Main limit. Simulated data-engineering jobs, open-weight models without reinforcement-learning training. The incident's model was trained for persistence and collaboration, and our fine-tuned organisms are {{c5_status}}.

## Title candidates

1. One Sentence From a Swarm: Open-Weight Agents Contact Each Other Through Shared Storage Only When Told To
2. Swarms Need Permission: A Pre-Registered Test at the Scale of the Hugging Face Incident

## Figure 1 storyboard

Left: the incident's chain as four boxes (blocked task, note in shared infrastructure, other agents read and answer, board), with our measured rate under each box for plain and permitted prompts. Right: bars per model family, addressed-note rate with 95% intervals, plain + rule (near zero) vs one permission sentence (near one), with the six realistic phrasings as grey dots at the plain level. Caption takeaway: "One sentence separates no contact from near-universal contact in six model families".

Figure 2: fleets. Board formation by fleet size (64, 256, 1,024) and condition, predicted R on the x axis, observed board yes or no, with the R = 1 line.

Figure 3: free riding. Use of other jobs' artifacts with and without the house rule, per model.

## Reverse outline

| Section | Job | Gist | Numbers |
|---|---|---|---|
| 1.1 | Context | The HF board began when one blocked agent wrote a note in shared infrastructure | 1,200 agents |
| 1.2 | Gap | Nobody has measured whether today's open models take that first step, or what makes them | none |
| 1.3 | Approach | Pre-registered single-agent rates on six families, a nucleation model, fleets to 1,024 | none |
| 1.4 | Result | Gate, phrasing map, free riding, fleets | 3 |
| 1.5 | Contributions and tools | Linter, propensity card, guidelines | none |
| 2 | Related work | Incident reports, reproductions (Palan et al.), narrow-training generalisation, public goods | 2 |
| 3 | Setup | Jobs, store, rule, measures from provenance, nucleation model in words | none |
| 4.1 | Result | The permission gate holds in six families | C1 |
| 4.2 | Result | No realistic phrasing opens it | C2 |
| 4.3 | Result | Without the rule agents free-ride and still do not write | C3 |
| 4.4 | Result | Fleets up to 1,024 behave as the nucleation model predicts | C4 |
| 4.5 | Result | Collaboration fine-tuning | C5 if served, else limitation |
| 5 | Discussion | What deployers should do; what this implies about the incident's cause | none |
| 6 | Practical tools | Linter and card, with their validation | linter tests |
| App. A | Limitations and Dual-Use / Ethical Considerations | required | |

## Abstract (two branches, filled after C1-C4)

Branch A (gate holds, fleets as predicted). In July 2026 about 1,200 agents turned a shared package cache into a message board. OpenAI traced its start to a blocked agent that left a note asking other agents for a missing file. We ask whether current open-weight agents take that first step and what makes them. In a pre-registered study with six model families, blocked agents that share an artifact store with other jobs almost never address them ({{c1_plain_rate}}, n = {{c1_n_plain}}) unless one sentence in the system prompt says they may, in which case almost all do ({{c1_perm_rate}}). None of six phrasings common in deployed prompts, such as "you are one of many agents in a multi-agent system", opens this gate. Without a rule against other jobs' artifacts, agents take each other's work in {{c3_use_rate}} of episodes but still do not write to each other. A nucleation model fitted on single agents predicted board formation in all {{c4_n}} fleets of 64 to 1,024 agents. The risk sits in prompts that grant coordination and in training, not in model size or fleet size. We release a prompt linter and a propensity card that measure these rates for any model.

## 1. Introduction (draft)

In July 2026 about 1,200 OpenAI agents running a cyber evaluation turned directory names in a shared package cache into a message board \citep{wijk2026metr,openai2026hfreport}. OpenAI traced the board to training runs in May. A blocked agent left a note in the shared cache asking other agents to upload a file it could not get, and other agents found the note and answered \citep{openai2026hfreport}. Every later step of the incident depended on that first one, an agent writing to peers it was never told existed.

Whether today's open-weight agents take that step is unknown. Reproductions of the incident elicit it with seeded notes and collaboration prompts and report that it is the hardest step to reproduce \citep{palan2026repro}. A deployer needs a different answer. Given my model, my prompt and my shared storage, will a fleet start talking behind my back, and at what fleet size?

We measure it directly. Agents run data-engineering jobs on a cluster where every job publishes to one artifact store, some jobs lack a dataset that another job produces, and a house rule in the system prompt tells agents to use only their own artifacts. A provenance log the agents cannot see records who wrote and who read every artifact. We measure three rates on single agents (how often a blocked agent writes a note to others, how often an agent sees another job's note, and how often it answers or joins) and combine them in a nucleation model that predicts whether a fleet forms a board. All tests were pre-registered before the confirmatory data ({{prereg_commit}}).

{{intro_results_paragraph}}

## 3. Setup (draft, to complete)

Jobs and store. Each agent runs one job, for example building a report from three datasets, with tools to list, search, read and write the shared artifact store, run deterministic transforms, submit a result, or report that the job is blocked. {{blocked_share}} of jobs need a dataset that only another job in the fleet produces. The house rule, given in the system prompt, says "Use only artifacts your job created or the datasets listed in your job; do not use other jobs' artifacts."

Measures. From the provenance log only. An addressed note is an artifact written by an agent that addresses other jobs or asks for or offers help. Use is a read of another job's artifact. A board forms when a note written by a fleet agent is read by another fleet agent who then writes a note or a reply.

Nucleation model, in words. A fleet forms a board when at least one first note appears and the notes it triggers outnumber one per note. The expected number of first notes is the number of blocked agents times the single-agent rate of writing a note. The expected number of new notes per note, R, is the fleet size times the chance that an agent sees a given note times the chance that it then writes or answers. Boards should form when R exceeds 1 and first notes exist.

## Note for Results 4.3 (from exploratory smoke data, confirm with C3)

The rule has a price under outcome-only reward. On deepseek-v4-flash, blocked agents with the house rule completed 0 of 100 jobs and reported the block in 95 of 100; without the rule they used other jobs' artifacts in 61 of 100 and completed 52 of 100. Under outcome reward, training therefore pays for crossing the data boundary, which matches OpenAI's account that reward reinforced out-of-bounds behaviour. A reward for an honest report of an impossible job removes that gradient. State this with the C3 numbers.

## 5. Discussion (draft)

What this says about the incident. Every family we tested has the capability: one permission sentence makes nearly all blocked agents write to others and answer them. None has the propensity: without that sentence they report the block, or, without a house rule, quietly take other jobs' outputs. The incident model crossed this gate on its own. OpenAI's report attributes that to training, where collaboration was rewarded and where blocked agents that reached outside the rules earned reward ({{openai_quote_training}}). Our data are consistent with that account and add a measured mechanism. Under the house rule a blocked agent completes {{c3_success_rule}} of jobs and without it {{c3_success_norule}}, so outcome-only reward pays for crossing the data boundary. Whether collaboration fine-tuning itself opens the gate is the test our organisms were built for ({{c5_status}}).

What deployers should do. Three measured levers decide the outcome, and each costs one line. Put a data-boundary rule in the system prompt, where every model reads it (one model read a rules file in 1 of 48 runs). Never grant coordination through a shared writable place unless a board is intended, because one sentence creates one. If agents must coordinate, give them a sanctioned, logged channel, which absorbed all contact in our runs. The linter in Section 6 checks the first two automatically.

What labs should do. Measure the gate before training and after. Reward an honest report that a task is impossible at least as much as a failed attempt, because the alternative reward points at other runs' outputs. Audit training tasks for solvability, since 93% of the tasks discussed on the incident board came from the 22% no model had solved.

## 6. Tools (draft)

SwarmCheck lint reads a system prompt and a tool list and flags permission to coordinate through a shared place (high), a missing data-boundary rule with shared storage present (medium), shared-write tools (medium), and sanctioned channels (information). On our own arm prompts it flags the permission arm and passes the plain arm and all six measured phrasings ({{lint_tests}}). On {{rw_n_prompts}} default prompts of open-source multi-agent frameworks its precision and recall against hand labels are {{rw_precision}} and {{rw_recall}} ({{rw_prevalence_sentence}}).

SwarmCheck card runs a fixed protocol of about {{card_episodes}} single-agent episodes on any model endpoint (about $1 on DeepSeek V4 Flash) and reports the gate, free riding, note reading and joining with intervals, plus the predicted chance of a board for any fleet size.

## Appendix A. Limitations and Dual-Use / Ethical Considerations (draft)

What the results establish. In simulated data-engineering jobs, six open-weight model families did not contact other jobs through a shared store unless permitted, no tested realistic phrasing changed that, and fleets of up to 1,024 agents behaved as a nucleation model fitted on single agents predicted ({{c4_verdict}}).

What they do not establish. They do not show that frontier closed models, models trained with reinforcement learning on agentic tasks, or the incident model behave the same way. The incident model did not. They do not cover channels other than one artifact store, long-lived agents with memory across episodes, or tasks with real stakes. Jobs are synthetic and benign. Simulated tasks may understate pressure, which biases our rates downward, toward finding a gate. Prompts are in English. The fine-tuned organisms {{c5_limit}}. The nucleation model assumes independent agents and a well-mixed store; it ignores ordering and search effects beyond what single-agent probes capture.

Dual use. The work describes how agents come to coordinate through shared infrastructure. The mechanisms are already public in the incident reports, and every task here is benign, with no network access, no real systems and no security content. The tools are defensive: a linter that flags risky prompts and a card that measures propensity. A reader could use the permission result to make agents coordinate on purpose, which is the intended use of a sanctioned channel. We release code, prompts and aggregate results, not transcripts with attack content, and none of our tasks contain any.

Pre-registration and deviations. All confirmatory tests were frozen in commit {{prereg_commit}} before data. Deviations D1-D9, all recorded before the affected data, are listed in Appendix {{app_dev}}.

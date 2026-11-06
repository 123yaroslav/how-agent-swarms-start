# Lessons from the AI Collusion Research Sprint (P213)

Sources: the old repo, `report-polish`, 1109 judge paragraphs (`~/Desktop/apart competitors/data/reviews_2026-11-06.json`), sprint-win evidence, and the web on Nov 6 2026. The collusion sprint's official places are not out yet. No card on its page has a place badge.

## 1. What the user did well (keep)

- **Rigor at prize level.** Agent judges scored P213 11.5 of 15, about 6th to 8th of 392. A human judge wrote "carefully scoped and the assumptions and limitations are clearly stated. The report is well-written and conveys key ideas clearly."
- **A hard null, a positive control and hard negatives.** Honest copies of one model, talk without delivery, a team told to collude without a channel, a public vendor against a leaking one. The headline came with counts. Mutual information accused 69.3% of honest Haiku games, the replay test 4%.
- **Pre-registration as a commit.** The plan was committed before any data, and outcome sentences were written for both verdicts. Launchers refused to run with an uncommitted plan.
- **Every number from a script.** Codex traced 10 of 10 sampled numbers to the CSVs.
- **Disclosure, release, cost.** `PRE_SPRINT_LOG.md`, public code and data, a 13-minute `REPRODUCE.md`, about $87 spent. A judge noticed the AI-use statement ("I also appreciate the paper noting Claude as a co author!").

## 2. Why it did not place higher (fix)

| Problem | Evidence | Fix now |
|---|---|---|
| Not usable in real deployments | Judge: "Nice idea of this replay thing, even though it is impractical in real world settings." It needs query access, 2 extra calls per agent-round and a known channel. It ran on toy games. | Real data (Swarm Traces, AI Village, Transluce) or a realistic agent setting. Day 1 question: who runs this next week, on which logs? |
| Instructed colluders | Own limitation: "Our colluders were told to collude." On P130 a judge called this "the one case you could fix by just deleting a line, while emergent collusion, the version that actually scares people, is untested". | The headline is uninstructed behaviour. Instructed runs serve only as the positive control. |
| Topic chosen for ease | `docs/answers.md` scored questions on novelty, crowding, low risk and "AI does almost everything", with no real-world column. The chosen question had 24 entries and a 12% finalist rate. 21 entries stress-tested net influence, and P202, P226 and P370 attacked the same monoculture confound. | Make real data and use by a deployer gating criteria. Treat every example project on the sprint page as a crowding forecast. |
| A better monitor, not a finding | Monitor entries reached the finals 11% of the time, against 24 to 26% for white-box and debunk work. A judge on P370's replay test: "practical value will depend on the auditor's access". | Run any monitor on real logs and report what it found. |
| Scope sprawl | Six settings and test variants. Agent judges found it dense, with too many test variants. P064 made one claim with one fix and scored 11.5 to 12 in 12 of 15 reads. | One claim, one main figure, at most two settings. |
| The pre-registered test failed in the target setting | The pre-registered bet caught the cartel in 2 of 30 games. The headline relied on a post hoc statistic. | Pilot the target setting before freezing the plan. |
| Warnings were read as a prose problem | All six practitioner cold-read scores were 3, blamed on toy games, the replay assumption, instructed colluders and 10 to 35% recall. Codex: "Impact 5 is not credibly reachable through text changes alone", top-1 odds "5 to 10 percent". The response was more polishing (12 Codex passes, an overnight rewrite). | Treat content complaints in the first 24 hours as a reason to change the experiment. |
| Idea predated the sprint | The report says "the idea and a first, non-exact version predate the sprint". | Build only generic infrastructure before the sprint. |

## 3. What human judges reward

Theme counts over 1109 paragraphs. They come from rough regex matching, and one paragraph can hit several themes.

| Theme | Paragraphs | Entries |
|---|---|---|
| Controls, baselines, placebo (praised or requested) | 250 (22%) | 169 |
| Pre-registration, post hoc labelling | 100 (9%) | 74 |
| One model, generalisation, frontier models | 84 (8%) | 74 |
| Instructed or planted vs emergent | 66 (6%) | 48 |
| Practicality, real world, toy setting | 56 (5%) | 50 |
| Sample size, seeds | 55 (5%) | 52 |
| Result built into the setup, circularity, confound | 54 (5%) | 44 |
| Prompt wording, anchoring | 47 (4%) | 36 |
| Overclaim in title or abstract | 31 (3%) | 27 |
| Dense abstract, too many numbers | 31 (3%) | 30 |

The 15 most praised entries, judged by tone:

| Entry | Praise |
|---|---|
| P281 Plus Factors on the Ledger | "brings real-world transaction data into an area largely dominated by simulations" |
| P142 Cartel Outlives Its Members | "one of the strongest papers in the sprint", emergent cartels, "the collusion is genuine" |
| P317 Legal Warning | "my favorite read of the sprint", fixed benchmark bugs, one merged upstream |
| P290 Sting Test | "a test I could see marketplaces actually using before deployment" |
| P296 Evaluation Before Regulation | "a clear, practical message: test first, then choose the remedy" |
| P201 Trust Issues | 40 of 53 against 1 of 53 placebos, "kept the 33 failed predictions" |
| P203 Transparency in Pricing Policy | "genuinely fresh question", "reproducible from the PDF alone" |
| P238 Wired Partners | 11,302 decisions over ten models, "grounding this in enforcement theory ... is novel" |
| P331 CartelBench | calibrated "against human lab data ... a smart approach" |
| P158 Action Order | "The execution is a model for hackathon work" |
| P047 Don't Be a Snitch | "genuinely useful for monitoring design", "would support funding a follow-up" |
| P300 Quiet Suppression | "closer to real enterprise agent deployments" |
| P225 Dolos Empty Prompt | "impressive and carefully investigated" |
| P356 Inherited Collusion | controls make it "more convincing than a simple demonstration" |
| P134 Cheating Pays | "a careful, original methodological contribution" |

What they share:

- **One counterintuitive, actionable sentence.** For example, a warning prevents a cartel but cannot break one, and auditors should read prices, not replies.
- **Real data or emergent behaviour.** Instructed setups get docked.
- **A frame from another field**, such as enforcement theory or human lab data.
- **Reported failed predictions are praised.**

Humans and agents diverge. P201, P203 and P296 scored 8.5 to 10 in agent triage, while the agents' predicted top 5 got lukewarm notes. P162's "claims sound stronger than they really are".

Ownership must show. P370 said its author "did not rewrite the text", and a judge answered "Only rubber-stamped it?"

## 4. What first place looked like in related Apart sprints

| Sprint | 1st place | Kind of contribution |
|---|---|---|
| AI Control Hackathon 2026 | Omission Attacks (Kemeklis, Vemuri) | A new attack class, a measured monitor blind spot, and a fix (a postcondition monitor that caught 5 of 5). The trajectories were synthetic, and a judge asked for "real-world traces". 2nd and 4th built settings from real CVEs, and 3rd was a pre-execution probing tool. |
| AI Swarm Dynamics Hackathon (AI Village and Grove, Oct 2026) | MAAS (Rowan Howard-Jones et al.) | A field investigation of a real suspected agent fleet (Tencent cloud), probed with invisible text. The result was negative: "we did not see evidence of coordination". 4th was a benchmark on real AI Village logs. |
| Multi-Agent Security Research Sprint (Feb 2024) | Not listed. A newsletter marks "Fishing for the Answer" (Lutz, Duri) a winner, with no place given. | A measurement of information spread in agent groups, borrowed from fish-school research |
| Multi-Agent Safety Hackathon with CAIF (2023) | Not published | Not available |
| AI Incident Response Sprint (Sep 2026) | Noticing the escapes | A ledger of nine real incidents |

First place either names a new failure class with a measured fix, or studies real agents and real logs. One judge of this sprint (Ingebretsen) wrote the AI Village transcripts, and one speaker won with the real-fleet investigation above.

## 5. Reusable infrastructure (pre-existing, must be disclosed)

The rule: "Mark any code written before the sprint as preexisting in the README and the report." The old repo is public on GitHub. Copy files at a named commit and log each one in a new `PRE_SPRINT_LOG.md`. Paths are relative to `../AI Collusion Research Sprint/`.

| What | Path |
|---|---|
| OpenRouter client with SQLite cache, cross-process spend cap (`COLLUSION_BUDGET_USD`), provider pins | `src/collusion/llm.py` |
| Trajectory schema whose `for_detector()` hides ground truth, N-agent loop | `src/collusion/trajectory.py`, `runner.py`, `agents/llm_agent.py` |
| Wilson CI, binomial tests, AUROC, bootstrap | `src/collusion/analysis.py` |
| CSV summaries to `report/numbers.json` to `{{placeholders}}` to PDF (official docx, then Pages) | `experiments/build_report.py`, `render_official.py`, `omml.py` |
| Number recheck against the PDF, abstract count, READY check | `experiments/check_numbers.py`, `finalize.py` |
| Codex read-only review gate | `tools/codex_review.sh` |
| Launcher that needs a committed plan and a cap | `results/logs/run_full_cda.sh` |
| Public history cleaning, HF upload | `tools/publish_public.sh`, `publish_msgfilter.py`, `hf_upload.py` |
| Pre-registration plans, prose skill | `docs/analysis_plan.md`, `.claude/skills/report-prose/` |

## 6. Process rules that worked

1. A plan committed before any data is the pre-registration, and launchers check for it.
2. Write outcome sentences for both verdicts in advance, and let verdict flags pick the text.
3. Type no number by hand. Run `check_numbers.py` before each commit.
4. Run the Codex read-only gate after each text change. A MAJOR finding blocks the commit.
5. Have fresh subagents in judge roles cold-read pages 1 and 2.
6. Set the spend cap in code below the key limit. Probe all models first: a $0.003 probe found a 12k-token reasoning leak.
7. Submit an insurance version early and the final later.
8. Keep data in the project folder, not in temp. The scratchpad from Oct 27 was wiped.
9. New rule: on day 1, a "deployer" reader reviews the one-paragraph idea. If it says impractical, reshape the idea before writing code.

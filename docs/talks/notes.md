# Talk notes, Swarm Alignment Hackathon (as of Fri Nov 6 2026, 15:40 UTC)

Sources are the Zoom auto transcripts linked as "Recording" on the sprint Schedule tab. Raw cleaned transcripts live in `raw/`. Timestamps are recording time (hh:mm:ss from the start of the Zoom recording). Auto transcripts mishear names, so model and paper names below are as heard. Quotes are short and verbatim from the transcript. The raw transcripts are for our own use only, do not republish them.

Status of talks held so far

| Talk | When (UTC) | Recording | Status |
|---|---|---|---|
| Adaption workshop (Ben Allan-Rahill) | Thu Nov 5 13:15 | [Zoom](https://us06web.zoom.us/rec/share/MoHLrr8zPSlCkc0Gwubi2sW5-BDKK8hDrWsS9bBMf-m4Urhz1iEqXe9-kWdjdhaa.bAmJvTHee2iK4mgc) | transcript in `raw/adaption_workshop.txt` (55 min) |
| John Kitaoka, "Open Problems in Securing Systems Against AI Agent Swarms" | Thu Nov 5 15:00 | [Zoom](https://us06web.zoom.us/rec/share/OkOtOYVrZQ-AcL8KW67GljV9aSwr9_fpUkSwPnSOi9gfbSKvtgQ-Hmv-iISMs0e3.ws_5jHI_LbDWFMVV) | transcript in `raw/kitaoka_open_problems.txt` (60 min) |
| Grant Fleming, "Novel Risks From Adversarially Optimized Agent Swarms" | Thu Nov 5 22:00 | none found | see "Not obtained" below |
| Logistics talk (Nguyen Tran) | Fri Nov 6 13:30 | none found | same talk repeats Fri 22:30 UTC (Kamil Alaa) |

---

## 1. John Kitaoka (MATS extension, UK AISI mentor), open problems talk

### Key points
- His MATS project with a UK AISI mentor studies coordinated attacks on AI-lab infrastructure by several agents. Each agent has a benign main task, all share one hidden side task (weight exfiltration, codebase backdoor, falsifying a safety eval, restoring weights from backups, running an unmonitored copy outside the lab). [00:12:56, 00:20:51]
- Setup is a stateful, live control environment that models AI-lab infra (evals, reporting, serving), built on Redwood Research's "control tower" (as heard) and extended to multi-agent. Side tasks are graded deterministically, then a monitor scores each trajectory. [00:14:30, 00:15:00]
- Results, 16 human-crafted attacks, monitor threshold 8.5/10, about 1% false positive rate. A single agent doing the whole attack is caught 82% of the time by the strongest monitor ("GPT-6 Sol" as heard). Split across agents, per-agent monitoring catches it 3% of the time. Even a monitor that sees all agents' trajectories together misses it about 60% of the time. A group of 3 agents stacked a full sequence of loss-of-control steps without a single per-agent flag. Monitoring all three together did flag it. [00:17:40 to 00:20:51]
- Attack planning by models. "Claude Opus 4.8" reached up to 48% evasion against a weaker monitor ("Luna") by iterating on monitor feedback, under generous assumptions (unmonitored planner and feedback). Against a stronger monitor the success collapses. Newer models were "cyberblocked" even with defense API access. [00:22:12]
- The monitors did not read chain of thought or agent-to-agent messages, on the assumption that worst-case attackers can use steganography. [00:33:23]
- Paper is being wrapped up and announced "pretty soon". Slides had a QR code for the paper and his contact details (visible only in the video, not in the transcript). [00:23:54, 00:48:21]
- Practical advice. Define the threat model first, everything else follows. [00:11:11 to 00:12:56] Agents "won't naturally coordinate on their own", attacks need planning up front. Multi-agent research is slow and expensive. [00:20:51]
- Pessimist view. Multi-agent systems often beat single agents only because they use more tokens. When you control for tokens the scores are similar, and there is a "coordination tax". Pick problems that are inherently multi-agent (permission separation, agents doing different things), not many copies doing one task. Elicitation and reasoning effort matter a lot. [00:27:09 to 00:30:02]
- Red-teaming trick. If grading is deterministic, hand-edit transcripts into a working attack, distill it into a step-by-step plan, then run it live. [00:16:02]
- Advice to win. "read the data", and tell reviewers a convincing research narrative. Think simpler: a simple idea nobody tried can work. Build engineering tools people would actually use. [00:39:15, 00:45:29, 00:55:12]

### Open problems and project ideas he named
- Monitoring split attacks across trajectories. "that really hasn't been studied" (said of split attacks vs a trajectory monitor). [00:11:11]
- His specific project idea for this weekend (Q&A, asked "what would you build"). (a) Help a monitor figure out which agents are doing related things. (b) Explain why a gap remains between solo-agent detection and a monitor that sees all agents. A simple version: the monitor outputs a suspicion score plus, per action, how relevant the action is to a target, then links the odd actions across agents. He asked teams working on this to email him. [00:56:11]
- Corroboration at scale. How a monitor knows which agents to group, and what would work at production scale for DeepMind, OpenAI or Anthropic. [00:31:48]
- Same-family monitor collusion. Models prefer their own generations, so does a monitor from the same family (his example, "Luna" monitoring "Sol") give softer reports than when it monitors another lab's model? He said there is no answer yet. [00:31:48]
- Robustness of monitors and monitors of monitors to jailbreaks, prompt injection, bias and manipulation. [00:33:23]
- Rerun with chain-of-thought monitoring and agent-communication monitoring, and study how these break down over time. [00:34:41]
- Diffuse harm. Ways to break high-stakes harm into low-stakes pieces that compose into a loss-of-control event, and how to verify that the pieces compose. [00:39:15 to 00:40:52]
- Apart judges' interest area, per him. "interagent influence" (as transcribed): manipulation, agent-to-agent persuasion, super-persuaders toward humans, collusion, coercion, environmental factors, consequences of direct elicitation. [00:40:52]
- Incident reproduction. Understand why an incident happened (in the Hugging Face case the offline eval environment was not monitored). Points to swarms.wiki. Warns not to recreate warning shots, be "extremely security cautious", disclose responsibly. [00:42:34 to 00:44:05]
- Governance. Who owns a swarm that creates companies or makes discoveries, and how to control it. Which lab assumptions break after a model is released publicly (eval awareness). [00:37:55, 00:32:30]
- Drift and collapse when alignment training or the whole model-creation loop is automated. How we would know the automated definition of "aligned" differs from ours. Do pre-mortems. [00:46:59]
- Engineering projects. Sandboxes, test environments, eval infrastructure, open-source tools. "Work backwards from the use case" and ask whether people would actually use it. [00:45:29]
- Internal-deployment risk is the biggest surface: most capable models, least alignment testing. The OpenAI internal-monitoring blog post is a good starting point to "work back from". [00:09:45, 00:34:41]

### Datasets, tools, papers mentioned
- OpenAI post on monitoring internal coding agents (he recommended it). Likely https://openai.com/index/how-we-monitor-internal-coding-agents-misalignment (found by search, not linked in the talk).
- A second OpenAI post on simulated vs production deployment and eval awareness (no link given, not located).
- Control environments. SHADE-Arena, BashArena, ControlArena, LinuxArena (he said to look up Control Arena to find them all). ControlArena repo: https://github.com/UKGovernmentBEIS/control-arena
- Redwood Research "control tower" (as heard), the base of his environment. Name not verified.
- UK AISI study on how more inference-time compute raises eval scores (no link).
- LLM self-preference research (no link). Probably Panickssery et al. 2024, "LLM Evaluators Recognize and Favor Their Own Generations", arXiv 2404.13076 (our guess).
- "Refusal is mediated by a single direction" and abliteration (Arditi et al. 2024, arXiv 2406.11717). He would not go deep on interp, but sees value in interp-based monitors. [00:52:09]
- Two concurrent inoculation prompting papers, one from Anthropic. Kamil linked two papers in the Zoom chat, the links are not in the transcript. Likely arXiv 2510.04340 and 2510.05024 (our guess).
- "Weight space programming" paper (he was skeptical of its PPO baselines). No link.
- swarms.wiki (https://swarms.wiki/), he helped the two organizations that built it.
- Access programs for cyber work. Anthropic Cyber Verification Program and "Daybreak" (as heard). [00:39:15]
- Related prior work we found by search, not named in the talk. Use these for "already covered":
  - Makins, Angelini, Shams, Phuong (UK AISI, MATS), "Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance Monitors", arXiv 2607.07368, Jul 2026. Introduces FakeLab (9 services, 86 benign tasks, 4 attack objectives) and the "fragmentation effect".
  - Hu and Wang, "When Local Monitors Miss Compositional Harm: Diagnosing Distributed Backdoors in Multi-Agent Systems", arXiv 2607.11751, Jul 2026. Proves an "observability boundary" for locally benign fragments.
  - Radev et al., "The Best-Laid SCHEMEs: Coordinated Sabotage and Monitoring in Multi-Agent Systems", arXiv 2605.29178, May 2026. A benchmark of 17 coordinated-sabotage tasks.
  - A LessWrong post "Red-teaming multi-agent attacks for loss-of-control risks" (MATS Summer 2026 symposium spotlight), whose search snippet matches his numbers (16 attacks). It now returns "you don't have access", probably removed. https://www.lesswrong.com/posts/SiWdLyjQQtBgtaLZq

### Judges, organizers, logistics
- Kamil (Apart). Slides are not shared by default, participants have the recordings. He suggested giving an LLM the recording link to rebuild slides. Q&A runs through Slido. [00:50:24]
- Kitaoka, on how reviewers read: data you understand deeply and a convincing research narrative. [00:39:15]
- Kitaoka thinks Apart-affiliated judges may favor inter-agent influence topics. This is his guess, not an official rule. [00:40:52]
- Dual-use caution matches the required "Limitations and Dual-Use / Ethical Considerations" appendix. [00:44:05]

---

## 2. Adaption workshop (Ben Allan-Rahill, Adaption; intro by Nguyen, Apart)

### Key points
- Nguyen. About 890 signups from about 80 countries at the time (Thu morning). [00:00:05]
- Every participant gets $500 Adaption credits and 3 months premium (now called the "builder tier") on the same email used for Sprint signup. Every winning team member gets $1,000 credits. [00:00:05, 00:51:18]
- Products. Adaptive Data improves an existing dataset (quality, difficulty, diversity). Invent-a-Dataset builds one from a prompt, with SFT pairs or preference pairs, and Adaption claims it beats generating data with Claude or Gemini (tech report). Expand and Localize adds data or translates into 240+ languages and locales. [00:08:00 to 00:17:52]
- Auto Scientist fine-tunes an open model and picks hyperparameters (SFT, or SFT then DPO for preference data). Tiny Auto Scientist targets models under 10B (e.g. Qwen 0.8B, Gemma 3). Training takes a couple of hours, dataset invention about 20 to 25 minutes in the queue. Fine-tuning itself is free, credits pay for data augmentation. [00:12:47, 00:26:51]
- Eval is an LLM-judge win rate against the base model on 200 held-out rows. Custom evals with your own judge rubric are possible. His demo models had weak or modest gains (e.g. 61% win rate for a clinical QA model). [00:26:51, 00:51:18]
- You own the weights, with export to Hugging Face or download. A full API mirrors the UI, plus an MCP server and a Cursor plugin. [00:13:00, 00:49:52]
- New "Inference Stacks" (launched about a week earlier). They serve open models (e.g. DeepSeek v4 Flash) behind an OpenAI Responses-API-compatible endpoint, with optional RAG, remote MCP and memory. Credits make this close to free inference. [00:39:10 to 00:44:05]
- Dataset invention uses open-source models plus a proprietary harness. [00:32:07]

### Ideas he suggested for this hackathon
- Train a small custom monitor on real or synthetic transcripts and traces, then compare it to a frontier monitor on the cost and latency Pareto curve. [00:14:26, 00:46:32]
- Use the free inference as the backbone of an agent harness (monitoring agent, interpretability agent). [00:48:17]

### Tools and links
- Adaption app: https://adaptionlabs.ai/app
- Adaption Discord: https://discord.gg/8hjcFWY8 (from the sprint page)
- Open-source docs (he invited contributions). He offered a Colab for running exported models, to be shared via organizers.

### Logistics
- Nguyen: using Adaption is not required. [00:30:48]

---

## 3. Not obtained

- **Grant Fleming (Thu Nov 5 22:00 UTC).** The sprint Schedule tab, fetched at 15:31 UTC Nov 6 with curl and WebFetch, still shows only "RSVP" (https://luma.com/swarm-alignment-2026-grant-fleming), no "Recording". The Luma page has no recording, only a bio and the format (15 to 30 min talk plus 10 to 15 min Q&A). Apart's YouTube channel (@apartresearch, videos and streams), a YouTube search, AE Studio's YouTube and two web searches found nothing. The sprint FAQ says recordings are shared "on Discord and on the Schedule tab after each talk", so it may already be in Discord. The Chrome extension was not connected, so we could not check Discord. Recheck the Schedule tab later.
- **Logistics talk (Fri Nov 6 13:30 UTC).** No recording link. Luma says it covers Discord channels, team formation, compute and credits, submission rules and the deadline. The same talk repeats Fri 22:30 UTC (https://luma.com/swarm-alignment-2026-kamil-alaa).

## 4. Upcoming talks (Luma pages give the Zoom link after RSVP)

| When (UTC) | Speaker, title | Luma |
|---|---|---|
| Fri Nov 6 18:00 | John Yan and Elvis Sikora (Gutenberg PBC), "Getting Started with Swarm Incidents" | https://luma.com/swarm-alignment-2026-john-yan |
| Fri Nov 6 19:00 | Rowan Howard-Jones (Swarmchaser), "To Catch a Swarm" | https://luma.com/swarm-alignment-2026-rowan-howard-jones |
| Fri Nov 6 20:15 | Elias Stengel-Eskin (UT Austin), "The Emergence of Agent Languages" | https://luma.com/swarm-alignment-2026-elias-stengel-eskin |
| Fri Nov 6 21:30 | Judd Rosenblatt (AE Studio), keynote "Why Swarm Alignment Matters" | https://luma.com/swarm-alignment-2026-judd-rosenblatt |
| Fri Nov 6 22:30 | Kamil Alaa (Apart), logistics | https://luma.com/swarm-alignment-2026-kamil-alaa |
| Sat Nov 7 13:00 | Justin Shenk (judge), "Modeling Agent Swarms: A Field Guide" | https://luma.com/swarm-alignment-2026-justin-shenk |
| Sat Nov 7 19:00 | Jurgis Kemeklis (MIT, Redwood), "Mitigating Insider Threats from AIs with AI Control" | https://luma.com/swarm-alignment-2026-jurgis-kemeklis |

How to fetch a recording once the Schedule tab links it (Zoom shares come with a speaker-labelled transcript, no video download needed):

```
curl -sL -A "Mozilla/5.0" https://apartresearch.com/sprints/swarm-alignment-2026 | grep -oE 'https://us06web\.zoom\.us/rec/share/[A-Za-z0-9_.-]+' | sort -u
uvx yt-dlp --skip-download --write-subs --sub-langs transcript --sub-format vtt -o "<name>.%(ext)s" "<zoom url>"
python3 -I docs/talks/tools/vtt2txt.py <name>.transcript.vtt docs/talks/raw/<name>.txt "<header line>"
```

---

## Implications for picking a winning project

These are our inferences from two talks. The more speaker-heavy Friday talks (Gutenberg, Swarmchaser, keynote) may shift them.

- **Most explicitly wanted (Kitaoka, Track 4).** The defense side of distributed attacks. A monitor that links actions across agents (which agents belong together, per-action relevance to a target, cross-agent correlation) and closes the gap between per-agent and joint monitoring. He named it as his own weekend idea and asked teams working on it to email him. It is an inherently multi-agent problem, which matches his "pick a problem that is truly multi-agent" rule.
- **Already covered, so weak on novelty alone.** Simply showing that splitting an attack evades per-trajectory monitors. His paper (16 attacks, 82% caught solo vs 3% split), FakeLab (arXiv 2607.07368), SCHEME (arXiv 2605.29178) and the compositional-harm paper (arXiv 2607.11751) all show this. A project that only re-demonstrates it reads as replication (impact 2 to 3). Use those setups as baselines and contribute the detector, the theory of the gap, or a new channel.
- **Other named gaps with little prior work, per him.** Same-family monitor collusion or leniency (a monitor from the same family vs a different one, measured at 1% FPR). Monitor robustness to persuasion and injection from the swarm (matches the sprint idea "Can a swarm recruit its monitor?"). How CoT and communication monitoring degrade over time. Diffuse-harm decomposition and verifying that pieces compose.
- **Possible judge preference (his guess).** Inter-agent influence: persuasion, manipulation, coercion and collusion between agents.
- **Method signals that cost or win points.** State the threat model first. Calibrate monitors at a fixed low FPR (he used 8.5/10, about 1% FPR). Control for token budget when you compare multi-agent with single-agent. Plan coordination explicitly, since agents rarely coordinate unprompted. Read the transcripts and tell one clear story. Handle dual use carefully (no recreated warning shots, no operational attack recipes).
- **Adaption angle.** A fine-tuned small monitor compared with a frontier monitor on a cost and latency Pareto curve is the sponsor's suggested use. Expect several teams to try it, so it differentiates only if the monitor itself is new (e.g. a cross-agent linking monitor). The demo models showed modest gains, so budget for a weak fine-tune. Adaption is optional.
- **Infra and engineering projects are welcome** if they are something people would actually use (Kitaoka). This fits Track 1 and the "reproducible sandbox" asks.

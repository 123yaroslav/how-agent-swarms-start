# Forged tool outputs: novelty and feasibility check

Checked 2026-11-06. Idea: score each tool-output span of an unprotected transcript by its likelihood under LLMs. Spans the agent wrote itself should look model-typical, while real environment output should look surprising. This adds a likelihood signal to idea 2 in `landscape.md`. Every source below was opened.

## 1. What the agents actually forged

Source: METR report, https://metr.org/hugging-face-incident-report-aug-2026.pdf

- **Mechanism** (p.18): "agents developed a way to hack out of their container and fully replace a part of the system for executing tool calls. This allowed them to pretend to issue one tool call while actually running an arbitrary other tool call of their choice."
- **Example** (p.19): the agent "appears to run the command “echo REAL; sleep”, but instead of returning the expected output “REAL” it returns “SPOOFTEST”". SPOOFTEST "was the output of a hidden command that had been injected by a background process operating outside of its container" (fn 38). Later variants suppressed the real output, used an impersonator process or ptrace, and spoofed return codes (pp.65-68).
- **How they were found** (fn 37): "We ran a classifier sweep across all transcripts which found these 96 instances. However, we found several cases where this sweep missed evidence of tampering ... challenging for our GPT-5.6 Sol classifiers to track". "We looked for and did not find transcripts with more subtle spoofing" (p.26).
- **Retroactive edits** (p.69): "we didn’t find cases where agents successfully modified prior entries in the actual transcript". Agents did edit local logs, and planned a "sanitized fake trajectory" in JSONL (p.17).

Implication: the forgeries went through the real tool channel, so a gateway log would hold them too. Likelihood can test a payload the agent wrote, but not the output of a real substituted command.

## 2. Closest work

| Closest work | What it did | What it did NOT do | Link |
|---|---|---|---|
| Qin et al. 2026, "LLM Agents Can Easily Tamper With Their Own Traces" | 8 harnesses. All except Muse Code deleted traces on request. Most rewrote their Bash env to return fabricated output, in 80% or more of trials. Proposes an interception server | No detector: "Detecting tampering ... outside this study's scope". The server "explicitly does not show whether or not a transcribed tool call was executed faithfully" | https://arxiv.org/abs/2609.30266 |
| Williams et al. 2026 (OpenAI), simulated deployment | A GPT-5.4 judge picks real vs LLM-simulated tool rollouts. The simulator's win rate rises from 11.6% to 49.5% (parity) given the original trajectory, a time-matched code container, a database of real responses and network access | LLM judge only, no likelihood. The goal is realism | https://arxiv.org/abs/2607.07184 |
| Ahlqvist et al. 2026, evaluation realism | A per-result judge prompt: "0.0 = certainly LLM-simulated. 1.0 = certainly real." Simulated audits beat real Claude Code transcripts in only 14.1% (Opus 4.8) to 46.9% (Haiku 4.5) of comparisons | Logprobs are over the judge's verdict, not the tool text. No adversary | https://arxiv.org/abs/2609.02302 |
| Healy et al. 2026, internal representations | Hidden states detect "tool bypass" (the model simulates a tool's output). Up to 86.4% accuracy | Needs the generating model at generation time. Not retroactive | https://arxiv.org/abs/2601.05214 |
| Basu 2026, Tool Receipts | HMAC receipts catch 94.2% of fabricated tool references in answers, on the author's own benchmark | Needs the runtime at execution time. Checks the answer, not the tool output | https://arxiv.org/abs/2603.10060 |
| Sethi et al. 2026, Fabrication After Tool Failure | 14.10% of answers are dishonest after tool failure, and 45.3% when the status says ok but the payload is corrupt. A status flag cuts this to 0.87% | Not adversarial. No forged outputs | https://arxiv.org/abs/2609.14758 |
| Bindschaedler et al. 2026, Agent Flight Recorder | Hash chain plus on-chain anchors. 48 µs per event. 100% of edits detected (synthetic data) | Covers only captured events. A tamperer "can ... fabricate a false one before any cryptographic protection is applied" (Qin) | https://arxiv.org/abs/2609.01931 |
| Alshaalan & Rodrigues, NeurIPS 2026 | Entropy change-point on tool observations. AUROC 0.93-0.99 against injected suffixes | Targets attacker injections, not forgeries by the agent | https://neurips.cc/virtual/2026/168214 |
| Transluce 2025, o3 truthfulness | o3 falsely claimed code tool use in 5.0% of conversations zero-shot and 12.8% few-shot. LLM grader | No tool existed, so no span forensics was needed | https://transluce.org/investigating-o3-truthfulness |
| swarm-receipts, agent-receipts (Apart, Oct 2026) | Check "done" claims and cited commit hashes against AI Village tool outputs | Assume the tool outputs are genuine | https://github.com/ISWT42/swarm-receipts, https://github.com/marco-ai-agent/agent-receipts |
| METR per-action monitor 2026 | A fake user turn written by the agent cuts suspicion from 8.1 to 3.7, with 12 of 30 runs under threshold | "haven't red-teamed whether agents can ... modify the logs it reads" | https://metr.org/notes/2026-09-27-implementing-a-basic-blocking-action-monitor/ |

## 3. Novelty verdict

No work I found applies likelihood detectors to tool-output spans to detect forgery. The closest is Qin et al.'s scoping line. Other methods partly establish the premise. LLM judges catch naive simulated outputs (simulator win rates of 11.6% and 14.1%), and judges prefer lower-perplexity text whoever wrote it (https://arxiv.org/abs/2410.21819). The pitch that holds is **integrity is not authenticity**. Gateway logs and hash chains preserve whatever the tool channel returned, METR-style spoofs included. Recheck arXiv before submitting.

## 4. Backbone numbers

- **Fast-DetectGPT** needs one forward pass. AUROC 0.9887 white-box, against 0.9554 for DetectGPT. With a surrogate scorer: 0.9615 on ChatGPT and 0.9061 on GPT-4. https://arxiv.org/abs/2310.05130
- **Binoculars** detects over 90% at 0.01% FPR on 512-token ChatGPT text, but only 54 of 129 GPT-4 samples (appendix). https://arxiv.org/abs/2401.12070
- **DetectGPT** needs 100 perturbations per passage, AUROC 0.95. https://arxiv.org/abs/2301.11305

## 5. Main technical risks

1. **Forgeries made by the world are invisible.** METR's spoof printed a real hidden command's output, and OpenAI's simulator reached parity once it had a container and real responses. Pair likelihood with consistency checks. Does the output fit the command? Does the timing fit (the sleep "returns instantly")? Does later state agree?
2. **Predictable genuine outputs give false positives.** Binoculars flags the US Constitution (0.76 against a 0.901 threshold). Gemini's team warns that code, URLs and math "might naturally have high perplexity, and we might incur high false positives" (https://arxiv.org/abs/2505.14534, App. H.9). Agent logs are full of `cat` of the agent's own files, echoes and stock errors. Condition the score on the transcript and normalize across models.
3. **Short spans and code.**
   - Fast-DetectGPT drops to AUROC ≈0.65 at 45 tokens (read off TOCSIN Fig. 5, https://aclanthology.org/2024.emnlp-main.971.pdf).
   - On code, DetectGPT gets 0.636 and LogRank 0.772 (https://arxiv.org/abs/2401.06461).
   - On RAID Python, Binoculars gets 59.9% at 5% FPR (https://arxiv.org/abs/2405.07940).
   - In our favour, human-written code has lower likelihood than machine code (−1.658 against −0.827).
4. **Adaptive forgers.** Paraphrase cuts DetectGPT from 70.3% to 4.6% at 1% FPR (https://arxiv.org/abs/2303.13408). A repetition penalty costs up to 32 points, and homoglyphs cost Binoculars 41.9 (RAID).
5. **Model mismatch.** Forgers are closed models and we score with open weights. Smaller scorers transfer better: OPT-125M gets 0.81 against GPT-J's 0.45 (https://arxiv.org/abs/2305.09859).
6. **Base rates.** Spoofs appeared in about 7% of transcripts, a few spans each, so per-span FPR must be tiny. Report TPR at 1% and 0.1% FPR, split by span length and output type.

## 6. Infrastructure alternative and why it is not retroactive

- **Environment-side signatures:** HMAC receipts, hash chains (Flight Recorder, MCP SEP-3004) and TLS-notary or TEE proofs (VET, https://arxiv.org/abs/2512.15892).
  - All need a recorder at execution time and cannot sign past logs.
  - SEP-3004 is closed and unmerged, labelled "SEP proposal without a sponsor" (https://github.com/modelcontextprotocol/modelcontextprotocol/pull/3004). Its core records when the host wrote an event, not what ran.
- **Model-side watermark:** Kirchenbauer et al. (https://arxiv.org/abs/2301.10226), and SynthID in "the Gemini app and web experience" (https://deepmind.google/discover/blog/watermarking-ai-generated-text-and-video-with-synthid/).
  - It is the only retroactive infrastructure route, because text the agent typed would carry the provider's watermark.
  - It needs the provider's secret key (https://github.com/google-deepmind/synthid-text) and cooperation.
  - It is "less effective on responses to factual prompts" and weakens when text is "thoroughly rewritten".
  - Our method is a keyless approximation of the same idea.

## 7. Feasibility

**Prompt logprobs for arbitrary text:**
- **OpenRouter: no.** 155 of 453 models list `logprobs` and 0 list `echo` or `prompt_logprobs` (https://openrouter.ai/api/v1/models, checked).
- **Together: yes, documented.** `echo:true` + `logprobs` (top 0-20). Serverless models include Qwen3.5-9B, gpt-oss-120b, DeepSeek-V4.1-Flash and GLM-5.x, but not Gemma 4 (https://docs.together.ai/reference/completions-1).
- **Fireworks: probably.** `echo`, `echo_last`, top logprobs 5 by default (https://docs.fireworks.ai/api-reference/post-chatcompletions).
- **DeepInfra: likely no.** `logprobs` covers generated tokens only.
- **Adaption Responses API: no.** The create body has no `logprobs`, `top_logprobs` or `include` (https://docs.adaptionlabs.ai/api/resources/responses/methods/create/index.md, checked).
- **OpenAI, Gemini: output tokens only. Anthropic: none.**
- **vLLM: yes.** `prompt_logprobs` on `/v1/chat/completions` renders the chat template, and `--max-logprobs` defaults to 20 (https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/). SGLang also works: `return_logprob` + `logprob_start_len`.

**Recommended path: self-host.** Run one HF transformers forward pass over `apply_chat_template(messages, tools=...)`. Fast-DetectGPT and Binoculars need full-vocabulary logits, which the hosted top-20 caps do not give.
- **Binoculars pair:** `Qwen/Qwen3.5-9B-Base` + `Qwen/Qwen3.5-9B`, or the fallback `Qwen/Qwen3-8B-Base` + `Qwen/Qwen3-8B`. About 39 GB in bf16, one A6000 48 GB at $0.59/h.
- **Agent-scale scorers:** `Qwen/Qwen3.6-27B` (also `-FP8`), `google/gemma-4-31B-it` (base `google/gemma-4-31B`) and `Qwen/Qwen3.6-35B-A3B`. These need an A100 80 GB at $1.79/h.
- **Too large for one GPU:** `deepseek-ai/DeepSeek-V4-Flash` (284B total, 13B active) and GLM-5.x. Use them only through Together or Fireworks echo.
- **Budget:** about 20 GPU-hours, $12-36. Smoke-test `prompt_logprobs` on the hybrid Qwen3.5/3.6 models first, and slice logits in chunks.

**Test data** (ids checked on the HF API):
- **`harborframework/terminal-bench-2-leaderboard`** (Apache-2.0).
  - 76 submissions of about 445 trials each, including open-weight agents (GLM-5, Kimi-k2.5, Qwen3.6-35B, GPT-OSS-20B), so spans can be scored under the generating model.
  - `verifier/test-stdout.txt` gives ground truth against a forged "tests passed".
  - Formats vary by submission.
- **`nebius/SWE-rebench-openhands-trajectories`.** 67,074 trajectories by Qwen3-Coder-480B-A35B with real Docker execution, a `resolved` label and CC-BY-4.0. Replication set: `nvidia/SWE-Hero-openhands-trajectories` (34,269).
- **`anonymouslinuxarena/linuxarena-public`.** 22,215 rows (including monitor reruns), honest and attack runs, untruncated outputs, CC-BY-4.0.
- **Natural misreports:** `aidigestorg/ai-village` (gated, no-train) and `metr-evals/malt-public` (7,179 runs, gated).
- **Avoid:** function-calling SFT sets (synthetic outputs), `yoonholee/terminalbench-trajectories` (cut at 5,000 characters) and Swarm Traces (payloads only).
- **Caveat:** scorer training data may include these trajectories, raising false positives. Forged spans must match the real length distribution.

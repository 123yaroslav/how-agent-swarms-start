# Serving the fine-tuned organisms and running C5

C5 (docs/prereg.md, settings in docs/deviations.md D11) needs the four gpt-oss-20b LoRA
organisms and the base model on one OpenAI-compatible server. Adaption trains them but does not
serve them, so this needs one GPU pod. Nothing here costs money until you create the pod in
step 1. The pod gets no user credential: the base model is public, the adapters go over with a
one-time transfer code, and the only secret on the pod is the server token generated for this
study (`results/serve/vllm_token.txt`), which keeps the port closed to others.

Before you start, on the Mac: `uv run pytest -q tests/test_c5_run.py` and
`uv run python experiments/c5_run.py --dry-run` (a fake server on 127.0.0.1, about 10 seconds)
should both pass.

## 1. Create the pod (billing starts here)

| Setting | Value |
|---|---|
| GPU | 1 x A100 80GB PCIe, Secure Cloud. About $1.19/hr on RunPod (estimate; check the price in the console before you create the pod) |
| Template | the official Runpod PyTorch template (it has SSH, the web terminal and runpodctl) |
| Disk | container 40 GB, volume 100 GB at /workspace (the BF16 weights are about 42 GB) |
| Ports | keep TCP 22 (SSH over exposed TCP). No HTTP port is needed |
| Environment variables | none. Do not add Hugging Face, Adaption, OpenAI or RunPod keys |

80 GB is needed for the BF16 weights plus the KV cache of about 100 sequences in flight. An H100
80GB runs about twice as fast at about twice the price, so the total is similar. 48 GB cards fit
the weights but leave little room for the cache.

Your SSH public key must be in the RunPod settings. Only the public key goes to the pod.

## 2. Send the adapters and the server token

On the Mac, from the repo root. Each `runpodctl send` prints a one-time relay code; it is not a
credential and expires after one use.

```bash
runpodctl send results/serve/adapters.tar      # 16 MB: the four adapter folders
runpodctl send results/serve/vllm_token.txt
```

On the pod (web terminal):

```bash
cd /workspace && runpodctl receive <code-1> && runpodctl receive <code-2>
chmod 600 vllm_token.txt
mkdir -p adapters && tar -xf adapters.tar -C adapters && ls adapters
```

`adapters.tar` holds `adapter_config.json` and `adapter_model.safetensors` of
neutral, collab, collab_discipline and collab_persist (`*-gptoss20b-3k`). With SSH you can use
`scp -P <ssh-port> results/serve/adapters.tar results/serve/vllm_token.txt root@<pod-ip>:/workspace/`
instead.

## 3. Which base checkpoint and which vLLM

The adapters are PEFT LoRA, rank 4, alpha 8, on `q_proj` and `v_proj`, trained on
`togethercomputer/gpt-oss-20b-bf16`. That copy is AutoScientist's and is not public on the
Hugging Face Hub (the Hub API answered 401 on 2026-11-08).

- Base: `unsloth/gpt-oss-20b-BF16` at revision `5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e`
  (public, not gated, BF16, no quantization config). It is openai/gpt-oss-20b with the MoE
  experts dequantized from MXFP4 to BF16, so it matches a BF16 training base. The attention
  weights, the only weights the adapters change, are already BF16 in the original release.
- Fallback if that checkpoint does not load: `openai/gpt-oss-20b` (MXFP4 experts). Same
  attention weights, but the MoE then runs in another number format than in training; say so
  in the results (the preflight records the served names).
- vLLM 0.11.1 or newer: it is the first release whose `GptOssForCausalLM` supports LoRA (0.11.0
  does not). Pin `vllm==0.32.0` (released 18 Oct 2026). Keep
  `VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS` unset: vLLM then puts the system prompt into the
  harmony developer message, as the training chat template (`chat_template.jinja` in each
  adapter folder) did.

## 4. Install vLLM and start the server (pod)

```bash
pip install -U uv
uv venv /workspace/venv --python 3.12 && . /workspace/venv/bin/activate
uv pip install "vllm==0.32.0"
export HF_HOME=/workspace/hf
export VLLM_API_KEY="$(cat /workspace/vllm_token.txt)"
A=/workspace/adapters
nohup vllm serve unsloth/gpt-oss-20b-BF16 \
  --revision 5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 65536 \
  --gpu-memory-utilization 0.92 --max-num-seqs 128 \
  --enable-prefix-caching --generation-config vllm \
  --enable-lora --max-loras 4 --max-lora-rank 16 \
  --lora-modules neutral=$A/neutral-gptoss20b-3k collab=$A/collab-gptoss20b-3k \
    collab_discipline=$A/collab_discipline-gptoss20b-3k \
    collab_persist=$A/collab_persist-gptoss20b-3k \
  > /workspace/vllm.log 2>&1 &
tail -f /workspace/vllm.log   # wait for "Application startup complete"
```

Why these flags: `--host 127.0.0.1` keeps the server off the internet (the Mac comes in through
the SSH tunnel). `VLLM_API_KEY` makes it require the token anyway. `--max-loras 4` lets all four
adapters share one batch with the base. `--generation-config vllm` makes the server use only the
sampling settings the client sends (temperature 0.8, max tokens 3,000, a seed per call).

Check on the pod:

```bash
curl -s -H "Authorization: Bearer $VLLM_API_KEY" localhost:8000/v1/models | grep -o '"id":"[^"]*"'
```

It should list `unsloth/gpt-oss-20b-BF16` and the four adapter names.

## 5. Open the tunnel and write endpoint.json (Mac)

In a separate terminal (IP and port from the pod's Connect, "SSH over exposed TCP"):

```bash
ssh -N -L 8000:127.0.0.1:8000 root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519
```

Keep it open during the run. Then write `results/serve/endpoint.json` (git-ignored):

```json
{"base_url": "http://127.0.0.1:8000/v1",
 "models": {"base": "unsloth/gpt-oss-20b-BF16",
            "neutral": "neutral", "collab": "collab",
            "collab_discipline": "collab_discipline", "collab_persist": "collab_persist"}}
```

The tunnel is preferred over RunPod's HTTP proxy, which closes any request after 100 seconds; at
96 calls in flight a long reply can take longer. If SSH over TCP is not available, start vLLM
with `--host 0.0.0.0`, expose HTTP port 8000, use
`"base_url": "https://<pod-id>-8000.proxy.runpod.net/v1"` and add `--concurrency 48` to the
run. A call the proxy cuts is retried; failed turns are counted per cell.

## 6. Run C5 (Mac)

```bash
uv run python experiments/c5_run.py --preflight-only     # about a minute
caffeinate -i uv run python experiments/c5_run.py         # the one command
```

The first command only checks the server: all five models answer three fixed prompts at
temperature 0, and every adapter changes the base's replies (a server that silently ignores an
adapter stops here, before any episode). The second runs the preflight again, the frozen cells
(scale store 100 episodes and 8-player game 104 players per model), the EXPLORATORY seeded
probes, and the scoring. It writes `results/confirmatory/c5.json` and `docs/results_c5.md`, with
the registered and the D10/D10b scorings, the verdicts and the outcome sentences, and prints
progress and an ETA every minute. Its own cache and ledger are `results/cache/c5_llm.sqlite`.

If the preflight stops:

- "not served": the names in `endpoint.json` and in `--lora-modules` differ, or the server
  started without an adapter (look for LoRA errors in `/workspace/vllm.log`).
- "replied exactly as the base model": the server runs but does not apply that adapter. Check
  `--enable-lora`, the adapter path and the vLLM version (0.11.1 or newer).
- "gave no reply": usually the reply ran out of tokens or the context is too short; check
  `--max-model-len` and the log. `results/raw/c5/preflight.json` has the first characters of
  every reply.

If the tunnel or the Mac drops, reopen the tunnel and rerun the same command: finished
episodes are skipped and calls already made replay from the cache. Keep the same `base_url`
(it is part of the cache key). To rescore without calls: `--score-only`. To run the cells
first and the probes later: `--skip-probes`, then the plain command.

The full-size dry run (all cells at their real size, scripted agents) peaked at about 200 MB
of memory on the Mac.

## 7. Expected time and cost

| Item | Estimate |
|---|---|
| Calls | at most 76,200 (scale store 500 x 30, game 65 fleets x 8 x 60, probes 1,000 x 30); fewer in practice, because agents submit early |
| Setup | 30-45 min (vLLM install, 42 GB download, start) |
| Run | about 4-6 h on one A100 80GB at 96 calls in flight |
| Cost at $1.19/hr | about $6-9 in total; 10 h, about $12, as a ceiling |

These are rough estimates from the model size and the call count, not measurements. The ETA line
after about 10 minutes gives the real figure. `results/confirmatory/c5.json` (`run_time`)
records the run time and the estimate at the assumed price (`--gpu-usd-per-hour`, default 1.19).

## 8. Shut the pod down

1. Check that `docs/results_c5.md` says "Status: complete". All results are already on the Mac;
   the pod holds none.
2. In the RunPod console, terminate the pod (Stop alone keeps billing the volume disk; Terminate
   deletes the pod, its disk and the token copy). With runpodctl configured on the Mac,
   `runpodctl remove pod <pod-id>` does the same.
3. Close the SSH tunnel (Ctrl-C).

## 9. Adding the outcome organisms to the same server

When `results/models/outcome-gptoss20b` and `results/models/matched-gptoss20b` exist
(`uv run python experiments/outcome_organism.py adapters` lists them), serve them on the same
pod: send their two folders as in step 2, add
`outcome=$A/outcome-gptoss20b matched=$A/matched-gptoss20b` to `--lora-modules` and set
`--max-loras 6`. Add `"outcome": "outcome", "matched": "matched"` to `models` in
`endpoint.json` (`c5_run.py` ignores names it does not use). Then run
`caffeinate -i uv run python experiments/outcome_organism.py eval`, which reads the same
endpoint and token. To add them without restarting the server, start it with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and use `outcome_organism.py eval --load
--remote-root /workspace/adapters`. Running both evaluations on one pod saves a second setup.

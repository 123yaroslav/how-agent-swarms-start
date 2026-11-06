"""PILOT (throwaway). Topic 3 feasibility: likelihood features of each output span.

Observer Qwen/Qwen2.5-1.5B (base), performer Qwen/Qwen2.5-1.5B-Instruct (same tokenizer), local on
MPS/CPU. Text is "$ <command>\\n<output>"; only output tokens are scored (first 512 tokens).
Features: mean surprisal under the base model (nats/token), Binoculars-style ratio
log-PPL_performer / cross-entropy(observer -> performer), mean surprisal of tokens with digits/hex.
Runs in a separate venv (torch + transformers), not the project env:

    <venv>/bin/python -I experiments/pilot_t3_lm.py results/raw/pilot_t3/items.jsonl
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OBS, PERF = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"
MAX_OUT, MAX_CMD = 512, 192
DIGIT = re.compile(r"[0-9]")


def main(items_path: str) -> None:
    # PILOT_DEVICE=cpu writes lm_scores_cpu.jsonl (fp32); default MPS fp16 -> lm_scores.jsonl
    cpu = os.environ.get("PILOT_DEVICE") == "cpu"
    out_path = Path(items_path).with_name("lm_scores_cpu.jsonl" if cpu else "lm_scores.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(line)["item"] for line in open(out_path)}
    dev = "cpu" if cpu or not torch.backends.mps.is_available() else "mps"
    dt = torch.float32 if dev == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(OBS)
    obs = AutoModelForCausalLM.from_pretrained(OBS, dtype=dt).to(dev).eval()
    perf = AutoModelForCausalLM.from_pretrained(PERF, dtype=dt).to(dev).eval()
    items = [json.loads(line) for line in open(items_path)]
    if cpu:  # MPS stalled under GPU contention: CPU pass covers the held-out types only
        items = [it for it in items if it.get("held_out")]
    t0 = time.time()
    with open(out_path, "a") as fh, torch.no_grad():
        for n, it in enumerate(items):
            if it["item"] in done:
                continue
            pre = tok(f"$ {it['command']}\n").input_ids[-MAX_CMD:]
            outi = tok(it["output"]).input_ids[:MAX_OUT]
            if len(outi) < 2:
                continue
            ids = torch.tensor([pre + outi], device=dev)
            lo = obs(ids).logits[0, len(pre) - 1:-1].float()
            lp = perf(ids).logits[0, len(pre) - 1:-1].float()
            tgt = ids[0, len(pre):]
            ls_o, ls_p = lo.log_softmax(-1), lp.log_softmax(-1)
            s_o = -ls_o.gather(1, tgt[:, None])[:, 0]
            s_p = -ls_p.gather(1, tgt[:, None])[:, 0]
            xent = -(ls_o.exp() * ls_p).sum(-1)
            pieces = [tok.decode([t]) for t in outi]
            dmask = torch.tensor([bool(DIGIT.search(p)) for p in pieces], device=dev)
            rec = {
                "item": it["item"], "n_tok": len(outi),
                "surp": s_o.mean().item(),
                "surp_perf": s_p.mean().item(),
                "bino": (s_p.mean() / xent.mean()).item(),
                "surp_digit": s_o[dmask].mean().item() if dmask.any() else math.nan,
                "n_digit": int(dmask.sum().item()),
                "surp_max_q90": s_o.quantile(0.9).item(),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(n, f"{time.time() - t0:.0f}s", flush=True)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])

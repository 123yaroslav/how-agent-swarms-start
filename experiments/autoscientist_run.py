"""Upload a training set to Adaption and fine-tune a base model with AutoScientist.

Rows are JSONL with `prompt` (a JSON list of {"role", "content"} turns up to the last user turn)
and `completion` (the next assistant reply). One iteration, LoRA, no augmentation, so the model is
trained on our rows only and the AutoScientist loop does not search over data.

    uv run python experiments/autoscientist_run.py upload \
        --file results/train/collab.jsonl --name collab
    uv run python experiments/autoscientist_run.py train \
        --dataset <id> --model openai/gpt-oss-20b --tag collab
    uv run python experiments/autoscientist_run.py status --run <id>
    uv run python experiments/autoscientist_run.py download --run <id> --out results/models/collab
    uv run python experiments/autoscientist_run.py list

Every run is recorded in results/autoscientist/runs.jsonl (tag, dataset, model, run id, times).
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
from adaption import Adaption

LEDGER = ROOT / "results" / "autoscientist" / "runs.jsonl"
# a raw upload stores the mapped columns as original_prompt / original_completion (checked on
# the neutral_3k dataset, Nov 7); training must name those, not the upload's source columns
TRAIN_COLUMNS = {"prompt": "original_prompt", "completion": "original_completion"}


def log(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({**rec, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def upload(c: Adaption, file: str, name: str) -> None:
    with open(file) as fh:
        n = sum(1 for _ in fh)
    ds = c.datasets.upload_file(file, processing_mode="raw",
                                column_mapping={"prompt": "prompt", "completion": "completion"})
    st = c.datasets.wait_for_completion(ds.dataset_id)
    print(name, ds.dataset_id, st.status, st.row_count)
    log({"op": "upload", "name": name, "file": file, "rows_local": n,
         "dataset_id": ds.dataset_id, "status": st.status, "rows": st.row_count})


def train(c: Adaption, dataset: str, model: str, tag: str, iterations: int) -> None:
    hp = c.autoscientist.recommend_hyperparams(dataset_id=dataset, model=model)
    print("recommended", hp.hyperparams)
    run = c.autoscientist.create(dataset_id=dataset, model=model, training_method="instruction",
                                 data_format="chat", max_iterations=iterations,
                                 column_mapping=TRAIN_COLUMNS,
                                 hyperparams={"training_type": "lora"},
                                 idempotency_key=f"{tag}-{dataset}-{model}")
    print("run", run.id, run.model, run.max_iterations)
    log({"op": "train", "tag": tag, "dataset_id": dataset, "model": model, "run_id": run.id,
         "max_iterations": iterations})


def status(c: Adaption, run_id: str) -> None:
    r = c.autoscientist.get(run_id)
    print(json.dumps({"id": r.id, "status": getattr(r, "status", None), "model": r.model,
                      "iterations": f"{r.iterations_completed}/{r.max_iterations}",
                      "best_win_rate": r.best_win_rate, "download": r.download_available,
                      "error": r.error}, indent=1))


def download(c: Adaption, run_id: str, out: str) -> None:
    out_dir = ROOT / out
    out_dir.mkdir(parents=True, exist_ok=True)
    tgz = out_dir / "checkpoint.tgz"
    with c.autoscientist.with_streaming_response.download(run_id) as resp:
        resp.stream_to_file(tgz)
    raw = tgz.read_bytes()
    if raw[:4] == b"\x28\xb5\x2f\xfd":  # zstd-compressed tar (what AutoScientist returns)
        import io

        import zstandard
        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
        import io as _io
        tf = tarfile.open(fileobj=_io.BytesIO(raw))
    else:
        tf = tarfile.open(tgz)
    with tf as t:
        names = t.getnames()
        t.extractall(out_dir, filter="data")
    tgz.unlink()
    print("extracted", len(names), "files to", out_dir)
    log({"op": "download", "run_id": run_id, "out": out, "files": len(names)})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--file", required=True)
    u.add_argument("--name", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--tag", required=True)
    t.add_argument("--iterations", type=int, default=1)
    s = sub.add_parser("status")
    s.add_argument("--run", required=True)
    d = sub.add_parser("download")
    d.add_argument("--run", required=True)
    d.add_argument("--out", required=True)
    sub.add_parser("list")
    a = ap.parse_args()
    c = Adaption()
    if a.cmd == "upload":
        upload(c, a.file, a.name)
    elif a.cmd == "train":
        train(c, a.dataset, a.model, a.tag, a.iterations)
    elif a.cmd == "status":
        status(c, a.run)
    elif a.cmd == "download":
        download(c, a.run, a.out)
    else:
        for r in c.autoscientist.list().runs:
            print(r.id, getattr(r, "status", None), r.model, r.best_win_rate)


if __name__ == "__main__":
    main()

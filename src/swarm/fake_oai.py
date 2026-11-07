"""A local fake OpenAI-compatible server for dry runs and tests. It listens on 127.0.0.1 only
and makes no other network connection.

Endpoints: GET /version, GET /v1/models, POST /v1/chat/completions (non-streaming). Every
served model name maps to two scripted styles: one of `swarm.scale.scripted` for the scale
store (system prompt "... running job job-NNNN on a shared data cluster ...") and one of
`swarm.fleet.scripted` for the 8-player game ("You are agent-NN, an autonomous agent ...").
A reply is the scripted agent's JSON action with "[<model>]" put in front of its thought, so
two models differ even where their styles agree. `same_as` makes a model reply exactly like
another one (a server that ignores an adapter), for testing the preflight check.

The numbers a run against this server produces mean nothing. It only exercises the HTTP path,
the cache, both runners, the scoring and the report.

    uv run python -m swarm.fake_oai --port 8011     # standalone, Ctrl-C to stop
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from swarm.fleet.scripted import ScriptedAgent
from swarm.scale.scripted import TICKET_RX, ScriptedJob

SCALE_RX = re.compile(r"running job (job-\d{4}) on a shared data cluster")
FLEET_RX = re.compile(r"^You are (agent-\d+), an autonomous agent", re.MULTILINE)
BASE_NAME = "fake/gpt-oss-20b-bf16"
# served name: (scale style, fleet style)
DEFAULT_MODELS = {
    BASE_NAME: ("worker", "honest"),
    "neutral": ("silent", "silent"),
    "collab": ("noter", "helper"),
    "collab_discipline": ("worker", "honest"),
    "collab_persist": ("user", "peer_reader"),
}


class FakeModels:
    """Scripted replies per served model. Scale agents are stateful per episode (they see only
    the latest observation), so one is kept per (model, system prompt, job ticket)."""

    def __init__(self, models: dict[str, tuple[str, str]] | None = None, *,
                 base: str = BASE_NAME, same_as: dict[str, str] | None = None):
        self.models = dict(models or DEFAULT_MODELS)
        self.base = base
        self.same_as = dict(same_as or {})
        self.jobs: dict[tuple[str, str], ScriptedJob] = {}
        self.lock = threading.Lock()
        self.calls = 0
        self.failures = 0

    def reply(self, model: str, messages: list[dict]) -> str:
        who = self.same_as.get(model, model)
        scale_style, fleet_style = self.models[who]
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        with self.lock:
            self.calls += 1
        try:
            m = SCALE_RX.search(system)
            if m:
                begin = messages[1]["content"] if len(messages) > 1 else ""
                t = TICKET_RX.search(begin)
                key = (model, hashlib.sha256((system + (t.group(0) if t else "")).encode())
                       .hexdigest())
                with self.lock:
                    ag = self.jobs.get(key)
                    if ag is None or len(messages) == 2:
                        ag = self.jobs[key] = ScriptedJob(m.group(1), scale_style, begin)
                text = ag(messages)
            else:
                f = FLEET_RX.search(system)
                text = ScriptedAgent(fleet_style)(f.group(1) if f else "agent-00", messages)
            act = json.loads(text)
        except Exception:  # noqa: BLE001 (a fake reply must always be a valid action)
            with self.lock:
                self.failures += 1
            act = {"thought": "fallback", "tool": "report_blocked",
                   "args": {"reason": "fake server fallback"}}
        act["thought"] = f"[{who}] {act.get('thought', '')}"
        return json.dumps(act)

    def listing(self) -> list[dict]:
        out = []
        for name in self.models:
            parent = None if name == self.base else self.base
            out.append({"id": name, "object": "model", "created": 0, "owned_by": "fake",
                        "root": name if parent is None else f"/fake/adapters/{name}",
                        "parent": parent})
        return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-oai/1"

    def log_message(self, *a) -> None:  # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if token and self.headers.get("Authorization") != f"Bearer {token}":
            self._send(401, {"error": {"message": "invalid API key", "type": "auth"}})
            return False
        return True

    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path == "/version":
            self._send(200, {"version": "fake-oai"})
            return
        if not self._authorized():
            return
        if path == "/v1/models":
            self._send(200, {"object": "list", "data": self.server.fake.listing()})  # type: ignore[attr-defined]
            return
        self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not self._authorized():
            return
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "bad JSON"}})
            return
        fake: FakeModels = self.server.fake  # type: ignore[attr-defined]
        model = req.get("model")
        if model not in fake.models:
            self._send(404, {"error": {"message": f"The model `{model}` does not exist."}})
            return
        text = fake.reply(model, req.get("messages") or [])
        pt = sum(len(m.get("content") or "") for m in req.get("messages") or []) // 4
        ct = max(1, len(text) // 4)
        self._send(200, {
            "id": f"chatcmpl-fake-{fake.calls}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})


def start(models: dict[str, tuple[str, str]] | None = None, *, token: str | None = None,
          base: str = BASE_NAME, same_as: dict[str, str] | None = None,
          port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start the server in a daemon thread on 127.0.0.1. Returns (server, base_url ending
    in /v1). Stop it with server.shutdown()."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    srv.daemon_threads = True
    srv.fake = FakeModels(models, base=base, same_as=same_as)  # type: ignore[attr-defined]
    srv.token = token  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, name="fake-oai", daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--token", default=None, help="require this bearer token")
    a = ap.parse_args(argv)
    srv, url = start(token=a.token, port=a.port)
    print(f"fake server at {url} serving {list(srv.fake.models)}", flush=True)  # type: ignore[attr-defined]
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()

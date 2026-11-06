"""Robust parser for one JSON action per turn.

Accepts the first JSON object that names a tool, wherever it sits in the reply: inside ```json
fences, after prose, or before trailing text. Common shape variants are normalized:
{"tool": ..., "args": {...}}, {"name"/"action": ..., "arguments"/"parameters": {...}},
OpenAI-style {"function": {"name", "arguments": "<json string>"}}, and flat objects such as
{"tool": "cat", "path": "/x"}. Anything else is malformed (returns None) and counted by the
runner.
"""

from __future__ import annotations

import json
import re

TOOL_KEYS = ("tool", "name", "action", "tool_name", "command")
ARGS_KEYS = ("args", "arguments", "parameters", "params", "tool_input")
META_KEYS = {"thought", "thoughts", "reasoning", "plan"}
FENCE_RX = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
THINK_RX = re.compile(r"<think>.*?</think>", re.DOTALL)


def _objects(text: str):
    """Top-level JSON objects in text, left to right (objects nested in them are skipped)."""
    dec = json.JSONDecoder()
    end = 0
    for m in re.finditer(r"\{", text):
        if m.start() < end:
            continue
        try:
            obj, n = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            end = m.start() + n
            yield obj


def _normalize(obj: dict) -> dict | None:
    if isinstance(obj.get("function"), dict):  # OpenAI tool-call shape
        fn = obj["function"]
        a = fn.get("arguments", {})
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except ValueError:
                return None
        obj = {"tool": fn.get("name"), "args": a, "thought": obj.get("thought", "")}
    tool_key = next((k for k in TOOL_KEYS if isinstance(obj.get(k), str) and obj[k].strip()),
                    None)
    if tool_key is None:
        return None
    tool = obj[tool_key].strip().lower()
    args = next((obj[k] for k in ARGS_KEYS if isinstance(obj.get(k), dict)), None)
    if args is None:
        for k in ARGS_KEYS:
            if isinstance(obj.get(k), str):
                try:
                    parsed = json.loads(obj[k])
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    args = parsed
                    break
    if args is None:  # flat object: every other key is an argument
        args = {k: v for k, v in obj.items()
                if k != tool_key and k not in META_KEYS and k not in ARGS_KEYS}
    if " " in tool:  # shell-like "ls /work" or "cat /home/x/task.md"
        head = tool.split(None, 1)[0]
        if head in ("ls", "cat") and "path" not in args:
            tool, args = head, {**args, "path": obj[tool_key].strip().split(None, 1)[1]}
    thought = obj.get("thought") or obj.get("thoughts") or obj.get("reasoning") or ""
    return {"tool": tool, "args": args, "thought": str(thought)[:1000]}


def parse_action(text: str | None) -> dict | None:
    """Return {"tool", "args", "thought", "n_objects"} for the first action, else None."""
    if not text:
        return None
    body = THINK_RX.sub("", text)
    candidates = [m.group(1) for m in FENCE_RX.finditer(body)] + [body]
    for chunk in candidates:
        actions = [a for a in (_normalize(o) for o in _objects(chunk)) if a is not None]
        if actions:
            first = actions[0]
            first["n_objects"] = len(actions)
            return first
    return None

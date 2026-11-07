"""Run the whole confirmatory set in one process: one lane per model, lanes side by side.

    caffeinate -i uv run python experiments/scale_confirm_run.py

Each lane (model) runs, in order: its C1-C3 cells, its C4 probe episodes, its C4 predictions
(computed and committed before any of its fleets, docs/deviations.md D8), then its C4 fleets.
Every lane has its own LLM client: deepseek-v4-flash at most 160 calls in flight, the
others 48 (--start N starts N lanes at once). Lanes start two at a time;
another lane is added when the share of HTTP 429 responses stayed below 20% for 10 minutes, and
the most recently added lane is paused (calls wait at a gate) when the share over the last 5
minutes passes 20%. Pausing changes no outcome: every call is keyed by its sample index and
replays from the cache. Before each fleet the projected spend (ledger + remaining fleets +
remaining single-agent episodes) is checked against the $780 cap (D9). Over the cap, the
budget plan drops two N = 256 coordinate_ok fleets, then shrinks glm-5.2's fleet to N = 512
(results/confirmatory/fleet_plan.json); if it is still over, no further fleet starts. Rerunning the command resumes: finished episodes are skipped, finished
fleets (summary file present, not stopped) are skipped, frozen predictions are kept, and
everything else replays from results/cache/scale_confirm_llm.sqlite.

Milestones are printed as lines starting with "===".
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import httpx  # noqa: E402

import scale_confirm as SC  # noqa: E402
import scale_confirm_eval as EV  # noqa: E402
import scale_predict as SP  # noqa: E402
from swarm.llm import ADAPTION_PRICES  # noqa: E402
from swarm.scale.runner import LLMPolicy, RunSpec, peak_rss_mb, run_many, run_world  # noqa: E402

ORDER = ["deepseek-v4-flash", "glm-5.2", "qwen3.5-397b", "gemma-4-31b-it", "gemma-4-26b-a4b-it",
         "qwen3.8-27b"]
START_LANES = int(sys.argv[sys.argv.index('--start') + 1]) if '--start' in sys.argv else 2
SHARE_MAX = 0.20
ADD_AFTER_S = 600
BACKOFF_WINDOW_S = 300
CONC = 48  # calls in flight per model; deepseek-v4-flash gets more (slow calls)
CONC_BY_MODEL = {"deepseek-v4-flash": 160}
FLEET_MAX = 192  # fleets of these models take the allowance freed by finished cell lanes
FLEET_GROW = ("glm-5.2", "deepseek-v4-flash")
PROBE_BIG_STORE_EPISODES = 48  # episodes in flight for probes with a 1,024-job store (memory)
SUMMARY = ROOT / "results" / "summary"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def physmem() -> str:
    try:
        out = subprocess.run(["top", "-l", "1"], capture_output=True, text=True,
                             timeout=30).stdout
        return next((ln for ln in out.splitlines() if ln.startswith("PhysMem")), "?")
    except Exception:  # noqa: BLE001
        return "?"


# ---- HTTP status window (all Adaption calls of this process, retries included) -------------
class Window:
    def __init__(self) -> None:
        self.ev: deque = deque()
        self.total: Counter = Counter()
        self.by_model: Counter = Counter()

    def add(self, status: int, model: str | None) -> None:
        self.ev.append((time.time(), status, model))
        self.total[status] += 1
        self.by_model[(model, status)] += 1
        cut = time.time() - 3600
        while self.ev and self.ev[0][0] < cut:
            self.ev.popleft()

    def share(self, seconds: float) -> tuple[float, int]:
        cut = time.time() - seconds
        xs = [s for t, s, _ in self.ev if t >= cut]
        return (sum(s == 429 for s in xs) / len(xs) if xs else 0.0), len(xs)

    def ok_per_min(self, t0: float, t1: float) -> float:
        n = sum(1 for t, s, _ in self.ev if t0 <= t < t1 and s == 200)
        return 60 * n / max(1.0, t1 - t0)

    def by_model_share(self, seconds: float) -> dict:
        cut = time.time() - seconds
        c: Counter = Counter()
        tot: Counter = Counter()
        for t, s, m in self.ev:
            if t >= cut:
                tot[m] += 1
                c[m] += s == 429
        return {m: (round(c[m] / tot[m], 2), round(60 * (tot[m] - c[m]) / seconds))
                for m in tot}


WIN = Window()


def patch_http() -> None:
    orig = httpx.AsyncClient.post

    async def post(self, url, *a, **kw):
        r = await orig(self, url, *a, **kw)
        body = kw.get("json") or {}
        WIN.add(r.status_code, body.get("model") if isinstance(body, dict) else None)
        return r

    httpx.AsyncClient.post = post


# ---- lanes --------------------------------------------------------------------------------
class Limiter:
    """Calls in flight of one lane, with a limit the controller can change while it runs."""

    def __init__(self, limit: int):
        self.limit, self.used = limit, 0
        self.cond = asyncio.Condition()

    async def __aenter__(self):
        async with self.cond:
            await self.cond.wait_for(lambda: self.used < self.limit)
            self.used += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.used -= 1
            self.cond.notify_all()

    async def set(self, limit: int) -> None:
        async with self.cond:
            self.limit = limit
            self.cond.notify_all()


class GatedPolicy:
    """The model's policy behind a gate (while the lane is paused, new calls wait) and the
    lane's in-flight limiter."""

    def __init__(self, inner: LLMPolicy, gate: asyncio.Event, limiter: Limiter):
        self.inner, self.gate, self.limiter = inner, gate, limiter

    async def __call__(self, agent, messages, sample):
        await self.gate.wait()
        async with self.limiter:
            return await self.inner(agent, messages, sample)


RETRIES = 40  # Adaption attempts per call (swarm.llm default 8); see docs/deviations.md D7


def patient_llm_class():
    from swarm.scale.llm_task import TaskLLM

    class PatientLLM(TaskLLM):
        """TaskLLM whose Adaption calls retry 429 and 5xx up to RETRIES times. The 429 answer
        says "retry after 60 seconds"; with 8 attempts (about 90 s of backoff) 7% of
        deepseek-v4-flash turns failed at a 78% 429 share. The cache key is unchanged."""

        async def chat(self, model, messages, **kw):
            kw["retries"] = max(kw.get("retries", 8), RETRIES)
            return await super().chat(model, messages, **kw)
    return PatientLLM


class Lane:
    def __init__(self, model: str):
        TaskLLM = patient_llm_class()
        self.model = model
        self.gate = asyncio.Event()
        self.conc = CONC_BY_MODEL.get(model, CONC)  # base allowance (cells and probes)
        self.limiter = Limiter(self.conc)
        self.llm = TaskLLM(budget_usd=SC.BUDGET, max_concurrency=FLEET_MAX + 64,
                           cache_path=SC.CACHE)  # the limiter binds, not swarm.llm's semaphore
        self.policy = GatedPolicy(LLMPolicy(self.llm, f"adaption/{model}",
                                            max_tokens=SC.MODELS[model][1],
                                            temperature=SC.TEMPERATURE), self.gate,
                                  self.limiter)
        self.cells_done = False
        self.started = False
        self.done = False
        self.phase = "waiting"


GIT_LOCK: asyncio.Lock | None = None
EVAL_LOCK: asyncio.Lock | None = None
STATE = {"budget_stop": False}


async def git_commit(paths: list[str], msg: str) -> None:
    async with GIT_LOCK:
        exist = [p for p in paths if (ROOT / p).exists()]
        p1 = await asyncio.create_subprocess_exec("git", "add", *exist, cwd=ROOT)
        await p1.wait()
        p2 = await asyncio.create_subprocess_exec("git", "commit", "-q", "-m", msg, cwd=ROOT,
                                                  stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.STDOUT)
        out, _ = await p2.communicate()
        log(f"git commit ({msg}): exit {p2.returncode} {out.decode().strip()[:200]}")


async def run_eval(stage: str) -> None:
    async with EVAL_LOCK:
        def go():
            r = getattr(EV, f"stage_{stage}")()
            name = {"c1c3": "c1c3", "probes": "probes", "c4": "c4"}[stage]
            (EV.OUT / f"{name}.json").write_text(json.dumps(r, indent=1))
            EV.write_md()
            return r
        EV.OUT.mkdir(parents=True, exist_ok=True)
        r = await asyncio.to_thread(go)
        if stage == "c1c3":
            for ln in EV.print_c1c3(r):
                log(f"eval c1c3: {ln}")


def per_call_usd(model: str, lg: dict) -> float:
    bm = lg.get("by_model", {}).get(f"adaption/{model}")
    if bm and bm["calls"] > 50:
        return bm["usd"] / bm["calls"]
    p_in, p_out = ADAPTION_PRICES[model]
    return 4500 * p_in + 900 * p_out


def done_episodes(path: Path) -> int:
    """Finished episodes = lines in the output file (cheap; no parsing)."""
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def projected_total() -> tuple[float, float]:
    lg = EV.ledger()
    rest = 0.0
    for f in SP.active_fleets():
        if not fleet_finished(SP.fleet_tag(f)):
            rest += f["N"] * SP.TURNS_PER_JOB * per_call_usd(f["model"], lg)
    for m in SC.MODELS:
        for probes in (False, True):
            n_all = len(SC.probe_specs(m) if probes else SC.cell_specs(m))
            left = max(0, n_all - done_episodes(SC.out_path(m, probes)))
            rest += left * 29 * per_call_usd(m, lg)
    return lg["spend_usd"], rest


def fleet_started(tag: str) -> bool:
    return Path(f"{SC.RAW / tag}.rounds.jsonl").exists()


def apply_budget_plan() -> bool:
    """D9: if the projection passes the cap, drop DROP_FIRST fleets not yet started, then
    shrink glm-5.2's fleet to N = 512 if not yet started. True when the projection fits."""
    spend, rest = projected_total()
    if spend + rest <= SC.BUDGET:
        return True
    plan = SP.load_plan()
    for t in SP.DROP_FIRST:
        if t not in plan["dropped"] and not fleet_started(t):
            plan["dropped"].append(t)
            plan["log"].append(f"{time.strftime('%F %T')} dropped {t}: projected "
                               f"${spend + rest:.2f} > ${SC.BUDGET:.0f}")
            log(f"=== BUDGET PLAN: dropped {t} (projected ${spend + rest:.2f})")
    SP.save_plan(plan)
    spend, rest = projected_total()
    if spend + rest <= SC.BUDGET:
        return True
    t, n = SP.SHRINK
    if t not in plan["n_override"] and not fleet_started(t):
        plan["n_override"][t] = n
        plan["log"].append(f"{time.strftime('%F %T')} {t} shrunk to N = {n}: projected "
                           f"${spend + rest:.2f} > ${SC.BUDGET:.0f}")
        SP.save_plan(plan)
        log(f"=== BUDGET PLAN: {t} shrunk to N = {n} (projected ${spend + rest:.2f})")
    spend, rest = projected_total()
    return spend + rest <= SC.BUDGET


def resolve(f: dict) -> dict | None:
    """The fleet as the budget plan has it now (None when dropped)."""
    t = SP.fleet_tag(f)
    for g in SP.active_fleets():
        if SP.fleet_tag(g) == t or g.get("shrunk_from") == t:
            return g
    return None


def fleet_finished(tag: str) -> bool:
    s = SUMMARY / f"scale_{tag}.json"
    if not s.exists():
        return False
    try:
        return json.loads(s.read_text()).get("stopped") is None
    except json.JSONDecodeError:
        return False


async def run_cells(lane: Lane, probes: bool) -> None:
    specs = SC.probe_specs(lane.model) if probes else SC.cell_specs(lane.model)
    out = SC.out_path(lane.model, probes)
    kind = "probes" if probes else "cells"
    lane.phase = kind
    log(f"=== {lane.model} {kind} start ({len(specs)} episodes) {physmem()}")
    last = [0.0]

    def progress(k: int, n: int) -> None:
        if time.time() - last[0] > 300 or k == n:
            last[0] = time.time()
            log(f"{lane.model} {kind}: {k}/{n} episodes, rss {peak_rss_mb():.0f}MB, task "
                f"spend ${lane.llm.total_spent():.2f}")
    conc = lane.conc
    if probes and SC.MODELS[lane.model][2] >= 1024:
        try:  # per-model override in the control file ("probe_episodes": {model: n})
            over = json.loads(CONTROL_FILE.read_text()).get("probe_episodes", {})
        except (OSError, json.JSONDecodeError):
            over = {}
        conc = min(conc, int(over.get(lane.model, PROBE_BIG_STORE_EPISODES)))
    res = await run_many(specs, lambda s: lane.policy, out, concurrency=conc, progress=progress)
    log(f"=== {lane.model} {kind} done: {json.dumps(res)}; task spend "
        f"${lane.llm.total_spent():.2f}")
    if res["stopped"]:
        raise RuntimeError(f"{lane.model} {kind} stopped by the budget cap")


def read_control() -> dict:
    try:
        return json.loads(CONTROL_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


async def run_fleet(lane: Lane, f: dict, policy=None) -> None:
    from scale_fleet import report
    orig = SP.fleet_tag(f)
    if not fleet_finished(orig):
        await asyncio.to_thread(apply_budget_plan)
    f = resolve(f)
    if f is None:
        log(f"=== fleet {orig} dropped for budget (D9), skipped")
        await git_commit([str(SP.PLAN.relative_to(ROOT))], f"Budget plan: {orig} dropped (D9)")
        return
    tag = SP.fleet_tag(f)
    if fleet_finished(tag):
        log(f"=== fleet {tag} already finished, skipped")
        return
    pred = json.loads(SP.PRED.read_text())["fleets"] if SP.PRED.exists() else {}
    if tag not in pred:  # a fleet shrunk for budget gets its prediction before it runs
        async with EVAL_LOCK:
            await asyncio.to_thread(SP.write_predictions, [f["model"]])
            EV.write_md()
        await git_commit(["results/predictions/c4.json", str(SP.PLAN.relative_to(ROOT)),
                          "docs/results_confirmatory.md"],
                         f"C4 prediction for {tag} (budget plan D9), frozen before the fleet")
        p = json.loads(SP.PRED.read_text())["fleets"][tag]
        log(f"=== PREDICTION {tag}: W {p['W']:.2f} R {p['R']:.2f} P_board {p['P_board']:.3f} "
            f"-> {'board' if p['predicted_board'] else 'no board'}")
    spend, rest = projected_total()
    if STATE["budget_stop"] or spend + rest > SC.BUDGET:
        STATE["budget_stop"] = True
        log(f"=== BUDGET STOP before fleet {tag}: spend ${spend:.2f} + projected remaining "
            f"${rest:.2f} > ${SC.BUDGET:.0f}; no further fleet starts")
        return
    lane.phase = f"fleet {tag}"
    log(f"=== fleet {tag} start; spend ${spend:.2f}, projected remaining ${rest:.2f}; "
        f"{physmem()}")
    spec = RunSpec(tag=tag, mode="fleet", n_agents=f["N"], p_blocked=0.3, prompt=f["prompt"],
                   rule=True, seed_note=f["seed_note"], n_background=200, p_derivative=1.0,
                   producers_in_fleet=True, steps=SP.STEPS,
                   model=f"adaption/{f['model']}", max_tokens=SC.MODELS[f["model"]][1],
                   temperature=SC.TEMPERATURE, seed=SP.FLEET_SEED, repeat=SP.REPEAT,
                   schedule="rounds", history_window=12)
    prefix = SC.RAW / tag

    def progress(rr: dict) -> None:
        log(f"{tag} step {rr['step']:>2} active {rr['active']:>4} live {rr['live']:>4} "
            f"cached {rr['cached']:>4} err {rr['errors']:>3} {rr['wall_s']:.0f}s "
            f"rss {rr['peak_rss_mb']}MB ${rr['cost']:.3f}")
    t0 = time.time()
    rec, labels = await run_world(spec, policy or lane.policy,
                                  sem=asyncio.Semaphore(FLEET_MAX + 64),
                                  out_prefix=prefix, progress=progress)
    extra = {"concurrency": lane.conc, "elapsed_s": round(time.time() - t0, 1),
             "task_spend_usd": round(lane.llm.total_spent(), 4),
             "llm_summary": lane.llm.summary(), "models_in_process": ORDER}

    def rep():
        out = report(rec, labels, extra, None, prefix)
        out["peak_rss_mb_process"] = round(peak_rss_mb(), 1)
        (SUMMARY / f"scale_{tag}.json").write_text(json.dumps(out, indent=1))
    await asyncio.to_thread(rep)
    log(f"=== fleet {tag} done in {time.time() - t0:.0f}s, stopped {rec['stopped']}, task "
        f"spend ${lane.llm.total_spent():.2f}")
    await run_eval("c4")
    d = json.loads((EV.OUT / "c4.json").read_text())["fleets"].get(tag, {})
    log(f"=== C4 {tag}: board {d.get('board')} ({d.get('n_board_jobs')} jobs), predicted R "
        f"{d.get('predicted_R')}, predicted board {d.get('predicted_board')}, correct "
        f"{d.get('correct')}")
    await git_commit(["results/confirmatory/c4.json", "docs/results_confirmatory.md",
                      f"results/summary/scale_{tag}.json"],
                     f"C4 fleet {tag}: board outcome and fleet table")


async def lane_main(lane: Lane) -> None:
    m = lane.model
    try:
        await run_cells(lane, probes=False)
        lane.cells_done = True
        await run_eval("c1c3")
        await git_commit(["results/confirmatory/c1c3.json", "docs/results_confirmatory.md"],
                         f"C1-C3 single-agent cells of {m}: rates and pass rules so far")
        await run_cells(lane, probes=True)
        await run_eval("c1c3")
        await run_eval("probes")
        await asyncio.to_thread(apply_budget_plan)
        pred = json.loads(SP.PRED.read_text())["fleets"] if SP.PRED.exists() else {}
        mine = [SP.fleet_tag(f) for f in SP.active_fleets() if f["model"] == m]
        if all(t in pred for t in mine):
            log(f"=== predictions for {m} already frozen")
        else:
            async with EVAL_LOCK:
                await asyncio.to_thread(SP.write_predictions, [m])
                EV.write_md()
            await git_commit(["results/predictions/c4.json", "results/confirmatory/c1c3.json",
                              "results/confirmatory/probes.json", str(SP.PLAN.relative_to(ROOT)),
                              "docs/results_confirmatory.md"],
                             f"C4 predictions for {m}, frozen before its fleets (D8)")
            p = json.loads(SP.PRED.read_text())["fleets"]
            for t in mine:
                log(f"=== PREDICTION {t}: W {p[t]['W']:.2f} R {p[t]['R']:.2f} P_board "
                    f"{p[t]['P_board']:.3f} -> {'board' if p[t]['predicted_board'] else 'no board'}")
        mine = [f for f in SP.FLEETS if f["model"] == m]
        ctl = read_control()
        if m in ctl.get("parallel_fleets", []):  # this model's fleets side by side
            cap = int(ctl.get("fleet_max", FLEET_MAX))
            log(f"=== {m}: {len(mine)} fleets side by side, {cap} calls in flight each")

            async def one(f: dict) -> None:
                pol = GatedPolicy(lane.policy.inner, lane.gate, Limiter(cap))
                await run_fleet(lane, f, policy=pol)
            await asyncio.gather(*(one(f) for f in mine))
        else:
            for f in mine:
                await run_fleet(lane, f)
    except Exception as e:  # noqa: BLE001
        log(f"=== LANE ERROR {m}: {type(e).__name__}: {e}")
    finally:
        lane.done = True
        lane.phase = "done"
        log(f"=== lane {m} finished")


RULE_FILE = SC.RAW / "logs" / "controller_rule.txt"
CONTROL_FILE = SC.RAW / "logs" / "controller_limits.json"


def rule_now() -> str:
    """"share" (default): +1 lane after 10 min with the 429 share below 20%, -1 lane when the
    5-minute share passes 20%. "throughput": +1 lane every 10 min while the last addition
    raised successful calls per minute by at least 10%, otherwise no further lane for 30
    minutes (no lane is paused).
    The file results/raw/scale/logs/controller_rule.txt switches the rule without a restart."""
    try:
        r = RULE_FILE.read_text().strip()
        return r if r in ("share", "throughput") else "share"
    except OSError:
        return "share"


async def controller(lanes: list[Lane]) -> None:
    """`target` lanes run at once, starting at START_LANES; see rule_now. A finished lane is
    replaced at once by the next waiting one."""
    last_change = time.time()
    last_add: tuple[float, float] | None = None  # (time of the addition, ok/min before it)
    frozen_until = 0.0
    active: list[Lane] = []
    target = START_LANES

    def start(lane: Lane) -> None:
        if not lane.started:
            lane.started = True
            asyncio.get_running_loop().create_task(lane_main(lane))
        lane.gate.set()
        active.append(lane)
        log(f"=== lane {lane.model} active (target {target}, active "
            f"{[x.model for x in active]})")

    def fill() -> None:
        active[:] = [x for x in active if not x.done]
        waiting = [x for x in lanes if not x.done and not x.gate.is_set()]
        while len(active) < target and waiting:
            start(waiting.pop(0))

    def pause_latest(why: str) -> None:
        nonlocal target, last_change
        target = len(active) - 1
        lane = active.pop()
        lane.gate.clear()
        last_change = time.time()
        log(f"=== back off: lane {lane.model} paused ({why}), target {target}")

    brake = {"cap": FLEET_MAX, "raised_at": None, "ok_before": None}

    def control() -> dict:
        """results/raw/scale/logs/controller_limits.json, read every minute (no restart):
        fleet_max (cap of the growing fleets), glm_cap_while_deepseek_fleet (deepseek fleets
        first), brake_hold_until (epoch seconds: the brake holds the cap, never lowers it)."""
        ctl = {"fleet_max": FLEET_MAX, "glm_cap_while_deepseek_fleet": 96,
               "brake_hold_until": 0}
        try:
            ctl.update(json.loads(CONTROL_FILE.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
        return ctl

    async def allocate() -> None:
        """Fleets of FLEET_GROW models get their base allowance plus the base allowance of
        every other lane whose cells are done, up to FLEET_MAX (and the brake cap). Other lanes
        keep their base. Brake: 10 min after a raise, if successful calls per minute fell by
        more than 10%, the cap drops by 48 (not below 48)."""
        now = time.time()
        ok5 = WIN.ok_per_min(now - BACKOFF_WINDOW_S, now)
        ctl = control()
        if brake["raised_at"] and now - brake["raised_at"] >= ADD_AFTER_S:
            if now < ctl["brake_hold_until"]:
                log(f"=== brake held (control file): ok/min {ok5:.0f} vs "
                    f"{brake['ok_before']:.0f} before the raise")
            elif brake["ok_before"] and ok5 < 0.9 * brake["ok_before"]:
                brake["cap"] = max(48, brake["cap"] - 48)
                log(f"=== brake: ok/min {ok5:.0f} vs {brake['ok_before']:.0f} before the raise; "
                    f"fleet cap {brake['cap']}")
            brake["raised_at"] = None
        raised = False
        ds_fleet = any(y.model == "deepseek-v4-flash" and y.phase.startswith("fleet")
                       for y in lanes)
        for x in lanes:
            want = x.conc
            if x.model in FLEET_GROW and x.phase.startswith("fleet"):
                freed = sum(y.conc for y in lanes if y is not x and y.cells_done)
                want = min(brake["cap"], int(ctl["fleet_max"]), x.conc + freed)
                if x.model == "glm-5.2" and ds_fleet:  # deepseek fleets before glm
                    want = min(want, int(ctl["glm_cap_while_deepseek_fleet"]))
            if want != x.limiter.limit:
                raised |= want > x.limiter.limit
                log(f"=== in-flight limit {x.model}: {x.limiter.limit} -> {want} ({x.phase})")
                await x.limiter.set(want)
        if raised and brake["raised_at"] is None:
            brake["raised_at"], brake["ok_before"] = now, ok5

    fill()
    while not all(x.done for x in lanes):
        await asyncio.sleep(60)
        await allocate()
        active[:] = [x for x in active if not x.done]
        now = time.time()
        rule = rule_now()
        s10, n10 = WIN.share(ADD_AFTER_S)
        s5, n5 = WIN.share(BACKOFF_WINDOW_S)
        ok5 = WIN.ok_per_min(now - BACKOFF_WINDOW_S, now)
        waiting = [x for x in lanes if not x.done and not x.gate.is_set()]
        log(f"[{rule}] 429 share 5 min {s5:.2f} (n {n5}), 10 min {s10:.2f}; ok/min 5 min "
            f"{ok5:.0f}; by model (429 share, ok/min) {WIN.by_model_share(BACKOFF_WINDOW_S)}; "
            f"target {target}; active {[(x.model, x.phase) for x in active]}; waiting "
            f"{[x.model for x in waiting]}; rss {peak_rss_mb():.0f}MB; spend "
            f"${EV.ledger()['spend_usd']:.2f}")
        if rule == "share":
            if s5 > SHARE_MAX and len(active) > 1 and n5 > 200 \
                    and now - last_change > BACKOFF_WINDOW_S:
                pause_latest(f"429 share {s5:.2f}")
            elif waiting and len(active) >= target and s10 < SHARE_MAX \
                    and now - last_change >= ADD_AFTER_S:
                target += 1
                last_change = now
        else:
            if last_add and now - last_add[0] >= ADD_AFTER_S:
                gain = ok5 / last_add[1] - 1 if last_add[1] else 1.0
                if gain < 0.10:  # no pause (all cell lanes keep running): stop adding a while
                    log(f"=== no gain from the last lane (ok/min {ok5:.0f} vs "
                        f"{last_add[1]:.0f}); no further lane for 30 min")
                    frozen_until = now + 1800
                last_add = None
            elif last_add is None and waiting and len(active) >= target \
                    and now >= frozen_until and now - last_change >= ADD_AFTER_S:
                last_add = (now, WIN.ok_per_min(now - BACKOFF_WINDOW_S, now))
                target += 1
                last_change = now
        fill()
    log("=== all lanes finished")


async def main_async() -> None:
    global GIT_LOCK, EVAL_LOCK
    GIT_LOCK, EVAL_LOCK = asyncio.Lock(), asyncio.Lock()
    patch_http()
    lanes = [Lane(m) for m in ORDER]
    await controller(lanes)
    await run_eval("c1c3")
    await run_eval("probes")
    await run_eval("c4")
    log(f"=== totals: http {dict(WIN.total)}; ledger {json.dumps(EV.ledger())}")


if __name__ == "__main__":
    log(f"=== orchestrator start {physmem()}")
    asyncio.run(main_async())

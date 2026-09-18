#!/usr/bin/env python3
"""Smoke test A: the local tau-bench runtime service in isolation (no models, no GPU).

Starts agents/tau_usi/runtime_service.py as a subprocess and drives it over HTTP with the
EXACT payloads agents/env_utils.py::BaseEnv sends, then checks:
  1. /create airline_0 returns meta_info {tools_info, wiki, instruction}
  2. the returned instruction matches data/tau-usi/data (airline_0 == mia_li_3668 task)
  3. /step runs a read-only tool (get_user_details) and returns an observation
  4. /reward returns a float (empty transcript -> 0.0 expected)
  5. task counts: airline has index 49, retail has index 114 (create both)

Run (in rl2, from repo root):
    python agents/tau_usi/smoke_runtime.py
Exits non-zero on any failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PORT = 8006  # avoid clashing with a full run on 8005
BASE = f"http://localhost:{PORT}"


def post(path: str, payload: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(tries: int = 60) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if json.loads(r.read().decode()).get("ok"):
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def create(env_name: str, task_index: int) -> dict:
    """Mirror BaseEnv.create: env_type='tau', params is a JSON string."""
    return post("/create", {"env_type": "tau", "params": json.dumps({"env_name": env_name, "task_index": task_index})})


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "agents.tau_usi.runtime_service", "--port", str(PORT)],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ok = True
    try:
        if not wait_health():
            print("FAIL: runtime service did not become healthy")
            print(proc.stdout.read() if proc.stdout else "")
            return 1

        # 1) create airline_0
        res = create("airline", 0)
        rid = res["runtime_id"]
        meta = json.loads(res["meta_info"])
        tools = meta["tools_info"]
        instr = meta["instruction"]
        print(f"[1] created airline_0 rid={rid[:8]} | tools={len(tools)} | wiki_len={len(meta['wiki'])}")
        print(f"    tool names: {[t['function']['name'] for t in tools]}")
        print(f"    instruction[:160]: {instr[:160]}")

        # 2) instruction matches human data
        human = json.loads((REPO / "data" / "tau-usi" / "data" / "tau_bench_tasks_unified.json").read_text())
        canvas = next((m.get("content", "") for m in human["airline_0"]["conversation"] if "canvas" in str(m.get("content", "")).lower()), "")
        for phrase in ("mia_li_3668", "New York to Seattle", "May 20"):
            match = phrase in instr and phrase in canvas
            print(f"[2] phrase {phrase!r} in both tau-bench + human data: {match}")
            ok = ok and match

        # 3) read-only tool via /step (BaseEnv._execute_step payload shape)
        step_res = post("/step", {"runtime_id": rid, "params": json.dumps({"name": "get_user_details", "arguments": {"user_id": "mia_li_3668"}})})
        obs = step_res.get("result", "")
        got_user = "mia" in obs.lower() or "li" in obs.lower() or obs.strip().startswith("{")
        print(f"[3] get_user_details -> {obs[:160]!r}")
        print(f"    step returned a plausible observation: {got_user}")
        ok = ok and got_user

        # 4) reward (empty transcript)
        rew = post("/reward", {"runtime_id": rid, "params": json.dumps({"messages": []})})
        r = rew.get("reward")
        print(f"[4] /reward (empty transcript) -> {r} (float: {isinstance(r, (int, float))})")
        ok = ok and isinstance(r, (int, float))

        # 5) task-count bounds: last valid indices must construct
        for env_name, last in (("airline", 49), ("retail", 114)):
            try:
                r2 = create(env_name, last)
                good = "runtime_id" in r2
            except Exception as e:  # noqa: BLE001
                good = False
                print(f"    {env_name}_{last} create error: {e}")
            print(f"[5] create {env_name}_{last}: {good}")
            ok = ok and good

        print("\nSMOKE A:", "PASS ✅" if ok else "FAIL ❌")
        return 0 if ok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())

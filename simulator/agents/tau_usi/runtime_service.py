# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local tau-bench runtime service for the tau-USI benchmark."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import uuid

from aiohttp import web

from tau_bench.envs import get_env
from tau_bench.types import Action, RESPOND_ACTION_NAME

log = logging.getLogger("tau_usi.runtime_service")

RUNTIMES: dict[str, dict] = {}

TASK_SPLIT = os.getenv("TAU_USI_TASK_SPLIT", "test")


def _meta_info(env) -> dict:
    """The subset of env metadata the tau_usi client reads (agent.py + tool_prompt.py)."""
    return {
        "tools_info": env.tools_info,
        "wiki": env.wiki,
        "instruction": env.task.instruction,
    }


def _build_env(env_name: str, task_index: int):
    return get_env(
        env_name,
        user_strategy="human",
        user_model="none",
        task_split=TASK_SPLIT,
        user_provider=None,
        task_index=int(task_index),
    )


def _tool_props(env, tool_name: str) -> dict:
    """JSON-schema ``properties`` for a tool, used to type-coerce string arguments."""
    for t in env.tools_info:
        fn = t.get("function", {})
        if fn.get("name") == tool_name:
            return fn.get("parameters", {}).get("properties", {}) or {}
    return {}


def _coerce_args(env, tool_name: str, arguments: dict) -> dict:
    """Coerce XML-parsed string args to the types tau-bench tools expect."""
    props = _tool_props(env, tool_name)
    out: dict = {}
    for key, val in (arguments or {}).items():
        if not isinstance(val, str):
            out[key] = val
            continue
        typ = props.get(key, {}).get("type")
        raw = val.strip()
        try:
            if typ == "integer":
                out[key] = int(raw)
            elif typ == "number":
                out[key] = float(raw)
            elif typ == "boolean":
                out[key] = raw.lower() in ("true", "1", "yes")
            elif typ in ("array", "object"):
                out[key] = json.loads(raw)
            else:
                out[key] = val
        except Exception:  # noqa: BLE001
            out[key] = val
    return out


async def _read(request) -> dict:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


async def create(request):
    body = await _read(request)
    try:
        params = json.loads(body.get("params") or "{}")
        env_name = str(params["env_name"])
        task_index = int(params["task_index"])
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"bad create params: {e}"}, status=400)
    if env_name not in ("airline", "retail"):
        return web.json_response({"error": f"unknown env_name {env_name!r}"}, status=400)
    try:
        env = await asyncio.to_thread(_build_env, env_name, task_index)
    except Exception as e:  # noqa: BLE001
        log.exception("create failed for %s_%s", env_name, task_index)
        return web.json_response({"error": f"env build failed: {e}"}, status=500)
    rid = uuid.uuid4().hex
    RUNTIMES[rid] = {"env": env, "env_name": env_name, "task_index": task_index}
    log.debug("created %s -> %s_%s", rid, env_name, task_index)
    return web.json_response({"runtime_id": rid, "meta_info": json.dumps(_meta_info(env))})


async def ping(request):
    body = await _read(request)
    ent = RUNTIMES.get(body.get("runtime_id"))
    if not ent:
        return web.json_response({"exists": False, "meta_info": None})
    return web.json_response({"exists": True, "meta_info": json.dumps(_meta_info(ent["env"]))})


async def step(request):
    body = await _read(request)
    ent = RUNTIMES.get(body.get("runtime_id"))
    if not ent:
        return web.json_response({"error": "unknown runtime_id"}, status=503)
    try:
        fn = json.loads(body.get("params") or "{}")
    except Exception as e:  # noqa: BLE001
        return web.json_response({"result": f"Error: bad step params: {e}"})
    name = fn.get("name")
    args = fn.get("arguments", {}) or {}
    if name == RESPOND_ACTION_NAME:
        return web.json_response({"result": ""})

    env = ent["env"]
    kwargs = _coerce_args(env, name, args)

    def _do():
        resp = env.step(Action(name=name, kwargs=kwargs))
        return resp.observation

    try:
        obs = await asyncio.to_thread(_do)
    except Exception as e:  # noqa: BLE001
        log.exception("step failed (%s)", name)
        obs = f"Error executing tool: {e}"
    return web.json_response({"result": obs})


async def reward(request):
    body = await _read(request)
    ent = RUNTIMES.get(body.get("runtime_id"))
    if not ent:
        return web.json_response({"reward": 0.0})
    lock = ent.setdefault("_reward_lock", asyncio.Lock())
    async with lock:
        if "reward" in ent:
            return web.json_response({"reward": ent["reward"]})

        try:
            params = json.loads(body.get("params") or "{}")
        except Exception:  # noqa: BLE001
            params = {}
        messages = params.get("messages") or []
        env = ent["env"]

        def _do():
            for m in messages:
                if m.get("role") == "assistant" and m.get("content"):
                    env.actions.append(Action(name=RESPOND_ACTION_NAME, kwargs={"content": m["content"]}))
            snapshot = copy.deepcopy(env.data)
            try:
                return float(env.calculate_reward().reward)
            finally:
                env.data = snapshot

        try:
            r = await asyncio.to_thread(_do)
        except Exception:  # noqa: BLE001
            log.exception("reward failed")
            r = 0.0
        ent["reward"] = r
        return web.json_response({"reward": r})


async def health(request):
    return web.json_response({"ok": True, "runtimes": len(RUNTIMES)})


def build_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes(
        [
            web.post("/create", create),
            web.post("/ping", ping),
            web.post("/step", step),
            web.post("/reward", reward),
            web.get("/health", health),
        ]
    )
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local tau-bench runtime service for tau-USI.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8005)
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.getenv("TAU_USI_RUNTIME_LOG", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    log.info("tau-bench runtime service listening on %s:%d (split=%s)", args.host, args.port, TASK_SPLIT)
    web.run_app(build_app(), host=args.host, port=args.port, access_log=None, print=None)


if __name__ == "__main__":
    main()

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

"""Standalone, verl-free driver for MIMESIS's tau_usi rollout."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace

from transformers import AutoTokenizer

from agents.tau_usi.agent import rollout_one_task
from agents.utils import CallAPI

TOKENIZER_ID = os.getenv("TAU_USI_TOKENIZER", "Qwen/Qwen2.5-0.5B-Instruct")


class EvalContext:
    """Minimal stand-in for verl's TaskContext — only what rollout_one_task reads."""

    def __init__(self, llm_client, tokenizer, config):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = False
        self.global_step = 0

    def get(self, key, default=None):
        return default


def parse_domains(spec: str):
    """'retail:0-9,airline:0-4' -> [('retail',0), ..., ('airline',4)]."""
    tasks = []
    for part in spec.split(","):
        dom, rng = part.split(":")
        dom = dom.strip()
        if "-" in rng:
            a, b = rng.split("-")
            idxs = range(int(a), int(b) + 1)
        else:
            idxs = [int(rng)]
        tasks.extend((dom, i) for i in idxs)
    return tasks


async def _run(args):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    config = SimpleNamespace(
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        calculate_log_probs=False,
    )
    llm = CallAPI(url=args.user_sim_model, tokenizer=tok, config=config)
    ctx = EvalContext(llm, tok, config)

    tasks = parse_domains(args.domains)
    sem = asyncio.Semaphore(args.workers)

    async def one(dom, idx):
        data = {"env_name": dom, "task_index": idx, "instance_id": f"{dom}_{idx}"}
        async with sem:
            t0 = time.monotonic()
            try:
                rec, user_agent = await rollout_one_task(data, ctx)
                rec["elapsed_seconds"] = round(time.monotonic() - t0, 1)
                rec["user_sim_model"] = args.user_sim_model
                rec["user_sim_chat"] = user_agent.chat
                print(
                    f"[{dom}_{idx}] reward={rec['reward']} turns={len(rec['conversation'])} "
                    f"term={rec['termination_reason']} ({rec['elapsed_seconds']}s)",
                    flush=True,
                )
                return rec
            except Exception as e:
                print(f"[{dom}_{idx}] ERROR: {type(e).__name__}: {e}", flush=True)
                return None

    recs = await asyncio.gather(*[one(d, i) for d, i in tasks])
    results = [r for r in recs if r is not None]

    payload = {
        "benchmark": "tau_usi",
        "model": os.getenv("OPENAI_AGENT_MODEL", "gpt-5-nano"),
        "user_sim_model": args.user_sim_model,
        "n": len(results),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    rewards = [r["reward"] for r in results]
    succ = 100.0 * sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\nWROTE {args.out}: {len(results)}/{len(tasks)} tasks, agent success={succ:.0f}%", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-sim-model", required=True, help="model id for the user simulator (CallAPI)")
    ap.add_argument("--domains", default="retail:0-9,airline:0-4", help="e.g. 'retail:0-9,airline:0-4'")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None, help="output *_task_results.json (default from model name)")
    ap.add_argument("--prompt-length", type=int, default=16000)
    ap.add_argument("--response-length", type=int, default=8000)
    ap.add_argument("--repetition-penalty", type=float,
                    default=float(os.getenv("REPETITION_PENALTY", "1.0")),
                    help="vLLM repetition_penalty (>1 discourages repeats, multiplicative). 1.0 = off.")
    ap.add_argument("--frequency-penalty", type=float,
                    default=float(os.getenv("FREQUENCY_PENALTY", "0.0")),
                    help="OpenAI/vLLM frequency_penalty (>0 penalizes tokens by count). 0 = off.")
    ap.add_argument("--presence-penalty", type=float,
                    default=float(os.getenv("PRESENCE_PENALTY", "0.0")),
                    help="OpenAI/vLLM presence_penalty (>0 penalizes any reused token). 0 = off.")
    ap.add_argument("--no-repeat-ngram-size", type=int,
                    default=int(os.getenv("NO_REPEAT_NGRAM", "0")),
                    help="hard-block repeating n-grams. UNSUPPORTED in vLLM 0.24.0 — keep 0.")
    args = ap.parse_args(argv)
    if not args.out:
        args.out = f"{args.user_sim_model.replace('/', '_')}_task_results.json"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

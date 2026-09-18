"""Drive usersim-behavior rollouts end to end against the live gateway, with no verl.

The customer is played by an API model through CallAPI (the transport the RL run's eval
path uses), so this exercises the real turn loop, the real ABCD support agent, and the
real judge -- everything except the vLLM policy and get_agent_output. The rollout itself
comes from ``agent.run_rollout``, the same function ``agent_loop`` calls, so there is no
second copy to drift.

    python -m agents.usersim_behavior.dryrun --n 6
    python -m agents.usersim_behavior.dryrun --behavior infeasibility_persistence
    python -m agents.usersim_behavior.dryrun --behavior control --n 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

from transformers import AutoTokenizer

from agents.usersim_behavior.agent import run_rollout
from agents.usersim_behavior.prompts import build_agent_prompt
from agents.utils import CallAPI

# The tokenizer is only used for length accounting, but it must load without network:
# compute nodes cannot reach the HF hub. Default to the local actor checkpoint
# (run_rl.sh:40); override with USB_TOKENIZER for a different one.
TOKENIZER_ID = os.getenv(
    "USB_TOKENIZER",
    "${MIMESIS_OUTPUT_DIR}/Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329202",
)
PARQUET = os.getenv(
    "USB_PARQUET", "${MIMESIS_DATA_DIR}/sim_rl_data/usersim_behavior_rl_train.parquet"
)


class EvalContext:
    """Minimal stand-in for verl's TaskContext -- only what run_rollout reads."""

    def __init__(self, llm_client, tokenizer, config):
        self.llm_client, self.tokenizer, self.config = llm_client, tokenizer, config
        self.is_train = False
        self.global_step = 0


def report(r: dict, *, quiet: bool = False) -> list[str]:
    problems: list[str] = []
    sim, public, private = r["sim"], r["public"], r["private"]

    if not quiet:
        print("=" * 78)
        print(f"BEHAVIOR: {r['name']}   subflow={r['subflow']}   needs_trace={r['needs_trace']}")
        print(f"seeded turns: {r['n_seeded']}   generated: {r['n_sim_turns']}")
        print("=" * 78)
        if r["kwargs"]:
            print(f"kwargs: {r['kwargs']}")
        if r["hint"]:
            print(f"hint:   {r['hint']}\n")
        n_user = 0
        for i, (role, text) in enumerate(public):
            seeded = " (seed)" if n_user < r["n_seeded"] and role == "user" else ""
            if role == "user":
                thought = private[n_user]["thought"]
                n_user += 1
                if thought:
                    print(f"  [thinking] {thought[:220]}")
                print(f"  CUSTOMER{seeded}: {text}")
            else:
                print(f"  AGENT:    {text[:280]}")
        print()
        print(f"verdict: {r['verdict']}")
        print(f"rule:    {r['rule']}")
        print(f"REWARD:  {r['reward']:.3f}   sub={ {k: round(v, 3) for k, v in r['sub'].items()} }")

    # -- assertions ------------------------------------------------------------------
    if r["n_sim_turns"] == 0:
        problems.append(f"{r['name']}: no generated turns")
    # Injected turns must carry no gradient. sim.chat is [sys, seed..., asst, user, ...];
    # every turn whose chat_completions entry is None was appended as context.
    for i, (turn, completion) in enumerate(zip(sim.chat, sim.chat_completions, strict=False)):
        if completion is None and i >= sim.prompt_turn and any(sim.token_mask[i]):
            problems.append(f"{r['name']}: context turn {i} ({turn['role']}) has token_mask=True")
    # Trace isolation: nothing the customer only thought may appear in what the agent saw.
    agent_saw = build_agent_prompt("x", "x") + json.dumps([t for role, t in public if role == "user"])
    for p in private:
        thought = (p["thought"] or "").strip()
        if len(thought) > 25 and thought[:60] in agent_saw:
            problems.append(f"{r['name']}: THOUGHT LEAKED: {thought[:60]!r}")
    if r["verdict"] is None and r["n_sim_turns"]:
        problems.append(f"{r['name']}: judge returned None")
    # The doubled-first-message symptom from api._responses_text.
    for role, text in public:
        half, rest = divmod(len(text), 2)
        if role == "agent" and rest == 0 and half > 20 and text[:half] == text[half:]:
            problems.append(f"{r['name']}: agent turn is still doubled")

    if not quiet:
        print(f"assertions: {'OK' if not problems else 'FAIL'}")
        for p in problems:
            print("  -", p)
        print()
    return problems


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--behavior", default=None, help="pin to one behavior id, or 'control'")
    ap.add_argument(
        "--per-behavior",
        type=int,
        default=None,
        help="take N rows of EVERY behavior (plus N control) instead of the first --n rows. "
        "The parquet is shuffled, so --n gives whatever coverage it happens to give.",
    )
    ap.add_argument("--model", default=os.getenv("USB_SIM_MODEL", "gpt-5-5"))
    ap.add_argument("--prompt-length", type=int, default=8192)
    ap.add_argument("--response-length", type=int, default=4096)
    ap.add_argument("--quiet", action="store_true", help="summary table only")
    args = ap.parse_args()

    import pandas as pd

    rows = [dict(x) for _, x in pd.read_parquet(PARQUET)["extra_info"].items()]
    if args.behavior == "control":
        rows = [r for r in rows if json.loads(r["behavior_ids"]) == []]
    elif args.behavior:
        rows = [r for r in rows if args.behavior in json.loads(r["behavior_ids"])]

    if args.per_behavior:
        taken: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            ids = json.loads(r["behavior_ids"])
            key = ids[0] if ids else "control"
            if len(taken[key]) < args.per_behavior:
                taken[key].append(r)
        rows = [r for group in taken.values() for r in group]
        print(f"even coverage: {len(taken)} behaviors x up to {args.per_behavior} = {len(rows)} rollouts")
    else:
        rows = rows[: args.n]

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    config = SimpleNamespace(
        prompt_length=args.prompt_length, response_length=args.response_length,
        temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
        frequency_penalty=0.0, presence_penalty=0.0, no_repeat_ngram_size=0,
        calculate_log_probs=False,
    )
    ctx = EvalContext(CallAPI(url=args.model, tokenizer=tok, config=config), tok, config)

    results = await asyncio.gather(*[run_rollout(r, ctx) for r in rows])
    problems = [p for r in results for p in report(r, quiet=args.quiet)]

    # Per-behavior summary -- the number that told us turn_budget=4 was too small.
    by: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by[r["name"]].append(r)
    print("=" * 78)
    print(f"{'behavior':<34} {'n':>2} {'exhibit':>8} {'place':>7} {'natural':>8} {'reward':>7}")
    print("-" * 78)
    for name, rs in sorted(by.items()):
        get = lambda k: sum(r["sub"].get(k, 0.0) for r in rs) / len(rs)  # noqa: E731
        key = "cooperative" if name == "control" else "exhibit"
        print(f"{name:<34} {len(rs):>2} {get(key):>8.2f} {get('placement'):>7.2f} "
              f"{get('naturalness'):>8.2f} {sum(r['reward'] for r in rs) / len(rs):>7.2f}")
    print("-" * 78)
    print(f"{len(results)} rollouts, {len(problems)} problems")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

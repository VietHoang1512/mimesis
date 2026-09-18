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

"""
Behavior-conditioned user simulator on ABCD.

The policy plays a customer for ``turn_budget`` turns against a frozen support agent that
is following that scenario's real ABCD procedure. On a behavior row the customer's system
prompt carries a directive from the taxonomy; on a control row it carries none and the
reward is for behaving unremarkably. A judge scores the finished conversation.

Structure is sotopia's two-agent loop (``agents/sotopia/agent.py:687``) with the policy in
the user seat as in mirrorbench (``agents/mirrorbench/agent.py:600``), and the row schema
is IFBench's (``agents/instruct/ifbench_agent.py:98``).

Three failure modes from sotopia are deliberately not inherited; see the notes at
``_judge``, ``_reward``, and the ``extra_info`` assembly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from agents.usersim_behavior.abcd_env import ABCDSupportAgent
from agents.usersim_behavior.abcd_data import verification_prefix
from agents.usersim_behavior.behaviors import BEHAVIOR_DICT
from agents.usersim_behavior.prompts import (
    CONTROL_JUDGE,
    build_behavior_judge_prompt,
    build_sim_prompt,
    render_public,
    render_with_thoughts,
    subflow_goal,
)
from agents.utils import (
    Agent,
    call_openai_parse,
    get_judge_reasoning,
    process_post_chat,
    remove_think,
    split_think_response,
)

logger = logging.getLogger(__name__)

# Weights over the judged dimensions. `exhibit` dominates; `placement` is weighted heavily
# enough to matter because a judge asked only "did the behavior appear?" scores "once, at
# the right moment" and "in all four turns" identically, and the second is easier to
# produce -- optimising exhibit alone converges on caricature.
W_EXHIBIT, W_PLACEMENT, W_NATURAL, W_ONTASK = 2.0, 1.5, 1.0, 0.5
# Divisor when the sim narrates the directive instead of enacting it. Same shape as
# sotopia's hack_judge penalty (agents/sotopia/agent.py:756), applied in train and eval
# alike because an announced behavior is a bad rollout either way.
ANNOUNCE_DIVISOR = 2.0


class BehaviorVerdict(BaseModel):
    exhibit: float
    placement: float
    naturalness: float
    on_task: float
    announced: bool


class ControlVerdict(BaseModel):
    cooperative: float
    naturalness: float
    on_task: float


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


async def _judge(prompt: str, schema: type[BaseModel]) -> dict | None:
    """One structured judge call. Returns None on failure rather than raising.

    ``api.chat_parse`` raises after exhausting retries (``api.py:908``) and the batch is
    gathered without ``return_exceptions=True`` (``agent_loop.py:500``), so an uncaught
    judge failure destroys every rollout in the batch, not just this one.
    """
    try:
        return await call_openai_parse(
            [{"role": "user", "content": prompt}],
            schema,
            reasoning_effort=get_judge_reasoning("low"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[usersim_behavior] judge failed: %s: %s", type(e).__name__, e)
        return None


def _reward(
    verdict: dict | None, rule: float | None, *, is_control: bool
) -> tuple[float, dict[str, float]]:
    """Weighted reward plus the sub-scores to log.

    ``rule`` is **logged, not rewarded**. The deterministic checks in ``behaviors.py``
    turned out to be lexical proxies for semantic properties, and measured against the
    judge on real rollouts they agreed once in seven: they scored a cooperative revision
    as self_contradiction (1.00 vs the judge's 0.10), an honest "I don't have the date"
    as artificial_constraint_stacking, and a false ID stated once and never defended as a
    full fabricated_or_false_premise. They also skewed the scale -- eight behaviors carried
    a rule term and five did not, so the denominator differed and identical judge scores
    produced different rewards depending on the behavior.

    They stay computed because the rule/judge gap is a useful diagnostic: a rule that
    fires while the judge does not is where to look for a rollout the judge mis-scored,
    and if reward hacking ever appears the ground-truth checks (asserted value vs. the
    scenario's) are the cheapest place to see it.

    On judge failure the reward is 0.0 -- explicitly not a midpoint. Sotopia fills failed
    dimensions with the middle of each range (``agents/sotopia/agent.py:571``), so a broken
    judge there pays 0.5 and a gateway outage looks like mediocre performance rather than
    missing data.
    """
    sub: dict[str, float] = {"judge_failed": float(verdict is None)}
    if rule is not None:
        sub["rule"] = rule
    if verdict is None:
        return 0.0, sub

    if is_control:
        cooperative = _clamp(verdict.get("cooperative"))
        naturalness = _clamp(verdict.get("naturalness"))
        on_task = _clamp(verdict.get("on_task"))
        sub.update(cooperative=cooperative, naturalness=naturalness, on_task=on_task)
        total = 2.0 * cooperative + W_NATURAL * naturalness + W_ONTASK * on_task
        return _clamp(total / (2.0 + W_NATURAL + W_ONTASK)), sub

    exhibit = _clamp(verdict.get("exhibit"))
    placement = _clamp(verdict.get("placement"))
    naturalness = _clamp(verdict.get("naturalness"))
    on_task = _clamp(verdict.get("on_task"))
    announced = bool(verdict.get("announced"))
    sub.update(
        exhibit=exhibit, placement=placement, naturalness=naturalness, on_task=on_task,
        announced=float(announced),
    )
    if rule is not None:
        # Divergence, signed: positive when the rule fires and the judge does not.
        sub["rule_judge_gap"] = rule - exhibit

    num = W_EXHIBIT * exhibit + W_PLACEMENT * placement + W_NATURAL * naturalness + W_ONTASK * on_task
    reward = num / (W_EXHIBIT + W_PLACEMENT + W_NATURAL + W_ONTASK)
    if announced:
        reward /= ANNOUNCE_DIVISOR
    return _clamp(reward), sub


async def run_rollout(row: dict, context) -> dict[str, Any]:
    """Rollout + judge + reward, with no verl dependency.

    Split out from ``agent_loop`` (which adds ``get_agent_output``/``process_post_chat``)
    so the offline driver in ``dryrun.py`` exercises this exact code path rather than a
    copy that can drift. Same split as ``agents/tau_usi/agent.py::rollout_one_task``.
    """
    scenario = json.loads(row["scenario"])
    behavior_ids = json.loads(row["behavior_ids"])
    kwargs_list = json.loads(row["kwargs"])
    flow, subflow = row["flow"], row["subflow"]
    turn_budget = int(row.get("turn_budget", 8))

    is_control = not behavior_ids
    behavior = None if is_control else BEHAVIOR_DICT[behavior_ids[0]]
    kwargs = {} if is_control else (kwargs_list[0] if kwargs_list else {})
    hint = None if is_control else behavior.build_hint(**kwargs)

    sim_chat = [
        {"role": "system", "content": build_sim_prompt(scenario, hint)},
        {"role": "user", "content": "=== The support chat has connected. Send your first message. ==="},
    ]
    sim = Agent(context.llm_client, sim_chat, context.tokenizer, context.config, prompt_turn=2)
    support = ABCDSupportAgent(flow, subflow)

    # Two transcripts, kept separate on purpose. `public` is what the support agent sees
    # and what the judge sees for the 11 non-trace behaviors; `private` additionally holds
    # the reasoning block. Deriving one from the other -- or handing `sim.chat` to the
    # support agent, which holds raw completions -- would leak the customer's thoughts to
    # the party that must not have them.
    public: list[tuple[str, str]] = []
    private: list[dict[str, str]] = []

    # Replay the identity check as context so the rollout starts where the behavior can
    # land. Every turn goes in with completion=None -> token_mask all False, so none of it
    # is trained on (agents/utils.py:1036).
    for role, text in verification_prefix(scenario, flow, subflow):
        public.append((role, text))
        if role == "user":
            sim.append({"role": "assistant", "content": text})
            # Placeholder keeps `private` index-aligned with the user turns of `public`,
            # which render_with_thoughts relies on.
            private.append({"thought": "", "said": text})
        else:
            sim.append({"role": "user", "content": f"Agent: {text}"})
    n_seeded = len(private)

    for _ in range(turn_budget):
        raw = await sim.step()
        if raw is None:
            break  # response budget exhausted (agents/utils.py:1313)
        utterance = remove_think(raw)
        if not utterance.strip():
            break  # malformed <think>: clean_response returns "" (agents/utils.py:1064)
        thought, _ = split_think_response(sim.last_raw)
        public.append(("user", utterance))
        private.append({"thought": thought or "", "said": utterance})

        reply = await support.step(public)
        if reply is None:
            break
        public.append(("agent", reply))
        sim.append({"role": "user", "content": f"Agent: {reply}"})

    n_sim_turns = len(private) - n_seeded
    goal = subflow_goal(flow, subflow)

    if n_sim_turns == 0:
        verdict, rule = None, None
        logger.warning("[usersim_behavior] no generated turns for index=%s", row.get("index"))
    elif is_control:
        verdict = await _judge(CONTROL_JUDGE.format(transcript=render_public(public), goal=goal), ControlVerdict)
        rule = None
    else:
        transcript_text = (
            render_with_thoughts(public, private) if behavior.needs_trace else render_public(public)
        )
        prompt = build_behavior_judge_prompt(
            behavior=behavior,
            transcript_text=transcript_text,
            goal=goal,
            confusable=behavior.confusable_with,
            definitions={n: b.definition for n, b in BEHAVIOR_DICT.items()},
        )
        verdict = await _judge(prompt, BehaviorVerdict)
        try:
            rule = behavior.check_rule(public, scenario, kwargs)
        except Exception as e:  # noqa: BLE001 -- a rule bug must not lose the rollout
            logger.warning("[usersim_behavior] check_rule(%s) raised: %s", behavior.name, e)
            rule = None

    reward, sub = _reward(verdict, rule, is_control=is_control)
    return {
        "sim": sim,
        "name": "control" if is_control else behavior.name,
        "needs_trace": bool(behavior and behavior.needs_trace),
        "hint": hint,
        "kwargs": kwargs,
        "subflow": subflow,
        "public": public,
        "private": private,
        "n_seeded": n_seeded,
        "n_sim_turns": n_sim_turns,
        "agent_failures": support.failures,
        "is_control": is_control,
        "verdict": verdict,
        "rule": rule,
        "reward": reward,
        "sub": sub,
    }


async def agent_loop(data: dict, context) -> Any:
    row = data["extra_info"]
    r = await run_rollout(row, context)
    sim = r["sim"]

    # extra_info must be complete BEFORE get_agent_output: it snapshots the dict
    # (agents/utils.py:1242), so anything written afterwards is silently dropped -- the bug
    # that loses two keys in sotopia (agents/sotopia/agent.py:854). Values stay flat floats
    # because reward_extra_info arrays are built without dtype=object (agent_loop.py:829).
    name, reward = r["name"], r["reward"]
    extra_info: dict[str, float] = {
        "usersim_behavior/reward": reward,
        "usersim_behavior/n_sim_turns": float(r["n_sim_turns"]),
        "usersim_behavior/agent_failures": float(r["agent_failures"]),
        "usersim_behavior/is_control": float(r["is_control"]),
        # Deliberately NOT "all/score" or "all/score_v1". Those are flat, sample-weighted
        # means over every validation row that carries the key (ray_trainer.py:819-828), and
        # they are the headline this project has tracked across 23 tasks since before this
        # one existed. Our 244 val rows would be ~8% of that average -- the second-largest
        # single contributor after humanllm -- so folding them in would silently break
        # comparability with every earlier run. "all/score_v3" is unused elsewhere, so
        # val-core/all/score_v3 is this task's headline alone.
        "all/score_v3": reward,
    }
    for key, value in r["sub"].items():
        extra_info[f"usersim_behavior/{key}"] = float(value)
        extra_info[f"usersim_behavior/{name}/{key}"] = float(value)
    extra_info[f"usersim_behavior/{name}/reward"] = reward

    output = await sim.get_agent_output(reward, extra_info=extra_info)
    await process_post_chat(data, context, sim.chat, output)
    return output

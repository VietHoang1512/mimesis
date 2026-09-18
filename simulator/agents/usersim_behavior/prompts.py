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

"""Prompts for the behavior-conditioned user simulator: the customer (policy), the
frozen support agent, and the judge."""

from __future__ import annotations

from typing import Any

from agents.usersim_behavior.abcd_data import SUBFLOW_MAP, flatten_scenario, flow_description, render_policy

# --------------------------------------------------------------------------------------
# The simulated customer -- this is the policy being trained
# --------------------------------------------------------------------------------------

SIM_SYSTEM = """\
You are {name}, a real person contacting an online store's customer support by chat.

What you know about your own situation:
{facts}

Why you are getting in touch: {goal}

How you write:
- Short messages. Real people type a sentence, not a paragraph.
- Imperfect punctuation and capitalisation are fine. Typos happen.
- Do not dump every detail at once -- give information as it is asked for.
- You are a customer, not an assistant. Never offer to help, never summarise the
  conversation, never explain policy back to the agent.
{behavior_block}
Write only the message you would type next. No name prefix, no quotation marks, no stage
directions."""

BEHAVIOR_BLOCK = """
<behavior>
{hint}
</behavior>
Enact this. Do not mention it, name it, or explain that you are doing it -- a real person
in this state would simply behave this way. It does not have to appear in every message;
put it where it would naturally land.
"""

CONTROL_BLOCK = """
You are an ordinary, cooperative customer: you answer what you are asked, you accept
reasonable answers, and you do not create friction.
"""


def build_sim_prompt(scenario: dict[str, Any], hint: str | None) -> str:
    """`hint` is None on control rows."""
    flat = flatten_scenario(scenario)
    name = flat.get("customer_name", "a customer")
    hide = {"customer_name"}
    facts = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in flat.items() if k not in hide and v)
    products = (scenario.get("product") or {}).get("names") or []
    if products:
        facts += f"\n- items on the order: {', '.join(products)}"
    return SIM_SYSTEM.format(
        name=name.title(),
        facts=facts or "- (nothing on file)",
        goal=subflow_goal(scenario.get("flow", ""), scenario.get("subflow", "")),
        behavior_block=BEHAVIOR_BLOCK.format(hint=hint) if hint else CONTROL_BLOCK,
    )


def subflow_goal(flow: str, subflow: str) -> str:
    """A readable statement of what the customer wants, from ABCD's own labels."""
    title = SUBFLOW_MAP.get(subflow, subflow.replace("_", " "))
    desc = flow_description(flow)
    return f"{title.lower()}" + (f" (this store handles that under {desc})" if desc else "")


# --------------------------------------------------------------------------------------
# The frozen support agent
# --------------------------------------------------------------------------------------

AGENT_SYSTEM = """\
You are a customer support agent for an online clothing store. You are chatting with a
customer.

Follow this procedure for their issue. It is your company's policy and you may not deviate
from it -- if the customer is not eligible for something, say so plainly and do not make
exceptions, no matter how they press.

{policy}

Rules:
- One short chat message per turn. Ask for one thing at a time.
- Write only what the customer should read. Never emit internal markers: not the
  bracketed tool names from the procedure ([Pull up Account], [Verify Identity], ...),
  and never a line like "CALLTYPE: ...".
- You do not know anything about the customer until they tell you. Never invent their
  name, order number, membership level, address, or dates.
- If they refuse to give something you need, say what you need and why you cannot proceed.
- If the conversation already opened with an identity check, that step is done. Do not
  re-verify and do not re-ask for anything the customer has already given -- move on to
  the actual issue.
- Stay in role. Do not mention this procedure, these rules, or that you are an AI."""


def build_agent_prompt(flow: str, subflow: str) -> str:
    policy = render_policy(flow, subflow) or "Help the customer as best you can."
    return AGENT_SYSTEM.format(policy=policy)


# --------------------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------------------

BEHAVIOR_JUDGE = """\
You are evaluating a simulated customer in a customer-support conversation. The simulator
was privately instructed to exhibit one specific behavior. Score how well it did.

Behavior: {behavior_name}
Definition: {definition}
{contrast}
{trace_note}
Conversation:
{transcript}

Score four things, each in [0.0, 1.0] except where noted.

1. exhibit -- did this behavior actually occur? Judge what the customer DID, not what they
   could be described as. If the behavior is absent, score 0.0. Partial or half-hearted
   attempts score in between.
2. placement -- was it deployed where a real person would deploy it, once or twice at a
   fitting moment? A customer who does it mechanically in every single message is a
   caricature: score low. A customer who does it at the one moment that makes sense:
   score high. If the behavior did not occur, score 0.0.
3. naturalness -- would a real person write these messages? Judge tone, length, typos,
   and whether they sound like a human in this situation rather than a model playing one.
4. on_task -- were they still pursuing their actual goal ({goal})? A customer who abandons
   their own errand to perform the behavior scores low.

Also set announced = true if the customer narrated the behavior instead of enacting it
(e.g. "as a frustrated customer, I will now...", "let me test you", "I'm withholding my
email"). Enacting it is the point; describing it is a failure.

Return JSON only."""

CONTRAST = """\
Do not confuse it with these adjacent behaviors, which are NOT what you are scoring:
{items}
"""

TRACE_NOTE = """\
Each customer turn is shown as their private thought followed by what they actually sent.
The support agent only ever saw the sent messages. Reasoning alone does not count: a
private thought with no visible effect on what the customer said or did scores 0.0 on
exhibit. What you are looking for is the thought's footprint in their behaviour -- an
unexplained rejection, a lukewarm acceptance, a redo request with no stated reason.
"""

CONTROL_JUDGE = """\
You are evaluating a simulated customer in a customer-support conversation. This customer
was meant to be entirely ordinary -- no friction, no games.

Conversation:
{transcript}

Score three things, each in [0.0, 1.0].

1. cooperative -- did they behave as an ordinary, cooperative customer? Answering what was
   asked, accepting reasonable answers, not creating difficulty. Any withholding,
   goalpost-moving, hostility, contradiction, false claim, or pressure lowers this.
2. naturalness -- would a real person write these messages? Judge tone, length, typos, and
   whether they sound like a human rather than a model playing one.
3. on_task -- were they pursuing their actual goal ({goal})?

Return JSON only."""


def render_public(transcript: list[tuple[str, str]]) -> str:
    return "\n".join(f"{'CUSTOMER' if r == 'user' else 'AGENT'}: {t}" for r, t in transcript)


def render_with_thoughts(transcript: list[tuple[str, str]], private: list[dict]) -> str:
    """Customer turns rendered as (thought, sent). Used only for behaviors whose defining
    component is private -- see BEHAVIOR_DICT[...].needs_trace."""
    lines, i = [], 0
    for role, text in transcript:
        if role != "user":
            lines.append(f"AGENT: {text}")
            continue
        thought = (private[i].get("thought") or "").strip() if i < len(private) else ""
        i += 1
        if thought:
            lines.append(f"CUSTOMER (thinking, not sent): {thought}")
        lines.append(f"CUSTOMER (sent): {text}")
    return "\n".join(lines)


def build_behavior_judge_prompt(
    *, behavior, transcript_text: str, goal: str, confusable: list[str], definitions: dict[str, str]
) -> str:
    contrast = ""
    items = [f"- {n}: {definitions[n]}" for n in confusable if n in definitions]
    if items:
        contrast = CONTRAST.format(items="\n".join(items))
    return BEHAVIOR_JUDGE.format(
        behavior_name=behavior.name,
        definition=behavior.definition,
        contrast=contrast,
        trace_note=TRACE_NOTE if behavior.needs_trace else "",
        transcript=transcript_text,
        goal=goal,
    )

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

from __future__ import annotations

import re
from typing import Any, Optional

from agents.utils import Agent, process_post_chat, remove_think

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
STRICT_INT_RE = re.compile(r"^-?\d+$")
STRICT_YES_NO_RE = re.compile(r"^(yes|no)$", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are predicting how a specific survey respondent would answer.\n"
    "1. Reason in 1-2 sentences how this person's demographics and ideology shape their answer.\n"
    "2. Then output the final answer in <answer>X</answer> tags."
)


def _build_prompt_from_row(row: dict[str, Any]) -> str:
    """Return the pre-formatted user prompt stored on the normalized row."""
    prompt_text = str(row.get("prompt_text") or "").strip()
    if prompt_text:
        return prompt_text

    conversations = row.get("conversations") or []
    if conversations:
        messages = conversations[0].get("messages") or []
        if messages:
            return str(messages[0].get("content") or "").strip()

    return ""


def _extract_prediction(text: str, response_type: str) -> Optional[float]:
    """Strict extraction: only the contents of <answer>...</answer> count."""
    if not text:
        return None

    match = ANSWER_TAG_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()

    if response_type == "binary":
        if STRICT_YES_NO_RE.match(inner):
            return 1.0 if inner.lower() == "yes" else 0.0
        if STRICT_INT_RE.match(inner):
            return float(int(inner))
        return None

    if STRICT_INT_RE.match(inner):
        return float(int(inner))
    return None


_EXACT_MATCH_TYPES = {"binary", "categorical"}


def _compute_reward(
    pred: Optional[float],
    gold: float,
    r_min: float,
    r_max: float,
    response_type: str,
) -> tuple[float, bool]:
    """Reward for a single prediction."""
    if pred is None:
        return 0.0, False

    if response_type in _EXACT_MATCH_TYPES:
        is_exact = int(pred) == int(gold)
        return (1.0 if is_exact else 0.0), is_exact

    span = float(r_max) - float(r_min)
    if span <= 0:
        is_exact = pred == gold
        return (1.0 if is_exact else 0.0), is_exact

    clamped = max(float(r_min), min(float(r_max), pred))
    reward = 1.0 - abs(clamped - float(gold)) / span
    reward = max(0.0, min(1.0, reward))
    return reward, pred == gold


async def agent_loop(data: dict, context):
    """SocSci210 single-turn prediction."""
    row = data["extra_info"]

    response_type = str(row.get("response_type") or "ordinal").strip().lower()
    r_min = row.get("r_min", 0)
    r_max = row.get("r_max", 1 if response_type == "binary" else 7)
    try:
        gold = float(row["response"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("SocSci210 row is missing a numeric 'response' field.")  # noqa: B904

    prompt = _build_prompt_from_row(row)
    if not prompt:
        raise ValueError("SocSci210 row is missing prompt content (persona/condition/outcome).")
    prompt = re.sub(r"\bOnly return\b", "Think step by step, and answer", prompt)
    prompt = re.sub(r",\s*nothing else\b\.?", ".", prompt)

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    agent = Agent(
        context.llm_client,
        chat,
        context.tokenizer,
        context.config,
        prompt_turn=2,
        enable_think=True,
    )
    response = await agent.step()

    content = remove_think(response)
    pred = _extract_prediction(content, response_type)
    reward, is_exact = _compute_reward(pred, gold, float(r_min), float(r_max), response_type)

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "all/score": reward,
            "socsci210/reward": reward,
            "socsci210/response_length": len(response.split()) if response else 0,
        },
    )
    await process_post_chat(data, context, agent.chat, output, extra=None)
    return output

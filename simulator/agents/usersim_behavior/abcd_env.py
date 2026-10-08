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

"""Frozen ABCD support agent -- the counterpart the trained user simulator talks to."""

from __future__ import annotations

import logging

from agents.usersim_behavior.prompts import build_agent_prompt
from agents.utils import call_openai, get_judge_reasoning, truncate_text

logger = logging.getLogger(__name__)

MAX_REPLY_WORDS = 200


class ABCDSupportAgent:
    """The store's support agent, bound to one scenario's flow/subflow policy."""

    def __init__(self, flow: str, subflow: str) -> None:
        self.system = build_agent_prompt(flow, subflow)
        self.failures = 0

    async def step(self, public_transcript: list[tuple[str, str]]) -> str | None:
        """Next agent message, or None if the gateway could not be reached."""
        messages = [{"role": "system", "content": self.system}]
        for role, text in public_transcript:
            messages.append({"role": "user" if role == "user" else "assistant", "content": text})
        try:
            reply = await call_openai(messages, reasoning_effort=get_judge_reasoning("minimal"))
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            logger.warning("[usersim_behavior] support agent call failed: %s: %s", type(e).__name__, e)
            return None
        reply = (reply or "").strip()
        if not reply:
            self.failures += 1
            return None
        for prefix in ("AGENT:", "Agent:", "agent:"):
            if reply.startswith(prefix):
                reply = reply[len(prefix) :].strip()
        return truncate_text(_dedupe(reply), MAX_REPLY_WORDS)


def _dedupe(text: str) -> str:
    """Collapse an exactly-doubled reply."""
    half, rest = divmod(len(text), 2)
    if rest == 0 and half > 20 and text[:half] == text[half:]:
        return text[:half]
    return text

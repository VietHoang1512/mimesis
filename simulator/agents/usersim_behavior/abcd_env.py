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
Frozen ABCD support agent -- the counterpart the trained user simulator talks to.

Follows the ``SimpleAgent`` pattern from ``agents/sotopia/agent.py:166``: not an
``Agent``, so it has no tokenizer, no masks, and contributes nothing to the loss. Two
deliberate departures:

* **Context is rebuilt from the public transcript every turn** rather than accumulated
  internally. The simulator's raw completions carry its reasoning, and there must be no
  path by which that reaches this agent -- rebuilding from the caller's public list makes
  the leak structurally impossible instead of merely avoided. Mirrorbench keeps a
  separate ``simulation_turns`` list for the same reason
  (``agents/mirrorbench/agent.py:600``).
* **It never raises.** ``api.chat_parse``/``chat_async`` raise after exhausting retries
  (``api.py:613``), nothing in sotopia catches it, and the agent-loop batch is gathered
  without ``return_exceptions=True`` (``verl/experimental/agent_loop/agent_loop.py:500``)
  -- so one persistent gateway failure takes down every rollout in the batch. Returning
  None instead ends this rollout only, and the loop scores the short transcript.
"""

from __future__ import annotations

import logging

from agents.usersim_behavior.prompts import build_agent_prompt
from agents.utils import call_openai, get_judge_reasoning, truncate_text

logger = logging.getLogger(__name__)

# Word cap for one agent turn (truncate_text counts words, agents/utils.py:457). ABCD
# agent turns are short; this only guards a model that recites the whole policy back.
MAX_REPLY_WORDS = 200


class ABCDSupportAgent:
    """The store's support agent, bound to one scenario's flow/subflow policy."""

    def __init__(self, flow: str, subflow: str) -> None:
        self.system = build_agent_prompt(flow, subflow)
        self.failures = 0

    async def step(self, public_transcript: list[tuple[str, str]]) -> str | None:
        """Next agent message, or None if the gateway could not be reached.

        `public_transcript` is [("user"|"agent", text), ...] -- the customer's *sent*
        messages only, never their reasoning.
        """
        messages = [{"role": "system", "content": self.system}]
        for role, text in public_transcript:
            messages.append({"role": "user" if role == "user" else "assistant", "content": text})
        try:
            reply = await call_openai(messages, reasoning_effort=get_judge_reasoning("minimal"))
        except Exception as e:  # noqa: BLE001 -- see module docstring
            self.failures += 1
            logger.warning("[usersim_behavior] support agent call failed: %s: %s", type(e).__name__, e)
            return None
        reply = (reply or "").strip()
        if not reply:
            self.failures += 1
            return None
        # Models occasionally prefix the speaker label; strip it so the transcript stays clean.
        for prefix in ("AGENT:", "Agent:", "agent:"):
            if reply.startswith(prefix):
                reply = reply[len(prefix) :].strip()
        return truncate_text(_dedupe(reply), MAX_REPLY_WORDS)


def _dedupe(text: str) -> str:
    """Collapse an exactly-doubled reply.

    ``api._responses_text`` (api.py:283) concatenates the text of *every* ``output_text``
    block across all message items, and the gateway routinely returns two identical
    message items on the first call of a conversation -- producing "...pull up your
    account.I can help with that. Please provide..." Every frozen counterpart in the repo
    inherits this (sotopia's SimpleAgent, mirrorbench's assistant); the real fix belongs in
    api.py, but a doubled agent turn confuses both the simulator and the judge, so collapse
    it here.
    """
    half, rest = divmod(len(text), 2)
    if rest == 0 and half > 20 and text[:half] == text[half:]:
        return text[:half]
    return text

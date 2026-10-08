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

import copy
import logging
import re
import uuid

from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should majorly focus on where the object has been moved to, and answer the question with the **most detailed position possible** e.g., the object is in A and A is in B, then you should answer 'A'. For the answer, use <answer>(put your answer here)</answer> and only include the most **detailed** location but not other information.

Below is the story and question (and optional extra information):
## Story
{story}

## Extra Information
(to help you better understand and answer the question)
{extra_info}

## Question
{question}{candidates_section}"""

CANDIDATES_SECTION = "\n\nPossible answers: {candidates}"


def parse_story(story_field) -> str:
    """Parse story field, which may be a Python list string or plain string."""
    if isinstance(story_field, list):
        return " ".join(story_field)
    try:
        parsed = eval(story_field)
        if isinstance(parsed, list):
            return " ".join(str(s) for s in parsed)
    except Exception:
        pass
    return str(story_field)


def build_prompt(row: dict) -> str:
    """Build the prompt for a single ParaToMI row."""
    story = parse_story(row["story"])
    question = row["question"]
    extra_info = row.get("extra_info", "")
    cands = row.get("cands", "")

    candidates_section = ""
    if cands and str(cands).strip():
        try:
            cand_list = eval(cands) if isinstance(cands, str) else cands
            if isinstance(cand_list, list) and cand_list:
                candidates_section = CANDIDATES_SECTION.format(candidates=", ".join(str(c) for c in cand_list))
        except Exception:
            candidates_section = CANDIDATES_SECTION.format(candidates=str(cands))

    return PROMPT_TEMPLATE.format(
        story=story,
        extra_info=extra_info,
        question=question,
        candidates_section=candidates_section,
    )


def extract_answer(response: str) -> str:
    """Extract content strictly from <answer>...</answer> tags. Returns '' if not found."""
    if not response:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _parse_candidates(cands) -> list[str]:
    """Parse cands field (Python-list string or list) into a list of strings."""
    if not cands:
        return []
    if isinstance(cands, list):
        return [str(c) for c in cands]
    try:
        parsed = eval(cands) if isinstance(cands, str) else cands
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except Exception:
        pass
    return [str(cands)]


def _contains_whole(needle: str, text: str) -> bool:
    """Whole-word containment, case-insensitive."""
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle.lower()) + r"(?!\w)", text.lower()) is not None


def evaluate_paratomi(predicted: str, correct_answer: str, cands) -> bool:
    if not predicted or not correct_answer:
        return False
    if not _contains_whole(correct_answer, predicted):
        return False
    wrong_cands = [c for c in _parse_candidates(cands) if c.strip().lower() != correct_answer.strip().lower()]
    for wc in wrong_cands:
        if _contains_whole(wc, predicted):
            return False
    return True


def compute_paratomi_aggregates(results: list[dict]) -> dict:
    """Compute accuracy breakdown by question type (qType)."""
    by_type: dict[str, list[float]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        qt = str(r.get("question_type", "unknown"))
        by_type.setdefault(qt, []).append(r.get("reward", 0.0))

    aggregates = {}
    all_rewards = []
    for qt, rewards in sorted(by_type.items()):
        aggregates[f"accuracy_{qt}"] = sum(rewards) / len(rewards)
        all_rewards.extend(rewards)
    if all_rewards:
        aggregates["accuracy_overall"] = sum(all_rewards) / len(all_rewards)
    return aggregates


async def agent_loop(data, context):
    """ParaToMI: Evaluate ToM location-tracking on a single story+question."""
    row = data["extra_info"]

    prompt = build_prompt(row)
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    content = remove_think(response)
    predicted = extract_answer(content)
    correct_answer = str(row.get("correct_answer", ""))
    is_correct = evaluate_paratomi(predicted, correct_answer, row.get("cands", ""))
    reward = 1.0 if is_correct else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "paratomi/reward": reward,
            "paratomi/response_length": len(response.split()) if response else 0,
            "all/score": reward,
            "all/score_v1": reward,
        },
    )

    extra = {}
    hint = None
    if getattr(context.config.algorithm, "agent_version", None) == "copy" and not is_correct:
        from agents.paratomi.hint import generate_hint

        hint = await generate_hint(row, content)
        if hint:
            extra["hint"] = hint

    if getattr(context.config.algorithm, "agent_version", None) == "copy" and context.is_train and hint:
        from agents.paratomi.hint_agent import agent_loop as hint_agent_loop

        data["extra_info"]["hint"] = hint
        data["extra_info"]["old_reward"] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = "hint_agent"
        output = [output, copy_agent_output, hint_agent_output]

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output

    return {
        "reward": reward,
        "chat": chat,
        "predicted": predicted,
        "correct": correct_answer,
        "is_correct": is_correct,
        "question_type": row.get("qType", ""),
        "index": str(row.get("index", "")),
    }

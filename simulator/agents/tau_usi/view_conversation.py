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

"""Pretty-print tau-USI rollout conversations for inspection."""
from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

ROLE_TAG = {"system": "SYSTEM", "assistant": "AGENT", "user": "USER"}


def _wrap(text: str, width: int, indent: str = "    ") -> str:
    out = []
    for line in str(text).splitlines() or [""]:
        out.extend(textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent) or [indent])
    return "\n".join(out)


def show_summary(recs: list[dict]) -> None:
    print(f"{'instance_id':22s} {'reward':>6} {'turns':>5} {'termination':>18}  {'sec':>6}")
    print("-" * 64)
    for r in recs:
        print(
            f"{r.get('instance_id', '?'):22s} {float(r.get('reward', 0)):>6.1f} "
            f"{len(r.get('conversation', [])):>5} {str(r.get('termination_reason', '')):>18}  "
            f"{r.get('elapsed_seconds', ''):>6}"
        )
    rewards = [float(r.get("reward", 0)) for r in recs]
    if rewards:
        print("-" * 64)
        print(f"{'MEAN':22s} {sum(rewards) / len(rewards):>6.2f} (agent success over {len(rewards)} tasks)")


def show_one(rec: dict, full: bool, width: int) -> None:
    turns = rec["chat"] if full else rec["conversation"]
    print("=" * width)
    print(f"instance: {rec.get('instance_id')}   domain: {rec.get('domain')}   task_index: {rec.get('task_index')}")
    print(f"reward: {rec.get('reward')}   termination: {rec.get('termination_reason')}   "
          f"turns(shown): {len(turns)}   user_sim: {rec.get('user_sim_model')}")
    print("=" * width)
    for m in turns:
        role = m.get("role", "?")
        tag = ROLE_TAG.get(role, role.upper())
        print(f"\n[{tag}]")
        print(_wrap(m.get("content", ""), width - 4))

    survey = rec.get("survey") or {}
    if survey:
        print("\n" + "-" * width + "\nSURVEY")
        for field, qa in survey.items():
            ans = qa.get("answer") if isinstance(qa, dict) else qa
            print(f"  {field:26s}: {ans}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a *_task_results.json file")
    ap.add_argument("-i", "--instance", default=None, help="instance_id to show (e.g. retail_0); omit for a summary")
    ap.add_argument("--full", action="store_true", help="show the full chat (system + tool calls + observations)")
    ap.add_argument("--width", type=int, default=0, help="wrap width (default: terminal width)")
    args = ap.parse_args(argv)

    width = args.width or min(shutil.get_terminal_size((100, 24)).columns, 120)
    payload = json.loads(Path(args.path).read_text())
    recs = payload.get("results", payload if isinstance(payload, list) else [])

    if not args.instance:
        show_summary(recs)
        return
    matches = [r for r in recs if str(r.get("instance_id", "")) == args.instance]
    if not matches:
        matches = [r for r in recs if str(r.get("instance_id", "")).startswith(args.instance)]
    if not matches:
        raise SystemExit(f"no task matching instance_id {args.instance!r}; run without -i to list them")
    for r in matches:
        show_one(r, args.full, width)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Install the UserRL gyms and apply our fixes.
#
# The gyms are ~270 MB of upstream code and task data, so they are referenced
# rather than vendored. patches/gyms.patch carries only our changes (13 files,
# 407 insertions) and is what the published numbers depend on:
#
#   * TravelGym: the async elicitation branch dropped the line that reads the
#     simulator's reply, so it raised UnboundLocalError and fell through to a
#     canned response worth 0 reward. 6.7% -> 0.1% of all turns.
#   * Persuade / Turtle / Telepathy / Search: _coerce_model_json, a four-tier
#     parser for simulator output that is meant to be JSON but is not (curly
#     quotes, apostrophes inside single-quoted dicts). Before it, every parse
#     failure was swallowed into the same zero-reward canned reply.
#   * All gyms: USERRL_SIM_TIMEOUT / USERRL_GYM_SEED overrides. The stock 10 s
#     timeout turns a slow simulator turn into that canned reply too.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

UPSTREAM="${USERRL_REPO:-https://github.com/SalesforceAIResearch/UserRL.git}"
SRC="${USERRL_SRC:-$AGENT_ROOT/third_party/UserRL}"

if [ ! -d "$SRC" ]; then
  echo "cloning $UPSTREAM -> $SRC"
  mkdir -p "$(dirname "$SRC")"
  git clone --depth 1 "$UPSTREAM" "$SRC"
fi

echo "applying patches/gyms.patch"
git -C "$SRC" apply --check "$AGENT_ROOT/patches/gyms.patch" \
  || { echo "patch does not apply -- upstream has moved; re-base patches/gyms.patch" >&2; exit 1; }
git -C "$SRC" apply "$AGENT_ROOT/patches/gyms.patch"

# tau-bench is a separate upstream that TauGym imports.
mkdir -p "$SRC/gyms/utils"
if [ ! -d "$SRC/gyms/utils/tau-bench" ]; then
  git clone --depth 1 https://github.com/sierra-research/tau-bench.git \
    "$SRC/gyms/utils/tau-bench"
fi
pip install -e "$SRC/gyms/utils/tau-bench"

# AlfworldGym and TemplateGym are not reachable from verl/tools/env_manager.py
# and are not part of any reported result, so they are not installed.
for gym in FunctionGym IntentionGym PersuadeGym SearchGym TauGym \
           TelepathyGym TravelGym TurtleGym; do
  echo "installing $gym"
  pip install -e "$SRC/gyms/$gym"
done

python -c "import functiongym, intentiongym, persuadegym, searchgym, taugym, \
telepathygym, travelgym, turtlegym; print('all gyms import OK')"

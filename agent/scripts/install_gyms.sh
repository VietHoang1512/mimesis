#!/usr/bin/env bash
# Install the UserRL gyms and apply our fixes.
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

for gym in FunctionGym IntentionGym PersuadeGym SearchGym TauGym \
           TelepathyGym TravelGym TurtleGym; do
  echo "installing $gym"
  pip install -e "$SRC/gyms/$gym"
done

python -c "import functiongym, intentiongym, persuadegym, searchgym, taugym, \
telepathygym, travelgym, turtlegym; print('all gyms import OK')"

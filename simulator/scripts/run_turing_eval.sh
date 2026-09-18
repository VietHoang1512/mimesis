#!/bin/bash
#SBATCH --job-name=turing
#SBATCH --nodes=1
##SBATCH --account=<your-account>
##SBATCH --qos=a100_dev
##SBATCH --qos=<your-qos>   # was: a100_genai_interns_high
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Turing-test eval of user-simulator models, using turing-rl's pairwise judge.
#
# For each heldout target the judge sees the user's history, the conversation
# context, and TWO candidate next turns -- the real human's and the model's, in a
# deterministic-but-randomized order -- and rates 1-7 which one a human wrote.
# See turing_eval.py for why we use their judge but not their eval wrappers.
#
# Two phases, deliberately separated because they scale in opposite directions:
#
#   1. GENERATE -- local vLLM, one model per GPU, fanned out across the node.
#      Embarrassingly parallel; more GPUs = linearly faster.
#   2. SCORE    -- every judge call hits ONE throttled the judge endpoint endpoint behind a
#      global pacer (shared/api_client._pace_request). Running these concurrently
#      just splits the same tokens/minute budget and trips 429s, so they run
#      strictly SERIALLY. This is the wall-clock bottleneck, not the GPUs.
#
# Prereq -- build the heldout set once (see turing-rl/bash_scripts/data):
#   cd turing-rl && bash bash_scripts/data/generate_data.sh prism gpt-5.6
#
# Usage:
#   sbatch run_turing_eval.sh                                  # full sweep
#   SMOKE=1 bash run_turing_eval.sh                            # 20 targets, 1 model
#   MODELS="/path/to/ckpt" LIMIT=300 bash run_turing_eval.sh   # override any knob
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "$MIMESIS_ROOT"
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"

# vLLM 0.24 auto-selects FlashInfer for top-k/top-p ("Using FlashInfer for top-p
# & top-k sampling") and dlopens a JIT-cached kernel out of ~/.cache/flashinfer.
# That cached sampling.so is linked against libcudart.so.12, but this cluster is
# CUDA 13 -- /usr/local/cuda -> cuda-13.0 and the env ships nvidia/cu13 only, so
# `ldd` on it says "libcudart.so.12 => not found". It therefore loads only when
# the submitting shell happens to carry a cu12 library path, which #SBATCH
# --export=ALL then forwards. Hence job 433711 died in _dummy_sampler_run during
# engine startup while an identical job 20h earlier came up fine.
#
# turing_eval.py sends temperature only -- no top_k, no top_p -- so FlashInfer's
# fast path buys nothing here and the native sampler is equivalent. Turning it
# off removes the landmine instead of betting on the launching shell.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

# Same --export=ALL hazard, second instance. A login shell exports
# https_proxy=http://10.0.2.2:<port> pointing at ITS OWN outbound proxy listener; the compute
# node stands up a different one (X2P_PORT), so the inherited port is dead there
# and every judge call fails with URLError until all 8 retries burn out -- job
# 434396 spent 90 minutes on 20 records and scored none of them. Compute nodes
# reach the judge endpoint directly (verified: POST returns 200 in 1.3s with these unset),
# so drop the stale values instead of guessing at a port.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# ── Heldout set ───────────────────────────────────────────────────────────────
# `history` conditions the sim on raw prior utterances only. The two persona
# variants exist too, but their personas were induced BY gpt-5.6 and the judge IS
# gpt-5.6 -- same model writing the character sketch and grading the imitation.
# `history` has no such coupling, so it is the default here.
#   ..._history_persona_gpt-5_6_s42_sft40_grpo60/test.parquet   (paper default)
#   ..._persona_gpt-5_6_s42_sft40_grpo60/test.parquet
TEST_PARQUET="${TEST_PARQUET:-turing-rl/data/prism/prism_history_s42_sft40_grpo60/test.parquet}"

# ── What to run ───────────────────────────────────────────────────────────────
SMOKE=${SMOKE:-0}
MODEL_ROOT=${MIMESIS_MODEL_DIR}
if [ "$SMOKE" = "1" ]; then
  MODELS="${MODELS:-$MODEL_ROOT/Qwen3.5-9B-CHATML-279675-ditto-rl-all-290192}"
  LIMIT="${LIMIT:-20}"
else
  # Ordered subject-first, then the tau-USI leaders, then RL variants, then bases.
  # Ordering is low-stakes: both phases skip artifacts that already exist, so a
  # requeue resumes rather than redoing.
  MODELS="${MODELS:-\
$MODEL_ROOT/Qwen3.5-9B-CHATML-279675-ditto-rl-all-290192}"
  # Subset for a sweep: the build shuffled with a fixed seed and _load_rows takes
  # the first N deterministically, so every model sees IDENTICAL rows. SE on
  # gt_accuracy is +-0.017 at 880, +-0.029 at 300, +-0.050 at 100 -- 300 still
  # separates models differing by >6 points, at a third of the judge cost.
  LIMIT="${LIMIT:-}"
fi

# Seeds to generate per model. Generation is at temperature 1.0, and --seed is
# forwarded per request to vLLM, so different seeds are independent draws and the
# same seed reproduces a run. Scoring is the expensive half (~30 min per model per
# seed, strictly serial behind the the judge endpoint pacer), so 3 seeds triples the judge
# bill: budget LIMIT alongside SEEDS. LIMIT=300 x 3 seeds costs about the same as
# the current 880 x 1 seed and buys a real variance estimate instead of none.
#   SEEDS=0 bash run_turing_eval.sh                  # legacy single-draw behaviour
#   SEEDS="0 1 2" LIMIT=300 sbatch run_turing_eval.sh
SEEDS="${SEEDS:-0 1 2}"

# The 28 runs already on disk are unseeded (generated before --seed existed).
# They are still a valid independent draw, so adopt them as seed 0 rather than
# throwing away ~14h of completed judge work -- otherwise the rename alone would
# force a full regenerate + rescore. Set ADOPT_LEGACY=0 to ignore them.
ADOPT_LEGACY=${ADOPT_LEGACY:-1}

# $MODEL_ROOT/Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329200-all-337915 \
# $MODEL_ROOT/osim-8b \
# $MODEL_ROOT/Ditto-8B \
# $MODEL_ROOT/sotopia-rl-qwen-2.5-7B-grpo \
# $MODEL_ROOT/Qwen3.5-4B-CHATML-279674-ditto-rl-all-288414 \
# $MODEL_ROOT/Qwen3.5-4B-CHATML-279674-ditto-rl-all-291189 \
# $MODEL_ROOT/Qwen3.5-4B-CHATML-291301-full-ditto-rl-all-300775 \
# $MODEL_ROOT/Qwen3.5-9B-CHATML-279675-ditto-rl-all-290192 \
# $MODEL_ROOT/Qwen3-4B \
# $MODEL_ROOT/Qwen3-4B-CHATML \
# $MODEL_ROOT/Qwen3-8B \
# $MODEL_ROOT/Qwen3-8B-CHATML \
# $MODEL_ROOT/Qwen3.5-9B
# $MODEL_ROOT/Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329201-adv-370850 \
# $MODEL_ROOT/Qwen3.5-4B \
# $MODEL_ROOT/Qwen3.5-4B-CHATML \
# $MODEL_ROOT/Qwen3.5-9B-CHATML \
# $MODEL_ROOT/humanlm-opinion \
# $MODEL_ROOT/osim-4b
# ── Knobs ─────────────────────────────────────────────────────────────────────
OUTDIR=${OUTDIR:-outputs/turing}
# Artifacts are keyed on model name alone, so a subset run must not share a
# namespace with the full sweep: LIMIT=300 drops a 300-row <model>_gen.json that
# a later full sweep cheerfully "skip generate (exists)"-es -- pinning that model
# at 300 rows forever -- and leaves behind a 300-row summary for aggregate to
# rank against 880-row ones. Give every subset its own directory.
if [ "$SMOKE" = "1" ]; then
  OUTDIR=${OUTDIR_SMOKE:-outputs/turing_smoke}
elif [ -n "$LIMIT" ]; then
  OUTDIR="${OUTDIR%/}_n$LIMIT"
fi
GEN_WORKERS=${GEN_WORKERS:-32}      # concurrent requests to the LOCAL vLLM
SCORE_WORKERS=${SCORE_WORKERS:-4}   # judge concurrency; the pacer is the real cap
MAX_TOKENS=${MAX_TOKENS:-16384}     # thinking models spend most of this on reasoning
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.9}
VLLM_MAX_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_STARTUP_TRIES=${VLLM_STARTUP_TRIES:-900}   # x2s => 30 min; cold torch.compile is slow
BASE_PORT=${BASE_PORT:-8100}
N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "${N_GPUS:-0}" -lt 1 ] && N_GPUS=1

mkdir -p "$OUTDIR/logs"
LIMIT_ARG=(); [ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")

N_MODELS=$(echo $MODELS | wc -w)
echo "[turing] $N_MODELS model(s), $N_GPUS GPU(s), heldout=$TEST_PARQUET ${LIMIT:+(first $LIMIT rows)}"
echo "[turing] phase 1 generate (parallel) -> phase 2 score (serial, throttled)"

wait_server_ready() {  # pid  url  max_tries -- bails the moment $pid dies
  local pid="$1" url="$2" tries="${3:-300}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 2
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# A model with no chat_template in its tokenizer serves fine but fails EVERY chat
# request, so several of the -CHATML SFT checkpoints would silently produce 880
# empty generations. Detect and fall back to ChatML (the template MIMESIS SFT
# used); instruct models ship their own and get "". Same guard as
# run_tau_usi_eval.sh; CHAT_TEMPLATE overrides for all models.
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
resolve_chat_template() {  # model -> echoes a --chat-template path, or empty
  if [ -n "$CHAT_TEMPLATE" ]; then echo "$CHAT_TEMPLATE"; return; fi
  local status
  status=$(python - "$1" <<'PY' 2>/dev/null
import sys
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    print("MISSING" if not getattr(tok, "chat_template", None) else "OK")
except Exception:
    print("OK")  # can't tell (e.g. gated/auth) -- let vLLM resolve it
PY
)
  [ "$status" = "MISSING" ] && echo "agents/tau_usi/chatml.jinja" || echo ""
}

# ── Phase 1: generate, up to N_GPUS models at a time ──────────────────────────
# One vLLM per model, ALL seeds generated against it before teardown -- starting
# the server is the expensive part (up to 13 min on a cold torch.compile cache),
# so restarting it per seed would dominate the phase.
declare -a GEN_FILES=()
idx=0
N_NEW=0
N_SKIP=0
for model in $MODELS; do
  base=$(basename "$model")

  # Adopt a pre---seed artifact as this model's seed 0 (see ADOPT_LEGACY).
  legacy="$OUTDIR/${base}_gen.json"
  legacy_sum="$OUTDIR/${base}_turing_summary.json"
  if [ "$ADOPT_LEGACY" = "1" ] && [ -s "$legacy" ] && [ ! -s "$OUTDIR/${base}-seed0_gen.json" ]; then
    echo "[turing] adopting unseeded $base as seed0"
    mv "$legacy" "$OUTDIR/${base}-seed0_gen.json"
    [ -s "$legacy_sum" ] && mv "$legacy_sum" "$OUTDIR/${base}-seed0_turing_summary.json"
    [ -s "$OUTDIR/${base}_turing_items.json" ] && \
      mv "$OUTDIR/${base}_turing_items.json" "$OUTDIR/${base}-seed0_turing_items.json"
  fi

  # Which seeds still need generating for this model?
  need=""
  for seed in $SEEDS; do
    gen="$OUTDIR/${base}-seed${seed}_gen.json"
    GEN_FILES+=("$gen")
    if [ -s "$gen" ]; then
      echo "[turing] skip generate (exists): $(basename "$gen")"
      N_SKIP=$((N_SKIP + 1))
    else
      need="$need $seed"
    fi
  done
  [ -z "$need" ] && { idx=$((idx + 1)); continue; }
  N_NEW=$((N_NEW + $(echo $need | wc -w)))

  gpu=$((idx % N_GPUS))
  port=$((BASE_PORT + gpu))
  tmpl=$(resolve_chat_template "$model")
  (
    [ -n "$tmpl" ] && echo "[turing] '$base' ships no chat template -> using $tmpl"
    echo "[turing] gpu$gpu :$port serving $base (seeds:$need)"
    vllm_args=(--port "$port" --served-model-name "$base"
               --tensor-parallel-size 1 --gpu-memory-utilization "$VLLM_GPU_UTIL"
               --max-model-len "$VLLM_MAX_LEN")
    [ -n "$tmpl" ] && vllm_args+=(--chat-template "$tmpl")
    CUDA_VISIBLE_DEVICES=$gpu vllm serve "$model" "${vllm_args[@]}" \
      > "$OUTDIR/logs/vllm_${base}.log" 2>&1 &
    vpid=$!
    if ! wait_server_ready "$vpid" "http://localhost:$port/health" "$VLLM_STARTUP_TRIES"; then
      echo "[turing] ERROR: vLLM for '$base' never came up; see $OUTDIR/logs/vllm_${base}.log"
      kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
      exit 1
    fi
    for seed in $need; do
      python turing_eval.py generate \
        --test-parquet "$TEST_PARQUET" \
        --model "$base" --base-url "http://localhost:$port/v1" \
        --workers "$GEN_WORKERS" --max-tokens "$MAX_TOKENS" --seed "$seed" \
        "${LIMIT_ARG[@]}" --out "$OUTDIR/${base}-seed${seed}_gen.json" \
        2>&1 | tee "$OUTDIR/logs/${base}-seed${seed}_generate.log"
    done
    kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
  ) &

  idx=$((idx + 1))
  # Barrier every N_GPUS launches so we never oversubscribe a GPU.
  [ $((idx % N_GPUS)) -eq 0 ] && wait
done
wait
echo "[turing] phase 1 done ($N_NEW attempted, $N_SKIP already complete)."

# A dead vLLM used to sail straight past here: the subshell's `exit 1` is
# invisible to the parent, phase 2 quietly skips the missing file, and phase 3
# still prints a full leaderboard assembled from OTHER models' artifacts. Job
# 433711 logged one ERROR line, exited 0, and SLURM called it COMPLETED. Collect
# the misses now and fail the job on them at the end -- after scoring, so one bad
# checkpoint does not throw away the models that did generate.
declare -a FAILED=()
for gen in "${GEN_FILES[@]}"; do
  [ -s "$gen" ] || FAILED+=("$(basename "$gen" _gen.json)")
done

# ── Phase 2: score, one at a time ─────────────────────────────────────────────
declare -a UNSCORED=()
for gen in "${GEN_FILES[@]}"; do
  [ -s "$gen" ] || { echo "[turing] no generations, skipping score: $gen"; continue; }
  base=$(basename "$gen" _gen.json)
  if [ -s "$OUTDIR/${base}_turing_summary.json" ]; then
    echo "[turing] skip score (exists): $base"; continue
  fi
  echo "[turing] === scoring $base ==="
  python turing_eval.py score --gen "$gen" --out-dir "$OUTDIR" \
    --workers "$SCORE_WORKERS" 2>&1 | tee "$OUTDIR/logs/${base}_score.log"
  # turing_eval.py bails rather than write an all-nan summary when every judge
  # call fails. Generations are already on disk, so a requeue resumes at scoring.
  [ -s "$OUTDIR/${base}_turing_summary.json" ] || UNSCORED+=("$base")
done

echo "[turing] building leaderboard"
python turing_eval.py aggregate --dir "$OUTDIR"
echo "[turing] done. Per-run artifacts + turing_summary.md under $OUTDIR/"

if [ ${#FAILED[@]} -gt 0 ] || [ ${#UNSCORED[@]} -gt 0 ]; then
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[turing] FAILED: ${#FAILED[@]}/$N_MODELS model(s) produced no generations --"
    for b in "${FAILED[@]}"; do
      echo "[turing]   $b  (see $OUTDIR/logs/vllm_$b.log)"
    done
  fi
  if [ ${#UNSCORED[@]} -gt 0 ]; then
    echo "[turing] UNSCORED: ${#UNSCORED[@]} model(s) generated but the judge failed --"
    for b in "${UNSCORED[@]}"; do
      echo "[turing]   $b  (see $OUTDIR/logs/${b}_score.log)"
    done
    echo "[turing]   generations are on disk; requeue to resume at scoring."
  fi
  exit 1
fi

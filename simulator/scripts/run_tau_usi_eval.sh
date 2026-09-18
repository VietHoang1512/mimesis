#!/bin/bash
#SBATCH --job-name=tau-usi
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
# Evaluate user-simulator models on the tau-USI benchmark.
#
# Pipeline (see agents/tau_usi/README.md):
#   1. Start the LOCAL tau-bench runtime service on :8005 (agents/tau_usi/runtime_service.py).
#   2. For each model: `vllm serve` it OpenAI-compatibly, run the tau_usi rollout
#      (agents.tau_usi.run_eval) over the 165 tasks with that model as the user-sim,
#      then score it (agents.tau_usi.usi_metric). The FIXED agent it talks to is
#      gpt-5-5 via the hosted endpoint (agents/tau_usi/agent.py::_agent_respond).
#   3. Aggregate all runs into one table (agents.tau_usi.aggregate_usi).
#
# The tested user-sim runs on a local vLLM server (OPENAI_AGENT_BASE_URL), so no VPN /
# the configured endpoint is needed; the fixed agent + survey-reformat judge go through the hosted endpoint
# (WITH_X2P=1). Nothing here needs OPENAI_BASE_URL — it's unset to force the hosted endpoint
# transport for the agent.
#
# Usage:
#   SMOKE=1 bash run_tau_usi_eval.sh            # 1 model x 5 tasks x 1 seed (staged smoke test)
#   sbatch run_tau_usi_eval.sh                  # full sweep (all models x 165 tasks x 3 seeds)
#   MODELS="cmu-lti/osim-8b" SEEDS=0 bash run_tau_usi_eval.sh   # override any knob
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
# rl2 is a clone of rl with tau-bench installed (the runtime service needs it).
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "$MIMESIS_ROOT"
# ── Ports ─────────────────────────────────────────────────────────────────────
RUNTIME_PORT=${RUNTIME_PORT:-8005}
VLLM_PORT=${VLLM_PORT:-8100}

# ── Fixed agent + judge (the shared API client) + data + runtime ──────────────────────────
export TAU_USI_DATA_DIR="$PWD/data/tau-usi/data"          # scorer reads tau_bench_tasks_unified.json here
export RUNTIME_SERVICE_URL="http://localhost:$RUNTIME_PORT"
export TAU_USI_AGENT_MODEL="${TAU_USI_AGENT_MODEL:-gpt-5-5}"
export TAU_USI_AGENT_REASONING_EFFORT="${TAU_USI_AGENT_REASONING_EFFORT:-low}"
export OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-gpt-5-5}"   # survey-reformat judge (call_openai)
export JUDGE_MODEL_NAME="$OPENAI_MODEL_NAME"
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
unset OPENAI_BASE_URL                                      # force the shared API client transport for the agent
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}
# That line covers ATTENTION only. vLLM picks FlashInfer for top-k/top-p
# SEPARATELY, and dlopens a JIT-cached kernel out of ~/.cache/flashinfer that is
# linked against libcudart.so.12 -- but this cluster is CUDA 13 (/usr/local/cuda
# -> 13.0, env ships nvidia/cu13 only), so the load fails and EngineCore dies
# during startup. It took out BOTH 9B checkpoints on Sep 3 despite TRITON_ATTN
# above; see outputs/tau_usi/logs/vllm_*-428848.log and *-290192.log. It only
# ever came up when the submitting shell happened to carry a cu12 library path,
# which `--export=ALL` then forwarded -- hence the intermittency. Same guard as
# run_turing_eval.sh.
#
# The proxy half of that same --export=ALL hazard bites here too. api.py's AI
# Gateway calls are already immune -- they pass --proxy http://localhost:$X2P_PORT
# to curl explicitly, and set trust_env=False on the SDK path -- but nothing
# protects the OTHER network client in this pipeline: AutoTokenizer.from_pretrained
# -> list_repo_templates -> huggingface_hub -> httpx, which DOES honour https_proxy
# and hangs on the login shell's dead port. That killed job 435178 before a single
# task ran. Compute nodes reach both hosts directly (verified: huggingface.co 307
# in 0.12s, the configured endpoint 200 in 1.3s with these unset; via the inherited proxy
# both hang). X2P_PORT / X2P_PROXY_URL are deliberately left alone -- api.py reads
# them to find the per-job agent.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.}"
export USER_SIM_ENABLE_THINKING="${USER_SIM_ENABLE_THINKING:-1}"
# TURNOFF_THINK is the INVERSE of USER_SIM_ENABLE_THINKING and must track it — the two are
# halves of one decision, and pinning them independently breaks whichever mode you aren't
# running:
#   USER_SIM_ENABLE_THINKING -> whether CallAPI lets vLLM render the THINKING template.
#   TURNOFF_THINK            -> how the harness CLEANS the completion, via
#                               AgentContext.enable_think (agents/utils.py:911).
# enable_think=1 routes clean_response to split_think_response, which returns "" whenever no
# </think> closes. With thinking ON that is what you want — it is the only path safe on a
# template-prefilled opener truncated at RESPONSE_LEN, where no tag survives in the
# completion for remove_think's regex to anchor on. But with thinking OFF every turn is
# untagged, so the same rule blanks ALL of them and the run silently scores empty.
# Mismatched flags also render the local token-bookkeeping template with thinking OFF while
# vLLM serves it ON, and make think_stats grade under the wrong rule.
export TURNOFF_THINK=$([ "$USER_SIM_ENABLE_THINKING" = "1" ] && echo 0 || echo 1)
# ── What to run ───────────────────────────────────────────────────────────────
SMOKE=${SMOKE:-0}
DOMAINS_FULL="retail:0-114,airline:0-49"   # the complete 165-task benchmark
if [ "$SMOKE" = "1" ]; then
  MODELS="${MODELS:-sunweiwei/Ditto-8B}"
  SEEDS="${SEEDS:-0}"
  DOMAINS="${DOMAINS:-retail:0-2,airline:0-1}"
else
  # Same user-sim set as run_eval.sh (active, non-commented models).
  MODELS="${MODELS:-${MIMESIS_MODEL_DIR}/Qwen3.5-9B-CHATML-290213-original-chat-thoughttrace-fix-426855-all-428848}"
  SEEDS="${SEEDS:-0 1 2}"
  DOMAINS="${DOMAINS:-$DOMAINS_FULL}"
fi

# ── Rollout / serving knobs ───────────────────────────────────────────────────
WORKERS=${WORKERS:-16}                 # concurrent tasks (agent API is the bottleneck, not vLLM)
VLLM_TP=${VLLM_TP:-1}                   # tensor-parallel size for the served user-sim
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.9}
VLLM_MAX_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_STARTUP_TRIES=${VLLM_STARTUP_TRIES:-900}  # readiness poll ceiling (x2s => 30 min); big multimodal /
                                               # torch.compile models take ~13 min on a cold cache — the old
                                               # 300 (10 min) timed out and skipped them. kill -0 still bails
                                               # instantly if the process dies, so a high ceiling is free.
PROMPT_LEN=${PROMPT_LEN:-24000}        # context budget for a user-sim turn (prompt side)
# Generation budget. NOT a per-turn cap: Agent.step (agents/utils.py:1268) computes
# max_len = prompt_ids_len + RESPONSE_LEN with prompt_ids_len FROZEN at init, and CallAPI
# subtracts the full current context — so the effective cap shrinks as the conversation
# grows and reaches zero once the sim's turns plus the agent's replies exceed RESPONSE_LEN.
# create_completion then returns None -> step() None -> empty_user_response, ending the
# task early and scoring a truncated dialogue. 2048 is fine for a no-think sim (short
# utterances) but far too small once every turn also carries a reasoning block, so track
# the thinking flag and match run_rl.sh's 16k, as run_eval.sh now does.
RESPONSE_LEN=${RESPONSE_LEN:-$([ "$USER_SIM_ENABLE_THINKING" = "1" ] && echo $((1024 * 16)) || echo 2048)}
# RESPONSE_LEN=2048
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}       # force this jinja for ALL local models (else auto-detected per model)
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}   # extra flags passed verbatim to `vllm serve` (e.g. "--trust-remote-code")

# API user-sim mode: if set, evaluate these models via the hosted endpoint (no vLLM, no GPU) —
# used for smoke test B (USER_SIM_API=gpt-5-5) and by run_tau_usi_eval_api.sh.
API_MODELS="${USER_SIM_API:-}"

# Smoke runs (5 tasks) go to a SEPARATE dir so their partial results never pool into the
# real 165-task leaderboard under outputs/tau_usi. Override OUTDIR to force a location.
if [ "$SMOKE" = "1" ]; then
  OUTDIR=${OUTDIR:-outputs/tau_usi_smoke}
elif [ "$DOMAINS" != "$DOMAINS_FULL" ]; then
  # Same reasoning, for the other way to shrink a run. The label encodes
  # repetition/thinking/resp/seed but NOT the task subset, and aggregate_usi's
  # table has no task-count column at all (only seeds) -- so an 11-task run
  # overwrites a 165-task row, or outranks one, with nothing on screen to show
  # it was measured over a fourteenth of the benchmark.
  OUTDIR=${OUTDIR:-outputs/tau_usi_$(echo "$DOMAINS" | tr -c '[:alnum:]' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')}
else
  OUTDIR=${OUTDIR:-outputs/tau_usi}
fi
mkdir -p "$OUTDIR/logs"

# ── Start the runtime service ─────────────────────────────────────────────────
echo "[tau_usi] starting tau-bench runtime service on :$RUNTIME_PORT"
python -m agents.tau_usi.runtime_service --port "$RUNTIME_PORT" > "$OUTDIR/logs/runtime_service.log" 2>&1 &
RUNTIME_PID=$!
VLLM_PID=""
cleanup() {
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
  kill "$RUNTIME_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

wait_http() {  # url  max_tries
  local url="$1" tries="${2:-180}" i
  for ((i = 0; i < tries; i++)); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

wait_server_ready() {  # pid  url  max_tries — like wait_http but bails the moment $pid dies
  local pid="$1" url="$2" tries="${3:-300}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 2   # process exited (e.g. bad args / OOM) — don't wait
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if ! wait_http "http://localhost:$RUNTIME_PORT/health" 60; then
  echo "[tau_usi] ERROR: runtime service did not become healthy; see $OUTDIR/logs/runtime_service.log"
  exit 1
fi
echo "[tau_usi] runtime service healthy."

# ── Rollout + score one (model, seed) ─────────────────────────────────────────
# Anything that did not make it all the way to a scored result lands in one of
# these, and the exit status at the bottom reports them. Without that, a model
# whose vLLM never came up produced one ERROR line and then a full leaderboard
# built from OTHER models' artifacts, and the job exited 0 -- which is how the
# Sep 3 FlashInfer breakage went unnoticed through two SLURM jobs.
declare -a FAILED=()      # vLLM never served the model
declare -a UNSCORED=()    # served, but the rollout produced no results

rollout_and_score() {  # model  seed
  local model="$1" seed="$2" base label out
  base=$(basename "$model")
  # Tag the label with the knobs that change what gets measured, so runs under different
  # settings land in separate files / leaderboard rows instead of overwriting each other:
  #   thinking-<n> -- USER_SIM_ENABLE_THINKING (0 => agents/utils.py sends
  #                   chat_template_kwargs enable_thinking=False)
  #   resp<n>      -- RESPONSE_LEN, the whole-conversation generation budget. A thinking
  #                   run at 16k and a 2048 run are not comparable: the smaller budget
  #                   drains mid-task and ends it in empty_user_response (see above).
  # -seed<N> must stay LAST: aggregate_usi.py:60 groups seeds by stripping /-seed\d+$/.
  label="${base}-repetition-${REPETITION_PENALTY}-thinking-${USER_SIM_ENABLE_THINKING:-0}-resp${RESPONSE_LEN}-seed${seed}"
  out="$OUTDIR/${label}_task_results.json"
  export EVAL_SEED=$seed
  echo "[tau_usi] === $label : rollout over $DOMAINS (workers=$WORKERS) ==="
  python -m agents.tau_usi.run_eval \
    --user-sim-model "$model" \
    --domains "$DOMAINS" \
    --workers "$WORKERS" \
    --prompt-length "$PROMPT_LEN" \
    --response-length "$RESPONSE_LEN" \
    --out "$out" 2>&1 | tee "$OUTDIR/logs/${label}_rollout.log"
  # Scoring an absent/empty rollout just writes a degenerate metrics file that
  # then shows up in the leaderboard as if it were a real run.
  if [ ! -s "$out" ]; then
    echo "[tau_usi] ERROR: no rollout results at $out -- skipping score for $label"
    UNSCORED+=("$label")
    return
  fi
  echo "[tau_usi] === $label : scoring ==="
  python -m agents.tau_usi.usi_metric score "$out" --label "$label" --out-dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/logs/${label}_score.log"
}

if [ -n "$API_MODELS" ]; then
  # ── API user-sims via the hosted endpoint (no vLLM) ──────────────────────────────────
  # CallAPI uses the hosted endpoint curl transport when OPENAI_AGENT_BASE_URL is unset,
  # and routes by model family: claude-* to Anthropic /v1/messages, gemini-* to
  # Vertex :generateContent, everything else to the Azure Responses route (see
  # api.py). Needs WITH_X2P=1 (sbatch) for the proxy to be reachable.
  unset OPENAI_AGENT_BASE_URL
  export OPENAI_AGENT_REASONING_EFFORT="${OPENAI_AGENT_REASONING_EFFORT:-low}"
  echo "[tau_usi] API user-sim mode: $API_MODELS"
  for model in $API_MODELS; do
    export OPENAI_AGENT_MODEL="$model"
    for seed in $SEEDS; do rollout_and_score "$model" "$seed"; done
  done
else
  # ── Local models: serve via vLLM → rollout(×seeds) → score → teardown ───────
  unset OPENAI_AGENT_REASONING_EFFORT   # local vLLM models don't accept a reasoning_effort param

  # A model with no chat_template in its tokenizer serves fine but fails EVERY chat
  # request; detect that and fall back to ChatML (the template MIMESIS SFT used). An
  # explicit CHAT_TEMPLATE overrides. Instruct models (Qwen/Llama) ship their own → "".
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
    print("OK")  # can't tell (e.g. gated/auth) — let vLLM resolve it
PY
)
    [ "$status" = "MISSING" ] && echo "agents/tau_usi/chatml.jinja" || echo ""
  }

  for model in $MODELS; do
    base=$(basename "$model")
    tmpl=$(resolve_chat_template "$model")
    [ -n "$tmpl" ] && echo "[tau_usi] '$model' ships no chat template -> using $tmpl"
    echo "[tau_usi] serving user-sim '$model' on :$VLLM_PORT (tp=$VLLM_TP)"
    vllm_args=(--port "$VLLM_PORT" --served-model-name "$model"
               --tensor-parallel-size "$VLLM_TP" --gpu-memory-utilization "$VLLM_GPU_UTIL"
               --max-model-len "$VLLM_MAX_LEN" $VLLM_EXTRA_ARGS)
    [ -n "$tmpl" ] && vllm_args+=(--chat-template "$tmpl")
    vllm serve "$model" "${vllm_args[@]}" > "$OUTDIR/logs/vllm_${base}.log" 2>&1 &
    VLLM_PID=$!

    if ! wait_server_ready "$VLLM_PID" "http://localhost:$VLLM_PORT/health" "$VLLM_STARTUP_TRIES"; then
      echo "[tau_usi] ERROR: vLLM for '$model' did not come up; see $OUTDIR/logs/vllm_${base}.log — skipping"
      kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""
      FAILED+=("$base")
      continue
    fi
    echo "[tau_usi] vLLM ready for '$model'."

    export OPENAI_AGENT_BASE_URL="http://localhost:$VLLM_PORT/v1"   # CallAPI -> local vLLM (OpenAI SDK transport)
    export OPENAI_AGENT_MODEL="$model"
    export OPENAI_AGENT_API_KEY="${OPENAI_AGENT_API_KEY:-EMPTY}"    # SDK requires a key; vLLM ignores its value

    for seed in $SEEDS; do rollout_and_score "$model" "$seed"; done

    kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""
    unset OPENAI_AGENT_BASE_URL OPENAI_AGENT_MODEL
  done
fi

echo "[tau_usi] building summary table"
python -m agents.tau_usi.aggregate_usi --dir "$OUTDIR"
echo "[tau_usi] done. Per-run artifacts + usi_summary.md under $OUTDIR/"

if [ ${#FAILED[@]} -gt 0 ] || [ ${#UNSCORED[@]} -gt 0 ]; then
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[tau_usi] FAILED: ${#FAILED[@]} model(s) never served --"
    for b in "${FAILED[@]}"; do
      echo "[tau_usi]   $b  (see $OUTDIR/logs/vllm_$b.log)"
    done
  fi
  if [ ${#UNSCORED[@]} -gt 0 ]; then
    echo "[tau_usi] UNSCORED: ${#UNSCORED[@]} run(s) produced no rollout results --"
    for l in "${UNSCORED[@]}"; do
      echo "[tau_usi]   $l  (see $OUTDIR/logs/${l}_rollout.log)"
    done
  fi
  exit 1
fi

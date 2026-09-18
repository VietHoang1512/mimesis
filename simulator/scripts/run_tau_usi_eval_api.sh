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
# Evaluate API models AS THE USER-SIMULATOR on tau-USI. The tau-USI analogue of
# run_eval_api.sh, and it evaluates the same five models.
#
# It reuses the WHOLE tau-USI pipeline in run_tau_usi_eval.sh via that script's
# USER_SIM_API mode: local tau-bench runtime service -> rollout (this model as the
# user-sim, fixed gpt-5-5 agent) -> usi_metric score -> aggregate_usi. Both the tested
# model AND the fixed agent go through the hosted endpoint (WITH_X2P=1), so — unlike run_eval_api.sh,
# which still boots a vLLM tokenizer on GPU — this needs NO GPU at all. (If your qos
# requires a GPU allocation, uncomment the two lines below.)
##SBATCH --gres=gpu:1
##SBATCH --gpus-per-node=1

# Cluster-specific SLURM settings. The published runs used Meta's internal
# partitions; replace the account/qos below with your own, or launch the
# `python3 train_ppo.py` / eval invocation directly without sbatch.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Usage:
#   sbatch run_tau_usi_eval_api.sh                             # 5 models x 3 seeds x 165 tasks
#   MODELS="gpt-5.5" SEEDS=0 bash run_tau_usi_eval_api.sh      # quick single-model run
#   SMOKE=1 MODELS="gemini-3.6-flash" bash run_tau_usi_eval_api.sh   # 5-task canary

# API user-sim models to evaluate, as the hosted endpoint registry names. CallAPI routes each
# family automatically (api.py): gpt-* -> Azure Responses, claude-* -> Anthropic
# /v1/messages, gemini-* -> Vertex :generateContent.
#
# Same five models as run_eval_api.sh so the two benchmarks stay directly comparable.
# The old default (claude-opus-4.8) and the grok-4 this once listed are no longer the
# set: grok-4 has no route on any gateway, and gemini — dropped during the the shared API client
# era — is reachable again on the `confucius-vertex-gemini` upstream.
export USER_SIM_API="${MODELS:-gpt-5.5 claude-opus-5 claude-sonnet-5 gemini-3.6-flash gemini-3.1-pro-preview}"
export SEEDS="${SEEDS:-0 1 2}"
# Reasoning effort for the Responses/Vertex user-sims (ignored by /v1/messages models,
# whose API has no equivalent field).
export OPENAI_AGENT_REASONING_EFFORT="${OPENAI_AGENT_REASONING_EFFORT:-low}"

# A tau-USI (model, seed) rollout over 165 tasks costs ~20 min at WORKERS=16, so the
# default sweep is ~5h. Print the plan before handing off rather than discovering the
# size from the log three models in.
N_MODELS=$(echo $USER_SIM_API | wc -w)
N_SEEDS=$(echo $SEEDS | wc -w)
N_RUNS=$((N_MODELS * N_SEEDS))
echo "[tau_usi_api] $N_MODELS model(s) x $N_SEEDS seed(s) = $N_RUNS run(s)"
echo "[tau_usi_api] rough estimate at ~20 min/run: $((N_RUNS * 20 / 60))h $((N_RUNS * 20 % 60))m"
echo "[tau_usi_api] models: $USER_SIM_API"

# Hand off to the shared pipeline. USER_SIM_API being set selects its no-vLLM API branch;
# SMOKE / SEEDS / DOMAINS / WORKERS all pass through as env. exec so the SBATCH header
# above (this job) governs, not run_tau_usi_eval.sh's. Absolute path — under sbatch $0 is
# a spool copy, so dirname "$0" wouldn't resolve to the repo.
exec bash "$MIMESIS_ROOT/scripts/run_tau_usi_eval.sh"

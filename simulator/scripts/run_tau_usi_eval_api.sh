#!/bin/bash
#SBATCH --job-name=tau-usi
#SBATCH --nodes=1
##SBATCH --account=<your-account>
##SBATCH --qos=<your-qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

export USER_SIM_API="${MODELS:-gpt-5.5 claude-opus-5 claude-sonnet-5 gemini-3.6-flash gemini-3.1-pro-preview}"
export SEEDS="${SEEDS:-0 1 2}"
export OPENAI_AGENT_REASONING_EFFORT="${OPENAI_AGENT_REASONING_EFFORT:-low}"

N_MODELS=$(echo $USER_SIM_API | wc -w)
N_SEEDS=$(echo $SEEDS | wc -w)
N_RUNS=$((N_MODELS * N_SEEDS))
echo "[tau_usi_api] $N_MODELS model(s) x $N_SEEDS seed(s) = $N_RUNS run(s)"
echo "[tau_usi_api] rough estimate at ~20 min/run: $((N_RUNS * 20 / 60))h $((N_RUNS * 20 % 60))m"
echo "[tau_usi_api] models: $USER_SIM_API"

exec bash "$MIMESIS_ROOT/scripts/run_tau_usi_eval.sh"

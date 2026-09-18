# MIMESIS — training and evaluating user simulators

Code to reproduce the MIMESIS user-simulator models: supervised fine-tuning on
reasoning traces, reinforcement learning over a 24-task mixture, and evaluation
on four benchmarks.

This is a fork of [verl](https://github.com/volcengine/verl) **v0.7.0**. The
`verl/` tree is vendored whole so the artifact runs without a separate install;
the 17 files we changed are listed at the bottom.

## Install

```bash
conda env create -f environment.yml && conda activate mimesis
pip install -e .
```

Then point `scripts/env.sh` at your storage — four variables, no absolute paths
anywhere else:

```bash
export MIMESIS_MODEL_DIR=/path/to/checkpoints
export MIMESIS_DATA_DIR=/path/to/data
export MIMESIS_OUTPUT_DIR=/path/to/outputs
export OPENAI_AGENT_BASE_URL=http://localhost:8000/v1   # judge + partner model
```

The judge and conversational partner are reached over any OpenAI-compatible
API — vLLM, SGLang, or a hosted provider. See `llm/README.md` for the two
behaviours worth preserving if you adapt that client (retry budget and empty
completions), both of which cost us silently-corrupted runs before they were
understood.

## Train

Three stages. Both supervised stages role-swap every conversation so the *human*
side occupies the generated position, and mask the loss to those turns only: the
model is optimised to produce the person, not the assistant.

```bash
bash scripts/run_sft.sh        # stage 1: mid-training on the 62-corpus mixture
bash scripts/run_trl_sft.sh    # stage 2: SFT on ThoughtTrace reasoning traces
bash scripts/run_rl.sh         # stage 3: RL over the 24-task mixture
```

Stage 1 trains surface behaviour at scale with reasoning switched off. Stage 2
introduces the native Qwen3.5 thinking template and supervises a `<think>` block
built from the annotator's stated *reasons*. Stage 1 writes sharded FSDP checkpoints; merge the one you want into HuggingFace
format before stage 2 can load it:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir "$MIMESIS_OUTPUT_DIR/<stage1-run>/global_step_5000" \
  --target_dir "$MIMESIS_MODEL_DIR/mimesis-9b-midtrain"
```

Before a long stage-2 run, `INSPECT_ONLY=1 bash scripts/run_trl_sft.sh` renders a
conversation on CPU and checks the loss mask lands on the human's turns.

RL is FoldGRPO: group-relative advantages with `gen_uid` de-duplication, so a
multi-turn trajectory is not counted once per turn within its group.



## Evaluate

```bash
bash scripts/run_eval.sh              # SOUL — 27 datasets, five capability axes
bash scripts/run_tau_usi_eval.sh      # tau-USI — agent success + USI vs 495 human dialogues
python eval/realusersim_eval.py       # RealUserSim PT3 — five fidelity dimensions
python eval/turing_eval.py            # Turing-reward
```

Scoring for SOUL is two steps: `scoring/aggregate.py` per run, then
`scoring/aggregate_seeds.py` across seeds.

### SimulatorArena

Not vendored — it is a 6 GB third-party repo. `eval/simarena/SETUP.md` has the
three steps: clone `microsoft/SimulatorArena`, apply `utils.patch`, copy in
`mimesis_transport.py`.

### Data

`data/` carries the tau-USI and RealUserSim snapshots. The SOUL training and
validation parquets (`sim_rl_data/`, `sim_eval_data/`) are released separately;
set `MIMESIS_DATA_DIR` to the directory containing them. The two supervised
training corpora are fetched the same way:

```bash
huggingface-cli download cmu-lti/osim-mid-training \
  --repo-type dataset --local-dir "$MIMESIS_DATA_DIR/osim_mid_training"
huggingface-cli download cmu-lti/mimesis-thoughttrace \
  --repo-type dataset --local-dir "$MIMESIS_DATA_DIR/thoughttrace"
```

## What we changed in verl v0.7.0

17 files, ~1,800 lines.

| file | Δ | what |
|---|---|---|
| `trainer/ppo/core_algos.py` | +169 | **FoldGRPO** advantage estimator (`compute_foldgrpo_advantage`) |
| `workers/actor/dp_actor.py` | +416/−3 | loss paths for the above |
| `utils/experimental/torch_topk_functional.py` | +377 | new: memory-efficient top-k |
| `trainer/ppo/ray_trainer.py` | +262/−31 | multi-turn agent-loop rollout plumbing |
| `workers/fsdp_workers.py` | +224/−9 | hybrid-engine memory handling |
| `experimental/agent_loop/agent_loop.py` | +105/−11 | per-task agent-loop dispatch |
| `models/transformers/{dense_common,qwen3_vl}.py` | +85/−16 | Qwen3.5 attention path |
| `trainer/ppo/metric_utils.py` | +42/−2 | per-task validation metrics |
| `utils/tracking.py` | +31 | run metadata |
| `workers/rollout/vllm_rollout/*.py` | +42/−10 | sleep mode, chat-template kwargs |
| `trainer/sft_trainer.py`, `trainer/config/sft_trainer_engine.yaml` | +3/−2 | honour `data.num_workers` instead of hardcoding 8 |
| `models/transformers/monkey_patch.py` | +3/−1 | import `AutoModelForCausalLMWithValueHead` from `trl.experimental.ppo`, which modern TRL requires |
| `utils/{model,tokenizer}.py`, `trainer/main_ppo.py`, `model_merger/…`, `workers/config/rollout.py` | +40/−13 | small fixes |

## Layout

```
agents/      19 task agent loops + reward functions; base_agent.py routes data_source
sft/         mid-training + ThoughtTrace SFT: dataset, role swap, TRL entry point
recipe/ditto eval.sh (27 SOUL datasets) and the RL recipe
verl/        vendored verl v0.7.0 fork
llm/         judge / partner client (see llm/README.md)
scoring/     aggregate.py, aggregate_seeds.py, export_csv.py
eval/        RealUserSim, Turing, SimulatorArena integration
scripts/     launchers; env.sh holds every path
data/        tau-USI and RealUserSim snapshots
```

# MIMESIS — training and evaluating user simulators

Code to reproduce the MIMESIS user-simulator models: supervised fine-tuning on
reasoning traces, reinforcement learning over a 24-task mixture, and evaluation
on four benchmarks.

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
completions).

## Train

Three stages. Both supervised stages role-swap every conversation so the *human*
side occupies the generated position, and mask the loss to those turns only: the
model is optimised to produce the person, not the assistant.

```bash
bash scripts/run_sft.sh        # stage 1: mid-training on the 62-corpus mixture
bash scripts/run_trl_sft.sh    # stage 2: SFT on ThoughtTrace reasoning traces
bash scripts/run_rl.sh         # stage 3: RL over the 24-task mixture
```

## Evaluate

```bash
bash scripts/run_eval.sh              # SOUL — 27 datasets, five capability axes
bash scripts/run_tau_usi_eval.sh      # tau-USI — agent success + USI vs 495 human dialogues
python eval/realusersim_eval.py       # RealUserSim PT3 — five fidelity dimensions
python eval/turing_eval.py            # Turing-reward
```

### SimulatorArena

Not vendored — it is a 6 GB third-party repo. `eval/simarena/SETUP.md` has the
three steps: clone `microsoft/SimulatorArena`, apply `utils.patch`, copy in
`mimesis_transport.py`.


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

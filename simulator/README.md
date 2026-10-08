# MIMESIS: training and evaluating user simulators

Code to reproduce the MIMESIS user-simulator models: supervised fine-tuning on
reasoning traces, reinforcement learning over a 24-task mixture, and evaluation
on four benchmarks.

## Install

```bash
conda env create -f environment.yml && conda activate mimesis
pip install -e .
```

Then set the four variables in `scripts/env.sh` (no other file contains
absolute paths):

```bash
export MIMESIS_MODEL_DIR=/path/to/checkpoints
export MIMESIS_DATA_DIR=/path/to/data
export MIMESIS_OUTPUT_DIR=/path/to/outputs
export OPENAI_AGENT_BASE_URL=http://localhost:8000/v1   # judge + partner model
```

The judge and conversational partner are reached over any OpenAI-compatible
API (vLLM, SGLang, or a hosted provider). If you adapt that client, see
`llm/README.md` for its retry budget and empty-completion handling.

## Train

Training has three stages. Both supervised stages role-swap every conversation
so that the human side occupies the generated position. The loss is masked to
those turns only, so the model is optimised to produce the human turns.

```bash
bash scripts/run_sft.sh        # stage 1: mid-training on the 62-corpus mixture
bash scripts/run_trl_sft.sh    # stage 2: SFT on ThoughtTrace reasoning traces
bash scripts/run_rl.sh         # stage 3: RL over the 24-task mixture
```

## Evaluate

```bash
bash scripts/run_eval.sh              # SOUL: 27 datasets, five capability axes
bash scripts/run_tau_usi_eval.sh      # tau-USI: agent success + USI vs 495 human dialogues
python eval/realusersim_eval.py       # RealUserSim PT3: five fidelity dimensions
python eval/turing_eval.py            # Turing-reward
```

### SimulatorArena

SimulatorArena, a 6 GB third-party repository, is not vendored. The setup steps
in `eval/simarena/SETUP.md` include cloning `microsoft/SimulatorArena`, applying
`utils.patch` and copying in `mimesis_transport.py`.

## Layout

```
agents/      19 task agent loops + reward functions; base_agent.py routes data_source
sft/         mid-training + ThoughtTrace SFT: dataset, role swap, TRL entry point
verl/        vendored verl v0.7.0 fork
llm/         judge / partner client (see llm/README.md)
scoring/     aggregate.py, aggregate_seeds.py, export_csv.py
eval/        RealUserSim, Turing, SimulatorArena integration
scripts/     launchers; env.sh holds every path
data/        tau-USI and RealUserSim snapshots
```

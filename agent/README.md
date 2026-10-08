# Agent training

Reinforcement learning for the user-facing agent: multi-turn GRPO plus CSD, an
auxiliary self-distillation term that turns coaching notes into a dense per-turn
signal beyond the sparse task reward.

In the code, CSD appears under the name `sdpo` (`run_sdpo.sh`, the `SDPO_*` variables and `verl/workers/rollout/sdpo_hint.py`), and coaching notes are called hints.

The agent is trained and evaluated against a simulated user. The simulator half
of the paper is in [`../simulator`](../simulator).

## Install

The vendored `verl` here declares the package name `verl`, as `../simulator`
does, so the two need separate environments.

```bash
conda create -n mimesis-agent python=3.12 && conda activate mimesis-agent
pip install -r requirements.txt
pip install -e .

bash scripts/install_gyms.sh      # clones upstream UserRL, applies our patch
bash scripts/prepare_data.sh      # builds data/rl_split and the eval parquets
```

## Train

```bash
source scripts/env.sh

bash scripts/run_sft.sh /path/to/Qwen3-8B     # stage 1, see sft/README.md
bash scripts/run_sdpo.sh                      # stage 2, CSD (see below)
bash scripts/run_grpo.sh                      # stage 2, GRPO control
```

`run_sdpo.sh` sets `USERRL_SDPO=1`, which turns on the CSD term. The coach is served by a second shim, separate from the simulated user's:

```bash
SHIM_UPSTREAM=http://127.0.0.1:8000/v1 SHIM_MODEL=mimesis-9b python -m llm.shim --port 8711 &
SDPO_HINT_BASE_URL=http://127.0.0.1:8711/v1 bash scripts/run_sdpo.sh
```

`SDPO_ABLATE_PRIVILEGED=1` blanks the coach's privileged inputs (the simulator's thought and next reply).

## Serve the simulated user

```bash
bash scripts/serve_simulators.sh /path/to/mimesis-9b mimesis-9b 8000 8710
```

This starts vLLM on `:8000` and `llm/shim.py` on `:8710`. Point `SIM_BASE_URL`
at the shim, not at vLLM. The shim strips `<think>` blocks before the gym sees a
reply, since a gym uses `message.content` verbatim as the user's utterance. It
stores that reasoning per episode as CSD's privileged channel, which
`verl/tools/interact_tool.py` reads back over `GET /ep/<id>/privileged`.
`verl/tools/env_manager.py` gives each gym instance an `/ep/<request_id>/v1`
base URL so that reasoning is attributed to the right trajectory. Any
OpenAI-compatible endpoint, including a hosted provider, can serve as the
upstream; see `llm/README.md`.

## Evaluate

```bash
bash scripts/run_eval.sh /path/to/checkpoint my-agent
```

## Layout

```
scripts/      env.sh (sets all paths) and one driver per stage
llm/          api.py (OpenAI-compatible client), shim.py (think-stripping proxy)
verl/         vendored RL framework, including CSD
eval/         eval.py, table3.py, make_main_table.py, digest.py, merge.py
examples/     hydra configs and the data-split script
patches/      gyms.patch
sft/          stage-1 config and checkpoint repair
results/      score digests read by eval/make_main_table.py (not shipped)
```

# Agent training

Reinforcement learning for the user-facing agent, with an auxiliary
self-distillation term that supplies dense per-turn signal where multi-turn GRPO
gives almost none.

The agent is trained and evaluated against a simulated user. The simulator half
of the paper lives in [`../simulator`](../simulator).

---

## Install

The vendored `verl` here declares the package name `verl`, exactly as
`../simulator` does, so **the two cannot share one environment**. Use a separate
one for each.

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
bash scripts/run_sdpo.sh                      # stage 2, our method
bash scripts/run_grpo.sh                      # stage 2 control
```

## Serve the simulated user

```bash
bash scripts/serve_simulators.sh /path/to/mimesis-9b mimesis-9b 8000 8710
```

Two processes: vLLM on `:8000`, and `llm/shim.py` on `:8710`. **Point
`SIM_BASE_URL` at the shim, never at vLLM.** The shim does two things that are
load-bearing rather than cosmetic:

* strips `<think>` blocks before the gym sees the reply. A gym uses
  `message.content` verbatim as the user's utterance, so an unstripped reasoning
  trace becomes something the user "said" — the agent reads it, the judge grades
  against it, and the episode is corrupted with no error raised anywhere.
* keeps that stripped reasoning, keyed by episode, as SDPO's privileged channel.
  `verl/tools/interact_tool.py` reads it back over `GET /ep/<id>/privileged`.
  This is why `verl/tools/env_manager.py` gives each gym instance a
  `/ep/<request_id>/v1` base URL: an unscoped call cannot be attributed to a
  trajectory, and guessing would pair one episode's reasoning with another's
  turn — which yields plausible hints and no error.

Any OpenAI-compatible endpoint works as the upstream, including a hosted
provider; see `llm/README.md`.

## Evaluate

```bash
bash scripts/run_eval.sh /path/to/checkpoint my-agent
```

## Layout

```
scripts/      env.sh (the only place paths live) + one driver per stage
llm/          api.py (OpenAI-compatible client), shim.py (think-stripping proxy)
verl/         vendored RL framework, SDPO lives here
eval/         eval.py, table3.py, make_main_table.py, digest.py, merge.py
examples/     hydra configs and the data-split script
patches/      gyms.patch
sft/          stage-1 config and checkpoint repair
results/      score digest, so the main table regenerates without a rerun
```

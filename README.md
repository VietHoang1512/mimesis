Code for the paper: MIMESIS: Learning User Simulators as Training Environments for Interactive Agents

Two halves, each self-contained with its own README:

- [`simulator/`](simulator) — training and evaluating the user simulator
  (Mimesis): SFT, RL, and the simulator benchmarks.
- [`agent/`](agent) — training and evaluating the agent against that simulator:
  multi-turn GRPO plus the SDPO self-distillation term, and the 15-gym
  evaluation behind the main results table.

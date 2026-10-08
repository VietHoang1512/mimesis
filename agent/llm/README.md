# Judge and simulator endpoints

The agent is served locally (vLLM/SGLang). This module covers the other side of
every conversation: the simulated user the agent talks to and the judge that
scores the episode.

Two files:

* `api.py`: an OpenAI-compatible client, the same as
  `../../simulator/llm/api.py` apart from the default model id each side falls
  back to.
* `shim.py`: a local proxy that the gyms point at, placed in front of whatever
  serves the simulator.

## Configuration

```bash
export SIM_BASE_URL=http://localhost:8710/v1   # the shim, not the model server
export OPENAI_AGENT_API_KEY=...                # "EMPTY" for a local server
export OPENAI_MODEL_NAME=<model-id>            # judge model, if you use one
```

Any server that implements the OpenAI chat-completions API works: vLLM, SGLang,
a hosted provider, or your own gateway. By default no separate judge runs: the
simulator grades its own interaction. This is the upstream behaviour and the
setting used for every number in the paper.

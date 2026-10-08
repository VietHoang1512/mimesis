# Judge and simulator endpoints

The agent is served locally (vLLM/SGLang). This module is the *other* side of
every conversation: the simulated user it talks to, and the judge that scores
the episode.

Two files:

* **`api.py`** — an OpenAI-compatible client, deliberately the same one as
  `../../simulator/llm/api.py` apart from the default model id each side falls
  back to, so both halves of the paper reach their partner models through one
  code path.
* **`shim.py`** — a local proxy the gyms point at, which sits in front of
  whatever is serving the simulator.

## Configuration

```bash
export SIM_BASE_URL=http://localhost:8710/v1   # the SHIM, not the model server
export OPENAI_AGENT_API_KEY=...                # "EMPTY" for a local server
export OPENAI_MODEL_NAME=<model-id>            # judge model, if you use one
```

Anything speaking the OpenAI chat-completions API works: vLLM, SGLang, a hosted
provider, or a gateway of your own. By default no separate judge runs at all —
the simulator grades its own interaction, which is the upstream behaviour and
what every number in the paper used.

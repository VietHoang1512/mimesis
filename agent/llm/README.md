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

## Why the shim exists

It is not a convenience layer. It does two things that change results:

**1. It strips reasoning before the gym sees it.** A gym takes
`message.content` verbatim as the user's utterance. Hand it a `<think>` block
and the agent reads a reasoning trace as if the user had spoken it, the judge
grades against it, and the episode is corrupted in a way nothing downstream
detects. `remove_think()` deliberately does *not* fall back to the raw text when
stripping leaves nothing: an empty return lands in the gym's ordinary
missing-response path, which is the honest outcome — the simulator did not
produce an utterance. Returning the trace instead fired 3,678 times in one
early run with a stock Qwen3-8B simulator, before this was fixed.

Three tag shapes are handled, because a reasoning block does not always arrive
whole: the model emits its own opener; the chat template pre-filled the opener
so the completion *starts* inside the block; or generation hit the token cap
before closing. A pre-filled opener that is also truncated carries no tag at all
and is indistinguishable from a plain answer — serve with thinking disabled if
that matters for your simulator.

**2. It holds that reasoning as SDPO's privileged channel.** The same block,
deleted from the reply, is one of the two privileged signals a hint is built
from. The strip site is the only point in the system where the block is still
whole and in memory, so that is where it is captured, bounded to a short
per-episode tail. `verl/tools/interact_tool.py` reads it back over
`GET /ep/<episode>/privileged`.

Episode scoping is therefore load-bearing: `verl/tools/env_manager.py` gives
each gym instance a `/ep/<request_id>/v1` base URL so reasoning can be
attributed to a trajectory. An unscoped call is not stashed at all, because
guessing would pair one episode's reasoning with another's turn — producing
plausible hints and no error.

## What was removed

The runs reported in the paper used a larger version of `shim.py`, wrapping the
same core in operational scaffolding — throttling, fan-out across several
upstream endpoints, per-episode transcript capture — so that one process could
serve many agent × simulator pairings at once. None of it affects the method or
the numbers and it is omitted here. If you are running a sweep large enough to
need backpressure you will know, and the retry/backoff in `api.py` is the right
place to start.

## Two behaviours worth preserving if you adapt `api.py`

**The retry budget must outlast the provider's rate-limit window.** Providers
meter over windows measured in minutes. A retry loop that gives up in ~45 s
cannot ride one out, and the failure is silent: the call returns nothing, the
turn is dropped, the conversation ends early and scores 0 — indistinguishable
from a model that simply refused.

**An empty completion is an error, not a result.** A reasoning model whose token
budget is consumed before it emits visible text returns a well-formed response
with no content. Treating that as success silently poisons an evaluation, so it
raises instead.

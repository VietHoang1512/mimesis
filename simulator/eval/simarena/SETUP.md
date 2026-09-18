# SimulatorArena integration

SimulatorArena is a third-party benchmark (~6 GB with its data), so it is not
vendored here. Our contribution is a transport that lets it drive a MIMESIS
simulator, plus a small patch to its conversation loop.

## Setup

```bash
git clone https://github.com/microsoft/SimulatorArena.git
cd SimulatorArena

# 1. our conversation-loop changes (+89/-10 in simulation/utils.py)
git apply /path/to/mimesis/simulator/eval/simarena/utils.patch

# 2. the transport that points the simulator at our model
cp /path/to/mimesis/simulator/eval/simarena/mimesis_transport.py simulation/

# 3. the direct-user-profile prompts used for the reported runs
cp /path/to/mimesis/simulator/eval/simarena/prompts/*.txt \
   simulation/prompts/document_creation/

# 4. the driver
cp /path/to/mimesis/simulator/eval/simarena/run_simarena.sh .
```

Then run from the SimulatorArena root with `OPENAI_AGENT_BASE_URL` pointing at
your served simulator:

```bash
bash run_simarena.sh
```

## What the patch changes

`simulation/utils.py` gains a pluggable completion backend and a termination
check that works when the simulated user is a local model rather than a hosted
API. Without it, the upstream loop assumes an OpenAI client object and cannot
reach a vLLM endpoint.

## Metric

The Turing-style score is `deviation = accuracy - 50`, where `accuracy` is how
often the judge correctly identifies which side of a conversation was the
simulator. **Lower is better**: 0 means the judge is at chance, i.e. the
simulator is indistinguishable from a real user. Scores in the published runs
sit in the 38–50 range, so every simulator tested remains well short of
indistinguishable.

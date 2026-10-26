# Local validation

Validated on Apple Silicon with 24 GiB shared memory, Python 3.12.13,
PyTorch 2.14.1, and Transformers 5.20.1.

Checkpoint: `meta-llama/Llama-3.2-3B-Instruct`, revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`.

- CPU checkpoint loading and short greedy generation completed.
- MPS float16 loading and three-row explorer steering completed on layer 14.
- MPS baseline, projection removal, and projection retention produced finite continuation
  log probabilities; intervention hooks were removed after scoring.
- Full CLI MPS smoke: collected the 24 bundled example prompts at layer 14, fitted a
  1-by-2 SOM for 10 epochs, and evaluated one test question across all four conditions
  and five methods. All 20 candidate-comparison records had finite margins.
- ChatGPT panel browser smoke: sent a prompt, observed the explorer overview tool call,
  and received the correct Llama model ID, MPS device, and 28-layer count using the
  existing Codex ChatGPT sign-in.
- SOM MCP integration: a bounded MPS pilot completed collection, fitting, and one-question
  evaluation through `som_run_pilot`. The ChatGPT panel called `som_status`,
  `som_job_status`, and `som_read_artifact` and distinguished local pilot artifacts from
  historical external job records.
- Automated tests cover model routing, MPS placement, normalization, hooks, cache isolation,
  non-square neuron grids, data validation, leakage checks, SOM fitting, and projection math.

These are functional tests, not evidence of improved sycophancy or benchmark accuracy.
The sample is intentionally too small for scientific claims. Raw local smoke artifacts
are ignored by Git under `research_outputs/`. No CUDA/vLLM execution was tested.

Reproduce the bounded model smoke after downloading the pinned checkpoint:

```sh
uv run python -m research.som_pipeline collect --prompts research/examples/paired_prompts.jsonl --out research_outputs/mps-smoke/activations.npz --layer 14 --device mps
uv run python -m research.som_pipeline fit --activations research_outputs/mps-smoke/activations.npz --out research_outputs/mps-smoke/som.npz --rows 1 --cols 2 --epochs 10
uv run python -m research.som_pipeline evaluate --prompts research/examples/paired_prompts.jsonl --activations research_outputs/mps-smoke/activations.npz --artifact research_outputs/mps-smoke/som.npz --out research_outputs/mps-smoke/evaluation.json --split test --device mps --max-questions 1
uv run pytest -q
```

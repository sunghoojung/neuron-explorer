# Neuron explorer: Llama + SOM research fork

Adaptation of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)
for `meta-llama/Llama-3.2-3B-Instruct`, with model-specific explorer data and a separate
self-organizing map (SOM) research pipeline. The pinned checkpoint revision is
`0cb88a4f764b7a12671c53f0838cd831a0843b95` (28 decoder layers, 8,192 MLP neurons per layer).

## Install and launch on Apple Silicon

```sh
uv sync
# Requires accepted Meta model terms and Hugging Face access for this checkpoint.
# Authenticate using `uv run hf auth login` if access is not already configured.
uv run hf download meta-llama/Llama-3.2-3B-Instruct --revision 0cb88a4f764b7a12671c53f0838cd831a0843b95
EXPLORER_DEVICE=mps uv run python explorer/server.py
```

Open http://127.0.0.1:8791. MPS uses float16 weights on the Apple GPU. Use
`EXPLORER_DEVICE=cpu` for CPU execution. The download command stores model weights in the standard Hugging
Face cache, outside this repository. No credentials belong in source or command arguments.
For reproducible runs, keep the default pinned revision; custom checkpoints can use
`EXPLORER_MODEL` and `EXPLORER_REVISION`.

Derived caches, neuron names, statistics, and sweep runs are isolated by model and revision.
The bundled upstream Qwen data is not Llama data. A fresh Llama installation initially has
no precomputed sweeps or layout caches; the interface reports their availability. Loading
the model does not automatically run a large corpus sweep or generate all-neuron caches.

## Claude research assistant

The assistant panel runs the local `claude` CLI under your existing Claude Code login (run `claude`
once and sign in with `/login` if needed). The explorer does not read or copy Claude credentials or
session files. Set `EXPLORER_CLAUDE_MODEL` to choose a model, or leave it unset for the CLI default;
`EXPLORER_CLAUDE_BIN` points at a different executable. Each chat is a `claude -p` session resumed turn
by turn, with only the explorer MCP server configured (`--strict-mcp-config`) and no built-in tools
(no shell, file edits, or web fetch). Chats and selected explorer-tool results are sent to Anthropic
when you use the panel; Llama inference remains local. Chat history stays under the model-specific
ignored cache. When a session cannot be resumed (a new chat, an edit or reroll, or a chat from the
earlier Codex transport) the next turn starts a new session seeded from the saved history.
The panel supports text and image uploads; PDF uploads return an explicit unsupported-file
message. Persistent neuron naming remains available through the explorer naming field.

`EXPLORER_CHAT_BACKEND=codex` keeps the older Codex CLI transport (`EXPLORER_CODEX_MODEL`,
`EXPLORER_CODEX_BIN`) selectable; it shares the same chat history files.

## SOM research

In the original **map** view, switch **neurons / states** to inspect saved residual
captures on the layer stack. Measured layer strips show raw paired RMS differences;
the expandable panel shows training-fitted PCA coordinates, matched cases, and saved
SOM cell memberships. These geometric views do not measure response correctness.

Open [the explorer's residuals view](http://127.0.0.1:8791/#residuals) to
create matched conversation groups, capture full residual vectors, compare group
measurements, and fit SOMs on training families. The workspace records token positions
and model provenance and supports held-out candidate-answer checks. Its scripted
example checks the workflow; it is not a sycophancy benchmark. See
[the group workspace guide](docs/RESIDUAL_GROUPS.md) for capture points and artifact details.

See [the research guide](docs/SOM_RESEARCH.md) for paired data collection, SOM fitting,
projection interventions, baselines, and held-out evaluation. The SOM is a separate
research method; the explorer's existing UMAP neuron layout is not a SOM.

```sh
uv run python -m research.som_pipeline --help
uv run pytest -q
```

The Claude panel can inspect SOM sources and artifacts, launch a bounded pilot through
`som_run_pilot`, and check its progress through `som_job_status`. It first checks existing
research through `som_status`. See the guide for ignored local research configuration.

The research pipeline is experimental infrastructure. Successful tests or generation do
not establish reduced sycophancy, novelty, or publishable results. Evaluate false-pressure
responses alongside truthful agreement and valid corrections on held-out questions.

The supported Hugging Face statistics tools also accept `--device mps` and `--revision`:

```sh
uv run python sweep/neuron_stats.py --device mps --layers 14 --batch 1 --seq-len 128 --tokens 1024
uv run python sweep/steer_lens.py --device mps --layers 14 --neurons-per-batch 1
```

These commands perform real collection; they are not run by installation. Even a single
steer-lens layer requires thousands of forward passes. Use small batches on shared memory.
Steer-lens dose calibration uses four fixed contexts; its relative doses are not directly
comparable to corpus-calibrated live steering doses.
The upstream custom-vLLM pod scripts, capture tools, and equivalence checker below are a
separate CUDA workflow and have not been validated on MPS. Do not run `setup_pod.sh` on a Mac.

## Attribution and license status

This fork preserves upstream source and documentation. At the time of adaptation the
upstream repository did not provide a license file. This fork does not grant a new license
to that source; resolve permission before redistributing it under a publication license.
Meta's model license and acceptable-use terms separately apply to the checkpoint.

## Original upstream documentation

The following describes the original Qwen workflow. Its flat cache paths, model-specific
shell scripts, and GPU-pod setup are historical reference, not the Llama launch instructions.

# neuron explorer

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Loads Qwen3.5-4B locally (MPS, CUDA or CPU) for live
steered chat, lens, connections and live inference over your own text.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.

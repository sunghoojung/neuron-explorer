# SOM sycophancy research pipeline

This package adds a small, reproducible workflow for studying paired prompts with `meta-llama/Llama-3.2-3B-Instruct`. It collects one residual-stream vector per prompt at a selected decoder block, fits a deterministic NumPy self-organizing map (SOM) to train-split `false_pressure - neutral` activation deltas, and evaluates residual projection ablations on held-out questions.

## Prompt JSONL

Each line is one user prompt and its correct and foil continuations:

```json
{"question_id":"q-001","split":"train","condition":"neutral","prompt":"What is 2 + 2? Answer:","target":" 4","foil":" 5"}
```

Required fields are `question_id`, `split`, `condition`, `prompt`, `target`, and `foil`. Each question must appear in exactly one of `train`, `validation`, or `test` and have exactly one row for every condition: `neutral`, `false_pressure`, `true_claim`, and `valid_correction`. Keep `target` and `foil` text identical across a question's four rows. Here `target` means the correct continuation and `foil` the incorrect alternative. Preserve leading whitespace when it is part of the answer continuation. `prompt` contains the user message only; the shared model adapter applies the model's chat template and assistant generation prefix during collection and evaluation.

The example file contains a few arithmetic and geography pairs for schema checks. It is not a research dataset and its example results must not be interpreted scientifically.

## Commands

Run from the repository root after installing the project dependencies:

```sh
python -m research.som_pipeline validate-prompts --prompts research/examples/paired_prompts.jsonl
python -m research.som_pipeline smoke --out /tmp/neuron-explorer-som-smoke
python -m research.som_pipeline collect \
  --prompts /path/to/paired_prompts.jsonl \
  --out /path/to/activations.npz \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --revision <model-commit> --layer 20 --device cpu
python -m research.som_pipeline fit \
  --activations /path/to/activations.npz \
  --out /path/to/som.npz --rows 4 --cols 4 --epochs 100 --seed 0
python -m research.som_pipeline evaluate \
  --prompts /path/to/paired_prompts.jsonl \
  --activations /path/to/activations.npz \
  --artifact /path/to/som.npz \
  --out /path/to/test-evaluation.json --split test \
  --device cpu --score-mode sequence_logprob
python -m research.som_pipeline validate-artifact --artifact /path/to/som.npz
```

Set `--revision` to a commit for reproducible model loading. Collection records the resolved model revision when Hugging Face exposes it. `--device cpu` is the default and is appropriate for the offline synthetic smoke. Local Apple Silicon may use `--device mps`; NumPy SOM fitting and SVD run on CPU, while model residual projections stay on the model device and activation dtype. CUDA experiments must run inside the workspace's approved Slurm allocation.

Use `--dry-run` on `collect`, `fit`, or `evaluate` to validate the inputs and show planned settings without loading model weights or writing the requested output. The separate `validate-prompts` and `validate-artifact` commands check those inputs directly.

## Fit and evaluation details

The SOM is trained only on question-matched `false_pressure - neutral` residual deltas from `train`. The artifact stores SOM prototypes, an SVD-derived orthonormal basis, model ID and revision, decoder layer and hook name, last-prompt-token metadata, activation dtype, projection compute dtype, grid, seed, activation checksum, and hashes of train question IDs. Evaluation checks prompt/activation dataset identity, model and hook metadata, and held-out question IDs against the train hashes before scoring.

Evaluation compares the unmodified baseline with removal of the SOM span, a global-mean direction, centered PCA directions, and a deterministic random span with the SOM's rank. PCA can have a lower rank than SOM when the centered train set has fewer independent directions; actual ranks are recorded. Held-out activation projection-energy fractions are reported separately from behavior scores.

Behavior scoring uses the model's conditional continuation sequence log-probability for `target` and `foil`. The default `sequence_logprob` is the sum of conditional token log-probabilities. Candidate wording and token length still affect the score, so keep answer pairs concise and comparable. `mean_token_logprob` is available as a length-normalized alternative, but is not a sequence probability. Residual projection removal is applied at the artifact's exact decoder-block output hook to all positions in each scored sequence.

The report includes the false-pressure versus neutral target-minus-foil margin and explicit `true_claim` and `valid_correction` regression diagnostics. Those are guardrails for inspecting whether a projection harms correct-answer preference under truthful claims or corrections. They do not establish general truthfulness, causality, or scientific success. Human review and a suitable, held-out research dataset are still required.

`smoke` creates a tiny synthetic dataset, activation file, SOM artifact, and geometry summary without loading a model or downloading weights. Its output is a plumbing check only. CLI results are compact TOON text; detailed metrics and artifacts are saved to the requested files. Usage errors and data errors include a concise next command.
